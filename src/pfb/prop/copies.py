"""Several accounts on one signal: N accounts of one plan copying one strategy's trades, beside
N independent accounts of the same marginal, so the output shows that copies give no diversification.

Copies. The N accounts see the SAME day path draw by draw: the seed, the resampled days and the Monte Carlo
paths of the single-account result (run_mc). Each account has the plan's own fees, evaluation, funded life
and payout rules. The rule engine is a deterministic function of (plan, day path, soft daily-loss booking),
so on each path the N copies end alike: one account's per-path outcome is computed once and counted N times.

Independent (the comparison column). N accounts whose outcomes are independent draws of the same marginal:
for each of n_paths joint draws, every account draws its own path outcome uniformly, with replacement, from
the copies' n_paths path outcomes (the empirical marginal; generator SeedSequence([seed,
INDEPENDENT_STREAM])).
The marginal is then exactly the copies' marginal, so the two columns differ in dependence only, not in
Monte Carlo noise of a second path set. Beside the draws, "product_of_marginals" gives the exact values for
independent accounts of that marginal: P(all pass) = p^N, P(all fail) = (1 - p)^N, mean N m, SD sqrt(N) s
(s the marginal's SD, ddof 0) and correlation 0. The drawn column carries what has no closed form (the
quantiles of the sum) and checks the product within Monte Carlo error.

Per account and path: passed (0/1), payout (the trader's payouts in cents, 0 when not passed) and net =
passed x (payout - activation - funded fees) - the attempt fee: the per-path value whose mean is
ev_per_purchase, so N = 1 reproduces the single-account result exactly. Money is integer cents; means and
SDs are floats in cents; no float decides a rule boundary (the engine's integer rules decide every outcome).

Plan rules about holding several accounts (an account-count limit, a copy-trading rule, a contract cap across
accounts) are applied where the plan file holds them; pfb-rules-1 has no such field, so each is listed under
"not_modelled" with the plan key that is missing, and a plan file that one day holds one is refused until it
is applied here. Nothing here registers a trial, appends to the ledger or counts toward N.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from pfb.prop.config import PlanRules
from pfb.prop.days import check_day_paths
from pfb.prop.matrix import (
    ROUND_DIGITS,
    UNITS,
    CapCheck,
    McSettings,
    PlanRefused,
    compact_estimate,
    compact_result,
    default_cap_check,
    display_path,
    undated_rules_warning,
    unknown_field,
)
from pfb.prop.matrix import _load as load_newest
from pfb.prop.mc import CHUNK_PATHS as DEFAULT_CHUNK
from pfb.prop.mc import (
    FUNDED_OUTPUTS,
    STANDING_WARNING,
    Engine,
    MCResult,
    _assemble,
    _check_args,
    _concat,
    _Fees,
    _fees,
    _PathOutcomes,
    _run_chunk,
    _soft_warning,
    _unknown_warning,
    binomial,
    default_engine,
    eval_horizon,
    make_sequences,
)
from pfb.prop.paths import stress
from pfb.prop.types import END_PASS, DayPaths, RuleUnknown

COPIES_SCHEMA = "pfb.prop_copies/1"
MAX_ACCOUNTS = 1000  # a bound on memory and time, not a modelling choice
INDEPENDENT_STREAM = 8  # SeedSequence([seed, 8]): the independent draws' own stream
QUANTILES = (("q05", 0.05), ("q25", 0.25), ("q50", 0.5), ("q75", 0.75), ("q95", 0.95))
QUANTILE_METHOD = "inverted_cdf"  # an observed sum, so integer cents
# A firm's rules about holding several accounts. pfb-rules-1 (pfb.prop.config.SECTION_FIELDS) has none of
# these keys; each is named as missing, never guessed. A plan file holding one is refused until applied here.
COPY_RULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("account_count_limit", "the firm's limit on evaluation or funded accounts held at once",
     ("eval.max_open_accounts", "funded.max_open_accounts")),
    ("copy_trading", "the firm's rule on copying one signal across accounts (allowed, banned or capped)",
     ("funded.copy_trading",)),
    ("aggregate_contract_cap", "a contract cap across all of a trader's accounts together",
     ("funded.max_contracts_all_accounts",)),
)  # fmt: skip
MISSING_REASON = "no pfb-rules-1 field holds it (pfb.prop.config.SECTION_FIELDS); not guessed"
SCOPE_NOT_MODELLED: tuple[tuple[str, str], ...] = (
    ("resets_per_account", "each account is one purchase along the path (no reset chain along the shared "
     "path); fees.reset_cents enters only the linear expectation cost_to_funded_with_resets x N"),
    ("staggered_starts", "every copy is bought on the path's first day (no account bought later)"),
    ("per_account_fills", "every copy gets the strategy's own fills (no per-account slippage or rejection)"),
)  # fmt: skip
APPLIED = (
    "per-account contract caps: eval.max_contracts_minis and the funded.scaling tiers, checked by the rule "
    "engine for every copy (each copy trades the strategy's own size)",
    "per-account fees: the attempt fee (fees.billing, eval_price_cents, platform_monthly_cents), "
    "fees.activation_cents and the funded fees, as in ev_per_purchase",
    "per-account evaluation, funded and payout rules of the plan file (the single result's payout policy)",
)  # fmt: skip
READING = (
    "copies: on every path the N accounts end as one does, so P(all pass) = P(one passes), P(all fail) = "
    "P(one fails) and the SD of the summed money is N times one account's; independent: the SD of the sum "
    "grows as sqrt(N) and P(all fail) falls as (1 - p)^N. Copying one signal adds exposure, not "
    "diversification."
)  # fmt: skip


# ----------------------------------------------------------------- the per-path run ----
@dataclass(frozen=True)
class PathRun:
    """run_mc's result with the per-path outcomes behind it (the same paths, the same numbers)."""

    result: MCResult
    outcomes: _PathOutcomes
    fees: _Fees
    want_funded: bool


def run_paths(plan: Any, days: DayPaths, s: McSettings, *, engine: Engine | None = None,
              chunk_paths: int = DEFAULT_CHUNK) -> PathRun:  # fmt: skip
    """run_mc (pfb.prop.mc) step for step, keeping the per-path outcomes and fees it reduces to estimates.
    tests/prop/test_copies.py pins result == run_mc(...) on the same arguments, so the two cannot drift."""
    _check_args(s.kind, s.n_paths, s.seed, s.horizon_eval, s.horizon_funded, s.stress_ticks, s.tick_cents,
                s.soft_daily_loss)  # fmt: skip
    sha = getattr(plan, "sha256", None)
    if not isinstance(sha, str) or not sha:
        raise ValueError("the plan carries no sha256 (load it with pfb.prop.config.load_plan)")
    engine = engine or default_engine()
    check_day_paths(days)
    if s.stress_ticks:
        days = stress(days, s.stress_ticks, int(s.tick_cents))  # type: ignore[arg-type]
    h_e, h_f = eval_horizon(plan, s.horizon_eval), s.horizon_funded
    path_ss, se_ss = np.random.SeedSequence(s.seed).spawn(2)
    rng = np.random.Generator(np.random.PCG64(path_ss))
    warnings = [STANDING_WARNING]
    soft = _soft_warning(plan, s.soft_daily_loss)
    if soft:
        warnings.append(soft)
    out: list[_PathOutcomes] = []
    want_funded = True
    for start in range(0, s.n_paths, chunk_paths):
        size = min(chunk_paths, s.n_paths - start)
        seqs = make_sequences(s.generator, days.n_days, h_e + h_f, size, s.mean_block, rng)
        try:
            out.append(_run_chunk(plan, days, seqs, h_e, h_f, engine, want_funded, s.soft_daily_loss))
        except RuleUnknown as exc:
            if not want_funded:
                raise
            warnings.append(_unknown_warning(exc, FUNDED_OUTPUTS, "funded phase not run"))
            want_funded = False
            out.append(_run_chunk(plan, days, seqs, h_e, h_f, engine, want_funded, s.soft_daily_loss))
    po = _concat(out)
    fees = _fees(plan, po, engine, want_funded, warnings)
    se_rng = np.random.Generator(np.random.PCG64(se_ss))
    run = dict(kind=s.kind, plan_sha256=sha, seed=s.seed, generator=s.generator,
               mean_block=s.mean_block if s.generator == "block" else None, n_paths=s.n_paths,
               horizon_eval=h_e, horizon_funded=h_f, chunk_paths=chunk_paths, stress_ticks=s.stress_ticks,
               tick_cents=s.tick_cents, days_source=tuple(days.source), n_days=days.n_days,
               soft_daily_loss=s.soft_daily_loss, plan_file_sha256=getattr(plan, "file_sha256", None) or sha,
               plan_overrides=tuple(getattr(plan, "overrides", ())))  # fmt: skip
    result = _assemble(run, plan, days, po, fees, want_funded, warnings, se_rng)
    return PathRun(result, po, fees, want_funded)


@dataclass(frozen=True)
class AccountPaths:
    """One account's outcome on each path (int64 cents); None where an output cannot be computed."""

    passed: np.ndarray
    payout: np.ndarray | None  # None: the funded phase did not run
    net: np.ndarray | None  # None: the funded phase did not run or a fee is unknown
    missing: tuple[str, ...]  # why payout or net is None


def account_paths(pr: PathRun) -> AccountPaths:
    """passed, payout and net per path, the net exactly as ev_per_purchase's per-path value (pfb.prop.mc)."""
    po, fees = pr.outcomes, pr.fees
    passed = (po.eval_end == END_PASS).astype(np.int64)
    if not pr.want_funded:
        why = "the funded phase did not run (a rule value it needs is unknown)"
        return AccountPaths(passed, None, None, (f"payout_sum_cents and net_sum_cents: {why}",))
    payout = passed * po.trader_cents
    if fees.first is None or fees.funded is None or fees.activation is None:
        return AccountPaths(passed, payout, None, ("net_sum_cents: a fee value is unknown",))
    net = passed * (po.trader_cents - fees.activation - fees.funded) - fees.first
    return AccountPaths(passed, payout, net, ())


# ---------------------------------------------------------------------- statistics ----
def _rounded(value: float | Fraction) -> float:
    return round(float(value), ROUND_DIGITS)


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    """Pearson correlation in exact integer sums (Python ints, no int64 overflow); None when either side has
    no variance. Identical arrays give exactly 1.0."""
    n = int(x.shape[0])
    xs, ys = [int(v) for v in x], [int(v) for v in y]
    sx, sy = sum(xs), sum(ys)
    sxx = sum(v * v for v in xs)
    syy = sum(v * v for v in ys)
    sxy = sum(a * b for a, b in zip(xs, ys, strict=True))
    num, d1, d2 = n * sxy - sx * sy, n * sxx - sx * sx, n * syy - sy * sy
    if d1 <= 0 or d2 <= 0:
        return None
    prod = d1 * d2
    root = math.isqrt(prod)
    if root * root == prod:
        return float(Fraction(num, root))
    return float(num) / math.sqrt(float(prod))


def distribution(x: np.ndarray) -> dict[str, Any]:
    """Mean (with its SE, ddof 1), SD (ddof 0: the distribution's own), min, max and the QUANTILES (observed
    values, integer cents) of a per-draw sum."""
    n = int(x.shape[0])
    sd1 = float(np.std(x, ddof=1)) if n > 1 else 0.0
    out: dict[str, Any] = {
        "mean": _rounded(np.mean(x)), "mean_se": _rounded(sd1 / math.sqrt(n)), "sd": _rounded(np.std(x)),
        "min": int(x.min()), "max": int(x.max()),
        "quantiles": {k: int(np.quantile(x, q, method=QUANTILE_METHOD)) for k, q in QUANTILES},
    }  # fmt: skip
    return out


def _prob(k: int, n: int) -> dict[str, Any]:
    return compact_estimate(binomial(k, n))


def column(n_accounts: int, k_pass: np.ndarray, payout_sum: np.ndarray | None, net_sum: np.ndarray | None,
           pair_pass: tuple[np.ndarray, np.ndarray] | None,
           pair_net: tuple[np.ndarray, np.ndarray] | None) -> dict[str, Any]:  # fmt: skip
    """One column: k_pass[i] is the number of the N accounts that passed in draw i; the sums are over the N
    accounts; pair_* are two accounts' per-draw values (None when N = 1)."""
    n = int(k_pass.shape[0])
    net = None
    if net_sum is not None:
        net = {**distribution(net_sum), "p_negative": _prob(int((net_sum < 0).sum()), n)}
    return {
        "n_draws": n,
        "p_all_pass": _prob(int((k_pass == n_accounts).sum()), n),
        "p_all_fail": _prob(int((k_pass == 0).sum()), n),
        "p_any_pass": _prob(int((k_pass > 0).sum()), n),
        "accounts_passed_mean": _rounded(np.mean(k_pass)),
        "corr_pass_two_accounts": None if pair_pass is None else _maybe(pearson(*pair_pass)),
        "corr_net_two_accounts": None if pair_net is None else _maybe(pearson(*pair_net)),
        "payout_sum_cents": None if payout_sum is None else distribution(payout_sum),
        "net_sum_cents": net,
    }


def _maybe(value: float | None) -> float | None:
    return None if value is None else _rounded(value)


def copies_column(acc: AccountPaths, n_accounts: int) -> dict[str, Any]:
    """N accounts on the same path: every sum is N times one account's value on that path."""
    pair = n_accounts >= 2
    return column(
        n_accounts, n_accounts * acc.passed,
        None if acc.payout is None else n_accounts * acc.payout,
        None if acc.net is None else n_accounts * acc.net,
        (acc.passed, acc.passed) if pair else None,
        (acc.net, acc.net) if pair and acc.net is not None else None,
    )  # fmt: skip


def independent_column(acc: AccountPaths, n_accounts: int, seed: int) -> dict[str, Any]:
    """N accounts drawing their path outcomes independently from the copies' n path outcomes (module doc)."""
    n = int(acc.passed.shape[0])
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, INDEPENDENT_STREAM])))
    k_pass = np.zeros(n, dtype=np.int64)
    payout_sum = None if acc.payout is None else np.zeros(n, dtype=np.int64)
    net_sum = None if acc.net is None else np.zeros(n, dtype=np.int64)
    firsts: list[np.ndarray] = []
    for _ in range(n_accounts):
        idx = rng.integers(0, n, size=n)
        k_pass += acc.passed[idx]
        if payout_sum is not None and acc.payout is not None:
            payout_sum += acc.payout[idx]
        if net_sum is not None and acc.net is not None:
            net_sum += acc.net[idx]
        if len(firsts) < 2:
            firsts.append(idx)
    pair = len(firsts) == 2
    return column(
        n_accounts, k_pass, payout_sum, net_sum,
        (acc.passed[firsts[0]], acc.passed[firsts[1]]) if pair else None,
        (acc.net[firsts[0]], acc.net[firsts[1]]) if pair and acc.net is not None else None,
    )  # fmt: skip


def product_of_marginals(acc: AccountPaths, n_accounts: int) -> dict[str, Any]:
    """Exact values for N independent accounts of the empirical marginal (module docstring)."""
    n = int(acc.passed.shape[0])
    p = Fraction(int(acc.passed.sum()), n)
    root_n = math.sqrt(n_accounts)

    def moments(x: np.ndarray | None) -> dict[str, float] | None:
        if x is None:
            return None
        return {"mean": _rounded(n_accounts * float(np.mean(x))), "sd": _rounded(root_n * float(np.std(x)))}

    return {
        "p_all_pass": _rounded(p**n_accounts), "p_all_fail": _rounded((1 - p) ** n_accounts),
        "p_any_pass": _rounded(1 - (1 - p) ** n_accounts),
        "corr_two_accounts": None if n_accounts < 2 else 0.0,
        "payout_sum_cents": moments(acc.payout), "net_sum_cents": moments(acc.net),
    }  # fmt: skip


# ------------------------------------------------------------------ plan rules ----
def copy_rules(plan: PlanRules) -> list[dict[str, Any]]:
    """The "not_modelled" list: each COPY_RULES rule with its missing plan keys, then the scope limits. A plan
    holding one of the keys is refused (PlanRefused) until this module applies it: never silently ignored."""
    out: list[dict[str, Any]] = []
    for rule, what, keys in COPY_RULES:
        held = [k for k in keys if k in plan.leaves]
        if held:
            raise PlanRefused(f"plan key {held[0]} ({what}) is in the plan file, but pfb.prop.copies "
                              "does not apply it yet", field=held[0], error="CopyRuleNotApplied")  # fmt: skip
        out.append({"rule": rule, "what": what, "plan_keys_missing": list(keys), "reason": MISSING_REASON})
    out += [{"rule": rule, "what": what, "plan_keys_missing": None, "reason": "outside this model"}
            for rule, what in SCOPE_NOT_MODELLED]  # fmt: skip
    return out


def check_accounts(n_accounts: Any) -> int:
    if isinstance(n_accounts, bool) or not isinstance(n_accounts, int) or not 1 <= n_accounts <= MAX_ACCOUNTS:
        raise PlanRefused(f"copies must be an integer from 1 to {MAX_ACCOUNTS}, got {n_accounts!r}",
                          field="copies", error="BadRequest")  # fmt: skip
    return n_accounts


# ---------------------------------------------------------------------- document ----
def _linear(record: dict[str, Any], n_accounts: int) -> dict[str, Any]:
    """Expectations of a sum do not depend on dependence: N x the single-account value, in both columns."""
    out: dict[str, Any] = {}
    for name in ("ev_per_purchase", "cost_to_funded_with_resets"):
        value = record["outputs"][name]["value"]
        out[f"{name}_cents"] = None if value is None else _rounded(n_accounts * value)
    return out


def copies_document(plan: PlanRules, record: dict[str, Any], pr: PathRun, n_accounts: int,
                    not_modelled: list[dict[str, Any]]) -> dict[str, Any]:  # fmt: skip
    """The pfb.prop_copies/1 object: the rules labels and the single-account record, the copies and the
    independent column side by side, the exact product of marginals, and what is applied and not modelled."""
    acc = account_paths(pr)
    labels = {**plan.rules_label(), "rules_in_force": record["rules_in_force"]}
    return {
        "schema": COPIES_SCHEMA, **labels, "kind": record["kind"], "trial": False,
        "n_accounts": n_accounts, "seed": record["seed"], "n_paths": record["n_paths"],
        "soft_daily_loss": record["soft_daily_loss"], "units": UNITS,
        "copies": copies_column(acc, n_accounts),
        "independent": independent_column(acc, n_accounts, record["seed"]),
        "product_of_marginals": product_of_marginals(acc, n_accounts),
        "linear_expectations": _linear(record, n_accounts),
        "method": {
            "copies": "the single-account Monte Carlo paths; each path's one-account outcome counted N times",
            "independent": f"each of N accounts draws its path outcome uniformly with replacement from the "
                           f"{record['n_paths']} copies' path outcomes (SeedSequence([seed, "
                           f"{INDEPENDENT_STREAM}])): the same marginal, independent accounts",
            "product_of_marginals": "exact for independent accounts of that marginal: p^N, (1-p)^N, "
                                    "1-(1-p)^N, N x mean, sqrt(N) x SD (ddof 0), correlation 0",
            "quantiles": f"{QUANTILE_METHOD} (observed sums, integer cents); SD ddof 0, mean_se ddof 1",
        },
        "reading": READING, "applied": list(APPLIED), "not_modelled": not_modelled,
        "missing": list(acc.missing), "single": record,
        "warnings": list(record["warnings"]),
    }  # fmt: skip


def evaluate_copies(path: Path, days: DayPaths, settings: McSettings, n_accounts: int, *, repo_root: Path,
                    engine: Engine | None = None, cap_check: CapCheck = default_cap_check,
                    run: Callable[..., PathRun] = run_paths) -> tuple[dict, dict, MCResult]:  # fmt: skip
    """evaluate_plan's steps (load the newest version, check caps, run the Monte Carlo) keeping the per-path
    outcomes: (copies document, the single-account compact record, MCResult); PlanRefused if it cannot run.
    The record equals evaluate_plan's on the same arguments."""
    check_accounts(n_accounts)
    plan = load_newest(path)
    not_modelled = copy_rules(plan)
    try:
        extra = [*cap_check(days, plan), undated_rules_warning(days, plan)]
        pr = run(plan, days, settings, engine=engine)
    except RuleUnknown as exc:
        field = unknown_field(exc)
        raise PlanRefused(f"rule value unknown: {exc}", field=field, error="RuleUnknown") from exc
    record = compact_result(pr.result, plan, display_path(path, repo_root), extra)
    return copies_document(plan, record, pr, n_accounts, not_modelled), record, pr.result


# ------------------------------------------------------------- the result contract ----
ROW_KEYS = ("p_all_pass", "p_all_fail", "p_any_pass", "corr_pass_two_accounts")


def _row_column(col: dict[str, Any]) -> dict[str, Any]:
    out = {k: (col[k]["value"] if isinstance(col[k], dict) else col[k]) for k in ROW_KEYS}
    for name in ("payout_sum_cents", "net_sum_cents"):
        dist = col[name]
        stem = name.removesuffix("_cents")
        out[f"{stem}_mean_cents"] = None if dist is None else dist["mean"]
        out[f"{stem}_sd_cents"] = None if dist is None else dist["sd"]
    net = col["net_sum_cents"]
    for q in ("q05", "q50", "q95"):
        out[f"net_sum_{q}_cents"] = None if net is None else net["quantiles"][q]
    return out


def copies_row(doc: dict[str, Any]) -> dict[str, Any]:
    """The "copies" object of a result-contract prop row (docs/result.schema.json $defs copies_row)."""
    return {
        "n_accounts": doc["n_accounts"],
        "copies": _row_column(doc["copies"]),
        "independent": _row_column(doc["independent"]),
        "not_modelled": [item["rule"] for item in doc["not_modelled"]],
    }


def copies_artifact_name(single_name: str, n_accounts: int) -> str:
    """<single-plan artifact stem>_copies<N>.json."""
    return f"{single_name.removesuffix('.json')}_copies{n_accounts}.json"
