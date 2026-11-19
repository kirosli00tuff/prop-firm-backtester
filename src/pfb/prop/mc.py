"""Prop Monte Carlo on a strategy's own day paths: estimators with standard errors (docs/PROP_MC.md).

run_mc draws sequences of day indices (block bootstrap or random start), runs one evaluation attempt along
each from a fresh account (horizon H_e), continues every passing path into one funded life from the day after
the pass (horizon H_f), prices both with the plan's fees and returns MCResult. The rule engine is passed in
as an Engine (default: pfb.prop.kernel and pfb.prop.fees, imported lazily) so the estimators are testable on
a fake engine. Nothing here registers a trial.

Rules added after the first review (docs/PROP_MC.md): an unknown fee or funded rule nulls only the outputs
that need it; a retry after END_TIME_LIMIT is a new purchase, after any other failure the reset;
random_start draws only full windows when the history allows and otherwise counts padded paths and marks
the outputs they bias as lower bounds; END_SIZE_LIMIT ends are counted apart from breaches and warned
with the share of days over the limits.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from pfb.prop.days import check_day_paths
from pfb.prop.paths import block_bootstrap, padded_paths, random_start, stress
from pfb.prop.rules import SOFT_DLL_LIMIT, SOFT_DLL_LOW, SOFT_DLL_MODES
from pfb.prop.types import (
    END_BREACH,
    END_PASS,
    END_PAYOUT_LIMIT,
    END_SIZE_LIMIT,
    END_TIME_LIMIT,
    END_TIMEOUT,
    DayPaths,
    RuleUnknown,
)

MC_SCHEMA = "pfb.prop_mc/1"
KINDS = ("synthetic", "engine_check", "trial")  # a trial: only its declared secondary outputs (TRIALS.md s6)
GENERATORS = ("block", "random_start")
CHUNK_PATHS = 10_000  # paths per engine call: 10,000 x 504 int64 = 40 MB of sequences
QUANTILE_RESAMPLES = 200
TRADING_DAYS_PER_30_CALENDAR = 21  # PROP_MC.md section 4: 30 calendar = 21 trading days
CALENDAR_PER_PERIOD = 30
STANDING_WARNING = "price-limit and locked-market rules are not modelled"
END_NAMES = {END_PASS: "pass", END_BREACH: "breach", END_TIMEOUT: "timeout", END_PAYOUT_LIMIT: "payout_limit",
             END_TIME_LIMIT: "time_limit", END_SIZE_LIMIT: "size_limit"}  # fmt: skip
SE_SCOPE = ("every SE is a Monte Carlo error given the day pool (paths resampled from these days); it says "
            "nothing about how well a finite history represents the strategy")  # fmt: skip


# ------------------------------------------------------------------ engine ----
@dataclass(frozen=True)
class Engine:
    """The rule-engine entry points run_mc calls (PROP_MC.md section 2)."""

    eval_many: Callable[..., Any]  # (plan, days, seqs, *, soft_daily_loss) -> batch
    funded_many: Callable[..., Any]  # (plan, days, seqs, *, soft_daily_loss) -> batch
    attempt_fee: Callable[..., int]  # (plan, outcome_days, calendar_days, *, first) -> cents
    funded_fee: Callable[[Any, int, int], int]  # (plan, days_alive, calendar_days) -> cents


def default_engine() -> Engine:
    """The real engine: Numba batch kernels and the fee functions (imported here, not at module load)."""
    from pfb.prop import fees, kernel  # pyright: ignore[reportAttributeAccessIssue]

    return Engine(eval_many=kernel.simulate_eval_many, funded_many=kernel.simulate_funded_many,
                  attempt_fee=fees.attempt_fee_cents, funded_fee=fees.funded_fee_cents)  # fmt: skip


@dataclass(frozen=True)
class _ScalarBatch:
    end: np.ndarray
    days: np.ndarray
    payouts_trader_cents: np.ndarray  # per-path total
    first_payout_day: np.ndarray


def _scalar_eval_many(plan: Any, days: DayPaths, seqs: np.ndarray, *,
                      soft_daily_loss: str = SOFT_DLL_LIMIT) -> _ScalarBatch:  # fmt: skip
    from pfb.prop import rules

    outs = [rules.simulate_eval(plan, days, row, soft_daily_loss=soft_daily_loss) for row in seqs]
    zeros = np.zeros(len(outs), dtype=np.int64)
    return _ScalarBatch(np.array([o.end for o in outs], dtype=np.int64),
                        np.array([o.days for o in outs], dtype=np.int64), zeros, zeros - 1)  # fmt: skip


def _scalar_funded_many(plan: Any, days: DayPaths, seqs: np.ndarray, *,
                        soft_daily_loss: str = SOFT_DLL_LIMIT) -> _ScalarBatch:  # fmt: skip
    from pfb.prop import rules

    outs = [rules.simulate_funded(plan, days, row, soft_daily_loss=soft_daily_loss) for row in seqs]
    return _ScalarBatch(
        np.array([o.end for o in outs], dtype=np.int64), np.array([o.days for o in outs], dtype=np.int64),
        np.array([sum(o.payouts_trader_cents) for o in outs], dtype=np.int64),
        np.array([o.first_payout_day for o in outs], dtype=np.int64),
    )  # fmt: skip


def scalar_engine() -> Engine:
    """The scalar reference (rules.py, one path at a time) with the real fees: slow, for checks on small n."""
    from pfb.prop import fees  # pyright: ignore[reportAttributeAccessIssue]

    return Engine(eval_many=_scalar_eval_many, funded_many=_scalar_funded_many,
                  attempt_fee=fees.attempt_fee_cents, funded_fee=fees.funded_fee_cents)  # fmt: skip


def plan_value(plan: Any, section: str, name: str) -> Any:
    """A rule value: PlanRules.value("<section>.<name>"), None for "none"; an Unknown raises RuleUnknown.

    A plan object without value() (a test fake) is read as plan.<section>[<name>], "none" mapped to None.
    """
    getter = getattr(plan, "value", None)
    if callable(getter):
        return getter(f"{section}.{name}")
    raw = getattr(plan, section)[name]
    return None if raw is None or raw == "none" else raw


def calendar_days(trading_days: int | np.ndarray) -> Any:
    """Trading days to calendar days at 21 : 30, rounded up (integer arithmetic)."""
    t = trading_days
    return (t * CALENDAR_PER_PERIOD + TRADING_DAYS_PER_30_CALENDAR - 1) // TRADING_DAYS_PER_30_CALENDAR


def eval_horizon(plan: Any, horizon_eval: int) -> int:
    """horizon_eval, or the plan's time limit converted at 30 calendar = 21 trading days when shorter."""
    limit = plan_value(plan, "eval", "time_limit_calendar_days")
    if limit is None:
        return horizon_eval
    return max(1, min(horizon_eval, int(limit) * TRADING_DAYS_PER_30_CALENDAR // CALENDAR_PER_PERIOD))


# ----------------------------------------------------------------- results ----
@dataclass(frozen=True)
class Estimate:
    """One output: value, SE, the n it is over, a note, and lower_bound when padded paths bias it down."""

    value: float | None
    se: float | None
    n: int
    note: str = ""
    lower_bound: bool = False


@dataclass(frozen=True)
class MCResult:
    """Every PROP_MC.md section 4 output with its SE, the counts, the run parameters and the warnings.

    soft_daily_loss names the soft daily-loss booking ("limit": exactly -DLL; "minute_low": the minute low);
    plan_sha256 is the plan's hash (config.overridden_sha256 under with_values), plan_file_sha256 the file's.

    counts: eval_padded and funded_padded (phases that ran into the end of history under random_start),
    eval_size_limit and funded_size_limit (END_SIZE_LIMIT ends, apart from breaches), funded_open_no_payout
    (funded lives open at the horizon with no payout yet: the bound on p_breach_before_first_payout).
    """

    kind: str
    plan_sha256: str
    seed: int
    generator: str
    mean_block: int | None
    n_paths: int
    horizon_eval: int
    horizon_funded: int
    chunk_paths: int
    stress_ticks: int
    tick_cents: int | None
    days_source: tuple[str, ...]
    n_days: int
    p_pass: Estimate
    days_to_pass_median: Estimate
    days_to_pass_p90: Estimate
    cost_per_attempt: Estimate
    cost_to_funded: Estimate
    cost_to_funded_with_resets: Estimate
    funded_ev: Estimate
    p_breach_before_first_payout: Estimate
    ev_per_purchase: Estimate
    eval_ends: dict[str, int]
    funded_ends: dict[str, int]
    counts: dict[str, int]
    warnings: tuple[str, ...]
    se_scope: str = SE_SCOPE
    schema: str = MC_SCHEMA
    soft_daily_loss: str = SOFT_DLL_LIMIT  # the soft daily-loss booking the run used (N1)
    plan_file_sha256: str | None = None  # the file hash; differs from plan_sha256 under overrides (N10)
    plan_overrides: tuple[tuple[str, Any], ...] = ()  # the sensitivity overrides (config.with_values)
    trial: bool = field(default=False)  # never a trial (PROP_MC.md section 4)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)


# --------------------------------------------------------------- per path ----
@dataclass(frozen=True)
class _PathOutcomes:
    """Per-path engine results over all n paths (funded fields are 0, or -1, where the attempt did not pass).

    funded_ok is False when the funded phase could not run (a rule value it needs is unknown)."""

    eval_end: np.ndarray
    eval_days: np.ndarray
    eval_padded: np.ndarray  # bool: the attempt ran into the -1 padding
    funded_end: np.ndarray  # -1 where not passed
    funded_days: np.ndarray
    funded_padded: np.ndarray  # bool: the funded life ran into the -1 padding
    trader_cents: np.ndarray
    first_payout_day: np.ndarray


def _arr(batch: Any, name: str) -> np.ndarray:
    return np.asarray(getattr(batch, name)).astype(np.int64)


def _trader_total(batch: Any) -> np.ndarray:
    """Trader payouts per path: trader_total_cents when the batch has it, else the payouts_trader_cents
    field summed (2-d, zero or -1 padded) or taken as a per-path total (1-d)."""
    if hasattr(batch, "trader_total_cents"):
        return _arr(batch, "trader_total_cents")
    name = "payouts_trader_cents"
    arr = np.asarray(getattr(batch, name)).astype(np.int64)
    if arr.ndim == 2:
        return np.where(arr > 0, arr, 0).sum(axis=1)
    if arr.ndim == 1:
        return arr
    raise TypeError(f"{name} must be 1-d or 2-d, got shape {arr.shape}")


def _fee_by_days(fn: Callable[[int], int], days: np.ndarray) -> np.ndarray:
    """Apply a fee function of a day count to every path (computed once per distinct count)."""
    if days.size == 0:
        return np.zeros(0, dtype=np.int64)
    uniq, inverse = np.unique(days, return_inverse=True)
    values = np.array([int(fn(int(d))) for d in uniq], dtype=np.int64)
    return values[inverse]


def _run_chunk(plan: Any, days: DayPaths, seqs: np.ndarray, h_e: int, h_f: int, engine: Engine,
               want_funded: bool, soft_daily_loss: str) -> _PathOutcomes:  # fmt: skip
    """Engine calls for one chunk of paths; a RuleUnknown from the funded phase propagates to run_mc."""
    n = seqs.shape[0]
    eseq = np.ascontiguousarray(seqs[:, :h_e])
    ev = engine.eval_many(plan, days, eseq, soft_daily_loss=soft_daily_loss)
    end, used = _arr(ev, "end"), _arr(ev, "days")
    f_end = np.full(n, -1, dtype=np.int64)
    f_days, trader, first_pay = (np.zeros(n, dtype=np.int64) for _ in range(3))
    f_pad = np.zeros(n, dtype=bool)
    passed = np.flatnonzero(end == END_PASS)
    if want_funded and passed.size:
        cols = used[passed][:, None] + np.arange(h_f, dtype=np.int64)[None, :]
        fseq = np.ascontiguousarray(np.take_along_axis(seqs[passed], cols, axis=1))
        fb = engine.funded_many(plan, days, fseq, soft_daily_loss=soft_daily_loss)
        f_end[passed], f_days[passed] = _arr(fb, "end"), _arr(fb, "days")
        trader[passed] = _trader_total(fb)
        first_pay[passed] = _arr(fb, "first_payout_day")
        f_pad[passed] = padded_paths(fseq) & (f_end[passed] == END_TIMEOUT)
    e_pad = padded_paths(eseq) & (end == END_TIMEOUT)
    return _PathOutcomes(end, used, e_pad, f_end, f_days, f_pad, trader, first_pay)


def _concat(parts: list[_PathOutcomes]) -> _PathOutcomes:
    return _PathOutcomes(*(np.concatenate([getattr(p, f) for p in parts])
                           for f in _PathOutcomes.__dataclass_fields__))  # fmt: skip


# --------------------------------------------------------------- estimators ----
def binomial(k: int, n: int) -> Estimate:
    """k / n with the binomial SE; at p = 0 or 1 the SE is 0 and the note says so (degenerate)."""
    if n == 0:
        return Estimate(None, None, 0, "no paths")
    p = k / n
    note = "" if 0 < k < n else f"SE 0 at p = {p:.0f} is degenerate; 95% bound 3/n = {3 / n:.2g} (rule of 3)"
    return Estimate(p, math.sqrt(p * (1.0 - p) / n), n, note)


def mean_se(x: np.ndarray) -> Estimate:
    """Mean and sample sd / sqrt(n) (ddof 1; SE 0 for a single value)."""
    n = int(x.shape[0])
    if n == 0:
        return Estimate(None, None, 0, "no paths")
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    return Estimate(float(np.mean(x)), sd / math.sqrt(n), n)


def quantile_bootstrap(x: np.ndarray, q: float, rng: np.random.Generator,
                       resamples: int = QUANTILE_RESAMPLES) -> Estimate:  # fmt: skip
    """Order-statistic quantile (inverted CDF: an observed value) and its bootstrap SE over paths."""
    n = int(x.shape[0])
    if n == 0:
        return Estimate(None, None, 0, "no passing paths")
    value = float(np.quantile(x, q, method="inverted_cdf"))
    boot = np.empty(resamples, dtype=np.float64)
    step = max(1, min(resamples, 2_000_000 // n))  # about 16 MB of draws at a time
    for lo in range(0, resamples, step):
        k = min(step, resamples - lo)
        draws = rng.integers(0, n, size=(k, n))
        boot[lo : lo + k] = np.quantile(x[draws], q, axis=1, method="inverted_cdf")
    return Estimate(value, float(np.std(boot, ddof=1)), n, f"bootstrap SE over {resamples} resamples")


def delta_se(influence: np.ndarray) -> float:
    """Delta-method SE from per-path influence values z_i: sqrt(mean(z^2) / n)."""
    n = influence.shape[0]
    return math.sqrt(float(np.mean(influence.astype(np.float64) ** 2)) / n)


def cost_to_funded(fee: np.ndarray, passed: np.ndarray, activation: int) -> Estimate:
    """cost_per_attempt / p_pass + activation; delta method on the ratio (covariance included)."""
    n = fee.shape[0]
    p = float(np.mean(passed))
    if p == 0.0:
        return Estimate(None, None, n, "no pass: undefined")
    c = float(np.mean(fee))
    z = (fee - (c / p) * passed) / p
    return Estimate(c / p + activation, delta_se(z), n)


def cost_with_resets(fee: np.ndarray, reset: np.ndarray, passed: np.ndarray, time_limit: np.ndarray,
                     activation: int) -> Estimate:  # fmt: skip
    """First purchase + the expected retries + activation (Wald's identity; PROP_MC.md).

    Attempts are independent paths; attempt k > 1 follows a failure of attempt k - 1 and costs a new purchase
    (fee, first=True) when that failure was END_TIME_LIMIT, else the reset (reset, first=False: the reset if
    any, else a new purchase), each with its own duration's monthly and platform fees. With T, B and P the
    shares of time-limit failures, other failures and passes:
    value = mean(fee) + (T mean(fee) + B mean(reset)) / P + activation, since (1/P - 1) retries follow
    failures whose mix is T : B. SE by the delta method on (mean fee, mean reset, T, B, P). With only
    time-limit failures it equals cost_to_funded exactly.
    """
    n = fee.shape[0]
    pi = passed.astype(np.float64)
    ti = time_limit.astype(np.float64)
    bi = 1.0 - pi - ti
    p = float(np.mean(pi))
    if p == 0.0:
        return Estimate(None, None, n, "no pass: undefined")
    f, r, t, b = float(np.mean(fee)), float(np.mean(reset)), float(np.mean(ti)), float(np.mean(bi))
    retries = (t * f + b * r) / p
    z = ((fee - f) + ((ti - t) * f + t * (fee - f) + (bi - b) * r + b * (reset - r)) / p
         - retries / p * (pi - p))  # fmt: skip
    return Estimate(
        f + retries + activation,
        delta_se(z),
        n,
        "Wald: first fee + retries (new purchase after a time limit, else reset) + activation",
    )


def _missing(n: int, needs: str) -> Estimate:
    return Estimate(None, None, n, f"not computed: needs {needs}")


# --------------------------------------------------------------- run_mc ----
def make_sequences(generator: str, n_days: int, horizon: int, n_paths: int, mean_block: int,
                   rng: np.random.Generator) -> np.ndarray:  # fmt: skip
    if generator == "block":
        return block_bootstrap(n_days, horizon, n_paths, mean_block, rng)
    if generator == "random_start":
        return random_start(n_days, horizon, n_paths, rng)
    raise ValueError(f"generator must be one of {GENERATORS}, got {generator!r}")


def _check_args(kind: str, n_paths: int, seed: int, horizon_eval: int, horizon_funded: int,
                stress_ticks: int, tick_cents: int | None, soft_daily_loss: str) -> None:  # fmt: skip
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if soft_daily_loss not in SOFT_DLL_MODES:
        raise ValueError(f"soft_daily_loss must be one of {SOFT_DLL_MODES}, got {soft_daily_loss!r}")
    for name, v in (("n_paths", n_paths), ("horizon_eval", horizon_eval), ("horizon_funded", horizon_funded)):
        if not isinstance(v, int) or v < 1:
            raise ValueError(f"{name} must be an integer >= 1, got {v!r}")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be an integer >= 0, got {seed!r}")
    if stress_ticks and tick_cents is None:
        raise ValueError("stress_ticks > 0 needs tick_cents")


def run_mc(plan: Any, days: DayPaths, *, n_paths: int, seed: int, generator: str = "block",
           mean_block: int = 5, horizon_eval: int = 252, horizon_funded: int = 252, stress_ticks: int = 0,
           tick_cents: int | None = None, kind: str, engine: Engine | None = None,
           chunk_paths: int = CHUNK_PATHS, soft_daily_loss: str = SOFT_DLL_LIMIT) -> MCResult:  # fmt: skip
    """The PROP_MC.md section 4 outputs for one plan on one DayPaths (see the module docstring).

    A rule value the evaluation needs raises RuleUnknown. One the funded phase or a fee needs nulls only the
    outputs that use it, with a warning naming the field. soft_daily_loss: "limit" (default, as
    upstream)
    or "minute_low" (sensitivity: a soft daily loss booked at the triggering minute's low)."""
    _check_args(kind, n_paths, seed, horizon_eval, horizon_funded, stress_ticks, tick_cents, soft_daily_loss)
    sha = getattr(plan, "sha256", None)
    if not isinstance(sha, str) or not sha:
        raise ValueError("the plan carries no sha256 (load it with pfb.prop.config.load_plan)")
    engine = engine or default_engine()
    check_day_paths(days)
    if stress_ticks:
        days = stress(days, stress_ticks, int(tick_cents))  # type: ignore[arg-type]
    h_e, h_f = eval_horizon(plan, horizon_eval), horizon_funded
    path_ss, se_ss = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.Generator(np.random.PCG64(path_ss))
    warnings = [STANDING_WARNING]
    soft = _soft_warning(plan, soft_daily_loss)
    if soft:
        warnings.append(soft)
    out, want_funded = [], True
    for start in range(0, n_paths, chunk_paths):
        size = min(chunk_paths, n_paths - start)
        seqs = make_sequences(generator, days.n_days, h_e + h_f, size, mean_block, rng)
        try:
            out.append(_run_chunk(plan, days, seqs, h_e, h_f, engine, want_funded, soft_daily_loss))
        except RuleUnknown as exc:
            if not want_funded:
                raise
            warnings.append(_unknown_warning(exc, FUNDED_OUTPUTS, "funded phase not run"))
            want_funded = False
            out.append(_run_chunk(plan, days, seqs, h_e, h_f, engine, want_funded, soft_daily_loss))
    po = _concat(out)
    fees = _fees(plan, po, engine, want_funded, warnings)
    se_rng = np.random.Generator(np.random.PCG64(se_ss))
    run = dict(kind=kind, plan_sha256=sha, seed=seed, generator=generator,
               mean_block=mean_block if generator == "block" else None, n_paths=n_paths, horizon_eval=h_e,
               horizon_funded=h_f, chunk_paths=chunk_paths, stress_ticks=stress_ticks, tick_cents=tick_cents,
               days_source=tuple(days.source), n_days=days.n_days, soft_daily_loss=soft_daily_loss,
               plan_file_sha256=getattr(plan, "file_sha256", None) or sha,
               plan_overrides=tuple(getattr(plan, "overrides", ())))  # fmt: skip
    return _assemble(run, plan, days, po, fees, want_funded, warnings, se_rng)


# --------------------------------------------------------------- fees ----
FUNDED_OUTPUTS = ("funded_ev", "p_breach_before_first_payout", "ev_per_purchase")


def _unknown_warning(exc: RuleUnknown, outputs: tuple[str, ...], what: str = "") -> str:
    tail = f" ({what})" if what else ""
    return f"rule unknown: {exc}; not computed: {', '.join(outputs)}{tail}"


@dataclass(frozen=True)
class _Fees:
    """Per-path fee arrays and the activation fee; None where a fee value is unknown (warned)."""

    first: np.ndarray | None  # the attempt as a purchase
    reset: np.ndarray | None  # the attempt as a retry after a failure other than a time limit
    funded: np.ndarray | None  # funded fees per path (0 where not passed)
    activation: int | None


def _try(fn: Callable[[], Any], outputs: tuple[str, ...], warnings: list[str]) -> Any:
    try:
        return fn()
    except RuleUnknown as exc:
        warnings.append(_unknown_warning(exc, outputs))
        return None


def _fees(plan: Any, po: _PathOutcomes, engine: Engine, want_funded: bool, warnings: list[str]) -> _Fees:
    """Each fee in its own try, so an unknown one nulls only the outputs that need it."""
    used = po.eval_days
    act = _try(lambda: int(plan_value(plan, "fees", "activation_cents")),
               ("cost_to_funded", "cost_to_funded_with_resets", "ev_per_purchase"), warnings)  # fmt: skip

    def attempt(first: bool) -> np.ndarray:
        return _fee_by_days(lambda d: engine.attempt_fee(plan, d, calendar_days(d), first=first), used)

    first = _try(lambda: attempt(True),
                 ("cost_per_attempt", "cost_to_funded", "cost_to_funded_with_resets", "ev_per_purchase"),
                 warnings)  # fmt: skip
    reset = _try(lambda: attempt(False), ("cost_to_funded_with_resets",), warnings)
    funded = None
    if want_funded:
        mask = po.funded_end >= 0

        def funded_fees() -> np.ndarray:
            arr = np.zeros(po.funded_days.shape[0], dtype=np.int64)
            fee = _fee_by_days(lambda d: engine.funded_fee(plan, d, calendar_days(d)), po.funded_days[mask])
            arr[mask] = fee
            return arr

        funded = _try(funded_fees, ("funded_ev", "ev_per_purchase"), warnings)
    return _Fees(first, reset, funded, act)


def _soft_warning(plan: Any, mode: str) -> str | None:
    """Names the soft daily-loss booking when the plan has a soft daily loss in either phase."""
    phases = []
    for section in ("eval", "funded"):
        try:
            if plan_value(plan, section, "daily_loss_kind") == "soft":
                phases.append(section)
        except (RuleUnknown, KeyError, TypeError, AttributeError):
            continue
    if not phases:
        return None
    if mode == SOFT_DLL_LOW:
        return (f"soft daily loss ({', '.join(phases)}) booked at the triggering minute's low "
                "(soft_daily_loss=minute_low, the pessimistic sensitivity)")  # fmt: skip
    return (f"soft daily loss ({', '.join(phases)}) booked at exactly -daily_loss, as upstream: "
            "optimistic when a minute's low is far below it "
            "(sensitivity: soft_daily_loss=minute_low)")  # fmt: skip


def _reset_window_warning(plan: Any) -> str | None:
    """A reset after a breach is modelled with a full access window (the firm may give only the days left)."""
    try:
        limit = plan_value(plan, "eval", "time_limit_calendar_days")
        reset = plan_value(plan, "fees", "reset_cents")
    except (RuleUnknown, KeyError, TypeError, AttributeError):
        return None
    if limit is None or reset is None:
        return None
    return (f"a reset is modelled with a full {int(limit)}-calendar-day access window (the firm may give "
            "only the days left); a retry after a time-limit end is priced as a new purchase")  # fmt: skip


# --------------------------------------------------------------- assemble ----
def _end_counts(ends: np.ndarray) -> dict[str, int]:
    return {END_NAMES.get(int(e), str(int(e))): int(c) for e, c in zip(*np.unique(ends, return_counts=True),
                                                                        strict=True)}  # fmt: skip


def _limit(plan: Any, section: str, name: str, pick: Callable[[Any], Any] = lambda v: v) -> int | None:
    try:
        value = pick(plan_value(plan, section, name))
    except (RuleUnknown, KeyError, TypeError, IndexError, AttributeError):
        return None
    return None if value is None else int(value)


def _size_warning(plan: Any, days: DayPaths, counts: dict[str, int]) -> str | None:
    """Share of pool days above the eval limit and above the first funded tier, and the size-limit ends."""
    shares = []
    for label, limit in (("eval max_contracts_minis", _limit(plan, "eval", "max_contracts_minis")),
                         ("the first funded scaling tier", _limit(plan, "funded", "scaling",
                                                                  lambda v: v[0][1]))):  # fmt: skip
        if limit is not None:
            share = float(np.mean(days.max_contracts > limit)) if days.n_days else 0.0
            shares.append((label, limit, share))
    ends = counts["eval_size_limit"] + counts["funded_size_limit"]
    if not ends and all(s == 0.0 for _, _, s in shares):
        return None
    parts = [f"{s:.1%} of pool days above {label} ({lim})" for label, lim, s in shares]
    return (
        f"size: {'; '.join(parts) or 'limits unknown'}; END_SIZE_LIMIT ends (counted apart from "
        f"breaches): {counts['eval_size_limit']} attempts, {counts['funded_size_limit']} funded lives"
    )


def _counts(po: _PathOutcomes, passed: np.ndarray) -> dict[str, int]:
    mask = passed == 1
    return {
        "eval_padded": int(po.eval_padded.sum()), "funded_padded": int(po.funded_padded[mask].sum()),
        "eval_size_limit": int((po.eval_end == END_SIZE_LIMIT).sum()),
        "funded_size_limit": int((po.funded_end[mask] == END_SIZE_LIMIT).sum()),
        "funded_open_no_payout": int(((po.funded_end[mask] == END_TIMEOUT)
                                      & (po.first_payout_day[mask] < 0)).sum()),
    }  # fmt: skip


def _flag(est: Estimate, lower: bool) -> Estimate:
    return replace(est, lower_bound=True) if lower and est.value is not None else est


def _assemble(run: dict[str, Any], plan: Any, days: DayPaths, po: _PathOutcomes, fees: _Fees,
              want_funded: bool, warnings: list[str], se_rng: np.random.Generator) -> MCResult:  # fmt: skip
    n_paths, h_e, h_f = run["n_paths"], run["horizon_eval"], run["horizon_funded"]
    passed = (po.eval_end == END_PASS).astype(np.int64)
    n_pass = int(passed.sum())
    counts = _counts(po, passed)
    _eval_warnings(po, counts, n_paths, h_e, n_pass, warnings)
    for extra in (_reset_window_warning(plan), _size_warning(plan, days, counts)):
        if extra:
            warnings.append(extra)
    costs = _cost_estimates(po, fees, passed)
    funded_ev, p_breach = _funded_estimates(po, passed, fees, want_funded, n_pass, h_f, counts, warnings)
    ev_purchase = _ev_per_purchase(po, fees, passed, want_funded, n_paths)
    eval_pad, fund_pad = counts["eval_padded"] > 0, counts["funded_padded"] > 0
    pass_days = po.eval_days[passed == 1]
    return MCResult(
        **run, p_pass=_flag(binomial(n_pass, n_paths), eval_pad),
        days_to_pass_median=quantile_bootstrap(pass_days, 0.5, se_rng),
        days_to_pass_p90=quantile_bootstrap(pass_days, 0.9, se_rng),
        cost_per_attempt=costs[0], cost_to_funded=costs[1], cost_to_funded_with_resets=costs[2],
        funded_ev=_flag(funded_ev, fund_pad), p_breach_before_first_payout=_flag(p_breach, fund_pad),
        ev_per_purchase=_flag(ev_purchase, eval_pad or fund_pad),
        eval_ends=_end_counts(po.eval_end),
        funded_ends=_end_counts(po.funded_end[passed == 1]) if want_funded else {},
        counts=counts, warnings=tuple(warnings),
    )  # fmt: skip


def _eval_warnings(po: _PathOutcomes, counts: dict[str, int], n_paths: int, h_e: int, n_pass: int,
                   warnings: list[str]) -> None:  # fmt: skip
    timeouts = int((po.eval_end == END_TIMEOUT).sum())
    if timeouts:
        warnings.append(f"{timeouts} of {n_paths} attempts timed out at the {h_e}-day horizon "
                        "(counted as not passed)")  # fmt: skip
    if counts["eval_padded"]:
        warnings.append(
            f"{counts['eval_padded']} of those timeouts ran into the end of history (-1 padding):"
            " the history is shorter than H_e + H_f; p_pass and ev_per_purchase are lower bounds"
        )
    if n_pass == 0:
        warnings.append("no attempt passed: cost_to_funded and funded outputs are undefined")


def _cost_estimates(po: _PathOutcomes, fees: _Fees, passed: np.ndarray) -> tuple[Estimate, ...]:
    n = passed.shape[0]
    if fees.first is None:
        need = "the attempt fee (unknown fee value)"
        return _missing(n, need), _missing(n, need), _missing(n, need)
    per_attempt = mean_se(fees.first)
    if fees.activation is None:
        return per_attempt, _missing(n, "fees.activation_cents"), _missing(n, "fees.activation_cents")
    ctf = cost_to_funded(fees.first, passed, fees.activation)
    if fees.reset is None:
        return per_attempt, ctf, _missing(n, "the reset fee (fees.reset_cents unknown)")
    time_limit = (po.eval_end == END_TIME_LIMIT).astype(np.int64)
    return per_attempt, ctf, cost_with_resets(fees.first, fees.reset, passed, time_limit, fees.activation)


def _ev_per_purchase(po: _PathOutcomes, fees: _Fees, passed: np.ndarray, want_funded: bool,
                     n_paths: int) -> Estimate:  # fmt: skip
    if not want_funded:
        return _missing(n_paths, "the funded phase (not run)")
    if fees.first is None or fees.funded is None or fees.activation is None:
        return _missing(n_paths, "every fee (one is unknown)")
    return mean_se(passed * (po.trader_cents - fees.activation - fees.funded) - fees.first)


def _funded_estimates(po: _PathOutcomes, passed: np.ndarray, fees: _Fees, want_funded: bool, n_pass: int,
                      h_f: int, counts: dict[str, int],
                      warnings: list[str]) -> tuple[Estimate, Estimate]:  # fmt: skip
    if not want_funded:
        return _missing(n_pass, "the funded phase (not run)"), _missing(n_pass, "the funded phase (not run)")
    mask = passed == 1
    open_ = int((po.funded_end[mask] == END_TIMEOUT).sum())
    if open_:
        warnings.append(f"{open_} of {n_pass} funded lives still open at the {h_f}-day horizon "
                        "(value after it not counted)")  # fmt: skip
    if counts["funded_padded"]:
        warnings.append(f"{counts['funded_padded']} of {n_pass} funded lives ran into the end of history "
                        "(-1 padding): funded_ev, p_breach_before_first_payout and ev_per_purchase are lower "
                        "bounds")  # fmt: skip
    breach_first = int(((po.funded_end[mask] == END_BREACH) & (po.first_payout_day[mask] < 0)).sum())
    if counts["funded_open_no_payout"] and n_pass:
        hi = (breach_first + counts["funded_open_no_payout"]) / n_pass
        warnings.append(f"{counts['funded_open_no_payout']} of {n_pass} funded lives were open with no "
                        "payout at the end: p_breach_before_first_payout lies in "
                        f"[{breach_first / n_pass:.4f}, {hi:.4f}]")  # fmt: skip
    p_breach = binomial(breach_first, n_pass)
    if fees.funded is None:
        return _missing(n_pass, "the funded fee (unknown fee value)"), p_breach
    return mean_se(po.trader_cents[mask] - fees.funded[mask]), p_breach
