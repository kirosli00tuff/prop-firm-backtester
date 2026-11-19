"""The firm-by-plan EV matrix and the compact per-plan JSON behind `pfb prop` (docs/PROP_MC.md).

evaluate_plan loads one pfb-rules-1 file, checks the contract caps, runs run_mc and returns the compact
record (each PROP_MC.md section 4 output with its SE and n, plus seed, kind, plan sha256, warnings). A plan
that cannot be run is refused with PlanRefused (the reason and, for an unknown rule value, the dotted
field); run_matrix records such plans as skipped rows with that reason. Money is integer cents in the
inputs; the estimates are means, so they are floats in cents. Nothing here registers a trial.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pfb.prop.config import PlanConfigError, PlanRules, load_versions
from pfb.prop.mc import SE_SCOPE, Engine, Estimate, MCResult, run_mc
from pfb.prop.rules import RULES_NEWEST, SOFT_DLL_LIMIT
from pfb.prop.types import DayPaths, RuleUnknown

CLI_SCHEMA = "pfb.prop_cli/1"
MATRIX_SCHEMA = "pfb.prop_matrix/1"
OUTPUTS = (
    "p_pass", "days_to_pass_median", "days_to_pass_p90", "cost_per_attempt", "cost_to_funded",
    "cost_to_funded_with_resets", "funded_ev", "p_breach_before_first_payout", "ev_per_purchase",
)  # fmt: skip
PROBABILITIES = frozenset({"p_pass", "p_breach_before_first_payout"})
DAY_COUNTS = frozenset({"days_to_pass_median", "days_to_pass_p90"})
UNITS = "money in cents (USD x 100); days in trading days; p_* are probabilities"
ROUND_DIGITS = 6
PAYOUT_POLICY = (
    'funded_ev is the value of the "largest request at the first eligible close" payout policy '
    "(docs/PROP_RULES_SCHEMA.md section 8), not of the plan under every policy"
)  # fmt: skip
CAPS_UNCHECKED = (
    "contract caps not pre-checked: pfb.prop.pool is absent (the rule engine still applies per-day caps)"
)


class PlanRefused(RuntimeError):
    """A plan cannot be run: ``reason`` says why; ``field`` names the unknown rule value when that is why."""

    def __init__(self, reason: str, *, field: str | None = None, error: str = "PlanRefused") -> None:
        super().__init__(reason)
        self.reason = reason
        self.field = field
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {"error": self.error, "reason": self.reason, "field": self.field}


@dataclass(frozen=True)
class McSettings:
    """The run_mc parameters shared by every plan of one command."""

    n_paths: int
    seed: int
    kind: str
    generator: str = "block"
    mean_block: int = 5
    stress_ticks: int = 0
    tick_cents: int | None = None
    horizon_eval: int = 252
    horizon_funded: int = 252
    soft_daily_loss: str = SOFT_DLL_LIMIT  # "limit" (as upstream) or "minute_low" (sensitivity)


CapCheck = Callable[[DayPaths, PlanRules], list[str]]


# ------------------------------------------------------------------- caps ----
def first_tier_warning(days: DayPaths, plan: PlanRules) -> list[str]:
    """A warning when pool days exceed the first funded scaling tier (the limit a new funded account opens
    with; those days end a funded life as END_SIZE_LIMIT until the balance reaches a higher tier)."""
    try:
        tier = int(plan.value("funded.scaling")[0][1])
    except RuleUnknown:
        return []
    over = int((days.max_contracts > tier).sum())
    if not over:
        return []
    return [f"contract caps: {over} of {days.n_days} pool days ({over / days.n_days:.1%}) exceed the first "
            f"funded scaling tier ({tier} minis); such a day ends a funded life as END_SIZE_LIMIT while the "
            "balance is in that tier"]  # fmt: skip


def default_cap_check(days: DayPaths, plan: PlanRules) -> list[str]:
    """pfb.prop.pool.check_caps for the eval and the funded phase (top tier) when the module exists
    (PoolOverLimit becomes PlanRefused), then a warning for days above the first funded tier; a warning
    when the pool module is absent."""
    try:
        from pfb.prop import pool  # pyright: ignore[reportAttributeAccessIssue]
    except ImportError:
        return [CAPS_UNCHECKED, *first_tier_warning(days, plan)]
    for phase in ("eval", "funded"):
        try:
            pool.check_caps(days, plan, phase=phase)
        except pool.PoolOverLimit as exc:
            raise PlanRefused(f"contract caps: {exc}", error="PoolOverLimit") from exc
    return first_tier_warning(days, plan)


# ------------------------------------------------------------------ plans ----
def plan_path(firms_dir: Path, firm: str, plan: str, size: str) -> Path:
    """config/firms/<firm>/<plan>_<size>.toml; PlanRefused (listing the firm's files) when absent."""
    for name, value in (("firm", firm), ("plan", plan), ("size", size)):
        if not value or "/" in value or "\\" in value or value.startswith("."):
            raise PlanRefused(f"--{name} {value!r} is not a plain name")
    path = firms_dir / firm / f"{plan}_{size}.toml"
    if not path.is_file():
        have = (
            sorted(p.stem for p in (firms_dir / firm).glob("*.toml")) if (firms_dir / firm).is_dir() else []
        )
        raise PlanRefused(f"no plan file {path.name} for firm {firm!r} (have: {have or 'none'})")
    return path


def discover_plans(firms_dir: Path) -> list[Path]:
    """Every <firm>/<plan>_<size>.toml under firms_dir, sorted by firm then file name."""
    if not firms_dir.is_dir():
        raise PlanRefused(f"firms directory {firms_dir} does not exist")
    return sorted(p for p in firms_dir.glob("*/*.toml") if p.is_file())


def display_path(path: Path, repo_root: Path) -> str:
    """Repo-relative when inside repo_root, else absolute."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), ROUND_DIGITS)


def compact_estimate(est: Estimate) -> dict[str, Any]:
    out: dict[str, Any] = {"value": _rounded(est.value), "se": _rounded(est.se), "n": est.n}
    if est.note:
        out["note"] = est.note
    if est.lower_bound:
        out["lower_bound"] = True
    return out


def compact_result(
    result: MCResult, plan: PlanRules, plan_display: str, extra_warnings: Sequence[str]
) -> dict:
    """The compact record of one plan (CLI_SCHEMA); the full MCResult is kept under "mc"."""
    return {
        "schema": CLI_SCHEMA, "firm": plan.firm, "plan": plan.plan, "size": plan.size, "label": plan.label,
        "plan_status": plan.status, "plan_path": plan_display, "plan_sha256": result.plan_sha256,
        "kind": result.kind, "trial": False, "seed": result.seed, "n_paths": result.n_paths,
        "generator": result.generator, "mean_block": result.mean_block, "horizon_eval": result.horizon_eval,
        "horizon_funded": result.horizon_funded, "stress_ticks": result.stress_ticks,
        "tick_cents": result.tick_cents, "days_source": list(result.days_source), "n_days": result.n_days,
        "units": UNITS, "outputs": {name: compact_estimate(getattr(result, name)) for name in OUTPUTS},
        "eval_ends": result.eval_ends, "funded_ends": result.funded_ends, "counts": result.counts,
        "se_scope": result.se_scope, "payout_policy": PAYOUT_POLICY,
        "soft_daily_loss": result.soft_daily_loss, "plan_file_sha256": result.plan_file_sha256,
        "plan_overrides": [list(pair) for pair in result.plan_overrides],
        "plan_version": plan.version, "effective_from": plan.effective_from, "rules_read": plan.rules_read,
        "rules_in_force": RULES_NEWEST,
        "warnings": [*result.warnings, *extra_warnings, *plan.unknown_notes()],
    }  # fmt: skip


def undated_rules_warning(days: DayPaths, plan: PlanRules) -> str:
    """: a resampled (undated) path is charged under the newest rules version, and says so."""
    early = int((days.trade_dates < plan.effective_from_int).sum())
    return (f"rules: resampled day paths are undated, so every day is charged under the newest rules version "
            f"(version {plan.version}, effective_from {plan.effective_from}); {early} of {days.n_days} "
            "source days predate it")  # fmt: skip


def _load(path: Path) -> PlanRules:
    """The plan file, checked to be the newest of its dated versions (undated runs use the newest)."""
    try:
        return load_versions(path).newest
    except PlanConfigError as exc:
        raise PlanRefused(f"invalid plan file: {exc}", error="PlanConfigError") from exc


def unknown_field(exc: RuleUnknown) -> str:
    """The dotted path a RuleUnknown names (its message is '<path>: <reason>')."""
    return str(exc).split(":", 1)[0].strip()


def evaluate_plan(path: Path, days: DayPaths, settings: McSettings, *, repo_root: Path,
                  engine: Engine | None = None,
                  cap_check: CapCheck = default_cap_check) -> tuple[dict, MCResult]:  # fmt: skip
    """Load, check caps, run the Monte Carlo: (compact record, MCResult); PlanRefused if it cannot run."""
    plan = _load(path)
    try:
        extra = [*cap_check(days, plan), undated_rules_warning(days, plan)]
        result = run_mc(
            plan, days, n_paths=settings.n_paths, seed=settings.seed, generator=settings.generator,
            mean_block=settings.mean_block, horizon_eval=settings.horizon_eval,
            horizon_funded=settings.horizon_funded, stress_ticks=settings.stress_ticks,
            tick_cents=settings.tick_cents, kind=settings.kind, engine=engine,
            soft_daily_loss=settings.soft_daily_loss,
        )  # fmt: skip
    except RuleUnknown as exc:
        field = unknown_field(exc)
        raise PlanRefused(f"rule value unknown: {exc}", field=field, error="RuleUnknown") from exc
    return compact_result(result, plan, display_path(path, repo_root), extra), result


# ----------------------------------------------------------------- matrix ----
@dataclass(frozen=True)
class MatrixRow:
    firm: str
    plan: str
    size: str
    plan_path: str
    status: str  # "ok" or "skipped"
    reason: str | None
    field: str | None
    seconds: float
    record: dict | None  # the compact record when ok


def _names(path: Path) -> tuple[str, str, str]:
    """(firm, plan, size) from <firm>/<plan>_<size>.toml (the size is the text after the last underscore)."""
    plan, _, size = path.stem.rpartition("_")
    return path.parent.name, plan or path.stem, size


def run_matrix(days: DayPaths, plan_paths: Sequence[Path], settings: McSettings, *, repo_root: Path,
               engine: Engine | None = None, cap_check: CapCheck = default_cap_check,
               progress: Callable[[str], None] | None = None) -> list[MatrixRow]:  # fmt: skip
    """evaluate_plan on every path; a refused plan is a skipped row with its reason."""
    rows: list[MatrixRow] = []
    for path in plan_paths:
        firm, plan, size = _names(path)
        shown = display_path(path, repo_root)
        t0 = time.perf_counter()
        try:
            record, _ = evaluate_plan(path, days, settings, repo_root=repo_root, engine=engine,
                                      cap_check=cap_check)  # fmt: skip
            row = MatrixRow(record["firm"], record["plan"], record["size"], shown, "ok", None, None,
                            round(time.perf_counter() - t0, 3), record)  # fmt: skip
        except PlanRefused as exc:
            row = MatrixRow(firm, plan, size, shown, "skipped", exc.reason, exc.field,
                            round(time.perf_counter() - t0, 3), None)  # fmt: skip
        if progress is not None:
            progress(f"{row.firm}/{row.plan}_{row.size}: {row.status} in {row.seconds:.1f}s")
        rows.append(row)
    return rows


def matrix_document(rows: Sequence[MatrixRow], header: dict[str, Any]) -> dict[str, Any]:
    """The prop_matrix.json object: the header (run, settings), the rows, and the counts."""
    ok = sum(1 for r in rows if r.status == "ok")
    return {**header, "schema": MATRIX_SCHEMA, "units": UNITS, "payout_policy": PAYOUT_POLICY,
            "se_scope": SE_SCOPE, "n_plans": len(rows), "n_ok": ok, "n_skipped": len(rows) - ok,
            "rows": [asdict(r) for r in rows]}  # fmt: skip


def _fmt(name: str, est: dict[str, Any]) -> str:
    value, se = est.get("value"), est.get("se")
    if value is None or se is None:
        return "n/a"
    mark = " lb" if est.get("lower_bound") else ""
    if name in PROBABILITIES:
        return f"{value:.4f} ({se:.4f}){mark}"
    if name in DAY_COUNTS:
        return f"{value:.0f} ({se:.1f}){mark}"
    return f"{value / 100:,.0f} ({se / 100:,.0f}){mark}"


MD_COLUMNS = (
    ("p_pass", "p_pass"), ("days_to_pass_median", "days med"), ("days_to_pass_p90", "days p90"),
    ("cost_per_attempt", "cost/attempt $"), ("cost_to_funded", "cost to funded $"),
    ("cost_to_funded_with_resets", "with resets $"), ("funded_ev", "funded EV $"),
    ("p_breach_before_first_payout", "p breach pre-payout"), ("ev_per_purchase", "EV/purchase $"),
)  # fmt: skip


def matrix_markdown(doc: dict[str, Any]) -> str:
    """Rows firm/plan/size, columns the outputs as value (SE); money in whole USD; skipped plans listed."""
    head = ["firm", "plan", "size", *(label for _, label in MD_COLUMNS)]
    lines = [
        f"# Prop matrix: {doc.get('run_id')}", "",
        f"kind {doc.get('kind')}, n_paths {doc.get('n_paths')}, seed {doc.get('seed')}, generator "
        f"{doc.get('generator')} (mean block {doc.get('mean_block')}), "
        f"stress_ticks {doc.get('stress_ticks')}, soft_daily_loss {doc.get('soft_daily_loss', 'limit')}. "
        "Each cell is value (SE); lb marks a lower bound (paths cut by the end of history). Not a trial.", "",
        f"Payout policy: {PAYOUT_POLICY}.", f"SEs: {SE_SCOPE}.", "",
        "| " + " | ".join(head) + " |", "|" + "---|" * len(head),
    ]  # fmt: skip
    rows = doc["rows"]
    for r in rows:
        if r["status"] != "ok":
            continue
        outs = r["record"]["outputs"]
        cells = [r["firm"], r["plan"], r["size"], *(_fmt(n, outs[n]) for n, _ in MD_COLUMNS)]
        lines.append("| " + " | ".join(cells) + " |")
    skipped = [r for r in rows if r["status"] != "ok"]
    if skipped:
        lines += ["", "Skipped plans:", ""]
        lines += [f"- {r['firm']}/{r['plan']}_{r['size']}: {r['reason']}" for r in skipped]
    warned = sorted({w for r in rows if r["status"] == "ok" for w in r["record"]["warnings"]
                     if not w[:1].isdigit()})  # fmt: skip
    if warned:
        lines += ["", "Warnings (count-free ones, all rows):", ""] + [f"- {w}" for w in warned]
    return "\n".join(lines) + "\n"


# ------------------------------------------------- secondary output of a trial (docs/TRIALS.md s6) ----
SECONDARY_SCHEMA = "pfb.prop_secondary/1"
ALL_PLANS = "*"
MINIS_PER_TYPE = {"mini": 1, "micro": 10}  # contracts per mini equivalent by contract_type


def declared_plan_paths(firms_dir: Path, plans: Sequence[str]) -> list[Path]:
    """The plan files a hypothesis's [secondary] prop_plans names: every file for ["*"], else
    <firm>/<plan file stem>.toml each (PlanRefused when one is absent)."""
    if list(plans) == [ALL_PLANS]:
        return discover_plans(firms_dir)
    out: list[Path] = []
    for name in plans:
        firm, _, stem = name.partition("/")
        path = firms_dir / firm / f"{stem}.toml"
        if not firm or not stem or not path.is_file():
            raise PlanRefused(f"declared plan {name!r} has no file <firm>/<plan>.toml under {firms_dir.name}")
        out.append(path)
    return out


def is_declared(plans: Sequence[str], firm: str, plan: str, size: str) -> bool:
    """Whether `pfb prop` on a trial asks for a plan its hypothesis declared (["*"] declares every plan)."""
    return list(plans) == [ALL_PLANS] or f"{firm}/{plan}_{size}" in plans


def prop_row(
    firm: str, plan_stem: str, status: str, reason: str | None, record: dict | None
) -> dict[str, Any]:
    """One row of the result contract's prop list (docs/TRIALS.md section 7): firm, plan (the file stem
    <plan>_<size>), p_pass and EV per purchase with their SEs, plus the row's status and reason and the
    settings it ran at (n_paths, seed, soft_daily_loss), read from the compact record (None: skipped)."""
    rec = record or {}
    outputs = rec.get("outputs", {})
    p_pass, ev = outputs.get("p_pass", {}), outputs.get("ev_per_purchase", {})
    return {
        "firm": firm, "plan": plan_stem, "status": status, "reason": reason,
        "p_pass": p_pass.get("value"), "p_pass_se": p_pass.get("se"),
        "ev_per_purchase_cents": ev.get("value"), "ev_se_cents": ev.get("se"),
        "n_paths": rec.get("n_paths"), "seed": rec.get("seed"), "soft_daily_loss": rec.get("soft_daily_loss"),
        "plan_version": rec.get("plan_version"), "effective_from": rec.get("effective_from"),
        "plan_file_sha256": rec.get("plan_file_sha256"), "rules_in_force": rec.get("rules_in_force"),
    }  # fmt: skip


def secondary_row(row: MatrixRow) -> dict[str, Any]:
    """One row of the result contract's prop list for a matrix row (see prop_row)."""
    return prop_row(row.firm, f"{row.plan}_{row.size}", row.status, row.reason, row.record)
