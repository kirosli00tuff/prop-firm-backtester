"""The prop Monte Carlo of a saved run, as a service (docs/TRIALS.md section 6).

Moved here from pfb.cli.prop_cmd so that the CLI and pfb.app share one path: `run_single` runs one plan,
`run_matrix_cmd` every config/firms/<firm>/<plan>_<size>.toml. Run ids always start with their kind
("engine_check-", "synthetic-", "trial-"). The source run's kind is read from its begin record in the ledger
a run (pool members included) with no begin record is refused (RunNotFound), and a run id
whose prefix differs from that record's kind is refused, so a trial's folder copied under another name is
never free of the trial's restrictions. The root comes from that record (or summary.json) or the caller.
Nothing here counts toward N. A trial run is accepted only for what its hypothesis's [secondary] declares: a
declared plan (or the matrix when it declares ["*"]), at the declared prop_mc and prop_seed, with the default
generator, block and soft daily-loss booking, the root's own contracts per mini (MINIS_PER_TYPE, the value
pfb.runner.secondary uses) and no stress or pool; anything else is refused (TrialNotDeclared).
Outputs: runs/<run_id>/prop/<firm>_<plan>_<size>_<seed>.json (suffix _stress<k> under a cost stress,
_minute_low under that soft daily-loss booking and _pool-<sha8 of the pool ids> with a pool), and
runs/<run_id>/prop_matrix.json and .md for the matrix. With copies N (one plan, never on a
trial) the same name with _copies<N> holds pfb.prop.copies' document and the row gains a "copies" object.
a source folder whose files do not hash to its result record's artifact_sha256 is refused
(ArtifactMismatch; records written before the key are not checked), and on a trial a ledger other than the
registry's own or a firms directory other than the repository's config/firms is refused
(TrialInputsNotRepository).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pfb.instruments.table import InstrumentTableError, UnknownRoot, load_instruments
from pfb.prop.copies import copies_artifact_name, copies_row, evaluate_copies
from pfb.prop.days import DayPathsError, day_paths_from_run
from pfb.prop.matrix import (
    ALL_PLANS,
    MINIS_PER_TYPE,
    McSettings,
    PlanRefused,
    discover_plans,
    display_path,
    evaluate_plan,
    is_declared,
    matrix_document,
    matrix_markdown,
    plan_path,
    prop_row,
    run_matrix,
    secondary_row,
)
from pfb.prop.mc import GENERATORS
from pfb.prop.rules import SOFT_DLL_LIMIT, SOFT_DLL_MODES
from pfb.prop.types import DayPaths

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_KINDS = ("engine_check", "synthetic", "trial")
REPO_LEDGER = REPO_ROOT / "ledger" / "registry.jsonl"
REPO_FIRMS_DIR = REPO_ROOT / "config" / "firms"  # pfb.runner.secondary.FIRMS_DIR, a trial's secondary input
TRIAL_DEFAULTS = {"generator": "block", "block": 5, "stress_ticks": 0, "soft_daily_loss": SOFT_DLL_LIMIT}


class PropRefused(RuntimeError):
    """A refusal before any Monte Carlo: an error code, the reason and, when one is at fault, the field."""

    def __init__(self, reason: str, *, error: str = "Refused", field: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.error = error
        self.field = field

    @property
    def payload(self) -> dict[str, Any]:
        return {"error": self.error, "reason": self.reason, "field": self.field}


# ------------------------------------------------------------- source run ----
@dataclass(frozen=True)
class SourceRun:
    run_id: str
    run_dir: Path
    kind: str
    root: str
    declared: dict[str, Any] | None = None  # a trial's [secondary] table from its hypothesis record


def ledger_records(ledger: Path, key: str, value: str) -> list[dict[str, Any]]:
    """Every ledger record whose `key` equals `value` (a plain scan; the chain is verified elsewhere)."""
    if not ledger.is_file():
        return []
    out = []
    with ledger.open(encoding="utf-8") as fh:
        for line in fh:
            if f'"{value}"' in line:
                rec = json.loads(line)
                if rec.get(key) == value:
                    out.append(rec)
    return out


def resolve_source(run_id: str, runs_dir: Path, ledger: Path, root: str | None) -> SourceRun:
    """The run directory, its kind (from its begin record in the ledger; the run id's prefix must agree) and
    its root; PropRefused if any of them is missing or inconsistent."""
    if not run_id or "/" in run_id or run_id.startswith("."):
        raise PropRefused(f"run id {run_id!r} is not a plain name")
    prefix = run_id.split("-", 1)[0]
    if prefix not in RUN_KINDS:
        raise PropRefused(f"run id {run_id!r} does not start with a run kind {RUN_KINDS}")
    run_dir = runs_dir / run_id
    if not run_dir.is_dir():
        raise PropRefused(f"no run directory {run_dir}")
    records = ledger_records(ledger, "run_id", run_id)
    begin = next((r for r in records if r.get("kind") != "result"), None)
    result = next((r for r in records if r.get("kind") == "result"), None)
    if begin is None:
        raise PropRefused(f"the ledger has no begin record for {run_id!r} (a run's kind is read from it)",
                          error="RunNotFound")  # fmt: skip
    kind = str(begin.get("kind"))
    if kind != prefix:
        raise PropRefused(f"ledger kind {kind!r} differs from the run id's kind {prefix!r}")
    if result is not None and result.get("status") != "ok":
        raise PropRefused(f"the run's result record has status {result.get('status')!r}, not ok")
    _require_folder_matches(run_dir, run_id, result)
    ledger_root = _trial_root(run_dir, begin) if kind == "trial" else _ledger_root(run_dir, ledger)
    if root and ledger_root and root != ledger_root:
        raise PropRefused(f"--root {root} differs from the recorded root {ledger_root} for {run_id}")
    resolved = root or ledger_root
    if not resolved:
        raise PropRefused(f"root of {run_id} unknown (not in summary.json or the ledger): pass --root")
    declared = _declared_secondary(ledger, begin) if kind == "trial" else None
    return SourceRun(run_id, run_dir, kind, str(resolved), declared)


def _require_folder_matches(run_dir: Path, run_id: str, result: dict[str, Any] | None) -> None:
    """A folder whose files do not hash to its result record's artifact_sha256 is
    refused (ArtifactMismatch); a record written before the key carries no hashes and is not checked."""
    from pfb.registry.artifacts import require_folder_matches
    from pfb.registry.errors import ArtifactMismatch

    try:
        require_folder_matches(run_dir, run_id, result)
    except ArtifactMismatch as exc:
        raise PropRefused(str(exc), error="ArtifactMismatch") from exc


def _declared_secondary(ledger: Path, begin: dict[str, Any]) -> dict[str, Any]:
    """The [secondary] table of the trial's hypothesis record; refused when the ledger has none."""
    hyp_id = begin.get("hypothesis_id")
    for record in ledger_records(ledger, "hypothesis_id", str(hyp_id)):
        if record.get("kind") == "hypothesis":
            return dict(record["secondary"])
    raise PropRefused(f"hypothesis {hyp_id!r} of the trial is not in the ledger", error="TrialNotDeclared")


def _trial_root(run_dir: Path, begin: dict[str, Any]) -> str:
    """A trial's root is its begin record's, which fixed its contracts per mini; summary.json, an
    artifact file, may only agree with it."""
    root = begin.get("root")
    if not isinstance(root, str) or not root:
        raise PropRefused(f"the trial's begin record for {run_dir.name} names no root")
    from pfb.prop import pool

    summary = run_dir / pool.SUMMARY_FILE
    if summary.is_file():
        try:
            data = json.loads(summary.read_text())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PropRefused(f"{pool.SUMMARY_FILE} of {run_dir.name} is not JSON: {exc}") from exc
        named = data.get("root") if isinstance(data, dict) else None
        if named is not None and named != root:
            raise PropRefused(
                f"{pool.SUMMARY_FILE} of {run_dir.name} names root {named!r}; the trial's begin record "
                f"names {root!r}"
            )
    return root


def _ledger_root(run_dir: Path, ledger: Path) -> str | None:
    """pfb.prop.pool.run_root (summary.json "root", else the ledger); None when neither names one."""
    from pfb.prop import pool

    try:
        return pool.run_root(run_dir, ledger=ledger)
    except pool.PoolError:
        return None


@dataclass(frozen=True)
class RootFacts:
    contracts_per_mini: int
    tick_cents: int | None  # None when the tick value is not whole cents


def root_facts(root: str, contracts_per_mini: int | None) -> RootFacts:
    try:
        inst = load_instruments().get(root)
    except (UnknownRoot, InstrumentTableError) as exc:
        raise PropRefused(f"root {root!r}: {exc}") from exc
    per_mini = contracts_per_mini or MINIS_PER_TYPE.get(inst.contract_type)
    if per_mini is None:
        raise PropRefused(f"contract_type {inst.contract_type!r} of {root} has no mini equivalent here: pass "
                          "--contracts-per-mini")  # fmt: skip
    tick = inst.tick_value_cents
    return RootFacts(per_mini, int(tick) if tick.denominator == 1 else None)


@dataclass(frozen=True)
class Source:
    days: DayPaths
    kind: str
    run_ids: tuple[str, ...]
    tick_cents: int | None
    out_dir: Path


def build_source(run_id: str, pool: list[str], *, runs_dir: Path, ledger: Path, root: str | None,
                 contracts_per_mini: int | None, stress_ticks: int) -> Source:  # fmt: skip
    """DayPaths of the run (or of the pool: the run first, then the pool runs), its kind and tick cents."""
    first = resolve_source(run_id, runs_dir, ledger, root)
    runs = [first] + [resolve_source(r, runs_dir, ledger, None) for r in pool]
    kinds = {r.kind for r in runs}
    if len(kinds) > 1:
        raise PropRefused(f"a pool mixes run kinds {sorted(kinds)}")
    facts = [root_facts(r.root, contracts_per_mini) for r in runs]
    ticks = {f.tick_cents for f in facts}
    tick = ticks.pop() if len(ticks) == 1 else None
    if stress_ticks and tick is None:
        raise PropRefused("--stress-ticks needs one whole-cent tick value across the run(s); roots "
                          f"{sorted({r.root for r in runs})} do not share one")  # fmt: skip
    try:
        days = _days(runs, facts, ledger)
    except DayPathsError as exc:
        raise PropRefused(f"day paths: {exc}", error="DayPathsError") from exc
    return Source(days, first.kind, tuple(r.run_id for r in runs), tick, first.run_dir)


def _days(runs: list[SourceRun], facts: list[RootFacts], ledger: Path) -> DayPaths:
    if len(runs) == 1:
        return day_paths_from_run(runs[0].run_dir, contracts_per_mini=facts[0].contracts_per_mini)
    from pfb.prop import pool

    try:
        return pool.pool_runs([r.run_dir for r in runs], ledger=ledger,
                              contracts_per_mini=[f.contracts_per_mini for f in facts])  # fmt: skip
    except pool.PoolError as exc:
        raise PropRefused(f"pool: {exc}", error="PoolError") from exc


# ---------------------------------------------------------------- request ----
@dataclass(frozen=True)
class Request:
    """One `pfb prop` request. settings_args: mc, seed, block, generator, stress_ticks and, optionally,
    soft_daily_loss (default "limit")."""

    matrix: bool
    run_id: str
    pool: list[str]
    firm: str | None
    plan: str | None
    size: str | None
    settings_args: dict[str, Any]
    root: str | None
    contracts_per_mini: int | None
    runs_dir: Path
    firms_dir: Path
    ledger: Path
    registry_ledger: Path = REPO_LEDGER  # the ledger trials are begun in (pfb.app passes its deps' own)


def _arg(req: Request, name: str) -> Any:
    return req.settings_args.get(name, TRIAL_DEFAULTS.get(name))


def require_declared(source: SourceRun, req: Request) -> None:
    """TRIALS.md section 6: on a trial run only the declared plan(s), prop_mc and prop_seed, with defaults and
    the root's own contracts per mini (the value the runner's secondary output used)."""
    declared = source.declared
    if source.kind != "trial" or declared is None:
        return
    per_mini = root_facts(source.root, None).contracts_per_mini
    plans = declared["prop_plans"]
    if req.matrix:
        planned = plans == [ALL_PLANS]
    else:
        firm, plan, size = req.firm or "", req.plan or "", req.size or ""
        planned = bool(firm and plan and size) and is_declared(plans, firm, plan, size)
    if (
        not planned
        or _arg(req, "mc") != declared["prop_mc"]
        or _arg(req, "seed") != declared["prop_seed"]
        or any(_arg(req, k) != v for k, v in TRIAL_DEFAULTS.items())
        or req.contracts_per_mini not in (None, per_mini)
        or req.pool
        or req.settings_args.get("copies") is not None  # a hypothesis cannot declare copies
    ):
        raise PropRefused(
            f"a trial's prop outputs are the declared ones only: plans {plans}, mc {declared['prop_mc']}, "
            f"seed {declared['prop_seed']}, contracts per mini {per_mini} ({source.root}), default "
            "generator, block and soft daily-loss booking, no stress, no pool, no copies",
            error="TrialNotDeclared",
        )


def settings_of(source: Source, req: Request) -> McSettings:
    """The Monte Carlo settings of the request, checked."""
    generator, soft = _arg(req, "generator"), _arg(req, "soft_daily_loss")
    if generator not in GENERATORS:
        raise PropRefused(f"--generator must be one of {GENERATORS}", field="generator")
    if soft not in SOFT_DLL_MODES:
        raise PropRefused(f"--soft-daily-loss must be one of {SOFT_DLL_MODES}", field="soft_daily_loss")
    for name in ("mc", "block"):
        if _arg(req, name) < 1:
            raise PropRefused(f"--{name} must be >= 1", field=name)
    if _arg(req, "seed") < 0 or _arg(req, "stress_ticks") < 0:
        raise PropRefused("--seed and --stress-ticks must be >= 0")
    stress = _arg(req, "stress_ticks")
    return McSettings(n_paths=_arg(req, "mc"), seed=_arg(req, "seed"), kind=source.kind, generator=generator,
                      mean_block=_arg(req, "block"), stress_ticks=stress,
                      tick_cents=source.tick_cents if stress else None, soft_daily_loss=soft)  # fmt: skip


def artifact_name(firm: str, plan: str, size: str, settings: McSettings, pool: tuple[str, ...]) -> str:
    """<firm>_<plan>_<size>_<seed>.json, plus _stress<k>, _minute_low and _pool-<sha8> where they apply."""
    stress = f"_stress{settings.stress_ticks}" if settings.stress_ticks else ""
    soft = f"_{settings.soft_daily_loss}" if settings.soft_daily_loss != SOFT_DLL_LIMIT else ""
    pooled = f"_pool-{hashlib.sha256(' '.join(pool).encode()).hexdigest()[:8]}" if pool else ""
    return f"{firm}_{plan}_{size}_{settings.seed}{stress}{soft}{pooled}.json"


# ----------------------------------------------------------------- output ----
def write_json(path: Path, obj: Any) -> None:
    """Write via a temporary file in the same directory, then rename (no half-written artifact)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, sort_keys=True, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def _write_text(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


@dataclass(frozen=True)
class PropOutcome:
    """What one request produced: the summary object (single: the compact record; matrix: the matrix
    document without its full rows), the result-contract prop rows and the files written (names relative to
    the run directory)."""

    kind: str
    summary: dict[str, Any]
    rows: list[dict[str, Any]]
    artifacts: list[str]


def require_repository_inputs(source: SourceRun, req: Request) -> None:
    """On a trial, the ledger must be the one trials are begun in (the repository's
    ledger/registry.jsonl unless pfb.app runs on other deps) and the firms directory the repository's
    config/firms, the one the trial's secondary output read (TrialInputsNotRepository otherwise)."""
    if source.kind != "trial":
        return
    if (
        req.ledger.resolve() != req.registry_ledger.resolve()
        or req.firms_dir.resolve() != REPO_FIRMS_DIR.resolve()
    ):
        raise PropRefused(
            "a trial's prop outputs read the registry's own ledger and the repository's config/firms only; "
            "--ledger and --firms-dir must be left at their defaults",
            error="TrialInputsNotRepository",
        )


def _source(req: Request) -> Source:
    source = resolve_source(req.run_id, req.runs_dir, req.ledger, req.root)
    require_declared(source, req)
    require_repository_inputs(source, req)
    return build_source(req.run_id, req.pool, runs_dir=req.runs_dir, ledger=req.ledger, root=req.root,
                        contracts_per_mini=req.contracts_per_mini,
                        stress_ticks=_arg(req, "stress_ticks"))  # fmt: skip


def run_single(req: Request, repo_root: Path = REPO_ROOT) -> PropOutcome:
    """One plan: the compact record (also written with the full MCResult under "mc") and its artifact."""
    if not (req.firm and req.plan and req.size):
        raise PropRefused("a single-plan request needs firm, plan and size")
    path = plan_path(req.firms_dir, req.firm, req.plan, req.size)
    source = _source(req)
    settings = settings_of(source, req)
    t0 = time.perf_counter()
    n_copies: int | None = req.settings_args.get("copies")
    copies_doc = None
    if n_copies is None:
        record, result = evaluate_plan(path, source.days, settings, repo_root=repo_root)
    else:  # the same Monte Carlo, its per-path outcomes kept for N accounts (pfb.prop.copies)
        copies_doc, record, result = evaluate_copies(
            path, source.days, settings, n_copies, repo_root=repo_root
        )
    out = source.out_dir / "prop" / artifact_name(req.firm, req.plan, req.size, settings, source.run_ids[1:])
    shown = display_path(out, repo_root)
    record = {**record, "run_id": req.run_id, "pool": list(source.run_ids[1:]), "status": "ok",
              "seconds": round(time.perf_counter() - t0, 3), "artifacts": [shown]}  # fmt: skip
    write_json(out, {**record, "mc": json.loads(result.to_json())})
    row = prop_row(req.firm, f"{req.plan}_{req.size}", "ok", None, record)
    names = [f"prop/{out.name}"]
    if copies_doc is not None and n_copies is not None:
        cout = out.with_name(copies_artifact_name(out.name, n_copies))
        write_json(cout, {**copies_doc, "run_id": req.run_id, "pool": list(source.run_ids[1:])})
        summary = copies_row(copies_doc)
        record = {**record, "artifacts": [shown, display_path(cout, repo_root)], "copies": summary}
        row = {**row, "copies": summary}
        names.append(f"prop/{cout.name}")
    return PropOutcome(source.kind, record, [row], names)


def run_matrix_cmd(
    req: Request, repo_root: Path = REPO_ROOT, progress: Callable[[str], None] | None = None
) -> PropOutcome:
    """Every plan file: prop_matrix.json and .md in the run directory."""
    source = _source(req)
    paths = discover_plans(req.firms_dir)
    settings = settings_of(source, req)
    t0 = time.perf_counter()
    rows = run_matrix(source.days, paths, settings, repo_root=repo_root, progress=progress)
    header = {"run_id": req.run_id, "pool": list(source.run_ids[1:]), "kind": source.kind,
              "trial": source.kind == "trial",
              "n_paths": settings.n_paths, "seed": settings.seed, "generator": settings.generator,
              "mean_block": settings.mean_block if settings.generator == "block" else None,
              "stress_ticks": settings.stress_ticks, "tick_cents": settings.tick_cents,
              "soft_daily_loss": settings.soft_daily_loss,
              "firms_dir": display_path(req.firms_dir, repo_root),
              "seconds": round(time.perf_counter() - t0, 3)}  # fmt: skip
    doc = matrix_document(rows, header)
    json_path, md_path = source.out_dir / "prop_matrix.json", source.out_dir / "prop_matrix.md"
    write_json(json_path, doc)
    _write_text(md_path, matrix_markdown(doc))
    shown = [display_path(json_path, repo_root), display_path(md_path, repo_root)]
    summary_rows = [{"firm": r.firm, "plan": r.plan, "size": r.size, "status": r.status, "reason": r.reason,
                     "field": r.field,
                     "outputs": r.record["outputs"] if r.record else None} for r in rows]  # fmt: skip
    summary = {**{k: v for k, v in doc.items() if k != "rows"}, "status": "ok", "rows": summary_rows,
               "artifacts": shown}  # fmt: skip
    return PropOutcome(source.kind, summary, [secondary_row(r) for r in rows], [json_path.name, md_path.name])


def run_request(req: Request, repo_root: Path = REPO_ROOT,
                progress: Callable[[str], None] | None = None) -> PropOutcome:  # fmt: skip
    """The matrix or one plan, as the request says; PlanRefused becomes PropRefused."""
    if req.matrix and req.settings_args.get("copies") is not None:
        raise PropRefused("copies runs on one plan: drop the matrix or --copies", field="copies")
    try:
        return run_matrix_cmd(req, repo_root, progress) if req.matrix else run_single(req, repo_root)
    except PlanRefused as exc:
        raise PropRefused(exc.reason, error=exc.error, field=exc.field) from exc
