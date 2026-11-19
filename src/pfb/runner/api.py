"""pfb.runner.run: the one entry that runs a strategy and logs it (ADR A1, A2, A11, A14; docs/TRIALS.md).

Order: (1) lint, load, params resolved against the SPEC grid (an off-grid value is refused here, for every
kind), truncation check on synthetic bars; (2) resolve the configuration and every hash (data
hashes from DataGateway.files_for, no bar read); (3)-(4) registry begin, which checks the kind's
preconditions and appends and fsyncs the begin record; (5) the engine runs (a real-data run carries a
RunPermit built from the receipt); (6) registry finish appends the result record BEFORE anything is printed
or written; (7) artifacts are written. A crash after begin still counts and leaves the run "open or
crashed" (G7). Kinds: engine_check (real data, allowlisted), synthetic, and trial (TRIALS.md C6).
A gateway refusal (pfb.data GatewayRefusal, PathsConfigError included) raised before begin is re-raised as
RunRefused: nothing was logged. After begin it is the run's failure: a failed result record, then re-raised.
The outcome carries the sha256 of the run's two ledger lines (begin_sha256, result_sha256).
Artifact names in the result record are repo-relative inside the repository and "external:<runs_dir
name>/<run_id>/<file>" outside it, never an absolute path.
A real-data run whose store roots overlap a real store location (the roots of
config/paths.toml) is refused before begin unless its ledger is this repository's ledger/registry.jsonl.
Warnings: step 1's look-ahead warnings are prefixed "step 1: ", go into the begin record
and are appended to the result's summary warnings; the engine's in-run look-ahead warning is prefixed
"in-run: ". lookahead_signal is "none" when step 1 saw no signal, else "seen".
A trial (pfb.runner.trial) takes root, window, session, cost model, fees, suite and secondary outputs from its
hypothesis; it checkpoints the ledger before begin when needed, after begin and at the end; its DSR uses the
grid of the hypothesis's "ok" configurations; its declared prop plans run after the artifacts.
Seed: a trial runs at its hypothesis's lookahead_seed (step 1, the engine, the begin record);
a seed given for a trial that differs from it is refused as SeedMismatch before step 1, so before any begin.
The other kinds take the seed given, DEFAULT_SEED (the program seed) when none is. The seed is resolved once,
in prepare, and carried by the Plan; execute_planned runs at the Plan's seed.
Artifact hashes: every artifact is serialized in memory before the result record
(artifact_files); the record's artifact_sha256 names each with the sha256 of those bytes, and exactly those
bytes are written after it, so the order "result record before anything is written" holds.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, is_dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

from pfb.data.errors import GatewayRefusal
from pfb.data.guards import real_store_locations
from pfb.data.paths import DEFAULT_PATHS_TOML
from pfb.engine.strategy import StrategyError, resolve_params
from pfb.registry.api import (
    LEDGER_REL,
    Receipt,
    RegistryConfig,
    RunRequest,
    begin,
    checkpoint,
    finish,
    read_state,
)
from pfb.registry.artifacts import bytes_sha256
from pfb.registry.chain import LedgerLine
from pfb.registry.errors import BadRequest, EngineCheckNotDeclared, SeedMismatch
from pfb.registry.hypfile import PROGRAM_LOOKAHEAD_SEED
from pfb.registry.inputs import EngineCheckEntry, load_engine_checks
from pfb.registry.records import (
    ENGINE_CHECK,
    EVENT_WINDOWS_OFF,
    EVENT_WINDOWS_ON,
    SIGNAL_NONE,
    SIGNAL_SEEN,
    STATUS_FAILED,
    STATUS_OK,
    SYNTHETIC,
    TRIAL,
)
from pfb.runner.deps import DataFiles, RunnerDeps, default_deps

RUNNER_KINDS = (ENGINE_CHECK, SYNTHETIC, TRIAL)
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"
FRAME_ARTIFACTS = ("trades", "equity_minute", "daily")
SUMMARY_ARTIFACT = "summary.json"
SUITE_ARTIFACT = "suite.json"
SUITE_KEY = "suite"
EXTERNAL_PREFIX = "external:"
METRICS_NOT_JSON = "MetricsNotJson"
PATHS_TOML = DEFAULT_PATHS_TOML  # names the real store roots (tests point it at a temporary file)
REPO_LEDGER = REPO_ROOT / LEDGER_REL
SUMMARY_WARNINGS = "warnings"
NO_SIGNAL_TEXT = "look-ahead check saw no signal"  # pfb.engine.lookahead.NO_SIGNAL_WARNING
LOOKAHEAD_TEXT = "look-ahead check"
STEP1_PREFIX = "step 1: "
INRUN_PREFIX = "in-run: "
SECONDARY_FAILED = "secondary prop outputs failed"
DEFAULT_SEED = PROGRAM_LOOKAHEAD_SEED  # one program-wide look-ahead seed
EVENT_COSTS_OFF, EVENT_COSTS_ON = EVENT_WINDOWS_OFF, EVENT_WINDOWS_ON  # a hypothesis's cost.event_costs


def deflation_n(n_after: int) -> int:
    """The N a run's DSR is deflated at: the registry's N after its begin record, and at least 1 (the run
    itself) on a fresh ledger where no trial has been counted yet (N = 0)."""
    return max(int(n_after), 1)


class RunRefused(RuntimeError):
    """The runner refused before begin: unknown kind, failed lint or look-ahead, or a bad argument."""


class ProvenanceMismatch(RuntimeError):
    """The files the engine read differ from those files_for named before the run (A14 d)."""


@dataclass(frozen=True)
class RunOutcome:
    """The engine's result and the run's two ledger records with the sha256 of each line (no newline).
    secondary is a trial's prop index (None for other kinds); warnings are the runner's own (a failed
    secondary output), not the engine's."""

    result: Any
    begin_record: Mapping[str, Any]
    result_record: Mapping[str, Any]
    begin_sha256: str
    result_sha256: str
    secondary: Mapping[str, Any] | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Plan:
    root: str
    first: date | None
    last: date | None
    params: dict[str, Any]
    fees: str
    cost_model: str
    data_files: DataFiles
    check_id: str | None
    lookahead_warnings: tuple[str, ...] = ()
    hypothesis_id: str | None = None
    hypothesis_path: Path | None = None
    suite: Any = None  # pfb.analysis.spec.SuiteSpec or None
    secondary: Mapping[str, Any] | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    seed: int = DEFAULT_SEED  # the run's one seed: step 1, the engine and the begin record (set by prepare)
    # a trial's is its hypothesis's cost.event_costs (no switch sets it); other kinds "off".
    # cost_model stays the base spec ("book", never "book+events"): the engine and the suite wrap it.
    event_costs: str = EVENT_COSTS_OFF


def params_json(params: Mapping[str, Any]) -> str:
    return json.dumps(dict(params), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def step1_warnings(raw: Sequence[str]) -> tuple[str, ...]:
    return tuple(f"{STEP1_PREFIX}{w}" for w in raw)


def lookahead_signal(raw: Sequence[str]) -> str:
    return SIGNAL_NONE if any(NO_SIGNAL_TEXT in w for w in raw) else SIGNAL_SEEN


def checked_strategy(
    path: Path, deps: RunnerDeps, params: Mapping[str, Any] | None, seed: int
) -> tuple[Any, dict, tuple[str, ...]]:
    """Step 1: lint (nothing executed), load, resolve params against the SPEC grid, truncation check on
    synthetic bars. Returns the strategy, the resolved params and the check's raw warnings."""
    violations = deps.lint_strategy(path)
    if violations:
        raise RunRefused(f"{path.name}: {len(violations)} lint violation(s); first: {violations[0]}")
    strategy = deps.load_strategy(path)
    try:  # the engine's own resolver: an unknown name or a value outside the SPEC grid (C6)
        resolved = resolve_params(strategy.spec, params)
    except StrategyError as exc:
        raise RunRefused(str(exc)) from exc
    try:
        warnings = deps.lookahead_check(strategy, resolved, seed)
    except Exception as exc:
        raise RunRefused(f"look-ahead check failed: {type(exc).__name__}: {exc}") from exc
    return strategy, resolved, tuple(warnings or ())


def _overlaps(a: Path, b: Path) -> bool:
    return a.is_relative_to(b) or b.is_relative_to(a)


def refuse_throwaway_ledger(kind: str, deps: RunnerDeps, config: RegistryConfig) -> None:
    """A real-data run on the real store roots logs to this repository's ledger only.
    Paths are resolved; a present but broken paths file raises PathsConfigError (fail closed)."""
    if kind == SYNTHETIC:  # source="synthetic": the engine opens no store
        return
    real = [location.resolve() for location in real_store_locations(PATHS_TOML)]
    roots = [Path(root).resolve() for root in deps.store_roots().values()]
    if not any(_overlaps(root, location) for root in roots for location in real):
        return
    if config.registry_path.resolve() != REPO_LEDGER.resolve():
        raise RunRefused(
            f"a {kind} run reads the real store roots, so it logs to this repository's {LEDGER_REL} only; "
            f"the registry config names another ledger ({config.registry_path.name})"
        )


def _entry(config: RegistryConfig, check_id: str | None) -> EngineCheckEntry:
    checks = load_engine_checks(config.engine_checks_path)
    if checks is None or check_id is None or check_id not in checks.entries:
        raise EngineCheckNotDeclared(f"engine check {check_id!r} is not on config/engine_checks.toml")
    return checks.entries[check_id]


def _plan_engine_check(
    spec_defaults: dict[str, Any],
    resolved: dict[str, Any],
    args: dict[str, Any],
    deps: RunnerDeps,
    config: RegistryConfig,
) -> Plan:
    if resolved != spec_defaults:
        raise RunRefused("an engine check runs at the SPEC default params only (A2)")
    if args["hypothesis_path"] is not None or args["window"] is not None:
        raise RunRefused("an engine check takes no hypothesis and no window index")
    entry = _entry(config, args["check_id"])
    first, last = date.fromisoformat(entry.first), date.fromisoformat(entry.last)
    given = (args["root"], args["first"], args["last"])
    if any(g is not None and g != e for g, e in zip(given, (entry.root, first, last), strict=True)):
        raise RunRefused("an engine check takes root and window from its allowlist entry")
    return Plan(
        root=entry.root,
        first=first,
        last=last,
        params=resolved,
        fees=entry.fees,
        cost_model=entry.cost_model,
        data_files=deps.data_files(entry.root, first, last),
        check_id=entry.id,
        suite=args["suite"],
    )


def _plan_synthetic(strategy: Any, resolved: dict[str, Any], args: dict[str, Any]) -> Plan:
    if any(args[k] is not None for k in ("first", "last", "check_id", "hypothesis_path", "window")):
        raise RunRefused("a synthetic run takes no window, no check id and no hypothesis")
    return Plan(
        root=args["root"] or strategy.spec.root,
        first=None,
        last=None,
        params=resolved,
        fees=args["fees"] or "topstep",
        cost_model=args["cost_model"] or "book",
        data_files=(),
        check_id=None,
        suite=args["suite"],
    )


def _request(kind: str, strategy: Any, path: Path, plan: Plan, deps: RunnerDeps) -> RunRequest:
    model_id, model_sha = deps.cost_model_ids(plan.cost_model, plan.event_costs)
    return RunRequest(
        kind=kind,
        strategy_path=path,
        strategy_sha256=strategy.sha256,
        hypothesis_id=plan.hypothesis_id or strategy.spec.hypothesis_id,
        root=plan.root,
        session=strategy.spec.session,
        params=plan.params,
        seed=plan.seed,
        engine_version=deps.engine_version(),
        instruments_sha256=deps.instruments_sha256(),
        cost_model=plan.cost_model,
        cost_model_id=model_id,
        cost_config_sha256=model_sha,
        fees=plan.fees,
        fees_sha256=deps.fees_sha256(plan.fees),
        closed_windows_sha256=deps.closed_windows_sha256(),
        lint_passed=True,
        lookahead_passed=True,
        first=plan.first.isoformat() if plan.first else None,
        last=plan.last.isoformat() if plan.last else None,
        data_files=plan.data_files,
        hypothesis_path=plan.hypothesis_path,
        check_id=plan.check_id,
        warnings=step1_warnings(plan.lookahead_warnings),
        calendars_sha256=deps.calendars_sha256(),
        lookahead_signal=lookahead_signal(plan.lookahead_warnings),
        event_windows=plan.event_costs,
    )


def _execute(kind: str, path: Path, plan: Plan, receipt: Receipt, deps: RunnerDeps) -> Any:
    common: dict[str, Any] = {
        "root": plan.root,
        "params": plan.params,
        "fees": plan.fees,
        "cost_model": plan.cost_model,
        "seed": plan.seed,
        "n_trials": deflation_n(receipt.n_after),
        "event_costs": plan.event_costs,
    }
    if plan.suite is not None:
        common["suite"] = plan.suite
    if kind == SYNTHETIC:
        return deps.run_strategy(path, source="synthetic", **common)
    permit = deps.make_permit(
        run_id=receipt.run_id,
        kind=kind,
        record_sha256=receipt.record_sha256,
        strategy_sha256=receipt.record["strategy_sha256"],
        root=plan.root,
        first=plan.first,
        last=plan.last,
        params_json=params_json(plan.params),
        fees=plan.fees,
        cost_model=plan.cost_model,
        event_costs=plan.event_costs,
    )
    result = deps.run_strategy(path, source="real", first=plan.first, last=plan.last, permit=permit, **common)
    read = deps.provenance_files(result)
    if read != plan.data_files:
        raise ProvenanceMismatch(
            f"the engine read {len(read)} file(s) that differ from the {len(plan.data_files)} "
            "named before the run"
        )
    return result


def _extras(result: Any) -> Mapping[str, Any]:
    extras = getattr(result, "extras", None)
    return extras if isinstance(extras, Mapping) else {}


def artifact_names(result: Any) -> list[str]:
    """The files a run writes: the three frames, extras frames, summary.json, and suite.json with a suite."""
    names = [f"{name}.parquet" for name in FRAME_ARTIFACTS]
    names += [f"{name}.parquet" for name in sorted(_extras(result))]
    names.append(SUMMARY_ARTIFACT)
    if result.summary.get(SUITE_KEY) is not None:
        names.append(SUITE_ARTIFACT)
    return names


def planned_artifacts(
    runs_dir: Path, run_id: str, repo_root: Path, names: Sequence[str] | None = None
) -> list[str]:
    """Repo-relative names inside repo_root, "external:<runs_dir name>/<run_id>/<file>" outside it."""
    if names is None:
        names = [f"{name}.parquet" for name in FRAME_ARTIFACTS] + [SUMMARY_ARTIFACT]
    out = (runs_dir / run_id).resolve()
    try:
        base = out.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        base = f"{EXTERNAL_PREFIX}{runs_dir.name}/{run_id}"
    return [f"{base}/{name}" for name in names]


def _parquet_bytes(frame: Any) -> bytes:
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    return buffer.getvalue()


def _json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=2) + "\n").encode("utf-8")


def artifact_files(result: Any) -> dict[str, bytes]:
    """Every artifact serialized in memory, in artifact_names order: the result record
    names each with the sha256 of these bytes, and exactly these bytes are written after it."""
    files = {f"{name}.parquet": _parquet_bytes(getattr(result, name)) for name in FRAME_ARTIFACTS}
    files.update(
        {f"{name}.parquet": _parquet_bytes(frame) for name, frame in sorted(_extras(result).items())}
    )
    files[SUMMARY_ARTIFACT] = _json_bytes(result.summary)
    suite = result.summary.get(SUITE_KEY)
    if suite is not None:
        files[SUITE_ARTIFACT] = _json_bytes(suite)
    if list(files) != artifact_names(result):
        raise RuntimeError("artifact files and artifact names disagree")  # an internal invariant
    return files


def _write_artifacts(files: Mapping[str, bytes], runs_dir: Path, run_id: str) -> None:
    out = runs_dir / run_id
    out.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (out / name).write_bytes(data)


def _with_summary(result: Any, summary: dict[str, Any]) -> Any:
    if is_dataclass(result) and not isinstance(result, type):
        return replace(result, summary=summary)
    return type(result)(**{**vars(result), "summary": summary})


def with_warnings(result: Any, extra: tuple[str, ...]) -> Any:
    """A copy of the result whose engine look-ahead warnings carry "in-run: " and whose summary warnings end
    with `extra` (the step-1 warnings, already prefixed); the engine's result is not mutated."""
    engine = list(result.summary.get(SUMMARY_WARNINGS, ()))
    marked = [f"{INRUN_PREFIX}{w}" if w.startswith(LOOKAHEAD_TEXT) else w for w in engine]
    if not extra and marked == engine:
        return result
    return _with_summary(result, {**result.summary, SUMMARY_WARNINGS: [*marked, *extra]})


def _with_grid_dsr(result: Any, receipt: Receipt, deps: RunnerDeps, config: RegistryConfig) -> Any:
    """A trial: the DSR over the grid of the hypothesis's "ok" configurations, this one included (A14 G8)."""
    own = result.summary.get("sharpe_daily")
    if not isinstance(own, int | float) or isinstance(own, bool):
        return result
    mine = receipt.record["configurations"][0]
    sharpes = read_state(config).ok_daily_sharpes(receipt.record["hypothesis_id"])
    grid = [v for c, v in sorted(sharpes.items()) if c != mine] + [float(own)]
    if len(grid) < 2:
        return result
    return _with_summary(result, {**result.summary, **deps.grid_dsr(result, receipt.n_after, grid)})


def _finish_ok(
    receipt: Receipt, result: Any, files: Mapping[str, bytes], runs_dir: Path, config: RegistryConfig
) -> LedgerLine:
    names = list(files)
    planned = planned_artifacts(runs_dir, receipt.run_id, config.repo_root, names)
    shas = {path: bytes_sha256(files[name]) for path, name in zip(planned, names, strict=True)}
    try:
        return finish(
            receipt,
            STATUS_OK,
            result.summary,
            planned,
            config=config,
            artifact_sha256=shas,
        )
    except BadRequest:
        finish(receipt, STATUS_FAILED, None, [], config=config, error=METRICS_NOT_JSON)
        raise


def _plan(
    kind: str,
    strategy: Any,
    resolved: dict[str, Any],
    args: dict[str, Any],
    deps: RunnerDeps,
    config: RegistryConfig,
) -> Plan:
    if kind == ENGINE_CHECK:
        defaults = {name: p.default for name, p in strategy.spec.params.items()}
        return _plan_engine_check(defaults, resolved, args, deps, config)
    if kind == TRIAL:
        from pfb.runner.trial import plan_trial

        return plan_trial(strategy, resolved, args, deps, config)
    return _plan_synthetic(strategy, resolved, args)


def run_seed(kind: str, seed: int | None, args: Mapping[str, Any], config: RegistryConfig) -> int:
    """The run's seed: a trial's is its hypothesis's lookahead_seed (another seed given is SeedMismatch);
    another kind's is the seed given, DEFAULT_SEED when none is."""
    if kind != TRIAL:
        return DEFAULT_SEED if seed is None else seed
    from pfb.runner.trial import trial_hypothesis

    declared = trial_hypothesis(args, config)[0].lookahead_seed
    if seed is not None and seed != declared:
        raise SeedMismatch(
            f"a trial runs at its hypothesis's lookahead_seed {declared}; seed {seed} was given "
            "(leave the seed out)"
        )
    return declared


def require_event_coverage(plan: Plan, deps: RunnerDeps) -> None:
    """Under event_costs "on" the pinned release calendar must cover the run's window
    before anything is begun (the engine checks it again before any bar is read). RunRefused otherwise."""
    if plan.event_costs != EVENT_COSTS_ON:
        return
    from pfb.costs.model import CostModelError  # EventCoverageError and ReleaseCalendarError subclass it

    if plan.first is None or plan.last is None:
        raise RunRefused('event_costs "on" needs a window (a trial of a hypothesis that declares it)')
    try:
        deps.event_coverage(plan.first, plan.last)
    except CostModelError as exc:
        raise RunRefused(f"{type(exc).__name__}: {exc}") from exc


def prepare(
    kind: str,
    path: Path,
    args: dict[str, Any],
    params: Mapping[str, Any] | None,
    seed: int | None,
    deps: RunnerDeps | None,
) -> tuple[RunnerDeps, RegistryConfig, Plan, RunRequest]:
    """Steps 1-2 (nothing is logged): a gateway refusal here is a RunRefused. The seed is resolved first
    (run_seed), so a trial's SeedMismatch comes before step 1."""
    try:
        deps = deps or default_deps()
        config = deps.registry_config()
        used = run_seed(kind, seed, args, config)
        strategy, resolved, warnings = checked_strategy(path, deps, params, used)
        refuse_throwaway_ledger(kind, deps, config)
        plan = replace(
            _plan(kind, strategy, resolved, args, deps, config), lookahead_warnings=warnings, seed=used
        )
        require_event_coverage(plan, deps)
        return deps, config, plan, _request(kind, strategy, path, plan, deps)
    except GatewayRefusal as exc:
        raise RunRefused(f"{type(exc).__name__}: {exc}") from exc


def _secondary(
    kind: str, plan: Plan, runs_dir: Path, run_id: str, deps: RunnerDeps
) -> tuple[Mapping[str, Any] | None, tuple[str, ...]]:
    """A trial's declared prop plans on its saved artifacts (after the result record; never an input)."""
    if kind != TRIAL or plan.secondary is None:
        return None, ()
    try:
        return deps.prop_secondary(runs_dir / run_id, plan.root, plan.secondary), ()
    except Exception as exc:  # the run is recorded; a failed secondary output is reported, not fatal
        return None, (f"{SECONDARY_FAILED}: {type(exc).__name__}: {exc}",)


def execute_planned(
    kind: str,
    path: Path,
    prepared: tuple[RunnerDeps, RegistryConfig, Plan, RunRequest],
    seed: int | None,
    runs_dir: Path,
) -> RunOutcome:
    """Steps 3-7 for a prepared run at the Plan's seed; a seed given here must equal it (None: the Plan's).
    A trial checkpoints the ledger as TRIALS.md C6 orders."""
    deps, config, plan, request = prepared
    if (seed is not None and seed != plan.seed) or request.seed != plan.seed:
        raise SeedMismatch(f"the run was prepared at seed {plan.seed}, not {seed}")
    is_trial = kind == TRIAL
    if is_trial:
        checkpoint(config)
    receipt = begin(request, config=config)
    try:
        if is_trial:
            checkpoint(config)
        try:
            result = _execute(kind, path, plan, receipt, deps)
            result = with_warnings(result, step1_warnings(plan.lookahead_warnings))
            if is_trial:
                result = _with_grid_dsr(result, receipt, deps, config)
            files = artifact_files(result)  # serialized before the result record, which names their sha256
        except Exception as exc:
            finish(receipt, STATUS_FAILED, None, [], config=config, error=type(exc).__name__)
            raise
        result_line = _finish_ok(receipt, result, files, runs_dir, config)
        _write_artifacts(files, runs_dir, receipt.run_id)
        secondary, warnings = _secondary(kind, plan, runs_dir, receipt.run_id, deps)
    finally:
        if is_trial:
            checkpoint(config)
    return RunOutcome(
        result=result,
        begin_record=receipt.record,
        result_record=result_line.record,
        begin_sha256=receipt.record_sha256,
        result_sha256=result_line.sha256,
        secondary=secondary,
        warnings=warnings,
    )


def run_args(
    *,
    check_id: str | None = None,
    root: str | None = None,
    first: date | None = None,
    last: date | None = None,
    fees: str | None = None,
    cost_model: str | None = None,
    hypothesis_path: Path | None = None,
    window: int | None = None,
    suite: Any = None,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "root": root,
        "first": first,
        "last": last,
        "fees": fees,
        "cost_model": cost_model,
        "hypothesis_path": None if hypothesis_path is None else Path(hypothesis_path),
        "window": window,
        "suite": suite,
        "allow_neighbourhood": False,
    }


def run(
    strategy_path: Path | str,
    *,
    kind: str,
    check_id: str | None = None,
    root: str | None = None,
    first: date | None = None,
    last: date | None = None,
    params: Mapping[str, Any] | None = None,
    fees: str | None = None,
    cost_model: str | None = None,
    seed: int | None = None,
    deps: RunnerDeps | None = None,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    hypothesis_path: Path | None = None,
    window: int | None = None,
    suite: Any = None,
) -> RunOutcome:
    """Run one strategy and log it: a begin record before the engine, a result record before any output.

    A trial names its hypothesis file; window is the 1-based index of its [[window]] (needed only when it
    declares more than one); fees, cost_model, suite and seed, when given, must equal the hypothesis's (its
    seed is lookahead_seed). A trial whose hypothesis declares neighbourhood = true is run through
    run_with_neighbours (pfb.runner.neighbours). Another kind runs at seed, DEFAULT_SEED when None."""
    if kind not in RUNNER_KINDS:
        raise RunRefused(f"kind {kind!r} is not one of {list(RUNNER_KINDS)}")
    path = Path(strategy_path).resolve()
    args = run_args(
        check_id=check_id, root=root, first=first, last=last, fees=fees, cost_model=cost_model,
        hypothesis_path=hypothesis_path, window=window, suite=suite,
    )  # fmt: skip
    prepared = prepare(kind, path, args, params, seed, deps)
    return execute_planned(kind, path, prepared, seed, runs_dir)
