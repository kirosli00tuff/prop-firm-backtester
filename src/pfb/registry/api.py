"""Public registry API: genesis, sync, status, hypothesis registration, and a run's begin
and finish (ADR 13, A1,
A2, A10, A14).

Every append goes through chain.locked_append: the state, n_before, n_after and the budget already used are
derived INSIDE the exclusive lock, and the new record is replayed with the whole ledger before a byte is
written.
Before any append other than genesis and sync, upstream's two ledger files must hash as at the last genesis or
sync.
Anchor: when git HEAD of the registry's repository tracks the ledger (at its path
inside the repository), its bytes at HEAD must be a byte prefix of the working file before every append
and in every read (status, read_state), else LedgerAnchorBroken; genesis is refused while HEAD tracks a
ledger. A ledger outside the repository, or a repository that is not a git work tree, has no anchor.
LIMIT: lines appended after the last commit are not anchored; cutting them off is not detected until they
are committed.

(docs/TRIALS.md sections 2-4): registration of schema-2 hypotheses (with the facts the runner resolves
from the strategy's SPEC and the cost and fee files), begin records of tier 3 (calendars_sha256,
lookahead_signal, power), a trial only on a tree where nothing differs from HEAD (the ledger included, tracked
at HEAD), finish bound to the begin Receipt (FinishNotPermitted) and free of the upstream sync check, and
`checkpoint`, which commits the ledger alone.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pfb.registry.bookfile import BookHypothesis, is_book_record, member_windows
from pfb.registry.chain import (
    ZERO_HASH,
    LedgerLine,
    canonical_bytes,
    locked_append,
    read_ledger,
    sha256_hex,
)
from pfb.registry.codehash import PFB_SRC, code_sha256
from pfb.registry.errors import (
    BadRequest,
    BudgetExceeded,
    ChecksNotPassed,
    DirtyTree,
    EngineCheckNotDeclared,
    EngineCheckStrategy,
    FinishNotPermitted,
    GenesisExists,
    HypothesisNotRegistered,
    LedgerNotTracked,
    RegistryError,
    RunNotBegun,
    SeedMismatch,
    SpecMismatch,
    StrategyChanged,
    UndeclaredConfiguration,
)
from pfb.registry.genesis import genesis_body
from pfb.registry.gitstate import (
    commit_only,
    head_blob,
    head_commit,
    is_dirty,
    repo_relative,
    require_at_head,
    require_committed,
    with_trailer,
)
from pfb.registry.hypfile import Hypothesis, load_hypothesis
from pfb.registry.inputs import EngineCheckEntry, load_engine_checks
from pfb.registry.powercheck import (
    begin_power,
    book_begin_power,
    hypothesis_power,
    power_at,
    require_each_window,
    require_power,
)
from pfb.registry.records import (
    ARTIFACT_SHA_KEY,
    BOOK_KEY,
    CHECKS_PASSED,
    ENGINE_CHECK,
    EVENT_WINDOWS,
    EVENT_WINDOWS_OFF,
    EVENTS_MODEL_SUFFIX,
    GENESIS,
    HYPOTHESIS,
    PURPOSE_STORED,
    RESULT,
    RUN_KINDS,
    SCHEMA,
    SEED_KEY,
    SEED_REASON_KEY,
    SIGNAL_SEEN,
    STATUS_BEGUN,
    STATUS_FAILED,
    STATUS_OK,
    SYNC,
    SYNTHETIC,
    TRIAL,
    LedgerState,
    config_id,
    events_id_matches,
    replay,
)
from pfb.registry.sync import changed_pe_files, require_synced, sync_body

REPO_ROOT = Path(__file__).resolve().parents[3]
LEDGER_REL = "ledger/registry.jsonl"
ENGINE_CHECKS_REL = "config/engine_checks.toml"
OUTSIDE_REPO = "<outside repo>"


@dataclass(frozen=True)
class RegistryConfig:
    """Where the ledger, the optional upstream checkout (None: no upstream ledger), the repository, the
    allowlist and the code (code_sha256) are."""

    registry_path: Path
    pe_root: Path | None
    repo_root: Path = REPO_ROOT
    engine_checks_path: Path = REPO_ROOT / ENGINE_CHECKS_REL
    code_root: Path = PFB_SRC

    @classmethod
    def for_repo(cls, pe_root: Path | None, repo_root: Path = REPO_ROOT) -> RegistryConfig:
        return cls(
            registry_path=repo_root / LEDGER_REL,
            pe_root=pe_root,
            repo_root=repo_root,
            engine_checks_path=repo_root / ENGINE_CHECKS_REL,
        )


@dataclass(frozen=True)
class RunRequest:
    """Everything a run's begin record holds, resolved by pfb.runner before begin (no bar is read)."""

    kind: str
    strategy_path: Path
    strategy_sha256: str
    hypothesis_id: str
    root: str
    session: str
    params: Mapping[str, Any]
    seed: int
    engine_version: str
    instruments_sha256: str
    cost_model: str
    cost_model_id: str
    cost_config_sha256: str
    fees: str
    fees_sha256: str
    closed_windows_sha256: str
    lint_passed: bool
    lookahead_passed: bool
    first: str | None = None
    last: str | None = None
    data_files: tuple[tuple[str, str], ...] = ()
    hypothesis_path: Path | None = None
    check_id: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    calendars_sha256: str = ""
    lookahead_signal: str = SIGNAL_SEEN
    # "on" only for a trial whose hypothesis declares event_costs "on" (cost_model_id then ends
    # in "+events"); the begin record's existing key event_windows carries it.
    event_windows: str = EVENT_WINDOWS_OFF


@dataclass(frozen=True)
class HypothesisFacts:
    """What registration needs from outside the registry, resolved by pfb.runner (no bar is read): the
    strategy's SPEC, the cost model's id and config sha256, and the sha256 of the fee file and of the
    closed-window table."""

    spec_root: str
    spec_bar_minutes: int
    spec_session: str
    spec_params: tuple[str, ...]
    cost_model_id: str
    cost_config_sha256: str
    fees_sha256: str
    closed_windows_sha256: str
    spec_grids: Mapping[str, tuple[object, ...]]  # each SPEC parameter's own grid
    spec_contracts: int | None = None  # the SPEC's contracts (a book member's must equal it)


@dataclass(frozen=True)
class Receipt:
    """What begin returns: pfb.runner builds the engine's RunPermit from it."""

    run_id: str
    kind: str
    record_sha256: str
    n_before: int
    n_after: int
    record: Mapping[str, Any]


@dataclass(frozen=True)
class RegistryStatus:
    n: int
    n_pe: int
    n_local: int
    records: int
    last_sha256: str
    pe_changed: tuple[str, ...]
    open_runs: tuple[str, ...]
    hypotheses: tuple[str, ...]

    @property
    def pe_in_sync(self) -> bool:
        return not self.pe_changed


def now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _common(kind: str, state: LedgerState | None, added: int) -> dict[str, Any]:
    n_before = state.n if state is not None else 0
    return {
        "schema": SCHEMA,
        "kind": kind,
        "time_utc": now_utc(),
        "n_before": n_before,
        "n_after": n_before + added,
    }


def _validate(records: tuple[Mapping[str, Any], ...]) -> None:
    replay(records)


def _state(lines: tuple[LedgerLine, ...]) -> LedgerState:
    return replay([line.record for line in lines])


def _ledger_rel_in_repo(config: RegistryConfig) -> str | None:
    try:
        return config.registry_path.resolve().relative_to(config.repo_root.resolve()).as_posix()
    except ValueError:
        return None


def ledger_anchor(config: RegistryConfig) -> bytes | None:
    """The ledger's bytes at git HEAD; None when the ledger lies outside the repository or HEAD lacks it."""
    rel = _ledger_rel_in_repo(config)
    return None if rel is None else head_blob(config.repo_root, rel)


def _read_anchored(config: RegistryConfig) -> tuple[LedgerLine, ...]:
    return read_ledger(config.registry_path, ledger_anchor(config))


def read_state(config: RegistryConfig) -> LedgerState:
    """Read, anchor-check, chain-verify and replay the whole ledger."""
    return _state(_read_anchored(config))


def current_n(config: RegistryConfig) -> int:
    return read_state(config).n


def create_genesis(config: RegistryConfig) -> LedgerLine:
    """Write the genesis record into a new or empty ledger: the upstream lines unchanged and N = the
    upstream's N, or N = 0 without an upstream."""

    def build(_: tuple[LedgerLine, ...]) -> dict[str, Any]:
        body = genesis_body(config.pe_root)
        return {**_common(GENESIS, None, body["n_pe"]), **body}

    if ledger_anchor(config) is not None:
        raise GenesisExists(
            f"HEAD tracks {_ledger_rel_in_repo(config)}: genesis is refused (restore the committed ledger)"
        )
    config.registry_path.parent.mkdir(parents=True, exist_ok=True)
    return locked_append(config.registry_path, build, _validate, genesis=True)


def sync(config: RegistryConfig) -> LedgerLine | None:
    """Import upstream's new ledger lines; None (nothing appended) when upstream's ledger is unchanged."""

    def build(lines: tuple[LedgerLine, ...]) -> dict[str, Any]:
        state = _state(lines)
        body = sync_body(state, config.pe_root)
        if body is None:
            raise _NothingToSync
        return {**_common(SYNC, state, body["n_pe"] - body["n_pe_before"]), **body}

    try:
        return locked_append(config.registry_path, build, _validate, anchor=ledger_anchor(config))
    except _NothingToSync:
        return None


class _NothingToSync(Exception):
    """Internal: Upstream's ledger is unchanged, so sync appends nothing."""


def status(config: RegistryConfig) -> RegistryStatus:
    lines = _read_anchored(config)
    state = _state(lines)
    return RegistryStatus(
        n=state.n,
        n_pe=state.n_pe,
        n_local=state.n_local,
        records=state.records,
        last_sha256=lines[-1].sha256,
        pe_changed=changed_pe_files(state, config.pe_root),
        open_runs=state.open_runs,
        hypotheses=tuple(sorted(state.hypotheses)),
    )


def _check_against_spec(hyp: Hypothesis, facts: HypothesisFacts, where: str) -> None:
    if facts.spec_root not in hyp.roots:
        raise SpecMismatch(f"{where}: roots {list(hyp.roots)} must include the SPEC root {facts.spec_root!r}")
    if hyp.bar_minutes != facts.spec_bar_minutes or hyp.session != facts.spec_session:
        raise SpecMismatch(f"{where}: bar_minutes and session must equal the SPEC's")
    if set(hyp.grid) != set(facts.spec_params):
        raise SpecMismatch(
            f"{where}: the grid names {sorted(hyp.grid)} differ from the SPEC parameters "
            f"{sorted(facts.spec_params)}"
        )
    for name, values in hyp.grid.items():
        spec_grid = tuple(facts.spec_grids.get(name, ()))
        allowed = {canonical_bytes(v) for v in spec_grid}
        outside = [v for v in values if canonical_bytes(v) not in allowed]
        if outside:
            raise SpecMismatch(
                f"{where}: grid.{name} holds {outside}, outside the SPEC parameter's own grid "
                f"{list(spec_grid)} (value and type)"
            )
    most = len(hyp.roots) * len(hyp.windows) * hyp.grid_points
    if hyp.budget > most:
        raise BudgetExceeded(f"{where}: budget {hyp.budget} exceeds roots x windows x grid points = {most}")


def _check_allowlist(hyp: Hypothesis, config: RegistryConfig, where: str) -> None:
    checks = load_engine_checks(config.engine_checks_path)
    if checks is not None and any(e.strategy_sha256 == hyp.strategy_sha256 for e in checks.entries.values()):
        raise EngineCheckStrategy(f"{where}: the strategy is on {ENGINE_CHECKS_REL} (rule C5)")


def require_events_id(event_costs: str, model_id: str, where: str) -> None:
    """The cost model id the runner resolved carries the declared event costs ("on": it ends
    in "+events"; "off": it does not). Replay checks the record the same way."""
    if not events_id_matches(event_costs, model_id):
        raise BadRequest(
            f"{where}: event_costs {event_costs!r} with cost model id {model_id!r}: "
            f'"on" needs an id ending in "{EVENTS_MODEL_SUFFIX}" and "off" one that does not'
        )


def hypothesis_body(hyp: Hypothesis, hyp_rel: str, facts: HypothesisFacts) -> dict[str, Any]:
    """The hypothesis record's own fields (TRIALS.md section 3) except power, which needs N."""
    return {
        "hypothesis_id": hyp.id,
        "hypothesis_sha256": hyp.sha256,
        "hypothesis_path": hyp_rel,
        "purpose": PURPOSE_STORED[hyp.purpose],
        "statement": hyp.statement,
        "strategy_path": hyp.strategy,
        "strategy_sha256": hyp.strategy_sha256,
        "roots": list(hyp.roots),
        "bar_minutes": hyp.bar_minutes,
        "session": hyp.session,
        "windows": [list(w) for w in hyp.windows],
        "grid": {k: list(v) for k, v in hyp.grid.items()},
        "budget": hyp.budget,
        "no_signal_reason": hyp.no_signal_reason,
        "cost": {
            "model": hyp.cost.model,
            "fees": hyp.cost.fees,
            "model_id": facts.cost_model_id,
            "config_sha256": facts.cost_config_sha256,
            "fees_sha256": facts.fees_sha256,
            "event_costs": hyp.cost.event_costs,
            "event_stance": hyp.cost.event_stance,
        },
        "suite": hyp.suite.to_json(),
        "secondary": hyp.secondary.to_json(),
        "closed_windows_sha256": facts.closed_windows_sha256,
        SEED_KEY: hyp.lookahead_seed,  # hypothesis record tier 3
        SEED_REASON_KEY: hyp.lookahead_seed_reason,
    }


def register_hypothesis(path: Path, config: RegistryConfig, facts: HypothesisFacts) -> LedgerLine:
    """Append a hypothesis record (N unchanged) after every registry-side refusal of TRIALS.md section 2.

    The window and data-file refusals need the gateway and are made by pfb.runner.register before this call;
    the power check runs inside the ledger lock at N + budget for each declared window on its own (the record
    stores the shortest window's figures); a grid value outside the SPEC parameter's own grid is refused;
    replay checks rule C5's ledger half and binds sd_source to its run. A book file is refused here
    (register_book registers it)."""
    build = _hypothesis_builder(path, config, facts)
    return locked_append(config.registry_path, build, _validate, anchor=ledger_anchor(config))


def _hypothesis_builder(
    path: Path, config: RegistryConfig, facts: HypothesisFacts
) -> Callable[[tuple[LedgerLine, ...]], dict[str, Any]]:
    """register_hypothesis's checks outside the lock; returns the record builder run inside it."""
    hyp_rel = repo_relative(config.repo_root, path)
    require_committed(config.repo_root, hyp_rel)
    hyp = load_hypothesis(config.repo_root / hyp_rel)
    if isinstance(hyp, BookHypothesis):
        raise BadRequest(f"{hyp_rel} is a book (schema 3): it is registered with register_book")
    where = f"hypothesis {hyp.id!r}"
    strategy_rel = repo_relative(config.repo_root, Path(hyp.strategy))
    if strategy_rel != hyp.strategy:
        raise BadRequest(f"{where}: strategy must be a plain repo-relative path, not {hyp.strategy!r}")
    require_committed(config.repo_root, strategy_rel)
    strategy_file = config.repo_root / strategy_rel
    if not strategy_file.is_file() or sha256_hex(strategy_file.read_bytes()) != hyp.strategy_sha256:
        raise StrategyChanged(f"{hyp.strategy} does not hash to the hypothesis's strategy_sha256")
    _check_against_spec(hyp, facts, where)
    _check_allowlist(hyp, config, where)
    require_events_id(hyp.cost.event_costs, facts.cost_model_id, where)
    body = hypothesis_body(hyp, hyp_rel, facts)
    inputs = hyp.power.to_json()

    def build(lines: tuple[LedgerLine, ...]) -> dict[str, Any]:
        state = _state(lines)
        require_synced(state, config.pe_root)
        n_used = state.n + hyp.budget
        require_each_window(inputs, body["windows"], n_used, where)
        return {
            **_common(HYPOTHESIS, state, 0),
            **body,
            "power": hypothesis_power(inputs, body["windows"], n_used),
            BOOK_KEY: None,  # hypothesis record tier 2: a schema-2 hypothesis is no book
        }

    return build


def dry_append(config: RegistryConfig, build: Callable[[tuple[LedgerLine, ...]], dict[str, Any]]) -> None:
    """Every check an append makes (anchor, chain, the builder's own checks, replay of the whole ledger with
    the new record) WITHOUT writing: `pfb hypothesis register --dry-run`. No lock is taken; nothing
    changes."""
    lines = _read_anchored(config)
    built = dict(build(lines))
    built["prev_sha256"] = lines[-1].sha256 if lines else ZERO_HASH
    _validate(tuple(line.record for line in lines) + (json.loads(canonical_bytes(built)),))


def hypothesis_refusals(path: Path, config: RegistryConfig, facts: HypothesisFacts) -> list[RegistryError]:
    """The registry-side refusal of a schema-2 file (at most one: its checks stop at the first), nothing
    appended."""
    try:
        dry_append(config, _hypothesis_builder(path, config, facts))
    except RegistryError as exc:
        return [exc]
    return []


def hypothesis_record(config: RegistryConfig, hypothesis_id: str) -> Mapping[str, Any]:
    """The registered hypothesis record (HypothesisNotRegistered when the id has none)."""
    record = read_state(config).hypotheses.get(hypothesis_id)
    if record is None:
        raise HypothesisNotRegistered(f"hypothesis {hypothesis_id!r} is not registered")
    return record


def _display_path(config: RegistryConfig, path: Path) -> str:
    """Repo-relative POSIX path; a file outside the repository is named by file name (no home path)."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(config.repo_root.resolve()).as_posix()
    except ValueError:
        return f"{OUTSIDE_REPO}/{resolved.name}"


def _require_ledger_at_head(config: RegistryConfig) -> None:
    """A trial: the ledger lies in the repository and is tracked at HEAD (TRIALS.md section 4)."""
    if ledger_anchor(config) is None:
        raise LedgerNotTracked(
            "a trial needs the ledger tracked at git HEAD inside the repository "
            "(run `pfb registry checkpoint`)"
        )


def _trial_fields(request: RunRequest, config: RegistryConfig) -> dict[str, Any]:
    if request.hypothesis_path is None:
        raise BadRequest("a trial names its hypothesis file")
    hyp_rel = repo_relative(config.repo_root, request.hypothesis_path)
    strategy_rel = repo_relative(config.repo_root, request.strategy_path)
    for rel in (strategy_rel, hyp_rel):
        require_committed(config.repo_root, rel)
    _require_ledger_at_head(config)
    if is_dirty(config.repo_root, "") is not False:
        raise DirtyTree("a trial needs a clean git tree: nothing may differ from HEAD, the ledger included")
    hyp = load_hypothesis(config.repo_root / hyp_rel)
    if hyp.id != request.hypothesis_id:
        raise BadRequest(
            f"hypothesis file id {hyp.id!r} differs from the request's {request.hypothesis_id!r}"
        )
    if request.seed != hyp.lookahead_seed:
        raise SeedMismatch(
            f"a trial runs at its hypothesis's lookahead_seed {hyp.lookahead_seed}; the request's seed is "
            f"{request.seed}"
        )
    _require_event_costs(hyp.cost.event_costs, request)
    return {"hypothesis_sha256": hyp.sha256, "check": None, "declared_in": None}


def _require_event_costs(declared: str, request: RunRequest) -> None:
    """A trial runs at its hypothesis's event_costs. "on" needs
    event_windows "on" and a cost_model_id ending in "+events"; "off" needs neither. Replay checks the
    same."""
    if request.event_windows != declared:
        raise UndeclaredConfiguration(
            f"a trial of {request.hypothesis_id!r} runs at event_costs {declared!r}; the request says "
            f"event_windows {request.event_windows!r}"
        )
    if not events_id_matches(declared, request.cost_model_id):
        raise UndeclaredConfiguration(
            f'event_costs {declared!r} with cost_model_id {request.cost_model_id!r}: "on" needs an id ending '
            f'in "{EVENTS_MODEL_SUFFIX}" and "off" one that does not'
        )


def _engine_check_fields(request: RunRequest, config: RegistryConfig) -> dict[str, Any]:
    checks = load_engine_checks(config.engine_checks_path)
    if checks is None:
        raise EngineCheckNotDeclared(f"{ENGINE_CHECKS_REL} is absent: every engine_check is refused")
    entry: EngineCheckEntry | None = checks.entries.get(request.check_id or "")
    if entry is None:
        raise EngineCheckNotDeclared(f"check id {request.check_id!r} is not on {ENGINE_CHECKS_REL}")
    require_committed(config.repo_root, repo_relative(config.repo_root, config.engine_checks_path))
    require_committed(config.repo_root, entry.strategy)
    require_at_head(config.repo_root, entry.declared_in)
    declared = (entry.session, entry.seed, canonical_bytes(dict(entry.params)))
    if (request.session, request.seed, canonical_bytes(dict(request.params))) != declared:
        raise EngineCheckNotDeclared(
            f"check {entry.id!r} runs at session {entry.session!r}, seed {entry.seed} and the params of its "
            "entry only (R4)"
        )
    return {"hypothesis_sha256": None, "check": entry.to_json(), "declared_in": entry.declared_in}


def _ledger_rel(config: RegistryConfig) -> str:
    try:
        return config.registry_path.resolve().relative_to(config.repo_root.resolve()).as_posix()
    except ValueError:
        return LEDGER_REL


def _run_fields(request: RunRequest, config: RegistryConfig) -> dict[str, Any]:
    """Every begin field except the common ones, checked before the lock is taken."""
    if request.kind not in RUN_KINDS:
        raise BadRequest(f"kind {request.kind!r} is not one of {sorted(RUN_KINDS)}")
    if request.kind == SYNTHETIC and request.data_files:
        raise BadRequest("a synthetic run reads no data file")
    if not (request.lint_passed and request.lookahead_passed):
        raise ChecksNotPassed("lint and look-ahead must both pass before a run is begun")
    if not request.calendars_sha256:
        raise BadRequest("a begin record names calendars_sha256 (config/calendars/*.json)")
    if request.event_windows not in EVENT_WINDOWS or (
        request.kind != TRIAL and request.event_windows != EVENT_WINDOWS_OFF
    ):
        raise BadRequest(f'event_windows {request.event_windows!r}: "on" is a trial\'s only, else "off"')
    if (
        not request.strategy_path.is_file()
        or sha256_hex(request.strategy_path.read_bytes()) != request.strategy_sha256
    ):
        raise StrategyChanged(f"{request.strategy_path.name} does not hash to the request's strategy_sha256")
    strategy_rel = _display_path(config, request.strategy_path)
    if request.kind == TRIAL:
        kind_fields = _trial_fields(request, config)
    elif request.kind == ENGINE_CHECK:
        kind_fields = _engine_check_fields(request, config)
    else:
        kind_fields = {"hypothesis_sha256": None, "check": None, "declared_in": None}
    checks = load_engine_checks(config.engine_checks_path)
    fields = {
        "run_id": f"{request.kind}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{secrets.token_hex(4)}",
        "status": STATUS_BEGUN,
        "hypothesis_id": request.hypothesis_id,
        "strategy_path": strategy_rel,
        "strategy_sha256": request.strategy_sha256,
        "git_commit": head_commit(config.repo_root),
        "git_dirty": is_dirty(config.repo_root, _ledger_rel(config)),
        "data_files": [{"path": p, "sha256": s} for p, s in request.data_files],
        "instruments_sha256": request.instruments_sha256,
        "cost_model": request.cost_model,
        "cost_model_id": request.cost_model_id,
        "cost_config_sha256": request.cost_config_sha256,
        "fees": request.fees,
        "fees_sha256": request.fees_sha256,
        "closed_windows_sha256": request.closed_windows_sha256,
        "engine_checks_sha256": checks.sha256 if checks is not None else None,
        "root": request.root,
        "first": request.first,
        "last": request.last,
        "session": request.session,
        "params": dict(request.params),
        "seed": request.seed,
        "engine_version": request.engine_version,
        "code_sha256": code_sha256(config.code_root),
        "metrics": None,
        "artifacts": [],
        "event_windows": request.event_windows,
        "warnings": list(request.warnings),
        "static_checks": dict(CHECKS_PASSED),
        "calendars_sha256": request.calendars_sha256,
        "lookahead_signal": request.lookahead_signal,
        "power": None,
        **kind_fields,
    }
    fields["configurations"] = [config_id(fields)]
    return fields


def _trial_power(state: LedgerState, request: RunRequest) -> dict[str, Any]:
    """The trial's power object at n_used = n_before + budget - configurations already counted, over the
    trial's own window."""
    hypothesis_id = request.hypothesis_id
    hyp = state.hypotheses.get(hypothesis_id)
    if hyp is None:
        raise HypothesisNotRegistered(f"hypothesis {hypothesis_id!r} is not registered")
    first, last = request.first, request.last
    if first is None or last is None or [first, last] not in hyp["windows"]:
        raise UndeclaredConfiguration(f"window {first}..{last} is not declared")
    window = [first, last]
    n_used = state.n + hyp["budget"] - len(state.configurations.get(hypothesis_id, frozenset()))
    if is_book_record(hyp):  # the BOOK's check over the union of its member windows
        return book_begin_power(
            hyp["power"], member_windows(hyp), n_used, f"book member of {hypothesis_id!r}"
        )
    require_power(power_at(hyp["power"], [window], n_used), f"trial of {hypothesis_id!r}")
    return begin_power(hyp["power"], [window], n_used)


def _require_lines_at_head(lines: tuple[LedgerLine, ...], config: RegistryConfig) -> None:
    """Inside the lock: the ledger read equals its bytes at HEAD (no line appended since the checkpoint)."""
    if b"".join(line.raw + b"\n" for line in lines) != ledger_anchor(config):
        raise DirtyTree("the ledger has lines not committed at HEAD: run `pfb registry checkpoint` first")


def begin(request: RunRequest, *, config: RegistryConfig) -> Receipt:
    """Append and fsync the run's begin record (a trial's N rises HERE, A1 step 4); return a receipt."""
    fields = _run_fields(request, config)

    def build(lines: tuple[LedgerLine, ...]) -> dict[str, Any]:
        state = _state(lines)
        require_synced(state, config.pe_root)
        record = dict(fields)
        added = 0
        if request.kind == TRIAL:
            _require_lines_at_head(lines, config)
            record["power"] = _trial_power(state, request)
            seen = state.configurations.get(request.hypothesis_id, frozenset())
            added = sum(1 for c in fields["configurations"] if c not in seen)
        return {**_common(request.kind, state, added), **record}

    line = locked_append(config.registry_path, build, _validate, anchor=ledger_anchor(config))
    return Receipt(
        run_id=fields["run_id"],
        kind=request.kind,
        record_sha256=line.sha256,
        n_before=line.record["n_before"],
        n_after=line.record["n_after"],
        record=line.record,
    )


def plan_configurations(requests: Sequence[RunRequest], *, config: RegistryConfig) -> int:
    """C3: before any of a set of trials of one hypothesis is begun, the number of its configurations not
    yet counted; the whole set is refused (BudgetExceeded) when the budget cannot cover them. Nothing is
    appended; each begin still checks its own record."""
    hyp_ids = {r.hypothesis_id for r in requests}
    if not requests or any(r.kind != TRIAL for r in requests) or len(hyp_ids) != 1:
        raise BadRequest("a planned set is one or more trials of one hypothesis")
    ids = {_run_fields(r, config)["configurations"][0] for r in requests}
    state = read_state(config)
    hyp_id = hyp_ids.pop()
    hyp = state.hypotheses.get(hyp_id)
    if hyp is None:
        raise HypothesisNotRegistered(f"hypothesis {hyp_id!r} is not registered")
    seen = state.configurations.get(hyp_id, frozenset())
    new = ids - seen
    if len(seen) + len(new) > hyp["budget"]:
        raise BudgetExceeded(
            f"the set of {len(requests)} runs needs {len(new)} new configurations; {len(seen)} of the budget "
            f"{hyp['budget']} of {hyp_id!r} are spent: the whole set is refused and nothing is begun"
        )
    return len(new)


def finish(
    receipt: Receipt,
    status: str,
    metrics: Mapping[str, Any] | None,
    artifacts: Sequence[str],
    *,
    config: RegistryConfig,
    error: str | None = None,
    warnings: Sequence[str] = (),
    artifact_sha256: Mapping[str, str] | None = None,
) -> LedgerLine:
    """Append the run's result record (N unchanged): status "ok", or "failed" with the error class name.

    Bound to the begin Receipt: refused (FinishNotPermitted) unless receipt.record_sha256 is the sha256 of
    the run's begin line. A upstream ledger change does not block a result (it moves no N); begin stays
    blocked.
    artifact_sha256 maps every name in artifacts to the sha256 of the bytes that will
    be written under it; replay refuses a map that does not name exactly the artifacts (None means {})."""
    if status not in (STATUS_OK, STATUS_FAILED) or (status == STATUS_FAILED) != isinstance(error, str):
        raise BadRequest(f"status must be {STATUS_OK!r} (no error) or {STATUS_FAILED!r} (+ error class)")
    body = {
        "run_id": receipt.run_id,
        "status": status,
        "error": error,
        "metrics": dict(metrics) if metrics is not None else None,
        "artifacts": list(artifacts),
        "warnings": list(warnings),
        "begin_sha256": receipt.record_sha256,
        ARTIFACT_SHA_KEY: dict(artifact_sha256 or {}),
    }

    def build(lines: tuple[LedgerLine, ...]) -> dict[str, Any]:
        state = _state(lines)
        begun = state.begin_sha256.get(receipt.run_id)
        if begun is None:
            raise RunNotBegun(f"run {receipt.run_id!r} has no begin record")
        if begun != receipt.record_sha256:
            raise FinishNotPermitted(
                f"the receipt for {receipt.run_id!r} does not carry the sha256 of its begin line"
            )
        return {**_common(RESULT, state, 0), **body}

    return locked_append(config.registry_path, build, _validate, anchor=ledger_anchor(config))


@dataclass(frozen=True)
class Checkpoint:
    commit: str
    n: int
    records: int


def checkpoint(config: RegistryConfig) -> Checkpoint | None:
    """Commit ledger/registry.jsonl alone ("ledger checkpoint: N <n>, records <r>" plus the lines of
    $PFB_COMMIT_TRAILER); None when the ledger already equals its bytes at HEAD. Refused (LedgerNotTracked)
    when the ledger lies outside the repository or the repository is not a git work tree."""
    rel = _ledger_rel_in_repo(config)
    if rel is None or is_dirty(config.repo_root, "") is None:
        raise LedgerNotTracked("the ledger lies outside a git work tree: there is nothing to checkpoint into")
    state = read_state(config)  # anchor, chain and replay verified before anything is committed
    if config.registry_path.read_bytes() == ledger_anchor(config):
        return None
    message = with_trailer(f"ledger checkpoint: N {state.n}, records {state.records}")
    commit = commit_only(config.repo_root, rel, message)
    return Checkpoint(commit=commit, n=state.n, records=state.records)


KINDS = (GENESIS, SYNC, HYPOTHESIS, TRIAL, ENGINE_CHECK, SYNTHETIC, RESULT)
