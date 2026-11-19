"""Registration of a book (schema "pfb.hypothesis/3") and the gate of its pooled statistic.

register_book appends the book's hypothesis record (N unchanged) after every registry-side refusal: the
book file committed at HEAD; per member (each a check a schema-2 hypothesis gets) a plain repo-relative
strategy committed at HEAD and hashing to its strategy_sha256, params naming exactly the SPEC parameters with
one value inside each SPEC grid, contracts equal to the SPEC's (the engine runs at the SPEC's contracts, so a
member at other contracts would not be the configuration whose trades_per_year and trade_sd_cents it
declares), and rule C5 against the allowlist; inside the ledger lock upstream in sync and the BOOK's power
check
at N + K over the union of the member windows. Replay re-checks the record (pfb.registry.records). The window
and data-file refusals need the gateway and are made per member by pfb.runner.register.

book_refusals is the same set of checks with nothing appended (`pfb hypothesis register --dry-run`): every
member's refusal is listed, not only the first.

book_runs reads the ledger only: it returns the latest ok run of every member and the live N, and
refuses (BookIncomplete) while any member lacks an ok result. It appends nothing, so N is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from pfb.registry.api import (
    ENGINE_CHECKS_REL,
    HypothesisFacts,
    RegistryConfig,
    _common,  # pyright: ignore[reportPrivateUsage]
    _state,  # pyright: ignore[reportPrivateUsage]
    _validate,  # pyright: ignore[reportPrivateUsage]
    dry_append,
    ledger_anchor,
    read_state,
    require_events_id,
)
from pfb.registry.bookfile import (
    BookHypothesis,
    BookMember,
    derived_power,
    is_book_record,
    member_index,
    pool_members,
    root_family,
)
from pfb.registry.chain import LedgerLine, canonical_bytes, locked_append, sha256_hex
from pfb.registry.errors import (
    BadRequest,
    BookIncomplete,
    EngineCheckStrategy,
    HypothesisNotRegistered,
    RegistryError,
    SpecMismatch,
    StrategyChanged,
)
from pfb.registry.gitstate import repo_relative, require_committed
from pfb.registry.hypfile import load_hypothesis
from pfb.registry.inputs import load_engine_checks
from pfb.registry.powercheck import book_hypothesis_power
from pfb.registry.records import (
    BOOK_KEY,
    HYPOTHESIS,
    PURPOSE_STORED,
    SEED_KEY,
    SEED_REASON_KEY,
    STATUS_OK,
    TRIAL,
)
from pfb.registry.sync import require_synced

Builder = Callable[[tuple[LedgerLine, ...]], dict[str, Any]]


def load_book(path: Path, config: RegistryConfig) -> tuple[BookHypothesis, str]:
    """The book file (parsed) and its repo-relative path; BadRequest when the file is not a book."""
    hyp_rel = repo_relative(config.repo_root, path)
    hyp = load_hypothesis(config.repo_root / hyp_rel)
    if not isinstance(hyp, BookHypothesis):
        raise BadRequest(f"{hyp_rel} is not a book (schema 3); register it as a schema-2 hypothesis")
    return hyp, hyp_rel


def _check_member(member: BookMember, facts: HypothesisFacts, config: RegistryConfig, where: str) -> None:
    strategy_rel = repo_relative(config.repo_root, Path(member.strategy))
    if strategy_rel != member.strategy:
        raise BadRequest(f"{where}: strategy must be a plain repo-relative path, not {member.strategy!r}")
    require_committed(config.repo_root, strategy_rel)
    strategy_file = config.repo_root / strategy_rel
    if not strategy_file.is_file() or sha256_hex(strategy_file.read_bytes()) != member.strategy_sha256:
        raise StrategyChanged(f"{where}: {member.strategy} does not hash to the member's strategy_sha256")
    if set(member.params) != set(facts.spec_params):
        raise SpecMismatch(
            f"{where}: params {sorted(member.params)} must name exactly the SPEC parameters "
            f"{sorted(facts.spec_params)}"
        )
    for name, value in member.params.items():
        spec_grid = tuple(facts.spec_grids.get(name, ()))
        if canonical_bytes(value) not in {canonical_bytes(v) for v in spec_grid}:
            raise SpecMismatch(
                f"{where}: params.{name} = {value!r} lies outside the SPEC parameter's own grid "
                f"{list(spec_grid)}"
            )
    if facts.spec_contracts is None or member.contracts != facts.spec_contracts:
        raise SpecMismatch(
            f"{where}: contracts {member.contracts} must equal the SPEC's contracts {facts.spec_contracts} "
            "(the engine runs a member at its SPEC's contracts)"
        )
    checks = load_engine_checks(config.engine_checks_path)
    if checks is not None and any(
        e.strategy_sha256 == member.strategy_sha256 for e in checks.entries.values()
    ):
        raise EngineCheckStrategy(f"{where}: the strategy is on {ENGINE_CHECKS_REL} (rule C5)")


def member_refusals(
    book: BookHypothesis, facts: Sequence[HypothesisFacts], config: RegistryConfig
) -> list[RegistryError]:
    """Every member's first refusal (one per member at most), in member order."""
    if len(facts) != len(book.members):
        raise BadRequest(f"{len(facts)} member facts for the {len(book.members)} members of {book.id!r}")
    out: list[RegistryError] = []
    for i, (member, fact) in enumerate(zip(book.members, facts, strict=True)):
        try:
            _check_member(member, fact, config, f"book {book.id!r} member {i + 1}")
        except RegistryError as exc:
            out.append(exc)
    return out


def _member_record(member: BookMember, facts: HypothesisFacts) -> dict[str, Any]:
    return {
        "strategy_path": member.strategy,
        "strategy_sha256": member.strategy_sha256,
        "root": member.root,
        "contracts": member.contracts,
        "params": dict(member.params),
        "window": list(member.window),
        "trades_per_year": member.trades_per_year,
        "trade_sd_cents": member.trade_sd_cents,
        "bar_minutes": facts.spec_bar_minutes,
        "session": facts.spec_session,
        "root_family": root_family(member.root),  # clause 7a (config/instruments.toml exposure)
    }


def _cost_facts(facts: Sequence[HypothesisFacts], where: str) -> HypothesisFacts:
    """The cost model, fee file and closed-window table are the book's own: one identity for every member."""
    first = facts[0]
    keys = ("cost_model_id", "cost_config_sha256", "fees_sha256", "closed_windows_sha256")
    if any(tuple(getattr(f, k) for k in keys) != tuple(getattr(first, k) for k in keys) for f in facts):
        raise BadRequest(f"{where}: the members' cost, fee and closed-window identities differ")
    return first


def book_body(book: BookHypothesis, hyp_rel: str, facts: Sequence[HypothesisFacts]) -> dict[str, Any]:
    """The book's hypothesis record fields except the common ones and power (which needs N)."""
    where = f"book {book.id!r}"
    cost = _cost_facts(facts, where)
    require_events_id(book.cost.event_costs, cost.cost_model_id, where)  # as a schema-2 file
    derived = derived_power(pool_members(book.members), Fraction(book.member_correlation))
    return {
        "hypothesis_id": book.id,
        "hypothesis_sha256": book.sha256,
        "hypothesis_path": hyp_rel,
        "purpose": PURPOSE_STORED[book.purpose],
        "statement": book.statement,
        "strategy_path": None,
        "strategy_sha256": None,
        "roots": list(book.roots),
        "bar_minutes": None,
        "session": None,
        "windows": [list(w) for w in book.windows],
        "grid": None,
        "budget": book.budget,
        "no_signal_reason": book.no_signal_reason,
        "cost": {
            **asdict(book.cost),
            "model_id": cost.cost_model_id,
            "config_sha256": cost.cost_config_sha256,
            "fees_sha256": cost.fees_sha256,
        },
        "suite": book.suite.to_json(),
        "secondary": book.secondary.to_json(),
        "closed_windows_sha256": cost.closed_windows_sha256,
        BOOK_KEY: {
            "account": book.account,
            "cap": book.cap,
            "statistic": book.statistic,
            "members": [_member_record(m, f) for m, f in zip(book.members, facts, strict=True)],
            "derived": derived.to_json(),
            "member_correlation": book.member_correlation,
            "member_correlation_reason": book.member_correlation_reason,
        },
        SEED_KEY: book.lookahead_seed,  # hypothesis record tier 3
        SEED_REASON_KEY: book.lookahead_seed_reason,
    }


def _builder(book: BookHypothesis, body: Mapping[str, Any], config: RegistryConfig) -> Builder:
    where = f"book {book.id!r}"
    windows = [list(m.window) for m in book.members]

    def build(lines: tuple[LedgerLine, ...]) -> dict[str, Any]:
        state = _state(lines)
        require_synced(state, config.pe_root)
        n_used = state.n + book.budget  # clause 5: registration at N + K
        power = book_hypothesis_power(book.power.to_json(), windows, n_used, where)
        return {**_common(HYPOTHESIS, state, 0), **body, "power": power}

    return build


def register_book(path: Path, config: RegistryConfig, facts: Sequence[HypothesisFacts]) -> LedgerLine:
    """Append the book's hypothesis record (N unchanged); the first refusal is raised."""
    book, hyp_rel = load_book(path, config)
    require_committed(config.repo_root, hyp_rel)
    refusals = member_refusals(book, facts, config)
    if refusals:
        raise refusals[0]
    build = _builder(book, book_body(book, hyp_rel, facts), config)
    return locked_append(config.registry_path, build, _validate, anchor=ledger_anchor(config))


def book_refusals(
    path: Path, config: RegistryConfig, facts: Sequence[HypothesisFacts]
) -> list[RegistryError]:
    """register_book's checks with nothing appended: every refusal found (the file's, each member's, then
    the in-lock ones on a replay of the whole ledger with the would-be record)."""
    book, hyp_rel = load_book(path, config)
    out: list[RegistryError] = []
    try:
        require_committed(config.repo_root, hyp_rel)
    except RegistryError as exc:
        out.append(exc)
    out.extend(member_refusals(book, facts, config))
    try:
        dry_append(config, _builder(book, book_body(book, hyp_rel, facts), config))
    except RegistryError as exc:
        out.append(exc)
    return out


@dataclass(frozen=True)
class BookRuns:
    """The K counted runs a book's pooled statistic is a fixed function of, and the live N it is read at."""

    hypothesis_id: str
    account: str
    statistic: str
    n: int
    run_ids: tuple[str, ...]  # the latest ok run of each member, in member order


def book_runs(config: RegistryConfig, hypothesis_id: str) -> BookRuns:
    """Clause 3: read-only; BookIncomplete unless every one of the K members has an ok result."""
    state = read_state(config)
    hyp = state.hypotheses.get(hypothesis_id)
    if hyp is None:
        raise HypothesisNotRegistered(f"hypothesis {hypothesis_id!r} is not registered")
    if not is_book_record(hyp):
        raise BadRequest(f"hypothesis {hypothesis_id!r} is not a book")
    latest: dict[int, str] = {}
    for run_id, begin in state.begun.items():  # ledger order: a later ok run of a member replaces it
        result = state.results.get(run_id)
        if begin["kind"] != TRIAL or begin["hypothesis_id"] != hypothesis_id or result is None:
            continue
        if result["status"] == STATUS_OK:
            latest[member_index(begin, hyp, run_id)] = run_id
    members = hyp["book"]["members"]
    missing = [i + 1 for i in range(len(members)) if i not in latest]
    if missing:
        raise BookIncomplete(
            f"book {hypothesis_id!r}: members {missing} of {len(members)} have no ok result; the pooled "
            "statistic is refused until all K have (no partial book)"
        )
    book = hyp["book"]
    return BookRuns(
        hypothesis_id=hypothesis_id,
        account=book["account"],
        statistic=book["statistic"],
        n=state.n,
        run_ids=tuple(latest[i] for i in range(len(members))),
    )
