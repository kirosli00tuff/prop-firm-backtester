"""`pfb hypothesis register <file>` (docs/TRIALS.md section 2): resolve what the registry cannot see, then
register.

Here, with the runner's deps (no bar is read): the strategy is linted and loaded for its SPEC; the cost model
and fee file are resolved to their ids and hashes; every (root, declared window) pair is asked of the
gateway's files_for, so a window touching a closed window (WindowTable.request_refusal) is refused as
ClosedWindowTouched and one for which the catalog names no file as NoDataFile; the declared prop plans and the
sizing plan must name plan files. Then pfb.registry.api.register_hypothesis makes the registry-side checks
(committed at HEAD, strategy hash, SPEC, budget, rule C5, upstream sync, power at N + budget) and appends the
record. N is unchanged.

A book (schema 3) is checked per member: each member's strategy linted and loaded for its SPEC,
each member's (root, window) asked of the gateway, the declared plans and the book's account must name plan
files; then pfb.registry.bookreg.register_book. dry_run makes every check and appends nothing.

Event costs: the cost model is identified with the declared event_costs ("on": <model>+events
and the calendar's sha256 in its config sha256), and under "on" every declared window (a book: each member
window) must be covered by the pinned release calendar (RunRefused otherwise).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

from pfb.costs.model import CostModelError
from pfb.data.errors import ClosedWindowRefused, GatewayRefusal
from pfb.registry.api import HypothesisFacts, RegistryConfig, hypothesis_refusals, register_hypothesis
from pfb.registry.bookfile import BookHypothesis
from pfb.registry.bookreg import book_refusals, register_book
from pfb.registry.chain import LedgerLine
from pfb.registry.errors import ClosedWindowTouched, HypothesisInvalid, NoDataFile, RegistryError
from pfb.registry.hypfile import ALL_PLANS, CostDecl, Hypothesis, load_hypothesis
from pfb.registry.inputs import EVENT_COSTS_ON
from pfb.runner.api import RunRefused
from pfb.runner.deps import RunnerDeps, default_deps
from pfb.runner.secondary import FIRMS_DIR

Refusal = RegistryError | RunRefused | GatewayRefusal


def _check_pair(root: str, first: str, last: str, deps: RunnerDeps) -> None:
    try:
        files = deps.data_files(root, date.fromisoformat(first), date.fromisoformat(last))
    except ClosedWindowRefused as exc:
        raise ClosedWindowTouched(f"{root} {first}..{last} touches a closed window: {exc}") from exc
    except GatewayRefusal as exc:
        raise NoDataFile(f"{root} {first}..{last}: {type(exc).__name__}: {exc}") from exc
    if not files:
        raise NoDataFile(f"the catalog names no data file for {root} {first}..{last}")


def _pairs(hyp: Hypothesis | BookHypothesis) -> list[tuple[str, str, str]]:
    """(root, first, last) to ask of the gateway: every root x window, or each book member's own pair."""
    if isinstance(hyp, BookHypothesis):
        return list(dict.fromkeys((m.root, *m.window) for m in hyp.members))
    return [(root, first, last) for root in hyp.roots for first, last in hyp.windows]


def _check_data(hyp: Hypothesis | BookHypothesis, deps: RunnerDeps) -> None:
    for root, first, last in _pairs(hyp):
        _check_pair(root, first, last, deps)


def _event_windows(hyp: Hypothesis | BookHypothesis) -> list[tuple[str, str]]:
    """The windows event_costs "on" must find covered: every declared window (a book: each member's)."""
    if hyp.cost.event_costs != EVENT_COSTS_ON:
        return []
    return list(dict.fromkeys((first, last) for _, first, last in _pairs(hyp)))


def _check_event_window(first: str, last: str, deps: RunnerDeps) -> None:
    """The pinned release calendar covers the window, else RunRefused."""
    try:
        deps.event_coverage(date.fromisoformat(first), date.fromisoformat(last))
    except CostModelError as exc:  # EventCoverageError, ReleaseCalendarError
        raise RunRefused(f"{type(exc).__name__}: {exc}") from exc


def _check_events(hyp: Hypothesis | BookHypothesis, deps: RunnerDeps) -> None:
    for first, last in _event_windows(hyp):
        _check_event_window(first, last, deps)


def _check_plans(hyp: Hypothesis | BookHypothesis, firms_dir: Path) -> None:
    named = [p for p in hyp.secondary.prop_plans if p != ALL_PLANS]
    if hyp.suite.sizing_plan:
        named.append(hyp.suite.sizing_plan)
    if isinstance(hyp, BookHypothesis):
        named.append(hyp.account)  # the one account the book is pooled into
    for name in named:
        firm, _, stem = name.partition("/")
        if not (firms_dir / firm / f"{stem}.toml").is_file():
            raise HypothesisInvalid(f"plan {name!r} names no file <firm>/<plan>.toml under config/firms")


def _facts(strategy_path: Path, cost: CostDecl, deps: RunnerDeps) -> HypothesisFacts:
    violations = deps.lint_strategy(strategy_path)
    if violations:
        raise RunRefused(f"{strategy_path.name}: {len(violations)} lint violation(s); first: {violations[0]}")
    spec = deps.load_strategy(strategy_path).spec
    model_id, model_sha = deps.cost_model_ids(cost.model, cost.event_costs)  # "on": <model>+events
    contracts = getattr(spec, "contracts", None)
    return HypothesisFacts(
        spec_root=spec.root,
        spec_bar_minutes=spec.bar_minutes,
        spec_session=spec.session,
        spec_params=tuple(spec.params),
        cost_model_id=model_id,
        cost_config_sha256=model_sha,
        fees_sha256=deps.fees_sha256(cost.fees),
        closed_windows_sha256=deps.closed_windows_sha256(),
        spec_grids={name: tuple(p.grid) for name, p in spec.params.items()},
        spec_contracts=contracts if isinstance(contracts, int) and not isinstance(contracts, bool) else None,
    )


def resolve_facts(hyp: Hypothesis, deps: RunnerDeps, config_root: Path) -> HypothesisFacts:
    """The SPEC (lint first: nothing is executed before it passes), cost and fee identities."""
    return _facts(config_root / hyp.strategy, hyp.cost, deps)


def member_facts(book: BookHypothesis, deps: RunnerDeps, config_root: Path) -> list[HypothesisFacts]:
    """Each member's SPEC facts (its own strategy, linted first) with the book's cost and fee identities."""
    return [_facts(config_root / m.strategy, book.cost, deps) for m in book.members]


def _load(path: Path, deps: RunnerDeps) -> tuple[Hypothesis | BookHypothesis, RegistryConfig, Path]:
    config = deps.registry_config()
    path = path if path.is_absolute() else config.repo_root / path
    return load_hypothesis(path), config, path


def register(path: Path, deps: RunnerDeps | None = None, firms_dir: Path = FIRMS_DIR) -> LedgerLine:
    """Every refusal of TRIALS.md section 2 (a book: of each member), then the record (N unchanged)."""
    deps = deps or default_deps()
    hyp, config, path = _load(path, deps)
    _check_plans(hyp, firms_dir)
    try:
        if isinstance(hyp, BookHypothesis):
            facts = member_facts(hyp, deps, config.repo_root)
        else:
            facts = resolve_facts(hyp, deps, config.repo_root)
    except GatewayRefusal as exc:
        raise RunRefused(f"{type(exc).__name__}: {exc}") from exc
    _check_data(hyp, deps)
    _check_events(hyp, deps)
    if isinstance(facts, list):
        return register_book(path, config, facts)
    return register_hypothesis(path, config, facts)


def _collect(out: list[Refusal], check: Callable[[], object]) -> object:
    try:
        return check()
    except (RegistryError, RunRefused, GatewayRefusal) as exc:
        out.append(exc)
        return None


def dry_run(path: Path, deps: RunnerDeps | None = None, firms_dir: Path = FIRMS_DIR) -> list[Refusal]:
    """`pfb hypothesis register --dry-run`: every check of register, nothing appended; the refusals found
    (empty when the file would register). A file that does not parse is one refusal; a book lists each
    member's refusals (a still-closed member window among them). The registry-side checks need every
    member's SPEC facts, so they run only when all of them resolved."""
    used = deps or default_deps()
    out: list[Refusal] = []
    loaded = _collect(out, lambda: _load(path, used))
    if not isinstance(loaded, tuple):
        return out
    hyp, config, path = loaded
    _collect(out, lambda: _check_plans(hyp, firms_dir))
    for root, first, last in _pairs(hyp):
        _collect(out, lambda r=root, f=first, la=last: _check_pair(r, f, la, used))
    for first, last in _event_windows(hyp):
        _collect(out, lambda f=first, la=last: _check_event_window(f, la, used))
    if isinstance(hyp, BookHypothesis):
        cost = hyp.cost
        facts = [
            _collect(out, lambda m=m: _facts(config.repo_root / m.strategy, cost, used)) for m in hyp.members
        ]
        if all(isinstance(f, HypothesisFacts) for f in facts):
            book_facts = [f for f in facts if isinstance(f, HypothesisFacts)]
            out.extend(_collect_list(out, lambda: book_refusals(path, config, book_facts)))
        return out
    single = _collect(out, lambda: resolve_facts(hyp, used, config.repo_root))
    if isinstance(single, HypothesisFacts):
        out.extend(_collect_list(out, lambda: hypothesis_refusals(path, config, single)))
    return out


def _collect_list(out: list[Refusal], check: Callable[[], list[RegistryError]]) -> list[RegistryError]:
    result = _collect(out, check)
    return result if isinstance(result, list) else []
