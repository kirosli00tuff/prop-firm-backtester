"""The plan of a counted trial (docs/TRIALS.md C4, C6, section 4): everything bound to the hypothesis.

The hypothesis file (schema 2) fixes the strategy, roots, windows, session, cost model, event costs (no
switch), fees, the suite and the secondary outputs. A command-line value that differs is refused
(SuiteMismatch); one left out is taken from the hypothesis. The root defaults to the SPEC root; the window is
the 1-based index of a [[window]] table, needed only when more than one is declared. Parameters outside the
grid and roots or windows not declared are refused here (UndeclaredConfiguration) before anything is logged,
and again by the registry's replay at begin.

A book (schema 3): the trial is one member; (strategy, root, params, window) must be exactly one
declared member (NotABookMember), and fees, cost model, suite and secondary outputs are the book's.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

from pfb.registry.api import RegistryConfig
from pfb.registry.bookfile import BookHypothesis, BookMember, member_identity
from pfb.registry.errors import NotABookMember, SuiteMismatch, UndeclaredConfiguration
from pfb.registry.hypfile import Hypothesis, load_hypothesis
from pfb.registry.records import in_grid
from pfb.runner.api import Plan, RunRefused
from pfb.runner.deps import RunnerDeps


def trial_hypothesis(
    args: Mapping[str, Any], config: RegistryConfig
) -> tuple[Hypothesis | BookHypothesis, Path]:
    """The trial's hypothesis file, parsed (the registry checks it is committed and registered at begin)."""
    path = args["hypothesis_path"]
    if path is None:
        raise RunRefused("a trial names its hypothesis file (--hypothesis hypotheses/<id>.toml)")
    path = path if path.is_absolute() else config.repo_root / path
    return load_hypothesis(path), path.resolve()


def _window(hyp: Hypothesis | BookHypothesis, args: Mapping[str, Any]) -> tuple[date, date]:
    index = args["window"]
    if index is None:
        if len(hyp.windows) != 1:
            raise RunRefused(
                f"{hyp.id!r} declares {len(hyp.windows)} windows: pass --window 1..{len(hyp.windows)}"
            )
        index = 1
    if not 1 <= index <= len(hyp.windows):
        raise UndeclaredConfiguration(f"window {index} is not one of the {len(hyp.windows)} declared windows")
    first, last = (date.fromisoformat(d) for d in hyp.windows[index - 1])
    for given, declared in ((args["first"], first), (args["last"], last)):
        if given is not None and given != declared:
            raise UndeclaredConfiguration(f"window {first}..{last} of {hyp.id!r} is fixed by its [[window]]")
    return first, last


def _bound(hyp: Hypothesis | BookHypothesis, args: Mapping[str, Any]) -> None:
    """C4: fees, cost model and suite are the hypothesis's; a differing value given for them is refused."""
    if args["fees"] not in (None, hyp.cost.fees) or args["cost_model"] not in (None, hyp.cost.model):
        raise SuiteMismatch(
            f"fees and cost model of a trial of {hyp.id!r} are {hyp.cost.fees!r}, {hyp.cost.model!r}"
        )
    if args["suite"] is not None and args["suite"] != hyp.suite:
        raise SuiteMismatch(f"the suite of a trial is the [suite] of {hyp.id!r}; no switch changes it")
    if args["check_id"] is not None:
        raise RunRefused("a trial takes no check id")


def plan_trial(
    strategy: Any, resolved: dict[str, Any], args: Mapping[str, Any], deps: RunnerDeps, config: RegistryConfig
) -> Plan:
    """The trial's Plan; data files resolved through the gateway (no bar read)."""
    hyp, hyp_path = trial_hypothesis(args, config)
    if isinstance(hyp, BookHypothesis):
        return _plan_member(hyp, hyp_path, strategy, resolved, args, deps, config)
    if (config.repo_root / hyp.strategy).resolve() != Path(strategy.path).resolve():
        raise RunRefused(f"{hyp.id!r} pins {hyp.strategy}, not {Path(strategy.path).name}")
    if hyp.suite.neighbourhood and not args.get("allow_neighbourhood", False):
        raise SuiteMismatch(f"{hyp.id!r} declares neighbourhood = true: run it with run_with_neighbours")
    _bound(hyp, args)
    root = args["root"] or strategy.spec.root
    if root not in hyp.roots:
        raise UndeclaredConfiguration(f"root {root!r} is not one of {hyp.id!r}'s roots {list(hyp.roots)}")
    if not in_grid(resolved, {k: list(v) for k, v in hyp.grid.items()}):
        raise UndeclaredConfiguration(f"params {resolved} lie outside the declared grid of {hyp.id!r}")
    first, last = _window(hyp, args)
    return Plan(
        root=root,
        first=first,
        last=last,
        params=resolved,
        fees=hyp.cost.fees,
        cost_model=hyp.cost.model,
        data_files=deps.data_files(root, first, last),
        check_id=None,
        hypothesis_id=hyp.id,
        hypothesis_path=hyp_path,
        suite=hyp.suite,
        secondary=hyp.secondary.to_json(),
        event_costs=hyp.cost.event_costs,  # from the hypothesis only
    )


def book_member(
    book: BookHypothesis, strategy: Any, root: str, params: Mapping[str, Any], window: tuple[date, date],
    config: RegistryConfig,
) -> BookMember:  # fmt: skip
    """Clause 2 (the runner's half; replay refuses it again at begin): exactly one declared member is this
    (strategy file and sha256, root, params, window), else NotABookMember before anything is logged."""
    pair = (window[0].isoformat(), window[1].isoformat())
    mine = member_identity(strategy.sha256, root, params, pair)
    found = [
        m
        for m in book.members
        if m.identity() == mine and (config.repo_root / m.strategy).resolve() == Path(strategy.path).resolve()
    ]
    if len(found) != 1:
        raise NotABookMember(
            f"strategy {Path(strategy.path).name}, root {root}, params {dict(params)}, window "
            f"{pair[0]}..{pair[1]} is not exactly one declared member of book {book.id!r}"
        )
    return found[0]


def _plan_member(
    book: BookHypothesis, hyp_path: Path, strategy: Any, resolved: dict[str, Any], args: Mapping[str, Any],
    deps: RunnerDeps, config: RegistryConfig,
) -> Plan:  # fmt: skip
    """A book member's Plan: the window from `--window K` over the book's distinct member windows (or the one
    window), the member matched exactly, the book's cost, fees, suite and secondary outputs."""
    _bound(book, args)
    root = args["root"] or strategy.spec.root
    first, last = _window(book, args)
    member = book_member(book, strategy, root, resolved, (first, last), config)
    return Plan(
        root=member.root,
        first=first,
        last=last,
        params=resolved,
        fees=book.cost.fees,
        cost_model=book.cost.model,
        data_files=deps.data_files(member.root, first, last),
        check_id=None,
        hypothesis_id=book.id,
        hypothesis_path=hyp_path,
        suite=book.suite,
        secondary=book.secondary.to_json(),
        event_costs=book.cost.event_costs,  # every member runs at the book's event costs
    )
