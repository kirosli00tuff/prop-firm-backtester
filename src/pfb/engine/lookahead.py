"""Look-ahead checks (ADR-001 section 12). The automatic check runs in pfb.runner step 1 (synthetic bars, with
the K-session sweep) and inside simulate on the run's own bars (without it); a failure refuses the run.
The planted-future canaries are in pfb.engine.canaries.

Call order: the random and the sweep cuts in one ascending sequence; then
generate(bars[:n-1]), whose signals choose the cuts after entry bars and after exit bars; those cuts, each
group in ascending k; the call on the full bars LAST. Its Signals is the check's result and the one the
kernel gets: no further call. Each output is compared with the next longer call of its sequence, the longest
of a sequence with the n-1 call, and that with the full call; equality is transitive, so every cut k is
checked against generate(bars)[:k] while at most three outputs are held.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from pfb.engine.strategy import (
    SIGNAL_KEYS,
    Bars,
    GenerateFn,
    IsolatedGenerate,
    ParamValue,
    Signals,
    bar_count,
    frozen_params,
    normalize_signals,
    owned_bars,
    slice_bars,
)

TRUNCATION_CUTS = 20
ENTRY_BAR_CUTS = 200  # cuts after entry bars first ...
EXIT_BAR_CUTS = 100  # ... then a second budget after exit bars; both chosen without the seed
SIGNAL_BAR_CUTS = ENTRY_BAR_CUTS
SWEEP_SESSIONS = 5  # step 1 cuts after every bar of 5 sessions spread evenly
NO_SIGNAL_WARNING = "look-ahead check saw no signal"
ANCHOR_TAG = "cut before the last bar"
FULL_TAG = "full bars"


class LookAheadDetected(RuntimeError):
    """generate(bars[:k]) differs from generate(bars)[:k]: a value at bar i used a bar after i."""


@dataclass(frozen=True)
class CheckResult:
    signals: Signals  # the LAST call's output, on the full bars: the Signals the kernel is handed
    calls: int  # generate calls, the full one included
    entry_bars: int  # bars with an entry flag in the full output
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _Out:
    k: int
    tag: str
    signals: Signals


def _call(generate: GenerateFn, bars: Bars, k: int, params: Mapping[str, ParamValue], tag: str) -> _Out:
    """One call on the first k bars. A loaded strategy copies in IsolatedGenerate; any other callable gets
    its own read-only copies here, so no call is ever handed a view of a shared buffer."""
    part = slice_bars(bars, k)
    if not isinstance(generate, IsolatedGenerate):
        part = owned_bars(part)
    return _Out(k, tag, normalize_signals(generate(part, frozen_params(params)), k))


def _compare(short: _Out, long: _Out) -> None:
    for key in SIGNAL_KEYS:
        head, cut = long.signals.get(key)[: short.k], short.signals.get(key)
        if not np.array_equal(head, cut):
            first = int(np.flatnonzero(head != cut)[0])
            raise LookAheadDetected(
                f"truncation at k={short.k} ({short.tag}): output {key} differs at bar {first} from the call "
                f"on {long.k} bars ({long.tag}); a value at bar i depends on a later bar"
            )


# ----------------------------------------------------------------- cut points ----
def _cuts(n: int, seed: int) -> list[int]:
    """Up to TRUNCATION_CUTS distinct seeded cut points k in 1..n-1, ascending."""
    if n < 2:
        return []
    rng = np.random.default_rng(seed)
    count = min(TRUNCATION_CUTS, n - 1)
    return sorted(int(k) for k in rng.choice(np.arange(1, n), size=count, replace=False))


def select_evenly(items: Sequence[int], limit: int) -> list[int]:
    """All items when there are at most `limit`; otherwise the first, the last and evenly spaced ones
    between them, `limit` in all. Deterministic: no seed."""
    if limit < 2:
        raise ValueError("limit must be >= 2")
    if len(items) <= limit:
        return list(items)
    picks = np.unique(np.rint(np.linspace(0, len(items) - 1, limit)).astype(np.int64))
    return [items[int(p)] for p in picks]


def signal_bars(signals: Signals) -> np.ndarray:
    """Indices of the bars the kernel acts on: an entry or an exit flag is set."""
    return np.flatnonzero(signals.entry_long | signals.entry_short | signals.exit)


def _entry_bars(signals: Signals) -> np.ndarray:
    return np.flatnonzero(signals.entry_long | signals.entry_short)


def _after(indices: np.ndarray, budget: int | None) -> list[int]:
    cuts = [int(i) + 1 for i in indices]
    return cuts if budget is None else select_evenly(cuts, budget)


def _signal_groups(
    anchor: Signals, entry_budget: int | None, exit_budget: int | None
) -> list[tuple[str, list[int]]]:
    """Cuts right after entry bars (first), then after exit bars that are not entry bars."""
    entries = _entry_bars(anchor)
    exits = np.flatnonzero(anchor.exit & ~(anchor.entry_long | anchor.entry_short))
    return [("cut after a signal bar: entry", _after(entries, entry_budget)),
            ("cut after a signal bar: exit", _after(exits, exit_budget))]  # fmt: skip


def session_starts(bars: Bars) -> np.ndarray:
    """First bar index of each trade-day session, plus the bar count as a closing bound."""
    day = bars["trade_day"]
    n = bar_count(bars)
    if n == 0:
        return np.zeros(1, dtype=np.int64)
    inner = np.flatnonzero(np.diff(day) != 0) + 1
    return np.concatenate((np.zeros(1, dtype=np.int64), inner, np.full(1, n, dtype=np.int64)))


def _sweep_group(bounds: np.ndarray, session: int) -> tuple[str, list[int]]:
    if not 0 <= session < len(bounds) - 1:
        raise ValueError(f"session {session} is outside the {len(bounds) - 1} sessions of the bars")
    return f"sweep of session {session}", list(range(int(bounds[session]) + 1, int(bounds[session + 1]) + 1))


def _sweep_groups(bars: Bars, sessions: int) -> list[tuple[str, list[int]]]:
    """Cuts after every bar of `sessions` sessions spread evenly over the series (first and last included)."""
    if sessions <= 0 or bar_count(bars) == 0:
        return []
    bounds = session_starts(bars)
    every = list(range(len(bounds) - 1))
    chosen = every[:1] if sessions == 1 else select_evenly(every, sessions)
    return [_sweep_group(bounds, s) for s in chosen]


# ---------------------------------------------------------------- the engine ----
@dataclass(frozen=True)
class _Plan:
    seed: int | None = None
    entry_budget: int | None = None
    exit_budget: int | None = None
    signal_cuts: bool = False
    sweep: tuple[tuple[str, list[int]], ...] = ()


Tagged = list[tuple[int, str]]


def _ascending(plan: _Plan, n: int) -> Tagged:
    """The random and the sweep cuts in ONE ascending sequence, called before the n-1 call: each of these
    calls sees more bars than every call before it."""
    tagged: dict[int, str] = {}
    for k in _cuts(n, plan.seed) if plan.seed is not None else []:
        tagged.setdefault(k, f"seed {plan.seed}")
    for tag, cuts in plan.sweep:
        for k in cuts:
            tagged.setdefault(k, tag)
    return sorted((k, t) for k, t in tagged.items() if 1 <= k < n - 1)


def _chain(generate: GenerateFn, bars: Bars, params: Mapping[str, ParamValue], cuts: Tagged) -> _Out | None:
    """Call each cut in ascending k and compare it with the next one; returns the longest output."""
    previous: _Out | None = None
    for k, tag in cuts:
        current = _call(generate, bars, k, params, tag)
        if previous is not None:
            _compare(previous, current)
        previous = current
    return previous


def _signal_chains(
    generate: GenerateFn,
    bars: Bars,
    params: Mapping[str, ParamValue],
    plan: _Plan,
    anchor: _Out,
    done: set[int],
) -> int:
    """The entry then the exit cuts, chosen from the n-1 call (made before them, after the ascending ones)."""
    calls = 0
    for tag, cuts in _signal_groups(anchor.signals, plan.entry_budget, plan.exit_budget):
        fresh = [(k, tag) for k in sorted(set(cuts) - done) if 1 <= k < anchor.k]
        done.update(k for k, _ in fresh)
        tail = _chain(generate, bars, params, fresh)
        calls += len(fresh)
        if tail is not None:
            _compare(tail, anchor)
    return calls


def _run(generate: GenerateFn, bars: Bars, params: Mapping[str, ParamValue], plan: _Plan) -> CheckResult:
    n = bar_count(bars)
    ascending = _ascending(plan, n)
    tail = _chain(generate, bars, params, ascending)
    calls = len(ascending)
    anchor = _call(generate, bars, n - 1, params, ANCHOR_TAG) if n >= 2 else None
    if anchor is not None:
        calls += 1
        if tail is not None:
            _compare(tail, anchor)
        if plan.signal_cuts:
            calls += _signal_chains(generate, bars, params, plan, anchor, {k for k, _ in ascending})
    full = _call(generate, bars, n, params, FULL_TAG)  # LAST: its Signals is the result
    if anchor is not None:
        _compare(anchor, full)
    entries = len(_entry_bars(full.signals))
    warnings = () if entries else (NO_SIGNAL_WARNING,)
    return CheckResult(signals=full.signals, calls=calls + 1, entry_bars=entries, warnings=warnings)


# ----------------------------------------------------------------- public API ----
def truncation_check(
    generate: GenerateFn, bars: Bars, params: Mapping[str, ParamValue], seed: int
) -> CheckResult:
    """The 20 seeded random cuts (and the n-1 cut): generate(bars[:k]) must equal generate(bars)[:k]."""
    return _run(generate, bars, params, _Plan(seed=seed))


def signal_bar_check(
    generate: GenerateFn, bars: Bars, params: Mapping[str, ParamValue], max_cuts: int | None = None
) -> CheckResult:
    """Cuts right after entry bars, then after exit bars (each group bounded by max_cuts, all when None).
    A sparse strategy that peeks one bar ahead changes its outputs only near its signal bars."""
    return _run(generate, bars, params, _Plan(signal_cuts=True, entry_budget=max_cuts, exit_budget=max_cuts))


def session_sweep_check(
    generate: GenerateFn, bars: Bars, params: Mapping[str, ParamValue], session: int = 0
) -> CheckResult:
    """A cut after EVERY bar of one session: catches a later bar that only SUPPRESSES a signal."""
    sweep = (_sweep_group(session_starts(bars), session),) if bar_count(bars) else ()
    return _run(generate, bars, params, _Plan(sweep=sweep))


def lookahead_check(
    generate: GenerateFn,
    bars: Bars,
    params: Mapping[str, ParamValue],
    seed: int,
    *,
    sweep_sessions: int = 0,
) -> CheckResult:
    """The automatic check: 20 seeded random cuts, up to ENTRY_BAR_CUTS cuts after entry bars, up to
    EXIT_BAR_CUTS after exit bars and, at step 1, a cut after every bar of `sweep_sessions` sessions spread
    evenly. Raises LookAheadDetected; returns the full call's Signals and the warnings (NO_SIGNAL_WARNING when
    the full output holds no entry bar)."""
    plan = _Plan(seed=seed, signal_cuts=True, entry_budget=ENTRY_BAR_CUTS, exit_budget=EXIT_BAR_CUTS,
                 sweep=tuple(_sweep_groups(bars, sweep_sessions)))  # fmt: skip
    return _run(generate, bars, params, plan)


__all__ = [
    "ENTRY_BAR_CUTS", "EXIT_BAR_CUTS", "NO_SIGNAL_WARNING", "SIGNAL_BAR_CUTS", "SWEEP_SESSIONS",
    "TRUNCATION_CUTS", "CheckResult", "LookAheadDetected", "lookahead_check", "select_evenly",
    "session_starts", "session_sweep_check", "signal_bar_check", "signal_bars", "truncation_check",
]  # fmt: skip
