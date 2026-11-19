"""The automatic check (ADR A15): 20 seeded random cuts, at most 200 cuts after
entry bars, then at most 100 after exit bars, and at step 1 a cut after every bar of 5 sessions spread evenly;
the side-choosing and the suppressing mutants. Synthetic bars only."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
from lookahead_mutants import (  # pyright: ignore[reportMissingImports]
    next_bar_view,
    suppress_with_next_bar,
)

from pfb.engine.lookahead import (
    SIGNAL_BAR_CUTS,
    SWEEP_SESSIONS,
    LookAheadDetected,
    lookahead_check,
    select_evenly,
    session_sweep_check,
    truncation_check,
)
from pfb.engine.strategy import load_strategy, resolve_params
from pfb.engine.synthetic import synthetic_bars

REFERENCE = Path(__file__).resolve().parents[2] / "strategies" / "reference_breakout.py"
SEEDS = (0, 1, 7, 42, 2025)
ACCEPTANCE_SEEDS = tuple(range(30))
SWEEP_CATCHES = (1, 42)  # seeds whose first session holds a suppressed entry (measured; deterministic)
FIRST, LAST = date(2025, 4, 1), date(2025, 4, 11)


@pytest.fixture(scope="module")
def bars_by_seed():  # type: ignore[no-untyped-def]
    return {s: synthetic_bars("MNQ", seed=s, first=FIRST, last=LAST) for s in SEEDS}


@pytest.fixture(scope="module")
def reference():  # type: ignore[no-untyped-def]
    strategy = load_strategy(REFERENCE)
    return strategy, resolve_params(strategy.spec, None)


# ------------------------------------------------------------- selection ----
def test_select_evenly_keeps_everything_up_to_the_limit() -> None:
    assert select_evenly([3, 5, 9], 200) == [3, 5, 9]


def test_select_evenly_keeps_first_and_last_and_exactly_the_limit_without_a_seed() -> None:
    items = list(range(10, 1010))
    picked = select_evenly(items, SIGNAL_BAR_CUTS)
    assert len(picked) == SIGNAL_BAR_CUTS == len(set(picked))
    assert picked[0] == 10 and picked[-1] == 1009 and picked == sorted(picked)
    assert picked == select_evenly(items, SIGNAL_BAR_CUTS)


def test_select_evenly_refuses_a_limit_below_two() -> None:
    with pytest.raises(ValueError):
        select_evenly([1, 2, 3], 1)


# ------------------------------------------------------- the automatic check ----
@pytest.mark.parametrize("seed", SEEDS)
def test_the_automatic_check_passes_the_honest_reference(bars_by_seed, reference, seed: int) -> None:  # type: ignore[no-untyped-def]
    strategy, params = reference
    lookahead_check(strategy.generate, bars_by_seed[seed], params, seed)


@pytest.mark.parametrize("seed", SEEDS)
def test_the_automatic_check_catches_a_sparse_one_bar_peek(bars_by_seed, reference, seed: int) -> None:  # type: ignore[no-untyped-def]
    # Arrange
    strategy, params = reference

    def peeking(bars, p):  # type: ignore[no-untyped-def]
        return strategy.generate(next_bar_view(bars), p)

    # Act / Assert
    with pytest.raises(LookAheadDetected, match="cut after a signal bar"):
        lookahead_check(peeking, bars_by_seed[seed], params, seed)


def test_a_leaky_random_cut_still_raises_with_the_seed_in_the_message(bars_by_seed) -> None:  # type: ignore[no-untyped-def]
    def full_mean(bars, p):  # type: ignore[no-untyped-def]
        close = bars["close"]
        n = len(close)
        z = np.zeros(n, dtype=np.int64)
        return {"entry_long": close > close.mean(), "entry_short": np.zeros(n, dtype=np.bool_),
                "exit": np.zeros(n, dtype=np.bool_), "stop_ticks": z, "target_ticks": z.copy()}  # fmt: skip

    with pytest.raises(LookAheadDetected, match=r"\(seed 7\)"):
        lookahead_check(full_mean, bars_by_seed[7], {}, 7)


def _caught(fn) -> bool:  # type: ignore[no-untyped-def]
    try:
        fn()
    except LookAheadDetected:
        return True
    return False


@pytest.fixture(scope="module")
def month_by_seed():  # type: ignore[no-untyped-def]
    """Step 1's own bars: the default synthetic month, 30 fixed seeds."""
    return {s: synthetic_bars("MNQ", seed=s) for s in ACCEPTANCE_SEEDS}


# ------------------------------------------- E2: entry cuts before exit cuts ----
def _side_from_next_bar(base, every: int):  # type: ignore[no-untyped-def]
    """A dilution probe: on trade days % every == 0 the NEXT bar chooses the entry's side."""

    def generate(bars, p):  # type: ignore[no-untyped-def]
        out = {k: np.array(v, copy=True) for k, v in base(bars, p).items()}
        close, day, n = bars["close"], bars["trade_day"], len(bars["close"])
        for i in np.flatnonzero(out["entry_long"] | out["entry_short"]).tolist():
            if i + 1 < n and day[i] % every == 0 and day[i + 1] == day[i]:
                up = bool(close[i + 1] > close[i])
                out["entry_long"][i], out["entry_short"][i] = up, not up
        return out

    return generate


@pytest.mark.slow
@pytest.mark.parametrize("every", [1, 2, 5])
def test_e2_the_in_run_check_catches_the_side_choosing_peek_27_times_in_30(
    month_by_seed,
    reference,
    every: int,  # type: ignore[no-untyped-def]
) -> None:
    # Arrange: the in-run check (random, entry and exit cuts; no sweep) on 30 fixed seeds
    strategy, params = reference
    mutant = _side_from_next_bar(strategy.generate, every)
    # Act
    bars = month_by_seed
    caught = sum(_caught(lambda s=s: lookahead_check(mutant, bars[s], params, s)) for s in ACCEPTANCE_SEEDS)
    # Assert (measured 30, 30, 27; before E2: 20, 6, 0)
    assert caught >= 27, f"caught {caught}/30 with every={every}"


# ------------------------------------------------- the suppressing mutant ----
@pytest.mark.parametrize("seed", SEEDS)
def test_the_random_cuts_alone_miss_the_suppressing_mutant(bars_by_seed, reference, seed: int) -> None:  # type: ignore[no-untyped-def]
    # Documents why E3 wires the sweep: random cuts never see a later bar that only suppresses a signal.
    strategy, params = reference
    truncation_check(suppress_with_next_bar(strategy.generate), bars_by_seed[seed], params, seed)


@pytest.mark.slow
def test_e3_step_1_catches_the_suppressing_mutant_25_times_in_30_without_a_false_alarm(
    month_by_seed,
    reference,  # type: ignore[no-untyped-def]
) -> None:
    # Arrange: step 1's configuration (default month, SWEEP_SESSIONS sessions spread evenly)
    strategy, params = reference
    mutant = suppress_with_next_bar(strategy.generate)

    def check(gen, s: int):  # type: ignore[no-untyped-def]
        return lambda: lookahead_check(gen, month_by_seed[s], params, s, sweep_sessions=SWEEP_SESSIONS)

    # Act
    caught = sum(_caught(check(mutant, s)) for s in ACCEPTANCE_SEEDS)
    false_alarms = sum(_caught(check(strategy.generate, s)) for s in ACCEPTANCE_SEEDS)
    # Assert (was a strict xfail: 0/30 without the sweep)
    assert caught >= 25, f"caught {caught}/30"
    assert false_alarms == 0


@pytest.mark.parametrize("seed", SWEEP_CATCHES)
def test_the_session_sweep_catches_the_suppressing_mutant_on_these_seeds(
    bars_by_seed,
    reference,
    seed: int,  # type: ignore[no-untyped-def]
) -> None:
    strategy, params = reference
    with pytest.raises(LookAheadDetected, match="sweep of session 0"):
        session_sweep_check(suppress_with_next_bar(strategy.generate), bars_by_seed[seed], params)


@pytest.mark.parametrize("seed", SEEDS)
def test_the_session_sweep_passes_the_honest_reference(bars_by_seed, reference, seed: int) -> None:  # type: ignore[no-untyped-def]
    strategy, params = reference
    session_sweep_check(strategy.generate, bars_by_seed[seed], params)


def test_the_session_sweep_refuses_a_session_that_does_not_exist(bars_by_seed, reference) -> None:  # type: ignore[no-untyped-def]
    strategy, params = reference
    with pytest.raises(ValueError, match="outside"):
        session_sweep_check(strategy.generate, bars_by_seed[0], params, session=10_000)
