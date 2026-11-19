"""E5 truncation invariance (ADR-001 section 12): honest strategies pass, leaky strategy mutants are caught,
and the run path refuses a leaky strategy before the kernel. Synthetic bars only."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
from lookahead_mutants import (  # pyright: ignore[reportMissingImports]
    STRATEGY_MUTANTS,
    honest_momentum,
    next_bar_view,
    signals_from,
)

from pfb.engine.canaries import canary_reader, canary_strategy, planted_synthetic, zero_cost_inputs
from pfb.engine.lookahead import LookAheadDetected, signal_bar_check, signal_bars, truncation_check
from pfb.engine.run import simulate
from pfb.engine.strategy import Strategy, load_strategy, normalize_signals, resolve_params
from pfb.engine.synthetic import synthetic_bars, synthetic_frame

REFERENCE = Path(__file__).resolve().parents[2] / "strategies" / "reference_breakout.py"
SEEDS = (0, 1, 7, 42, 2025)
SHORT_FIRST, SHORT_LAST = date(2025, 4, 1), date(2025, 4, 11)


@pytest.fixture(scope="module")
def bars_by_seed():  # type: ignore[no-untyped-def]
    return {s: synthetic_bars("MNQ", seed=s, first=SHORT_FIRST, last=SHORT_LAST) for s in SEEDS}


@pytest.mark.parametrize("seed", SEEDS)
def test_the_honest_reference_strategy_passes_the_truncation_check(bars_by_seed, seed: int) -> None:  # type: ignore[no-untyped-def]
    # Arrange
    strategy = load_strategy(REFERENCE)
    params = resolve_params(strategy.spec, None)
    # Act / Assert: no LookAheadDetected
    truncation_check(strategy.generate, bars_by_seed[seed], params, seed)


@pytest.mark.parametrize("seed", SEEDS)
def test_honest_momentum_and_the_canary_reader_pass_the_truncation_check(bars_by_seed, seed: int) -> None:  # type: ignore[no-untyped-def]
    bars = bars_by_seed[seed]
    markers = {int(t): 1 for t in bars["ts_event"][::17]}
    truncation_check(honest_momentum, bars, {}, seed)
    truncation_check(canary_reader(markers), bars, {}, seed)


@pytest.mark.parametrize("name", sorted(STRATEGY_MUTANTS))
@pytest.mark.parametrize("seed", SEEDS)
def test_every_leaky_strategy_mutant_is_caught(bars_by_seed, name: str, seed: int) -> None:  # type: ignore[no-untyped-def]
    # Arrange
    mutant = STRATEGY_MUTANTS[name]
    # Act / Assert
    with pytest.raises(LookAheadDetected, match="truncation at k="):
        truncation_check(mutant, bars_by_seed[seed], {}, seed)


def test_a_strategy_fed_the_next_bar_is_caught_when_its_signals_are_dense(bars_by_seed) -> None:  # type: ignore[no-untyped-def]
    def peeking(bars, params):  # type: ignore[no-untyped-def]
        return honest_momentum(next_bar_view(bars), params)

    with pytest.raises(LookAheadDetected):
        truncation_check(peeking, bars_by_seed[0], {}, 0)


def test_the_run_path_refuses_a_leaky_strategy_before_the_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: a kernel that must never be reached
    frame = synthetic_frame("MNQ", SHORT_FIRST, SHORT_LAST, seed=3)
    inputs = zero_cost_inputs("MNQ", frame)
    honest = canary_strategy({})
    leaky = Strategy(honest.spec, STRATEGY_MUTANTS["next_session_open"], honest.sha256, honest.path)

    def no_kernel(*args: object, **kwargs: object) -> None:
        raise AssertionError("the kernel ran for a leaky strategy")

    monkeypatch.setattr("pfb.engine.run.K.run_kernel", no_kernel)
    # Act / Assert
    with pytest.raises(LookAheadDetected):
        simulate(leaky, frame, {}, inputs, seed=3)


def test_the_canary_reader_on_planted_bars_passes_the_check_inside_simulate() -> None:
    planted = planted_synthetic(first=SHORT_FIRST, last=SHORT_LAST)
    inputs = zero_cost_inputs("MNQ", planted.frame)
    out = simulate(canary_strategy(planted.leak_markers), planted.frame, {}, inputs, seed=5)
    assert out.trades.height > 0


# ------------------------------------------- signal-bar cuts (not wired by design) ----
@pytest.mark.parametrize("seed", SEEDS)
def test_signal_bar_cuts_catch_a_sparse_strategy_that_peeks_one_bar(bars_by_seed, seed: int) -> None:  # type: ignore[no-untyped-def]
    # Arrange: the reference breakout fed the next bar (few signal bars; 20 random cuts rarely see it)
    strategy = load_strategy(REFERENCE)
    params = resolve_params(strategy.spec, None)

    def peeking(bars, p):  # type: ignore[no-untyped-def]
        return strategy.generate(next_bar_view(bars), p)

    # Act / Assert
    with pytest.raises(LookAheadDetected, match="cut after a signal bar"):
        signal_bar_check(peeking, bars_by_seed[seed], params)


@pytest.mark.parametrize("seed", SEEDS)
def test_signal_bar_cuts_pass_the_honest_reference_and_the_canary_reader(bars_by_seed, seed: int) -> None:  # type: ignore[no-untyped-def]
    strategy = load_strategy(REFERENCE)
    bars = bars_by_seed[seed]
    signal_bar_check(strategy.generate, bars, resolve_params(strategy.spec, None))
    signal_bar_check(canary_reader({int(t): -1 for t in bars["ts_event"][::20]}), bars, {}, max_cuts=50)


def test_signal_bars_lists_entry_and_exit_bars_only() -> None:
    n = 6
    flags = np.zeros(n, dtype=np.bool_)
    raw = signals_from(flags.copy(), flags.copy())
    raw["entry_long"][1] = True
    raw["exit"][4] = True
    assert signal_bars(normalize_signals(raw, n)).tolist() == [1, 4]
