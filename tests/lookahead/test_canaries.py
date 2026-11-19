"""E5 planted-future canaries (ADR-001 section 12): the honest engine shows no
edge on the planted future and captures the one-bar-early control; each leaky engine mutant is detected."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import numpy as np
import polars as pl
import pytest
from lookahead_mutants import (  # pyright: ignore[reportMissingImports]
    ENGINE_MUTANTS,
    honest_kernel_engine,
    kernel_trades,
)

from pfb.engine.canaries import (
    JUMP_TICKS,
    MARK_EVERY_MINUTES,
    MIN_TRADES,
    CanaryReport,
    CanaryScore,
    PlantedBars,
    canary_strategy,
    edge_detected,
    honest_engine,
    plant_canaries,
    planted_synthetic,
    run_canaries,
    score_canary,
    shows_no_edge,
    zero_cost_inputs,
)
from pfb.engine.lookahead import truncation_check
from pfb.engine.synthetic import synthetic_bars, synthetic_frame

TRADE_KEYS = ["side", "entry_ts", "entry_price", "exit_price"]
SEEDS = (0, 3, 20_260_917)
FIRST, LAST = date(2025, 4, 1), date(2025, 4, 4)


@pytest.fixture(scope="module")
def planted() -> PlantedBars:
    return planted_synthetic()


@pytest.fixture(scope="module")
def honest_report(planted: PlantedBars) -> CanaryReport:
    return run_canaries(honest_engine, planted)


# ------------------------------------------------------------------ planting ----
def test_planting_returns_a_shifted_copy_and_leaves_the_input_unchanged() -> None:
    # Arrange
    frame = synthetic_frame("MNQ", FIRST, LAST, seed=1)
    before = frame.clone()
    # Act
    planted = plant_canaries(frame, seed=9)
    # Assert
    assert frame.equals(before)
    assert planted.frame.columns == frame.columns and planted.frame.height == frame.height
    assert all(planted.frame[c].dtype == pl.Int64 for c in ("open", "high", "low", "close"))
    assert len(planted.leak_markers) == len(planted.latency_markers) > 0


def test_each_marked_bar_carries_the_jump_and_later_bars_are_shifted() -> None:
    frame = synthetic_frame("MNQ", FIRST, LAST, seed=1)
    planted = plant_canaries(frame, seed=9)
    ts = frame["ts_event"].to_numpy()
    idx = {int(t): i for i, t in enumerate(ts)}
    marks = sorted(planted.leak_markers.items())
    shift = np.zeros(len(ts), dtype=np.int64)
    for t, d in marks:
        shift[idx[t] + 1 :] += d * JUMP_TICKS
    opens = planted.frame["open"].to_numpy() - frame["open"].to_numpy()
    assert np.array_equal(opens, shift)
    for t, d in marks:
        i = idx[t]
        jumped = planted.frame["close"][i] - frame["close"][i] - shift[i]
        assert jumped == d * JUMP_TICKS
        assert planted.latency_markers[int(ts[i - 1])] == d
    gaps = np.diff([t for t, _ in marks])
    assert int(gaps.min()) >= MARK_EVERY_MINUTES * 60_000_000_000


def test_planted_bars_keep_high_and_low_around_open_and_close(planted: PlantedBars) -> None:
    f = planted.frame
    assert f.select((pl.col("high") >= pl.max_horizontal("open", "close")).all()).item()
    assert f.select((pl.col("low") <= pl.min_horizontal("open", "close")).all()).item()


def test_planting_is_seeded() -> None:
    a, b = planted_synthetic(seed=5), planted_synthetic(seed=5)
    assert dict(a.leak_markers) == dict(b.leak_markers) and a.frame.equals(b.frame)


# ------------------------------------------------------------- the honest engine ----
def test_the_honest_engine_shows_no_edge_on_the_planted_future(honest_report: CanaryReport) -> None:
    leak = honest_report.leak
    assert leak.trades >= MIN_TRADES and leak.unmatched_trades == 0
    assert shows_no_edge(leak, JUMP_TICKS)
    assert not honest_report.leak_detected


def test_the_honest_engine_captures_the_one_bar_early_latency_control(honest_report: CanaryReport) -> None:
    assert edge_detected(honest_report.latency, JUMP_TICKS)
    assert honest_report.leak_free


@pytest.mark.parametrize("seed", SEEDS)
def test_the_honest_engine_is_leak_free_across_seeds(seed: int) -> None:
    assert run_canaries(honest_engine, planted_synthetic(seed=seed)).leak_free


def test_the_mutant_harness_without_a_bug_reproduces_simulate_trade_for_trade(planted: PlantedBars) -> None:
    # Arrange: the harness the mutants share must be the honest engine plus only the named bug
    inputs = zero_cost_inputs("MNQ", planted.frame)
    for markers in (planted.leak_markers, planted.latency_markers):
        strategy = canary_strategy(markers)
        # Act
        ours = kernel_trades(strategy, planted.frame, inputs)
        real = honest_engine(strategy, planted.frame, inputs).select(TRADE_KEYS)
        # Assert
        assert ours.select(TRADE_KEYS).equals(real)


# ---------------------------------------------------------------- the mutants ----
@pytest.mark.parametrize("name", sorted(ENGINE_MUTANTS))
def test_each_leaky_engine_mutant_is_detected_by_the_canary(planted: PlantedBars, name: str) -> None:
    # Arrange
    engine: Callable[..., pl.DataFrame] = ENGINE_MUTANTS[name]
    # Act
    report = run_canaries(engine, planted)
    # Assert
    assert report.leak_detected, report.leak
    assert not report.leak_free
    assert report.leak.mean_captured_ticks >= JUMP_TICKS / 2 and report.leak.z >= 5


@pytest.mark.parametrize("name", sorted(ENGINE_MUTANTS))
@pytest.mark.parametrize("seed", SEEDS)
def test_engine_mutants_are_detected_across_seeds(name: str, seed: int) -> None:
    assert run_canaries(ENGINE_MUTANTS[name], planted_synthetic(seed=seed)).leak_detected


def test_engine_mutants_slip_past_the_truncation_check_so_the_canary_is_needed() -> None:
    # The strategy handed to a leaky engine is honest: the truncation check alone cannot see the engine bug.
    bars = synthetic_bars("MNQ", seed=2)
    markers = {int(t): 1 for t in bars["ts_event"][::20]}
    truncation_check(canary_strategy(markers).generate, bars, {}, seed=2)


def test_the_honest_kernel_harness_passes_the_canary(planted: PlantedBars) -> None:
    assert run_canaries(honest_kernel_engine, planted).leak_free


# ------------------------------------------------------------------ scoring ----
def test_score_canary_matches_fills_at_the_marker_or_one_minute_later() -> None:
    minute = 60_000_000_000
    trades = pl.DataFrame(
        {"side": [1, -1, 1], "entry_ts": [10 * minute, 31 * minute, 99 * minute],
         "entry_price": [100, 200, 5], "exit_price": [116, 190, 0]}
    )  # fmt: skip
    score = score_canary(trades, {10 * minute: 1, 30 * minute: -1})
    assert isinstance(score, CanaryScore)
    assert (score.trades, score.unmatched_trades, score.hit_n) == (2, 1, 2)
    assert score.mean_captured_ticks == 13.0 and score.hit_rate == 1.0


def test_criteria_need_enough_trades() -> None:
    few = CanaryScore(MIN_TRADES - 1, 0, 16.0, 0.1, 160.0, 1.0, MIN_TRADES - 1, 0.05)
    assert not edge_detected(few, JUMP_TICKS) and not shows_no_edge(few, JUMP_TICKS)
