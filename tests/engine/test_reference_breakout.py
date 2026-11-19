"""The reference breakout (ADR-001 section 10) on hand-built bars, alone and through simulate."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
from engine_cases import frame_of, market_inputs, minute_bars

from pfb.engine.lookahead import truncation_check
from pfb.engine.run import simulate
from pfb.engine.strategy import load_strategy, make_bars, resolve_params

REFERENCE = Path(__file__).resolve().parents[2] / "strategies" / "reference_breakout.py"
DAY, NEXT = date(2025, 4, 8), date(2025, 4, 9)
RANGE = [(105, 110, 100, 105)] * 15  # 08:30..08:44 CT: high 110, low 100


def _day(after: list[tuple[int, int, int, int]], day: date = DAY) -> list:  # type: ignore[type-arg]
    evening = minute_bars(day, 7, 0, [(105, 106, 104, 105)] * 3)  # earlier bars of the same trade date
    return evening + minute_bars(day, 8, 30, RANGE) + minute_bars(day, 8, 45, after)


def _signals(bars_list: list) -> dict[str, np.ndarray]:  # type: ignore[type-arg]
    strategy = load_strategy(REFERENCE)
    bars = make_bars(frame_of(bars_list))
    return strategy.generate(bars, resolve_params(strategy.spec, None))  # type: ignore[return-value]


def test_spec_is_the_untuned_template_on_mnq() -> None:
    spec = load_strategy(REFERENCE).spec
    assert (spec.root, spec.session, spec.contracts, spec.bar_minutes) == ("MNQ", "globex", 1, 1)
    assert all(len(p.grid) == 1 for p in spec.params.values())
    assert resolve_params(spec, None) == {
        "day_open_minute": 510,
        "day_close_minute": 900,
        "range_minutes": 15,
        "buffer_ticks": 4,
        "hold_minutes": 75,
        "exit_before_close_minutes": 2,
    }


def test_first_close_four_ticks_above_the_range_is_the_long_entry() -> None:
    # Arrange: 08:45 closes at 113 (< 110 + 4), 08:46 at 114, 08:47 at 120 (a second breakout)
    after = [(110, 113, 109, 113), (113, 114, 112, 114), (114, 120, 113, 120)] + [(118, 119, 117, 118)] * 90
    # Act
    sig = _signals(_day(after))
    # Assert: index 3 + 15 + 1 = 19 is the 08:46 bar
    assert np.flatnonzero(sig["entry_long"]).tolist() == [19]
    assert not sig["entry_short"].any()


def test_exit_decision_comes_75_minutes_after_the_entry_decision() -> None:
    after = [(110, 113, 109, 113), (113, 114, 112, 114)] + [(118, 119, 117, 118)] * 90
    sig = _signals(_day(after))
    # entry decision bar opens 08:46; the first exit decision bar opens 10:01 (index 19 + 75)
    assert int(np.flatnonzero(sig["exit"])[0]) == 19 + 75
    assert not sig["exit"][:94].any()


def test_short_entry_and_the_exit_two_minutes_before_the_close() -> None:
    # Arrange: no breakout at 08:45; the 14:50 bar closes at 96 (= 100 - 4); bars run to 15:04
    later = [(100, 101, 95, 96)] + [(97, 98, 96, 97)] * 14
    bars = _day([(105, 106, 104, 105)]) + minute_bars(DAY, 14, 50, later)
    # Act
    sig = _signals(bars)
    # Assert: 14:50 is index 19; 14:58 (C - 2) is index 27, long before the 75-minute hold ends
    assert np.flatnonzero(sig["entry_short"]).tolist() == [19]
    assert int(np.flatnonzero(sig["exit"])[0]) == 27


def test_one_entry_per_trade_date_and_a_fresh_range_the_next_day() -> None:
    first = _day([(110, 115, 109, 115)] + [(99, 100, 90, 95)] * 5)
    second = _day([(110, 115, 109, 115)], NEXT)
    sig = _signals(first + second)
    assert np.flatnonzero(sig["entry_long"]).tolist() == [18, len(first) + 18]
    assert not sig["entry_short"].any()


def test_no_range_bar_means_no_entry() -> None:
    bars = minute_bars(DAY, 9, 0, [(200, 210, 190, 205)] * 20)
    sig = _signals(bars)
    assert not (sig["entry_long"].any() or sig["entry_short"].any() or sig["exit"].any())


def test_reference_is_truncation_invariant_on_its_test_bars() -> None:
    strategy = load_strategy(REFERENCE)
    bars = make_bars(frame_of(_day([(110, 115, 109, 115)] + [(118, 119, 117, 118)] * 90)))
    truncation_check(strategy.generate, bars, resolve_params(strategy.spec, None), seed=3)


def test_through_the_engine_entry_fills_at_0847_and_exit_at_1002() -> None:
    # Arrange: entry decision 08:46 -> fill 08:47 open; exit decision 10:01 -> fill 10:02 open
    after = [(110, 113, 109, 113), (113, 114, 112, 114)] + [(118 + i % 3, 121, 117, 118) for i in range(90)]
    frame = frame_of(_day(after))
    strategy = load_strategy(REFERENCE)
    # Act
    out = simulate(strategy, frame, resolve_params(strategy.spec, None), market_inputs(frame), seed=1,
                   price_limit_rule=False)  # fmt: skip
    # Assert
    details = out.trades.to_dicts()
    assert [(d["entry_bar"], d["exit_bar"], d["exit_reason"]) for d in details] == [(20, 95, "SIGNAL")]
    minutes = frame["ct_minute"].to_list()
    assert (minutes[20], minutes[95]) == (8 * 60 + 47, 10 * 60 + 2)
