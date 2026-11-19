"""A must-be-flat interval that holds no bar is enforced. Bar i before the
interval's start with bar i + 1 at or after its end is a session end (exit at close[i], its signals ignored),
counted in session_ends_at_flat_gap. Hand answers for three cases.

Bars 09:00..09:09 CT, minutes 4 and 5 must be flat. Rows: minutes 0-3 (100, 101, 99, 100), 4-5 flat at 100,
6-9 (90, 91, 89, 90). Tick 50 cents, commission 10 cents a fill, slippage 5 cents a buy and 7 a sell."""

from __future__ import annotations

from datetime import date

import numpy as np
from engine_cases import NS_MIN, frame_of, market_inputs, minute_bars, scripted

from pfb.engine.run import flat_gap_ends, simulate

DAY = date(2025, 6, 3)
ROWS = [(100, 101, 99, 100)] * 4 + [(100, 100, 100, 100)] * 2 + [(90, 91, 89, 90)] * 4
BARS = minute_bars(DAY, 9, 0, ROWS)
T0 = BARS[0].ts
FLAT = [(T0 + 4 * NS_MIN, T0 + 6 * NS_MIN)]
WITHOUT_4_5 = [0, 1, 2, 3, 6, 7, 8, 9]


def _sim(keep: list[int], long: list[int]):  # type: ignore[no-untyped-def]
    frame = frame_of([BARS[i] for i in keep])
    inputs = market_inputs(frame, flat=FLAT)
    return simulate(scripted(long=long), frame, {}, inputs, seed=0, price_limit_rule=False)


def _minute(ts: int) -> int:
    return (ts - T0) // NS_MIN


def test_case_1_all_minutes_present_exits_flat_by_at_minute_4_as_before() -> None:
    # Hand: long decided at 09:00, filled at open[09:01] = 100; 09:04 lies in the interval -> FLAT_BY at its
    # open 100; gross 0, commission 2 x 10, slippage 5 + 7 -> net -32. No gap.
    out = _sim(list(range(10)), long=[0])
    t = out.trades.row(0, named=True)
    got = (t["exit_reason"], _minute(t["exit_ts"]), t["exit_price"], t["net_cents"])
    assert got == ("FLAT_BY", 4, 100, -32)
    assert out.counters["session_ends_at_flat_gap"] == 0


def test_case_2_minutes_4_and_5_absent_end_the_session_at_minute_3() -> None:
    # Hand: same entry; 09:03 is before the interval and the next bar 09:06 at its end -> SESSION_END at
    # close[09:03] = 100, stamped 09:04; net -32 (was SESSION_END at 09:10 at 90, net -532).
    out = _sim(WITHOUT_4_5, long=[0])
    t = out.trades.row(0, named=True)
    assert (t["exit_reason"], _minute(t["exit_ts"]), t["exit_price"], t["net_cents"]) == (
        "SESSION_END",
        4,
        100,
        -32,
    )
    assert (t["exit_bar"], out.trades.height) == (3, 1)
    assert out.counters["session_ends_at_flat_gap"] == 1


def test_case_3_an_entry_decided_just_before_the_hole_is_dropped_and_counted() -> None:
    # Hand: the entry decided at 09:03 sits on that session-end bar: ignored, counted (was: filled at 09:06)
    out = _sim(WITHOUT_4_5, long=[3])
    assert out.trades.height == 0
    assert out.counters["entry_signals_at_session_end"] == 1
    assert out.counters["session_ends_at_flat_gap"] == 1
    assert out.counters["signals_ignored"] == 1


def test_an_entry_decided_two_bars_before_the_hole_cannot_fill_into_it() -> None:
    # Hand: decided at 09:02, would fill at open[09:03]; the next minute 09:04 is must-be-flat -> dropped
    out = _sim(WITHOUT_4_5, long=[2])
    assert out.trades.height == 0
    assert out.counters["entries_dropped"] == 1


def test_flat_gap_ends_marks_only_an_interval_without_a_bar() -> None:
    # Arrange: minutes as ns; intervals [4, 6) and [20, 30)
    ts = np.array([0, 1, 3, 6, 7, 19, 25, 31], dtype=np.int64) * NS_MIN
    starts = np.array([4, 20], dtype=np.int64) * NS_MIN
    ends = np.array([6, 30], dtype=np.int64) * NS_MIN
    # Act
    gap = flat_gap_ends(ts, starts, ends)
    # Assert: 3 -> 6 jumps the empty [4, 6); 19 -> 25 lands inside [20, 30) (flat_by handles it)
    assert gap.tolist() == [False, False, True, False, False, False, False, False]
    assert not flat_gap_ends(ts, starts[:0], ends[:0]).any()
    assert not flat_gap_ends(ts[:1], starts, ends).any()


def test_a_hole_at_a_trade_date_change_is_not_counted_twice() -> None:
    # Arrange: the interval lies between two trade dates; the day change already ends the session
    d2 = date(2025, 6, 4)
    bars = minute_bars(DAY, 9, 0, ROWS[:4]) + minute_bars(d2, 9, 0, ROWS[:4])
    gap_start = BARS[3].ts + NS_MIN
    frame = frame_of(bars)
    inputs = market_inputs(frame, flat=[(gap_start, gap_start + NS_MIN)])
    # Act
    out = simulate(scripted(long=[0]), frame, {}, inputs, seed=0, price_limit_rule=False)
    # Assert
    assert out.trades["exit_reason"].to_list() == ["SESSION_END"]
    assert out.counters["session_ends_at_flat_gap"] == 0
