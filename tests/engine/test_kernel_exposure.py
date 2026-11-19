"""Exposure inside a bar (ADR-001 A6, A14 c): eq_low, eq_high, mae_ticks and mfe_ticks per exit reason.

Hand-computed with 50 cents a tick, commission 10 per fill, slippage 5 (buy) and 7 (sell). A long enters at
open[1] = 100 after a decision at bar 0, so its entry costs are 15 and equity right after the fill is -15.
mae_ticks and mfe_ticks are non-negative magnitudes.
"""

from __future__ import annotations

from engine_cases import run

from pfb.engine import kernel as K

DECISION = (100, 100, 100, 100)
WIDE = (100, 103, 97, 101)  # the entry bar: opens at the fill price, trades 97..103, closes 101


def _bounds(out: K.KernelOutput, bar: int) -> tuple[int, int, int]:
    return int(out.equity[bar, K.E_LOW]), int(out.equity[bar, K.E_HIGH]), int(out.equity[bar, K.E_CLOSE])


def _excursions(out: K.KernelOutput) -> tuple[int, int, int]:
    trade = out.trades[0]
    return int(trade[K.T_MAE]), int(trade[K.T_MFE]), int(trade[K.T_REASON])


def test_stop_in_the_entry_bar_uses_the_stop_fill_and_the_favourable_extreme() -> None:
    # Arrange: stop 2 ticks -> 98; the bar's low 97 touches it, its high 103 is the favourable extreme
    out = run([DECISION, WIDE, DECISION], long=[0], stops={0: 2})
    # Act
    low, high, close = _bounds(out, 1)
    # Assert: points 0 (open, flat), -15 (fill), -15+150 (high 103), -15-100 (stop 98), -132 (after the exit)
    assert _excursions(out) == (2, 3, K.STOP)  # the low 97 does not count: mae 2, not 3
    assert (low, high, close) == (-132, 135, -132)


def test_target_in_the_entry_bar_uses_the_target_and_the_adverse_extreme() -> None:
    # Arrange: target 2 ticks -> 102; high 103 trades one tick through it
    out = run([DECISION, WIDE, DECISION], long=[0], targets={0: 2})
    # Act
    low, high, close = _bounds(out, 1)
    # Assert: points 0, -15, -165 (low 97), +85 (target 102), 75 (after the exit: +100 - 20 - 5)
    assert _excursions(out) == (3, 2, K.TARGET)  # the high 103 does not count: mfe 2, not 3
    assert (low, high, close) == (-165, 85, 75)


def test_session_end_in_the_entry_bar_uses_the_whole_range() -> None:
    # Arrange: the entry bar is the session's last bar
    out = run([DECISION, WIDE], long=[0])
    # Act
    low, high, close = _bounds(out, 1)
    # Assert: points 0, -15, -165 (97), +135 (103), close exit at 101: +50 - 20 - 12 = 18
    assert _excursions(out) == (3, 3, K.SESSION_END)
    assert (low, high, close) == (-165, 135, 18)


def test_a_bar_the_position_stays_open_through_exposes_the_whole_range() -> None:
    out = run([DECISION, WIDE, (101, 104, 99, 102), (102, 102, 102, 102)], long=[0])
    # bar 2: open 101 (+35), low 99 (-65), high 104 (+185), close 102 (+85)
    assert _bounds(out, 2) == (-65, 185, 85)
    assert _excursions(out) == (3, 4, K.SESSION_END)


def test_signal_exit_at_the_open_contributes_the_open_only() -> None:
    # Arrange: exit decision at bar 1; bar 2 opens 104 and then trades 90..110
    out = run([DECISION, WIDE, (104, 110, 90, 100), DECISION], long=[0], exits=[1])
    # Act
    low, high, close = _bounds(out, 2)
    # Assert: points +185 (open 104) and 168 after the exit (+200 - 20 - 12); 110 and 90 do not count
    assert (low, high, close) == (168, 185, 168)
    assert _excursions(out) == (3, 4, K.SIGNAL)


def test_flat_by_exit_at_the_open_contributes_the_open_only() -> None:
    out = run([DECISION, WIDE, (104, 110, 90, 100), DECISION], long=[0], flat_by=[2], no_fill=[2])
    assert _bounds(out, 2) == (168, 185, 168)
    assert _excursions(out) == (3, 4, K.FLAT_BY)


def test_carried_stop_bar_counts_the_open_the_favourable_extreme_and_the_fill() -> None:
    # Arrange: stop 98 set at entry; bar 2 opens 99, trades up to 101, then down through the stop to 95
    out = run([DECISION, (100, 100, 100, 100), (99, 101, 95, 96), DECISION], long=[0], stops={0: 2})
    # Act
    low, high, close = _bounds(out, 2)
    # Assert: open 99 (-65), high 101 (+35), stop fill 98 (-115), after exit -100 - 20 - 12 = -132
    assert (low, high, close) == (-132, 35, -132)
    assert _excursions(out) == (2, 1, K.STOP)


def test_short_target_exposure_is_mirrored() -> None:
    # Arrange: short at 100 (entry costs 10 + 7), target 2 -> 98; bar 1 trades 96..103
    out = run([DECISION, (100, 103, 96, 99), DECISION], short=[0], targets={0: 2})
    # Act
    low, high, close = _bounds(out, 1)
    # Assert: -17 (fill), adverse 103 -> -167, target 98 -> +83, after exit +100 - 20 - 7 = 73
    assert (low, high, close) == (-167, 83, 73)
    assert _excursions(out) == (3, 2, K.TARGET)


def test_flat_bars_have_equity_equal_to_realized_everywhere() -> None:
    out = run([DECISION, WIDE, DECISION, (90, 120, 80, 95)], long=[0], exits=[1])
    assert _bounds(out, 3) == (_bounds(out, 2)[2],) * 3
