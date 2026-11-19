"""Hand-computed known answers for every fill rule of ADR-001 section 8 (the kernel alone).

Defaults (engine_cases): 50 cents a tick, commission 10 cents per fill, slippage 5 (buy) and 7 (sell) cents.
"""

from __future__ import annotations

import numpy as np
import pytest
from engine_cases import kernel_inputs, run

from pfb.engine import kernel as K
from pfb.engine.kernel import KernelInputError, run_kernel

FLAT = (100, 100, 100, 100)


def _trade(out: K.KernelOutput, row: int = 0) -> dict[str, int]:
    names = (
        "entry_idx",
        "exit_idx",
        "side",
        "entry_px",
        "exit_px",
        "gross",
        "comm",
        "slip",
        "net",
        "mae",
        "mfe",
        "reason",
        "after_lock",  # T_AFTER_LOCK
    )
    return dict(zip(names, out.trades[row].tolist(), strict=True))


def test_market_entry_fills_at_next_open_and_session_end_exits_at_close() -> None:
    # Arrange: long decided at bar 0, fills at open[1] = 101, exits when the session's last bar closes (104)
    rows = [(100, 101, 99, 100), (101, 103, 99, 102), (102, 104, 100, 103), (103, 105, 101, 104)]
    # Act
    t = _trade(run(rows, long=[0]))
    # Assert: gross 3 ticks = 150; commission 2 x 10; slippage buy 5 + sell 7
    assert (t["entry_idx"], t["entry_px"], t["exit_idx"], t["exit_px"]) == (1, 101, 3, 104)
    assert (t["gross"], t["comm"], t["slip"], t["net"]) == (150, 20, 12, 118)
    assert t["reason"] == K.SESSION_END


def test_long_stop_gapped_through_fills_at_the_open() -> None:
    # Arrange: entry 100, stop 97; bar 2 opens at 95, below the stop
    rows = [(100, 101, 99, 100), (100, 102, 99, 101), (95, 96, 94, 95), (95, 95, 95, 95)]
    # Act
    t = _trade(run(rows, long=[0], stops={0: 3}))
    # Assert: fill at min(97, 95) = 95, a market fill with slippage: -5 ticks = -250 - 20 - 12
    assert (t["exit_idx"], t["exit_px"], t["reason"]) == (2, 95, K.STOP)
    assert (t["gross"], t["net"]) == (-250, -282)


def test_short_stop_gapped_through_fills_at_the_open() -> None:
    # Arrange: short entry 100, stop 103; bar 2 opens at 106
    rows = [(100, 101, 99, 100), (100, 101, 99, 100), (106, 107, 105, 106), (106, 106, 106, 106)]
    # Act
    t = _trade(run(rows, short=[0], stops={0: 3}))
    # Assert: max(103, 106) = 106; -6 ticks = -300; slippage sell 7 + buy 5
    assert (t["side"], t["exit_px"], t["reason"]) == (-1, 106, K.STOP)
    assert (t["gross"], t["slip"], t["net"]) == (-300, 12, -332)


def test_stop_touched_without_gap_fills_at_the_stop_price() -> None:
    rows = [(100, 101, 99, 100), (100, 101, 99, 100), (99, 99, 97, 98), FLAT]
    t = _trade(run(rows, long=[0], stops={0: 3}))
    assert (t["exit_px"], t["reason"], t["net"]) == (97, K.STOP, -150 - 20 - 12)


def test_stop_and_target_touched_in_one_bar_the_stop_fills() -> None:
    # Arrange: stop 98, target 103; bar 2 trades 97..105
    rows = [(100, 101, 99, 100), (100, 101, 99, 100), (100, 105, 97, 101), FLAT]
    # Act
    t = _trade(run(rows, long=[0], stops={0: 2}, targets={0: 3}))
    # Assert
    assert (t["exit_idx"], t["exit_px"], t["reason"], t["net"]) == (2, 98, K.STOP, -100 - 20 - 12)


def test_target_touched_but_not_traded_through_does_not_fill() -> None:
    # Arrange: target 103; bar 2 high is exactly 103, bar 3 high 104 trades one tick through
    rows = [(100, 101, 99, 100), (100, 101, 99, 100), (101, 103, 100, 102), (102, 104, 101, 103), FLAT]
    # Act
    t = _trade(run(rows, long=[0], targets={0: 3}))
    # Assert: fills on bar 3 at exactly 103, commission only (entry slippage 5 remains)
    assert (t["exit_idx"], t["exit_px"], t["reason"]) == (3, 103, K.TARGET)
    assert (t["gross"], t["comm"], t["slip"], t["net"]) == (150, 20, 5, 125)


def test_target_gapped_through_fills_at_the_target_never_better() -> None:
    rows = [(100, 101, 99, 100), (100, 101, 99, 100), (110, 112, 109, 111), FLAT]
    t = _trade(run(rows, long=[0], targets={0: 3}))
    assert (t["exit_px"], t["reason"], t["net"]) == (103, K.TARGET, 125)


def test_short_target_needs_one_tick_through_below() -> None:
    rows = [(100, 101, 99, 100), (100, 101, 99, 100), (99, 100, 97, 98), (98, 99, 96, 97), FLAT]
    t = _trade(run(rows, short=[0], targets={0: 3}))
    assert (t["exit_idx"], t["exit_px"], t["reason"], t["slip"]) == (3, 97, K.TARGET, 7)


def test_exit_signal_fills_at_the_next_open() -> None:
    rows = [(100, 101, 99, 100), (101, 102, 100, 101), (102, 103, 101, 102), (104, 105, 103, 104), FLAT]
    t = _trade(run(rows, long=[0], exits=[2]))
    assert (t["exit_idx"], t["exit_px"], t["reason"], t["net"]) == (3, 104, K.SIGNAL, 150 - 20 - 12)


def test_flat_by_exits_at_the_open_of_the_must_be_flat_bar() -> None:
    rows = [(100, 101, 99, 100), (101, 102, 100, 101), (102, 103, 101, 102), (103, 104, 102, 103), FLAT]
    t = _trade(run(rows, long=[0], flat_by=[3], no_fill=[3]))
    assert (t["exit_idx"], t["exit_px"], t["reason"]) == (3, 103, K.FLAT_BY)


def test_pending_entry_that_cannot_fill_is_dropped_and_counted() -> None:
    # Arrange: decision at bar 0, bar 1 is not entry_fill_ok (its minute or the next is must-be-flat)
    rows = [(100, 101, 99, 100)] * 4
    # Act
    out = run(rows, long=[0], no_fill=[1])
    # Assert: not carried to bar 2
    assert len(out.trades) == 0
    assert out.counters["entries_dropped"] == 1


def test_session_end_ignores_the_bars_signals_and_nothing_crosses() -> None:
    # Arrange: bar 1 ends a session and carries an entry signal; bar 2 starts the next session
    rows = [(100, 101, 99, 100)] * 4
    # Act
    out = run(rows, long=[1], session_end=[1, 3])
    # Assert
    assert len(out.trades) == 0
    assert out.counters["entry_signals_at_session_end"] == 1  # the split counter
    assert out.counters["exit_signals_at_session_end"] == 0


def test_e6_an_exit_flag_while_flat_on_a_session_end_bar_is_not_counted() -> None:
    # Arrange: a dense exit flag (the reference's) on bar 1, which ends a session; no position
    out = run([(100, 101, 99, 100)] * 4, exits=[1], session_end=[1, 3])
    # Assert
    assert out.counters["entry_signals_at_session_end"] == 0
    assert out.counters["exit_signals_at_session_end"] == 0


def test_e6_entry_and_exit_flags_on_a_session_end_bar_in_position_are_counted_apart() -> None:
    # Arrange: long decided at bar 0 (fills at bar 1); bar 1 ends the session and carries both flags
    out = run([(100, 101, 99, 100)] * 4, long=[0, 1], exits=[1], session_end=[1, 3])
    # Assert: closed at bar 1's close; the entry flag is not counted as "in position"
    assert len(out.trades) == 1
    assert out.counters["entry_signals_at_session_end"] == 1
    assert out.counters["exit_signals_at_session_end"] == 1
    assert out.counters["entry_signals_in_position"] == 0


def test_conflicting_signals_are_ignored_and_counted() -> None:
    out = run([(100, 101, 99, 100)] * 3, long=[0], short=[0])
    assert len(out.trades) == 0
    assert out.counters["conflicting_signals"] == 1


def test_entry_signal_in_a_position_is_ignored_no_reversal() -> None:
    rows = [(100, 101, 99, 100)] * 5
    out = run(rows, long=[0], short=[2])
    assert len(out.trades) == 1
    assert _trade(out)["side"] == 1
    assert out.counters["entry_signals_in_position"] == 1


def test_entry_allowed_false_blocks_the_entry_at_its_decision_bar() -> None:
    out = run([(100, 101, 99, 100)] * 3, long=[0], blocked=[0])
    assert len(out.trades) == 0
    assert out.counters["entries_blocked"] == 1


def test_a_market_fill_on_an_uncalibrated_minute_is_counted() -> None:
    out = run([(100, 101, 99, 100)] * 3, long=[0], uncalibrated=[1])
    assert out.counters["uncalibrated_fills"] == 1


def test_fractional_cent_gross_floors_toward_minus_infinity() -> None:
    # Arrange: tick value 1562.5 cents (156250 x 1e-4 USD, as ZN); one tick won, then one tick lost
    win = [(100, 100, 100, 100), (100, 100, 100, 100), (101, 101, 101, 101)]
    loss = [(100, 100, 100, 100), (100, 100, 100, 100), (99, 99, 99, 99)]
    # Act
    t_win = _trade(run(win, long=[0], tv_cc=156_250))
    t_loss = _trade(run(loss, long=[0], tv_cc=156_250))
    # Assert: floor(1562.5) = 1562 and floor(-1562.5) = -1563
    assert t_win["gross"] == 1562
    assert t_loss["gross"] == -1563


def test_open_mark_of_a_fractional_cent_position_floors_too() -> None:
    rows = [(100, 100, 100, 100), (100, 100, 100, 100), (99, 99, 99, 99), (99, 99, 99, 99)]
    out = run(rows, long=[0], tv_cc=156_250)
    # bar 2: realized -15 (entry costs) + floor(-1562.5) = -1578
    assert out.equity[2, K.E_CLOSE] == -15 - 1563


def test_whole_quantity_scales_gross_and_takes_costs_as_given() -> None:
    rows = [(100, 100, 100, 100), (100, 100, 100, 100), (102, 102, 102, 102)]
    t = _trade(run(rows, long=[0], contracts=3, commission=30))
    assert (t["gross"], t["comm"]) == (300, 60)


def test_last_eq_close_equals_the_sum_of_net_cents() -> None:
    # Arrange: three round trips with mixed outcomes
    rows = [
        (100, 102, 98, 101),
        (101, 104, 99, 103),
        (103, 104, 96, 97),
        (97, 99, 95, 98),
        (98, 103, 97, 102),
        (102, 103, 100, 101),
        (101, 102, 99, 100),
        (100, 101, 99, 100),
    ]
    # Act
    out = run(rows, long=[0, 3], short=[5], exits=[2, 4], stops={3: 2})
    # Assert
    assert len(out.trades) == 3
    assert int(out.equity[-1, K.E_CLOSE]) == int(out.trades[:, K.T_NET].sum())
    assert int(out.equity[-1, K.E_POS]) == 0


def test_a_float_array_is_refused_before_the_kernel() -> None:
    inputs = kernel_inputs([(100, 101, 99, 100)] * 2)
    bad = K.KernelInputs(**{**inputs.__dict__, "open": inputs.open.astype(np.float64)})
    with pytest.raises(KernelInputError):
        run_kernel(bad)


def test_a_float_scalar_is_refused_before_the_kernel() -> None:
    inputs = kernel_inputs([(100, 101, 99, 100)] * 2)
    bad = K.KernelInputs(**{**inputs.__dict__, "commission_cents": 10.0})
    with pytest.raises(KernelInputError):
        run_kernel(bad)
