"""simulate on hand-built frames: session_end causes (A14 d), removed bars, flat-by, costs, frames."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest
from engine_cases import COMM, NS_MIN, Bar, ct_ns, frame_of, market_inputs, minute_bars, scripted, strategy_of

from pfb.engine.lookahead import LookAheadDetected
from pfb.engine.run import DAILY_COLUMNS, EQUITY_COLUMNS, EngineInputError, simulate
from pfb.engine.strategy import SignalError
from pfb.metrics.core import TRADE_COLUMNS, summarize

D1, D2 = date(2025, 4, 7), date(2025, 4, 8)
UP = [(100 + i, 101 + i, 99 + i, 100 + i) for i in range(8)]


def _sim(bars: list[Bar], strategy=None, **inputs_kw):  # type: ignore[no-untyped-def]
    frame = frame_of(bars)
    return simulate(strategy or scripted(long=[0]), frame, {}, market_inputs(frame, **inputs_kw), seed=7,
                    price_limit_rule=False)  # fmt: skip


def _reasons(out) -> list[str]:  # type: ignore[no-untyped-def]
    return out.trades["exit_reason"].to_list()


def test_trade_date_change_forces_session_end_at_the_last_bar_of_the_day() -> None:
    # Arrange: day 1 has 4 bars, day 2 has 4 bars; long decided at bar 0, never exited by signal
    bars = minute_bars(D1, 10, 0, UP[:4]) + minute_bars(D2, 10, 0, UP[4:])
    # Act
    out = _sim(bars, scripted(long=[0, 3]))
    # Assert: exit at day 1's last close (bar 3, price 103); the entry signal on bar 3 is ignored
    assert _reasons(out) == ["SESSION_END"]
    assert out.trades["exit_price"].to_list() == [103]
    assert out.counters["entry_signals_at_session_end"] == 1  # the split counter
    assert out.counters["exit_signals_at_session_end"] == 0


def test_roll_splice_forces_session_end_before_the_instrument_change() -> None:
    bars = minute_bars(D1, 10, 0, UP[:3]) + minute_bars(D1, 10, 3, UP[3:6], instrument_id=2)
    out = _sim(bars)
    assert _reasons(out) == ["SESSION_END"]
    assert out.trades["exit_bar"].to_list() == [2]


def test_era_change_forces_session_end_like_a_roll() -> None:
    # A14 (d): an era change mid trade date forces a flat position
    bars = minute_bars(D1, 10, 0, UP[:3], era="step2") + minute_bars(D1, 10, 3, UP[3:6], era="research")
    out = _sim(bars)
    assert _reasons(out) == ["SESSION_END"]
    assert out.trades["exit_bar"].to_list() == [2]


def test_removed_closure_bar_ends_the_session_before_it_and_is_counted() -> None:
    # Arrange: bar 3 of 6 is a scheduled-closure bar
    bars = minute_bars(D1, 10, 0, UP[:3]) + minute_bars(D1, 10, 3, UP[3:4], closure=True)
    bars += minute_bars(D1, 10, 4, UP[4:6])
    # Act
    out = _sim(bars)
    # Assert: 5 usable bars; the position closes at bar 2's close (102)
    assert out.equity_minute.height == 5
    assert out.counters["bars_dropped_closure"] == 1
    assert out.trades["exit_price"].to_list() == [102]


def test_unsourced_days_are_dropped_and_counted() -> None:
    bars = minute_bars(D1, 10, 0, UP[:4]) + minute_bars(D2, 10, 0, UP[4:])
    out = _sim(bars, unsourced=[D2])
    assert out.equity_minute.height == 4
    assert (out.counters["bars_dropped_unsourced"], out.counters["days_dropped_unsourced"]) == (4, 1)


def test_bars_outside_the_session_window_are_dropped_and_counted() -> None:
    bars = minute_bars(D1, 10, 0, UP[:6])
    window = (ct_ns(D1, 10, 1), ct_ns(D1, 10, 5))  # keeps 10:01..10:04
    out = _sim(bars, window=window)
    assert out.equity_minute.height == 4
    assert out.counters["bars_dropped_outside_session"] == 2


def test_must_be_flat_interval_exits_at_the_open_of_its_first_bar() -> None:
    # Arrange: a flat interval from 10:04 CT; entry fills at 10:01
    bars = minute_bars(D1, 10, 0, UP[:6])
    out = _sim(bars, flat=[(ct_ns(D1, 10, 4), ct_ns(D1, 11, 0))])
    # Assert: FLAT_BY at open[4] = 104
    assert _reasons(out) == ["FLAT_BY"]
    assert out.trades["exit_price"].to_list() == [104]


@pytest.mark.parametrize(("flat_from_minute", "fills"), [(1, False), (2, False), (3, True)])
def test_entry_fill_ok_probes_the_fill_minute_and_the_next(flat_from_minute: int, fills: bool) -> None:
    # Arrange: decision at 10:00, fill bar 10:01; must-be-flat from 10:01, 10:02 or 10:03
    bars = minute_bars(D1, 10, 0, UP[:6])
    flat = [(ct_ns(D1, 10, flat_from_minute), ct_ns(D1, 11, 0))]
    # Act
    out = _sim(bars, flat=flat)
    # Assert
    assert (out.trades.height == 1) is fills
    assert out.counters["entries_dropped"] == (0 if fills else 1)


def test_grain_pause_flattens_and_blocks_entries_around_it() -> None:
    # Arrange: bars 07:40..07:47 CT and the grain pause 07:43-08:30 CT
    bars = minute_bars(D1, 7, 40, UP[:8])
    pause = [(ct_ns(D1, 7, 43), ct_ns(D1, 8, 30))]
    # Act: an entry decided at 07:40 fills at 07:41 and must exit at 07:43; one decided at 07:41 cannot fill
    held = _sim(bars, scripted(long=[0]), flat=pause)
    blocked = _sim(bars, scripted(long=[1]), flat=pause)
    # Assert
    assert _reasons(held) == ["FLAT_BY"]
    assert held.trades["exit_bar"].to_list() == [3]
    assert blocked.trades.height == 0
    assert blocked.counters["entries_dropped"] == 1


def test_slippage_is_looked_up_per_ct_minute_and_side() -> None:
    # Arrange: slippage of minute m is m cents on a buy and 2000 + m on a sell
    minutes = np.arange(1440, dtype=np.int64)
    table = np.stack([minutes, 2000 + minutes], axis=1)
    bars = minute_bars(D1, 10, 0, UP[:6])
    # Act: long fills at 10:01 (minute 601, buy) and exits by signal at 10:04 (minute 604, sell)
    out = _sim(bars, scripted(long=[0], exits=[3]), slippage=table)
    # Assert
    assert out.trades["slippage_cents"].to_list() == [601 + 2604]
    assert out.trades["commission_cents"].to_list() == [2 * COMM]


def test_frames_have_the_ruled_columns_and_feed_summarize() -> None:
    # Arrange: two days, one trade per day
    bars = minute_bars(D1, 10, 0, UP[:4]) + minute_bars(D2, 10, 0, UP[4:])
    # Act
    out = _sim(bars, scripted(long=[0, 4], stops={4: 1}))
    summary = summarize(out.trades, out.equity_minute, out.daily, n_trials=None)
    # Assert
    assert tuple(out.trades.columns) == TRADE_COLUMNS
    labels = {"exit_reason": pl.String, "trade_date": pl.Date}
    assert dict(out.trades.schema) == {c: labels.get(c, pl.Int64) for c in TRADE_COLUMNS}
    assert out.trades["exit_reason"].to_list() == ["SESSION_END", "STOP"]
    assert out.trades["trade_date"].to_list() == [D1, D2]
    assert tuple(out.equity_minute.columns) == EQUITY_COLUMNS
    assert tuple(out.daily.columns) == DAILY_COLUMNS
    assert summary["net_cents"] == int(out.trades["net_cents"].sum())


def test_daily_rows_chain_and_sum_to_the_ledger() -> None:
    bars = minute_bars(D1, 10, 0, UP[:4]) + minute_bars(D2, 10, 0, UP[4:])
    out = _sim(bars, scripted(long=[0, 4]))
    daily = out.daily
    assert daily["trade_date"].to_list() == [D1, D2]
    assert daily["eq_open"][1] == daily["eq_close"][0]
    assert (
        int(daily["net_cents"].sum())
        == int(out.trades["net_cents"].sum())
        == out.equity_minute["eq_close"][-1]
    )


def test_mae_and_mfe_are_non_negative_magnitudes() -> None:
    bars = minute_bars(D1, 10, 0, [(100, 103, 97, 100)] * 5)
    out = _sim(bars, scripted(short=[0]))
    assert out.trades["mae_ticks"].to_list() == [3]
    assert out.trades["mfe_ticks"].to_list() == [3]


def test_close_fill_is_stamped_at_the_bar_close() -> None:
    bars = minute_bars(D1, 10, 0, UP[:3])
    out = _sim(bars)
    assert out.trades["exit_ts"].to_list() == [ct_ns(D1, 10, 2) + NS_MIN]
    assert out.trades["entry_ts"].to_list() == [ct_ns(D1, 10, 1)]


def test_a_leaky_strategy_is_refused_before_the_kernel() -> None:
    def peek(bars, params):  # type: ignore[no-untyped-def]
        close = bars["close"]
        nxt = np.append(close[1:], close[-1:])
        n = len(close)
        zeros = np.zeros(n, dtype=np.int64)
        return {
            "entry_long": nxt > close,
            "entry_short": np.zeros(n, dtype=bool),
            "exit": np.zeros(n, dtype=bool),
            "stop_ticks": zeros,
            "target_ticks": zeros,
        }

    bars = minute_bars(D1, 10, 0, [(100, 101, 99, 100 + (i % 3)) for i in range(40)])
    with pytest.raises(LookAheadDetected):
        _sim(bars, strategy_of(peek))


def test_a_negative_stop_distance_refuses_the_run() -> None:
    bars = minute_bars(D1, 10, 0, UP[:4])
    with pytest.raises(SignalError):
        _sim(bars, scripted(long=[0], stops={0: -1}))


def test_entry_allowed_must_cover_the_usable_bars() -> None:
    frame = frame_of(minute_bars(D1, 10, 0, UP[:4]))
    with pytest.raises(EngineInputError):
        simulate(scripted(), frame, {}, market_inputs(frame), seed=1, entry_allowed=np.ones(3, dtype=bool))


def test_non_increasing_time_is_refused() -> None:
    bars = minute_bars(D1, 10, 0, UP[:3])
    frame = frame_of([bars[0], bars[2], bars[1]])
    with pytest.raises(EngineInputError):
        simulate(scripted(), frame, {}, market_inputs(frame), seed=1)


def test_engine_and_metrics_name_the_same_exit_reasons() -> None:
    from pfb.engine.kernel import EXIT_REASONS
    from pfb.metrics.core import EXIT_REASONS as METRICS_REASONS

    assert EXIT_REASONS == METRICS_REASONS
