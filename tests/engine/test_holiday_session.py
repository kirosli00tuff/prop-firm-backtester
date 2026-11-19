"""A holiday trade date that spans two calendar days, on synthetic bars with the real equity calendar.

MLK 2025-01-20: the trade date runs Sun 01-19 17:00 to Mon 01-20 12:00 CT (early halt), must-be-flat from
11:30.
Christmas Eve 2024-12-24: Mon 12-23 17:00 to Tue 12-24 12:15 CT, must-be-flat from 11:45 to Wed 17:00.
Checks session_end, flat_by and entry_fill_ok across the CT midnight inside the trade date and at the halt.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from engine_cases import NS_MIN, ct_ns, scripted

from pfb.engine.run import MarketInputs, in_intervals, prepare_bars, simulate
from pfb.engine.synthetic import synthetic_frame
from pfb.instruments import sessions

ROOT = "MNQ"
EPOCH = date(1970, 1, 1)
CASES = [
    pytest.param(date(2025, 1, 20), (11, 30), id="mlk-2025-01-20"),
    pytest.param(date(2024, 12, 24), (11, 45), id="christmas-eve-2024-12-24"),
]


def _inputs(frame: pl.DataFrame, first: date, last: date) -> MarketInputs:
    """Real must-be-flat intervals and real globex windows, flat costs."""
    starts, ends = sessions.must_be_flat_intervals_ns(ROOT, first, last)
    days = np.unique(frame["trade_date"].cast(pl.Int32).to_numpy()).astype(np.int32)
    windows = [sessions.session_window_ns(ROOT, EPOCH + timedelta(days=int(d)), "globex") for d in days]
    return MarketInputs(
        must_be_flat_starts=np.asarray(starts, dtype=np.int64),
        must_be_flat_ends=np.asarray(ends, dtype=np.int64),
        session_days=days,
        session_starts=np.array([w[0] for w in windows], dtype=np.int64),
        session_ends=np.array([w[1] for w in windows], dtype=np.int64),
        unsourced_days=np.zeros(0, dtype=np.int32),
        slippage_cents=np.tile(np.array([5, 7], dtype=np.int64), (1440, 1)),
        calibrated=np.ones(1440, dtype=bool),
        commission_cents=10,
        tick_value_cc=5000,
    )


def _setup(holiday: date) -> tuple[pl.DataFrame, MarketInputs, np.ndarray]:
    first, last = holiday - timedelta(days=3), holiday + timedelta(days=2)
    frame = synthetic_frame(ROOT, first, last, seed=17)
    inputs = _inputs(frame, first, last)
    prepared = prepare_bars(frame, inputs)
    on_day = (prepared.frame["trade_date"] == holiday).to_numpy()
    return prepared.frame, inputs, on_day


def _index_at(frame: pl.DataFrame, ts: int) -> int:
    hits = np.flatnonzero(frame["ts_event"].to_numpy() == ts)
    assert len(hits) == 1, "the synthetic bar at this minute is absent; pick another seed"
    return int(hits[0])


@pytest.mark.parametrize(("holiday", "halt_flat"), CASES)
def test_holiday_trade_date_spans_two_calendar_days_and_ends_once(
    holiday: date, halt_flat: tuple[int, int]
) -> None:
    # Arrange
    frame, inputs, on_day = _setup(holiday)
    prepared = prepare_bars(frame, inputs)
    # Act
    ct_days = (
        frame.filter(pl.Series(on_day))
        .select(pl.from_epoch("ts_event", time_unit="ns").dt.replace_time_zone("UTC"))
        .to_series()
        .dt.convert_time_zone("America/Chicago")
        .dt.date()
        .unique()
        .sort()
        .to_list()
    )
    ends = prepared.session_end[on_day]
    # Assert: bars on the evening before and on the holiday itself; one session_end, at the last bar
    assert ct_days == [holiday - timedelta(days=1), holiday]
    assert int(ends.sum()) == 1 and bool(ends[-1])
    last_ts = int(frame["ts_event"].to_numpy()[on_day][-1])
    assert last_ts < ct_ns(holiday, 12, 30)


@pytest.mark.parametrize(("holiday", "halt_flat"), CASES)
def test_flat_by_and_entry_fill_ok_across_midnight_and_at_the_halt(
    holiday: date, halt_flat: tuple[int, int]
) -> None:
    # Arrange
    frame, inputs, on_day = _setup(holiday)
    ts = frame["ts_event"].to_numpy()[on_day]
    flat_from = ct_ns(holiday, *halt_flat)
    # Act
    flat_by = in_intervals(ts, inputs.must_be_flat_starts, inputs.must_be_flat_ends)
    fill_ok = ~flat_by & ~in_intervals(ts + NS_MIN, inputs.must_be_flat_starts, inputs.must_be_flat_ends)
    # Assert: nothing is must-be-flat across CT midnight; flat from the halt rule's minute on
    assert np.array_equal(flat_by, ts >= flat_from)
    assert np.array_equal(fill_ok, ts + NS_MIN < flat_from)
    midnight = ct_ns(holiday, 0, 0)
    assert not flat_by[(ts >= midnight - 5 * NS_MIN) & (ts < midnight + 5 * NS_MIN)].any()


@pytest.mark.parametrize(("holiday", "halt_flat"), CASES)
def test_a_position_held_over_midnight_is_flattened_at_the_halt_rule(
    holiday: date, halt_flat: tuple[int, int]
) -> None:
    # Arrange: a long decided at 23:00 CT on the evening before (fills 23:01), never exited by signal
    frame, inputs, _ = _setup(holiday)
    decide = _index_at(frame, ct_ns(holiday - timedelta(days=1), 23, 0))
    flat_from = ct_ns(holiday, *halt_flat)
    # Act
    out = simulate(scripted(long=[decide]), frame, {}, inputs, seed=2)
    # Assert: the trade crosses CT midnight and leaves at the first bar at or after the flat-by minute
    trade = out.trades.row(0, named=True)
    assert trade["exit_reason"] == "FLAT_BY"
    assert trade["trade_date"] == holiday
    assert trade["entry_ts"] < ct_ns(holiday, 0, 0) <= trade["exit_ts"]
    first_flat = int(frame["ts_event"].to_numpy()[frame["ts_event"].to_numpy() >= flat_from][0])
    assert trade["exit_ts"] == first_flat


@pytest.mark.parametrize(("holiday", "halt_flat"), CASES)
def test_an_entry_whose_next_minute_is_must_be_flat_is_dropped(
    holiday: date, halt_flat: tuple[int, int]
) -> None:
    # Arrange: decision two minutes before the flat-by minute; the fill minute's successor is must-be-flat
    frame, inputs, _ = _setup(holiday)
    flat_from = ct_ns(holiday, *halt_flat)
    decide = _index_at(frame, flat_from - 2 * NS_MIN)
    _index_at(frame, flat_from - NS_MIN)  # the fill bar exists
    # Act
    out = simulate(scripted(long=[decide]), frame, {}, inputs, seed=2)
    # Assert
    assert out.trades.height == 0
    assert out.counters["entries_dropped"] == 1
