"""DataGateway reads: columns and dtypes, tick conversion, splicing, clock windows, provenance."""

from datetime import UTC, date, datetime, time, timedelta

import numpy as np
import polars as pl
import pytest
from gateway_synth import BASE_TICKS, TICKS_FIXED, SynthStores, ct_ns, session_rows

from pfb.data.errors import BadRequest, OffGridPrice
from pfb.data.gateway import CT, OUTPUT_COLUMNS, ClockWindow, bound_ns, ct_minute_expr, to_ticks

JAN4, JAN8, JAN11, JAN22 = date(2021, 1, 4), date(2021, 1, 8), date(2021, 1, 11), date(2021, 1, 22)

EXPECTED_DTYPES = {
    "ts_event": pl.Int64,
    "open": pl.Int64,
    "high": pl.Int64,
    "low": pl.Int64,
    "close": pl.Int64,
    "volume": pl.Int64,
    "trade_date": pl.Date,
    "ct_minute": pl.Int16,
    "is_roll_session": pl.Boolean,
    "in_flatten_window": pl.Boolean,
    "in_no_new_positions_window": pl.Boolean,
    "in_scheduled_closure": pl.Boolean,
    "vendor_degraded_day": pl.Boolean,
    "early_halt_ct": pl.String,
    "gap_before_minutes": pl.Int64,
    "instrument_id": pl.Int64,
    "raw_symbol": pl.String,
    "era": pl.String,
}


@pytest.fixture
def spliced(stores: SynthStores) -> SynthStores:
    stores.hist("NQ", session_rows(JAN4, JAN8))
    stores.step2("NQ", session_rows(JAN11, JAN22, start_ticks=BASE_TICKS + 100))
    return stores


def test_columns_order_and_dtypes(spliced: SynthStores) -> None:
    frame = spliced.gateway().bars("NQ", JAN4, JAN22)

    assert tuple(frame.columns) == OUTPUT_COLUMNS
    assert dict(frame.schema) == {k: v() for k, v in EXPECTED_DTYPES.items()}


def test_prices_come_back_as_vendor_ticks(spliced: SynthStores) -> None:
    frame = spliced.gateway().bars("NQ", JAN4, JAN8)

    assert frame["open"].to_list()[:3] == [BASE_TICKS, BASE_TICKS + 1, BASE_TICKS + 2]


def test_two_store_splice_hist_then_step2(spliced: SynthStores) -> None:
    gateway = spliced.gateway()

    frame = gateway.bars("NQ", JAN4, JAN22)

    assert frame.height == 15 * 3
    assert frame["era"].unique(maintain_order=True).to_list() == ["hist", "step2"]
    assert frame["ts_event"].is_sorted() and frame["ts_event"].n_unique() == frame.height
    assert [(p.era, p.rows) for p in gateway.provenance] == [("hist", 15), ("step2", 30)]


def test_only_files_overlapping_the_request_are_read(spliced: SynthStores) -> None:
    gateway = spliced.gateway()

    frame = gateway.bars("NQ", JAN11, JAN22)

    assert frame["era"].unique().to_list() == ["step2"]
    assert [p.era for p in gateway.provenance] == ["step2"]


def test_label_range_filter_keeps_the_previous_evening_bar(spliced: SynthStores) -> None:
    frame = spliced.gateway().bars("NQ", JAN11, JAN11)

    assert frame["ts_event"].to_list()[0] == ct_ns(date(2021, 1, 10), time(17, 30))
    assert frame["trade_date"].unique().to_list() == [JAN11]


def test_clock_window_filter(spliced: SynthStores) -> None:
    gateway = spliced.gateway()

    day = gateway.bars("NQ", JAN4, JAN8, ClockWindow(8 * 60 + 30, 15 * 60))
    wrap = gateway.bars("NQ", JAN4, JAN8, ClockWindow(17 * 60, 9 * 60))

    assert day["ct_minute"].unique().to_list() == [510]
    assert sorted(wrap["ct_minute"].unique().to_list()) == [510, 1050]
    assert gateway.provenance[0].rows == 10


@pytest.mark.parametrize(("start", "end"), [(-1, 10), (0, 1441), (600, 600), (1440, 10)])
def test_invalid_clock_window_is_refused(start: int, end: int) -> None:
    with pytest.raises(BadRequest):
        ClockWindow(start, end)


def test_ct_minute_matches_zoneinfo_across_dst() -> None:
    instants = [datetime(2021, 3, 14, 7, 59, tzinfo=UTC) + timedelta(minutes=m) for m in range(0, 180, 7)]
    instants += [datetime(2021, 11, 7, 5, 30, tzinfo=UTC) + timedelta(minutes=m) for m in range(0, 180, 11)]
    ns = [int(i.timestamp()) * 1_000_000_000 for i in instants]

    got = pl.DataFrame({"ts_event": ns}).select(ct_minute_expr())["ct_minute"].to_list()

    expected = [i.astimezone(CT).hour * 60 + i.astimezone(CT).minute for i in instants]
    assert got == expected


def test_bound_is_17_ct_on_the_last_calendar_day() -> None:
    assert bound_ns(date(2021, 1, 29)) == ct_ns(date(2021, 1, 29), time(17, 0))
    assert bound_ns(date(2021, 7, 2)) == int(datetime(2021, 7, 2, 22, tzinfo=UTC).timestamp()) * 1_000_000_000


@pytest.mark.parametrize(
    "tick_fixed",
    [15_625_000, 250_000_000, 25_000_000, 3_906_250, 500, 5_000_000_000, TICKS_FIXED["ZL"]],
    ids=["1/64", "factor-100 grain (ZC 0.0025*100)", "factor-100 livestock", "1/256", "6J", "MBT", "ZL"],
)
def test_tick_round_trip_on_grid(tick_fixed: int) -> None:
    counts = np.array([0, 1, 7, 63, 64, 1_234, 98_765, 4_000_001], dtype=np.int64)
    prices = counts * tick_fixed / 1e9

    ticks = to_ticks(prices, tick_fixed, "SYN", "close")

    assert ticks.dtype == np.int64
    assert np.array_equal(ticks, counts)
    assert np.array_equal(ticks * tick_fixed / 1e9, prices)


def test_tick_off_grid_raises() -> None:
    with pytest.raises(OffGridPrice, match="1 price"):
        to_ticks(np.array([110.015625, 110.0078125]), 15_625_000, "ZN", "open")


def test_empty_result_inside_coverage_keeps_the_schema(stores: SynthStores) -> None:
    stores.research("NQ", session_rows(JAN4, date(2021, 1, 15)))

    frame = stores.gateway().bars("NQ", date(2021, 1, 9), date(2021, 1, 10))

    assert frame.height == 0
    assert tuple(frame.columns) == OUTPUT_COLUMNS
