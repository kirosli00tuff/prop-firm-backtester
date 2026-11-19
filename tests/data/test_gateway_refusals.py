"""DataGateway refusals: every one raises, nothing is filtered silently."""

from datetime import date, datetime, time

import pytest
from gateway_synth import TICKS_FIXED, Row, SynthStores, ct_ns, research_name, session_rows, write_store

from pfb.data.catalog import CatalogEntry
from pfb.data.errors import (
    BadRequest,
    BoundBreached,
    ClosedWindowRefused,
    GatewayRefusal,
    HashMismatch,
    NoBarsFound,
    OffGridPrice,
    PathRefused,
    StoreRefused,
    TickMismatch,
)

JAN4, JAN29, FEB12, MAR31 = date(2021, 1, 4), date(2021, 1, 29), date(2021, 2, 12), date(2021, 3, 31)


@pytest.fixture
def full(stores: SynthStores) -> SynthStores:
    """NQ, ZL and ZM research files over 2021-01-04..2021-03-31 (closed rows included, as on disk today)."""
    for root in ("NQ", "ZL", "ZM"):
        stores.research(root, session_rows(JAN4, MAR31))
    return stores


@pytest.mark.parametrize(
    ("root", "first", "last", "window_id"),
    [
        ("NQ", date(2021, 3, 1), date(2021, 3, 5), "syn_holdout"),
        ("NQ", date(2021, 2, 15), date(2021, 2, 19), "syn_embargo"),
        ("ZL", date(2021, 1, 11), date(2021, 1, 15), "syn_test"),
        ("ZM", date(2021, 2, 1), date(2021, 2, 5), "syn_decision"),
    ],
)
def test_each_window_kind_refuses_a_request_inside(
    full: SynthStores, root: str, first: date, last: date, window_id: str
) -> None:
    gateway = full.gateway()

    with pytest.raises(ClosedWindowRefused) as caught:
        gateway.bars(root, first, last)

    assert caught.value.window_id == window_id
    assert gateway.provenance == ()


def test_request_merely_touching_a_window_is_refused(full: SynthStores) -> None:
    with pytest.raises(ClosedWindowRefused, match="syn_embargo"):
        full.gateway().bars("NQ", date(2021, 2, 8), date(2021, 2, 15))


def test_product_scoped_window_refuses_only_its_roots(full: SynthStores) -> None:
    gateway = full.gateway()

    frame = gateway.bars("NQ", date(2021, 1, 11), date(2021, 1, 15))
    with pytest.raises(ClosedWindowRefused, match="syn_test"):
        gateway.bars("ZL", date(2021, 1, 11), date(2021, 1, 15))

    assert frame.height == 15


def test_open_ended_window_refuses_a_far_future_request(full: SynthStores) -> None:
    with pytest.raises(ClosedWindowRefused, match="syn_holdout"):
        full.gateway().bars("NQ", date(2400, 1, 3), date(2400, 1, 7))


def test_request_refusal_comes_before_any_file_is_opened(full: SynthStores) -> None:
    catalog = full.catalog()
    (full.paths.research / "NQ" / research_name("NQ")).unlink()

    with pytest.raises(ClosedWindowRefused):
        full.gateway(catalog).bars("NQ", date(2021, 2, 15), date(2021, 2, 16))


def test_mislabelled_row_caught_by_time_span_while_label_open(stores: SynthStores) -> None:
    rows = session_rows(date(2021, 2, 8), FEB12)
    rows.append(Row(ct_ns(FEB12, time(16, 30)), FEB12.isoformat(), 90_000))  # inside syn_embargo's span
    stores.research("NQ", rows)

    with pytest.raises(ClosedWindowRefused, match="syn_embargo"):
        stores.gateway().bars("NQ", date(2021, 2, 8), FEB12)


def test_row_after_the_bound_instant_is_refused(stores: SynthStores) -> None:
    rows = session_rows(date(2021, 1, 25), JAN29)
    rows.append(
        Row(ct_ns(JAN29, time(17, 0)), JAN29.isoformat(), 90_000)
    )  # labelled inside, stamped at bound
    stores.research("NQ", rows)

    with pytest.raises(BoundBreached):
        stores.gateway().bars("NQ", date(2021, 1, 25), JAN29)


def test_row_one_minute_before_the_bound_is_returned(stores: SynthStores) -> None:
    rows = session_rows(date(2021, 1, 25), JAN29)
    rows.append(Row(ct_ns(JAN29, time(16, 59)), JAN29.isoformat(), 90_000))
    stores.research("NQ", rows)

    frame = stores.gateway().bars("NQ", date(2021, 1, 25), JAN29)

    assert frame.height == 16


def test_null_label_is_refused(stores: SynthStores) -> None:
    rows = session_rows(date(2021, 1, 25), JAN29)
    rows.append(Row(ct_ns(JAN29, time(16, 0)), None, 90_000))
    stores.research("NQ", rows)

    with pytest.raises(StoreRefused, match="null"):
        stores.gateway().bars("NQ", date(2021, 1, 25), JAN29)


def test_ts_not_strictly_increasing_is_refused(stores: SynthStores) -> None:
    rows = session_rows(date(2021, 1, 25), JAN29)
    rows.insert(4, rows[3])
    stores.research("NQ", rows)

    with pytest.raises(StoreRefused, match="strictly increasing"):
        stores.gateway().bars("NQ", date(2021, 1, 25), JAN29)


def test_duplicate_trade_date_across_stores_is_refused(stores: SynthStores) -> None:
    stores.hist("NQ", session_rows(JAN4, date(2021, 1, 11))[:-2])  # 01-11 evening and 08:30 bars
    late = ct_ns(date(2021, 1, 11), time(15, 0))
    stores.step2("NQ", [r for r in session_rows(date(2021, 1, 11), JAN29) if r.ts_ns >= late])

    with pytest.raises(StoreRefused, match="two store files"):
        stores.gateway().bars("NQ", JAN4, JAN29)


@pytest.mark.parametrize("raw_price", [10_000.125, 10_000.25 + 2e-6, float("nan")])
def test_off_grid_price_is_refused_without_snapping(stores: SynthStores, raw_price: float) -> None:
    rows = session_rows(date(2021, 1, 25), JAN29)
    rows[2] = Row(rows[2].ts_ns, rows[2].label, 0, raw_price=raw_price)
    stores.research("NQ", rows)

    with pytest.raises(OffGridPrice):
        stores.gateway().bars("NQ", date(2021, 1, 25), JAN29)


@pytest.mark.parametrize("meta_tick", [TICKS_FIXED["NQ"] * 2, TICKS_FIXED["NQ"] // 10, TICKS_FIXED["NQ"] + 1])
def test_metadata_tick_disagreeing_with_table_is_refused(stores: SynthStores, meta_tick: int) -> None:
    stores.research("NQ", session_rows(date(2021, 1, 25), JAN29), meta_tick_fixed=meta_tick)

    with pytest.raises(TickMismatch):
        stores.gateway().bars("NQ", date(2021, 1, 25), JAN29)


def test_metadata_exchange_tick_times_vendor_factor_is_accepted(stores: SynthStores) -> None:
    stores.research("ZC", session_rows(date(2021, 1, 25), JAN29), meta_tick_fixed=TICKS_FIXED["ZC"] // 100)

    frame = stores.gateway().bars("ZC", date(2021, 1, 25), JAN29)

    assert frame.height == 15


def test_hash_mismatch_after_catalog_was_written(stores: SynthStores) -> None:
    stores.research("NQ", session_rows(date(2021, 1, 25), JAN29))
    catalog = stores.catalog()
    stores.research("NQ", session_rows(date(2021, 1, 25), JAN29, start_ticks=50_000))

    with pytest.raises(HashMismatch):
        stores.gateway(catalog).bars("NQ", date(2021, 1, 25), JAN29)


def test_catalog_entry_pointing_outside_the_roots_is_refused(stores: SynthStores) -> None:
    stores.research("NQ", session_rows(date(2021, 1, 25), JAN29))
    catalog = stores.catalog()
    outside = write_store(
        stores.base / "elsewhere" / "NQ" / research_name("NQ"), session_rows(JAN4, JAN29), 1
    )
    entry = catalog.files[0]
    forged = CatalogEntry(**{**entry.__dict__, "path": str(outside)})
    tampered = type(catalog)(catalog.created_utc, catalog.closed_windows_sha256, (forged,), ())

    with pytest.raises(PathRefused):
        stores.gateway(tampered).bars("NQ", date(2021, 1, 25), JAN29)


def test_bad_requests_are_refused(stores: SynthStores) -> None:
    stores.research("NQ", session_rows(date(2021, 1, 25), JAN29))
    gateway = stores.gateway()

    with pytest.raises(BadRequest, match="unknown root"):
        gateway.bars("ES", date(2021, 1, 25), JAN29)
    with pytest.raises(BadRequest, match="after"):
        gateway.bars("NQ", JAN29, date(2021, 1, 25))
    with pytest.raises(BadRequest, match="datetime.date"):
        gateway.bars("NQ", datetime(2021, 1, 25), JAN29)


def test_no_store_file_covers_the_request(stores: SynthStores) -> None:
    stores.research("NQ", session_rows(date(2021, 1, 25), JAN29))

    with pytest.raises(NoBarsFound):
        stores.gateway().bars("NQ", date(2020, 6, 1), date(2020, 6, 5))


def test_every_refusal_is_a_gateway_refusal_runtime_error() -> None:
    for cls in (BadRequest, BoundBreached, ClosedWindowRefused, HashMismatch, NoBarsFound, OffGridPrice):
        assert issubclass(cls, GatewayRefusal)
        assert issubclass(cls, RuntimeError)
