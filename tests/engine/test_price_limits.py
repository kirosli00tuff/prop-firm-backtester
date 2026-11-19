"""Known answers of the price-limit port without upstream."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from pfb.engine import price_limits as pl

DAY = date(2025, 6, 3)  # MNQ period 2020-11-02..2026-06-19: ETH 7 %, RTH 7 % until 14:25 CT
S = pl.tick_price("MNQ", 80_000)  # 20000.00


def _ct_ns(hh: int, mm: int, day: date = DAY) -> int:
    from datetime import UTC, datetime

    return pl.utc_to_ns(datetime(day.year, day.month, day.day, hh, mm, tzinfo=pl.CT).astimezone(UTC))


def test_tick_price_is_exact_vendor_units() -> None:
    assert pl.tick_price("MNQ", 80_001) == Decimal("20000.25")
    assert pl.tick_price("ZC", 1801) == Decimal("450.25")  # vendor cents: 0.25 cent x 100... ticks of 0.25


def test_stop_levels_rth_down_only_and_thresholds() -> None:
    band = pl.limit_band("MNQ", DAY, _ct_ns(10, 0), S)
    levels = pl.stop_levels(band, S)
    assert levels.upper is None and levels.lower == Decimal("19000.00")  # 20000 - 1400 + 400
    assert levels.lower is not None and pl.ticks_at_or_below("MNQ", levels.lower) == 76_000
    assert pl.beyond_stop("MNQ", 76_000, levels) == "below" and pl.beyond_stop("MNQ", 76_001, levels) is None


def test_eth_band_both_sides_and_buffer_is_an_argument() -> None:
    band = pl.limit_band("MNQ", DAY, _ct_ns(3, 0), S)
    assert (band.up, band.down) == (Decimal("0.07"), Decimal("0.07"))
    levels = pl.stop_levels(band, S, Decimal("0.03"))
    assert (levels.upper, levels.lower) == (Decimal("20800.00"), Decimal("19200.00"))


def test_entry_refusal_and_forced_exit_at_the_level() -> None:
    ts = _ct_ns(10, 0)
    assert pl.check_price_limit_entry("MNQ", DAY, ts, S, 76_000) is not None
    assert pl.check_price_limit_entry("MNQ", DAY, ts, S, 76_001) is None
    ex = pl.required_price_limit_exit("MNQ", DAY, ts, 1, S, 76_000)
    assert ex is not None and (ex.side, ex.quantity) == ("sell", 1)
    assert pl.required_price_limit_exit("MNQ", DAY, ts, 0, S, 76_000) is None


def test_limit_pending_outside_table_and_dcb_no_lock() -> None:
    with pytest.raises(pl.LimitPending):
        pl.limit_period("MNQ", date(2026, 6, 22))
    with pytest.raises(pl.LimitPending):
        pl.limit_period("ZN", date(2019, 6, 3))
    band = pl.limit_band("ZN", DAY, _ct_ns(10, 0), Decimal("110"))
    assert band.kind is pl.LimitKind.DCB_ONLY and band.up_amount is None and band.down_amount is None


def test_locked_exit_waits_for_a_bar_trading_through() -> None:
    band = pl.limit_band("MNQ", DAY, _ct_ns(3, 0), S)
    up, down = pl.limit_prices(band, S)
    assert down == Decimal("18600.00")
    locked = pl.Bar(74_400, 74_400, 74_400, 74_400)
    through = pl.Bar(74_400, 74_401, 74_400, 74_401)
    got = pl.locked_exit_fill("MNQ", "sell", up, down, [locked, locked, through])
    assert (got.index, got.price, got.flagged, got.bars_waited) == (2, 74_400, True, 2)
    assert pl.locked_exit_fill("MNQ", "sell", up, down, [locked]).index is None


def test_settlement_proxy_vwap_and_fallback() -> None:
    w0, w1 = pl.settlement_window_ns("MNQ", DAY)
    minute = 60_000_000_000
    got = pl.settlement_proxy("MNQ", [w0 - 30 * 10**9, w0 - 30 * 10**9 + minute], [80_000, 80_004], [1, 1],
                              (w0, w1))  # fmt: skip
    assert got is not None and got.method == "vwap_close" and got.value == Decimal("20000.00")
    fallback = pl.settlement_proxy("MNQ", [w0 - 10 * minute], [80_004], [5], (w0, w1))
    assert fallback is not None and fallback.method == "last_close_before_window_end"
    assert pl.settlement_proxy("MNQ", [w1], [80_000], [5], (w0, w1)) is None


def test_rates_window_refused_not_guessed() -> None:
    with pytest.raises(pl.SettlementWindowUnavailable):
        pl.settlement_window_ct("ZN", DAY)
