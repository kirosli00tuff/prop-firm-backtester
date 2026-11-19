"""Price-limit proximity and locked-market rules, ported from the author's research repo. Upstream's readings
are kept: the initial hard limit only, the first RTH
equity level for the whole window, DCB-only products without a lock limit by default.

Topstep's rule as upstream encodes it (short quotes from upstream's docstring): "Stop trading above:
Settlement x (1 +
Price Limit% - 2%)"; "Stop trading below: Settlement x (1 - Price Limit% + 2%)". No entry, and an immediate
exit, while the price is at or beyond either stop level. A locked market (a long at limit-down, a short at
limit-up) does not fill an exit until a bar trades through the limit price.

Prices cross this module's boundary as int64 ticks of the root (vendor units = ticks x vendor_tick_fixed /
1e9); inside, every level is an exact Decimal computed as upstream computes it (the same 28-digit context),
and a
level becomes a tick threshold through exact Fraction arithmetic: no float decides a boundary. The prior
settlement is upstream's proxy (settlement_proxy): the volume-weighted close of the one-minute bars in the
settlement window, an exact Decimal; it is a reference, not a price on the tick grid.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from enum import Enum
from fractions import Fraction
from functools import lru_cache
from zoneinfo import ZoneInfo

from pfb.engine.price_limit_tables import HARD_LIMIT_PRODUCTS, LIMITS, TABLE_WINDOW, LimitKind, LimitPeriod
from pfb.instruments.sessions import calendar_of
from pfb.instruments.table import PRICE_SCALE, Instrument, load_instruments

CT = ZoneInfo("America/Chicago")
D = Decimal
TOPSTEP_BUFFER = D("0.02")  # "Price Limit% - 2%"
# Python's default context, which upstream runs under
DECIMAL_CONTEXT = Context(prec=28, rounding=ROUND_HALF_EVEN)
NS_PER_S = 1_000_000_000
NS_PER_MINUTE = 60 * NS_PER_S
NS_PER_US = 1_000
UTC_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class DcbReading(Enum):
    NO_LOCK_LIMIT = "no_lock_limit"  # literal: no price lock limit, the rule does not apply
    VARIANT_AS_LIMIT = "variant_as_limit"  # conservative: the DCB variant as a daily limit


DCB_READING_DEFAULT = DcbReading.NO_LOCK_LIMIT


class LimitPending(LookupError):
    """No CME source for this product and date: the rule cannot be evaluated."""


class SettlementWindowUnavailable(LookupError):
    """No settlement window for this root and date (a full closure, or a group whose early settlement table
    is not ported)."""


# ------------------------------------------------------------------ units ----
@lru_cache(maxsize=64)
def _instrument(root: str) -> Instrument:
    return load_instruments().get(root)


def tick_price(root: str, ticks: int) -> Decimal:
    """A price in ticks as an exact vendor-unit Decimal."""
    return D(int(ticks) * _instrument(root).vendor_tick_fixed).scaleb(-9)


def _ticks_exact(root: str, level: Decimal) -> Fraction:
    return Fraction(level) * PRICE_SCALE / _instrument(root).vendor_tick_fixed


def ticks_at_or_above(root: str, level: Decimal) -> int:
    """The smallest tick count whose price is >= level."""
    return math.ceil(_ticks_exact(root, level))


def ticks_at_or_below(root: str, level: Decimal) -> int:
    """The largest tick count whose price is <= level."""
    return math.floor(_ticks_exact(root, level))


def ns_to_utc(ts_ns: int) -> datetime:
    return UTC_EPOCH + timedelta(microseconds=int(ts_ns) // NS_PER_US)


def utc_to_ns(ts: datetime) -> int:
    return ((ts - UTC_EPOCH) // timedelta(microseconds=1)) * NS_PER_US


# ----------------------------------------------------------- the period ----
def limit_period(root: str, trade_date: date) -> LimitPeriod:
    """The limit period in force on ``trade_date``; raises LimitPending when unsourced (upstream
    limit_period)."""
    periods = LIMITS.get(root)
    if periods is None:
        raise KeyError(f"{root!r} has no price-limit table")
    for p in periods:
        if p.start <= trade_date <= p.end:
            if p.status == "pending":
                raise LimitPending(
                    f"{root} {trade_date}: no CME source for the limit ({p.start}..{p.end}, pending)"
                )
            return p
    raise LimitPending(f"{root} {trade_date}: outside the table ({periods[0].start}..{periods[-1].end})")


def limit_kind(root: str) -> LimitKind:
    if root not in LIMITS:
        raise KeyError(f"{root!r} has no price-limit table")
    return LIMITS[root][0].kind


# -------------------------------------------------------- settlement window ----
@dataclass(frozen=True)
class Window:
    start_ct: time
    end_ct: time  # exclusive
    source: str


_WIKI = "cme_wiki_settlement_times"
SETTLEMENT_WINDOW_CT: dict[str, Window] = {
    "equity": Window(time(14, 59, 30), time(15, 0), "pl_* (equity fixing price, VWAP 14:59:30-15:00:00 CT)"),
    "rates": Window(time(13, 59, 30), time(14, 0), _WIKI + " Treasuries"),
    "fx": Window(time(13, 59, 30), time(14, 0), _WIKI + " FX"),
    "energy": Window(time(13, 28), time(13, 30), _WIKI + " Energy Products (14:28-14:30 ET)"),
    "gold": Window(time(12, 29), time(12, 30), _WIKI + " Gold (13:29-13:30 ET)"),
    "silver": Window(time(12, 24), time(12, 25), _WIKI + " Silver (13:24-13:25 ET)"),
    "copper": Window(time(11, 59), time(12, 0), _WIKI + " Copper (12:59-13:00 ET)"),
    "grains": Window(time(13, 14), time(13, 15), _WIKI + " Grains/Oilseeds"),
    "livestock": Window(time(12, 59, 30), time(13, 0), _WIKI + " Livestock"),
    "crypto": Window(time(14, 59), time(15, 0), _WIKI + " Bitcoin"),
}
# On an early-settlement day the rates window moves to a CME-published early settlement time; that table is
# not included, so a rates window is refused rather than guessed.
UNPORTED_EARLY_SETTLEMENT_GROUPS = frozenset({"rates"})


def _window_key(root: str) -> str:
    inst = _instrument(root)
    return inst.subgroup if inst.group == "metals" else inst.group


def _shift(t: time, delta: timedelta) -> time:
    return (datetime.combine(date(2000, 1, 3), t) + delta).time()


def settlement_window_ct(root: str, trade_date: date) -> Window:
    """Upstream settlement_window_ct: the regular window; on an early halt at or before its end, the regular
    length
    ending at the halt (proxy reading R-P3). A full closure has no settlement."""
    group = _instrument(root).group
    if group in UNPORTED_EARLY_SETTLEMENT_GROUPS:
        raise SettlementWindowUnavailable(f"{root}: the {group} early settlement table is not ported")
    reg = SETTLEMENT_WINDOW_CT[_window_key(root)]
    length = datetime.combine(date(2000, 1, 3), reg.end_ct) - datetime.combine(date(2000, 1, 3), reg.start_ct)
    era = calendar_of(root).era_for(trade_date)
    if trade_date in era.full_closures:
        raise SettlementWindowUnavailable(f"{root}: {trade_date} is a full closure; no settlement")
    halt_min = era.early_halts.get(trade_date)
    if halt_min is not None:
        halt = time(halt_min // 60, halt_min % 60)
        if halt <= reg.end_ct:
            return Window(_shift(halt, -length), halt, f"early halt {halt:%H:%M}; proxy reading R-P3")
    return reg


def settlement_window_ns(root: str, trade_date: date) -> tuple[int, int]:
    w = settlement_window_ct(root, trade_date)
    start = datetime.combine(trade_date, w.start_ct, tzinfo=CT).astimezone(UTC)
    end = datetime.combine(trade_date, w.end_ct, tzinfo=CT).astimezone(UTC)
    return utc_to_ns(start), utc_to_ns(end)


def previous_trade_date(root: str, day: date) -> date | None:
    """The previous trade date on the group calendar (at most 10 days back)."""
    era = calendar_of(root).era_for(day)
    for n in range(1, 11):
        if era.is_trade_date(day - timedelta(days=n)):
            return day - timedelta(days=n)
    return None


# ------------------------------------------------------- settlement proxy ----
@dataclass(frozen=True)
class SettlementProxy:
    value: Decimal
    method: str  # "vwap_close" | "last_close_before_window_end"
    bars_used: int


def settlement_proxy(
    root: str,
    starts_ns: Sequence[int],
    closes: Sequence[int],
    volumes: Sequence[int],
    window: tuple[int, int],
) -> SettlementProxy | None:
    """price-limit's proxy (upstream settlement_proxy): the volume-weighted close of the one-minute bars whose
    minute
    overlaps [window start, window end); with no volume there, the last close of a bar starting before the
    window end; None when no bar precedes the window end.
    Bars: start (UTC ns), close (ticks), volume."""
    w0, w1 = window
    lo = w0 - w0 % NS_PER_MINUTE
    with localcontext(DECIMAL_CONTEXT):
        num, den, used = D(0), 0, 0
        last: tuple[int, Decimal] | None = None
        for start, close_ticks, volume in zip(starts_ns, closes, volumes, strict=True):
            if start >= w1:
                continue
            close = tick_price(root, close_ticks)
            if last is None or start > last[0]:
                last = (start, close)
            if start >= lo and volume > 0:
                num += close * int(volume)
                den += int(volume)
                used += 1
        if den > 0:
            return SettlementProxy(num / den, "vwap_close", used)
    if last is None:
        return None
    return SettlementProxy(last[1], "last_close_before_window_end", 0)


# --------------------------------------------------------------- the band ----
@dataclass(frozen=True)
class Band:
    """Price Limit% up and down (fractions; None = no limit on that side) around the prior settlement."""

    root: str
    kind: LimitKind
    up: Decimal | None
    down: Decimal | None
    reading: str
    period: LimitPeriod | None
    up_amount: Decimal | None = None  # the limit in vendor price units
    down_amount: Decimal | None = None


EQUITY_REGIMES = ("eth", "rth7", "rth20")


def equity_regime(period: LimitPeriod, t: time) -> str:
    """Upstream _equity_sides by CT clock time: 7 % down only from 08:30 to rth_7pct_until, 20 % down only
    from then
    to 15:00, eth_pct up and down otherwise."""
    if period.rth_7pct_until is None:
        raise ValueError("not an equity period")
    if time(8, 30) <= t < period.rth_7pct_until:
        return "rth7"
    if period.rth_7pct_until <= t < time(15, 0):
        return "rth20"
    return "eth"


def _equity_sides(period: LimitPeriod, regime: str) -> tuple[Decimal | None, Decimal | None, str]:
    if period.eth_pct is None:
        raise ValueError("not an equity period")
    if regime == "rth7":
        return None, D("0.07"), "RTH: 7% down only"
    if regime == "rth20":
        return None, D("0.20"), "RTH late: 20% down only"
    eth = D(period.eth_pct) / 100
    return eth, eth, f"ETH: {period.eth_pct}% up and down"


def ct_time(ts_ns: int) -> time:
    return ns_to_utc(ts_ns).astimezone(CT).time()


def limit_band(
    root: str,
    trade_date: date,
    ts_ns: int,
    prior_settlement: Decimal,
    reading: DcbReading = DCB_READING_DEFAULT,
) -> Band:
    """Upstream limit_band at UTC instant ts_ns (the equity side depends on the CT clock time)."""
    regime = None
    if limit_kind(root) is LimitKind.EQUITY_BANDS:
        regime = equity_regime(limit_period(root, trade_date), ct_time(ts_ns))
    return band_for(root, trade_date, regime, prior_settlement, reading)


def band_for(
    root: str, trade_date: date, regime: str | None, prior_settlement: Decimal,
    reading: DcbReading = DCB_READING_DEFAULT,
) -> Band:  # fmt: skip
    """limit_band with the equity regime given (None for every other kind)."""
    s = prior_settlement
    if s <= 0:
        raise ValueError(f"{root}: prior settlement {s} must be positive")
    if limit_kind(root) is LimitKind.DCB_ONLY and reading is DcbReading.NO_LOCK_LIMIT:
        return Band(root, LimitKind.DCB_ONLY, None, None, "DCB only: no lock limit", None)
    period = limit_period(root, trade_date)
    factor = _instrument(root).vendor_price_factor
    with localcontext(DECIMAL_CONTEXT):
        if period.kind is LimitKind.HARD_DAILY:
            if period.amount is None:
                raise ValueError(f"{root}: hard period without an amount")
            amt = period.amount * factor
            return Band(root, period.kind, amt / s, amt / s, f"hard limit {period.amount}", period, amt, amt)
        if period.kind is LimitKind.EQUITY_BANDS:
            up, down, why = _equity_sides(period, regime or "eth")
            return Band(root, period.kind, up, down, why, period,
                        None if up is None else s * up, None if down is None else s * down)  # fmt: skip
        if period.dcb_value is None:
            raise ValueError(f"{root}: DCB period without a value")
        amt = s * period.dcb_value / 100 if period.dcb_kind == "pct" else period.dcb_value * factor
        why = f"DCB variant {period.dcb_value} {period.dcb_kind} as a limit (alternative reading)"
        return Band(root, period.kind, amt / s, amt / s, why, period, amt, amt)


@dataclass(frozen=True)
class StopLevels:
    upper: Decimal | None  # stop trading at or above
    lower: Decimal | None  # stop trading at or below
    band: Band


def stop_levels(band: Band, prior_settlement: Decimal, buffer: Decimal = TOPSTEP_BUFFER) -> StopLevels:
    """S + amount - buffer x S above, S - amount + buffer x S below (upstream stop_levels; buffer upstream's
    2 %)."""
    s = prior_settlement
    with localcontext(DECIMAL_CONTEXT):
        cushion = s * buffer
        upper = None if band.up_amount is None else s + band.up_amount - cushion
        lower = None if band.down_amount is None else s - band.down_amount + cushion
    return StopLevels(upper, lower, band)


def limit_prices(band: Band, prior_settlement: Decimal) -> tuple[Decimal | None, Decimal | None]:
    """The limit prices themselves, S + up amount and S - down amount, for the lock rule."""
    s = prior_settlement
    with localcontext(DECIMAL_CONTEXT):
        up = None if band.up_amount is None else s + band.up_amount
        down = None if band.down_amount is None else s - band.down_amount
    return up, down


def beyond_stop(root: str, price_ticks: int, levels: StopLevels) -> str | None:
    """"above" / "below" when the price is at or beyond a stop level, else None ("above" first)."""
    p = tick_price(root, price_ticks)
    if levels.upper is not None and p >= levels.upper:
        return "above"
    if levels.lower is not None and p <= levels.lower:
        return "below"
    return None


@dataclass(frozen=True)
class Refusal:
    reason: str
    detail: str


def check_price_limit_entry(
    root: str, trade_date: date, ts_ns: int, prior_settlement: Decimal, price_ticks: int,
    reading: DcbReading = DCB_READING_DEFAULT, buffer: Decimal = TOPSTEP_BUFFER,
) -> Refusal | None:  # fmt: skip
    """price-limit: no entry while the price is at or beyond either stop level."""
    band = limit_band(root, trade_date, ts_ns, prior_settlement, reading)
    levels = stop_levels(band, prior_settlement, buffer)
    side = beyond_stop(root, price_ticks, levels)
    if side is None:
        return None
    level = levels.upper if side == "above" else levels.lower
    return Refusal("price_limit_zone_no_entry", f"{root} {tick_price(root, price_ticks)} is {side} the stop "
                   f"level ({level}); {band.reading}")  # fmt: skip


@dataclass(frozen=True)
class PriceLimitExit:
    """The forced price-limit exit: fills at the next bar's open, subject to the locked-market rule."""

    root: str
    side: str
    quantity: int
    fill: str = "next_bar_open"


def required_price_limit_exit(
    root: str, trade_date: date, ts_ns: int, position: int, prior_settlement: Decimal, price_ticks: int,
    reading: DcbReading = DCB_READING_DEFAULT, buffer: Decimal = TOPSTEP_BUFFER,
) -> PriceLimitExit | None:  # fmt: skip
    if position == 0:
        return None
    band = limit_band(root, trade_date, ts_ns, prior_settlement, reading)
    if beyond_stop(root, price_ticks, stop_levels(band, prior_settlement, buffer)) is None:
        return None
    return PriceLimitExit(root, "sell" if position > 0 else "buy", abs(position))


# --------------------------------------------------------- locked markets ----
@dataclass(frozen=True)
class Bar:
    open: int  # ticks
    high: int
    low: int
    close: int


@dataclass(frozen=True)
class LockedFill:
    index: int | None  # the fill bar's index; None = still locked at the end
    price: int | None  # ticks
    flagged: bool  # at least one bar skipped as locked
    bars_waited: int


def locked_exit_fill(
    root: str, side: str, limit_up: Decimal | None, limit_down: Decimal | None, bars: Sequence[Bar]
) -> LockedFill:
    """R-08 (upstream locked_exit_fill, tolerance 0). bars[0] would fill the exit at its open. A sell is
    blocked
    while
    the bar's high <= limit_down, a buy while its low >= limit_up; the exit fills at the open of the first bar
    that trades through. A side without a limit cannot lock."""
    if side not in ("buy", "sell"):
        raise ValueError(f"side {side!r} is not buy or sell")
    if not bars:
        return LockedFill(None, None, False, 0)
    limit = limit_down if side == "sell" else limit_up
    if limit is None:
        return LockedFill(0, bars[0].open, False, 0)
    for i, bar in enumerate(bars):
        if not is_locked(root, side, bar.high, bar.low, limit):
            return LockedFill(i, bar.open, i > 0, i)
    return LockedFill(None, None, True, len(bars))


def is_locked(root: str, side: str, high: int, low: int, limit: Decimal) -> bool:
    if side == "sell":
        return tick_price(root, high) <= limit
    return tick_price(root, low) >= limit


__all__ = [
    "DCB_READING_DEFAULT", "HARD_LIMIT_PRODUCTS", "LIMITS", "TABLE_WINDOW", "TOPSTEP_BUFFER", "Band", "Bar",
    "DcbReading", "LimitKind", "LimitPending", "LimitPeriod", "LockedFill", "PriceLimitExit", "Refusal",
    "SettlementProxy", "SettlementWindowUnavailable", "StopLevels", "Window", "band_for", "beyond_stop",
    "check_price_limit_entry", "equity_regime", "limit_band", "limit_kind", "limit_period", "limit_prices",
    "locked_exit_fill", "previous_trade_date", "required_price_limit_exit", "settlement_proxy",
    "settlement_window_ct", "settlement_window_ns", "stop_levels", "tick_price", "ticks_at_or_above",
    "ticks_at_or_below",
]  # fmt: skip
