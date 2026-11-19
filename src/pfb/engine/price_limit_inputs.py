"""The kernel's per-bar price-limit inputs (the author's research repo price-limit and R-08 as the
engine applies them), built outside the kernel with the exact rule of pfb.engine.price_limits.

For each usable bar: the stop-level thresholds at its DECISION (bar open + 60 s, the close is the price),
the lock thresholds at its OPEN (the bar an exit would fill on), and whether its trade date lacks a prior
settlement reference. The reference of trade date d is upstream's settlement proxy over the bars the run holds
on
the previous trade date of the group calendar (no extra store read); without one, entries are refused and
neither the forced exit nor the lock applies (upstream reading RR-4). A date in a pending period
(LimitPending)
and a DCB-only root leave the rule inactive (sentinels). Each trade date is classified for
the result's warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import polars as pl

from pfb.engine import price_limits as plim
from pfb.engine.kernel import NO_LOWER, NO_UPPER

EPOCH = date(1970, 1, 1)
MINUTES_PER_DAY = 1440
STATUS_APPLIED, STATUS_NO_REFERENCE, STATUS_PENDING, STATUS_DCB, STATUS_OFF = (
    "applied", "no_reference", "pending", "dcb_no_lock_limit", "off",
)  # fmt: skip


@dataclass(frozen=True)
class PriceLimitArrays:
    upper: np.ndarray  # int64: a close >= this is at or beyond the upper stop level
    lower: np.ndarray  # int64: a close <= this is at or beyond the lower stop level
    lock_up: np.ndarray  # int64: a buy is locked while the bar's low >= this
    lock_down: np.ndarray  # int64: a sell is locked while the bar's high <= this
    no_reference: np.ndarray  # bool: the bar's trade date has no prior settlement reference
    root: str
    kind: str
    buffer: Decimal
    enabled: bool
    date_status: dict[str, str] = field(default_factory=dict)  # ISO trade date -> STATUS_*

    def dates_with(self, status: str) -> int:
        return sum(1 for s in self.date_status.values() if s == status)


def _ordinal_date(day: int) -> date:
    return EPOCH + timedelta(days=int(day))


def _day_proxy(root: str, day: date, ts: np.ndarray, close: np.ndarray, vol: np.ndarray) -> Decimal | None:
    """Upstream's settlement proxy of ``day`` from that day's bars (only the bars the proxy can use are
    passed:
    those in the window's minutes and the last one before the window end, so the result is identical)."""
    try:
        w0, w1 = plim.settlement_window_ns(root, day)
    except plim.SettlementWindowUnavailable:
        return None
    before_end = ts < w1
    if not bool(before_end.any()):
        return None
    lo = w0 - w0 % plim.NS_PER_MINUTE
    keep = before_end & (ts >= lo)
    keep[int(np.flatnonzero(before_end)[np.argmax(ts[before_end])])] = True
    got = plim.settlement_proxy(root, ts[keep].tolist(), close[keep].tolist(), vol[keep].tolist(), (w0, w1))
    return None if got is None else got.value


def settlement_references(root: str, raw: pl.DataFrame, days: np.ndarray) -> dict[int, Decimal | None]:
    """Trade day ordinal -> the prior settlement proxy (None: none in the run's bars)."""
    raw_days = raw["trade_date"].cast(pl.Int32).to_numpy()
    ts = raw["ts_event"].to_numpy().astype(np.int64)
    close = raw["close"].to_numpy().astype(np.int64)
    vol = raw["volume"].to_numpy().astype(np.int64)
    out: dict[int, Decimal | None] = {}
    for d in np.unique(days).tolist():
        prev = plim.previous_trade_date(root, _ordinal_date(d))
        if prev is None:
            out[d] = None
            continue
        mask = raw_days == (prev - EPOCH).days
        out[d] = _day_proxy(root, prev, ts[mask], close[mask], vol[mask]) if bool(mask.any()) else None
    return out


def _thresholds(
    root: str, day: date, regime: str | None, s: Decimal, buffer: Decimal
) -> tuple[int, int, int, int]:
    band = plim.band_for(root, day, regime, s)
    levels = plim.stop_levels(band, s, buffer)
    up, down = plim.limit_prices(band, s)
    return (
        NO_UPPER if levels.upper is None else plim.ticks_at_or_above(root, levels.upper),
        NO_LOWER if levels.lower is None else plim.ticks_at_or_below(root, levels.lower),
        NO_UPPER if up is None else plim.ticks_at_or_above(root, up),
        NO_LOWER if down is None else plim.ticks_at_or_below(root, down),
    )


def _regimes(period: plim.LimitPeriod, minutes: np.ndarray) -> np.ndarray:
    """The equity regime of each CT minute of day (the boundaries are whole minutes)."""
    until = period.rth_7pct_until
    if until is None:
        raise ValueError("not an equity period")
    until_min = until.hour * 60 + until.minute
    out = np.full(len(minutes), "eth", dtype=object)
    out[(minutes >= 510) & (minutes < until_min)] = "rth7"
    out[(minutes >= until_min) & (minutes < 900)] = "rth20"
    return out


def _fill_day(arrays: dict[str, np.ndarray], idx: np.ndarray, root: str, day: date, s: Decimal,
              minute: np.ndarray, buffer: Decimal) -> None:  # fmt: skip
    period = plim.limit_period(root, day)
    if period.kind is not plim.LimitKind.EQUITY_BANDS:
        up, lo, lu, ld = _thresholds(root, day, None, s, buffer)
        arrays["upper"][idx], arrays["lower"][idx] = up, lo
        arrays["lock_up"][idx], arrays["lock_down"][idx] = lu, ld
        return
    decision = _regimes(period, (minute[idx] + 1) % MINUTES_PER_DAY)
    at_open = _regimes(period, minute[idx])
    table = {r: _thresholds(root, day, r, s, buffer) for r in plim.EQUITY_REGIMES}
    arrays["upper"][idx] = [table[r][0] for r in decision]
    arrays["lower"][idx] = [table[r][1] for r in decision]
    arrays["lock_up"][idx] = [table[r][2] for r in at_open]
    arrays["lock_down"][idx] = [table[r][3] for r in at_open]


def build_price_limit_arrays(
    root: str, raw: pl.DataFrame, usable: pl.DataFrame, *, buffer: Decimal = plim.TOPSTEP_BUFFER,
    enabled: bool = True,
) -> PriceLimitArrays:  # fmt: skip
    """The arrays over the usable bars. raw: every bar the run holds (the settlement proxies come from it);
    usable: the bars the kernel runs (ts_event, trade_date, ct_minute)."""
    n = usable.height
    arrays = {
        "upper": np.full(n, NO_UPPER, dtype=np.int64), "lower": np.full(n, NO_LOWER, dtype=np.int64),
        "lock_up": np.full(n, NO_UPPER, dtype=np.int64), "lock_down": np.full(n, NO_LOWER, dtype=np.int64),
    }  # fmt: skip
    no_ref = np.zeros(n, dtype=bool)
    kind = plim.limit_kind(root)
    days = usable["trade_date"].cast(pl.Int32).to_numpy()
    status: dict[str, str] = {}
    unique = np.unique(days).tolist()
    if not enabled or kind is plim.LimitKind.DCB_ONLY:
        flag = STATUS_OFF if not enabled else STATUS_DCB
        status = {_ordinal_date(d).isoformat(): flag for d in unique}
        return PriceLimitArrays(**arrays, no_reference=no_ref, root=root, kind=kind.value, buffer=buffer,
                                enabled=enabled, date_status=status)  # fmt: skip
    refs = settlement_references(root, raw, days)
    minute = usable["ct_minute"].to_numpy().astype(np.int64)
    for d in unique:
        day = _ordinal_date(d)
        idx = np.flatnonzero(days == d)
        try:
            plim.limit_period(root, day)
        except plim.LimitPending:
            status[day.isoformat()] = STATUS_PENDING
            continue
        s = refs[d]
        if s is None:
            no_ref[idx] = True
            status[day.isoformat()] = STATUS_NO_REFERENCE
            continue
        _fill_day(arrays, idx, root, day, s, minute, buffer)
        status[day.isoformat()] = STATUS_APPLIED
    return PriceLimitArrays(**arrays, no_reference=no_ref, root=root, kind=kind.value, buffer=buffer,
                            enabled=enabled, date_status=status)  # fmt: skip


__all__ = ["PriceLimitArrays", "build_price_limit_arrays", "settlement_references"]
