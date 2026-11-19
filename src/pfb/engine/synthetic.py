"""Seeded synthetic bars built in code (ADR-001 section 9): gateway-shaped frames, no market data.

Bar times come from the root's calendar (pfb.instruments.sessions.open_intervals), so sessions, flat-by and
trade dates behave as on real data; prices are a seeded random walk in ticks. A synthetic frame carries one
era.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from pfb.data.gateway import OUTPUT_COLUMNS, ct_minute_expr
from pfb.engine.strategy import Bars, make_bars
from pfb.instruments.sessions import open_intervals

NS_PER_MINUTE = 60_000_000_000
SYNTHETIC_ERA = "synthetic"
SYNTHETIC_INSTRUMENT_ID = 1
DEFAULT_SYNTHETIC_FIRST = date(2025, 4, 1)
DEFAULT_SYNTHETIC_LAST = date(2025, 4, 30)
BASE_PRICE_TICKS = 80_000
STEP_TICKS = 4  # close-to-close steps drawn from -4..4 ticks
WICK_TICKS = 3  # wicks drawn from 0..3 ticks
MISSING_PER_MILLE = 5  # about 0.5 % of minutes are absent (nothing is forward filled)


def _minutes(
    starts_ns: np.ndarray, ends_ns: np.ndarray, trade_days: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if not (len(starts_ns) == len(ends_ns) == len(trade_days)):
        raise ValueError("starts, ends and trade_days differ in length")
    counts = (np.asarray(ends_ns, dtype=np.int64) - np.asarray(starts_ns, dtype=np.int64)) // NS_PER_MINUTE
    if np.any(counts < 0):
        raise ValueError("an interval ends before it starts")
    ts = np.concatenate(
        [
            np.arange(int(s), int(s) + int(c) * NS_PER_MINUTE, NS_PER_MINUTE, dtype=np.int64)
            for s, c in zip(starts_ns, counts, strict=True)
        ]
        or [np.zeros(0, dtype=np.int64)]
    )
    days = np.repeat(np.asarray(trade_days, dtype=np.int32), counts)
    return ts, days


def _prices(n: int, rng: np.random.Generator, base: int) -> dict[str, np.ndarray]:
    close = base + np.cumsum(rng.integers(-STEP_TICKS, STEP_TICKS + 1, n)).astype(np.int64)
    open_ = np.empty(n, dtype=np.int64)
    if n:
        open_[0] = base
        open_[1:] = close[:-1] + rng.integers(-1, 2, n - 1)
    high = np.maximum(open_, close) + rng.integers(0, WICK_TICKS + 1, n)
    low = np.minimum(open_, close) - rng.integers(0, WICK_TICKS + 1, n)
    return {"open": open_, "high": high, "low": low, "close": close}


def frame_from_intervals(
    root: str,
    starts_ns: np.ndarray,
    ends_ns: np.ndarray,
    trade_days: np.ndarray,
    *,
    seed: int,
    base_price_ticks: int = BASE_PRICE_TICKS,
) -> pl.DataFrame:
    """One bar per minute of each [start, end) interval (minus a seeded 0.5 % of absent minutes)."""
    rng = np.random.default_rng(seed)
    ts, days = _minutes(starts_ns, ends_ns, trade_days)
    keep = rng.integers(0, 1000, len(ts)) >= MISSING_PER_MILLE
    ts, days = ts[keep], days[keep]
    if len(ts) > 1 and not bool(np.all(np.diff(ts) > 0)):
        raise ValueError("synthetic intervals overlap or are not in time order")
    n = len(ts)
    prices = _prices(n, rng, base_price_ticks)
    gaps = np.zeros(n, dtype=np.int64)
    gaps[1:] = np.diff(ts) // NS_PER_MINUTE - 1
    false = pl.repeat(False, n, eager=True)
    frame = pl.DataFrame(
        {
            "ts_event": ts,
            **{k: pl.Series(k, v, dtype=pl.Int64) for k, v in prices.items()},
            "volume": rng.integers(1, 500, n).astype(np.int64),
            "trade_date": pl.Series(days, dtype=pl.Int32).cast(pl.Date),
            "is_roll_session": false,
            "in_flatten_window": false,
            "in_no_new_positions_window": false,
            "in_scheduled_closure": false,
            "vendor_degraded_day": false,
            "early_halt_ct": pl.repeat("", n, eager=True, dtype=pl.String),
            "gap_before_minutes": gaps,
            "instrument_id": pl.repeat(SYNTHETIC_INSTRUMENT_ID, n, eager=True, dtype=pl.Int64),
            "raw_symbol": pl.repeat(f"{root}SYN", n, eager=True, dtype=pl.String),
            "era": pl.repeat(SYNTHETIC_ERA, n, eager=True, dtype=pl.String),
        }
    )
    return frame.with_columns(ct_minute_expr()).select(OUTPUT_COLUMNS)


def synthetic_frame(root: str, first: date, last: date, *, seed: int) -> pl.DataFrame:
    """Synthetic bars on the root's calendar over first..last (raises outside calendar coverage)."""
    starts, ends, days = open_intervals(root, first, last)
    return frame_from_intervals(root, np.asarray(starts), np.asarray(ends), np.asarray(days), seed=seed)


def synthetic_bars(root: str, *, seed: int, first: date | None = None, last: date | None = None) -> Bars:
    """The strategy view of synthetic_frame (default window 2025-04-01..2025-04-30), for `pfb check`."""
    frame = synthetic_frame(root, first or DEFAULT_SYNTHETIC_FIRST, last or DEFAULT_SYNTHETIC_LAST, seed=seed)
    return make_bars(frame)


__all__ = [
    "DEFAULT_SYNTHETIC_FIRST",
    "DEFAULT_SYNTHETIC_LAST",
    "SYNTHETIC_ERA",
    "frame_from_intervals",
    "synthetic_bars",
    "synthetic_frame",
]
