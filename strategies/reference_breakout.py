"""Reference breakout (ADR-001 section 10): a port of an upstream template strategy at its defaults.

Range: bars opening in [O, O + 15) of the trade date (CT clock). From O + 15 to C, the first bar of the trade
date whose close is at least 4 ticks above the range high (long) or, failing that, at least 4 ticks below the
range low (short) is the day's one entry decision; it is made whether or not the entry later fills (A12). Exit
decision: any later bar of the same trade date with ts >= entry decision ts + 75 minutes (the template
times the hold from the entry DECISION) or a bar opening at or after C - 2 minutes. No stop, no target.
O = 08:30 and C = 15:00 CT are the template's fixed day-session clock for MNQ and NQ; size q_c = 1. No
parameter is tuned: every grid holds its default only.
"""

from __future__ import annotations

import numpy as np

SPEC = {
    "hypothesis_id": "engine_check:reference_breakout",
    "root": "MNQ",
    "bar_minutes": 1,
    "session": "globex",
    "contracts": 1,
    "params": {
        "day_open_minute": {"default": 510, "grid": [510]},
        "day_close_minute": {"default": 900, "grid": [900]},
        "range_minutes": {"default": 15, "grid": [15]},
        "buffer_ticks": {"default": 4, "grid": [4]},
        "hold_minutes": {"default": 75, "grid": [75]},
        "exit_before_close_minutes": {"default": 2, "grid": [2]},
    },
}

NS_PER_MINUTE = 60_000_000_000


def _day_bounds(trade_day):
    n = len(trade_day)
    starts = np.flatnonzero(trade_day[1:] != trade_day[:-1]) + 1
    first = np.zeros(1, dtype=np.int64)
    return np.concatenate((first, starts)).tolist(), np.append(starts, n).tolist()


def _entry_of_day(high, low, close, minute, params):
    """(index of the day's entry decision or -1, True if long)."""
    o = params["day_open_minute"]
    range_end = o + params["range_minutes"]
    buffer = params["buffer_ticks"]
    in_range = (minute >= o) & (minute < range_end)
    seen = np.cumsum(in_range) > 0
    hi = np.maximum.accumulate(np.where(in_range, high, high.min()))
    lo = np.minimum.accumulate(np.where(in_range, low, low.max()))
    window = seen & (minute >= range_end) & (minute < params["day_close_minute"])
    up = window & (close >= hi + buffer)
    down = window & ~up & (close <= lo - buffer)
    hits = np.flatnonzero(up | down)
    if len(hits) == 0:
        return -1, True
    first = int(hits[0])
    return first, bool(up[first])


def generate(bars, params):
    ts = bars["ts_event"]
    minute = bars["ct_minute"].astype(np.int64)
    n = len(ts)
    entry_long = np.zeros(n, dtype=np.bool_)
    entry_short = np.zeros(n, dtype=np.bool_)
    exit_ = np.zeros(n, dtype=np.bool_)
    hold_ns = params["hold_minutes"] * NS_PER_MINUTE
    late = params["day_close_minute"] - params["exit_before_close_minutes"]
    if n == 0:
        starts, ends = [], []
    else:
        starts, ends = _day_bounds(bars["trade_day"])
    for s, e in zip(starts, ends, strict=True):
        m = minute[s:e]
        k, is_long = _entry_of_day(bars["high"][s:e], bars["low"][s:e], bars["close"][s:e], m, params)
        if k < 0:
            continue
        entry_long[s + k] = is_long
        entry_short[s + k] = not is_long
        after = np.arange(e - s) > k
        exit_[s:e] = after & ((ts[s:e] - ts[s + k] >= hold_ns) | (m >= late))
    zeros = np.zeros(n, dtype=np.int64)
    return {
        "entry_long": entry_long,
        "entry_short": entry_short,
        "exit": exit_,
        "stop_ticks": zeros,
        "target_ticks": zeros.copy(),
    }
