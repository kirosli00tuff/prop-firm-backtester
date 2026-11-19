"""Test strategy (synthetic data only): one entry per day at a fixed CT minute, side by the minute stamp's
parity, never an exit signal, so every trade closes at the session end (at a bar's close)."""

from __future__ import annotations

import numpy as np

SPEC = {
    "hypothesis_id": "test:session_hold",
    "root": "MNQ",
    "bar_minutes": 1,
    "session": "rth",
    "contracts": 1,
    "params": {
        "entry_minute": {"default": 600, "grid": [600, 900]},
    },
}

NS_PER_MINUTE = 60_000_000_000


def generate(bars, params):
    minute = bars["ct_minute"].astype(np.int64)
    n = len(minute)
    entry = minute == params["entry_minute"]
    is_long = ((bars["ts_event"] // NS_PER_MINUTE) // 1440) % 2 == 0
    return {
        "entry_long": entry & is_long,
        "entry_short": entry & ~is_long,
        "exit": np.zeros(n, dtype=np.bool_),
        "stop_ticks": np.zeros(n, dtype=np.int64),
        "target_ticks": np.zeros(n, dtype=np.int64),
    }
