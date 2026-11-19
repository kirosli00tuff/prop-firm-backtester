"""Test strategy (synthetic data only): a zero-edge time-of-day rule. Long at the CT minutes inside
08:30-14:50 that are multiples of k (with every > 1, on about one such minute-day in `every`), exit decided
five minutes later; no stop, no target. On synthetic bars it has no edge, but it trades only in regular
hours, where the book slippage is lower than overnight (with every 1 it fills every
minute-day cell)."""

from __future__ import annotations

import numpy as np

SPEC = {
    "hypothesis_id": "test:tod_zero_edge",
    "root": "MNQ",
    "bar_minutes": 1,
    "session": "globex",
    "contracts": 1,
    "params": {"k": {"default": 10, "grid": [10, 20]}, "every": {"default": 1, "grid": [1, 4]}},
}

RTH_FIRST = 510  # 08:30 CT
RTH_END = 890  # 14:50 CT, exclusive
EXIT_AFTER = 5
NS_PER_MINUTE = 60_000_000_000
MULT = 2654435761
MOD = 1000003


def generate(bars, params):
    minute = bars["ct_minute"].astype(np.int64)
    n = len(minute)
    rth = (minute >= RTH_FIRST) & (minute < RTH_END)
    # every > 1 keeps about one entry minute-day cell in `every` (an integer hash of the minute stamp), so the
    # time-of-day draws have other days to draw from; every 1 fills every cell
    picked = ((bars["ts_event"] // NS_PER_MINUTE) * MULT % MOD) % params["every"] == 0
    entry = rth & (minute % params["k"] == 0) & picked
    exit_ = rth & (minute % params["k"] == EXIT_AFTER)
    zero = np.zeros(n, dtype=np.int64)
    return {
        "entry_long": entry,
        "entry_short": np.zeros(n, dtype=bool),
        "exit": exit_,
        "stop_ticks": zero,
        "target_ticks": zero,
    }
