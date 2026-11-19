"""Test strategy (synthetic data only): hourly_planted.py on even trade days only (trade_day, days since
1970-01-01, even). On bars whose planted moves exist on even trade days only (tests/analysis/suite_planting.py
even_day_planted_frame) its edge is in the day as well as the minute, so the time-of-day baseline, which
keeps the minute and draws the day, does not earn it either."""

from __future__ import annotations

import numpy as np

SPEC = {
    "hypothesis_id": "test:even_day_planted",
    "root": "MNQ",
    "bar_minutes": 1,
    "session": "globex",
    "contracts": 1,
    "params": {"exit_after": {"default": 6, "grid": [6]}},
}

FIRST_MINUTE = 480
LAST_MINUTE = 870


def generate(bars, params):
    minute = bars["ct_minute"].astype(np.int64)
    even = bars["trade_day"].astype(np.int64) % 2 == 0
    n = len(minute)
    window = even & (minute >= FIRST_MINUTE) & (minute <= LAST_MINUTE)
    exit_window = (minute >= FIRST_MINUTE) & (minute <= LAST_MINUTE + params["exit_after"])
    return {
        "entry_long": window & (minute % 60 == 0),
        "entry_short": window & (minute % 60 == 30),
        "exit": exit_window & (minute % 30 == params["exit_after"]),
        "stop_ticks": np.zeros(n, dtype=np.int64),
        "target_ticks": np.zeros(n, dtype=np.int64),
    }
