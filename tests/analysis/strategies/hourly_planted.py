"""Test strategy (synthetic data only): long at CT minutes 480, 540, ..., 840 and short at 510, 570, ..., 870
(the bar's close), exit decided 6 minutes later. On bars with a planted move two minutes after each entry
minute (tests/analysis/planting.py) it holds an edge the move creates."""

from __future__ import annotations

import numpy as np

SPEC = {
    "hypothesis_id": "test:hourly_planted",
    "root": "MNQ",
    "bar_minutes": 1,
    "session": "globex",
    "contracts": 1,
    "params": {
        "first_minute": {"default": 480, "grid": [480]},
        "last_minute": {"default": 870, "grid": [870]},
        "exit_after": {"default": 6, "grid": [6]},
    },
}


def generate(bars, params):
    minute = bars["ct_minute"].astype(np.int64)
    n = len(minute)
    lo, hi = params["first_minute"], params["last_minute"]
    window = (minute >= lo) & (minute <= hi)
    long_ = window & (minute % 60 == 0)
    short = window & (minute % 60 == 30)
    exit_ = (minute >= lo) & (minute <= hi + params["exit_after"]) & (minute % 30 == params["exit_after"])
    return {
        "entry_long": long_,
        "entry_short": short,
        "exit": exit_,
        "stop_ticks": np.zeros(n, dtype=np.int64),
        "target_ticks": np.zeros(n, dtype=np.int64),
    }
