"""Test strategy (synthetic data only): long at the close of CT minute 420 + 12 j for j < slots (07:00 CT on,
one slot every 12 minutes), exit decided at the first bar 6 to 9 minutes after the slot's entry minute. On the
planted market of tests/pipeline/planted_market.py each trade holds the slot's planted shift from its entry
fill to its exit fill, and nothing else of the plant: the edge per trade is known."""

from __future__ import annotations

import numpy as np

FIRST_MINUTE = 420
SPACING = 12
EXIT_FIRST, EXIT_LAST = 6, 9

SPEC = {
    "hypothesis_id": "test:planted_slots",
    "root": "MNQ",
    "bar_minutes": 1,
    "session": "globex",
    "contracts": 1,
    "params": {
        "slots": {"default": 10, "grid": [5, 10, 40]},
    },
}


def generate(bars, params):
    minute = bars["ct_minute"].astype(np.int64)
    n = len(minute)
    rel = minute - FIRST_MINUTE
    slot, offset = rel // SPACING, rel % SPACING
    in_slots = (rel >= 0) & (slot < params["slots"])
    return {
        "entry_long": in_slots & (offset == 0),
        "entry_short": np.zeros(n, dtype=np.bool_),
        "exit": in_slots & (offset >= EXIT_FIRST) & (offset <= EXIT_LAST),
        "stop_ticks": np.zeros(n, dtype=np.int64),
        "target_ticks": np.zeros(n, dtype=np.int64),
    }
