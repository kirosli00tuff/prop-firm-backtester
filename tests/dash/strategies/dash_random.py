"""Test strategy for the dashboard tests (synthetic data only): entries and exits at bars chosen by an integer
hash of the bar's minute stamp, so it has no edge by construction. One grid parameter (salt) with three
values, so a synthetic neighbourhood run has exactly two neighbours."""

from __future__ import annotations

import numpy as np

SPEC = {
    "hypothesis_id": "test:dash_random",
    "root": "MNQ",
    "bar_minutes": 1,
    "session": "globex",
    "contracts": 1,
    "params": {
        "salt": {"default": 1, "grid": [0, 1, 2]},
    },
}

NS_PER_MINUTE = 60_000_000_000
MULT_A = 2654435761
MULT_B = 40503
MOD = 1000003
ENTRY_EVERY = 200
EXIT_EVERY = 30


def generate(bars, params):
    stamp = bars["ts_event"] // NS_PER_MINUTE
    n = len(stamp)
    h = (stamp * MULT_A + params["salt"] * MULT_B + 12345) % MOD
    g = (stamp * MULT_B + params["salt"] * MULT_A + 777) % MOD
    entry = (h % ENTRY_EVERY) == 0
    is_long = (h // 7) % 2 == 0
    exit_ = (g % EXIT_EVERY) == 0
    return {
        "entry_long": entry & is_long,
        "entry_short": entry & ~is_long,
        "exit": exit_,
        "stop_ticks": np.zeros(n, dtype=np.int64),
        "target_ticks": np.zeros(n, dtype=np.int64),
    }
