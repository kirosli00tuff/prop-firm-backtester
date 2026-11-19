"""Test strategy (synthetic data only): entries and exits at bars chosen by an integer hash of the bar's
minute stamp, so it has no edge by construction and is truncation-invariant. salt picks the stream;
long_only 1 takes only longs; stop_ticks > 0 attaches that stop to every entry."""

from __future__ import annotations

import numpy as np

SPEC = {
    "hypothesis_id": "test:hash_random",
    "root": "MNQ",
    "bar_minutes": 1,
    "session": "globex",
    "contracts": 1,
    "params": {
        "salt": {"default": 0, "grid": list(range(64))},
        "entry_every": {"default": 200, "grid": [50, 200]},
        "exit_every": {"default": 30, "grid": [30]},
        "long_only": {"default": 0, "grid": [0, 1]},
        "stop_ticks": {"default": 0, "grid": [0, 40]},
    },
}

NS_PER_MINUTE = 60_000_000_000
MULT_A = 2654435761
MULT_B = 40503
MOD = 1000003


def generate(bars, params):
    stamp = bars["ts_event"] // NS_PER_MINUTE
    n = len(stamp)
    h = (stamp * MULT_A + params["salt"] * MULT_B + 12345) % MOD
    g = (stamp * MULT_B + params["salt"] * MULT_A + 777) % MOD
    entry = (h % params["entry_every"]) == 0
    is_long = ((h // 7) % 2 == 0) | (params["long_only"] == 1)
    exit_ = (g % params["exit_every"]) == 0
    return {
        "entry_long": entry & is_long,
        "entry_short": entry & ~is_long,
        "exit": exit_,
        "stop_ticks": np.full(n, params["stop_ticks"], dtype=np.int64),
        "target_ticks": np.zeros(n, dtype=np.int64),
    }
