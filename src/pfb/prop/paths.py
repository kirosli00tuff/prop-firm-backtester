"""Path generators and the cost stress (docs/PROP_MC.md section 3).

A sequence is an int64 array [n_paths, horizon] of day indices into a DayPaths; -1 pads past the end of
history (random_start only). The generators draw from the numpy Generator they are given (PCG64 from the
seed in mc.py), so a seed fixes every sequence.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pfb.prop.days import check_day_paths, fill_minutes_known
from pfb.prop.types import DayPaths

PAD = -1


def _check_shape(n_days: int, horizon: int, n_paths: int) -> None:
    if n_days < 1 or horizon < 1 or n_paths < 1:
        raise ValueError(f"n_days, horizon and n_paths must be >= 1, got {n_days}, {horizon}, {n_paths}")


def block_bootstrap(n_days: int, horizon: int, n_paths: int, mean_block: int,
                    rng: np.random.Generator) -> np.ndarray:  # fmt: skip
    """Stationary bootstrap (Politis-Romano) with integer mean block length.

    Day 0 of a path is uniform; each later day jumps to a new uniform day with probability exactly
    1/mean_block (an integer draw equal to 0), otherwise takes the next day, wrapping from the last day to the
    first. Block lengths are geometric with mean mean_block; mean_block 1 gives iid days.
    """
    _check_shape(n_days, horizon, n_paths)
    if not isinstance(mean_block, int | np.integer) or mean_block < 1:
        raise ValueError(f"mean_block must be an integer >= 1, got {mean_block!r}")
    seqs = np.empty((n_paths, horizon), dtype=np.int64)
    seqs[:, 0] = rng.integers(0, n_days, size=n_paths)
    for j in range(1, horizon):
        jump = rng.integers(0, mean_block, size=n_paths) == 0
        fresh = rng.integers(0, n_days, size=n_paths)
        nxt = seqs[:, j - 1] + 1
        nxt[nxt == n_days] = 0
        seqs[:, j] = np.where(jump, fresh, nxt)
    return seqs


def random_start(n_days: int, horizon: int, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """Historical order from a uniform start day.

    Starts are drawn only where `horizon` days of history remain (uniform on 0 .. n_days - horizon), so no
    sequence is padded. When the history is shorter than the horizon, starts are uniform over every day and
    positions past the last day are -1 (PAD); run_mc counts such paths and marks the outputs they bias
    (docs/PROP_MC.md, the rules added after the first review).
    """
    _check_shape(n_days, horizon, n_paths)
    last_start = n_days - horizon if n_days >= horizon else n_days - 1
    start = rng.integers(0, last_start + 1, size=n_paths)
    seqs = start[:, None] + np.arange(horizon, dtype=np.int64)[None, :]
    seqs[seqs >= n_days] = PAD
    return seqs.astype(np.int64)


def padded_paths(seqs: np.ndarray) -> np.ndarray:
    """bool[n_paths]: the sequence contains at least one PAD."""
    return np.any(seqs == PAD, axis=1)


def stress(days: DayPaths, k_ticks: int, tick_cents: int) -> DayPaths:
    """A new DayPaths, each minute lowered by k_ticks * tick_cents per contract side filled so far that day.

    When fill minutes are known (fill_sides adds up to sides every day) the cost lands at each fill minute;
    otherwise the whole day's cost lands at the first minute (conservative). Within a minute low and close
    carry the cost through that minute, high only the cost before it (conservative for a trailing peak).
    """
    if not isinstance(k_ticks, int | np.integer) or k_ticks < 0:
        raise ValueError(f"k_ticks must be an integer >= 0, got {k_ticks!r}")
    if not isinstance(tick_cents, int | np.integer) or tick_cents < 1:
        raise ValueError(f"tick_cents must be an integer >= 1, got {tick_cents!r}")
    check_day_paths(days)
    if k_ticks == 0:
        return days
    if fill_minutes_known(days):
        fills = days.fill_sides
    else:
        fills = np.zeros_like(days.fill_sides)
        fills[days.offsets[:-1]] = days.sides
    starts = days.offsets[:-1].astype(np.intp)
    total = np.cumsum(fills)
    day_base = np.repeat(total[starts] - fills[starts], np.diff(days.offsets))
    through = total - day_base  # sides filled from the day's start through this minute
    before = through - fills
    unit = int(k_ticks) * int(tick_cents)
    return replace(days, low=days.low - unit * through, high=days.high - unit * before,
                   close=days.close - unit * through)  # fmt: skip
