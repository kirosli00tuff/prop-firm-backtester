"""Synthetic DayPaths for the prop rule-engine tests (no market data: hand-written or seeded random
minutes)."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pfb.prop.types import DayPaths

REPO = Path(__file__).resolve().parents[2]
FIRMS_DIR = REPO / "config" / "firms"
TOPSTEP_STD = FIRMS_DIR / "topstep" / "combine_xfa_standard_50k.toml"
TOPSTEP_CONS = FIRMS_DIR / "topstep" / "combine_xfa_consistency_50k.toml"
FIRST_DATE = dt.date(2024, 1, 2)

Minute = tuple[int, int, int]  # (low, high, close) cents relative to the day's open


def weekdays(n: int, start: dt.date = FIRST_DATE) -> np.ndarray:
    """n consecutive weekdays as int yyyymmdd."""
    out: list[int] = []
    day = start
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day.year * 10000 + day.month * 100 + day.day)
        day += dt.timedelta(days=1)
    return np.asarray(out, dtype=np.int32)


def make_days(
    minutes: Sequence[Sequence[Minute]],
    sides: Sequence[int] | None = None,
    max_contracts: Sequence[int] | None = None,
    dates: np.ndarray | None = None,
) -> DayPaths:
    """DayPaths from per-day minute lists; a day given as [] gets one zero row and 0 sides."""
    rows: list[Minute] = []
    offsets = [0]
    auto_sides: list[int] = []
    for day in minutes:
        rows.extend(day if day else [(0, 0, 0)])
        offsets.append(len(rows))
        auto_sides.append(2 if day else 0)
    n = len(minutes)
    arr = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
    return DayPaths(
        trade_dates=weekdays(n) if dates is None else dates,
        offsets=np.asarray(offsets, dtype=np.int64),
        low=arr[:, 0].copy(),
        high=arr[:, 1].copy(),
        close=arr[:, 2].copy(),
        fill_sides=np.zeros(len(rows), dtype=np.int64),
        sides=np.asarray(auto_sides if sides is None else sides, dtype=np.int64),
        max_contracts=np.asarray([1] * n if max_contracts is None else max_contracts, dtype=np.int64),
        source=("synthetic",),
    )


def closes(nets_cents: Sequence[int]) -> DayPaths:
    """One-minute days with the given close P&L (low = min(0, close), high = max(0, close))."""
    return make_days([[(min(0, c), max(0, c), c)] for c in nets_cents])


def seq_all(days: DayPaths, pad: int = 0) -> np.ndarray:
    """The days in order, then ``pad`` entries of -1."""
    return np.concatenate([np.arange(days.n_days), np.full(pad, -1)]).astype(np.int64)


def random_days(
    rng: np.random.Generator,
    n_days: int,
    scale: int,
    drift: int = 0,
    max_minutes: int = 6,
    p_flat: float = 0.1,
    p_big: float = 0.03,
    max_minis: int = 2,
) -> DayPaths:
    """Seeded random minute paths: random-walk closes, wicks below and above, some untraded days and a few
    days above ``max_minis`` contracts (to reach the size rules)."""
    rows: list[Minute] = []
    offsets = [0]
    sides = np.zeros(n_days, dtype=np.int64)
    for d in range(n_days):
        if rng.random() < p_flat:
            rows.append((0, 0, 0))
            offsets.append(len(rows))
            continue
        sides[d] = int(rng.integers(1, 20))
        n_min = int(rng.integers(1, max_minutes + 1))
        steps = rng.normal(drift / n_min, scale / np.sqrt(n_min), n_min).round().astype(np.int64)
        prev = 0
        for c in np.cumsum(steps):
            c = int(c)
            wick_lo, wick_hi = (int(x) for x in np.abs(rng.normal(0, scale / 4, 2)).round())
            rows.append((min(prev, c) - wick_lo, max(prev, c) + wick_hi, c))
            prev = c
        offsets.append(len(rows))
    arr = np.asarray(rows, dtype=np.int64)
    max_c = np.where(rng.random(n_days) < p_big, max_minis + 3, rng.integers(1, max_minis + 1, n_days))
    return DayPaths(
        trade_dates=weekdays(n_days),
        offsets=np.asarray(offsets, dtype=np.int64),
        low=arr[:, 0].copy(),
        high=arr[:, 1].copy(),
        close=arr[:, 2].copy(),
        fill_sides=np.zeros(arr.shape[0], dtype=np.int64),
        sides=sides,
        max_contracts=max_c.astype(np.int64),
        source=("synthetic",),
    )


def random_seqs(
    rng: np.random.Generator, n_days: int, n_paths: int, horizon: int, p_pad: float = 0.2
) -> np.ndarray:
    """Uniform day indices; a share of paths is cut short with -1 padding."""
    seqs = rng.integers(0, n_days, size=(n_paths, horizon)).astype(np.int64)
    cut = rng.random(n_paths) < p_pad
    starts = rng.integers(1, horizon, size=n_paths)
    for i in np.flatnonzero(cut):
        seqs[i, starts[i] :] = -1
    return seqs
