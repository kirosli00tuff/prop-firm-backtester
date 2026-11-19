"""Helpers for the suite tests: synthetic frames with planted moves or drift, swapped in for
pfb.engine.run.synthetic_frame. Synthetic data only; nothing here reads a bar store."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import pfb.engine.run as run_module
from pfb.engine.synthetic import synthetic_frame

HERE = Path(__file__).resolve().parent
STRATEGIES = HERE / "strategies"
HASH_RANDOM = STRATEGIES / "hash_random.py"
HOURLY_PLANTED = STRATEGIES / "hourly_planted.py"
SESSION_HOLD = STRATEGIES / "session_hold.py"
TOD_ZERO_EDGE = STRATEGIES / "tod_zero_edge.py"
EVEN_DAY_PLANTED = STRATEGIES / "even_day_planted.py"
REFERENCE = HERE.parents[1] / "strategies" / "reference_breakout.py"
PRICE_COLUMNS = ("open", "high", "low", "close")
PLANT_TICKS = 40
PLANT_LAG = 2  # the move starts at the bar two minutes after the entry decision minute


def planted_shift(minutes: np.ndarray, ticks: int = PLANT_TICKS) -> np.ndarray:
    """Cumulative price shift by CT minute: +ticks from 482, 542, ..., 842; -ticks from 512, ..., 872; zero
    before 480 and after 872 (every move is undone within the day)."""
    steps = np.zeros(1440, dtype=np.int64)
    for m in range(480, 841, 60):
        steps[m + PLANT_LAG] += ticks
    for m in range(510, 871, 60):
        steps[m + PLANT_LAG] -= ticks
    return np.cumsum(steps)[minutes]


def shifted(frame: pl.DataFrame, shift: np.ndarray) -> pl.DataFrame:
    return frame.with_columns(pl.col(c) + pl.Series(shift, dtype=pl.Int64) for c in PRICE_COLUMNS)


def planted_frame(root: str, first: date, last: date, *, seed: int) -> pl.DataFrame:
    frame = synthetic_frame(root, first, last, seed=seed)
    return shifted(frame, planted_shift(frame["ct_minute"].to_numpy().astype(np.int64)))


def even_day_planted_frame(root: str, first: date, last: date, *, seed: int) -> pl.DataFrame:
    """planted_frame with the moves on even trade days only (days since 1970-01-01)."""
    frame = synthetic_frame(root, first, last, seed=seed)
    even = frame["trade_date"].cast(pl.Int32).to_numpy().astype(np.int64) % 2 == 0
    shift = np.where(even, planted_shift(frame["ct_minute"].to_numpy().astype(np.int64)), 0)
    return shifted(frame, shift.astype(np.int64))


def drifting_frame(root: str, first: date, last: date, *, seed: int) -> pl.DataFrame:
    """Up half a tick per bar on average (one tick every second bar), over the whole frame."""
    frame = synthetic_frame(root, first, last, seed=seed)
    return shifted(frame, np.arange(frame.height, dtype=np.int64) // 2)


def use_frame(monkeypatch: pytest.MonkeyPatch, maker: Callable[..., pl.DataFrame]) -> None:
    monkeypatch.setattr(run_module, "synthetic_frame", maker)
