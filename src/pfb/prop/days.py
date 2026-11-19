"""DayPaths from a run's artifacts, and synthetic constructors (docs/PROP_MC.md section 3).

A run directory holds equity_minute.parquet (eq_* cumulative equity in cents since the run start, one row per
usable bar), daily.parquet (one row per trade date) and trades.parquet (entry_bar and exit_bar index the
equity_minute rows). A DayPaths holds each day's minutes relative to the day's opening equity, which is the
previous minute's eq_close (0 before the first minute), so the day nets add up to the run's final equity.
Every column is checked; a mismatch refuses with DayPathsError.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from pfb.prop.types import DayPaths

EQUITY_FILE = "equity_minute.parquet"
DAILY_FILE = "daily.parquet"
TRADES_FILE = "trades.parquet"
EQUITY_COLUMNS = ("ts_event", "trade_date", "position", "eq_close", "eq_low", "eq_high")
DAILY_COLUMNS = ("trade_date", "eq_close")
TRADE_COLUMNS = ("contracts", "entry_ts", "exit_ts", "entry_bar", "exit_bar", "trade_date")
DEFAULT_SIDES_PER_DAY = 2  # synthetic days: one round trip of one contract
SYNTHETIC_FIRST_DAY = "2000-01-03"  # a Monday; synthetic trade dates are consecutive business days


class DayPathsError(ValueError):
    """A run's artifacts or a synthetic input do not make a valid DayPaths."""


# ------------------------------------------------------------------ checks ----
def check_day_paths(days: DayPaths) -> DayPaths:
    """Raise DayPathsError unless every array has the documented shape, dtype and ordering."""
    n = days.n_days
    if n < 1:
        raise DayPathsError("a DayPaths needs at least one day")
    _dtype(days.trade_dates, "trade_dates", np.int32, (n,))
    _dtype(days.offsets, "offsets", np.int64, (n + 1,))
    m = int(days.offsets[-1])
    for name in ("low", "high", "close", "fill_sides"):
        _dtype(getattr(days, name), name, np.int64, (m,))
    for name in ("sides", "max_contracts"):
        _dtype(getattr(days, name), name, np.int64, (n,))
    if bool(np.any(np.diff(days.trade_dates) <= 0)):
        raise DayPathsError("trade_dates must be strictly increasing (the sequence clock reads the gaps)")
    if int(days.offsets[0]) != 0 or bool(np.any(np.diff(days.offsets) < 1)):
        raise DayPathsError("offsets must start at 0 and give every day at least one minute")
    if bool(np.any(days.low > days.close)) or bool(np.any(days.high < days.close)):
        raise DayPathsError("a minute with low > close or high < close")
    if any(bool(np.any(a < 0)) for a in (days.sides, days.fill_sides, days.max_contracts)):
        raise DayPathsError("negative sides, fill_sides or max_contracts")
    return days


def fill_minutes_known(days: DayPaths) -> bool:
    """True when fill_sides adds up to sides on every day (fill minutes known)."""
    per_day = np.add.reduceat(days.fill_sides, days.offsets[:-1].astype(np.intp))
    return bool(np.array_equal(per_day, days.sides))


def _dtype(arr: np.ndarray, name: str, dtype: type, shape: tuple[int, ...]) -> None:
    if not isinstance(arr, np.ndarray) or arr.dtype != np.dtype(dtype) or arr.shape != shape:
        got = (getattr(arr, "dtype", type(arr)), getattr(arr, "shape", None))
        raise DayPathsError(f"{name} must be {np.dtype(dtype)}{shape}, got {got}")


# -------------------------------------------------------------- from a run ----
def day_paths_from_run(run_dir: Path, *, contracts_per_mini: int = 1) -> DayPaths:
    """Read a run's three frames and build its DayPaths (source = the run directory's name).

    contracts_per_mini: 1 for a mini or full-size contract, 10 for a micro; max_contracts is the day's largest
    exposure divided by it, rounded up (conservative against a contract cap). A minute's exposure is the
    larger of |position| at its close and the contracts of any trade entered or exited in it, so a round trip
    inside one minute shows its size.
    """
    if contracts_per_mini < 1:
        raise DayPathsError("contracts_per_mini must be >= 1")
    run_dir = Path(run_dir)
    equity = _read(run_dir / EQUITY_FILE, EQUITY_COLUMNS)
    daily = _read(run_dir / DAILY_FILE, DAILY_COLUMNS)
    trades = _read(run_dir / TRADES_FILE, TRADE_COLUMNS)
    ts = equity["ts_event"].to_numpy()
    if ts.shape[0] == 0:
        raise DayPathsError(f"{EQUITY_FILE} has no rows")
    if bool(np.any(np.diff(ts) <= 0)):
        raise DayPathsError(f"{EQUITY_FILE}: ts_event is not strictly increasing")
    minute_dates = _yyyymmdd(equity["trade_date"], EQUITY_FILE)
    day_dates = _yyyymmdd(daily["trade_date"], DAILY_FILE)
    _check_dates(minute_dates, day_dates)
    fill = _fill_sides(trades, ts, minute_dates)
    position = minute_exposure(equity["position"].to_numpy().astype(np.int64), trades)
    built = _assemble(
        day_dates, minute_dates,
        equity["eq_low"].to_numpy().astype(np.int64), equity["eq_high"].to_numpy().astype(np.int64),
        equity["eq_close"].to_numpy().astype(np.int64), fill, position, contracts_per_mini,
    )  # fmt: skip
    _check_daily_close(built, daily["eq_close"].to_numpy().astype(np.int64), day_dates, minute_dates)
    return check_day_paths(
        DayPaths(**{**built, "source": (run_dir.name,)})  # type: ignore[arg-type]
    )


def _read(path: Path, columns: tuple[str, ...]) -> pl.DataFrame:
    if not path.is_file():
        raise DayPathsError(f"missing run artifact {path.name}")
    present = pl.read_parquet_schema(path)
    missing = [c for c in columns if c not in present]
    if missing:
        raise DayPathsError(f"{path.name} lacks column(s) {missing}")
    frame = pl.read_parquet(path, columns=list(columns))
    for name in columns:
        if name == "trade_date":
            continue
        if frame.schema[name] != pl.Int64:
            raise DayPathsError(f"{path.name}: column {name} must be Int64, got {frame.schema[name]}")
        if frame[name].null_count():
            raise DayPathsError(f"{path.name}: column {name} has nulls")
    return frame


def _yyyymmdd(dates: pl.Series, where: str) -> np.ndarray:
    if dates.dtype != pl.Date or dates.null_count():
        raise DayPathsError(f"{where}: trade_date must be a Date column without nulls, got {dates.dtype}")
    y, m, d = (dates.dt.year().cast(pl.Int32), dates.dt.month().cast(pl.Int32), dates.dt.day().cast(pl.Int32))
    out = (y * 10000 + m * 100 + d).to_numpy()
    return np.asarray(out, dtype=np.int32)


def _check_dates(minute_dates: np.ndarray, day_dates: np.ndarray) -> None:
    if bool(np.any(np.diff(minute_dates) < 0)):
        raise DayPathsError(f"{EQUITY_FILE}: trade_date goes backwards")
    if day_dates.shape[0] == 0 or bool(np.any(np.diff(day_dates) <= 0)):
        raise DayPathsError(f"{DAILY_FILE}: trade dates must be non-empty and strictly increasing")
    extra = np.setdiff1d(np.unique(minute_dates), day_dates)
    if extra.size:
        raise DayPathsError(f"{EQUITY_FILE} has {extra.size} trade date(s) missing from {DAILY_FILE}")


def _fill_sides(trades: pl.DataFrame, ts: np.ndarray, minute_dates: np.ndarray) -> np.ndarray:
    """Contract sides filled per minute: a trade fills its contracts at entry_bar and again at exit_bar."""
    n = ts.shape[0]
    contracts = trades["contracts"].to_numpy().astype(np.int64)
    entry = trades["entry_bar"].to_numpy().astype(np.int64)
    exit_ = trades["exit_bar"].to_numpy().astype(np.int64)
    if bool(np.any(contracts < 1)):
        raise DayPathsError(f"{TRADES_FILE}: contracts < 1")
    for name, bars in (("entry_bar", entry), ("exit_bar", exit_)):
        if bool(np.any((bars < 0) | (bars >= n))):
            raise DayPathsError(f"{TRADES_FILE}: {name} outside the {n} equity_minute rows")
    if bool(np.any(exit_ < entry)):
        raise DayPathsError(f"{TRADES_FILE}: exit_bar before entry_bar")
    if not np.array_equal(ts[entry], trades["entry_ts"].to_numpy()):
        raise DayPathsError(f"{TRADES_FILE}: entry_ts differs from ts_event at entry_bar")
    if not np.array_equal(ts[exit_], trades["exit_ts"].to_numpy()):
        raise DayPathsError(f"{TRADES_FILE}: exit_ts differs from ts_event at exit_bar")
    if not np.array_equal(minute_dates[exit_], _yyyymmdd(trades["trade_date"], TRADES_FILE)):
        raise DayPathsError(f"{TRADES_FILE}: trade_date differs from the exit minute's trade date")
    fill = np.zeros(n, dtype=np.int64)
    np.add.at(fill, entry, contracts)
    np.add.at(fill, exit_, contracts)
    return fill


def minute_exposure(position: np.ndarray, trades: pl.DataFrame) -> np.ndarray:
    """Per equity_minute row: max(|position at the close|, contracts of each trade entered or exited there).

    A trade's own size is a lower bound on what was held inside its entry and exit minutes; several trades in
    one minute count by the largest, not the sum (a reversal exits and enters in one minute). The trades are
    already checked by _fill_sides (bars inside the rows, contracts >= 1)."""
    out = np.abs(position).astype(np.int64)
    contracts = trades["contracts"].to_numpy().astype(np.int64)
    for col in ("entry_bar", "exit_bar"):
        np.maximum.at(out, trades[col].to_numpy().astype(np.intp), contracts)
    return out


def _assemble(day_dates: np.ndarray, minute_dates: np.ndarray, eq_low: np.ndarray, eq_high: np.ndarray,
              eq_close: np.ndarray, fill: np.ndarray, position: np.ndarray,
              contracts_per_mini: int) -> dict[str, np.ndarray]:  # fmt: skip
    """Relative minutes per day; a daily date with no minute rows gets one zero minute."""
    opening = np.concatenate(([0], eq_close[:-1])).astype(np.int64)
    first = np.searchsorted(minute_dates, day_dates, side="left")
    last = np.searchsorted(minute_dates, day_dates, side="right")
    counts = last - first
    rows = np.maximum(counts, 1).astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(rows))).astype(np.int64)
    m = int(offsets[-1])
    low = np.zeros(m, dtype=np.int64)
    high = np.zeros(m, dtype=np.int64)
    close = np.zeros(m, dtype=np.int64)
    fill_out = np.zeros(m, dtype=np.int64)
    has = counts > 0
    # destination index of each source minute (source minutes are contiguous per day, days ascending)
    dest = np.repeat(offsets[:-1][has], counts[has]) + _ranks(counts[has])
    src = np.repeat(first[has], counts[has]) + _ranks(counts[has])
    day_open = np.repeat(opening[first[has]], counts[has])
    low[dest] = eq_low[src] - day_open
    high[dest] = eq_high[src] - day_open
    close[dest] = eq_close[src] - day_open
    fill_out[dest] = fill[src]
    sides = np.add.reduceat(fill_out, offsets[:-1].astype(np.intp)).astype(np.int64)
    max_abs = np.zeros(day_dates.shape[0], dtype=np.int64)
    if bool(np.any(has)):
        per_day = np.maximum.reduceat(np.abs(position), first[has].astype(np.intp))
        max_abs[has] = per_day
    max_contracts = (max_abs + contracts_per_mini - 1) // contracts_per_mini
    return {"trade_dates": day_dates.astype(np.int32), "offsets": offsets, "low": low, "high": high,
            "close": close, "fill_sides": fill_out, "sides": sides,
            "max_contracts": max_contracts.astype(np.int64)}  # fmt: skip


def _ranks(counts: np.ndarray) -> np.ndarray:
    """0..c-1 for each count c, concatenated."""
    total = int(counts.sum())
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    return np.arange(total, dtype=np.int64) - starts


def _check_daily_close(built: dict[str, np.ndarray], daily_close: np.ndarray, day_dates: np.ndarray,
                       minute_dates: np.ndarray) -> None:  # fmt: skip
    """daily.eq_close must equal the cumulative equity at each day's last minute (carried over empty days)."""
    cumulative = np.cumsum(built["close"][built["offsets"][1:] - 1])
    if not np.array_equal(cumulative, daily_close):
        bad = int(np.flatnonzero(cumulative != daily_close)[0])
        raise DayPathsError(
            f"{DAILY_FILE}: eq_close on {int(day_dates[bad])} is {int(daily_close[bad])}, the minutes give "
            f"{int(cumulative[bad])}"
        )


# --------------------------------------------------------------- synthetic ----
def synthetic_trade_dates(n_days: int) -> np.ndarray:
    """n consecutive business days from 2000-01-03 as int32 yyyymmdd."""
    start = np.datetime64(SYNTHETIC_FIRST_DAY, "D")
    days = np.busday_offset(start, np.arange(n_days), roll="forward")
    text = np.datetime_as_string(days, unit="D")
    return np.array([int(s.replace("-", "")) for s in text], dtype=np.int32)


def from_day_nets(nets: Sequence[int] | np.ndarray, lows: Sequence[int] | np.ndarray | None = None,
                  highs: Sequence[int] | np.ndarray | None = None,
                  sides: Sequence[int] | np.ndarray | None = None,
                  max_contracts: Sequence[int] | np.ndarray | None = None) -> DayPaths:  # fmt: skip
    """One minute per day: close = net, low = min(0, net) and high = max(0, net) unless given.

    sides defaults to 2 per day (one round trip), filled at the day's only minute; max_contracts to 1.
    """
    net = _int_array(nets, "nets")
    n = net.shape[0]
    low = np.minimum(net, 0) if lows is None else _int_array(lows, "lows", n)
    high = np.maximum(net, 0) if highs is None else _int_array(highs, "highs", n)
    if bool(np.any(low > np.minimum(net, 0))) or bool(np.any(high < np.maximum(net, 0))):
        raise DayPathsError("a one-minute day needs low <= min(0, net) and high >= max(0, net)")
    return from_minutes([(low[i : i + 1], high[i : i + 1], net[i : i + 1]) for i in range(n)],
                        sides=sides, max_contracts=max_contracts)  # fmt: skip


def from_minutes(minutes: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]] | Sequence[tuple],
                 sides: Sequence[int] | np.ndarray | None = None,
                 max_contracts: Sequence[int] | np.ndarray | None = None) -> DayPaths:  # fmt: skip
    """Days from per-day (low, high, close) arrays of cents relative to the day's open.

    sides defaults to 2 per day; each day's sides are filled at its first minute (the conservative placement
    the cost stress uses when fill minutes are unknown).
    """
    if len(minutes) == 0:
        raise DayPathsError("from_minutes needs at least one day")
    parts = [tuple(_int_array(a, f"day {d} {name}") for a, name in zip(day, ("low", "high", "close"),
                                                                          strict=True))
             for d, day in enumerate(minutes)]  # fmt: skip
    for d, (lo, hi, cl) in enumerate(parts):
        if not (lo.shape == hi.shape == cl.shape) or lo.shape[0] < 1:
            raise DayPathsError(f"day {d}: low, high, close must be equal-length and non-empty")
    n = len(parts)
    counts = np.array([p[0].shape[0] for p in parts], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    side_arr = (
        np.full(n, DEFAULT_SIDES_PER_DAY, dtype=np.int64) if sides is None else _int_array(sides, "sides", n)
    )
    fill = np.zeros(int(offsets[-1]), dtype=np.int64)
    fill[offsets[:-1]] = side_arr
    cap = (
        np.ones(n, dtype=np.int64) if max_contracts is None else _int_array(max_contracts, "max_contracts", n)
    )
    days = DayPaths(
        trade_dates=synthetic_trade_dates(n), offsets=offsets,
        low=np.concatenate([p[0] for p in parts]), high=np.concatenate([p[1] for p in parts]),
        close=np.concatenate([p[2] for p in parts]), fill_sides=fill, sides=side_arr, max_contracts=cap,
        source=("synthetic",),
    )  # fmt: skip
    return check_day_paths(days)


def _int_array(values: Sequence[int] | np.ndarray, name: str, n: int | None = None) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1 or (arr.size and arr.dtype.kind not in "iu"):
        raise DayPathsError(f"{name} must be a 1-d sequence of integers (cents), got dtype {arr.dtype}")
    arr = arr.astype(np.int64)
    if n is not None and arr.shape[0] != n:
        raise DayPathsError(f"{name} has {arr.shape[0]} values, expected {n}")
    if n is None and arr.shape[0] < 1:
        raise DayPathsError(f"{name} is empty")
    return arr
