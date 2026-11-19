"""Several runs in one account: a pooled DayPaths on a common clock, and contract caps across the pool.

``pool_runs(run_dirs)`` returns one DayPaths for the account that holds every run at once, which then goes
through the same Monte Carlo (``run_mc``) as a single run. ``check_caps(pooled, plan, phase=...)`` refuses a
pool whose combined open contracts exceed the plan's cap (callers run it before ``run_mc``).

Common clock:
- trade dates: the union of the runs' daily.parquet dates; minutes within a day: the union of the runs'
  equity_minute ts_event values (two runs' rows at the same ts_event must carry the same trade date);
- each run's cumulative equity is forward-filled within its own history: 0 before its first row, its last
  eq_close at a minute where it has no row (on any day, a day without rows included);
- pooled eq_close = the sum of the runs' (carried) eq_close at that minute;
- pooled eq_low = the sum of the runs' eq_low at that minute, a run without a row there adding its carried
  close; likewise eq_high. This is a CONSERVATIVE BOUND, not the exact pooled extreme: the runs' lows inside
  one minute need not happen at the same instant, so the true pooled low is >= this sum and the true pooled
  high is <= this sum. Drawdown checks on the pool are therefore never optimistic; profit-target checks that
  read the high intraday could be optimistic by the same margin (an intraday target check on a pool should be
  read with that in mind). The bound assumes the runs share one bar grid (one-minute bars); with coarser or
  shifted bars a run's whole-bar extreme is attributed to its own bar's timestamp only;
- minutes and days are then made relative to the pooled day open exactly as in days.py (the pooled close of
  the previous minute, 0 before the first), a date where no run has a row getting one zero minute;
- fill_sides per minute and sides per day are summed over the runs;
- max_contracts[d] = the largest per-minute sum over the runs of their exposure in mini equivalents, rounded
  up once after summing (exact integer units). A run's exposure at its own row is days.minute_exposure (the
  larger of |position| and the size of a trade entered or exited in that minute); at a minute
  where it has no row, its carried |position|. Summing the runs' in-minute exposures is a conservative bound
  like the pooled low.

Mini equivalents: a run's root comes from summary.json ("root") when present, else from the run's record in
the registry ledger (read-only); config/instruments.toml gives its contract type. A mini (or full-size)
contract counts 1, a micro 1/10 (MICROS_PER_MINI). Other contract types (SIL, MBT) have no stated mini
equivalent and are refused unless the caller passes contracts_per_mini explicitly.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pfb.instruments.table import InstrumentTable, UnknownRoot, load_instruments
from pfb.prop.days import (
    DAILY_COLUMNS,
    DAILY_FILE,
    EQUITY_COLUMNS,
    EQUITY_FILE,
    TRADE_COLUMNS,
    TRADES_FILE,
    _assemble,
    _fill_sides,
    _read,
    _yyyymmdd,
    check_day_paths,
    day_paths_from_run,
    minute_exposure,
)
from pfb.prop.types import DayPaths

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LEDGER = REPO_ROOT / "ledger" / "registry.jsonl"
SUMMARY_FILE = "summary.json"
MICROS_PER_MINI = 10  # 10 micro contracts count as one mini (the prop firms' scaling convention)
CONTRACTS_PER_MINI = {"mini": 1, "micro": MICROS_PER_MINI}
PHASES = ("eval", "funded")
NO_ROW = -1  # PooledRuns.ts_event of a filler minute (a date on which no run has a row)


class PoolError(ValueError):
    """The runs cannot be pooled (bad artifacts, clashing clocks, unknown mini equivalent)."""


class PoolOverLimit(ValueError):
    """The pool's combined open contracts exceed the plan's cap.

    trade_date is int yyyymmdd; minute is the 0-based minute index within that day of the pooled DayPaths
    (None when only day-level maxima are known); ts_event is that minute's UTC ns timestamp (None likewise).
    """

    def __init__(self, *, phase: str, limit: int, contracts: int, trade_date: int, minute: int | None,
                 ts_event: int | None) -> None:  # fmt: skip
        self.phase, self.limit, self.contracts = phase, limit, contracts
        self.trade_date, self.minute, self.ts_event = trade_date, minute, ts_event
        where = f"trade date {trade_date}"
        if minute is not None:
            where += f", minute {minute} of the day"
        if ts_event is not None and ts_event != NO_ROW:
            stamp = dt.datetime.fromtimestamp(ts_event // 1_000_000_000, tz=dt.UTC).isoformat()
            where += f" ({stamp})"
        super().__init__(
            f"pool holds {contracts} mini equivalents at {where}, over the {phase} cap of {limit}"
        )


@dataclass(frozen=True)
class PooledRuns:
    """A pooled DayPaths with the per-minute detail the cap check needs to name the offending minute.

    ts_event[m] is the UTC ns timestamp of pooled minute m (NO_ROW for a filler minute); units[m] is the
    pool's open contracts at minute m in 1/units_per_mini mini equivalents (exact integers).
    """

    days: DayPaths
    ts_event: np.ndarray  # int64 [n_minutes]
    units: np.ndarray  # int64 [n_minutes]
    units_per_mini: int


@dataclass(frozen=True)
class _Run:
    name: str
    ts: np.ndarray  # int64, strictly increasing
    minute_dates: np.ndarray  # int32 yyyymmdd
    day_dates: np.ndarray  # int32 yyyymmdd (daily.parquet)
    daily_close: np.ndarray  # int64 cumulative cents at each daily date
    eq_low: np.ndarray
    eq_high: np.ndarray
    eq_close: np.ndarray
    position: np.ndarray
    exposure: np.ndarray  # days.minute_exposure per own row
    fill: np.ndarray
    contracts_per_mini: int


# ------------------------------------------------------------------------------------- public API ----
def pool_runs(run_dirs: Sequence[Path | str], *, contracts_per_mini: Sequence[int] | None = None,
              ledger: Path | None = None,
              instruments: InstrumentTable | None = None) -> DayPaths:  # fmt: skip
    """The pooled DayPaths of several runs held in one account (source = the run ids, in the given order).

    contracts_per_mini: one value per run (1 mini, 10 micro) to skip the root lookup; ledger: the registry
    file to look a run's root up in (default ledger/registry.jsonl of this repository).
    """
    return pool_detail(run_dirs, contracts_per_mini=contracts_per_mini, ledger=ledger,
                       instruments=instruments).days  # fmt: skip


def pool_detail(run_dirs: Sequence[Path | str], *, contracts_per_mini: Sequence[int] | None = None,
                ledger: Path | None = None,
                instruments: InstrumentTable | None = None) -> PooledRuns:  # fmt: skip
    """pool_runs plus the per-minute timestamps and contract units (see PooledRuns)."""
    dirs = [Path(d) for d in run_dirs]
    if not dirs:
        raise PoolError("pool_runs needs at least one run directory")
    resolved = [d.resolve() for d in dirs]
    if len(set(resolved)) != len(resolved):
        raise PoolError("a run directory appears twice in the pool")
    if contracts_per_mini is not None and len(contracts_per_mini) != len(dirs):
        raise PoolError(f"contracts_per_mini has {len(contracts_per_mini)} values for {len(dirs)} runs")
    if contracts_per_mini is not None:
        cpms = list(contracts_per_mini)
    else:
        cpms = [run_contracts_per_mini(d, ledger=ledger, instruments=instruments) for d in dirs]
    if any(isinstance(c, bool) or not isinstance(c, int | np.integer) or c < 1 for c in cpms):
        raise PoolError(f"contracts_per_mini must be integers >= 1, got {cpms}")
    runs = [_load_run(d, int(c)) for d, c in zip(dirs, cpms, strict=True)]
    return _pool(runs)


def check_caps(pooled: PooledRuns | DayPaths, plan: Any, *, phase: str = "eval") -> None:
    """Raise PoolOverLimit when the pool's open contracts exceed the plan's cap in this phase.

    eval: eval.max_contracts_minis. funded: the top tier of funded.scaling (the most the plan ever allows, a
    necessary condition; the balance-dependent lower tiers are not checked here). A DayPaths alone gives the
    first offending trade date only; a PooledRuns also gives the minute and its timestamp.
    """
    limit = cap_minis(plan, phase)
    days = pooled.days if isinstance(pooled, PooledRuns) else pooled
    over = np.flatnonzero(days.max_contracts > limit)
    if over.size == 0:
        return
    d = int(over[0])
    trade_date, contracts = int(days.trade_dates[d]), int(days.max_contracts[d])
    if not isinstance(pooled, PooledRuns):
        raise PoolOverLimit(phase=phase, limit=limit, contracts=contracts, trade_date=trade_date, minute=None,
                            ts_event=None)  # fmt: skip
    lo, hi = int(days.offsets[d]), int(days.offsets[d + 1])
    per_minute = -(-pooled.units[lo:hi] // pooled.units_per_mini)  # ceil to whole mini equivalents
    minute = int(np.flatnonzero(per_minute > limit)[0])
    raise PoolOverLimit(phase=phase, limit=limit, contracts=int(per_minute[minute]), trade_date=trade_date,
                        minute=minute, ts_event=int(pooled.ts_event[lo + minute]))  # fmt: skip


def cap_minis(plan: Any, phase: str) -> int:
    """The plan's contract cap in mini equivalents for the phase (RuleUnknown propagates)."""
    if phase == "eval":
        return int(plan.value("eval.max_contracts_minis"))
    if phase == "funded":
        tiers = plan.value("funded.scaling")
        return int(tiers[-1][1])
    raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")


def run_contracts_per_mini(run_dir: Path, *, ledger: Path | None = None,
                           instruments: InstrumentTable | None = None) -> int:  # fmt: skip
    """1 for a run on a mini root, 10 for a micro root; PoolError for any other contract type."""
    root = run_root(run_dir, ledger=ledger)
    table = instruments if instruments is not None else load_instruments()
    try:
        kind = table.get(root).contract_type
    except UnknownRoot as exc:
        raise PoolError(f"{Path(run_dir).name}: root {root} is not in config/instruments.toml") from exc
    if kind not in CONTRACTS_PER_MINI:
        raise PoolError(f"{Path(run_dir).name}: root {root} has contract type {kind!r} with no stated mini "
                        "equivalent; pass contracts_per_mini")  # fmt: skip
    return CONTRACTS_PER_MINI[kind]


def run_root(run_dir: Path, *, ledger: Path | None = None) -> str:
    """The run's root: summary.json "root" when present, else the ledger record(s) with this run_id."""
    run_dir = Path(run_dir)
    summary = run_dir / SUMMARY_FILE
    if summary.is_file():
        root = json.loads(summary.read_text()).get("root")
        if isinstance(root, str) and root:
            return root
    path = ledger if ledger is not None else DEFAULT_LEDGER
    if not path.is_file():
        raise PoolError(f"{run_dir.name}: no root in {SUMMARY_FILE} and no ledger at {path}")
    roots = set()
    with path.open() as fh:
        for line in fh:
            if run_dir.name not in line:
                continue
            record = json.loads(line)
            record = record.get("record", record) if isinstance(record, dict) else record
            if isinstance(record, dict) and record.get("run_id") == run_dir.name and record.get("root"):
                roots.add(str(record["root"]))
    if len(roots) != 1:
        raise PoolError(f"{run_dir.name}: the ledger gives {len(roots)} root(s) {sorted(roots)}; expected 1")
    return roots.pop()


# -------------------------------------------------------------------------------------- internals ----
def _load_run(run_dir: Path, cpm: int) -> _Run:
    """Read one run's frames; day_paths_from_run first validates every artifact (a mismatch refuses)."""
    try:
        day_paths_from_run(run_dir, contracts_per_mini=cpm)
    except ValueError as exc:
        raise PoolError(f"{run_dir.name}: {exc}") from exc
    equity = _read(run_dir / EQUITY_FILE, EQUITY_COLUMNS)
    daily = _read(run_dir / DAILY_FILE, DAILY_COLUMNS)
    trades = _read(run_dir / TRADES_FILE, TRADE_COLUMNS)
    ts = equity["ts_event"].to_numpy().astype(np.int64)
    minute_dates = _yyyymmdd(equity["trade_date"], EQUITY_FILE)

    def col(frame: pl.DataFrame, name: str) -> np.ndarray:
        return frame[name].to_numpy().astype(np.int64)

    return _Run(
        name=run_dir.name, ts=ts, minute_dates=minute_dates,
        day_dates=_yyyymmdd(daily["trade_date"], DAILY_FILE),
        daily_close=col(daily, "eq_close"), eq_low=col(equity, "eq_low"), eq_high=col(equity, "eq_high"),
        eq_close=col(equity, "eq_close"), position=col(equity, "position"),
        exposure=minute_exposure(col(equity, "position"), trades),
        fill=_fill_sides(trades, ts, minute_dates), contracts_per_mini=cpm,
    )  # fmt: skip


def _common_clock(runs: Sequence[_Run]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(union ts, trade date per union ts, union of daily dates); refuses clashing or backward dates."""
    all_ts = np.concatenate([r.ts for r in runs])
    all_dates = np.concatenate([r.minute_dates for r in runs])
    order = np.argsort(all_ts, kind="stable")
    ts_sorted, dates_sorted = all_ts[order], all_dates[order]
    same = ts_sorted[1:] == ts_sorted[:-1]
    clash = same & (dates_sorted[1:] != dates_sorted[:-1])
    if bool(np.any(clash)):
        i = int(np.flatnonzero(clash)[0])
        raise PoolError(f"ts_event {int(ts_sorted[i])} carries trade dates {int(dates_sorted[i])} and "
                        f"{int(dates_sorted[i + 1])} in different runs")  # fmt: skip
    keep = np.concatenate(([True], ~same))
    union_ts, union_dates = ts_sorted[keep], dates_sorted[keep]
    if bool(np.any(np.diff(union_dates) < 0)):
        raise PoolError("on the common clock a later minute carries an earlier trade date")
    day_dates = np.unique(np.concatenate([r.day_dates for r in runs])).astype(np.int32)
    return union_ts, union_dates.astype(np.int32), day_dates


def _carry(run: _Run, union_ts: np.ndarray) -> dict[str, np.ndarray]:
    """One run on the common clock: carried close and position, low/high where it has a row, fills."""
    k = np.searchsorted(run.ts, union_ts, side="right") - 1  # last own row at or before the minute
    seen = k >= 0
    kk = np.where(seen, k, 0)
    has_row = seen & (run.ts[kk] == union_ts)
    close = np.where(seen, run.eq_close[kk], 0)
    fill = np.zeros(union_ts.shape[0], dtype=np.int64)
    fill[np.searchsorted(union_ts, run.ts)] = run.fill
    return {"close": close, "low": np.where(has_row, run.eq_low[kk], close),
            "high": np.where(has_row, run.eq_high[kk], close), "fill": fill,
            "abs_position": np.where(has_row, run.exposure[kk],
                                     np.where(seen, np.abs(run.position[kk]), 0))}  # fmt: skip


def _pool(runs: Sequence[_Run]) -> PooledRuns:
    union_ts, union_dates, day_dates = _common_clock(runs)
    units_per_mini = math.lcm(*(r.contracts_per_mini for r in runs))
    n = union_ts.shape[0]
    low, high, close, fill, units = (np.zeros(n, dtype=np.int64) for _ in range(5))
    for run in runs:
        c = _carry(run, union_ts)
        low += c["low"]
        high += c["high"]
        close += c["close"]
        fill += c["fill"]
        units += c["abs_position"] * (units_per_mini // run.contracts_per_mini)
    built = _assemble(day_dates, union_dates, low, high, close, fill, units, units_per_mini)
    days = check_day_paths(DayPaths(**{**built, "source": tuple(r.name for r in runs)}))  # type: ignore[arg-type]
    _check_pooled_daily(days, runs)
    return PooledRuns(days=days, ts_event=_minute_ts(days, day_dates, union_dates, union_ts),
                      units=_minute_values(days, day_dates, union_dates, units),
                      units_per_mini=units_per_mini)  # fmt: skip


def _minute_index(days: DayPaths, day_dates: np.ndarray, union_dates: np.ndarray) -> np.ndarray:
    """For each pooled minute, the union-clock index it came from (NO_ROW for a filler minute)."""
    first = np.searchsorted(union_dates, day_dates, side="left")
    counts = np.searchsorted(union_dates, day_dates, side="right") - first
    out = np.full(int(days.offsets[-1]), NO_ROW, dtype=np.int64)
    has = counts > 0
    starts = np.repeat(first[has], counts[has])
    c = counts[has]
    ranks = np.arange(int(c.sum()), dtype=np.int64) - np.repeat(np.cumsum(c) - c, c)
    out[np.repeat(days.offsets[:-1][has], counts[has]) + ranks] = starts + ranks
    return out


def _minute_ts(days: DayPaths, day_dates: np.ndarray, union_dates: np.ndarray,
               union_ts: np.ndarray) -> np.ndarray:  # fmt: skip
    idx = _minute_index(days, day_dates, union_dates)
    return np.where(idx >= 0, union_ts[np.maximum(idx, 0)], NO_ROW).astype(np.int64)


def _minute_values(days: DayPaths, day_dates: np.ndarray, union_dates: np.ndarray,
                   values: np.ndarray) -> np.ndarray:  # fmt: skip
    idx = _minute_index(days, day_dates, union_dates)
    return np.where(idx >= 0, values[np.maximum(idx, 0)], 0).astype(np.int64)


def _check_pooled_daily(days: DayPaths, runs: Sequence[_Run]) -> None:
    """The pooled day nets must add up to the sum of the runs' daily eq_close carried to each pooled date."""
    expected = np.zeros(days.n_days, dtype=np.int64)
    for run in runs:
        k = np.searchsorted(run.day_dates, days.trade_dates, side="right") - 1
        expected += np.where(k >= 0, run.daily_close[np.maximum(k, 0)], 0)
    got = np.cumsum(days.day_net())
    if not np.array_equal(got, expected):
        bad = int(np.flatnonzero(got != expected)[0])
        raise PoolError(f"pooled equity on {int(days.trade_dates[bad])} is {int(got[bad])}, the runs' daily "
                        f"closes give {int(expected[bad])}")  # fmt: skip


__all__ = [
    "MICROS_PER_MINI", "NO_ROW", "PoolError", "PoolOverLimit", "PooledRuns", "cap_minis", "check_caps",
    "pool_detail", "pool_runs", "run_contracts_per_mini", "run_root",
]  # fmt: skip
