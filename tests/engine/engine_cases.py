"""Builders for the engine tests: kernel inputs from compact bar rows, frames and market inputs in code.

Every price is synthetic and in ticks. Defaults: tick value 5000 (1e-4 USD, 50 cents a tick), commission
10 cents per fill, slippage 5 cents on a buy and 7 cents on a sell.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from pfb.data.gateway import ct_minute_expr
from pfb.engine.kernel import KernelInputs, KernelOutput, run_kernel
from pfb.engine.run import MarketInputs
from pfb.engine.strategy import Bars, ParamValue, Strategy, StrategySpec

TV_CC = 5000
COMM = 10
SLIP_BUY = 5
SLIP_SELL = 7
NS_MIN = 60_000_000_000
CT = ZoneInfo("America/Chicago")
Row = tuple[int, int, int, int]


def _mask(n: int, idx: Iterable[int], default: bool = False) -> np.ndarray:
    out = np.full(n, default, dtype=bool)
    for i in idx:
        out[i] = not default
    return out


def _ints(n: int, values: Mapping[int, int]) -> np.ndarray:
    out = np.zeros(n, dtype=np.int64)
    for i, v in values.items():
        out[i] = v
    return out


def kernel_inputs(
    rows: Sequence[Row],
    *,
    long: Iterable[int] = (),
    short: Iterable[int] = (),
    exits: Iterable[int] = (),
    stops: Mapping[int, int] | None = None,
    targets: Mapping[int, int] | None = None,
    session_end: Iterable[int] | None = None,
    flat_by: Iterable[int] = (),
    no_fill: Iterable[int] = (),
    blocked: Iterable[int] = (),
    uncalibrated: Iterable[int] = (),
    tv_cc: int = TV_CC,
    commission: int = COMM,
    contracts: int = 1,
) -> KernelInputs:
    """rows = (open, high, low, close) per bar; session_end defaults to the last bar only."""
    n = len(rows)
    arr = np.array(rows, dtype=np.int64).reshape(n, 4)
    ends = [n - 1] if session_end is None else session_end
    return KernelInputs(
        open=arr[:, 0],
        high=arr[:, 1],
        low=arr[:, 2],
        close=arr[:, 3],
        slip_buy=np.full(n, SLIP_BUY, dtype=np.int64),
        slip_sell=np.full(n, SLIP_SELL, dtype=np.int64),
        session_end=_mask(n, ends),
        flat_by=_mask(n, flat_by),
        entry_fill_ok=_mask(n, no_fill, default=True),
        entry_allowed=_mask(n, blocked, default=True),
        calibrated=_mask(n, uncalibrated, default=True),
        entry_long=_mask(n, long),
        entry_short=_mask(n, short),
        exit=_mask(n, exits),
        stop_ticks=_ints(n, stops or {}),
        target_ticks=_ints(n, targets or {}),
        commission_cents=commission,
        tick_value_cc=tv_cc,
        contracts=contracts,
    )


def run(rows: Sequence[Row], **kwargs: object) -> KernelOutput:
    return run_kernel(kernel_inputs(rows, **kwargs))  # type: ignore[arg-type]


# ------------------------------------------------------------------ frames ----
def ct_ns(day: date, hour: int, minute: int) -> int:
    """UTC nanoseconds of a CT wall-clock time."""
    return int(datetime(day.year, day.month, day.day, hour, minute, tzinfo=CT).timestamp()) * 1_000_000_000


@dataclass(frozen=True)
class Bar:
    ts: int
    trade_date: date
    ohlc: Row
    closure: bool = False
    instrument_id: int = 1
    era: str = "research"


def minute_bars(day: date, hour: int, minute: int, rows: Sequence[Row], **extra: object) -> list[Bar]:
    start = ct_ns(day, hour, minute)
    return [Bar(start + i * NS_MIN, day, r, **extra) for i, r in enumerate(rows)]  # type: ignore[arg-type]


def frame_of(bars: Sequence[Bar]) -> pl.DataFrame:
    rows = np.array([b.ohlc for b in bars], dtype=np.int64).reshape(len(bars), 4)
    frame = pl.DataFrame(
        {
            "ts_event": pl.Series([b.ts for b in bars], dtype=pl.Int64),
            "open": rows[:, 0],
            "high": rows[:, 1],
            "low": rows[:, 2],
            "close": rows[:, 3],
            "volume": pl.Series([1] * len(bars), dtype=pl.Int64),
            "trade_date": pl.Series([b.trade_date for b in bars], dtype=pl.Date),
            "in_scheduled_closure": pl.Series([b.closure for b in bars], dtype=pl.Boolean),
            "instrument_id": pl.Series([b.instrument_id for b in bars], dtype=pl.Int64),
            "era": pl.Series([b.era for b in bars], dtype=pl.String),
        }
    )
    return frame.with_columns(ct_minute_expr())


def market_inputs(
    frame: pl.DataFrame,
    *,
    flat: Sequence[tuple[int, int]] = (),
    window: tuple[int, int] | None = None,
    unsourced: Sequence[date] = (),
    slippage: np.ndarray | None = None,
    calibrated: np.ndarray | None = None,
    tv_cc: int = TV_CC,
) -> MarketInputs:
    """A session window per trade day (default: one that holds every bar), flat intervals as given."""
    days = np.unique(frame["trade_date"].cast(pl.Int32).to_numpy()).astype(np.int32)
    ts = frame["ts_event"].to_numpy()
    lo, hi = window if window is not None else (int(ts.min()) - NS_MIN, int(ts.max()) + NS_MIN)
    table = (
        np.tile(np.array([SLIP_BUY, SLIP_SELL], dtype=np.int64), (1440, 1)) if slippage is None else slippage
    )
    return MarketInputs(
        must_be_flat_starts=np.array([s for s, _ in flat], dtype=np.int64),
        must_be_flat_ends=np.array([e for _, e in flat], dtype=np.int64),
        session_days=days,
        session_starts=np.full(len(days), lo, dtype=np.int64),
        session_ends=np.full(len(days), hi, dtype=np.int64),
        unsourced_days=pl.Series(list(unsourced), dtype=pl.Date).cast(pl.Int32).to_numpy().astype(np.int32),
        slippage_cents=table,
        calibrated=np.ones(1440, dtype=bool) if calibrated is None else calibrated,
        commission_cents=COMM,
        tick_value_cc=tv_cc,
    )


def scripted(
    long: Iterable[int] = (),
    short: Iterable[int] = (),
    exits: Iterable[int] = (),
    stops: Mapping[int, int] | None = None,
    targets: Mapping[int, int] | None = None,
    contracts: int = 1,
) -> Strategy:
    """A causal strategy whose signals sit at fixed bar indices (truncation-safe: a prefix stays a prefix)."""
    long_i, short_i, exit_i = list(long), list(short), list(exits)

    def generate(bars: Bars, params: Mapping[str, ParamValue]) -> dict[str, np.ndarray]:
        n = len(bars["ts_event"])
        pick = [i for i in range(n)]
        return {
            "entry_long": _mask(n, [i for i in long_i if i in pick]),
            "entry_short": _mask(n, [i for i in short_i if i in pick]),
            "exit": _mask(n, [i for i in exit_i if i in pick]),
            "stop_ticks": _ints(n, {i: v for i, v in (stops or {}).items() if i < n}),
            "target_ticks": _ints(n, {i: v for i, v in (targets or {}).items() if i < n}),
        }

    return strategy_of(generate, contracts=contracts)


def strategy_of(generate: Callable[[Bars, Mapping[str, ParamValue]], object], contracts: int = 1) -> Strategy:
    spec = StrategySpec("engine_check:test", "MNQ", 1, "globex", contracts, {})
    return Strategy(spec=spec, generate=generate, sha256="0" * 64, path=Path("scripted.py"))
