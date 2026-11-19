"""Planted-future canaries (ADR-001 section 12).

A copy of synthetic bars gets a JUMP_TICKS jump INSIDE a marked bar every MARK_EVERY_MINUTES of the CT day
window: the marked bar's close jumps and every later bar is shifted by the same amount. A test-only reader
strategy is handed the markers (bar ts -> direction), enters WITH the jump at the marked bar's close and exits
one bar after its fill. Through a point-in-time engine (decision at close i, fill at open i + 1) the entry
fills after the jump, so the reader captures noise only. An engine that fills at the decision bar's own open,
or that shows the strategy bar i + 1 while it is on bar i, captures the jump. The latency control puts the
marker one bar early (a legitimately tradeable signal): the honest engine must capture it, which shows the
canary is not blind.

The engine is a parameter (`CanaryEngine`): `honest_engine` is pfb.engine.run.simulate unchanged, including
its truncation check. Leaky mutant engines live in the test suite only; the kernel has no switch for them.
Prices are int64 ticks throughout; the statistics are floats and decide no rule boundary.
"""

from __future__ import annotations

import math
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from pfb.engine.run import MarketInputs, simulate
from pfb.engine.strategy import Bars, ParamValue, Strategy, bar_count, validate_spec
from pfb.engine.synthetic import synthetic_frame
from pfb.instruments import sessions
from pfb.instruments.table import load_instruments

NS_PER_MINUTE = 60_000_000_000
MINUTES_PER_DAY = 1440
JUMP_TICKS = 16  # Upstream: several times a typical 1-minute move (synthetic steps are -4..4 ticks)
MARK_EVERY_MINUTES = 20
MARK_WINDOW_CT = (8 * 60 + 35, 14 * 60 + 55)  # clear of the day open and of the flat-by clocks
MIN_TRADES = 100
NO_EDGE_MEAN_FRACTION = 4  # honest engine: mean captured < JUMP / 4
EDGE_MEAN_FRACTION = 2  # a detected leak: mean captured >= JUMP / 2
MAX_NULL_Z = 4.0
MIN_EDGE_Z = 5.0
HIT_SE_LIMIT = 4.0
CANARY_SEED = 20_260_917
CANARY_SHA256 = "0" * 64
CANARY_ROOT = "MNQ"
CANARY_FIRST, CANARY_LAST = date(2025, 4, 1), date(2025, 4, 30)
EPOCH = date(1970, 1, 1)

CanaryEngine = Callable[[Strategy, pl.DataFrame, MarketInputs], pl.DataFrame]


@dataclass(frozen=True)
class PlantedBars:
    frame: pl.DataFrame  # a shifted copy; the input frame is never changed
    leak_markers: Mapping[int, int]  # ts of the jumped bar -> direction (marker ON the jumped bar)
    latency_markers: Mapping[int, int]  # ts of the bar BEFORE the jumped bar -> direction
    jump_ticks: int


@dataclass(frozen=True)
class CanaryScore:
    trades: int
    unmatched_trades: int
    mean_captured_ticks: float
    se_ticks: float
    z: float
    hit_rate: float
    hit_n: int
    hit_se_under_null: float


# ------------------------------------------------------------------ planting ----
def _markable(ts: np.ndarray, minute: np.ndarray, day: np.ndarray, every: int) -> np.ndarray:
    """Indices i with a marked CT minute, bars i-1..i+2 one minute apart on one trade date."""
    n = len(ts)
    if n < 4:
        return np.zeros(0, dtype=np.int64)
    i = np.arange(1, n - 2)
    m = minute[i].astype(np.int64)
    lo, hi = MARK_WINDOW_CT
    on_clock = (m >= lo) & (m <= hi) & ((m - lo) % every == 0)
    contiguous = (ts[i] - ts[i - 1] == NS_PER_MINUTE) & (ts[i + 2] - ts[i] == 2 * NS_PER_MINUTE)
    one_day = (day[i - 1] == day[i]) & (day[i + 1] == day[i]) & (day[i + 2] == day[i])
    return i[on_clock & contiguous & one_day]


def plant_canaries(
    frame: pl.DataFrame, seed: int, jump_ticks: int = JUMP_TICKS, mark_every: int = MARK_EVERY_MINUTES
) -> PlantedBars:
    """A shifted copy of a gateway-shaped frame with a jump inside each marked bar (upstream
    plant_canaries)."""
    if jump_ticks < 1 or mark_every < 1:
        raise ValueError("jump_ticks and mark_every must be >= 1")
    ts = frame["ts_event"].to_numpy()
    day = frame["trade_date"].cast(pl.Int32).to_numpy()
    marked = _markable(ts, frame["ct_minute"].to_numpy(), day, mark_every)
    rng = np.random.default_rng(seed)
    direction = rng.choice(np.array([-1, 1], dtype=np.int64), size=len(marked))
    delta = np.zeros(len(ts), dtype=np.int64)
    delta[marked + 1] += direction * jump_ticks
    shift = np.cumsum(delta)
    cols = {c: frame[c].to_numpy().astype(np.int64) + shift for c in ("open", "high", "low", "close")}
    cols["close"][marked] += direction * jump_ticks
    cols["high"] = np.maximum(cols["high"], cols["close"])
    cols["low"] = np.minimum(cols["low"], cols["close"])
    out = frame.with_columns(pl.Series(c, v, dtype=pl.Int64) for c, v in cols.items())
    leak = {int(ts[i]): int(d) for i, d in zip(marked, direction, strict=True)}
    latency = {int(ts[i - 1]): int(d) for i, d in zip(marked, direction, strict=True)}
    return PlantedBars(out, types.MappingProxyType(leak), types.MappingProxyType(latency), jump_ticks)


# ------------------------------------------------------------- the reader ----
def _marker_directions(ts: np.ndarray, markers: Mapping[int, int]) -> np.ndarray:
    keys = np.array(sorted(markers), dtype=np.int64)
    values = np.array([markers[int(k)] for k in keys], dtype=np.int64)
    out = np.zeros(len(ts), dtype=np.int64)
    if len(keys) == 0 or len(ts) == 0:
        return out
    pos = np.minimum(np.searchsorted(keys, ts), len(keys) - 1)
    hit = keys[pos] == ts
    out[hit] = values[pos[hit]]
    return out


def canary_reader(markers: Mapping[int, int]) -> Callable[[Bars, Mapping[str, ParamValue]], object]:
    """Test-only generate: enter WITH the marker at its bar's close, exit at the close of the next bar.

    It reads only the bar's own ts (point in time), so it passes the truncation check."""
    frozen = types.MappingProxyType(dict(markers))

    def generate(bars: Bars, params: Mapping[str, ParamValue]) -> dict[str, np.ndarray]:
        n = bar_count(bars)
        direction = _marker_directions(bars["ts_event"], frozen)
        marked = direction != 0
        exit_ = np.zeros(n, dtype=np.bool_)
        exit_[1:] = marked[:-1]
        zeros = np.zeros(n, dtype=np.int64)
        return {"entry_long": direction > 0, "entry_short": direction < 0, "exit": exit_,
                "stop_ticks": zeros, "target_ticks": zeros.copy()}  # fmt: skip

    return generate


def canary_strategy(markers: Mapping[int, int], root: str = CANARY_ROOT) -> Strategy:
    """The reader as a Strategy (globex session, one contract, no parameters, no file)."""
    raw = {"hypothesis_id": "canary:reader", "root": root, "bar_minutes": 1, "session": "globex",
           "contracts": 1, "params": {}}  # fmt: skip
    return Strategy(validate_spec(raw), canary_reader(markers), CANARY_SHA256, Path("<canary>"))


# --------------------------------------------------------------- market ----
def zero_cost_inputs(root: str, frame: pl.DataFrame, session: str = "globex") -> MarketInputs:
    """The root's real calendar (sessions, flat-by, unsourced days) with zero slippage and commission."""
    days = np.unique(frame["trade_date"].cast(pl.Int32).to_numpy()).astype(np.int32)
    if len(days) == 0:
        raise ValueError("the frame has no bars")
    dates = [EPOCH + timedelta(days=int(d)) for d in days]
    windows = [sessions.session_window_ns(root, d, session) for d in dates]
    flat_starts, flat_ends = sessions.must_be_flat_intervals_ns(root, dates[0], dates[-1])
    return MarketInputs(
        must_be_flat_starts=np.asarray(flat_starts, dtype=np.int64),
        must_be_flat_ends=np.asarray(flat_ends, dtype=np.int64),
        session_days=days,
        session_starts=np.array([w[0] for w in windows], dtype=np.int64),
        session_ends=np.array([w[1] for w in windows], dtype=np.int64),
        unsourced_days=np.asarray(sessions.unsourced_days(root), dtype=np.int32),
        slippage_cents=np.zeros((MINUTES_PER_DAY, 2), dtype=np.int64),
        calibrated=np.ones(MINUTES_PER_DAY, dtype=np.bool_),
        commission_cents=0,
        tick_value_cc=int(load_instruments().get(root).tick_value_cc),
    )


def honest_engine(strategy: Strategy, frame: pl.DataFrame, inputs: MarketInputs) -> pl.DataFrame:
    """pfb.engine.run.simulate as it is (truncation check included), at no parameters."""
    return simulate(strategy, frame, {}, inputs, seed=0).trades


# --------------------------------------------------------------- scoring ----
def _stats(values: np.ndarray, hits: np.ndarray, unmatched: int) -> CanaryScore:
    n, k = len(values), len(hits)
    mean = float(values.mean()) if n else math.nan
    se = float(values.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    z = mean / se if n > 1 and se > 0 else math.nan
    hit_rate = float(hits.mean()) if k else math.nan
    hit_se = math.sqrt(0.25 / k) if k else math.nan
    return CanaryScore(n, unmatched, mean, se, z, hit_rate, k, hit_se)


def score_canary(trades: pl.DataFrame, markers: Mapping[int, int]) -> CanaryScore:
    """Captured gross ticks per trade the marker triggered (upstream score_canary).

    A trade belongs to the marker on bar M when its entry filled at M's open (a leaky engine) or at the next
    minute's open (a point-in-time engine); markers are MARK_EVERY_MINUTES apart, so the match is unique."""
    entry_ts = trades["entry_ts"].to_numpy().astype(np.int64)
    side = trades["side"].to_numpy().astype(np.int64)
    captured = side * (trades["exit_price"].to_numpy() - trades["entry_price"].to_numpy()).astype(np.int64)
    direction = _marker_directions(entry_ts, markers)
    late = _marker_directions(entry_ts - NS_PER_MINUTE, markers)
    matched = (direction != 0) | (late != 0)
    values = captured[matched].astype(np.float64)
    hits = values[values != 0] > 0
    return _stats(values, hits, int((~matched).sum()))


def shows_no_edge(score: CanaryScore, jump_ticks: int) -> bool:
    """Honest engine on the leak markers: enough trades, no capture, no z, hit rate near one half."""
    return bool(
        score.trades >= MIN_TRADES
        and score.mean_captured_ticks < jump_ticks / NO_EDGE_MEAN_FRACTION
        and abs(score.z) < MAX_NULL_Z
        and abs(score.hit_rate - 0.5) < HIT_SE_LIMIT * score.hit_se_under_null
    )


def edge_detected(score: CanaryScore, jump_ticks: int) -> bool:
    """A leaky engine on the leak markers, or any engine on the latency markers: the jump is captured."""
    return bool(
        score.trades >= MIN_TRADES
        and score.mean_captured_ticks >= jump_ticks / EDGE_MEAN_FRACTION
        and score.z >= MIN_EDGE_Z
    )


# ----------------------------------------------------------------- suite ----
@dataclass(frozen=True)
class CanaryReport:
    planted_jumps: int
    jump_ticks: int
    leak: CanaryScore  # markers ON the jumped bar
    latency: CanaryScore  # markers one bar early

    @property
    def leak_free(self) -> bool:
        """No edge on the planted future AND the one-bar-early control is captured (the canary can see)."""
        return shows_no_edge(self.leak, self.jump_ticks) and edge_detected(self.latency, self.jump_ticks)

    @property
    def leak_detected(self) -> bool:
        return edge_detected(self.leak, self.jump_ticks)


def planted_synthetic(
    root: str = CANARY_ROOT, first: date = CANARY_FIRST, last: date = CANARY_LAST, seed: int = CANARY_SEED
) -> PlantedBars:
    """Synthetic bars on the root's calendar, then planted (no market data)."""
    return plant_canaries(synthetic_frame(root, first, last, seed=seed), seed)


def run_canaries(engine: CanaryEngine, planted: PlantedBars, root: str = CANARY_ROOT) -> CanaryReport:
    """The reader on the leak markers and on the latency markers, through `engine`."""
    inputs = zero_cost_inputs(root, planted.frame)
    leak = score_canary(engine(canary_strategy(planted.leak_markers, root), planted.frame, inputs),
                        planted.leak_markers)  # fmt: skip
    latency = score_canary(engine(canary_strategy(planted.latency_markers, root), planted.frame, inputs),
                           planted.latency_markers)  # fmt: skip
    return CanaryReport(len(planted.leak_markers), planted.jump_ticks, leak, latency)


__all__ = [
    "JUMP_TICKS",
    "CanaryEngine",
    "CanaryReport",
    "CanaryScore",
    "PlantedBars",
    "canary_reader",
    "canary_strategy",
    "edge_detected",
    "honest_engine",
    "plant_canaries",
    "planted_synthetic",
    "run_canaries",
    "score_canary",
    "shows_no_edge",
    "zero_cost_inputs",
]
