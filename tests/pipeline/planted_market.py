"""Planted-edge market for the whole-pipeline recovery tests. Synthetic data only; nothing here
reads a bar store, registers a hypothesis or writes a ledger.

The market is pfb.engine.synthetic.synthetic_frame (the seeded random walk on the root's own calendar) plus,
for every trade date d and slot j of strategies/planted_slots.py, a transient shift X[d, j] ticks on the bars
of CT minutes [420 + 12 j + 3, 420 + 12 j + 11): the strategy decides at minute 420 + 12 j, fills at the next
bar's open (offset 1, or 2 when that minute is one of the frame's absent minutes), decides its exit at offset
6..9 and fills at offset 7..10, so every trade holds exactly X[d, j] plus the walk's own noise. X[d, j] =
rint(Normal(mu_ticks, sigma_ticks)) from a second seeded stream; the shift is undone at offset 11, so nothing
accumulates and the price stays near the walk (far from any price limit).

Truth: mean gross per trade = mu_ticks x tick value (the rounding of a normal with sigma >= 10 ticks moves the
mean by less than 1e-80 ticks), mean net = gross - the run's cost per round trip (2 x commission per side +
2 x fixed slippage per side, read from the repo's fee and instrument config, never typed in here).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from fractions import Fraction
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import pfb.engine.run as run_module
from pfb.analysis.power import TRADING_DAYS_PER_YEAR, PowerResult, min_edge
from pfb.analysis.spec import SuiteSpec
from pfb.costs.fees import load_fees
from pfb.costs.model import load_cost_model
from pfb.engine.run import RunResult
from pfb.engine.synthetic import synthetic_frame
from pfb.instruments import load_instruments
from pfb.instruments.sessions import open_intervals
from pfb.metrics.stats import bracket, norm_cdf, norm_ppf

HERE = Path(__file__).resolve().parent
PLANTED_SLOTS = HERE / "strategies" / "planted_slots.py"
ROOT = "MNQ"  # a hard-limit root (equity bands): the first trade date opens nothing (no settlement reference)
FEES = "topstep"
COST_MODEL = "fixed:1"  # one tick per side: a fixed, known charge (the book table needs config/paths.toml)
N_TRIALS = 1001  # an arbitrary trial count (the power figure and the DSR use it)
FIRST_DATE = date(2023, 1, 3)
FIRST_MINUTE, SPACING = 420, 12
SHIFT_FIRST, SHIFT_END = 3, 11  # offsets in the slot holding the plant (half-open)
SIGMA_TICKS = 40  # planted per-trade sd; with the walk's ~6.4 ticks over a hold the trade sd is ~40.5 ticks
PLANT_STREAM = 24  # second seed word of the plant's draws (the walk uses the run seed alone)
PSR_PASS = 0.95  # a detection: PSR (alpha test) or DSR at N (the deflated test) above 0.95, one-sided 5 %
WALK_STEP_VAR = Fraction(20, 3)  # var of U{-4..4}: the close-to-close step of pfb.engine.synthetic
JITTER_VAR = Fraction(2, 3)  # var of U{-1, 0, 1}: an open's jitter on the prior close
HOLD_STEPS = 6  # entry fill at offset 1, exit fill at offset 7 on a bar with no absent minute
PRICE_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class Case:
    """One stated trade count: slots per day x days with entries (the first trade date opens nothing)."""

    trades: int
    slots: int
    days_with_entries: int

    @property
    def run_days(self) -> int:
        return self.days_with_entries + 1


CASES = {250: Case(250, 5, 50), 1000: Case(1000, 10, 100), 4000: Case(4000, 40, 100)}


def trade_dates(root: str, first: date, count: int) -> list[date]:
    """The first ``count`` trade dates of the root's calendar from ``first`` on."""
    _s, _e, days = open_intervals(root, first, first + timedelta(days=2 * count + 30))
    unique = np.unique(np.asarray(days, dtype=np.int64))[:count]
    if len(unique) < count:
        raise ValueError(f"only {len(unique)} trade dates after {first}")
    return [date(1970, 1, 1) + timedelta(days=int(d)) for d in unique]


def window(case: Case) -> tuple[date, date]:
    dates = trade_dates(ROOT, FIRST_DATE, case.run_days)
    return dates[0], dates[-1]


@cache
def cost_per_trade_cents(root: str = ROOT) -> int:
    """2 x commission per side (fees config) + 2 x the fixed:1 slippage per side (cost model), 1 contract."""
    instruments = load_instruments()
    commission = load_fees(FEES, instruments=instruments).commission_side_cents(root)
    slip = np.asarray(load_cost_model(COST_MODEL, instruments, None).slippage_cents(root, 1), dtype=np.int64)
    if not (slip == slip[0, 0]).all():
        raise AssertionError("fixed:1 slippage is not one figure for every minute and side")
    return 2 * int(commission) + 2 * int(slip[0, 0])


@cache
def tick_value_cents(root: str = ROOT) -> Fraction:
    return Fraction(int(load_instruments().get(root).tick_value_cc), 100)


def trade_sd_cents(sigma_ticks: int = SIGMA_TICKS) -> float:
    """The design's per-trade sd: the plant, its rounding (1/12) and the walk over a 6-step hold."""
    var_ticks = sigma_ticks**2 + Fraction(1, 12) + HOLD_STEPS * WALK_STEP_VAR + 2 * JITTER_VAR
    return math.sqrt(float(var_ticks)) * float(tick_value_cents())


def mu_ticks_for_net(net_cents: float) -> float:
    """The plant's mean in ticks for a true mean NET per trade of net_cents."""
    return (net_cents + cost_per_trade_cents()) / float(tick_value_cents())


def plant_shift(frame: pl.DataFrame, *, seed: int, mu_ticks: float, slots: int, sigma: int) -> np.ndarray:
    minute = frame["ct_minute"].to_numpy().astype(np.int64)
    days = frame["trade_date"].cast(pl.Int32).to_numpy().astype(np.int64)
    _u, day_idx = np.unique(days, return_inverse=True)
    rng = np.random.default_rng([seed, PLANT_STREAM])
    draws = np.rint(rng.normal(mu_ticks, sigma, (len(_u), slots))).astype(np.int64)
    rel = minute - FIRST_MINUTE
    slot, offset = rel // SPACING, rel % SPACING
    on = (rel >= 0) & (slot < slots) & (offset >= SHIFT_FIRST) & (offset < SHIFT_END)
    return np.where(on, draws[day_idx, np.clip(slot, 0, slots - 1)], 0).astype(np.int64)


def planted_maker(mu_ticks: float, slots: int, sigma: int = SIGMA_TICKS) -> Callable[..., pl.DataFrame]:
    """A drop-in for pfb.engine.run.synthetic_frame with the plant added."""

    def make(root: str, first: date, last: date, *, seed: int) -> pl.DataFrame:
        frame = synthetic_frame(root, first, last, seed=seed)
        shift = pl.Series(plant_shift(frame, seed=seed, mu_ticks=mu_ticks, slots=slots, sigma=sigma))
        return frame.with_columns(pl.col(c) + shift for c in PRICE_COLUMNS)

    return make


def run_planted(case: Case, net_cents: float, seed: int, suite: SuiteSpec | None) -> RunResult:
    """One whole-pipeline run (strategy, engine with the price-limit rule, costs, metrics, the suite) on the
    planted market with a true mean net per trade of net_cents."""
    first, last = window(case)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run_module, "synthetic_frame", planted_maker(mu_ticks_for_net(net_cents), case.slots))
        return run_module.run_strategy(PLANTED_SLOTS, source="synthetic", first=first, last=last,
                                       params={"slots": case.slots}, fees=FEES, cost_model=COST_MODEL,
                                       seed=seed, n_trials=N_TRIALS, suite=suite)  # fmt: skip


@dataclass(frozen=True)
class Outcome:
    """What one seed's run leaves for the tests (no frame is kept: memory stays at one run)."""

    trades: int
    mean_net: float
    sd_net: float
    psr: float | None
    dsr: float | None
    power_passed: bool | None
    forced_exits: int
    off_cost_trades: int

    @property
    def se(self) -> float:
        return self.sd_net / math.sqrt(self.trades)


def outcome(result: RunResult) -> Outcome:
    net = result.trades["net_cents"].to_numpy().astype(np.int64)
    charged = (result.trades["commission_cents"] + result.trades["slippage_cents"]).to_numpy()
    suite = result.summary.get("suite")
    power = suite.get("power") if isinstance(suite, dict) else None
    psr, dsr = result.summary["psr"], result.summary["dsr"]
    return Outcome(
        trades=len(net), mean_net=float(net.mean()), sd_net=float(net.std(ddof=1)),
        psr=psr if isinstance(psr, float) else None, dsr=dsr if isinstance(dsr, float) else None,
        power_passed=power.get("passed") if isinstance(power, dict) else None,
        forced_exits=int(result.counters.get("price_limit_exits", 0)),
        off_cost_trades=int((charged != cost_per_trade_cents()).sum()),
    )  # fmt: skip


@cache
def probe_trades(trades: int) -> int:
    """The trades the calendar lets the strategy take in the case's window (seed 0): early closes and the
    frame's absent minutes take a few slots, so the module's power is evaluated at this count."""
    return run_planted(CASES[trades], 0.0, 0, None).trades.height


# ------------------------------------------------------------------------------- power oracle ----
def power_at(case: Case, trades: int, n_trials: int = N_TRIALS) -> PowerResult:
    """pfb.analysis.power.min_edge at the case's design: ``trades`` trades (the count the calendar lets the
    strategy take, see probe_trades), run days / 252 years, the design sd."""
    years = case.run_days / TRADING_DAYS_PER_YEAR
    return min_edge(trades_per_year=trades / years, trade_sd_cents=round(trade_sd_cents()), years=years,
                    n_trials=n_trials)  # fmt: skip


def predicted_power(
    case: Case, trades: int, edge_cents: float, *, deflated: bool, exact_design: bool = False
) -> float:
    """The power at a planted edge. The statistic SR_daily sqrt(T - 1) has mean c sqrt(n (T - 1) / T) and
    sd k; the test passes above b + z95 k (deflated, N = N_TRIALS) or z95 k (alpha).

    exact_design False: the module's own formula (docs/TRIALS.md s5, normal trades): c = mu / sqrt(s^2 + mu^2)
    (compound-Poisson daily variance) and each test's own k with daily raw kurtosis 3 + 3 / lam: at the
    deflated Sharpe s0 = (b + z) / sqrt(T - 1) for the deflated test, at z / sqrt(T - 1) for the alpha test
    (alpha rule 2). At min_edge_cents and min_edge_cents_alpha this is 0.80.
    exact_design True: this design's own large-T power: a FIXED count of normal trades per day, so c = mu / s
    and normal daily nets (raw kurtosis 3), with k = sqrt(1 + SR^2 / 2) at the Sharpe the edge gives (the
    PSR's own variance term at that Sharpe). It is the upper end of the bracket: the module's figure is the
    conservative one (the alpha k of the deflated Sharpe was)."""
    p = power_at(case, trades)
    s = float(p.trade_sd_cents)
    n, days = p.expected_trades, p.days
    b = bracket(N_TRIALS)
    z95 = norm_ppf(1.0 - 0.05)
    if exact_design:
        mean = edge_cents / s * math.sqrt(n * (days - 1) / days)
        sharpe = mean / math.sqrt(days - 1)
        k = math.sqrt(1.0 + sharpe * sharpe / 2.0)
    else:
        z = z95 + norm_ppf(0.80)
        lam = p.trades_per_year / TRADING_DAYS_PER_YEAR
        g4 = 3.0 + 3.0 / lam
        s0 = ((b if deflated else 0.0) + z) / math.sqrt(days - 1)
        k = math.sqrt(max(1.0, 1.0 + ((g4 - 1.0) / 4.0) * s0 * s0))
        c = edge_cents / math.sqrt(s * s + edge_cents * edge_cents)
        mean = c * math.sqrt(n * (days - 1) / days)
    threshold = (b if deflated else 0.0) + z95 * k
    return norm_cdf((mean - threshold) / k)


def binomial_se(p: float, draws: int) -> float:
    return math.sqrt(max(p * (1.0 - p), 1e-12) / draws)
