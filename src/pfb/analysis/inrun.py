"""The in-run analysis suite: computed inside pfb.engine.run.run_strategy on the bar frame the run already
loaded (no second bar read), returned as summary["suite"] plus extras frames (runs/<run_id>/<name>.parquet).

Parts: cost_stress (S1), baseline (S2), breakdowns (S6), walk_forward (S4, pfb.analysis.walkforward), power
(S3, pfb.analysis.power, on the run's realized figures), sizing (S7). A part turned off, or one that does not
apply (power without n_trials), is None. The suite is an output of the one configuration (docs/TRIALS.md C4).

A part that cannot be computed as specified (one of SUITE_PART_ERRORS: SuiteError; the
baseline's BaselineError; the walk-forward's FoldError; the power check's PowerInputError) is None too, and
compute_suite returns a warning "suite: <part> not computed: <error class>: <message>" for it; the engine adds
it to the summary's warnings, so it reaches the result record and the contract. The run itself is not
affected. A failed cross-check of the base run's own trades, days or nets (SuiteIntegrityError: ledger net
against daily net, a run day without a bar, bucket sums, a real trade that is not an eligible placement or
exits at its entry bar's open, a stress replay with another trade count) is not a part error: it propagates
and the run fails. Any other exception is a bug and propagates too.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import polars as pl

from pfb.analysis import baseline as B
from pfb.analysis.breakdowns import breakdowns
from pfb.analysis.power import PowerInputError
from pfb.analysis.sizing import load_sizing_plan, sizing_inputs, sizing_table
from pfb.analysis.spec import SuiteError, SuiteSpec
from pfb.analysis.stress import cost_stress
from pfb.analysis.walkforward import FoldError
from pfb.engine.run import (
    NS_PER_MINUTE,
    MarketInputs,
    PreparedBars,
    SimOutput,
    flat_gap_ends,
    in_intervals,
    prepare_bars,
)
from pfb.engine.strategy import ParamValue, Strategy, make_bars, normalize_signals
from pfb.instruments.table import InstrumentTable
from pfb.metrics import stats

SUITE_KEYS = ("cost_stress", "baseline", "breakdowns", "walk_forward", "power", "sizing")
TRADING_DAYS_PER_YEAR = 252
MIN_MOMENT_TRADES = 3
DEFAULT_SKEW, DEFAULT_KURTOSIS = 0.0, 3.0
# The known errors of a suite part: the part becomes None with a warning. FoldError is a
# SuiteError; it is named for the reader.
SUITE_PART_ERRORS: tuple[type[ValueError], ...] = (SuiteError, B.BaselineError, FoldError, PowerInputError)
SUITE_WARNING_PREFIX = "suite: "


@dataclass(frozen=True)
class RunContext:
    """What run_strategy holds after the base simulation: the inputs to simulate and its output."""

    strategy: Strategy
    frame: pl.DataFrame  # the bars as loaded, before prepare_bars
    params: Mapping[str, ParamValue]
    inputs: MarketInputs
    seed: int
    entry_allowed: np.ndarray | None
    cost_model: str
    root: str
    instruments: InstrumentTable
    sim: SimOutput
    n_trials: int | None


# ------------------------------------------------------------------------------------ market ----
def bar_slippage(ctx: RunContext, minute: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Buy and sell slippage of each usable bar as the kernel ran with them (ctx.sim.replay.kernel_in): under
    event_costs "on" the per-bar prices from the release calendar, under "off" exactly the
    minute table at each bar's CT minute. A SimOutput without a replay (built by hand) reads the minute
    table."""
    replay = ctx.sim.replay
    if replay is None:
        table = ctx.inputs.slippage_cents
        return table[minute, 0].astype(np.int64), table[minute, 1].astype(np.int64)
    kernel_in = replay.kernel_in
    if len(kernel_in.slip_buy) != len(minute) or len(kernel_in.slip_sell) != len(minute):
        raise RuntimeError("the kernel's per-bar slippage does not cover the usable bars")  # an invariant
    return np.asarray(kernel_in.slip_buy, dtype=np.int64), np.asarray(kernel_in.slip_sell, dtype=np.int64)


def baseline_market(ctx: RunContext, prepared: PreparedBars) -> B.BaselineMarket:
    """The usable bars and the engine's entry and segment rules, as pfb.engine.run.simulate builds them; the
    price-limit entry refusals come from ctx.sim.price_limits (upper, lower, no_reference), the arrays the
    kernel ran with."""
    bars = prepared.frame
    ts = bars["ts_event"].to_numpy()
    minute = bars["ct_minute"].to_numpy().astype(np.int64)
    starts, ends = ctx.inputs.must_be_flat_starts, ctx.inputs.must_be_flat_ends
    flat_now = in_intervals(ts, starts, ends)
    session_end = prepared.session_end | flat_gap_ends(ts, starts, ends)
    entry_fill_ok = ~flat_now & ~in_intervals(ts + NS_PER_MINUTE, starts, ends)
    allowed = ctx.entry_allowed if ctx.entry_allowed is not None else np.ones(len(ts), dtype=np.bool_)
    day_int = ctx.sim.daily["trade_date"].cast(pl.Int32).to_numpy()
    day_idx = np.searchsorted(day_int, bars["trade_date"].cast(pl.Int32).to_numpy()).astype(np.int64)
    close = bars["close"].to_numpy().astype(np.int64)
    # the kernel's own per-bar price-limit arrays; None: the rule refuses nothing
    limits = ctx.sim.price_limits
    refused = (
        None
        if limits is None
        else B.price_limit_refusals(close, limits.upper, limits.lower, limits.no_reference)
    )
    slip_buy, slip_sell = bar_slippage(ctx, minute)
    return B.BaselineMarket(
        open=bars["open"].to_numpy().astype(np.int64), close=close,
        slip_buy=slip_buy, slip_sell=slip_sell,
        ct_minute=minute, day_idx=day_idx,
        entry_ok=B.entry_ok_mask(session_end, entry_fill_ok, allowed, refused),
        room=B.segment_room(session_end, flat_now), n_days=len(day_int),
        commission_cents=ctx.inputs.commission_cents, tick_value_cc=ctx.inputs.tick_value_cc,
        contracts=ctx.strategy.spec.contracts,
    )  # fmt: skip


def _baseline(ctx: RunContext, prepared: PreparedBars, spec: SuiteSpec) -> tuple[object, pl.DataFrame | None]:
    if spec.baseline_draws == 0:
        return None, None
    market = baseline_market(ctx, prepared)
    daily = ctx.sim.daily["net_cents"].to_numpy().astype(np.int64)
    trades = B.trades_of(ctx.sim.trades, after_lock=ctx.sim.exit_after_lock)
    return B.random_baseline(market, trades, daily, draws=spec.baseline_draws,
                             seed=spec.baseline_seed)  # fmt: skip


# ---------------------------------------------------------------------------- walk-forward, power ----
def daily_with_trades(sim: SimOutput) -> pl.DataFrame:
    counts = sim.trades.group_by("trade_date").agg(pl.len().cast(pl.Int64).alias("trades"))
    joined = sim.daily.select("trade_date", "net_cents").join(counts, on="trade_date", how="left",
                                                               maintain_order="left")  # fmt: skip
    return joined.with_columns(pl.col("trades").fill_null(0))


def _walk_forward(ctx: RunContext, spec: SuiteSpec) -> object:
    if spec.walk_forward_folds == 0:
        return None
    from pfb.analysis.walkforward import fold_report

    return fold_report(daily_with_trades(ctx.sim), spec.walk_forward_folds)


def trade_moments(net: np.ndarray) -> tuple[float, float]:
    """Population skewness and RAW kurtosis of the trade nets (pfb.metrics.stats.moments); the normal
    defaults (0, 3) with fewer than MIN_MOMENT_TRADES trades or a zero sd."""
    if len(net) < MIN_MOMENT_TRADES:
        return DEFAULT_SKEW, DEFAULT_KURTOSIS
    mom = stats.moments([float(v) for v in net.tolist()])
    if mom["sd"] == 0:
        return DEFAULT_SKEW, DEFAULT_KURTOSIS
    return float(mom["skew"]), float(mom["kurtosis"])


def realized_power(ctx: RunContext) -> dict[str, object] | None:
    """S3 at the run's realized trades per year, per-trade population sd and days / 252, N = n_trials."""
    if ctx.n_trials is None:
        return None
    from pfb.analysis.power import min_edge

    net = ctx.sim.trades["net_cents"].to_numpy().astype(np.int64)
    days = ctx.sim.daily.height
    years = days / TRADING_DAYS_PER_YEAR
    edge = int(net.sum()) // len(net) if len(net) else None
    sd = round(float(net.std())) if len(net) >= 2 else 0
    skew, kurt = trade_moments(net)
    base: dict[str, object] = {"realized_edge_cents": edge, "trades": len(net), "days": days, "years": years,
                               "trade_sd_cents": sd, "trade_skew": skew, "trade_kurtosis": kurt}  # fmt: skip
    if len(net) < 2 or sd <= 0 or days < 2:
        return base | {
            "realized_min_edge_cents": None,
            "realized_min_edge_cents_alpha": None,
            "passed": None,
            "reason": "fewer than two trades, a zero per-trade sd or fewer than two days",
        }
    result = min_edge(trades_per_year=len(net) / years, trade_sd_cents=sd, years=years, n_trials=ctx.n_trials,
                      target_edge_cents=edge, trade_skew=skew, trade_kurtosis=kurt)  # fmt: skip
    return base | {"realized_min_edge_cents": result.min_edge_cents,
                   "realized_min_edge_cents_alpha": result.min_edge_cents_alpha, "passed": result.passed,
                   "detail": result.to_json()}  # fmt: skip


# ------------------------------------------------------------------------------------ sizing ----
def _stop_ticks(ctx: RunContext, prepared: PreparedBars) -> np.ndarray:
    """Each trade's stop distance, read at its entry decision bar from one more generate call."""
    bars = make_bars(prepared.frame)
    signals = normalize_signals(ctx.strategy.generate(bars, dict(ctx.params)), len(bars["ts_event"]))
    decision = ctx.sim.trades["entry_bar"].to_numpy().astype(np.int64) - 1
    return np.asarray(signals.stop_ticks, dtype=np.int64)[decision]


def _sizing(ctx: RunContext, prepared: PreparedBars, spec: SuiteSpec) -> dict[str, object]:
    plan = None
    if spec.sizing_plan:
        plan = load_sizing_plan(spec.sizing_plan, ctx.instruments.get(ctx.root).contract_type)
    stops = _stop_ticks(ctx, prepared) if ctx.sim.trades.height else None
    inputs = sizing_inputs(ctx.sim.trades, ctx.sim.daily["trade_date"], ctx.strategy.spec.contracts,
                           ctx.inputs.tick_value_cc, stops)  # fmt: skip
    return sizing_table(inputs, plan)


# ------------------------------------------------------------------------------------- entry ----
def _clean(value: object) -> object:
    """NaN and infinities to None, recursively, so the suite is strict JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    return value


def suite_error_warning(part: str, exc: Exception) -> str:
    """The warning for a part that a known suite error left None."""
    return f"{SUITE_WARNING_PREFIX}{part} not computed: {type(exc).__name__}: {exc}"


def _guarded[T](part: str, compute: Callable[[], T], failed: T) -> tuple[T, tuple[str, ...]]:
    """compute(), or `failed` and one warning when it raises one of SUITE_PART_ERRORS."""
    try:
        return compute(), ()
    except SUITE_PART_ERRORS as exc:
        return failed, (suite_error_warning(part, exc),)


def compute_suite(
    ctx: RunContext, spec: SuiteSpec
) -> tuple[dict[str, object], dict[str, pl.DataFrame], tuple[str, ...]]:
    """summary["suite"] (keys SUITE_KEYS, a part off or failed is None, plus "spec"), the extras frames and
    the warnings of the parts a known suite error left None, in SUITE_KEYS order."""
    prepared = prepare_bars(ctx.frame, ctx.inputs)
    stress, w_stress = _guarded("cost_stress", lambda: cost_stress(ctx, spec.cost_stress_ticks), None)
    (base, draws), w_base = _guarded("baseline", lambda: _baseline(ctx, prepared, spec), (None, None))
    (tables, rolling), w_break = _guarded(
        "breakdowns", lambda: breakdowns(ctx.sim.trades, ctx.sim.daily, prepared.frame), (None, None)
    )
    walk, w_walk = _guarded("walk_forward", lambda: _walk_forward(ctx, spec), None)
    power, w_power = _guarded("power", lambda: realized_power(ctx), None)
    sizing, w_sizing = _guarded("sizing", lambda: _sizing(ctx, prepared, spec), None)
    extras: dict[str, pl.DataFrame] = {}
    if draws is not None:
        extras["baseline_draws"] = draws
    if rolling is not None:
        extras["rolling_sharpe"] = rolling
    suite: dict[str, object] = {
        "cost_stress": stress,
        "baseline": base,
        "breakdowns": tables,
        "walk_forward": walk,
        "power": power,
        "sizing": sizing,
        "spec": spec.to_json(),
    }
    warnings = (*w_stress, *w_base, *w_break, *w_walk, *w_power, *w_sizing)
    return dict(_clean(suite)), extras, warnings  # type: ignore[call-overload]
