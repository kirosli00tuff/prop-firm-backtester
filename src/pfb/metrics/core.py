"""Run summary: every row of design brief section 4 plus PSR, DSR at the live N and MinBTL (section 5).

ADR-001 sections 9 and 11 with A8 and A14 G8. Inputs are the engine's three polars frames; money columns must
be integer cents and stay integers in every sum. Conventions (the author's research repo `moments`):
per-day Sharpe = mean / POPULATION sd of daily net over every trade date (zero-trade days included), skewness
and RAW kurtosis from population moments, annual = daily x sqrt(252). A ratio with a zero denominator is None.
Imports nothing from pfb outside this package and never reads the registry: N and the grid are arguments.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

import numpy as np
import polars as pl

from pfb.metrics import stats

SUMMARY_SCHEMA = "pfb.metrics/1"
TRADING_DAYS_PER_YEAR = 252
NS_PER_MINUTE = 60_000_000_000
MIN_GRID_CONFIGURATIONS = 2

# The trade ledger's columns, defined once here; the engine builds exactly these, in this order.
TRADE_COLUMNS = (
    "side",
    "contracts",
    "entry_ts",
    "exit_ts",
    "entry_price",
    "exit_price",
    "gross_cents",
    "commission_cents",
    "slippage_cents",
    "net_cents",
    "mae_ticks",
    "mfe_ticks",
    "entry_bar",
    "exit_bar",
    "exit_reason",
    "trade_date",
)
# PRICE_LIMIT: price-limit
EXIT_REASONS = ("SIGNAL", "STOP", "TARGET", "FLAT_BY", "SESSION_END", "PRICE_LIMIT")
EQUITY_COLUMNS = ("position", "eq_close", "eq_low", "eq_high")
DAILY_COLUMNS = ("net_cents",)

Scalar = int | float | str | bool | None
Summary = dict[str, Scalar]


class MetricsInputError(ValueError):
    """The frames or arguments handed to summarize break the input contract."""


# --------------------------------------------------------------------------- input checks


def _require(frame: pl.DataFrame, name: str, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise MetricsInputError(f"{name} frame lacks columns {missing}")


def _ints(frame: pl.DataFrame, name: str, column: str) -> np.ndarray:
    series = frame.get_column(column)
    if not series.dtype.is_integer():
        raise MetricsInputError(f"{name}.{column} must be an integer column, got {series.dtype}")
    if series.null_count():
        raise MetricsInputError(f"{name}.{column} holds {series.null_count()} null values")
    return series.cast(pl.Int64).to_numpy()


def _check_ledger_labels(trades: pl.DataFrame) -> None:
    """exit_reason is text from EXIT_REASONS, trade_date a Date, the bar indices integers (ADR section 9)."""
    for column in ("entry_bar", "exit_bar"):
        _ints(trades, "trades", column)
    if trades.schema["trade_date"] != pl.Date or trades["trade_date"].null_count():
        raise MetricsInputError("trades.trade_date must be a Date column without nulls")
    reasons = trades["exit_reason"]
    if reasons.dtype != pl.String or not reasons.is_in(list(EXIT_REASONS)).all():
        raise MetricsInputError(f"trades.exit_reason must be text from {EXIT_REASONS}")


def _check_args(
    n_trials: int | None, capital_base_cents: int | None, grid_daily_sharpes: Sequence[float] | None
) -> None:
    if n_trials is not None and (isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1):
        raise MetricsInputError(f"n_trials must be an int >= 1 or None, got {n_trials!r}")
    if capital_base_cents is not None and (
        isinstance(capital_base_cents, bool)
        or not isinstance(capital_base_cents, int)
        or capital_base_cents <= 0
    ):
        raise MetricsInputError(f"capital_base_cents must be a positive int or None: {capital_base_cents!r}")
    if grid_daily_sharpes is not None and not all(math.isfinite(float(v)) for v in grid_daily_sharpes):
        raise MetricsInputError("grid_daily_sharpes holds a non-finite value")


def _ratio(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


# --------------------------------------------------------------------------- blocks


def _pnl_block(trades: pl.DataFrame) -> Summary:
    net = _ints(trades, "trades", "net_cents")
    n = len(net)
    wins, losses = net[net > 0], net[net < 0]
    win_sum, loss_sum = int(wins.sum()), int(losses.sum())
    ticks = (
        (_ints(trades, "trades", "exit_price") - _ints(trades, "trades", "entry_price"))
        * _ints(trades, "trades", "side")
        * _ints(trades, "trades", "contracts")
    )
    mae, mfe = np.abs(_ints(trades, "trades", "mae_ticks")), np.abs(_ints(trades, "trades", "mfe_ticks"))
    avg_win, avg_loss = _ratio(win_sum, len(wins)), _ratio(loss_sum, len(losses))
    return {
        "net_cents": int(net.sum()),
        "gross_cents": int(_ints(trades, "trades", "gross_cents").sum()),
        "commission_cents": int(_ints(trades, "trades", "commission_cents").sum()),
        "slippage_cents": int(_ints(trades, "trades", "slippage_cents").sum()),
        "n_trades": n,
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate": _ratio(len(wins), n),
        "profit_factor": _ratio(win_sum, abs(loss_sum)),
        "expectancy_cents": _ratio(int(net.sum()), n),
        "expectancy_gross_ticks": _ratio(int(ticks.sum()), n),
        "avg_win_cents": avg_win,
        "avg_loss_cents": avg_loss,
        "payoff_ratio": None if avg_win is None or avg_loss is None else avg_win / abs(avg_loss),
        "mae_ticks_mean": _ratio(int(mae.sum()), n),
        "mae_ticks_worst": int(mae.max()) if n else None,
        "mfe_ticks_mean": _ratio(int(mfe.sum()), n),
        "mfe_ticks_best": int(mfe.max()) if n else None,
    }


def _daily_block(nets: np.ndarray) -> tuple[Summary, dict[str, float | int] | None]:
    """Daily statistics and the moments (None when no day has a non-zero sd)."""
    n = len(nets)
    total = int(nets.sum())
    mom = stats.moments([float(v) for v in nets]) if n else None
    flat = mom is None or mom["sd"] == 0
    sharpe = None if flat or mom is None else float(mom["sharpe"])
    downside_sq = int((np.minimum(nets, 0) ** 2).sum())
    mean = _ratio(total, n)
    out: Summary = {
        "n_days": n,
        "mean_daily_net_cents": mean,
        "sd_daily_net_cents": None if mom is None else float(mom["sd"]),
        "skewness_daily": None if flat or mom is None else float(mom["skew"]),
        "kurtosis_daily": None if flat or mom is None else float(mom["kurtosis"]),
        "sharpe_daily": sharpe,
        "sharpe_annual": None if sharpe is None else sharpe * math.sqrt(TRADING_DAYS_PER_YEAR),
        "sortino_annual": (
            None
            if mean is None or downside_sq == 0
            else mean / math.sqrt(downside_sq / n) * math.sqrt(TRADING_DAYS_PER_YEAR)
        ),
        "annualized_net_cents": None if mean is None else mean * TRADING_DAYS_PER_YEAR,
        "best_day_cents": int(nets.max()) if n else None,
        "best_day_share": int(nets.max()) / total if n and total > 0 else None,
    }
    return out, None if flat else mom


def _drawdown_block(equity: pl.DataFrame) -> Summary:
    """Peak-to-trough on eq_close and the intrabar worst case (A8); equity is 0 before the first bar."""
    close = _ints(equity, "equity_minute", "eq_close")
    low, high = _ints(equity, "equity_minute", "eq_low"), _ints(equity, "equity_minute", "eq_high")
    if not len(close):
        return {"max_drawdown_cents": 0, "max_drawdown_intrabar_cents": 0}
    peak_close = np.maximum(np.maximum.accumulate(close), 0)
    peak_high = np.maximum(np.maximum.accumulate(high), 0)
    return {
        "max_drawdown_cents": int((peak_close - close).max()),
        "max_drawdown_intrabar_cents": int((peak_high - low).max()),
    }


def _exposure_block(trades: pl.DataFrame, equity: pl.DataFrame, n_days: int) -> Summary:
    n = trades.height
    held_ns = int((_ints(trades, "trades", "exit_ts") - _ints(trades, "trades", "entry_ts")).sum())
    position = _ints(equity, "equity_minute", "position")
    return {
        "trades_per_day": _ratio(n, n_days),
        "avg_hold_minutes": _ratio(held_ns, n * NS_PER_MINUTE),
        "n_minutes": len(position),
        "share_minutes_in_position": _ratio(int((position != 0).sum()), len(position)),
    }


# --------------------------------------------------------------------------- PSR, DSR, MinBTL


def _psr(mom: dict[str, float | int], benchmark: float, label: str, warnings: list[str]) -> float | None:
    try:
        return stats.probabilistic_sharpe_ratio(
            float(mom["sharpe"]), int(mom["n"]), float(mom["skew"]), float(mom["kurtosis"]), benchmark
        )
    except ValueError as exc:
        warnings.append(f"{label}: {exc}")
        return None


def _grid_variance(grid: Sequence[float] | None) -> float | None:
    if grid is None or len(grid) < MIN_GRID_CONFIGURATIONS:
        return None
    return statistics.pvariance([float(v) for v in grid])


def _dsr_block(
    mom: dict[str, float | int] | None,
    n_days: int,
    sharpe_annual: float | None,
    n_trials: int | None,
    grid: Sequence[float] | None,
    warnings: list[str],
) -> Summary:
    keys = (
        "dsr",
        "dsr_floor",
        "dsr_grid",
        "dsr_variance",
        "dsr_variance_floor",
        "dsr_variance_grid",
        "dsr_bracket",
        "dsr_sr0_daily",
        "minbtl_years",
    )
    out: Summary = dict.fromkeys(keys)
    out["n_trials"], out["dsr_n_grid"] = n_trials, 0 if grid is None else len(grid)
    if n_trials is None:
        return out
    b = stats.bracket(n_trials)
    floor = 1.0 / (n_days - 1) if n_days >= 2 else None
    grid_var = _grid_variance(grid)
    variance = None if floor is None else max(floor, grid_var if grid_var is not None else floor)
    out.update(dsr_bracket=b, dsr_variance_floor=floor, dsr_variance_grid=grid_var, dsr_variance=variance)
    if variance is not None:
        out["dsr_sr0_daily"] = math.sqrt(variance) * b
    if sharpe_annual is not None and sharpe_annual > 0:  # the square would hide a negative Sharpe
        out["minbtl_years"] = stats.min_btl_years(n_trials, sharpe_annual)
    if mom is None or variance is None or floor is None:
        return out
    out["dsr"] = _psr(mom, math.sqrt(variance) * b, "DSR", warnings)
    out["dsr_floor"] = _psr(mom, math.sqrt(floor) * b, "DSR at the floor variance", warnings)
    if grid_var is not None:
        out["dsr_grid"] = _psr(mom, math.sqrt(grid_var) * b, "DSR at the grid variance", warnings)
    return out


# --------------------------------------------------------------------------- entry point


def _check_consistency(net_cents: int, daily_nets: np.ndarray, equity: pl.DataFrame) -> None:
    if int(daily_nets.sum()) != net_cents:
        raise MetricsInputError(f"daily net_cents sum {int(daily_nets.sum())} != ledger net {net_cents}")
    close = _ints(equity, "equity_minute", "eq_close")
    if len(close) and int(close[-1]) != net_cents:
        raise MetricsInputError(f"last eq_close {int(close[-1])} != ledger net {net_cents}")


def summarize(
    trades: pl.DataFrame,
    equity_minute: pl.DataFrame,
    daily: pl.DataFrame,
    *,
    n_trials: int | None,
    capital_base_cents: int | None = None,
    grid_daily_sharpes: Sequence[float] | None = None,
) -> dict[str, object]:
    """JSON-ready summary of one run (values are int, float, str, bool or None).

    n_trials is the registry's N after the run's own begin record (None leaves every DSR field and MinBTL
    null). grid_daily_sharpes are the per-day Sharpes of every configuration of the same hypothesis with an
    "ok" result, this run included (A14 G8); with fewer than two only the floor 1 / (T - 1) applies.
    capital_base_cents is carried as a declared field and does not enter any ratio.
    """
    _require(trades, "trades", TRADE_COLUMNS)
    _require(equity_minute, "equity_minute", EQUITY_COLUMNS)
    _require(daily, "daily", DAILY_COLUMNS)
    _check_ledger_labels(trades)
    _check_args(n_trials, capital_base_cents, grid_daily_sharpes)
    pnl = _pnl_block(trades)
    daily_nets = _ints(daily, "daily", "net_cents")
    _check_consistency(int(pnl["net_cents"] or 0), daily_nets, equity_minute)
    day, mom = _daily_block(daily_nets)
    dd = _drawdown_block(equity_minute)
    annual_net, dd_intrabar = day["annualized_net_cents"], int(dd["max_drawdown_intrabar_cents"] or 0)
    warnings: list[str] = []
    sharpe_annual = day["sharpe_annual"]
    summary: Summary = {
        "summary_schema": SUMMARY_SCHEMA,
        **pnl,
        **day,
        **dd,
        "calmar": None if not isinstance(annual_net, float) else _ratio(annual_net, dd_intrabar),
        **_exposure_block(trades, equity_minute, len(daily_nets)),
        "capital_base_cents": capital_base_cents,
        "psr": None if mom is None else _psr(mom, 0.0, "PSR", warnings),
        **_dsr_block(
            mom,
            len(daily_nets),
            sharpe_annual if isinstance(sharpe_annual, float) else None,
            n_trials,
            grid_daily_sharpes,
            warnings,
        ),
    }
    summary["metrics_warnings"] = "; ".join(warnings) if warnings else None
    return dict(summary)
