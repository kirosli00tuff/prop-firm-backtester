"""S6 breakdowns and trade analytics, from the frames the run holds in memory (no bar read).

Every Sharpe here is the summary's: mean / population sd of a daily net series x sqrt(252), zero-trade days
included, None when the sd is 0. A bucket of days (calendar year, volatility tercile, weekday) uses the run's
days in that bucket; a bucket of trades (CT hour of entry) uses every run day, each day carrying only that
hour's trades. Nets of every breakdown sum to the run's net (checked; a mismatch is an error).
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from pfb.analysis.spec import SuiteIntegrityError

TRADING_DAYS_PER_YEAR = 252
NS_PER_MINUTE = 60_000_000_000
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
ROLLING_WINDOWS = (63, 252)
TERCILES = 3
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def sharpe_annual(nets: np.ndarray) -> float | None:
    """The summary's Sharpe of one daily series; None for no day or sd 0."""
    if len(nets) == 0:
        return None
    x = nets.astype(np.float64)
    sd = float(x.std())
    return None if sd == 0 else float(x.mean()) / sd * math.sqrt(TRADING_DAYS_PER_YEAR)


def quantiles(values: np.ndarray) -> dict[str, int | None]:
    """5/25/50/75/95 quantiles as observed values (inverted CDF), None when empty."""
    keys = [f"q{round(q * 100):02d}" for q in QUANTILES]
    if len(values) == 0:
        return dict.fromkeys(keys)
    qs = np.quantile(values, QUANTILES, method="inverted_cdf")
    return {k: int(v) for k, v in zip(keys, qs, strict=True)}


# ---------------------------------------------------------------------------------- buckets ----
def _day_buckets(keys: np.ndarray, labels: list[object], nets: np.ndarray,
                 trades_per_day: np.ndarray) -> list[dict[str, object]]:  # fmt: skip
    out: list[dict[str, object]] = []
    for code, label in enumerate(labels):
        mask = keys == code
        if not bool(mask.any()):
            continue
        series = nets[mask]
        out.append({"bucket": label, "days": int(mask.sum()), "trades": int(trades_per_day[mask].sum()),
                    "net_cents": int(series.sum()), "sharpe_annual": sharpe_annual(series)})  # fmt: skip
    return out


def _check_sum(name: str, rows: list[dict[str, object]], total: int) -> None:
    got = sum(int(r["net_cents"]) for r in rows)  # type: ignore[call-overload]
    if got != total:
        raise SuiteIntegrityError(f"breakdown {name}: bucket nets sum to {got}, the run's net is {total}")


def _by_year(days: pl.Series, nets: np.ndarray, per_day: np.ndarray) -> list[dict[str, object]]:
    years = days.dt.year().to_numpy()
    labels = sorted(set(years.tolist()))
    codes = np.searchsorted(np.asarray(labels), years)
    return _day_buckets(codes, list(labels), nets, per_day)


def _by_weekday(days: pl.Series, nets: np.ndarray, per_day: np.ndarray) -> list[dict[str, object]]:
    codes = days.dt.weekday().to_numpy().astype(np.int64) - 1  # polars: Monday = 1
    return _day_buckets(codes, list(WEEKDAYS), nets, per_day)


def day_ranges(bars: pl.DataFrame, days: pl.Series) -> np.ndarray:
    """High minus low in ticks over each run day's usable session bars."""
    ranges = bars.group_by("trade_date").agg((pl.col("high").max() - pl.col("low").min()).alias("range"))
    joined = pl.DataFrame({"trade_date": days}).join(
        ranges, on="trade_date", how="left", maintain_order="left"
    )
    if joined["range"].null_count():
        raise SuiteIntegrityError("a run day has no usable bar for its range")
    return joined["range"].to_numpy().astype(np.int64)


def _by_vol_tercile(ranges: np.ndarray, nets: np.ndarray, per_day: np.ndarray) -> list[dict[str, object]]:
    """Days ranked by range (ties by date), tercile = rank x 3 // days; bounds are the ranges each holds."""
    n = len(ranges)
    rank = np.empty(n, dtype=np.int64)
    rank[np.lexsort((np.arange(n), ranges))] = np.arange(n)
    codes = rank * TERCILES // max(n, 1)
    rows = _day_buckets(codes, ["low", "mid", "high"], nets, per_day)
    for row, code in zip(rows, sorted(set(codes.tolist())), strict=True):
        held = ranges[codes == code]
        row |= {"range_ticks_min": int(held.min()), "range_ticks_max": int(held.max())}
    return rows


def _by_entry_hour(hours: np.ndarray, trade_day: np.ndarray, trade_net: np.ndarray,
                   n_days: int) -> list[dict[str, object]]:  # fmt: skip
    out: list[dict[str, object]] = []
    for hour in sorted(set(hours.tolist())):
        mask = hours == hour
        series = np.zeros(n_days, dtype=np.int64)
        np.add.at(series, trade_day[mask], trade_net[mask])
        out.append({"bucket": int(hour), "trades": int(mask.sum()), "net_cents": int(trade_net[mask].sum()),
                    "sharpe_annual": sharpe_annual(series)})  # fmt: skip
    return out


# ------------------------------------------------------------------------------ rolling, trades ----
def rolling_sharpe(nets: np.ndarray, window: int) -> np.ndarray:
    """Annual Sharpe of each trailing window of `window` days (NaN before it fills or when sd is 0)."""
    out = np.full(len(nets), np.nan)
    if len(nets) < window:
        return out
    view = np.lib.stride_tricks.sliding_window_view(nets.astype(np.float64), window)
    sd = view.std(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[window - 1 :] = np.where(sd > 0, view.mean(axis=1) / sd, np.nan) * math.sqrt(
            TRADING_DAYS_PER_YEAR
        )
    return out


def _rolling_summary(series: np.ndarray) -> dict[str, object]:
    valid = series[~np.isnan(series)]
    if len(valid) == 0:
        return {"windows": 0, "min": None, "median": None, "max": None, "last": None}
    return {"windows": len(valid), "min": float(valid.min()), "median": float(np.median(valid)),
            "max": float(valid.max()),
            "last": None if np.isnan(series[-1]) else float(series[-1])}  # fmt: skip


def _excursions(trades: pl.DataFrame) -> dict[str, object]:
    net = trades["net_cents"].to_numpy()
    out: dict[str, object] = {}
    for name in ("mae_ticks", "mfe_ticks"):
        values = np.abs(trades[name].to_numpy().astype(np.int64))
        out[name] = {
            "all": {"n": len(values), **quantiles(values)},
            "winners": {"n": int((net > 0).sum()), **quantiles(values[net > 0])},
            "losers": {"n": int((net < 0).sum()), **quantiles(values[net < 0])},
        }
    return out


def _hold_minutes(trades: pl.DataFrame) -> dict[str, object]:
    held = (trades["exit_ts"].to_numpy() - trades["entry_ts"].to_numpy()) // NS_PER_MINUTE
    mean = None if len(held) == 0 else float(held.mean())
    return {"n": len(held), "mean": mean, **quantiles(held.astype(np.int64))}


def _trades_per_day(per_day: np.ndarray) -> dict[str, object]:
    n_days, active = len(per_day), int((per_day > 0).sum())
    total = int(per_day.sum())
    return {"all_days": None if n_days == 0 else total / n_days,
            "active_days": None if active == 0 else total / active, "days": n_days, "days_active": active,
            "max": int(per_day.max()) if n_days else 0}  # fmt: skip


# ------------------------------------------------------------------------------------- entry ----
def breakdowns(
    trades: pl.DataFrame, daily: pl.DataFrame, bars: pl.DataFrame
) -> tuple[dict[str, object], pl.DataFrame]:
    """The S6 tables and the rolling-Sharpe frame. `bars` are the run's usable bars (pfb.engine.run's
    prepare_bars), whose row index is the ledger's entry_bar."""
    days = daily["trade_date"]
    nets = daily["net_cents"].to_numpy().astype(np.int64)
    total = int(nets.sum())
    trade_net = trades["net_cents"].to_numpy().astype(np.int64)
    if int(trade_net.sum()) != total:
        raise SuiteIntegrityError(f"ledger net {int(trade_net.sum())} differs from the daily net {total}")
    day_int = days.cast(pl.Int32).to_numpy()
    trade_day = np.searchsorted(day_int, trades["trade_date"].cast(pl.Int32).to_numpy())
    per_day = np.bincount(trade_day, minlength=len(nets)).astype(np.int64)
    hours = bars["ct_minute"].to_numpy().astype(np.int64)[trades["entry_bar"].to_numpy()] // 60
    tables = {
        "by_year": _by_year(days, nets, per_day),
        "by_vol_tercile": _by_vol_tercile(day_ranges(bars, days), nets, per_day),
        "by_entry_hour_ct": _by_entry_hour(hours, trade_day, trade_net, len(nets)),
        "by_weekday": _by_weekday(days, nets, per_day),
    }
    for name, rows in tables.items():
        _check_sum(name, rows, total)
    rolling = {w: rolling_sharpe(nets, w) for w in ROLLING_WINDOWS}
    best = int(nets.max()) if len(nets) else None
    out: dict[str, object] = {
        **tables,
        "best_day_cents": best,
        "best_day_share": best / total if best is not None and total > 0 else None,
        "rolling_sharpe": {str(w): _rolling_summary(s) for w, s in rolling.items()},
        "excursions": _excursions(trades),
        "hold_minutes": _hold_minutes(trades),
        "trades_per_day": _trades_per_day(per_day),
    }
    frame = pl.DataFrame({"trade_date": days, **{f"sharpe_{w}": pl.Series(s).fill_nan(None)
                                                 for w, s in rolling.items()}})  # fmt: skip
    return out, frame
