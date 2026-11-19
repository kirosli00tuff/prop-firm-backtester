"""Walk-forward (S4): the fold report of one run, and selection across runs already recorded.

Pure functions on daily frames: no bar read, no registry access, no parameter values. A fold report is an
output of one configuration (docs/TRIALS.md C4) and never an input to choosing one; `select` reads the daily
frames of counted runs only.

Daily frame: one row per trade date of the run (zero-trade days included), in date order, with net_cents
(integer cents) and optionally trade_date (Date) and trades (integer count per day). Folds are contiguous
blocks of days: `days // folds` days each, the remainder going to the last fold. Sharpe per fold follows
pfb.metrics.core: per-day mean / population sd of daily net, times sqrt(252); None when the sd is zero.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from datetime import date

import polars as pl

from pfb.analysis.spec import SuiteError
from pfb.data.windows import WindowTable, load_windows
from pfb.metrics import stats

TRADING_DAYS_PER_YEAR = 252
MIN_FOLDS = 2
MODES = ("anchored", "rolling")


class FoldError(SuiteError):
    """The run has fewer days than the folds asked for (a SuiteError, so a ValueError too)."""


# --------------------------------------------------------------------------- helpers


def assert_open(root: str, first: date, last: date, table: WindowTable | None = None) -> None:
    """Raise ClosedWindowRefused when [first, last] touches a closed window of root (WindowTable)."""
    refusal = (table if table is not None else load_windows()).request_refusal(root, first, last)
    if refusal is not None:
        raise refusal


def sharpe_annual(nets: Sequence[int]) -> float | None:
    """Annualized Sharpe of daily nets (population sd); None with fewer than 2 days or a zero sd."""
    if len(nets) < 2:
        return None
    mom = stats.moments([float(v) for v in nets])
    if mom["sd"] == 0:
        return None
    return float(mom["sharpe"]) * math.sqrt(TRADING_DAYS_PER_YEAR)


def fold_bounds(n_days: int, folds: int) -> list[tuple[int, int]]:
    """[start, stop) row ranges of the folds: n_days // folds each, remainder to the last."""
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < MIN_FOLDS:
        raise ValueError(f"folds {folds!r} must be an int >= {MIN_FOLDS}")
    if n_days < folds:
        raise FoldError(f"{n_days} day(s) cannot make {folds} folds")
    size = n_days // folds
    return [(k * size, (k + 1) * size if k < folds - 1 else n_days) for k in range(folds)]


def _check_daily(daily: pl.DataFrame, label: str) -> None:
    if "net_cents" not in daily.columns or not daily.schema["net_cents"].is_integer():
        raise ValueError(f"{label}: needs an integer net_cents column")
    if daily["net_cents"].null_count():
        raise ValueError(f"{label}: net_cents has nulls")
    if "trade_date" in daily.columns:
        if daily.schema["trade_date"] != pl.Date or daily["trade_date"].null_count():
            raise ValueError(f"{label}: trade_date must be a Date column without nulls")
        dates = daily["trade_date"].to_list()
        if any(a >= b for a, b in zip(dates, dates[1:], strict=False)):
            raise ValueError(f"{label}: trade_date must be strictly increasing")
    if "trades" in daily.columns and not daily.schema["trades"].is_integer():
        raise ValueError(f"{label}: trades must be an integer column")


def _dates(daily: pl.DataFrame, start: int, stop: int) -> tuple[date | None, date | None]:
    if "trade_date" not in daily.columns:
        return None, None
    return daily["trade_date"][start], daily["trade_date"][stop - 1]


def _fold(daily: pl.DataFrame, index: int, start: int, stop: int) -> dict[str, object]:
    nets = [int(v) for v in daily["net_cents"][start:stop]]
    first, last = _dates(daily, start, stop)
    trades = int(daily["trades"][start:stop].sum()) if "trades" in daily.columns else None
    return {
        "fold": index,
        "first": None if first is None else first.isoformat(),
        "last": None if last is None else last.isoformat(),
        "days": stop - start,
        "trades": trades,
        "net_cents": sum(nets),
        "sharpe_annual": sharpe_annual(nets),
    }


def _same_sign(a: float, b: float) -> bool:
    return (a > 0 and b > 0) or (a < 0 and b < 0) or (a == 0 and b == 0)


def _pairs(nets: Sequence[int], bounds: Sequence[tuple[int, int]], anchored: bool) -> dict[str, object]:
    """In-sample (folds 0..k-1 anchored, or k-1 rolling) against out-of-sample fold k, k = 1..folds-1."""
    rows: list[dict[str, object]] = []
    agree, total, ratios = 0, 0, []
    for k in range(1, len(bounds)):
        is_start = 0 if anchored else bounds[k - 1][0]
        ins = sharpe_annual(nets[is_start : bounds[k - 1][1]])
        oos = sharpe_annual(nets[bounds[k][0] : bounds[k][1]])
        rows.append({"oos_fold": k, "is_sharpe": ins, "oos_sharpe": oos})
        if ins is None or oos is None:
            continue
        total += 1
        agree += _same_sign(ins, oos)
        if ins > 0:
            ratios.append(oos / ins)
    return {
        "pairs": rows,
        "n_pairs_scored": total,
        "same_sign_share": agree / total if total else None,
        "mean_oos_over_is": sum(ratios) / len(ratios) if ratios else None,
    }


# --------------------------------------------------------------------------- fold report


def fold_report(
    daily: pl.DataFrame, folds: int, root: str | None = None, table: WindowTable | None = None
) -> dict[str, object]:
    """Per-fold first/last date, days, trades, net_cents, sharpe_annual, and the stability figures.

    With root, every fold's date range is checked against the closed windows (needs trade_date).
    Keys: n_folds, n_days, net_cents, folds (list), sharpe_spread (population sd of the fold Sharpes that are
    not None), sharpe_min, sharpe_max, positive_folds (folds with net_cents > 0), anchored, rolling.
    """
    _check_daily(daily, "daily")
    bounds = fold_bounds(daily.height, folds)
    if root is not None:
        if "trade_date" not in daily.columns:
            raise ValueError("a closed-window check by root needs a trade_date column")
        for start, stop in bounds:
            assert_open(root, daily["trade_date"][start], daily["trade_date"][stop - 1], table)
    nets = [int(v) for v in daily["net_cents"]]
    rows = [_fold(daily, k, start, stop) for k, (start, stop) in enumerate(bounds)]
    fold_nets = [sum(nets[start:stop]) for start, stop in bounds]
    fold_sharpes = [sharpe_annual(nets[start:stop]) for start, stop in bounds]
    sharpes = [v for v in fold_sharpes if v is not None]
    return {
        "n_folds": folds,
        "n_days": daily.height,
        "net_cents": sum(nets),
        "folds": rows,
        "sharpe_spread": statistics.pstdev(sharpes) if sharpes else None,
        "sharpe_min": min(sharpes) if sharpes else None,
        "sharpe_max": max(sharpes) if sharpes else None,
        "positive_folds": sum(1 for v in fold_nets if v > 0),
        "anchored": _pairs(nets, bounds, anchored=True),
        "rolling": _pairs(nets, bounds, anchored=False),
    }


# --------------------------------------------------------------------------- selection


def _check_runs(runs: Mapping[str, pl.DataFrame]) -> int:
    if len(runs) < 1:
        raise ValueError("select needs at least one run")
    heights = {name: frame.height for name, frame in runs.items()}
    for name, frame in runs.items():
        _check_daily(frame, f"run {name}")
    if len(set(heights.values())) != 1:
        raise ValueError(f"runs must cover the same days (rows per run: {heights})")
    with_dates = [name for name, frame in runs.items() if "trade_date" in frame.columns]
    if with_dates and len(with_dates) != len(runs):
        raise ValueError("either every run has trade_date or none has")
    if with_dates:
        ref = runs[with_dates[0]]["trade_date"]
        for name in with_dates[1:]:
            if not runs[name]["trade_date"].equals(ref):
                raise ValueError(f"run {name} trade dates differ from run {with_dates[0]}")
    return next(iter(heights.values()))


def _best(runs: Mapping[str, pl.DataFrame], start: int, stop: int) -> tuple[str, float | None]:
    """Highest in-sample Sharpe; a None Sharpe ranks last; ties go to the first name in sorted order."""
    scored = [
        (name, sharpe_annual([int(v) for v in runs[name]["net_cents"][start:stop]])) for name in sorted(runs)
    ]
    best_name, best = scored[0]
    for name, value in scored[1:]:
        if value is not None and (best is None or value > best):
            best_name, best = name, value
    return best_name, best


def select(runs: Mapping[str, pl.DataFrame], folds: int, mode: str = "anchored") -> dict[str, object]:
    """Per fold k >= 1, the run with the best in-sample Sharpe; its fold-k days are stitched out of sample.

    runs maps a run id to its daily frame (same days for every run). Returns mode, n_folds, choices (oos_fold,
    chosen, is_sharpe, first, last, days, net_cents), the stitched net_cents, days and sharpe_annual, and
    "daily": the stitched out-of-sample daily frame (the only value that is not JSON-ready).
    """
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} must be one of {MODES}")
    n_days = _check_runs(runs)
    bounds = fold_bounds(n_days, folds)
    choices: list[dict[str, object]] = []
    pieces: list[pl.DataFrame] = []
    for k in range(1, len(bounds)):
        is_start = 0 if mode == "anchored" else bounds[k - 1][0]
        chosen, is_sharpe = _best(runs, is_start, bounds[k - 1][1])
        start, stop = bounds[k]
        piece = runs[chosen].slice(start, stop - start).with_columns(pl.lit(chosen).alias("run_id"))
        first, last = _dates(runs[chosen], start, stop)
        choices.append(
            {
                "oos_fold": k,
                "chosen": chosen,
                "is_sharpe": is_sharpe,
                "first": None if first is None else first.isoformat(),
                "last": None if last is None else last.isoformat(),
                "days": stop - start,
                "net_cents": int(piece["net_cents"].sum()),
            }
        )
        pieces.append(piece)
    stitched = pl.concat(pieces, how="diagonal_relaxed")
    nets = [int(v) for v in stitched["net_cents"]]
    return {
        "mode": mode,
        "n_folds": folds,
        "choices": choices,
        "net_cents": sum(nets),
        "days": len(nets),
        "sharpe_annual": sharpe_annual(nets),
        "daily": stitched,
    }
