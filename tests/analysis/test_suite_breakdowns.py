"""S6 breakdowns on hand-built frames with known answers."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl
import pytest

from pfb.analysis.breakdowns import breakdowns, quantiles, rolling_sharpe, sharpe_annual
from pfb.analysis.spec import SuiteIntegrityError

DAYS = [date(2024, 12, 30), date(2024, 12, 31), date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6),
        date(2025, 1, 7)]  # Mon Tue Thu Fri Mon Tue  # fmt: skip
DAILY = [100, -50, 0, 200, -30, 60]
RANGES = [10, 20, 30, 40, 50, 60]
NS_MIN = 60_000_000_000
# (day index, entry hour CT, net, entry bar = 2 x day + (hour == 10), hold minutes, mae, mfe)
TRADES = [(0, 9, 100, 0, 10, 2, 8), (1, 10, -50, 3, 20, 9, 1), (3, 9, 150, 6, 30, 1, 12),
          (3, 10, 50, 7, 40, 3, 5), (4, 9, -30, 8, 50, 6, 2), (5, 10, 60, 11, 60, 0, 7)]  # fmt: skip


def _frames() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    daily = pl.DataFrame({"trade_date": DAYS, "net_cents": DAILY})
    bars = pl.DataFrame({
        "trade_date": [d for d in DAYS for _ in range(2)],
        "high": [1000 + v for r in RANGES for v in (r, 0)],
        "low": [1000] * 12,
        "ct_minute": [m for _ in DAYS for m in (540, 600)],
    })  # fmt: skip
    trades = pl.DataFrame({
        "trade_date": [DAYS[t[0]] for t in TRADES],
        "net_cents": [t[2] for t in TRADES],
        "entry_bar": [t[3] for t in TRADES],
        "entry_ts": [0] * len(TRADES),
        "exit_ts": [t[4] * NS_MIN for t in TRADES],
        "mae_ticks": [t[5] for t in TRADES],
        "mfe_ticks": [t[6] for t in TRADES],
    })  # fmt: skip
    return trades, daily, bars


def _rows(out: dict[str, object], name: str) -> list[dict[str, object]]:
    rows = out[name]
    assert isinstance(rows, list)
    return rows


def _sh(values: list[int]) -> float | None:
    x = np.asarray(values, dtype=float)
    return None if x.std() == 0 else x.mean() / x.std() * math.sqrt(252)


def test_buckets_have_the_known_nets_trades_and_sharpes() -> None:
    out, _ = breakdowns(*_frames())
    assert [(r["bucket"], r["net_cents"], r["trades"], r["days"]) for r in _rows(out, "by_year")] == [
        (2024, 50, 2, 2), (2025, 230, 4, 4)]  # type: ignore[index, union-attr]  # fmt: skip
    assert out["by_year"][1]["sharpe_annual"] == pytest.approx(_sh([0, 200, -30, 60]))  # type: ignore[index]
    terciles = _rows(out, "by_vol_tercile")
    assert [(r["bucket"], r["net_cents"], r["range_ticks_min"], r["range_ticks_max"]) for r in terciles] == [
        ("low", 50, 10, 20), ("mid", 200, 30, 40), ("high", 30, 50, 60)]  # type: ignore[union-attr]  # fmt: skip
    assert [(r["bucket"], r["net_cents"], r["days"]) for r in _rows(out, "by_weekday")] == [
        ("Mon", 70, 2), ("Tue", 10, 2), ("Thu", 0, 1), ("Fri", 200, 1)]  # type: ignore[union-attr]  # fmt: skip
    hours = _rows(out, "by_entry_hour_ct")
    assert [(r["bucket"], r["net_cents"], r["trades"]) for r in hours] == [(9, 220, 3), (10, 60, 3)]  # type: ignore[union-attr]
    # an hour's Sharpe is over every run day, each carrying only that hour's trades
    assert hours[0]["sharpe_annual"] == pytest.approx(_sh([100, 0, 0, 150, -30, 0]))  # type: ignore[index]
    assert hours[1]["sharpe_annual"] == pytest.approx(_sh([0, -50, 0, 50, 0, 60]))  # type: ignore[index]


def test_every_breakdown_sums_to_the_run_net() -> None:
    out, _ = breakdowns(*_frames())
    for name in ("by_year", "by_vol_tercile", "by_entry_hour_ct", "by_weekday"):
        assert sum(r["net_cents"] for r in _rows(out, name)) == sum(DAILY)  # type: ignore[union-attr]


def test_trade_analytics() -> None:
    out, _ = breakdowns(*_frames())
    assert out["best_day_cents"] == 200 and out["best_day_share"] == pytest.approx(200 / 280)
    assert out["trades_per_day"] == {"all_days": 1.0, "active_days": 6 / 5, "days": 6, "days_active": 5,
                                     "max": 2}  # fmt: skip
    assert out["hold_minutes"] == {"n": 6, "mean": 35.0, "q05": 10, "q25": 20, "q50": 30, "q75": 50,
                                   "q95": 60}  # fmt: skip
    mae = out["excursions"]["mae_ticks"]  # type: ignore[index]
    assert mae["winners"] == {"n": 4, "q05": 0, "q25": 0, "q50": 1, "q75": 2, "q95": 3}  # [0, 1, 2, 3]
    assert mae["losers"] == {"n": 2, "q05": 6, "q25": 6, "q50": 6, "q75": 9, "q95": 9}


def test_a_ledger_that_disagrees_with_the_daily_net_is_an_error() -> None:
    trades, daily, bars = _frames()
    with pytest.raises(SuiteIntegrityError):  # a base run that disagrees with itself
        breakdowns(trades.with_columns(pl.col("net_cents") + 1), daily, bars)


def test_rolling_sharpe_matches_a_direct_window_computation() -> None:
    rng = np.random.default_rng(3)
    nets = rng.integers(-1000, 1200, 300)
    got = rolling_sharpe(nets, 63)
    assert np.isnan(got[:62]).all()
    for i in (62, 100, 299):
        assert got[i] == pytest.approx(_sh(nets[i - 62 : i + 1].tolist()))


def test_rolling_frame_and_summary() -> None:
    rng = np.random.default_rng(4)
    n = 300
    days = pl.date_range(date(2020, 1, 1), date(2020, 1, 1) + (n - 1) * (date(2020, 1, 2) - date(2020, 1, 1)),
                         eager=True)  # fmt: skip
    nets = rng.integers(-1000, 1200, n)
    daily = pl.DataFrame({"trade_date": days, "net_cents": nets})
    trades = pl.DataFrame({"trade_date": days, "net_cents": nets, "entry_bar": np.zeros(n, np.int64),
                           "entry_ts": np.zeros(n, np.int64), "exit_ts": np.full(n, NS_MIN),
                           "mae_ticks": np.ones(n, np.int64), "mfe_ticks": np.ones(n, np.int64)})  # fmt: skip
    bars = pl.DataFrame({"trade_date": days, "high": np.full(n, 5), "low": np.zeros(n, np.int64),
                         "ct_minute": np.full(n, 600)})  # fmt: skip
    out, frame = breakdowns(trades, daily, bars)
    assert frame.columns == ["trade_date", "sharpe_63", "sharpe_252"]
    assert frame["sharpe_63"].null_count() == 62 and frame["sharpe_252"].null_count() == 251
    summary = out["rolling_sharpe"]["252"]  # type: ignore[index]
    assert summary["windows"] == n - 251
    assert summary["last"] == pytest.approx(_sh(nets[-252:].tolist()))


def test_helpers_on_empty_and_flat_series() -> None:
    assert sharpe_annual(np.zeros(0, np.int64)) is None
    assert sharpe_annual(np.full(5, 7)) is None
    assert quantiles(np.zeros(0, np.int64))["q50"] is None
