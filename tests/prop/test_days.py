"""DayPaths from run artifacts and the synthetic constructors (src/pfb/prop/days.py)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pfb.prop.days import (
    DayPathsError,
    check_day_paths,
    day_paths_from_run,
    fill_minutes_known,
    from_day_nets,
    from_minutes,
    synthetic_trade_dates,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
NS = 60_000_000_000
D1, D2, D3 = dt.date(2024, 3, 4), dt.date(2024, 3, 5), dt.date(2024, 3, 6)


def _write_run(tmp: Path, *, equity: dict, daily: dict, trades: dict) -> Path:
    run = tmp / "run-x"
    run.mkdir()
    dates = {"trade_date": pl.Date}
    pl.DataFrame(equity, schema_overrides=dates).write_parquet(run / "equity_minute.parquet")
    pl.DataFrame(daily, schema_overrides={"trade_date": pl.Date}).write_parquet(run / "daily.parquet")
    pl.DataFrame(trades, schema_overrides={"trade_date": pl.Date}).write_parquet(run / "trades.parquet")
    return run


def _base(tmp: Path, **override: dict) -> Path:
    """Two traded days with minutes and one daily date without minute rows."""
    ts = [i * NS for i in range(6)]
    equity = {
        "ts_event": ts, "trade_date": [D1, D1, D1, D2, D2, D2], "position": [0, 2, 0, 0, -1, 0],
        "eq_close": [0, 300, 250, 250, 100, 120], "eq_low": [0, -50, 200, 250, 80, 100],
        "eq_high": [0, 320, 300, 250, 260, 130],
    }  # fmt: skip
    daily = {"trade_date": [D1, D2, D3], "eq_close": [250, 120, 120]}
    trades = {"contracts": [2, 1], "entry_ts": [ts[1], ts[4]], "exit_ts": [ts[2], ts[5]],
              "entry_bar": [1, 4], "exit_bar": [2, 5], "trade_date": [D1, D2]}  # fmt: skip
    for key, value in override.items():
        {"equity": equity, "daily": daily, "trades": trades}[key].update(value)
    return _write_run(tmp, equity=equity, daily=daily, trades=trades)


def test_run_minutes_relative_to_previous_close(tmp_path: Path) -> None:
    days = day_paths_from_run(_base(tmp_path))
    assert days.n_days == 3
    assert days.trade_dates.tolist() == [20240304, 20240305, 20240306]
    assert days.offsets.tolist() == [0, 3, 6, 7]
    assert days.close.tolist() == [0, 300, 250, 0, -150, -130, 0]  # day 2 opens at 250
    assert days.low.tolist() == [0, -50, 200, 0, -170, -150, 0]
    assert days.high.tolist() == [0, 320, 300, 0, 10, -120, 0]
    assert days.day_net().tolist() == [250, -130, 0]
    assert int(days.day_net().sum()) == 120  # adds up to the run's final equity
    assert days.source == ("run-x",)


def test_run_sides_fill_minutes_and_contracts(tmp_path: Path) -> None:
    days = day_paths_from_run(_base(tmp_path))
    assert days.fill_sides.tolist() == [0, 2, 2, 0, 1, 1, 0]
    assert days.sides.tolist() == [4, 2, 0]
    assert days.max_contracts.tolist() == [2, 1, 0]
    assert fill_minutes_known(days)


def test_run_micro_contracts_round_up_to_minis(tmp_path: Path) -> None:
    run = _base(tmp_path, equity={"position": [0, 11, 0, 0, -10, 0]})
    days = day_paths_from_run(run, contracts_per_mini=10)
    assert days.max_contracts.tolist() == [2, 1, 0]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"daily": {"eq_close": [250, 121, 121]}}, "eq_close on 20240305"),
        ({"trades": {"entry_ts": [NS * 9, NS * 4]}}, "entry_ts differs"),
        ({"trades": {"exit_ts": [NS * 2, NS * 9]}}, "exit_ts differs"),
        ({"trades": {"exit_bar": [2, 6]}}, "outside"),
        ({"trades": {"trade_date": [D2, D2]}}, "trade_date differs"),
        ({"daily": {"trade_date": [D1, D3, D2]}}, "strictly increasing"),
        ({"daily": {"trade_date": [D1, D3, dt.date(2024, 3, 7)]}}, "missing from daily"),
        ({"equity": {"ts_event": [0, NS, NS, 3 * NS, 4 * NS, 5 * NS]}}, "strictly increasing"),
    ],
)
def test_run_refuses_mismatch(tmp_path: Path, override: dict, message: str) -> None:
    run = _base(tmp_path, **override)
    with pytest.raises(DayPathsError, match=message):
        day_paths_from_run(run)


def test_run_refuses_missing_column_and_file(tmp_path: Path) -> None:
    run = _base(tmp_path)
    pl.read_parquet(run / "daily.parquet").drop("eq_close").write_parquet(run / "daily.parquet")
    with pytest.raises(DayPathsError, match="lacks column"):
        day_paths_from_run(run)
    (tmp_path / "b").mkdir()
    run2 = _base(tmp_path / "b")
    (run2 / "trades.parquet").unlink()
    with pytest.raises(DayPathsError, match="missing run artifact trades.parquet"):
        day_paths_from_run(run2)


def test_run_refuses_float_equity(tmp_path: Path) -> None:
    run = _base(tmp_path)
    frame = pl.read_parquet(run / "equity_minute.parquet")
    frame.with_columns(pl.col("eq_low").cast(pl.Float64)).write_parquet(run / "equity_minute.parquet")
    with pytest.raises(DayPathsError, match="must be Int64"):
        day_paths_from_run(run)


def test_from_day_nets_defaults() -> None:
    days = from_day_nets([500, -300, 0])
    assert days.offsets.tolist() == [0, 1, 2, 3]
    assert days.low.tolist() == [0, -300, 0]
    assert days.high.tolist() == [500, 0, 0]
    assert days.day_net().tolist() == [500, -300, 0]
    assert days.sides.tolist() == [2, 2, 2]
    assert days.max_contracts.tolist() == [1, 1, 1]
    assert days.trade_dates.tolist() == [20000103, 20000104, 20000105]
    assert fill_minutes_known(days)


def test_from_day_nets_custom_and_refusals() -> None:
    days = from_day_nets([100], lows=[-40], highs=[150], sides=[0])
    assert (days.low[0], days.high[0], int(days.sides[0])) == (-40, 150, 0)
    with pytest.raises(DayPathsError, match="low <= min"):
        from_day_nets([100], lows=[10])
    with pytest.raises(DayPathsError, match="integers"):
        from_day_nets([1.5])  # type: ignore[list-item]
    with pytest.raises(DayPathsError, match="expected 1"):
        from_day_nets([1], sides=[1, 2])


def test_from_minutes_ragged() -> None:
    days = from_minutes([([0, -10], [5, 5], [5, -10]), ([0], [0], [0])], sides=[4, 0])
    assert days.offsets.tolist() == [0, 2, 3]
    assert days.fill_sides.tolist() == [4, 0, 0]
    assert days.day_net().tolist() == [-10, 0]
    with pytest.raises(DayPathsError, match="low > close"):
        from_minutes([([5], [5], [0])])
    with pytest.raises(DayPathsError, match="equal-length"):
        from_minutes([([0, 0], [0], [0])])


def test_check_day_paths_refuses_bad_dtype() -> None:
    days = from_day_nets([1, 2])
    from dataclasses import replace

    with pytest.raises(DayPathsError, match="low must be int64"):
        check_day_paths(replace(days, low=days.low.astype(np.float64)))


def test_synthetic_trade_dates_skip_weekends() -> None:
    assert synthetic_trade_dates(6).tolist()[-2:] == [20000107, 20000110]


