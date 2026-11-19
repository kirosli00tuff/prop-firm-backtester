"""Pooled accounts (src/pfb/prop/pool.py, -P2): several runs on one clock, contract caps."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pfb.prop.config import load_plan, with_values
from pfb.prop.days import day_paths_from_run
from pfb.prop.pool import (
    NO_ROW,
    PoolError,
    PoolOverLimit,
    cap_minis,
    check_caps,
    pool_detail,
    pool_runs,
    run_contracts_per_mini,
    run_root,
)
from pfb.prop.types import DayPaths

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN = REPO_ROOT / "tests" / "prop" / "fixtures" / "ka_static_50k.toml"
NS = 60_000_000_000
D1, D2, D3 = dt.date(2024, 3, 4), dt.date(2024, 3, 5), dt.date(2024, 3, 6)


def _write_run(base: Path, name: str, *, equity: dict, daily: dict, trades: dict) -> Path:
    run = base / name
    run.mkdir()
    dates = {"trade_date": pl.Date}
    pl.DataFrame(equity, schema_overrides=dates).write_parquet(run / "equity_minute.parquet")
    pl.DataFrame(daily, schema_overrides=dates).write_parquet(run / "daily.parquet")
    pl.DataFrame(trades, schema_overrides=dates).write_parquet(run / "trades.parquet")
    return run


def _trades(ts: list[int], contracts: list[int], entry: list[int], exit_: list[int],
            dates: list[dt.date]) -> dict:  # fmt: skip
    return {"contracts": contracts, "entry_ts": [ts[i] for i in entry], "exit_ts": [ts[i] for i in exit_],
            "entry_bar": entry, "exit_bar": exit_, "trade_date": dates}  # fmt: skip


def _base_tables() -> tuple[dict, dict, dict]:
    """Two traded days with minutes and one daily date without minute rows (as in test_days)."""
    ts = [i * NS for i in range(6)]
    equity = {
        "ts_event": ts, "trade_date": [D1, D1, D1, D2, D2, D2], "position": [0, 2, 0, 0, -1, 0],
        "eq_close": [0, 300, 250, 250, 100, 120], "eq_low": [0, -50, 200, 250, 80, 100],
        "eq_high": [0, 320, 300, 250, 260, 130],
    }  # fmt: skip
    daily = {"trade_date": [D1, D2, D3], "eq_close": [250, 120, 120]}
    return equity, daily, _trades(ts, [2, 1], [1, 4], [2, 5], [D1, D2])


def _negated(equity: dict, daily: dict) -> tuple[dict, dict]:
    """Every equity column and the position negated; the negated low is the old high and vice versa."""
    neg = {
        "ts_event": equity["ts_event"], "trade_date": equity["trade_date"],
        "position": [-p for p in equity["position"]], "eq_close": [-c for c in equity["eq_close"]],
        "eq_low": [-h for h in equity["eq_high"]], "eq_high": [-lo for lo in equity["eq_low"]],
    }  # fmt: skip
    return neg, {"trade_date": daily["trade_date"], "eq_close": [-c for c in daily["eq_close"]]}


def _assert_same(a: DayPaths, b: DayPaths) -> None:
    for f in fields(DayPaths):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, np.ndarray):
            assert x.dtype == y.dtype, f.name
            assert np.array_equal(x, y), f.name
        else:
            assert x == y, f.name


# ------------------------------------------------------------------------------ P2: one run, negation ----
def test_pooling_one_run_equals_the_run(tmp_path: Path) -> None:
    equity, daily, trades = _base_tables()
    run = _write_run(tmp_path, "run-a", equity=equity, daily=daily, trades=trades)
    _assert_same(pool_runs([run], contracts_per_mini=[1]), day_paths_from_run(run))


def test_pooling_one_micro_run_equals_the_run(tmp_path: Path) -> None:
    equity, daily, trades = _base_tables()
    equity["position"] = [0, 11, 0, 0, -10, 0]
    run = _write_run(tmp_path, "run-a", equity=equity, daily=daily, trades=trades)
    _assert_same(pool_runs([run], contracts_per_mini=[10]), day_paths_from_run(run, contracts_per_mini=10))


def test_pooling_a_run_with_its_negation_is_flat(tmp_path: Path) -> None:
    equity, daily, trades = _base_tables()
    run = _write_run(tmp_path, "run-a", equity=equity, daily=daily, trades=trades)
    neg_equity, neg_daily = _negated(equity, daily)
    neg = _write_run(tmp_path, "run-neg", equity=neg_equity, daily=neg_daily, trades=trades)
    pooled = pool_runs([run, neg], contracts_per_mini=[1, 1])
    single = day_paths_from_run(run)
    assert pooled.source == ("run-a", "run-neg")
    assert np.array_equal(pooled.offsets, single.offsets)
    assert not pooled.close.any()  # eq_close 0 at every minute
    assert not pooled.day_net().any()  # day net 0
    # the conservative low/high bound: each minute's own range of one run minus the other's
    assert pooled.low.tolist() == (single.low - single.high).tolist()
    assert pooled.high.tolist() == (single.high - single.low).tolist()
    assert pooled.sides.tolist() == (2 * single.sides).tolist()
    assert pooled.fill_sides.tolist() == (2 * single.fill_sides).tolist()
    assert pooled.max_contracts.tolist() == (2 * single.max_contracts).tolist()


# --------------------------------------------------------------------- two runs, different days, by hand ----
def _two_runs(tmp_path: Path) -> tuple[Path, Path]:
    """Run A (mini) trades D1, D2; run B (micro) trades D2, D3; one shared minute (ts 11) on D2."""
    ts_a = [0, NS, 10 * NS, 11 * NS]
    a = _write_run(
        tmp_path, "run-a",
        equity={"ts_event": ts_a, "trade_date": [D1, D1, D2, D2], "position": [1, 0, 2, 0],
                "eq_close": [100, 150, 150, 90], "eq_low": [-20, 100, 120, 80],
                "eq_high": [110, 160, 170, 150]},
        daily={"trade_date": [D1, D2], "eq_close": [150, 90]},
        trades=_trades(ts_a, [1, 2], [0, 2], [1, 3], [D1, D2]),
    )  # fmt: skip
    ts_b = [11 * NS, 12 * NS, 20 * NS]
    b = _write_run(
        tmp_path, "run-b",
        equity={"ts_event": ts_b, "trade_date": [D2, D2, D3], "position": [25, 0, 5],
                "eq_close": [-30, -40, 10], "eq_low": [-50, -45, -40], "eq_high": [0, -40, 20]},
        daily={"trade_date": [D2, D3], "eq_close": [-40, 10]},
        trades=_trades(ts_b, [25, 5], [0, 2], [1, 2], [D2, D3]),
    )  # fmt: skip
    return a, b


def test_two_runs_on_different_days_by_hand(tmp_path: Path) -> None:
    # Common clock ts 0,1 10,11,12 20.
    # Carried A close 100,150,150,90,90,90; B 0,0,0,-30,-40,10.
    # Pooled close 100,150,150,60,50,100; low -20,100,120,30,45,50 (A's carried 90 at ts 12, 20; B's 0 before
    # its first row); high 110,160,170,150,50,110. Day opens 0, 150, 50.
    a, b = _two_runs(tmp_path)
    pooled = pool_runs([a, b], contracts_per_mini=[1, 10])
    assert pooled.trade_dates.tolist() == [20240304, 20240305, 20240306]
    assert pooled.offsets.tolist() == [0, 2, 5, 6]
    assert pooled.close.tolist() == [100, 150, 0, -90, -100, 50]
    assert pooled.low.tolist() == [-20, 100, -30, -120, -105, 0]
    assert pooled.high.tolist() == [110, 160, 20, 0, -100, 60]
    assert pooled.day_net().tolist() == [150, -100, 50]
    assert int(pooled.day_net().sum()) == 90 + 10  # the runs' final equities
    assert pooled.fill_sides.tolist() == [1, 1, 2, 27, 25, 10]
    assert pooled.sides.tolist() == [2, 54, 10]
    # mini equivalents (tenths), a run's exposure at its own row is max(|position|, the size of
    # a trade entered or exited there): A 10,10,20,20,0,0 (exits of 1 at ts 1 and of 2 at ts 11);
    # B 0,0,0,25,25,5 (exit of 25 at ts 12) -> sums 10,10,20,45,25,5, per day max 10, 45, 5 -> 1, 5, 1 (ceil)
    assert pooled.max_contracts.tolist() == [1, 5, 1]
    assert pooled.source == ("run-a", "run-b")


def test_pool_detail_minutes_and_units(tmp_path: Path) -> None:
    a, b = _two_runs(tmp_path)
    detail = pool_detail([a, b], contracts_per_mini=[1, 10])
    assert detail.units_per_mini == 10
    assert detail.ts_event.tolist() == [0, NS, 10 * NS, 11 * NS, 12 * NS, 20 * NS]
    assert detail.units.tolist() == [10, 10, 20, 45, 25, 5]  # trade sizes at entry and exit minutes (N9)


def test_filler_day_has_no_timestamp(tmp_path: Path) -> None:
    equity, daily, trades = _base_tables()
    run = _write_run(tmp_path, "run-a", equity=equity, daily=daily, trades=trades)
    detail = pool_detail([run], contracts_per_mini=[1])
    assert detail.ts_event[-1] == NO_ROW  # D3 has no minute rows
    assert detail.units[-1] == 0


def test_pool_order_only_changes_source(tmp_path: Path) -> None:
    a, b = _two_runs(tmp_path)
    ab, ba = pool_runs([a, b], contracts_per_mini=[1, 10]), pool_runs([b, a], contracts_per_mini=[10, 1])
    assert ba.source == ("run-b", "run-a")
    _assert_same(ab, DayPaths(**{**{f.name: getattr(ba, f.name) for f in fields(DayPaths)},
                                 "source": ab.source}))  # fmt: skip


# ----------------------------------------------------------------------------------------------- caps ----
def _flat_run(base: Path, name: str, contracts: int) -> Path:
    """One day, a position of `contracts` minis held over ts 1-2."""
    ts = [0, NS, 2 * NS, 3 * NS]
    return _write_run(
        base, name,
        equity={"ts_event": ts, "trade_date": [D1] * 4, "position": [0, contracts, contracts, 0],
                "eq_close": [0, 10, 20, 0], "eq_low": [0, 0, 0, 0], "eq_high": [0, 10, 20, 20]},
        daily={"trade_date": [D1], "eq_close": [0]},
        trades=_trades(ts, [contracts], [1], [3], [D1]),
    )  # fmt: skip


def _plan(eval_cap: int, top_tier: int) -> object:
    return with_values(load_plan(PLAN), {"eval.max_contracts_minis": eval_cap,
                                         "funded.scaling": [[100_000, 1], ["none", top_tier]]})  # fmt: skip


def test_caps_refuse_an_over_limit_pool(tmp_path: Path) -> None:
    a, b = _flat_run(tmp_path, "run-a", 2), _flat_run(tmp_path, "run-b", 2)
    plan = _plan(2, 3)
    for run in (a, b):  # each run alone is at the limit and passes
        check_caps(pool_detail([run], contracts_per_mini=[1]), plan, phase="eval")
    detail = pool_detail([a, b], contracts_per_mini=[1, 1])
    assert detail.days.max_contracts.tolist() == [4]
    with pytest.raises(PoolOverLimit, match="over the eval cap of 2") as info:
        check_caps(detail, plan, phase="eval")
    assert (info.value.trade_date, info.value.minute, info.value.ts_event) == (20240304, 1, NS)
    assert (info.value.contracts, info.value.limit) == (4, 2)
    with pytest.raises(PoolOverLimit, match="over the funded cap of 3"):
        check_caps(detail, plan, phase="funded")
    check_caps(detail, _plan(4, 4), phase="eval")  # at the cap is allowed
    check_caps(detail, _plan(4, 4), phase="funded")


def test_caps_on_a_bare_day_paths_name_the_day_only(tmp_path: Path) -> None:
    a, b = _flat_run(tmp_path, "run-a", 2), _flat_run(tmp_path, "run-b", 2)
    with pytest.raises(PoolOverLimit) as info:
        check_caps(pool_runs([a, b], contracts_per_mini=[1, 1]), _plan(3, 3))
    assert (info.value.trade_date, info.value.minute, info.value.ts_event) == (20240304, None, None)


def test_caps_count_micros_as_tenths(tmp_path: Path) -> None:
    a, b = _flat_run(tmp_path, "run-a", 2), _flat_run(tmp_path, "run-b", 11)
    detail = pool_detail([a, b], contracts_per_mini=[1, 10])  # 2 minis + 11 micros = 3.1 -> 4
    assert detail.days.max_contracts.tolist() == [4]
    with pytest.raises(PoolOverLimit):
        check_caps(detail, _plan(3, 3))
    check_caps(detail, _plan(4, 4))


def test_cap_minis_reads_the_plan() -> None:
    plan = _plan(5, 7)
    assert cap_minis(plan, "eval") == 5
    assert cap_minis(plan, "funded") == 7
    with pytest.raises(ValueError, match="phase"):
        cap_minis(plan, "payout")


# ------------------------------------------------------------------------------------ refusals, roots ----
def test_pool_refuses_bad_input(tmp_path: Path) -> None:
    a, b = _two_runs(tmp_path)
    with pytest.raises(PoolError, match="at least one"):
        pool_runs([])
    with pytest.raises(PoolError, match="twice"):
        pool_runs([a, a], contracts_per_mini=[1, 1])
    with pytest.raises(PoolError, match="2 runs"):
        pool_runs([a, b], contracts_per_mini=[1])
    with pytest.raises(PoolError, match=">= 1"):
        pool_runs([a, b], contracts_per_mini=[1, 0])


def test_pool_refuses_clashing_trade_dates(tmp_path: Path) -> None:
    a, _ = _two_runs(tmp_path)
    ts = [11 * NS]  # A has ts 11 on D2; this run puts it on D3
    c = _write_run(tmp_path, "run-c",
                   equity={"ts_event": ts, "trade_date": [D3], "position": [0], "eq_close": [0],
                           "eq_low": [0], "eq_high": [0]},
                   daily={"trade_date": [D3], "eq_close": [0]},
                   trades=_trades(ts, [1], [0], [0], [D3]))  # fmt: skip
    with pytest.raises(PoolError, match="trade dates"):
        pool_runs([a, c], contracts_per_mini=[1, 1])


def test_pool_refuses_a_broken_run(tmp_path: Path) -> None:
    equity, daily, trades = _base_tables()
    daily["eq_close"] = [250, 120, 999]
    run = _write_run(tmp_path, "run-bad", equity=equity, daily=daily, trades=trades)
    with pytest.raises(PoolError, match="run-bad"):
        pool_runs([run], contracts_per_mini=[1])


def test_run_root_from_summary_then_ledger(tmp_path: Path) -> None:
    a, b = _two_runs(tmp_path)
    (a / "summary.json").write_text(json.dumps({"root": "NQ"}))
    (b / "summary.json").write_text(json.dumps({"net_cents": 0}))
    ledger = tmp_path / "registry.jsonl"
    ledger.write_text(json.dumps({"kind": "trial", "run_id": "run-b", "root": "MNQ"}) + "\n"
                      + json.dumps({"kind": "finish", "run_id": "run-b"}) + "\n")  # fmt: skip
    assert run_root(a, ledger=ledger) == "NQ"
    assert run_root(b, ledger=ledger) == "MNQ"
    assert run_contracts_per_mini(a, ledger=ledger) == 1
    assert run_contracts_per_mini(b, ledger=ledger) == 10
    _assert_same(pool_runs([a, b], ledger=ledger), pool_runs([a, b], contracts_per_mini=[1, 10]))
    with pytest.raises(PoolError, match="0 root"):
        run_root(tmp_path / "run-missing", ledger=ledger)


def test_contract_type_without_mini_equivalent_is_refused(tmp_path: Path) -> None:
    a, _ = _two_runs(tmp_path)
    (a / "summary.json").write_text(json.dumps({"root": "MBT"}))
    with pytest.raises(PoolError, match="no stated mini equivalent"):
        run_contracts_per_mini(a)
    (a / "summary.json").write_text(json.dumps({"root": "NOPE"}))
    with pytest.raises(PoolError, match="not in config/instruments.toml"):
        run_contracts_per_mini(a)


