"""S7 sizing table: hand-computed replays (integer arithmetic throughout)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pfb.analysis.sizing import (
    KELLY_FLAG,
    SizingInputs,
    SizingPlan,
    load_sizing_plan,
    risk_per_contract,
    sizing_table,
)
from pfb.analysis.spec import SuiteError

PLAN = SizingPlan("test/plan", drawdown_cents=200_000, daily_loss_cents=None, cap_contracts=50)
RISK_CC = 50_000  # 10 ticks of MNQ (5000 cc per tick): 500 cents per contract


def _inputs(net: list[int], day: list[int], n_days: int, risk: int = RISK_CC) -> SizingInputs:
    return SizingInputs(np.asarray(net, np.int64), np.asarray(day, np.int64), n_days, 1,
                        np.full(len(net), risk, np.int64), "stop")  # fmt: skip


FOUR = _inputs([1000, -500, 2000, -1000], [0, 0, 1, 2], 3)


def test_fixed_reproduces_the_run() -> None:
    table = sizing_table(FOUR, PLAN)
    fixed = table["rules"]["fixed"]  # type: ignore[index]
    assert fixed["net_cents"] == 1500 and fixed["max_contracts"] == 1 and fixed["share_capped"] == 0
    assert fixed["max_drawdown_cents"] == 1000  # closed-trade marks 0, 1000, 500, 2500, 1500


def test_fixed_fraction_by_hand() -> None:
    # q = floor(0.10 x distance / 500 cents); distance from an end-of-day trailing floor 200000 below the peak
    # t0: 200000 -> 40, +40000; t1 (same day): 240000 -> 48, -24000 (equity 16000); t2: peak 16000,
    # distance 200000 -> 40, +80000; t3: peak 96000 -> 40, -40000 -> net 56000
    rule = sizing_table(FOUR, PLAN)["rules"]["fixed_fraction_0.10"]  # type: ignore[index]
    assert rule["net_cents"] == 56_000 and rule["max_contracts"] == 48 and rule["share_capped"] == 0
    capped = sizing_table(FOUR, PLAN.__class__("t/p", 200_000, None, 45))["rules"]["fixed_fraction_0.10"]  # type: ignore[index]
    # t1 capped at 45: -22500, equity 17500; then 40 and 40: 17500 + 80000 - 40000
    assert capped["net_cents"] == 57_500 and capped["max_contracts"] == 45 and capped["share_capped"] == 0.25


def test_quarter_kelly_by_hand() -> None:
    # mean / variance per cent = c n S / (n Q - S^2) = 4 x 1500 / (4 x 6.25e6 - 1500^2) = 6000 / 22.75e6
    # q = floor(6000 x distance / (4 x 22.75e6)): 13, 14, 13, 13 -> 13000 - 7000 + 26000 - 13000
    table = sizing_table(FOUR, PLAN)
    kelly = table["kelly"]
    assert kelly["flag"] == KELLY_FLAG  # type: ignore[index]
    assert kelly["f_star_contracts_per_cent"] == pytest.approx(6000 / 22.75e6)  # type: ignore[index]
    quarter = kelly["kelly_quarter"]  # type: ignore[index]
    assert quarter["replayed"] is True and quarter["net_cents"] == 19_000 and quarter["max_contracts"] == 14


def test_vol_target_by_hand() -> None:
    # daily nets alternate +1000 / -1000: the 20-day population sd is 1000; target drawdown / 10 = 20000 ->
    # 20 contracts from day 20 on (SPEC size before): days 20..24 net +20000 overall
    net = [1000 if d % 2 == 0 else -1000 for d in range(25)]
    inputs = _inputs(net, list(range(25)), 25)
    rule = sizing_table(inputs, PLAN)["rules"]["vol_target"]  # type: ignore[index]
    assert rule["target_cents"] == 20_000 and rule["max_contracts"] == 20 and rule["net_cents"] == 20_000
    with_dll = sizing_table(inputs, SizingPlan("t/p", 200_000, 30_000, 50))["rules"]["vol_target"]  # type: ignore[index]
    assert with_dll["target_cents"] == 10_000 and with_dll["max_contracts"] == 10


def test_without_a_plan_only_fixed_and_kelly_fractions() -> None:
    table = sizing_table(FOUR, None)
    assert list(table["rules"]) == ["fixed"]  # type: ignore[arg-type]
    assert table["rules"]["fixed"]["net_cents"] == 1500  # type: ignore[index]
    assert table["kelly"]["kelly_half"] == {"fraction": pytest.approx(6000 / 22.75e6 / 2), "replayed": False}  # type: ignore[index]


def test_a_distance_rule_stops_at_the_floor_and_fixed_does_not() -> None:
    inputs = _inputs([-150_000, -100_000, 5_000], [0, 1, 2], 3)
    table = sizing_table(inputs, PLAN)
    assert table["rules"]["fixed"]["net_cents"] == -245_000  # type: ignore[index]
    assert table["rules"]["fixed"]["floor_breached"] is True  # type: ignore[index]
    rule = table["rules"]["fixed_fraction_0.05"]  # type: ignore[index]
    assert rule["floor_breached"] is True


def test_risk_from_stops_or_the_mae_quantile() -> None:
    risk, source = risk_per_contract(np.array([5, 9, 1]), np.array([4, 4, 8]), 5000)
    assert source == "stop" and risk.tolist() == [20_000, 20_000, 40_000]
    mae = np.arange(1, 101)
    risk, source = risk_per_contract(mae, np.array([4] * 99 + [0]), 5000)
    assert source == "mae_p95" and set(risk.tolist()) == {95 * 5000}


def test_no_risk_turns_the_fixed_fraction_rule_off() -> None:
    table = sizing_table(_inputs([100, -50], [0, 1], 2, risk=0), PLAN)
    assert table["rules"]["fixed_fraction_0.05"] is None  # type: ignore[index]


def test_unknown_plan_file_is_a_suite_error(tmp_path: Path) -> None:
    with pytest.raises(SuiteError, match="no file"):
        load_sizing_plan("nofirm/noplan", "micro")


def test_the_topstep_plan_gives_its_drawdown_and_a_micro_cap() -> None:
    plan = load_sizing_plan("topstep/combine_xfa_standard_50k", "micro")
    assert plan.drawdown_cents == 200_000 and plan.cap_contracts == 50 and plan.daily_loss_cents is None
    assert load_sizing_plan("topstep/combine_xfa_standard_50k", "mini").cap_contracts == 5
