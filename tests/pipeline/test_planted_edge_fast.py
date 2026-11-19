"""Planted-edge recovery, the fast part (default suite): the power oracle against pfb.analysis.power, the
registration gate below the minimum edge, one run through the whole suite and a 20-seed recovery at 250
trades. The many-seed detection rates are in test_planted_edge_slow.py (marked slow). Synthetic data only."""

from __future__ import annotations

import math

import pytest
from planted_market import (
    CASES,
    N_TRIALS,
    cost_per_trade_cents,
    outcome,
    power_at,
    predicted_power,
    probe_trades,
    run_planted,
    trade_sd_cents,
)

from pfb.analysis.inrun import SUITE_KEYS
from pfb.analysis.spec import SuiteSpec

ORACLE_CEIL_SLACK = 0.01  # min_edge is a ceil to whole cents: the power there is 0.80 to at most 0.81
COUNT_TOLERANCE = 0.03  # early closes and absent minutes take up to 3 % of the stated slots
FAST_SEEDS = range(1, 21)
SE_TOLERANCE = 3.0  # recovered mean within 3 standard errors of the truth (two-sided 0.27 %)
ONE_RUN_SE_TOLERANCE = 4.0  # a single seed: within 4 standard errors (two-sided 6e-5)
SD_TOLERANCE = 0.06  # one run's trade sd within 6 % of the design sd (about 2.7 sd of the estimate at 1,000)


@pytest.mark.parametrize("trades", sorted(CASES))
def test_the_power_oracle_is_the_modules_formula_at_its_own_minimum(trades: int) -> None:
    case = CASES[trades]
    p = power_at(case, trades)
    assert p.min_edge_cents is not None and p.min_edge_cents_alpha is not None
    for edge, deflated in ((p.min_edge_cents_alpha, False), (p.min_edge_cents, True)):
        at = predicted_power(case, trades, edge, deflated=deflated)
        below = predicted_power(case, trades, edge - 1, deflated=deflated)
        assert 0.80 <= at <= 0.80 + ORACLE_CEIL_SLACK
        assert below < 0.80
        assert predicted_power(case, trades, edge, deflated=deflated, exact_design=True) >= at


@pytest.mark.parametrize("trades", sorted(CASES))
def test_the_gate_refuses_a_target_below_the_minimum_edge(trades: int) -> None:
    p = power_at(CASES[trades], trades)
    assert p.min_edge_cents is not None
    years = CASES[trades].run_days / 252
    from pfb.analysis.power import min_edge

    def gate(target: int) -> bool | None:
        return min_edge(trades_per_year=trades / years, trade_sd_cents=p.trade_sd_cents, years=years,
                        n_trials=N_TRIALS, target_edge_cents=target).passed  # fmt: skip

    assert gate(p.min_edge_cents - 1) is False
    assert gate(p.min_edge_cents) is True


def test_one_run_through_the_whole_suite_recovers_the_planted_edge() -> None:
    case = CASES[1000]
    trades = probe_trades(1000)
    edge = power_at(case, trades).min_edge_cents_alpha
    assert edge is not None
    result = run_planted(case, float(edge), seed=7, suite=SuiteSpec())
    got = outcome(result)
    assert abs(got.trades - case.trades) <= COUNT_TOLERANCE * case.trades
    assert got.off_cost_trades == 0  # every trade charged 2 x commission + 2 x one tick, as configured
    assert got.forced_exits == 0 and "PRICE_LIMIT" not in set(result.trades["exit_reason"].to_list())
    first_date = result.daily["trade_date"][0]
    assert first_date not in set(result.trades["trade_date"].to_list())  # no reference on the first date
    limits = result.summary["price_limits"]
    assert isinstance(limits, dict) and limits["enabled"] and limits["dates_without_reference"] == 1
    assert abs(got.mean_net - edge) <= ONE_RUN_SE_TOLERANCE * got.se
    assert abs(got.sd_net / trade_sd_cents() - 1.0) <= SD_TOLERANCE
    suite = result.summary["suite"]
    assert isinstance(suite, dict) and all(suite[k] is not None for k in SUITE_KEYS)
    power = suite["power"]
    assert power["trades"] == got.trades
    assert power["realized_edge_cents"] == int(result.trades["net_cents"].sum()) // got.trades
    design_alpha = power_at(case, got.trades).min_edge_cents_alpha
    assert design_alpha is not None
    assert abs(power["realized_min_edge_cents_alpha"] / design_alpha - 1.0) <= SD_TOLERANCE


def test_twenty_seeds_at_250_trades_recover_the_edge_within_three_standard_errors() -> None:
    case = CASES[250]
    edge = power_at(case, probe_trades(250)).min_edge_cents_alpha
    assert edge is not None
    runs = [outcome(run_planted(case, float(edge), seed=s, suite=None)) for s in FAST_SEEDS]
    total = sum(r.trades for r in runs)
    pooled = sum(r.mean_net * r.trades for r in runs) / total
    sd = math.sqrt(sum((r.sd_net**2) * (r.trades - 1) for r in runs) / (total - len(runs)))
    assert abs(pooled - edge) <= SE_TOLERANCE * sd / math.sqrt(total)
    assert all(r.off_cost_trades == 0 and r.forced_exits == 0 for r in runs)
    assert cost_per_trade_cents() > 0
