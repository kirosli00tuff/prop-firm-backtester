"""Planted-edge recovery over seeds, whole pipeline (marked slow; about 13 minutes; one run of
about 140,000 bars in memory at a time). Synthetic data only; nothing is registered.

At each stated trade count (250, 1,000, 4,000; CASES in planted_market.py) the same SEEDS are run at three
planted NET edges per trade: zero, min_edge_cents_alpha and min_edge_cents of pfb.analysis.power at the
case's design (N = N_TRIALS). A detection is the run summary's PSR (the alpha test) or DSR at N (the deflated
test) above 0.95. Tolerances (binomial SE at the predicted rate over the seeds):
- recovered mean net per trade, pooled over seeds: within 3 SE of the truth; seeds whose own 95 % interval
  misses the truth: at most 5 % + 3 SE;
- detection rate at a planted edge: between the module's power - 3 SE (the module's claim: at least 0.80 at
  its minimum) and this design's exact power + 3 SE (predicted_power exact_design; the module's figure is the
  conservative one). The upper end was first the fixed-count power with the module's k; it failed at 250
  trades (0.905 measured against at most 0.897) and was replaced after that run by the exact power, for the
  reason that the module's alpha k belongs to the deflated Sharpe;
- zero edge: PSR and DSR detection rates at most the nominal 5 % + 3 SE;
- an edge below min_edge_cents (the alpha minimum, smaller at every count) is not claimed: the gate refuses
  it, its DSR detection rate stays under 0.80 - 3 SE, and the suite's realized power check passes on at most
  5 % + 3 SE of the seeds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

import pytest
from planted_market import (
    CASES,
    Outcome,
    binomial_se,
    outcome,
    power_at,
    predicted_power,
    probe_trades,
    run_planted,
)

from pfb.analysis.spec import SuiteSpec

pytestmark = pytest.mark.slow

SEEDS = range(1, 201)  # 200 seeds: SE of a rate near 0.80 is 0.028; probe_trades uses seed 0
SE_TOLERANCE = 3.0
NOMINAL = 0.05
POWER_TARGET = 0.80
PSR_PASS = 0.95
LEVELS = ("zero", "alpha", "dsr")


@dataclass(frozen=True)
class Level:
    edge: int
    runs: tuple[Outcome, ...]

    def rate(self, key: str) -> float:
        hits = [getattr(r, key) is not None and getattr(r, key) > PSR_PASS for r in self.runs]
        return sum(hits) / len(hits)


@cache
def measured(trades: int) -> dict[str, Level]:
    case = CASES[trades]
    p = power_at(case, probe_trades(trades))
    assert p.min_edge_cents is not None and p.min_edge_cents_alpha is not None
    edges = {"zero": 0, "alpha": p.min_edge_cents_alpha, "dsr": p.min_edge_cents}
    # SuiteSpec.off() keeps the suite's power part (n_trials is given) and turns the costly parts off
    out = {k: Level(e, tuple(outcome(run_planted(case, float(e), s, SuiteSpec.off())) for s in SEEDS))
           for k, e in edges.items()}  # fmt: skip
    for k, lv in out.items():  # the table for the result file (pytest -rP shows it)
        n = sum(r.trades for r in lv.runs) / len(lv.runs)
        print(
            f"planted trades={trades} level={k} edge={lv.edge} mean_trades={n:.1f} "
            f"mean_net={sum(r.mean_net for r in lv.runs) / len(lv.runs):.2f} psr_rate={lv.rate('psr'):.3f} "
            f"dsr_rate={lv.rate('dsr'):.3f} power_passed={sum(bool(r.power_passed) for r in lv.runs)}"
            f" alpha_pred={predicted_power(case, probe_trades(trades), lv.edge, deflated=False):.3f}"
            f"/{predicted_power(case, probe_trades(trades), lv.edge, deflated=False, exact_design=True):.3f}"
            f" dsr_pred={predicted_power(case, probe_trades(trades), lv.edge, deflated=True):.3f}"
            f"/{predicted_power(case, probe_trades(trades), lv.edge, deflated=True, exact_design=True):.3f}"
        )
    return out


def _bracket(trades: int, edge: int, deflated: bool) -> tuple[float, float]:
    case, n = CASES[trades], probe_trades(trades)
    low = predicted_power(case, n, edge, deflated=deflated)
    high = predicted_power(case, n, edge, deflated=deflated, exact_design=True)
    draws = len(SEEDS)
    return low - SE_TOLERANCE * binomial_se(low, draws), high + SE_TOLERANCE * binomial_se(high, draws)


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("trades", sorted(CASES))
def test_recovered_mean_net_per_trade_is_within_three_standard_errors(trades: int, level: str) -> None:
    lv = measured(trades)[level]
    total = sum(r.trades for r in lv.runs)
    pooled = sum(r.mean_net * r.trades for r in lv.runs) / total
    sd = math.sqrt(sum(r.sd_net**2 * (r.trades - 1) for r in lv.runs) / (total - len(lv.runs)))
    assert abs(pooled - lv.edge) <= SE_TOLERANCE * sd / math.sqrt(total)
    missed = sum(abs(r.mean_net - lv.edge) > 1.96 * r.se for r in lv.runs) / len(lv.runs)
    assert missed <= NOMINAL + SE_TOLERANCE * binomial_se(NOMINAL, len(lv.runs))
    assert all(r.off_cost_trades == 0 and r.forced_exits == 0 for r in lv.runs)


@pytest.mark.parametrize("trades", sorted(CASES))
def test_detection_rate_brackets_the_predicted_power(trades: int) -> None:
    m = measured(trades)
    low, high = _bracket(trades, m["alpha"].edge, deflated=False)
    assert low <= m["alpha"].rate("psr") <= high
    low, high = _bracket(trades, m["dsr"].edge, deflated=True)
    assert low <= m["dsr"].rate("dsr") <= high


@pytest.mark.parametrize("trades", sorted(CASES))
def test_a_zero_edge_is_detected_at_no_more_than_the_nominal_rate(trades: int) -> None:
    zero = measured(trades)["zero"]
    bound = NOMINAL + SE_TOLERANCE * binomial_se(NOMINAL, len(zero.runs))
    assert zero.rate("psr") <= bound
    assert zero.rate("dsr") <= bound


@pytest.mark.parametrize("trades", sorted(CASES))
def test_an_edge_below_min_edge_is_not_claimed(trades: int) -> None:
    below = measured(trades)["alpha"]
    p = power_at(CASES[trades], probe_trades(trades))
    assert p.min_edge_cents is not None and below.edge < p.min_edge_cents  # the gate refuses it (fast test)
    assert below.rate("dsr") < POWER_TARGET - SE_TOLERANCE * binomial_se(POWER_TARGET, len(below.runs))
    low, high = _bracket(trades, below.edge, deflated=True)
    assert low <= below.rate("dsr") <= high
    passed = sum(bool(r.power_passed) for r in below.runs) / len(below.runs)
    assert passed <= NOMINAL + SE_TOLERANCE * binomial_se(NOMINAL, len(below.runs))
