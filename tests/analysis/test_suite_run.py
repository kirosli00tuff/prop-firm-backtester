"""run_strategy with and without a suite (synthetic bars only; nothing is registered)."""

from __future__ import annotations

import hashlib
import json

import pytest
from suite_planting import REFERENCE

from pfb.analysis.inrun import SUITE_KEYS
from pfb.analysis.spec import SuiteSpec
from pfb.engine.run import run_strategy

# sha256 of json.dumps(summary) for this run (pfb-engine/0.3.1, N = 1000, an arbitrary test value), pinned
# from the engine; a suite must leave it byte for byte unchanged. No PRICE_LIMIT exit here.
GOLDEN_SUMMARY_SHA256 = "ca4f6cbb306f8080f3011cd9c0eef5f659dd349b1e8291e859382d8e7f01db46"
GOLDEN_SORTED_SHA256 = {
    (3, 1000): "303b382a84f0602595bb86697a46cb8fc273db1bb7a7ab161ba641d1c8f7e85a",
    (5, None): "4d8913dd64617cfe44052cf1a6e3acff33b1de6a032aee24383f6fcf988bcb46",
}


def _sha(obj: object, *, sort: bool) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=sort).encode()).hexdigest()


def test_a_run_without_a_suite_is_byte_identical_to_the_engine_before_the_suite() -> None:
    result = run_strategy(REFERENCE, source="synthetic", cost_model="fixed:1", seed=3, n_trials=1000)
    assert _sha(result.summary, sort=False) == GOLDEN_SUMMARY_SHA256
    assert "suite" not in result.summary
    assert result.extras == {}


@pytest.mark.parametrize(("seed", "n_trials"), list(GOLDEN_SORTED_SHA256))
def test_suite_none_changes_nothing(seed: int, n_trials: int | None) -> None:
    result = run_strategy(REFERENCE, source="synthetic", cost_model="fixed:1", seed=seed, n_trials=n_trials,
                          suite=None)  # fmt: skip
    assert _sha(result.summary, sort=True) == GOLDEN_SORTED_SHA256[(seed, n_trials)]


def test_a_suite_leaves_every_other_summary_key_unchanged_and_adds_the_parts() -> None:
    plain = run_strategy(REFERENCE, source="synthetic", cost_model="fixed:1", seed=3, n_trials=1000)
    full = run_strategy(REFERENCE, source="synthetic", cost_model="fixed:1", seed=3, n_trials=1000,
                        suite=SuiteSpec())  # fmt: skip
    suite = full.summary.pop("suite")
    assert full.summary == plain.summary
    assert isinstance(suite, dict)
    assert set(suite) == {*SUITE_KEYS, "spec"}
    assert all(suite[k] is not None for k in SUITE_KEYS)
    json.dumps(suite, allow_nan=False)  # strict JSON
    assert set(full.extras) == {"baseline_draws", "rolling_sharpe"}
    assert full.extras["baseline_draws"].height == 3 * 200  # uniform, tod, cost_matched


def test_parts_turned_off_are_none() -> None:
    result = run_strategy(REFERENCE, source="synthetic", cost_model="fixed:1", seed=3, suite=SuiteSpec.off())
    suite = result.summary["suite"]
    assert isinstance(suite, dict)
    assert suite["cost_stress"] is None and suite["baseline"] is None and suite["walk_forward"] is None
    assert suite["power"] is None  # n_trials None
    assert suite["breakdowns"] is not None and suite["sizing"] is not None  # always computed
    assert "baseline_draws" not in result.extras


def test_power_uses_the_realized_figures_and_the_runs_n() -> None:
    from pfb.analysis.power import min_edge

    result = run_strategy(REFERENCE, source="synthetic", cost_model="fixed:1", seed=3, n_trials=1000,
                          suite=SuiteSpec.off())  # fmt: skip
    power = result.summary["suite"]["power"]  # type: ignore[index]
    net = result.trades["net_cents"].to_numpy()
    days = result.daily.height
    assert power["trades"] == len(net) and power["days"] == days
    assert power["realized_edge_cents"] == int(net.sum()) // len(net)
    assert power["trade_sd_cents"] == round(float(net.std()))
    expected = min_edge(trades_per_year=len(net) / (days / 252), trade_sd_cents=power["trade_sd_cents"],
                        years=days / 252, n_trials=1000, target_edge_cents=power["realized_edge_cents"],
                        trade_skew=power["trade_skew"], trade_kurtosis=power["trade_kurtosis"])  # fmt: skip
    assert power["realized_min_edge_cents"] == expected.min_edge_cents
    assert power["passed"] == expected.passed


def test_walk_forward_folds_sum_to_the_run_net() -> None:
    folds_only = SuiteSpec(cost_stress_ticks=(), baseline_draws=0, walk_forward_folds=3)
    result = run_strategy(REFERENCE, source="synthetic", cost_model="fixed:1", seed=3, suite=folds_only)
    wf = result.summary["suite"]["walk_forward"]  # type: ignore[index]
    assert sum(f["net_cents"] for f in wf["folds"]) == result.summary["net_cents"]
    assert sum(f["trades"] for f in wf["folds"]) == result.trades.height
