"""pfb.contract.build_result and validate on hand-built records (no run, no ledger)."""

from __future__ import annotations

from typing import Any

import pytest

from pfb.contract import (
    BASELINE_SD0_WARNING,
    KEYS,
    POWER_NONE_WARNING,
    POWER_WARNING,
    ContractError,
    build_result,
    refusal,
    validate,
)

BEGIN: dict[str, Any] = {
    "run_id": "synthetic-20261116T000000000000Z-00000001", "kind": "synthetic", "hypothesis_id": "h_x",
    "strategy_path": "strategies/x.py", "root": "MNQ", "first": None, "last": None, "params": {"k": 1},
    "n_before": 1000, "n_after": 1000, "power": None,
}  # fmt: skip
SHA = "a" * 64


def _suite(power: dict[str, Any] | None) -> dict[str, Any]:
    return {"cost_stress": None, "baseline": None, "walk_forward": None, "power": power}


def test_an_open_run_has_status_open_and_null_metrics() -> None:
    doc = build_result(begin=BEGIN, result=None, begin_sha256=SHA, result_sha256=None)
    validate(doc)
    assert doc["status"] == "open" and doc["ledger"] == {"begin_sha256": SHA, "result_sha256": None}
    assert all(doc[k] is None for k in ("net_cents", "sharpe", "dsr", "cost_stress", "power", "prop"))
    assert sorted(doc) == sorted(KEYS) and doc["window"] is None and doc["n_added"] == 0


def test_metrics_map_from_the_summary_names_and_nan_becomes_null() -> None:
    metrics = {"net_cents": 5, "n_trades": 3, "sharpe_annual": float("nan"), "sortino_annual": 1.5,
               "warnings": ["w"], "suite": _suite(None)}  # fmt: skip
    result = {"status": "ok", "metrics": metrics, "artifacts": ["runs/x/summary.json"], "warnings": []}
    doc = build_result(begin=BEGIN, result=result, begin_sha256=SHA, result_sha256=SHA, warnings=["late"])
    validate(doc)
    assert (doc["net_cents"], doc["trades"], doc["sharpe"], doc["sortino"]) == (5, 3, None, 1.5)
    assert doc["warnings"] == ["w", "late"] and doc["artifacts"] == ["runs/x/summary.json"]


def test_a_realized_minimum_above_the_stated_one_warns() -> None:
    begin = {**BEGIN, "kind": "trial", "run_id": "trial-1", "first": "2025-04-01", "last": "2025-04-30",
             "power": {"n_used": 1002, "min_edge_cents": 900, "target_edge_cents": 1000}}  # fmt: skip
    realized = {"realized_min_edge_cents": 1200, "realized_edge_cents": 800, "passed": False}
    result = {"status": "ok", "metrics": {"suite": _suite(realized), "warnings": []}}
    doc = build_result(begin=begin, result=result, begin_sha256=SHA, result_sha256=SHA)
    validate(doc)
    assert doc["power"]["min_edge_cents_alpha"] is None  # no hypothesis record given: not filled
    assert doc["warnings"] == [POWER_WARNING.format(real=1200, stated=900)]
    assert doc["window"] == ["2025-04-01", "2025-04-30"]


def _baseline(mode: str, *, tod_sd: float = 1500.0) -> dict[str, Any]:
    def part(pct: float, p: float, sd: float) -> dict[str, Any]:
        return {
            "net": {"percentile": pct, "p": p, "sd": sd},
            "sharpe": {"percentile": pct, "p": p, "sd": 0.1},
        }

    modes = {"uniform": part(0.97, 0.03, 7000.0), "tod": part(0.6, 0.4, tod_sd)}
    head = modes[mode]
    return {"draws": 1000, "seed": 7, "trades": 50, **modes, "mode": mode,
            "net_percentile": head["net"]["percentile"], "sharpe_percentile": head["sharpe"]["percentile"],
            "p_net": head["net"]["p"], "p_sharpe": head["sharpe"]["p"]}  # fmt: skip


def _with_baseline(baseline: dict[str, Any]) -> dict[str, Any]:
    suite = {**_suite(None), "baseline": baseline}
    return {"status": "ok", "metrics": {"suite": suite, "warnings": []}}


def test_the_baseline_names_its_mode_and_carries_that_modes_four_figures() -> None:
    doc = build_result(
        begin=BEGIN, result=_with_baseline(_baseline("tod")), begin_sha256=SHA, result_sha256=SHA
    )
    validate(doc)
    assert doc["baseline"] == {"draws": 1000, "mode": "tod", "net_percentile": 0.6, "sharpe_percentile": 0.6,
                               "p_net": 0.4, "p_sharpe": 0.4}  # fmt: skip
    assert doc["warnings"] == []


def test_the_schema_refuses_a_baseline_without_a_known_mode() -> None:
    doc = build_result(
        begin=BEGIN, result=_with_baseline(_baseline("tod")), begin_sha256=SHA, result_sha256=SHA
    )
    doc["baseline"]["mode"] = "headline"
    with pytest.raises(ContractError):
        validate(doc)
    del doc["baseline"]["mode"]
    with pytest.raises(ContractError):
        validate(doc)


def test_a_reported_mode_whose_draws_are_all_equal_warns_that_the_percentile_is_uninformative() -> None:
    result = _with_baseline(_baseline("tod", tod_sd=0.0))
    doc = build_result(begin=BEGIN, result=result, begin_sha256=SHA, result_sha256=SHA)
    validate(doc)
    assert doc["warnings"] == [BASELINE_SD0_WARNING.format(mode="tod")]
    # the other mode's sd 0 is not the reported one: no warning
    other = _with_baseline(_baseline("uniform", tod_sd=0.0))
    assert build_result(begin=BEGIN, result=other, begin_sha256=SHA, result_sha256=SHA)["warnings"] == []


TRIAL_BEGIN: dict[str, Any] = {
    **BEGIN, "kind": "trial", "run_id": "trial-1", "first": "2025-04-01", "last": "2025-04-30",
    "power": {"n_used": 1002, "min_edge_cents": 900, "target_edge_cents": 1000},
}  # fmt: skip


@pytest.mark.parametrize(
    ("realized", "reason"),
    [
        ({"realized_min_edge_cents": None, "realized_edge_cents": 800, "passed": False, "detail": {}},
         "undetectable at the run's trades, sd and days"),
        ({"realized_min_edge_cents": None, "realized_edge_cents": 800, "passed": None,
          "reason": "fewer than two trades, a zero per-trade sd or fewer than two days"},
         "fewer than two trades, a zero per-trade sd or fewer than two days"),
        (None, "the run has no realized power figures"),
    ],
)  # fmt: skip
def test_a_realized_minimum_of_none_warns_with_the_reason(
    realized: dict[str, Any] | None, reason: str
) -> None:
    result = {"status": "ok", "metrics": {"suite": _suite(realized), "warnings": []}}
    doc = build_result(begin=TRIAL_BEGIN, result=result, begin_sha256=SHA, result_sha256=SHA)
    validate(doc)
    assert doc["warnings"] == [POWER_NONE_WARNING.format(reason=reason, stated=900)]


def test_no_power_warning_without_a_stated_minimum() -> None:
    realized = {"realized_min_edge_cents": None, "realized_edge_cents": 800, "passed": False}
    result = {"status": "ok", "metrics": {"suite": _suite(realized), "warnings": []}}
    doc = build_result(begin=BEGIN, result=result, begin_sha256=SHA, result_sha256=SHA)
    assert doc["warnings"] == []  # a synthetic run states no minimum


def test_the_stated_alpha_minimum_uses_the_runs_own_window_not_the_sum_of_windows() -> None:
    from datetime import date

    from pfb.analysis.power import min_edge, years_of
    from pfb.registry.powercheck import window_years

    inputs = {"trades_per_year": 220.0, "trade_sd_cents": 21000, "trade_skew": 0.0, "trade_kurtosis": 3.0,
              "target_edge_cents": 30000}  # fmt: skip
    own = ["2024-01-02", "2025-12-31"]
    hypothesis = {"purpose": "research", "power": inputs, "windows": [["2016-01-04", "2026-05-08"], own]}
    begin = {**TRIAL_BEGIN, "first": own[0], "last": own[1]}
    doc = build_result(begin=begin, result=None, begin_sha256=SHA, result_sha256=None,
                       hypothesis=hypothesis)  # fmt: skip
    validate(doc)
    want = min_edge(trades_per_year=220.0, trade_sd_cents=21000, trade_skew=0.0, trade_kurtosis=3.0,
                    years=window_years([own]), n_trials=1002, target_edge_cents=30000)  # fmt: skip
    summed_years = years_of(
        [(date.fromisoformat(f), date.fromisoformat(t)) for f, t in hypothesis["windows"]]
    )
    summed = min_edge(trades_per_year=220.0, trade_sd_cents=21000, trade_skew=0.0, trade_kurtosis=3.0,
                      years=summed_years, n_trials=1002, target_edge_cents=30000)  # fmt: skip
    assert doc["power"]["min_edge_cents_alpha"] == want.min_edge_cents_alpha
    assert want.min_edge_cents_alpha is not None and want.min_edge_cents_alpha != summed.min_edge_cents_alpha


@pytest.mark.parametrize("mutate", ["extra", "missing", "kind", "sha"])
def test_the_schema_refuses_a_malformed_result(mutate: str) -> None:
    doc = build_result(begin=BEGIN, result=None, begin_sha256=SHA, result_sha256=None)
    if mutate == "extra":
        doc["metrics"] = {}
    elif mutate == "missing":
        del doc["dsr"]
    elif mutate == "kind":
        doc["kind"] = "experiment"
    else:
        doc["ledger"]["begin_sha256"] = "x"
    with pytest.raises(ContractError):
        validate(doc)


def test_refusal_objects() -> None:
    validate(refusal("SuiteMismatch", "fixed by the hypothesis"))
    validate(refusal("ZeroDivisionError", "boom", status="error"))
    with pytest.raises(ContractError):
        validate({**refusal("X", "y"), "extra": 1})
