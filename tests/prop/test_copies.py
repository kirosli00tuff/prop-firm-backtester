"""Several accounts on one signal (pfb.prop.copies). Synthetic day paths only: coin-flip days
of +/-500 USD on the known-answer fixture plans (tests/prop/fixtures), nothing read from a store, no trial."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from _synth import closes

from pfb.prop.config import FIELD_TAGS, Leaf, load_plan
from pfb.prop.copies import (
    COPY_RULES,
    MAX_ACCOUNTS,
    SCOPE_NOT_MODELLED,
    account_paths,
    check_accounts,
    column,
    copies_artifact_name,
    copies_row,
    copy_rules,
    evaluate_copies,
    pearson,
    run_paths,
)
from pfb.prop.matrix import McSettings, PlanRefused, evaluate_plan
from pfb.prop.mc import run_mc

FIXTURES = Path(__file__).resolve().parent / "fixtures"
KA_STATIC = FIXTURES / "ka_static_50k.toml"
KA_EOD = FIXTURES / "ka_eod_trailing_50k.toml"


def _coin_days(n_days: int = 120, seed: int = 3) -> object:
    rng = np.random.default_rng(seed)
    return closes([50_000 if x else -50_000 for x in rng.random(n_days) < 0.5])


def _settings(n_paths: int = 400, seed: int = 5, **kw: object) -> McSettings:
    return McSettings(n_paths=n_paths, seed=seed, kind="synthetic", horizon_eval=40, horizon_funded=40,
                      **kw)  # type: ignore[arg-type]  # fmt: skip


def _copies(n_accounts: int, path: Path = KA_STATIC, days: object = None, **kw: object) -> tuple:
    days = _coin_days() if days is None else days
    return evaluate_copies(path, days, _settings(**kw), n_accounts, repo_root=FIXTURES,  # type: ignore[arg-type]
                           cap_check=lambda _d, _p: [])  # fmt: skip


# ------------------------------------------------------------ the per-path run is run_mc's ----
@pytest.mark.parametrize("chunk", [10_000, 150])
def test_run_paths_reproduces_run_mc_exactly(chunk: int) -> None:
    # Arrange
    plan, days, s = load_plan(KA_EOD), _coin_days(), _settings(n_paths=400)
    # Act
    pr = run_paths(plan, days, s, chunk_paths=chunk)  # type: ignore[arg-type]
    ref = run_mc(plan, days, n_paths=400, seed=5, horizon_eval=40, horizon_funded=40, kind="synthetic",  # type: ignore[arg-type]
                 chunk_paths=chunk)  # fmt: skip
    # Assert
    assert pr.result == ref
    assert pr.outcomes.eval_end.shape == (400,)


def test_the_engine_is_deterministic_so_copies_are_counted_once() -> None:
    plan, days, s = load_plan(KA_STATIC), _coin_days(), _settings()
    a, b = account_paths(run_paths(plan, days, s)), account_paths(run_paths(plan, days, s))  # type: ignore[arg-type]
    assert np.array_equal(a.passed, b.passed) and np.array_equal(a.net, b.net)  # type: ignore[arg-type]
    assert np.array_equal(a.payout, b.payout)  # type: ignore[arg-type]


# --------------------------------------------------------- N = 1 is the single-account result ----
def test_one_account_equals_the_single_account_result_exactly() -> None:
    # Arrange
    days = _coin_days()
    single, _ = evaluate_plan(KA_STATIC, days, _settings(), repo_root=FIXTURES, cap_check=lambda _d, _p: [])  # type: ignore[arg-type]
    # Act
    doc, record, _ = _copies(1, days=days)
    # Assert
    assert record == single and doc["single"] == single
    p, ev = single["outputs"]["p_pass"], single["outputs"]["ev_per_purchase"]
    for col in (doc["copies"],):
        assert col["p_all_pass"]["value"] == p["value"] and col["p_all_pass"]["se"] == p["se"]
        assert col["p_any_pass"]["value"] == p["value"]
        assert col["net_sum_cents"]["mean"] == ev["value"] and col["net_sum_cents"]["mean_se"] == ev["se"]
        assert col["corr_pass_two_accounts"] is None and col["corr_net_two_accounts"] is None
    prod = doc["product_of_marginals"]
    assert prod["p_all_pass"] == p["value"] and prod["corr_two_accounts"] is None
    assert prod["net_sum_cents"]["mean"] == ev["value"]
    assert doc["linear_expectations"]["ev_per_purchase_cents"] == ev["value"]
    assert doc["copies"]["net_sum_cents"]["sd"] == prod["net_sum_cents"]["sd"]


def test_the_result_carries_the_single_result_rules_labels() -> None:
    doc, record, _ = _copies(2)
    for key in ("firm", "plan", "size", "plan_version", "effective_from", "rules_read", "plan_file_sha256",
                "plan_sha256", "rules_in_force", "soft_daily_loss", "seed", "n_paths", "kind"):  # fmt: skip
        assert doc[key] == record[key], key
    assert doc["trial"] is False and doc["schema"] == "pfb.prop_copies/1" and doc["n_accounts"] == 2
    assert doc["warnings"] == record["warnings"]


# ------------------------------------------------- copies: correlation 1, no diversification ----
def test_copies_move_together_and_all_pass_is_the_single_pass_rate() -> None:
    # Arrange
    n = 3
    # Act
    doc, record, _ = _copies(n)
    # Assert
    col, p = doc["copies"], record["outputs"]["p_pass"]["value"]
    assert 0.0 < p < 1.0
    assert col["corr_pass_two_accounts"] == 1.0 and col["corr_net_two_accounts"] == 1.0
    assert col["p_all_pass"]["value"] == p == col["p_any_pass"]["value"]
    k = round(p * record["n_paths"])
    assert col["p_all_fail"]["value"] == round((record["n_paths"] - k) / record["n_paths"], 6)
    assert col["accounts_passed_mean"] == pytest.approx(n * p, abs=1e-6)
    ev = record["outputs"]["ev_per_purchase"]["value"]
    assert col["net_sum_cents"]["mean"] == pytest.approx(n * ev, abs=1e-3)
    # every sum is N times a one-account value, so every quantile is a multiple of N
    assert all(v % n == 0 for v in col["net_sum_cents"]["quantiles"].values())


def test_copies_sd_is_sqrt_n_times_the_independent_sd() -> None:
    n = 4
    doc, _, _ = _copies(n)
    for name in ("net_sum_cents", "payout_sum_cents"):
        copies_sd = doc["copies"][name]["sd"]
        indep_sd = doc["product_of_marginals"][name]["sd"]
        assert indep_sd > 0 and copies_sd / indep_sd == pytest.approx(math.sqrt(n), rel=1e-6)


def test_a_degenerate_marginal_has_no_correlation_and_all_pass_one() -> None:
    days = closes([50_000] * 30)  # every day +500 USD: every path passes the 1,000 USD target
    doc, record, _ = _copies(3, days=days)
    col = doc["copies"]
    assert record["outputs"]["p_pass"]["value"] == 1.0
    assert col["p_all_pass"]["value"] == 1.0 and col["p_all_fail"]["value"] == 0.0
    assert col["corr_pass_two_accounts"] is None  # no variance: undefined, not 1
    assert doc["product_of_marginals"]["p_all_pass"] == 1.0


# -------------------------------------------- independent: the product of marginals, in tolerance ----
def test_the_independent_column_matches_the_product_of_marginals() -> None:
    # Arrange: 4 SEs is the stated tolerance on every drawn value
    n, n_paths = 3, 3000
    # Act
    doc, record, _ = _copies(n, n_paths=n_paths)
    # Assert
    ind, prod = doc["independent"], doc["product_of_marginals"]
    for key in ("p_all_pass", "p_all_fail", "p_any_pass"):
        est = ind[key]
        assert abs(est["value"] - prod[key]) <= 4 * max(est["se"], 1 / n_paths), key
    assert abs(ind["corr_pass_two_accounts"]) <= 4 / math.sqrt(n_paths)
    assert abs(ind["corr_net_two_accounts"]) <= 4 / math.sqrt(n_paths)
    for name in ("net_sum_cents", "payout_sum_cents"):
        assert abs(ind[name]["mean"] - prod[name]["mean"]) <= 4 * ind[name]["mean_se"], name
        assert ind[name]["sd"] == pytest.approx(prod[name]["sd"], rel=0.1), name
    # and the dependence is what differs: copies all-fail = 1 - p, independent (1 - p)^N
    p = record["outputs"]["p_pass"]["value"]
    assert prod["p_all_fail"] == pytest.approx((1 - p) ** n, abs=1e-6)
    assert prod["p_all_fail"] < doc["copies"]["p_all_fail"]["value"]


def test_the_independent_draws_are_seeded() -> None:
    a, _, _ = _copies(3)
    b, _, _ = _copies(3)
    c, _, _ = _copies(3, seed=6)
    assert a["independent"] == b["independent"] and a["independent"] != c["independent"]


# ------------------------------------------------------------- plan rules and refusals ----
def test_no_copy_rule_is_a_pfb_rules_1_field_and_each_is_named_missing() -> None:
    doc, _, _ = _copies(2)
    named = {item["rule"]: item for item in doc["not_modelled"]}
    for rule, _what, keys in COPY_RULES:
        assert not set(keys) & set(FIELD_TAGS), rule
        assert named[rule]["plan_keys_missing"] == list(keys)
    for rule, _what in SCOPE_NOT_MODELLED:
        assert named[rule]["plan_keys_missing"] is None
    assert any("eval.max_contracts_minis" in a for a in doc["applied"])


def test_a_plan_holding_a_copy_rule_is_refused_until_applied() -> None:
    plan = load_plan(KA_STATIC)
    leaf = Leaf("funded.copy_trading", "allowed", None, "S01")
    with pytest.raises(PlanRefused) as info:
        copy_rules(replace(plan, leaves={**plan.leaves, "funded.copy_trading": leaf}))
    assert info.value.field == "funded.copy_trading" and info.value.error == "CopyRuleNotApplied"


@pytest.mark.parametrize("bad", [0, -1, True, MAX_ACCOUNTS + 1, 2.0, "3"])
def test_the_account_count_is_checked(bad: object) -> None:
    with pytest.raises(PlanRefused) as info:
        check_accounts(bad)
    assert info.value.field == "copies"


def test_unknown_fees_or_funded_phase_null_only_the_money() -> None:
    plan, days, s = load_plan(KA_STATIC), _coin_days(), _settings()
    pr = run_paths(plan, days, s)  # type: ignore[arg-type]
    no_fee = account_paths(replace(pr, fees=replace(pr.fees, activation=None)))
    assert no_fee.net is None and no_fee.payout is not None and "fee" in no_fee.missing[0]
    no_funded = account_paths(replace(pr, want_funded=False))
    assert no_funded.payout is None and no_funded.net is None
    col = column(2, 2 * no_funded.passed, None, None, (no_funded.passed, no_funded.passed), None)
    assert col["payout_sum_cents"] is None and col["net_sum_cents"] is None
    assert col["corr_pass_two_accounts"] == 1.0


def test_pearson_is_exact_on_identical_and_opposite_arrays() -> None:
    x = np.array([3, -7, 11, 0, 5_000_000_000], dtype=np.int64)  # sums past int64 when squared
    assert pearson(x, x) == 1.0 and pearson(x, -x) == -1.0
    assert pearson(x, np.zeros(5, dtype=np.int64)) is None


def test_contract_row_and_artifact_name() -> None:
    doc, _, _ = _copies(2)
    row = copies_row(doc)
    assert row["n_accounts"] == 2 and set(row) == {"n_accounts", "copies", "independent", "not_modelled"}
    assert row["copies"]["p_all_pass"] == doc["copies"]["p_all_pass"]["value"]
    assert row["copies"]["net_sum_q50_cents"] == doc["copies"]["net_sum_cents"]["quantiles"]["q50"]
    assert row["independent"]["corr_pass_two_accounts"] == doc["independent"]["corr_pass_two_accounts"]
    assert copies_artifact_name("topstep_ka_static_50k_5.json", 3) == "topstep_ka_static_50k_5_copies3.json"


# ---------------------------------------------------------------------- through the CLI ----
def test_cli_copies_on_a_synthetic_run(tmp_path: Path) -> None:
    import json

    from test_cli_prop import RUN_ID, _firms, _invoke, _single, _write_run

    from pfb.contract import validate

    # Arrange
    _write_run(tmp_path / "runs", RUN_ID)
    firms = _firms(tmp_path)
    # Act
    res = _invoke([*_single(), "--copies", "2"], tmp_path, firms)
    # Assert
    assert res.exit_code == 0, res.output
    contract = json.loads(res.stdout)
    validate(contract)
    [row] = contract["prop"]
    saved = json.loads(
        (tmp_path / "runs" / RUN_ID / "prop" / "topstep_ka_static_50k_11_copies2.json").read_text()
    )
    assert row["copies"] == copies_row(saved) and saved["n_accounts"] == 2
    assert row["copies"]["copies"]["corr_pass_two_accounts"] == 1.0
    assert saved["run_id"] == RUN_ID and saved["single"]["outputs"]["p_pass"]["value"] == row["p_pass"]


def test_cli_copies_refused_on_a_trial_and_on_the_matrix(tmp_path: Path) -> None:
    from test_cli_prop import TRIAL_ID, _error, _firms, _invoke, _trial_ledger, _write_run

    _write_run(tmp_path / "runs", TRIAL_ID)
    firms = _firms(tmp_path)
    ledger = _trial_ledger(tmp_path, ["topstep/ka_static_50k"])
    declared = [TRIAL_ID, "--firm", "topstep", "--plan", "ka_static", "--size", "50k", "--mc", "300",
                "--seed", "11", "--copies", "2"]  # fmt: skip
    res = _invoke(declared, tmp_path, firms, ledger=ledger)
    assert res.exit_code == 2 and _error(res)["code"] == "TrialNotDeclared"
    res = _invoke(["matrix", TRIAL_ID, "--copies", "2"], tmp_path, firms, ledger=ledger)
    assert res.exit_code == 2
    assert not (tmp_path / "runs" / TRIAL_ID / "prop").exists()
