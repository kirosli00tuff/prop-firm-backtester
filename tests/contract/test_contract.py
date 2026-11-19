"""Every command path's result validates against docs/result.schema.json, and every key equals its source
(the verified ledger lines, suite.json, neighbourhood.json, the prop artifacts). Synthetic bars only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from app_env import FIRM, PLAN, AppEnv

from pfb import app
from pfb.contract import CONTRACT_METRICS, KEYS, SCHEMA_PATH, validate
from pfb.registry.chain import read_ledger

SCHEMA = json.loads(SCHEMA_PATH.read_text())


def _schema_ok(doc: dict[str, Any]) -> None:
    """Checked with the jsonschema package directly, besides pfb.contract.validate."""
    jsonschema.validate(doc, SCHEMA, cls=jsonschema.Draft202012Validator)
    validate(doc)


def _lines(ledger: Path, run_id: str) -> tuple[Any, Any]:
    lines = [ln for ln in read_ledger(ledger) if ln.record.get("run_id") == run_id]
    begin = next(ln for ln in lines if ln.record["kind"] != "result")
    result = next((ln for ln in lines if ln.record["kind"] == "result"), None)
    return begin, result


def assert_sources(doc: dict[str, Any], a: AppEnv) -> None:
    """Each key against the record or artifact it comes from (pfb.contract module docstring)."""
    assert sorted(doc) == sorted(KEYS) and len(KEYS) == 33
    begin, result = _lines(a.ledger, doc["run_id"])
    b, r = begin.record, result.record
    assert (doc["run_id"], doc["kind"], doc["status"]) == (b["run_id"], b["kind"], r["status"])
    assert (doc["hypothesis_id"], doc["strategy"], doc["root"]) == (
        b["hypothesis_id"],
        b["strategy_path"],
        b["root"],
    )
    assert doc["params"] == b["params"] and doc["n"] == b["n_after"]
    assert doc["n_added"] == b["n_after"] - b["n_before"]
    assert doc["window"] == ([b["first"], b["last"]] if b["first"] else None)
    assert doc["ledger"] == {"begin_sha256": begin.sha256, "result_sha256": result.sha256}
    metrics = r["metrics"]
    for key, source in CONTRACT_METRICS.items():
        assert doc[key] == metrics.get(source), key
    run_dir = a.runs / doc["run_id"]
    suite = json.loads((run_dir / "suite.json").read_text())
    assert suite == metrics["suite"]
    stress = suite["cost_stress"]
    if stress is None:
        assert doc["cost_stress"] is None
    else:
        assert doc["cost_stress"] == {
            k: {"net_cents": v["net_cents"], "sharpe": v["sharpe_annual"]} for k, v in stress.items()
        }
    base = suite["baseline"]
    keys = ("draws", "mode", "net_percentile", "sharpe_percentile", "p_net", "p_sharpe")
    want = None if base is None else {k: base[k] for k in keys}
    assert doc["baseline"] == want
    if base is not None:  # the four figures are those of the named mode
        head = base[base["mode"]]
        assert (base["p_net"], base["net_percentile"]) == (head["net"]["p"], head["net"]["percentile"])
    wf = suite["walk_forward"]
    want_wf = None if wf is None else {"folds": wf["n_folds"], "sharpe_spread": wf["sharpe_spread"],
                                        "positive_folds": wf["positive_folds"]}  # fmt: skip
    assert doc["walk_forward"] == want_wf
    power = suite["power"]
    for key in ("realized_min_edge_cents", "realized_edge_cents", "passed"):
        assert doc["power"][key] == power[key]
    assert doc["artifacts"][: len(r["artifacts"])] == r["artifacts"]
    # the record's artifact hashes, each matching the file written under its name
    assert doc["artifact_sha256"] == r["artifact_sha256"] and set(r["artifact_sha256"]) == set(r["artifacts"])
    for name, sha in r["artifact_sha256"].items():
        assert hashlib.sha256((run_dir / name.rsplit("/", 1)[1]).read_bytes()).hexdigest() == sha, name
    assert doc["warnings"][: len(metrics["warnings"])] == metrics["warnings"]


# ----------------------------------------------------------------------------- synthetic ----
def test_a_synthetic_run_with_the_suite_matches_its_sources_and_is_saved(app_env: AppEnv) -> None:
    # Act
    doc = app.run_backtest(str(app_env.strategy), kind="synthetic", cost_model="fixed:1")
    # Assert
    _schema_ok(doc)
    assert_sources(doc, app_env)
    assert doc["status"] == "ok" and doc["n_added"] == 0 and doc["purpose"] is None
    assert set(doc["cost_stress"]) == {"1", "2"} and doc["baseline"]["draws"] == 200
    assert doc["walk_forward"]["folds"] == 5 and doc["neighbourhood"] is None and doc["prop"] is None
    assert doc["power"]["min_edge_cents"] is None and doc["power"]["target_edge_cents"] is None
    saved = app_env.runs / doc["run_id"] / "result.json"
    assert json.loads(saved.read_text()) == doc and doc["artifacts"][-1].endswith("/result.json")
    assert app.get_run(doc["run_id"]) == doc


def test_a_synthetic_run_without_suite_parts_has_them_null(app_env: AppEnv) -> None:
    doc = app.run_backtest(
        str(app_env.strategy), kind="synthetic", cost_model="fixed:1", stress=False, baseline=False,
        walk_forward=False,
    )  # fmt: skip
    _schema_ok(doc)
    assert_sources(doc, app_env)
    assert doc["cost_stress"] is None and doc["baseline"] is None and doc["walk_forward"] is None
    assert doc["power"] is not None  # the realized power is computed whenever a suite is given


def test_a_synthetic_neighbourhood_reports_the_verdict_file(app_env: AppEnv) -> None:
    n_before = app.trial_count()["n"]
    doc = app.run_backtest(str(app_env.strategy), kind="synthetic", cost_model="fixed:1", neighbourhood=True,
                           baseline=False)  # fmt: skip
    _schema_ok(doc)
    assert_sources(doc, app_env)
    nb = json.loads((app_env.runs / doc["run_id"] / "neighbourhood.json").read_text())
    assert doc["neighbourhood"] == {"verdict": nb["verdict"]["verdict"], "ratio": nb["verdict"]["ratio"],
                                    "run_ids": [n["run_id"] for n in nb["neighbours"]]}  # fmt: skip
    assert len(doc["neighbourhood"]["run_ids"]) == 3  # range 30; buffer 2 and 6
    for run_id in doc["neighbourhood"]["run_ids"]:
        other = app.get_run(run_id)
        _schema_ok(other)
        assert_sources(other, app_env)
        assert other["neighbourhood"] is None
    assert app.trial_count()["n"] == n_before


def test_prop_on_a_synthetic_run_matches_its_artifact(app_env: AppEnv) -> None:
    # Act
    doc = app.run_backtest(
        str(app_env.strategy),
        kind="synthetic",
        cost_model="fixed:1",
        firm=FIRM,
        plan=PLAN,
        mc=300,
        seed=3,
        baseline=False,
    )
    # Assert
    _schema_ok(doc)
    assert_sources(doc, app_env)
    [row] = doc["prop"]
    record = json.loads((app_env.runs / doc["run_id"] / "prop" / f"{FIRM}_{PLAN}_3.json").read_text())
    out = record["outputs"]
    assert row == {
        "firm": FIRM, "plan": PLAN, "status": "ok", "reason": None,
        "p_pass": out["p_pass"]["value"], "p_pass_se": out["p_pass"]["se"],
        "ev_per_purchase_cents": out["ev_per_purchase"]["value"], "ev_se_cents": out["ev_per_purchase"]["se"],
        "n_paths": 300, "seed": 3, "soft_daily_loss": "limit",
        "plan_version": record["plan_version"], "effective_from": record["effective_from"],
        "plan_file_sha256": record["plan_file_sha256"], "rules_in_force": "newest_undated",
    }  # fmt: skip
    # the row names the rules that produced it (version, effective_from, the file's sha256)
    assert isinstance(row["plan_version"], int) and row["plan_version"] >= 1
    assert len(row["effective_from"]) == 10 and len(row["plan_file_sha256"]) == 64
    assert any(w.startswith("rules: resampled day paths are undated") for w in record["warnings"])
    assert record["soft_daily_loss"] == "limit" and record["plan_file_sha256"] == record["plan_sha256"]
    assert record["plan_overrides"] == [] and record["mc"]["soft_daily_loss"] == "limit"
    assert any(a.endswith(f"/prop/{FIRM}_{PLAN}_3.json") for a in doc["artifacts"])


def test_run_prop_mc_on_a_saved_run_with_the_minute_low_switch(app_env: AppEnv) -> None:
    base = app.run_backtest(str(app_env.strategy), kind="synthetic", cost_model="fixed:1", baseline=False)
    doc = app.run_prop_mc(base["run_id"], firm=FIRM, plan=PLAN, mc=300, seed=5, soft_daily_loss="minute_low")
    _schema_ok(doc)
    [row] = doc["prop"]
    path = app_env.runs / base["run_id"] / "prop" / f"{FIRM}_{PLAN}_5_minute_low.json"
    record = json.loads(path.read_text())
    assert row["soft_daily_loss"] == "minute_low" == record["soft_daily_loss"]
    assert row["p_pass"] == record["outputs"]["p_pass"]["value"]
    assert {k: v for k, v in doc.items() if k not in ("prop", "artifacts")} == {
        k: v for k, v in base.items() if k not in ("prop", "artifacts")
    }
    assert app.get_run(base["run_id"]) == base  # result.json is not rewritten


def test_the_prop_matrix_through_the_app(app_env: AppEnv) -> None:
    base = app.run_backtest(str(app_env.strategy), kind="synthetic", cost_model="fixed:1", baseline=False)
    doc = app.run_prop_mc(base["run_id"], firm="*", plan="*", mc=200, seed=1)
    _schema_ok(doc)
    matrix = json.loads((app_env.runs / base["run_id"] / "prop_matrix.json").read_text())
    assert len(doc["prop"]) == matrix["n_plans"] and matrix["soft_daily_loss"] == "limit"
    for row, mrow in zip(doc["prop"], matrix["rows"], strict=True):
        assert (row["firm"], row["plan"], row["status"]) == (mrow["firm"], f"{mrow['plan']}_{mrow['size']}",
                                                              mrow["status"])  # fmt: skip
        if mrow["status"] == "ok":
            assert row["p_pass"] == mrow["record"]["outputs"]["p_pass"]["value"]
            assert mrow["record"]["soft_daily_loss"] == "limit"
            assert row["plan_file_sha256"] == mrow["record"]["plan_file_sha256"]  # labels
            assert (row["plan_version"], row["effective_from"]) == (
                mrow["record"]["plan_version"], mrow["record"]["effective_from"])  # fmt: skip
        else:
            assert row["plan_version"] is None and row["rules_in_force"] is None


def test_prop_copies_row_validates_and_matches_its_artifact(app_env: AppEnv) -> None:
    # Arrange:, N accounts copying one signal; nothing counts toward N
    base = app.run_backtest(str(app_env.strategy), kind="synthetic", cost_model="fixed:1", baseline=False)
    n_before = app.trial_count()["n"]
    # Act
    doc = app.run_prop_mc(base["run_id"], firm=FIRM, plan=PLAN, mc=200, seed=4, copies=3)
    # Assert
    _schema_ok(doc)
    [row] = doc["prop"]
    run_dir = app_env.runs / base["run_id"] / "prop"
    copies = json.loads((run_dir / f"{FIRM}_{PLAN}_4_copies3.json").read_text())
    single = json.loads((run_dir / f"{FIRM}_{PLAN}_4.json").read_text())
    assert row["copies"]["n_accounts"] == 3 == copies["n_accounts"]
    assert row["copies"]["copies"]["p_all_pass"] == copies["copies"]["p_all_pass"]["value"]
    assert row["copies"]["independent"]["p_all_pass"] == copies["independent"]["p_all_pass"]["value"]
    assert (
        row["p_pass"]
        == single["outputs"]["p_pass"]["value"]
        == copies["single"]["outputs"]["p_pass"]["value"]
    )
    assert copies["plan_file_sha256"] == row["plan_file_sha256"] and copies["trial"] is False
    assert any(a.endswith(f"/prop/{FIRM}_{PLAN}_4_copies3.json") for a in doc["artifacts"])
    assert app.trial_count()["n"] == n_before
    # without copies the row has no copies key and stays as before
    plain = app.run_prop_mc(base["run_id"], firm=FIRM, plan=PLAN, mc=200, seed=4)
    assert "copies" not in plain["prop"][0]
    assert {k: v for k, v in plain["prop"][0].items()} == {k: v for k, v in row.items() if k != "copies"}


# ------------------------------------------------------------------------ get_run, refusals ----
def test_get_run_rebuilds_from_the_ledger_when_result_json_is_absent(app_env: AppEnv) -> None:
    doc = app.run_backtest(str(app_env.strategy), kind="synthetic", cost_model="fixed:1", baseline=False)
    (app_env.runs / doc["run_id"] / "result.json").unlink()
    rebuilt = app.get_run(doc["run_id"])
    _schema_ok(rebuilt)
    assert rebuilt == {**doc, "artifacts": [a for a in doc["artifacts"] if not a.endswith("/result.json")]}


@pytest.mark.parametrize(
    ("call", "code"),
    [
        (lambda s: app.run_backtest(s, kind="bogus"), "BadRequest"),
        (lambda s: app.run_backtest(s, kind="trial", hypothesis="h.toml", stress=False), "SuiteMismatch"),
        (
            lambda s: app.run_backtest(s, kind="trial", hypothesis="h.toml", firm=FIRM, plan=PLAN, mc=5),
            "SuiteMismatch",
        ),
        (lambda s: app.run_backtest(s, kind="synthetic", firm=FIRM), "BadRequest"),
        (lambda s: app.run_backtest(s, kind="synthetic", window=[1]), "BadRequest"),
        (lambda s: app.run_backtest(s, kind="synthetic", params={"nope": 1}), "RunRefused"),
        (lambda s: app.get_run("synthetic-nope"), "RunNotFound"),
        (lambda s: app.get_run("../x"), "BadRunId"),
        (lambda s: app.compare_runs([]), "BadRequest"),
    ],
)
def test_refusals_carry_a_code_and_validate(app_env: AppEnv, call: Any, code: str) -> None:
    n_before = app.trial_count()["n"]
    with pytest.raises(app.Refused) as caught:
        call(str(app_env.strategy))
    assert caught.value.code == code
    out = caught.value.to_json()
    _schema_ok(out)
    assert out == {
        "schema": "pfb.result/1",
        "status": "refused",
        "error": {"code": code, "reason": caught.value.reason},
    }
    assert app.trial_count()["n"] == n_before


def test_compare_runs_reads_each_saved_result(app_env: AppEnv) -> None:
    a = app.run_backtest(str(app_env.strategy), kind="synthetic", cost_model="fixed:1", baseline=False)
    b = app.run_backtest(str(app_env.strategy), kind="synthetic", cost_model="fixed:1", baseline=False,
                         params={"buffer_ticks": 6})  # fmt: skip
    out = app.compare_runs([a["run_id"], b["run_id"]])
    assert [r["run_id"] for r in out["rows"]] == [a["run_id"], b["run_id"]]
    assert out["rows"][1]["params"]["buffer_ticks"] == 6 and out["rows"][0]["net_cents"] == a["net_cents"]
    assert out["warnings"] == []
