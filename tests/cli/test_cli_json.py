"""J1-J2 through the command line: `pfb run --json` and `pfb prop --json` print the result contract and
nothing else on stdout; a refusal prints the refusal object and exits 2; `pfb new` and `pfb check`
on the template.
Temporary repository and ledger, synthetic bars (tests/contract/app_env.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
from app_env import FIRM, PLAN, TEMPLATE, AppEnv  # pyright: ignore[reportMissingImports]
from typer.testing import CliRunner

from pfb import app as service
from pfb.cli.main import app
from pfb.contract import SCHEMA_PATH

runner = CliRunner()
SCHEMA = json.loads(SCHEMA_PATH.read_text())


def _json(res: Any) -> dict[str, Any]:
    """stdout holds exactly one JSON object, valid under docs/result.schema.json."""
    lines = res.stdout.strip().splitlines()
    assert len(lines) == 1, res.stdout
    doc = json.loads(lines[0])
    jsonschema.validate(doc, SCHEMA, cls=jsonschema.Draft202012Validator)
    return doc


def _run(a: AppEnv, *extra: str) -> Any:
    args = ["run", str(a.strategy), "--kind", "synthetic", "--cost-model", "fixed:1", "--json", *extra]
    return runner.invoke(app, args)


def test_pfb_run_json_with_the_suite(app_env: AppEnv) -> None:
    res = _run(app_env)
    assert res.exit_code == 0, res.output
    doc = _json(res)
    assert doc["status"] == "ok" and doc["cost_stress"] is not None and doc["baseline"] is not None
    assert service.get_run(doc["run_id"]) == doc


def test_pfb_run_json_without_suite_parts(app_env: AppEnv) -> None:
    res = _run(app_env, "--no-stress", "--no-baseline", "--no-walk-forward")
    assert res.exit_code == 0, res.output
    doc = _json(res)
    assert (doc["cost_stress"], doc["baseline"], doc["walk_forward"]) == (None, None, None)


def test_pfb_run_json_neighbourhood(app_env: AppEnv) -> None:
    res = _run(app_env, "--neighbourhood", "--no-baseline")
    assert res.exit_code == 0, res.output
    doc = _json(res)
    assert doc["neighbourhood"]["verdict"] in ("plateau", "spike", "mixed", "n/a")
    assert len(doc["neighbourhood"]["run_ids"]) == 3


def test_pfb_run_json_with_a_prop_plan_then_pfb_prop(app_env: AppEnv) -> None:
    res = _run(app_env, "--no-baseline", "--firm", FIRM, "--plan", PLAN, "--mc", "200", "--seed", "2")
    assert res.exit_code == 0, res.output
    doc = _json(res)
    assert [r["plan"] for r in doc["prop"]] == [PLAN]
    stem, _, size = PLAN.rpartition("_")
    prop = runner.invoke(app, [
        "prop", doc["run_id"], "--firm", FIRM, "--plan", stem, "--size", size, "--mc", "200",
        "--seed", "2", "--soft-daily-loss", "minute_low", "--runs-dir", str(app_env.runs), "--ledger",
        str(app_env.ledger), "--json",
    ])  # fmt: skip
    assert prop.exit_code == 0, prop.output
    again = _json(prop)
    assert again["run_id"] == doc["run_id"] and again["prop"][0]["soft_daily_loss"] == "minute_low"
    assert again["prop"][0]["seed"] == 2 and again["net_cents"] == doc["net_cents"]


def test_a_refusal_prints_the_refusal_object_and_exits_2(app_env: AppEnv) -> None:
    res = runner.invoke(
        app, ["run", str(app_env.strategy), "--kind", "synthetic", "--param", "nope=1", "--json"]
    )
    assert res.exit_code == 2
    doc = _json(res)
    assert doc["status"] == "refused" and doc["error"]["code"] == "RunRefused"
    assert "refused: RunRefused:" in res.stderr
    where = ["--runs-dir", str(app_env.runs), "--ledger", str(app_env.ledger), "--json"]
    prop = runner.invoke(app, ["prop", "synthetic-x", "--firm", FIRM, "--plan", "p", "--size", "50k", *where])
    assert prop.exit_code == 2 and _json(prop)["error"]["code"] == "NoRunDirectory"


def test_pfb_check_and_pfb_new_on_the_template(tmp_path: Path) -> None:
    res = runner.invoke(app, ["check", str(TEMPLATE)])
    assert res.exit_code == 0, res.output
    assert "passes the lint and the truncation check" in res.stdout and "no signal" not in res.stdout
    with service.using(strategies_dir=tmp_path):
        made = runner.invoke(app, ["new", "orb_one", "--json"])
        assert made.exit_code == 0, made.output
        assert json.loads(made.stdout) == {"schema": "pfb.new/1", "path": "strategies/orb_one.py",
                                           "hypothesis_id": "h_orb_one"}  # fmt: skip
        again = runner.invoke(app, ["new", "orb_one"])
        assert again.exit_code == 2 and "refused: Exists:" in again.stderr
    checked = runner.invoke(app, ["check", str(tmp_path / "orb_one.py")])
    assert checked.exit_code == 0, checked.output
