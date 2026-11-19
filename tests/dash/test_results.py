"""The results view (pfb.dash.view_results) and the readers behind it and the Runs page
(pfb.dash.data). The conftest synthetic neighbourhood is copied into a temporary runs directory when a test
writes a result.json; nothing writes the shared fixture."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import streamlit as st
from conftest import DashEnv  # pyright: ignore[reportMissingImports]
from streamlit.testing.v1 import AppTest

from pfb.dash.data import (
    NOT_RECORDED,
    DashConfig,
    built_contract,
    ledger_view,
    main_end,
    plan_result_row,
    result_doc,
    run_artifacts,
    run_list,
    run_verdict,
)
from pfb.dash.view_results import KPI_SPEC, OFF, cost_rows, kpis
from pfb.dash.view_runs import chosen_run

TIMEOUT = 60
KPI_LABELS = [spec[0] for spec in KPI_SPEC]
SECTION_HEADS = ["KPIs", "Equity and drawdown", "Prop Monte Carlo per plan", "Costs", "Warnings", "Export"]


def _get_run(env: DashEnv, config: DashConfig, run_id: str) -> dict[str, Any]:
    """The contract pfb.app.get_run returns for run_id against the fixture's ledger and the given runs dir."""
    from pfb import app

    deps = SimpleNamespace(registry_config=lambda: env.env.config)
    with app.using(deps=deps, runs_dir=config.runs_dir, ledger=config.ledger_path):
        return app.get_run(run_id)


@pytest.fixture
def saved(dash_env: DashEnv, tmp_path: Path) -> DashConfig:
    """A copy of the runs directory where the centre has a result.json (pfb.app.get_run's contract, saved
    as pfb.app saves it)."""
    runs = tmp_path / "runs"
    shutil.copytree(dash_env.config.runs_dir, runs)
    config = replace(dash_env.config, runs_dir=runs)
    contract = _get_run(dash_env, config, dash_env.centre)
    (runs / dash_env.centre / "result.json").write_text(json.dumps(contract, indent=2) + "\n")
    return config


def _render(config: DashConfig, run_id: str) -> AppTest:
    def script(config, run_id) -> None:  # noqa: ANN001  # the source runs alone: no annotation names
        from pfb.dash import view_results

        view_results.render(config, run_id)

    at = AppTest.from_function(script, default_timeout=TIMEOUT, kwargs={"config": config, "run_id": run_id})
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _captions(at: AppTest) -> list[str]:
    return [c.value for c in at.caption]


# ------------------------------------------------------------------------------- the view ----
def test_the_results_view_shows_every_kpi_and_every_part(saved: DashConfig, dash_env: DashEnv) -> None:
    # Act
    at = _render(saved, dash_env.centre)
    # Assert
    metrics = {m.label: m.value for m in at.metric}
    assert list(metrics) == KPI_LABELS
    assert all(value != "n/a" for value in metrics.values()), metrics
    contract = json.loads((saved.runs_dir / dash_env.centre / "result.json").read_text())
    assert metrics["Trades"] == str(contract["trades"])
    heads = [m.value.removeprefix("#### ") for m in at.markdown if m.value.startswith("#### ")]
    assert heads == SECTION_HEADS
    assert not any(OFF in c for c in _captions(at)), _captions(at)
    assert len(at.get("vega_lite_chart")) == 1
    prop = [d.value for d in at.dataframe if "pass_rate" in d.value.columns]
    assert prop and {"ev_per_purchase", "payout_count", "main_breach_reason"} <= set(prop[0].columns)


def test_the_json_export_hands_out_the_saved_result_json_bytes(
    saved: DashConfig, dash_env: DashEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: capture what the view hands to st.download_button
    handed: list[bytes] = []
    real = st.download_button

    def spy(label: str, data: Any, **kwargs: Any) -> bool:
        handed.append(data)
        return real(label, data, **kwargs)

    monkeypatch.setattr(st, "download_button", spy)
    # Act
    at = _render(saved, dash_env.centre)
    # Assert
    assert handed == [(saved.runs_dir / dash_env.centre / "result.json").read_bytes()]
    assert any("as saved" in c for c in _captions(at))


def test_a_run_without_equity_or_prop_shows_those_parts_as_off(dash_env: DashEnv) -> None:
    # Act: the second neighbour lost its equity frames and never had prop output
    at = _render(dash_env.config, dash_env.neighbours[1])
    # Assert
    captions = _captions(at)
    assert f"Equity and drawdown: {OFF}." in captions
    assert f"Prop Monte Carlo per plan: {OFF}." in captions
    assert [m.label for m in at.metric] == KPI_LABELS
    assert not at.get("vega_lite_chart")


def test_an_open_run_shows_no_kpi_value_and_says_it_never_finished(dash_env: DashEnv) -> None:
    # Act
    at = _render(dash_env.config, dash_env.open_run)
    # Assert
    assert {m.value for m in at.metric} == {"n/a"}
    assert any("no result record" in str(i.value) for i in at.info)
    assert f"Costs: {OFF}." in _captions(at)


def test_an_unknown_run_is_an_error_box(dash_env: DashEnv) -> None:
    at = _render(dash_env.config, "synthetic-unknown")
    assert at.error and "no begin record" in str(at.error[0].value)


# ---------------------------------------------------------------------------- the readers ----
@pytest.mark.parametrize("which", ["centre", "neighbour", "open"])
def test_without_result_json_the_contract_is_the_one_pfb_app_builds(dash_env: DashEnv, which: str) -> None:
    # Arrange
    run_id = {"centre": dash_env.centre, "neighbour": dash_env.neighbours[0], "open": dash_env.open_run}[
        which
    ]
    art = run_artifacts(dash_env.config, run_id)
    # Act
    doc = result_doc(dash_env.config, art)
    # Assert
    assert doc is not None and not doc.saved
    assert dict(doc.contract) == _get_run(dash_env, dash_env.config, run_id)
    assert json.loads(doc.data) == dict(built_contract(dash_env.config, art))


def test_an_unreadable_result_json_gives_no_document(saved: DashConfig, dash_env: DashEnv) -> None:
    (saved.runs_dir / dash_env.centre / "result.json").write_text("{not json")
    art = run_artifacts(saved, dash_env.centre)
    assert result_doc(saved, art) is None
    assert any("result.json" in e for e in art.errors)


def test_kpis_fall_back_to_the_summary_names_without_a_contract() -> None:
    # Arrange
    summary = {"net_cents": -1234, "n_trades": 7, "sharpe_annual": 1.5, "max_drawdown_cents": 500}
    # Act
    found = {k.label: k.text for k in kpis(None, summary)}
    # Assert
    assert found["Net"] == "-$12.34"
    assert found["Trades"] == "7"
    assert found["Sharpe"] == "1.500"
    assert found["Win rate"] == "n/a"


def test_the_main_breach_reason_is_the_commonest_failing_end() -> None:
    assert main_end({"pass": 10, "breach": 6, "timeout": 4}) == "breach 30%"
    assert main_end({"pass": 3}) == "none"
    assert main_end({}) == NOT_RECORDED
    assert main_end(None) == NOT_RECORDED


def test_the_plan_row_says_when_the_payout_count_is_not_recorded(dash_env: DashEnv) -> None:
    doc = json.loads(dash_env.prop_file.read_text())
    row = plan_result_row(doc)
    assert row["payout_count"] == NOT_RECORDED
    assert row["pass_rate"] == doc["outputs"]["p_pass"]["value"]
    assert plan_result_row({**doc, "payout_count": 3})["payout_count"] == "3"


def test_event_costs_on_and_off_and_no_metrics() -> None:
    metrics = {"commission_cents": 250, "slippage_cents": 125}
    on = {r["cost"]: r["value"] for r in cost_rows(metrics, {"event_windows": "on"}) or []}
    off = {r["cost"]: r["value"] for r in cost_rows(metrics, {"event_windows": "off"}) or []}
    assert on["Fees (commission)"] == "$2.50" and on["Slippage"] == "$1.25"
    assert on["Event costs"].startswith("on")
    assert off["Event costs"].startswith(OFF)
    assert cost_rows(None, {}) is None


def test_the_run_list_names_every_run_with_its_strategy_and_verdict(dash_env: DashEnv) -> None:
    # Act
    rows = {r.run_id: r for r in run_list(dash_env.config, ledger_view(dash_env.config))}
    # Assert
    assert set(rows) == {dash_env.centre, *dash_env.neighbours, dash_env.open_run}
    assert rows[dash_env.open_run].verdict == "open"
    assert rows[dash_env.centre].verdict in {"plateau", "spike", "mixed", "n/a"}
    assert rows[dash_env.neighbours[0]].verdict == "ok"
    assert rows[dash_env.centre].strategy == "dash_random.py"
    assert len(rows[dash_env.centre].date) == len("2026-11-17")


def test_run_verdict_and_the_chosen_run() -> None:
    assert run_verdict("failed", None) == "failed"
    assert run_verdict("ok", {"neighbourhood": {"verdict": "spike"}}) == "spike"
    assert run_verdict("ok", {"neighbourhood": None}) == "ok"
    ids = ["a", "b"]
    assert chosen_run(ids, 1, "a") == "b"
    assert chosen_run(ids, None, "a") == "a"
    assert chosen_run(ids, None, "zzz") is None
    assert chosen_run(ids, None, None) is None
