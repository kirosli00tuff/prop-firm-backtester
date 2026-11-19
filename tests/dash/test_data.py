"""pfb.dash.data: the read-only views the pages show, checked against the ledger and the artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
from conftest import PROP_MC, REMOVED, DashEnv  # pyright: ignore[reportMissingImports]

from pfb.dash import data
from pfb.metrics.stats import bracket
from pfb.registry.api import read_state


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_ledger_view_reports_n_open_runs_and_every_run(dash_env: DashEnv) -> None:
    # Arrange
    state = read_state(dash_env.env.config)
    # Act
    view = data.ledger_view(dash_env.config)
    # Assert
    assert view.n == state.n == 302
    assert view.open_runs == (dash_env.open_run,)
    ids = [row.run_id for row in view.runs]
    assert set(ids) == {dash_env.centre, *dash_env.neighbours, dash_env.open_run}
    times = [row.time_utc for row in view.runs]
    assert times == sorted(times, reverse=True)  # newest first
    assert view.pe_in_sync is True


def test_run_rows_carry_the_result_records_figures(dash_env: DashEnv) -> None:
    # Arrange
    state = read_state(dash_env.env.config)
    # Act
    rows = {row.run_id: row for row in data.ledger_view(dash_env.config).runs}
    # Assert
    centre = rows[dash_env.centre]
    metrics = state.results[dash_env.centre]["metrics"]
    begin = state.begun[dash_env.centre]
    assert (centre.kind, centre.status, centre.hypothesis_id) == ("synthetic", "ok", "test:dash_random")
    assert centre.net_cents == metrics["net_cents"]
    assert centre.sharpe_annual == metrics["sharpe_annual"]
    assert centre.n_at_run == begin["n_after"] == 302
    opened = rows[dash_env.open_run]
    assert (opened.status, opened.net_cents, opened.sharpe_annual) == ("open", None, None)


def test_dsr_bar_is_the_bracket_and_the_floor_variance_sr0_of_a_ten_year_run() -> None:
    # Act
    bar = data.dsr_bar(1000)
    # Assert: T = 2520 days, floor variance 1 / (T - 1)
    assert bar.days == 2520
    assert bar.bracket == pytest.approx(bracket(1000))
    assert bar.sr0_daily == pytest.approx(bracket(1000) / math.sqrt(2519))
    assert bar.sr0_annual == pytest.approx(bar.sr0_daily * math.sqrt(252))


def test_artifacts_of_a_complete_run_are_all_present(dash_env: DashEnv) -> None:
    # Act
    art = data.run_artifacts(dash_env.config, dash_env.centre)
    # Assert
    run_dir = dash_env.config.runs_dir / dash_env.centre
    assert art.missing == ()
    assert art.summary == _json(run_dir / "summary.json")
    assert art.suite == _json(run_dir / "suite.json")
    assert art.neighbourhood is not None and art.neighbourhood["verdict"]["verdict"] in {
        "plateau",
        "mixed",
        "spike",
        "n/a",
    }
    assert len(art.prop) == 1 and art.prop_matrix is not None
    assert art.errors == ()


def test_missing_artifacts_are_named(dash_env: DashEnv) -> None:
    # Act
    art = data.run_artifacts(dash_env.config, dash_env.neighbours[1])
    # Assert
    assert set(art.missing) == set(REMOVED)
    assert art.summary is not None


def test_an_open_run_expects_nothing_and_says_so(dash_env: DashEnv) -> None:
    # Act
    art = data.run_artifacts(dash_env.config, dash_env.open_run)
    # Assert
    assert art.expected == () and art.missing == ()
    assert art.summary is None
    assert any("no result record" in note for note in art.notes)


def test_an_unknown_run_id_is_refused(dash_env: DashEnv) -> None:
    with pytest.raises(KeyError):
        data.run_artifacts(dash_env.config, "synthetic-nope")


def test_an_unreadable_json_artifact_is_reported_not_raised(dash_env: DashEnv, tmp_path: Path) -> None:
    # Arrange: a copy of the runs directory whose summary.json is cut short
    runs = tmp_path / "runs"
    src = dash_env.config.runs_dir / dash_env.centre
    dst = runs / dash_env.centre
    dst.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_file():
            (dst / f.name).write_bytes(f.read_bytes())
    (dst / "summary.json").write_text("{")
    config = replace(dash_env.config, runs_dir=runs)
    # Act
    art = data.run_artifacts(config, dash_env.centre)
    # Assert
    assert art.summary is None
    assert any("summary.json" in e for e in art.errors)


def test_equity_curve_from_daily_ends_at_the_run_net_and_draws_down_from_the_running_peak(
    dash_env: DashEnv,
) -> None:
    # Arrange
    run_dir = dash_env.config.runs_dir / dash_env.centre
    daily = pl.read_parquet(run_dir / "daily.parquet")
    summary = _json(run_dir / "summary.json")
    # Act
    curve = data.equity_curve(run_dir)
    # Assert
    assert curve is not None and curve.source == "daily.parquet"
    frame = curve.frame
    assert frame["equity_cents"].to_list() == daily["eq_close"].to_list()
    assert frame["equity_cents"][-1] - daily["eq_open"][0] == summary["net_cents"]
    peak = frame["equity_cents"].cum_max()
    assert frame["drawdown_cents"].to_list() == (frame["equity_cents"] - peak).to_list()


def test_equity_curve_falls_back_to_equity_minute(dash_env: DashEnv, tmp_path: Path) -> None:
    # Arrange: only the minute frame
    run_dir = dash_env.config.runs_dir / dash_env.centre
    minute = pl.read_parquet(run_dir / "equity_minute.parquet")
    (tmp_path / "equity_minute.parquet").write_bytes((run_dir / "equity_minute.parquet").read_bytes())
    # Act
    curve = data.equity_curve(tmp_path)
    # Assert
    assert curve is not None and curve.source == "equity_minute.parquet"
    last = minute.group_by("trade_date", maintain_order=True).agg(pl.col("eq_close").last())
    assert curve.frame["equity_cents"].to_list() == last["eq_close"].to_list()


def test_equity_curve_is_none_without_frames(dash_env: DashEnv) -> None:
    assert data.equity_curve(dash_env.config.runs_dir / dash_env.neighbours[1]) is None


def test_prop_rows_copy_the_prop_outputs(dash_env: DashEnv) -> None:
    # Arrange
    doc = _json(dash_env.prop_file)
    # Act
    (row,) = data.prop_rows([doc])
    # Assert
    out = doc["outputs"]
    assert (row["firm"], row["plan"], row["paths"]) == ("topstep", "combine_xfa_standard", PROP_MC)
    assert row["p_pass"] == out["p_pass"]["value"] and row["p_pass_se"] == out["p_pass"]["se"]
    assert row["ev_per_purchase_cents"] == out["ev_per_purchase"]["value"]
    assert row["days_to_pass_median"] == out["days_to_pass_median"]["value"]


def test_matrix_rows_cover_every_plan_and_pivot_by_firm_and_plan(dash_env: DashEnv) -> None:
    # Arrange
    matrix = _json(dash_env.config.runs_dir / dash_env.centre / "prop_matrix.json")
    # Act
    rows = data.matrix_rows(matrix)
    pivot = data.matrix_pivot(rows, "p_pass")
    # Assert
    assert len(rows) == len(matrix["rows"])
    firms = {r["firm"] for r in matrix["rows"]}
    assert set(pivot["firm"].to_list()) == firms
    ok = [r for r in matrix["rows"] if r["status"] == "ok"]
    first = ok[0]
    cell = pivot.filter(pl.col("firm") == first["firm"])[first["plan"]][0]
    assert cell == first["record"]["outputs"]["p_pass"]["value"]


def test_ledger_errors_surface_as_ledger_unreadable(broken_ledger: data.DashConfig) -> None:
    with pytest.raises(data.LedgerUnreadable):
        data.ledger_view(broken_ledger)


def test_config_from_env_reads_the_four_variables(tmp_path: Path) -> None:
    # Arrange
    env = {
        "PFB_DASH_REPO": str(tmp_path / "r"),
        "PFB_DASH_LEDGER": str(tmp_path / "l.jsonl"),
        "PFB_DASH_RUNS": str(tmp_path / "runs"),
        "PFB_DASH_PE": str(tmp_path / "pe"),
    }
    # Act
    config = data.DashConfig.from_env(env)
    # Assert
    assert config == data.DashConfig(tmp_path / "r", tmp_path / "l.jsonl", tmp_path / "runs", tmp_path / "pe")


def test_config_from_an_empty_env_is_this_repository() -> None:
    config = data.DashConfig.from_env({})
    assert config.ledger_path == config.repo_root / "ledger" / "registry.jsonl"
    assert config.runs_dir == config.repo_root / "runs"


def test_cents_format() -> None:
    assert data.fmt_cents(123456) == "$1,234.56"
    assert data.fmt_cents(-5) == "-$0.05"
    assert data.fmt_cents(None) == "n/a"
    assert data.fmt_cents(1250.0) == "$12.50"
    assert data.fmt_cents(float("nan")) == "n/a"
