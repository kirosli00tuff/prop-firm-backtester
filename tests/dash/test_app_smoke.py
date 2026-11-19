"""Every page of the Streamlit app renders without an exception (streamlit.testing.v1.AppTest) on a
synthetic run with a suite, its neighbourhood and its prop outputs, in a temporary ledger and runs
directory."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import streamlit as st
from conftest import REMOVED, DashEnv  # pyright: ignore[reportMissingImports]
from streamlit.testing.v1 import AppTest

import pfb.dash

APP = Path(pfb.dash.__file__).resolve().parent / "app.py"
TIMEOUT = 60
DEFAULT_PAGE = "pages/backtest.py"
NAV_TITLES = ["Backtest", "Runs"]
HIDDEN_TITLES = ["Run detail", "Prop Monte Carlo"]
STEP_HEADINGS = ["1 Strategy", "2 Check", "3 Data", "4 Run", "5 Results"]
RUN_COLUMNS = ["run_id", "date", "strategy", "kind", "root", "net", "trades", "verdict"]
RESULT_KPIS = ["Net", "Trades", "Win rate", "Average trade", "Sharpe", "Deflated Sharpe", "Max drawdown"]


@pytest.fixture
def app_env(dash_env: DashEnv, monkeypatch: pytest.MonkeyPatch) -> DashEnv:
    config = dash_env.config
    monkeypatch.setenv("PFB_DASH_REPO", str(config.repo_root))
    monkeypatch.setenv("PFB_DASH_LEDGER", str(config.ledger_path))
    monkeypatch.setenv("PFB_DASH_RUNS", str(config.runs_dir))
    monkeypatch.setenv("PFB_DASH_PE", str(config.pe_root))
    return dash_env


def _page(page: str, run_id: str | None = None) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    at.run()
    if page != DEFAULT_PAGE:
        at.switch_page(page)
    if run_id is not None:
        at.query_params["run"] = run_id
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [h.value for h in at.subheader]
    parts += [m.label + " " + str(m.value) for m in at.metric]
    parts += (
        [str(e.value) for e in at.error]
        + [str(w.value) for w in at.warning]
        + [str(i.value) for i in at.info]
    )
    return "\n".join(parts)


def test_the_header_shows_n_the_dsr_bar_and_the_open_runs_on_every_page(app_env: DashEnv) -> None:
    for page in (DEFAULT_PAGE, "pages/runs.py", "pages/detail.py", "pages/prop.py"):
        at = _page(page, app_env.centre)
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["N (global trials)"] == "302"
        assert metrics["Open runs"] == "1"
        assert "DSR bar" in metrics and "SR0 daily" in metrics


def test_the_run_list_shows_every_run(app_env: DashEnv) -> None:
    # Act
    at = _page("pages/runs.py")
    # Assert
    frame = at.dataframe[0].value
    assert set(frame["run_id"]) == {app_env.centre, *app_env.neighbours, app_env.open_run}
    assert list(frame.columns) == RUN_COLUMNS


def test_the_run_list_opens_no_run_until_a_row_is_clicked(app_env: DashEnv) -> None:
    # Act
    at = _page("pages/runs.py")
    # Assert
    assert "No run open: click a row of the table." in _text(at)
    assert not at.get("page_link")


def test_clicking_a_run_row_opens_its_results_and_the_links(app_env: DashEnv) -> None:
    # Arrange
    at = _page("pages/runs.py")
    ids = list(at.dataframe[0].value["run_id"])
    row = ids.index(app_env.centre)
    # Act: a single-row selection, as a click on the table sets it
    at.session_state["runs_table"] = {"selection": {"rows": [row], "columns": []}}
    at.run()
    # Assert
    assert not at.exception, [e.value for e in at.exception]
    assert f"Results: {app_env.centre}" in [h.value for h in at.subheader]
    labels = [m.label for m in at.metric]
    assert all(label in labels for label in RESULT_KPIS)
    links = at.get("page_link")
    assert [str(link.proto.label) for link in links] == ["Run detail", "Prop Monte Carlo"]
    assert all(link.proto.query_string == f"run={app_env.centre}" for link in links)
    assert at.query_params["run"] == app_env.centre


def test_the_url_run_parameter_opens_that_runs_results(app_env: DashEnv) -> None:
    # Act
    at = _page("pages/runs.py", app_env.neighbours[0])
    # Assert
    assert f"Results: {app_env.neighbours[0]}" in [h.value for h in at.subheader]
    assert "Export JSON" in [b.proto.label for b in at.get("download_button")]


def test_the_nav_lists_exactly_backtest_and_runs_with_backtest_as_default(
    app_env: DashEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: capture the pages the app hands to st.navigation
    seen: list[st.Page] = []
    real = st.navigation

    def spy(pages, **kwargs):  # noqa: ANN001, ANN003, ANN202
        seen.extend(pages)
        return real(pages, **kwargs)

    monkeypatch.setattr(st, "navigation", spy)
    # Act
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    at.run()
    # Assert
    assert not at.exception
    assert [p.title for p in seen if p.visibility == "visible"] == NAV_TITLES
    assert [p.title for p in seen if p.visibility == "hidden"] == HIDDEN_TITLES
    assert [p.title for p in seen if p._default] == ["Backtest"]  # pyright: ignore[reportPrivateUsage]
    assert [h.value for h in at.title] == ["Backtest"]


def test_the_backtest_page_opens_on_step_1_and_locks_the_later_steps(app_env: DashEnv) -> None:
    # Act
    at = _page(DEFAULT_PAGE)
    # Assert
    headings = [h.value for h in at.subheader]
    assert headings[0] == STEP_HEADINGS[0]
    assert headings[-3:] == STEP_HEADINGS[2:]
    assert "No strategy yet." in [c.value for c in at.caption]
    assert [c.value for c in at.caption].count("locked until the check in step 2 passes") == 3


def test_the_detail_page_renders_the_suite_and_the_neighbourhood(app_env: DashEnv) -> None:
    # Act
    at = _page("pages/detail.py", app_env.centre)
    # Assert
    text = _text(at)
    for heading in (
        "Cost stress",
        "Random-entry baseline",
        "Walk-forward",
        "Power",
        "Sizing",
        "Breakdowns",
        "Neighbourhood",
        "Equity and drawdown",
    ):
        assert heading in text, heading
    assert "No artifact is missing" in text


def test_the_detail_page_names_the_reported_baseline_mode(app_env: DashEnv) -> None:
    # Act
    at = _page("pages/detail.py", app_env.centre)
    # Assert: the baseline's four figures come from one mode, named on the page (TRIALS.md section 7)
    assert re.search(r"Reported mode \((tod|uniform|cost_matched)\): net percentile", _text(at))


def test_the_detail_page_names_the_missing_artifacts(app_env: DashEnv) -> None:
    # Act
    at = _page("pages/detail.py", app_env.neighbours[1])
    # Assert
    warnings = "\n".join(str(w.value) for w in at.warning)
    assert all(name in warnings for name in REMOVED)


def test_the_detail_page_of_an_open_run_says_it_has_no_result(app_env: DashEnv) -> None:
    at = _page("pages/detail.py", app_env.open_run)
    assert "no result record" in _text(at)


def test_the_prop_page_shows_pass_probability_and_the_matrix(app_env: DashEnv) -> None:
    # Act
    at = _page("pages/prop.py", app_env.centre)
    # Assert
    text = _text(at)
    assert "Firm-by-plan matrix" in text
    frames = [d.value for d in at.dataframe]
    assert any("p_pass" in f.columns for f in frames)


def test_the_prop_page_shows_a_linked_run_without_prop_output_and_says_so(app_env: DashEnv) -> None:
    # Arrange: a neighbour run has no prop/ directory
    linked = app_env.neighbours[0]
    # Act
    at = _page("pages/prop.py", linked)
    # Assert
    assert at.selectbox[0].value == linked
    assert "has no prop output" in _text(at)
    assert not at.dataframe


def test_a_broken_ledger_is_shown_as_an_error_not_a_crash(
    broken_ledger, monkeypatch: pytest.MonkeyPatch, dash_env: DashEnv
) -> None:
    # Arrange
    monkeypatch.setenv("PFB_DASH_REPO", str(broken_ledger.repo_root))
    monkeypatch.setenv("PFB_DASH_LEDGER", str(broken_ledger.ledger_path))
    monkeypatch.setenv("PFB_DASH_RUNS", str(broken_ledger.runs_dir))
    monkeypatch.setenv("PFB_DASH_PE", str(broken_ledger.pe_root))
    # Act
    at = _page(DEFAULT_PAGE)
    # Assert
    assert at.error and "ledger" in str(at.error[0].value).lower()
