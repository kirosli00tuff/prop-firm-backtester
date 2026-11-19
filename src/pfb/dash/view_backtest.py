"""Backtest page: the five steps top to bottom, each unlocking the next. A locked step shows its heading and
what it waits for."""

from __future__ import annotations

import streamlit as st

from pfb.dash import view_data, view_results, view_run, view_strategy
from pfb.dash.views import page_setup

STEP_DATA = "3 Data"
STEP_RUN = "4 Run"
STEP_RESULTS = "5 Results"
WAITS_FOR_CHECK = "locked until the check in step 2 passes"
WAITS_FOR_WINDOW = "locked until step 3 shows an open window on disk"
WAITS_FOR_RUN = "opens when a run from step 4 ends"


def _locked(steps: tuple[str, ...], why: str) -> None:
    for step in steps:
        st.subheader(step)
        st.caption(why)


def render() -> None:
    setup = page_setup("Backtest")
    if setup is None:
        return
    config, view = setup
    state = view_strategy.render(headings=True)
    if state is None or not state.passed or state.spec is None:
        _locked((STEP_DATA, STEP_RUN, STEP_RESULTS), WAITS_FOR_CHECK)
        return
    st.subheader(STEP_DATA)
    windows = view_data.render(state.spec)
    if not windows:
        _locked((STEP_RUN, STEP_RESULTS), WAITS_FOR_WINDOW)
        return
    st.subheader(STEP_RUN)
    run_id = view_run.render(config, state, windows)
    if run_id is None:
        _locked((STEP_RESULTS,), WAITS_FOR_RUN)
        return
    st.subheader(STEP_RESULTS)
    view_results.render(config, run_id, view)
