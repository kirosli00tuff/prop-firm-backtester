"""Prop Monte Carlo page: pass probability, days to pass and EV per purchase per plan run on a run
(runs/<run_id>/prop/*.json), and the firm-by-plan matrix (prop_matrix.json) when present."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import streamlit as st

from pfb.dash.data import (
    MATRIX,
    PROP_DIR,
    DashConfig,
    LedgerView,
    matrix_pivot,
    matrix_rows,
    prop_rows,
    run_artifacts,
)
from pfb.dash.views import RUN_PARAM, page_setup, select_run, table

MATRIX_META = ("n_paths", "seed", "generator", "mean_block", "stress_ticks", "n_plans", "n_ok", "n_skipped")


def runs_with_prop(config: DashConfig, view: LedgerView) -> list[str]:
    out = []
    for row in view.runs:
        directory = config.runs_dir / row.run_id
        if (directory / PROP_DIR).is_dir() or (directory / MATRIX).is_file():
            out.append(row.run_id)
    return out


def prop_choices(config: DashConfig, view: LedgerView, linked: str | None) -> list[str]:
    """The runs with prop output, plus the run the URL links to (?run=<id>) when the ledger has it, so the
    page shows the linked run and says so when that run has no prop output."""
    choices = runs_with_prop(config, view)
    if linked is not None and linked not in choices and any(r.run_id == linked for r in view.runs):
        choices.insert(0, linked)
    return choices


def _plans(docs: tuple[Mapping[str, Any], ...]) -> None:
    st.subheader("Plans run on this run")
    if not docs:
        st.caption("This run has no prop output (no prop/ directory).")
        return
    table(
        prop_rows(docs),
        "p_pass and EV per purchase with their Monte Carlo SEs; money in cents; days are "
        "trading days. Every SE is given the day pool; it says nothing about how well the history "
        "represents the strategy.",
    )
    for doc in docs:
        warnings = doc.get("warnings") or ()
        if warnings:
            with st.expander(f"Warnings: {doc.get('firm')}/{doc.get('plan')} ({doc.get('label')})"):
                for w in warnings:
                    st.markdown(f"- {w}")


def _matrix(matrix: Mapping[str, Any] | None) -> None:
    st.subheader("Firm-by-plan matrix")
    if matrix is None:
        st.caption("No prop_matrix.json for this run.")
        return
    st.caption(", ".join(f"{k} {matrix.get(k)}" for k in MATRIX_META if k in matrix))
    rows = matrix_rows(matrix)
    table(rows)
    st.markdown("**Pass probability (firm by plan)**")
    table(matrix_pivot(rows, "p_pass"))
    st.markdown("**EV per purchase, cents (firm by plan)**")
    table(matrix_pivot(rows, "ev_per_purchase_cents"))
    st.caption(str(matrix.get("se_scope") or ""))


def render() -> None:
    setup = page_setup("Prop Monte Carlo")
    if setup is None:
        return
    config, view = setup
    linked = st.query_params.get(RUN_PARAM)
    run_id = select_run(prop_choices(config, view, linked), "Run (with prop output)", key="prop_run")
    if run_id is None:
        return
    art = run_artifacts(config, run_id, view)
    for error in art.errors:
        st.error(f"Unreadable artifact: {error}")
    _plans(art.prop)
    _matrix(art.prop_matrix)
