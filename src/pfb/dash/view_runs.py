"""Runs page: one table of every run the ledger names (date, strategy, kind, root, net, trades, verdict).
Clicking a row (or opening the page with ?run=<id>) opens that run's Results view below the table, with links
to the hidden Run detail and Prop Monte Carlo pages."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import streamlit as st

from pfb.dash import view_results
from pfb.dash.data import DashConfig, LedgerView, RunListRow, fmt_cents, run_list
from pfb.dash.views import RUN_PARAM, frame_of, page_setup

if TYPE_CHECKING:
    from streamlit.elements.arrow import DataframeStateInput

DETAIL_PAGE = "pages/detail.py"
PROP_PAGE = "pages/prop.py"
TABLE_KEY = "runs_table"
COLUMNS = ("run_id", "date", "strategy", "kind", "root", "net", "trades", "verdict")
# an upload is named <sha256>.py; its 64 hex characters pushed the other columns
# off screen.
UPLOAD_NAME = re.compile(r"[0-9a-f]{64}\.py")
SHA8 = 8


def strategy_label(name: str) -> str:
    """The first 8 hex characters for a strategy file named by its sha256, else the file name."""
    return name[:SHA8] if UPLOAD_NAME.fullmatch(name) else name


def run_table_row(row: RunListRow) -> dict[str, Any]:
    return {
        "run_id": row.run_id,
        "date": row.date,
        "strategy": strategy_label(row.strategy),
        "kind": row.kind,
        "root": row.root,
        "net": fmt_cents(row.net_cents),
        "trades": row.trades,
        "verdict": row.verdict,
    }


def run_links(run_id: str) -> None:
    """Two links (run detail, prop Monte Carlo) that carry ?run=<id>."""
    cols = st.columns(2)
    cols[0].page_link(DETAIL_PAGE, label="Run detail", query_params={RUN_PARAM: run_id})
    cols[1].page_link(PROP_PAGE, label="Prop Monte Carlo", query_params={RUN_PARAM: run_id})


def _selected_index(event: object) -> int | None:
    selection = event.get("selection") if isinstance(event, Mapping) else None
    rows = selection.get("rows") if isinstance(selection, Mapping) else None
    return int(rows[0]) if rows else None


def chosen_run(run_ids: Sequence[str], selected: int | None, wanted: str | None) -> str | None:
    """The clicked row's run, else the URL's ?run=<id> when the table lists it, else none."""
    if selected is not None and 0 <= selected < len(run_ids):
        return run_ids[selected]
    return wanted if wanted in run_ids else None


def _table(rows: Sequence[Mapping[str, Any]], wanted: str | None) -> int | None:
    """The run table with single-row selection; the URL's run is pre-selected; the clicked row's index."""
    if not rows:
        st.caption("(no rows)")
        return None
    ids = [str(r["run_id"]) for r in rows]
    default: DataframeStateInput | None = (
        {"selection": {"rows": [ids.index(wanted)]}} if wanted is not None and wanted in ids else None
    )
    event = st.dataframe(
        frame_of(rows).select(COLUMNS),
        hide_index=True,
        key=TABLE_KEY,
        on_select="rerun",
        selection_mode="single-row",
        selection_default=default,
    )
    return _selected_index(event)


def _filtered(config: DashConfig, view: LedgerView) -> list[dict[str, Any]]:
    runs = run_list(config, view)
    kinds = sorted({r.kind for r in runs})
    verdicts = sorted({r.verdict for r in runs})
    cols = st.columns(2)
    chosen_kinds = cols[0].multiselect("Kind", kinds, default=kinds, key="runs_kind")
    chosen_verdicts = cols[1].multiselect("Verdict", verdicts, default=verdicts, key="runs_verdict")
    rows = [run_table_row(r) for r in runs if r.kind in chosen_kinds and r.verdict in chosen_verdicts]
    st.caption(
        f"{len(rows)} of {len(runs)} runs, newest first; click a row to open its results. net and trades "
        "come from the result record. verdict: open (begun, never finished) or failed, else the "
        "neighbourhood verdict of a neighbourhood centre (plateau, spike, mixed, n/a), else ok."
    )
    return rows


def render() -> None:
    setup = page_setup("Runs")
    if setup is None:
        return
    config, view = setup
    rows = _filtered(config, view)
    wanted = st.query_params.get(RUN_PARAM)
    selected = _table(rows, wanted)
    run_id = chosen_run([str(r["run_id"]) for r in rows], selected, wanted)
    if run_id is None:
        st.caption("No run open: click a row of the table.")
        return
    if wanted != run_id:
        st.query_params[RUN_PARAM] = run_id
    st.subheader(f"Results: {run_id}")
    run_links(run_id)
    view_results.render(config, run_id, view)
