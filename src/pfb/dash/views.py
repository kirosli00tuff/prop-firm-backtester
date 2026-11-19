"""Shared Streamlit pieces: the page setup with the header every page shows (N, the DSR bar, open runs), the
run selector and small table and section helpers. Read-only."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl
import streamlit as st

from pfb.dash.data import DashConfig, LedgerUnreadable, LedgerView, fmt_num, ledger_view

RUN_PARAM = "run"
OFF = "off or not applicable for this run"

GAIN = "#3FB950"
LOSS = "#F85149"
MUTED = "#8A9099"
CONTENT_MAX_WIDTH_PX = 1280
SPACING_PX = 8
METRIC_VALUE_REM = 1.3  # seven KPI tiles in a 1280 px column: full "-$123,456.78" without an ellipsis
STYLE = f"""
<style>
.block-container {{ max-width: {CONTENT_MAX_WIDTH_PX}px; padding-top: {3 * SPACING_PX}px; }}
[data-testid="stVerticalBlock"] {{ gap: {SPACING_PX}px; }}
[data-testid="stMetricValue"], [data-testid="stDataFrame"], [data-testid="stTable"] {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}}
[data-testid="stMetricValue"] {{ font-size: {METRIC_VALUE_REM}rem; }}
[data-testid="stMetricValue"] > div {{
    overflow: visible; text-overflow: clip; white-space: normal; overflow-wrap: anywhere;
}}
</style>
"""


def dash_config() -> DashConfig:
    return DashConfig.from_env(os.environ)


def apply_style() -> None:
    """The shared look: content width, 8 px spacing and monospace numbers (no gradients, no icons)."""
    st.markdown(STYLE, unsafe_allow_html=True)


def header(view: LedgerView) -> None:
    bar = view.bar
    cols = st.columns(4)
    cols[0].metric(
        "N (global trials)", str(view.n), help="Every configuration evaluated on real data counts."
    )
    cols[1].metric(
        "DSR bar",
        fmt_num(bar.bracket),
        help="bracket(N): expected maximum Sharpe of N null trials, "
        "in units of the Sharpe's standard deviation.",
    )
    cols[2].metric(
        "SR0 daily",
        fmt_num(bar.sr0_daily, 4),
        help=f"bracket(N) x sqrt(1 / (T - 1)), T = {bar.days} "
        f"days (a {bar.days // 252}-year run at the floor variance); annualized "
        f"{fmt_num(bar.sr0_annual)}.",
    )
    cols[3].metric(
        "Open runs", str(len(view.open_runs)), help="Begun without a result record: open or crashed."
    )
    sync = "unknown" if view.pe_in_sync is None else ("yes" if view.pe_in_sync else "NO")
    st.caption(
        f"Ledger: {view.records} records, N {view.n} (upstream {view.n_pe}, local {view.n - view.n_pe}); "
        f"upstream in sync: {sync} ({view.pe_note}). "
        f"SR0 annualized at {bar.days} days: {fmt_num(bar.sr0_annual)}."
    )
    if view.open_runs:
        st.caption("Open or crashed runs: " + ", ".join(view.open_runs))


def page_setup(title: str) -> tuple[DashConfig, LedgerView] | None:
    """Title, the ledger and the header; None (after an error box) when the ledger cannot be read."""
    apply_style()
    st.title(title)
    config = dash_config()
    try:
        view = ledger_view(config)
    except LedgerUnreadable as exc:
        st.error(f"The ledger cannot be read, so nothing is shown: {exc}")
        return None
    header(view)
    st.divider()
    return config, view


def select_run(run_ids: Sequence[str], label: str, key: str) -> str | None:
    """A run chosen from run_ids; the URL's ?run=<id> picks the first choice when it names one."""
    if not run_ids:
        st.info("No run to show.")
        return None
    choices = list(run_ids)
    wanted = st.query_params.get(RUN_PARAM)
    index = choices.index(wanted) if wanted is not None and wanted in choices else 0
    return st.selectbox(label, choices, index=index, key=key)


def table(rows: Sequence[Mapping[str, Any]] | pl.DataFrame, caption: str | None = None) -> None:
    frame = rows if isinstance(rows, pl.DataFrame) else frame_of(rows)
    if frame.height == 0:
        st.caption("(no rows)")
        return
    st.dataframe(frame, hide_index=True)
    if caption:
        st.caption(caption)


def _scalar(value: Any) -> Any:
    if isinstance(value, Mapping | list | tuple):
        return str(value)
    return value


def frame_of(rows: Sequence[Mapping[str, Any]]) -> pl.DataFrame:
    """A table from flat dicts; nested values become text so every column is a scalar column."""
    clean = [{str(k): _scalar(v) for k, v in row.items()} for row in rows]
    return pl.DataFrame(clean, infer_schema_length=None) if clean else pl.DataFrame()


def part(title: str, value: Any) -> bool:
    """A section heading; False (and an "off" caption) when the part is absent or null."""
    st.subheader(title)
    if value is None:
        st.caption(f"{title}: {OFF}.")
        return False
    return True
