"""Run detail page: the run's records and metrics, equity and drawdown, the suite, the breakdowns, its
neighbourhood, its warnings and what is missing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import streamlit as st

from pfb.dash.data import LedgerView, RunArtifacts, equity_curve, fmt_cents, fmt_num, prop_rows, run_artifacts
from pfb.dash.view_suite import breakdowns, render_suite
from pfb.dash.views import page_setup, select_run, table

TABS = ("Equity and drawdown", "Suite", "Breakdowns", "Neighbourhood", "Prop", "Warnings and files")
FACT_KEYS = (
    "kind",
    "status",
    "hypothesis_id",
    "root",
    "first",
    "last",
    "session",
    "params",
    "seed",
    "cost_model",
    "fees",
    "engine_version",
    "n_before",
    "n_after",
    "time_utc",
    "git_commit",
)


def _facts(art: RunArtifacts) -> None:
    rows = [{"field": k, "value": str(art.begin.get(k))} for k in FACT_KEYS if k in art.begin]
    with st.expander("Begin record"):
        table(rows)


def _missing(art: RunArtifacts) -> None:
    for note in art.notes:
        st.info(note)
    if art.missing:
        st.warning("Missing artifacts (named in the result record, not on disk): " + ", ".join(art.missing))
    elif art.expected:
        st.caption(f"No artifact is missing ({len(art.expected)} named in the result record).")
    else:
        st.caption("The ledger names no artifact for this run.")
    for error in art.errors:
        st.error(f"Unreadable artifact: {error}")


def _tiles(metrics: Mapping[str, Any]) -> None:
    first = st.columns(6)
    first[0].metric("Net", fmt_cents(metrics.get("net_cents")))
    first[1].metric("Trades", str(metrics.get("n_trades")))
    first[2].metric("Win rate", fmt_num(metrics.get("win_rate")))
    first[3].metric("Profit factor", fmt_num(metrics.get("profit_factor")))
    first[4].metric("Expectancy", fmt_cents(metrics.get("expectancy_cents")))
    first[5].metric("Max drawdown", fmt_cents(metrics.get("max_drawdown_cents")))
    second = st.columns(6)
    second[0].metric("Sharpe (annual)", fmt_num(metrics.get("sharpe_annual")))
    second[1].metric("Sortino (annual)", fmt_num(metrics.get("sortino_annual")))
    second[2].metric("PSR", fmt_num(metrics.get("psr")))
    second[3].metric("DSR", fmt_num(metrics.get("dsr")))
    second[4].metric("MinBTL (years)", fmt_num(metrics.get("minbtl_years")))
    second[5].metric("N at the run", str(metrics.get("n_trials")))


def _equity(art: RunArtifacts) -> None:
    st.subheader("Equity and drawdown")
    try:
        curve = equity_curve(art.directory)
    except Exception as exc:  # a corrupt frame is shown, not raised
        st.error(f"Equity frames unreadable: {type(exc).__name__}: {exc}")
        return
    if curve is None:
        st.warning("Equity curve unavailable: neither daily.parquet nor equity_minute.parquet is present.")
        return
    st.line_chart(curve.frame, x="trade_date", y="equity_cents")
    st.area_chart(curve.frame, x="trade_date", y="drawdown_cents")
    st.caption(f"End-of-day equity and drawdown from the running peak, in cents, from {curve.source}.")


def _neighbourhood(art: RunArtifacts, view: LedgerView) -> None:
    st.subheader("Neighbourhood")
    doc = art.neighbourhood
    if doc is None:
        contract = art.result.get("neighbourhood") if art.result else None
        if isinstance(contract, Mapping):
            st.json(dict(contract))
        else:
            st.caption("No neighbourhood.json: this run is not the centre of a neighbourhood run.")
        return
    raw = doc.get("verdict")
    found: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    st.markdown(
        f"Verdict: **{found.get('verdict')}**, ratio {fmt_num(found.get('ratio'))} (mean neighbour "
        f"Sharpe / centre Sharpe), {found.get('n_same_sign')} same sign, {found.get('n_other_sign')} "
        "other sign."
    )
    members = [("centre", doc.get("centre"))] + [("neighbour", n) for n in doc.get("neighbours") or ()]
    rows = []
    for role, member in members:
        member = member if isinstance(member, Mapping) else {}
        result = view.results.get(str(member.get("run_id")))
        metrics = result.get("metrics") if result else None
        metrics = metrics if isinstance(metrics, Mapping) else {}
        rows.append(
            {
                "role": role,
                "run_id": member.get("run_id"),
                "params": member.get("params"),
                "net": fmt_cents(metrics.get("net_cents")),
                "sharpe": metrics.get("sharpe_annual"),
            }
        )
    table(rows, f"N {doc.get('n_before')} before, {doc.get('n_after')} after the set.")


def _prop(art: RunArtifacts) -> None:
    st.subheader("Prop Monte Carlo")
    if not art.prop and art.prop_matrix is None:
        st.caption("No prop output (prop/ and prop_matrix.json absent).")
        return
    table(prop_rows(art.prop), "Details and the firm-by-plan matrix: the Prop Monte Carlo page.")


def _warnings(art: RunArtifacts) -> None:
    st.subheader("Warnings")
    metrics = art.metrics or {}
    found = [*(metrics.get("warnings") or ()), *(metrics.get("metrics_warnings") or ())]
    found += list((art.result_record or {}).get("warnings") or ())
    found += list(art.begin.get("warnings") or ())
    found += list((art.result or {}).get("warnings") or ())
    if found:
        for w in dict.fromkeys(str(x) for x in found):
            st.markdown(f"- {w}")
    else:
        st.caption("No warning.")
    st.subheader("Files")
    st.caption(f"{art.directory}: " + (", ".join(art.present) or "(no directory)"))
    if art.result is not None:
        with st.expander("Result contract (result.json)"):
            st.json(dict(art.result))


def render() -> None:
    setup = page_setup("Run detail")
    if setup is None:
        return
    config, view = setup
    run_id = select_run([r.run_id for r in view.runs], "Run", key="detail_run")
    if run_id is None:
        return
    art = run_artifacts(config, run_id, view)
    _facts(art)
    _missing(art)
    metrics = art.metrics
    if metrics is None:
        return
    _tiles(metrics)
    tabs = st.tabs(list(TABS))
    with tabs[0]:
        _equity(art)
    with tabs[1]:
        render_suite(art.suite, metrics, art.directory)
    with tabs[2]:
        breakdowns((art.suite or {}).get("breakdowns"), art.directory)
    with tabs[3]:
        _neighbourhood(art, view)
    with tabs[4]:
        _prop(art)
    with tabs[5]:
        _warnings(art)
