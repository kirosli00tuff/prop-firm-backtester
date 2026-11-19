"""Results view (Backtest step 5 and the Runs page): one run's KPI row, equity curve with its drawdown band,
per-plan prop table, costs, warnings and a JSON export, all from the run's own files (runs/<id>/result.json,
the equity frames, prop/*.json) read through pfb.dash.data. A part the run does not have is shown as "off or
not applicable". Read-only: the export hands out the saved result.json bytes (or, without the file, the
contract pfb.app.get_run would build); nothing is written."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import altair as alt
import polars as pl
import streamlit as st

from pfb.dash.data import (
    DashConfig,
    LedgerView,
    ResultDoc,
    RunArtifacts,
    equity_curve,
    fmt_cents,
    fmt_num,
    plan_result_row,
    result_doc,
    run_artifacts,
)
from pfb.dash.views import GAIN, LOSS, OFF, table

CENTS_PER_USD = 100
BAND_OPACITY = 0.3
CHART_HEIGHT_PX = 320
EVENT_COSTS_ON = "on"
EXPORT_LABEL = "Export JSON"
EXPORT_MIME = "application/json"
SECTIONS = ("KPIs", "Equity and drawdown", "Prop Monte Carlo per plan", "Costs", "Warnings", "Export")


@dataclass(frozen=True)
class Kpi:
    label: str
    text: str
    help: str


# (label, contract key, summary key, formatter, help)
KPI_SPEC: tuple[tuple[str, str, str, str, str], ...] = (
    ("Net", "net_cents", "net_cents", "cents", "Net of fees and slippage, USD."),
    ("Trades", "trades", "n_trades", "int", "Round trips."),
    ("Win rate", "win_rate", "win_rate", "num", "Share of trades with a positive net."),
    ("Average trade", "expectancy_cents", "expectancy_cents", "cents", "Net per trade, USD."),
    ("Sharpe", "sharpe", "sharpe_annual", "num", "Annualized from daily net (252 days)."),
    ("Deflated Sharpe", "dsr", "dsr", "num", "DSR: the probability the Sharpe beats N null trials' best."),
    (
        "Max drawdown",
        "max_drawdown_cents",
        "max_drawdown_cents",
        "cents",
        "From the running equity peak, USD.",
    ),
)


def _format(kind: str, value: object) -> str:
    if kind == "cents":
        return fmt_cents(value)
    if kind == "int":
        return "n/a" if value is None else str(value)
    return fmt_num(value)


def kpis(contract: Mapping[str, Any] | None, metrics: Mapping[str, Any] | None) -> tuple[Kpi, ...]:
    """The seven KPIs from the result contract, else from the run's summary (its own key names)."""
    out = []
    for label, key, summary_key, kind, help_text in KPI_SPEC:
        value = contract.get(key) if contract is not None else (metrics or {}).get(summary_key)
        out.append(Kpi(label, _format(kind, value), help_text))
    return tuple(out)


def _section(title: str, value: object) -> bool:
    """A section heading below the page's subheaders; False (and the "off" caption) when value is None."""
    st.markdown(f"#### {title}")
    if value is None:
        st.caption(f"{title}: {OFF}.")
        return False
    return True


def _kpi_row(doc: ResultDoc | None, art: RunArtifacts) -> None:
    _section(SECTIONS[0], True)
    row = kpis(doc.contract if doc is not None else None, art.metrics)
    cols = st.columns(len(row))
    for col, kpi in zip(cols, row, strict=True):
        col.metric(kpi.label, kpi.text, help=kpi.help)


def equity_frame(curve: pl.DataFrame) -> pl.DataFrame:
    """trade_date, equity and running peak in USD (display only; the cents stay in the run's files)."""
    return curve.select(
        "trade_date",
        (pl.col("equity_cents") / CENTS_PER_USD).alias("equity_usd"),
        ((pl.col("equity_cents") - pl.col("drawdown_cents")) / CENTS_PER_USD).alias("peak_usd"),
    )


def _equity_chart(frame: pl.DataFrame) -> alt.LayerChart | alt.FacetChart:
    last = frame["equity_usd"][-1] if frame.height else 0.0
    line_colour = GAIN if last is not None and last >= 0 else LOSS
    base = alt.Chart(frame).encode(x=alt.X("trade_date:T", title=None))
    band = base.mark_area(color=LOSS, opacity=BAND_OPACITY).encode(
        y=alt.Y("equity_usd:Q", title="Equity, USD"), y2="peak_usd:Q"
    )
    line = base.mark_line(color=line_colour).encode(y="equity_usd:Q")
    return alt.layer(band, line).properties(height=CHART_HEIGHT_PX)


def _equity(art: RunArtifacts) -> None:
    try:
        curve = equity_curve(art.directory)
    except Exception as exc:  # a corrupt frame is shown, not raised
        _section(SECTIONS[1], True)
        st.error(f"Equity frames unreadable: {type(exc).__name__}: {exc}")
        return
    if not _section(SECTIONS[1], curve):
        return
    assert curve is not None
    st.altair_chart(_equity_chart(equity_frame(curve.frame)), width="stretch")
    st.caption(f"End-of-day equity; the band is the drawdown from the running peak (from {curve.source}).")


def _prop(art: RunArtifacts, doc: ResultDoc | None) -> None:
    rows = [plan_result_row(d) for d in art.prop]
    if not rows:
        contract_rows = doc.contract.get("prop") if doc is not None else None
        rows = [dict(r) for r in contract_rows] if isinstance(contract_rows, list) else []
    if not _section(SECTIONS[2], rows or None):
        return
    table(
        rows,
        "pass_rate and EV per purchase with their Monte Carlo SEs (given the day pool). "
        "main_breach_reason: the commonest evaluation end other than a pass (breach = drawdown or hard "
        "daily loss). payout_count: pfb prop does not record one per plan yet.",
    )


def cost_rows(metrics: Mapping[str, Any] | None, begin: Mapping[str, Any]) -> list[dict[str, str]] | None:
    """Fees, slippage and event costs of the run; None when the run has no metrics."""
    if metrics is None:
        return None
    events = str(begin.get("event_windows") or "off")
    event_text = (
        "on: release-minute slippage is inside the slippage line"
        if events == EVENT_COSTS_ON
        else f"{OFF} (event windows off: release-minute slippage not charged)"
    )
    return [
        {"cost": "Fees (commission)", "value": fmt_cents(metrics.get("commission_cents"))},
        {"cost": "Slippage", "value": fmt_cents(metrics.get("slippage_cents"))},
        {"cost": "Event costs", "value": event_text},
        {"cost": "Cost model", "value": str(begin.get("cost_model") or "n/a")},
        {"cost": "Fee schedule", "value": str(begin.get("fees") or "n/a")},
    ]


def _costs(art: RunArtifacts) -> None:
    rows = cost_rows(art.metrics, art.begin)
    if _section(SECTIONS[3], rows):
        assert rows is not None
        table(rows)


def run_warnings(art: RunArtifacts, doc: ResultDoc | None) -> list[str]:
    """The contract's warnings (summary, result record, suite), the begin record's and the prop plans'."""
    found: list[Any] = list((doc.contract.get("warnings") or ()) if doc is not None else ())
    found += list(art.begin.get("warnings") or ())
    for plan in art.prop:
        found += [f"prop {plan.get('firm')}/{plan.get('plan')}: {w}" for w in plan.get("warnings") or ()]
    return list(dict.fromkeys(str(w) for w in found))


def _warnings(art: RunArtifacts, doc: ResultDoc | None) -> None:
    _section(SECTIONS[4], True)
    for note in art.notes:
        st.info(note)
    for error in art.errors:
        st.error(f"Unreadable artifact: {error}")
    found = run_warnings(art, doc)
    if not found:
        st.caption("No warning.")
    for warning in found:
        st.markdown(f"- {warning}")


def _export(run_id: str, doc: ResultDoc | None) -> None:
    if not _section(SECTIONS[5], doc):
        return
    assert doc is not None
    st.download_button(
        EXPORT_LABEL,
        data=doc.data,
        file_name=f"{run_id}.result.json",
        mime=EXPORT_MIME,
        key=f"export_{run_id}",
        on_click="ignore",
    )
    source = (
        "runs/<id>/result.json as saved"
        if doc.saved
        else (
            "no result.json: the contract built from the ledger and the artifacts "
            "(as pfb.app.get_run builds it)"
        )
    )
    st.caption(f"The run's result contract (pfb.result/1): {source}.")


def render(config: DashConfig, run_id: str, view: LedgerView | None = None) -> None:
    """The results of one run; an error box when the ledger does not name it."""
    try:
        art = run_artifacts(config, run_id, view)
    except KeyError:
        st.error(f"The ledger has no begin record for {run_id!r}.")
        return
    try:
        doc = result_doc(config, art)
    except Exception as exc:  # an unreadable ledger line or file is shown, not raised
        st.error(f"The result contract cannot be read: {type(exc).__name__}: {exc}")
        doc = None
    _kpi_row(doc, art)
    _equity(art)
    _prop(art, doc)
    _costs(art)
    _warnings(art, doc)
    _export(run_id, doc)
