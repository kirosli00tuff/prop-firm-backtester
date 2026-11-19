"""The analysis suite of one run (summary["suite"] or suite.json, plus the extras frames): cost stress, random
baseline, walk-forward folds, power, sizing and the breakdowns. A part that is off is said to be off."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import streamlit as st

from pfb.dash.data import fmt_cents, fmt_num, load_frame
from pfb.dash.views import part, table

HIST_BINS = 30
BREAKDOWN_TABLES = (
    ("by_year", "By calendar year"),
    ("by_vol_tercile", "By volatility tercile of the day (descriptive: the same day's range)"),
    ("by_entry_hour_ct", "By CT hour of the entry bar"),
    ("by_weekday", "By weekday"),
)


def _map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def cost_stress(stress: Any, metrics: Mapping[str, Any]) -> None:
    if not part("Cost stress", stress):
        return
    rows = [
        {
            "extra_ticks_per_side": 0,
            "cost_model": "(the run)",
            "net_cents": metrics.get("net_cents"),
            "sharpe_annual": metrics.get("sharpe_annual"),
            "trades": metrics.get("n_trades"),
            "slippage_cents": metrics.get("slippage_cents"),
        }
    ]
    for k, level in sorted(_map(stress).items(), key=lambda kv: int(kv[0])):
        rows.append({"extra_ticks_per_side": int(k), **_map(level)})
    table(rows, "Same trades, slippage raised by k ticks per side per fill.")


def _histogram(draws: pl.DataFrame, mode: str, strategy_net: Any) -> None:
    nets = draws.filter(pl.col("mode") == mode)["net_cents"].to_numpy()
    if nets.size == 0:
        return
    counts, edges = np.histogram(nets, bins=HIST_BINS)
    centres = (edges[:-1] + edges[1:]) / 2 / 100
    st.bar_chart(pl.DataFrame({"net_dollars": centres, "draws": counts}), x="net_dollars", y="draws")
    st.caption(f"{mode} draws' net (dollars); the strategy netted {fmt_cents(strategy_net)}.")


def baseline(value: Any, run_dir: Path) -> None:
    if not part("Random-entry baseline", value):
        return
    base = _map(value)
    mode = str(base.get("mode") or base.get("headline") or "uniform")  # "headline": an older run
    st.markdown(
        f"Reported mode ({mode}): net percentile {fmt_num(base.get('net_percentile'))}, "
        f"p {fmt_num(base.get('p_net'))}; Sharpe percentile {fmt_num(base.get('sharpe_percentile'))}, "
        f"p {fmt_num(base.get('p_sharpe'))}; {base.get('draws')} draws, seed {base.get('seed')}."
    )
    rows = []
    for mode in ("uniform", "tod", "cost_matched"):  # cost_matched: absent
        block = _map(base.get(mode))
        for measure in ("net", "sharpe"):
            stats = _map(block.get(measure))
            if stats:
                rows.append({"mode": mode, "measure": measure, "overlaps": block.get("overlaps"), **stats})
    table(
        rows,
        "percentile: share of draws strictly below plus half the ties; p = (1 + #draws >= strategy) "
        "/ (draws + 1).",
    )
    draws = load_frame(run_dir, "baseline_draws")
    if draws is None:
        st.caption("baseline_draws.parquet is not present: no histogram.")
        return
    _histogram(draws, mode, _map(base.get(mode)).get("net", {}).get("strategy"))


def walk_forward(value: Any) -> None:
    if not part("Walk-forward folds", value):
        return
    wf = _map(value)
    st.markdown(
        f"{wf.get('n_folds')} folds over {wf.get('n_days')} days; "
        f"Sharpe spread {fmt_num(wf.get('sharpe_spread'))} "
        f"(min {fmt_num(wf.get('sharpe_min'))}, max {fmt_num(wf.get('sharpe_max'))}); positive folds "
        f"{wf.get('positive_folds')}."
    )
    table(list(wf.get("folds") or ()))
    for name in ("anchored", "rolling"):
        pairs = _map(wf.get(name))
        st.caption(
            f"{name}: same-sign share {fmt_num(pairs.get('same_sign_share'))}, mean OOS / IS Sharpe "
            f"{fmt_num(pairs.get('mean_oos_over_is'))} over {pairs.get('n_pairs_scored')} pairs."
        )


def power(value: Any) -> None:
    if not part("Power", value):
        return
    pw = _map(value)
    rows = [{"field": k, "value": v} for k, v in pw.items() if k != "detail"]
    table(rows, "Realized: the run's own trades per year, sd, skew and kurtosis; N is the run's.")
    if pw.get("detail") is not None:
        with st.expander("Power detail"):
            st.json(dict(_map(pw.get("detail"))))


def sizing(value: Any) -> None:
    if not part("Sizing", value):
        return
    sz = _map(value)
    st.caption(
        f"Plan {sz.get('plan') or '(none)'}: drawdown {fmt_cents(sz.get('drawdown_cents'))}, daily loss "
        f"{fmt_cents(sz.get('daily_loss_cents'))}, cap {sz.get('cap_contracts')} contracts, risk from "
        f"{sz.get('risk_source')}. {sz.get('approximation')}."
    )
    table([{"rule": name, **_map(rule)} for name, rule in _map(sz.get("rules")).items()])
    kelly = _map(sz.get("kelly"))
    if kelly:
        rows = [{"rule": k, **_map(v)} for k, v in kelly.items() if isinstance(v, Mapping)]
        table(rows, f"Kelly: {kelly.get('flag')}.")


def _rolling(run_dir: Path, summary: Mapping[str, Any]) -> None:
    frame = load_frame(run_dir, "rolling_sharpe")
    if frame is not None and frame.height:
        st.line_chart(frame, x="trade_date", y=["sharpe_63", "sharpe_252"])
    rows = [{"window_days": k, **_map(v)} for k, v in _map(summary).items()]
    table(rows, "Rolling annual Sharpe over 63 and 252 trading days.")


def breakdowns(value: Any, run_dir: Path) -> None:
    if not part("Breakdowns", value):
        return
    bd = _map(value)
    for key, title in BREAKDOWN_TABLES:
        st.markdown(f"**{title}**")
        table(list(bd.get(key) or ()))
    st.markdown(
        f"**Best day** {fmt_cents(bd.get('best_day_cents'))} (share of a positive total: "
        f"{fmt_num(bd.get('best_day_share'))})"
    )
    st.markdown("**Rolling Sharpe**")
    _rolling(run_dir, _map(bd.get("rolling_sharpe")))
    exc = _map(bd.get("excursions"))
    rows = [
        {"measure": m, "trades": group, **_map(q)}
        for m, groups in exc.items()
        for group, q in _map(groups).items()
    ]
    st.markdown("**Excursions (ticks)**")
    table(rows)
    st.markdown("**Hold minutes and trades per day**")
    table([{"what": "hold_minutes", **_map(bd.get("hold_minutes"))}])
    table([_map(bd.get("trades_per_day"))])


def render_suite(suite: Mapping[str, Any] | None, metrics: Mapping[str, Any], run_dir: Path) -> None:
    if suite is None:
        st.info("No suite: this run was made without one (no suite.json and no summary suite).")
        return
    cost_stress(suite.get("cost_stress"), metrics)
    baseline(suite.get("baseline"), run_dir)
    walk_forward(suite.get("walk_forward"))
    power(suite.get("power"))
    sizing(suite.get("sizing"))
