"""Streamlit entry script of `pfb dash` (run by `streamlit run`, never imported by pfb): the left nav lists
Backtest (default) and Runs; Run detail and Prop Monte Carlo are hidden pages reached from Runs."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

PAGES = Path(__file__).resolve().parent / "pages"

st.set_page_config(page_title="pfb dashboard", layout="wide")
page = st.navigation(
    [
        st.Page(PAGES / "backtest.py", title="Backtest", url_path="backtest", default=True),
        st.Page(PAGES / "runs.py", title="Runs", url_path="runs"),
        st.Page(PAGES / "detail.py", title="Run detail", url_path="detail", visibility="hidden"),
        st.Page(PAGES / "prop.py", title="Prop Monte Carlo", url_path="prop", visibility="hidden"),
    ]
)
page.run()
