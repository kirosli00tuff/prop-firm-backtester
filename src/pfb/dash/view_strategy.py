"""Backtest page, steps 1 and 2 (docs/design/DASH_V2.md): a strategy pasted or dropped, its
sha256 and parsed SPEC, the template, and the free check (lint, look-ahead, a synthetic run; N unchanged).

This module writes nothing and spawns nothing itself: the upload and the check go through pfb.dash.actions,
the dashboard's one writer. Results are kept in st.session_state by sha256, so a rerun shows them again.
"""

from __future__ import annotations

import hashlib
import html
from collections.abc import Mapping
from typing import Any

import streamlit as st

from pfb.dash.actions import (
    MAX_UPLOAD_BYTES,
    CheckItem,
    StrategyState,
    UploadRefused,
    check_strategy,
    read_spec,
    save_upload,
)
from pfb.dash.views import GAIN, LOSS, dash_config, table
from pfb.registry.api import REPO_ROOT

TEMPLATE_REL = "strategies/_template.py"
STATE_KEY = "strategy_checks"
TEXT_KEY, FILE_KEY, CHECK_KEY = "strategy_text", "strategy_file", "strategy_check"
SPEC_FIELDS = ("root", "bar_minutes", "session", "contracts")
DOT = "&#9679;"


def _template_bytes() -> bytes | None:
    path = REPO_ROOT / TEMPLATE_REL
    return path.read_bytes() if path.is_file() else None


def _template() -> None:
    data = _template_bytes()
    if data is None:
        st.caption(f"{TEMPLATE_REL} is not in this checkout.")
        return
    st.download_button("Download the template", data, file_name="_template.py", mime="text/x-python")
    with st.expander("Template (strategies/_template.py)"):
        st.code(data.decode("utf-8"), language="python")


def _source() -> bytes | None:
    """The dropped file's bytes when there is one, else the pasted text (UTF-8); None when both are blank."""
    st.caption(
        f"Paste the strategy or drop a .py file (at most {MAX_UPLOAD_BYTES // 1024} KiB); the file wins."
    )
    text = st.text_area("Strategy code", key=TEXT_KEY, height=320)
    dropped = st.file_uploader("Strategy file", type=["py"], key=FILE_KEY)
    if dropped is not None:
        return dropped.getvalue()
    return text.encode("utf-8") if text.strip() else None


def _params_rows(params: Any) -> list[dict[str, Any]]:
    if not isinstance(params, Mapping):
        return []
    rows = []
    for name, item in params.items():
        found = item if isinstance(item, Mapping) else {}
        rows.append({"param": name, "default": found.get("default"), "grid": found.get("grid")})
    return rows


def _spec(spec: Mapping[str, Any] | None) -> None:
    if spec is None:
        st.caption("No SPEC dict literal could be read from this file (the check below says why).")
        return
    st.markdown("  \n".join(f"`{name}`: {html.escape(str(spec.get(name)))}" for name in SPEC_FIELDS))
    table(_params_rows(spec.get("params")), caption="Parameters: default and grid.")


def _item(item: CheckItem) -> None:
    colour = GAIN if item.ok else LOSS
    word = "pass" if item.ok else "fail"
    st.markdown(
        f'<span style="color:{colour}">{DOT}</span> <b>{html.escape(item.name)}</b> ({word}): '
        f"<code>{html.escape(item.message)}</code>",
        unsafe_allow_html=True,
    )


def _results() -> dict[str, StrategyState]:
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = {}
    return st.session_state[STATE_KEY]


def _run_check(source: bytes) -> StrategyState | None:
    config = dash_config()
    try:
        upload = save_upload(source, config.repo_root)
    except UploadRefused as exc:
        st.error(f"Not saved: {exc}")
        return None
    with st.spinner("Checking: lint, look-ahead, synthetic run"):
        state = check_strategy(upload, source, config=config)
    _results()[state.sha256] = state
    return state


def render(headings: bool = True) -> StrategyState | None:
    """Steps 1 and 2; the check's StrategyState for the current file once it was checked, else None."""
    if headings:
        st.subheader("1 Strategy")
    _template()
    source = _source()
    if source is None:
        st.caption("No strategy yet.")
        return None
    sha = hashlib.sha256(source).hexdigest()
    st.markdown(f"sha256 `{sha}`")
    _spec(read_spec(source))
    if headings:
        st.subheader("2 Check")
    st.caption("Free: lint, look-ahead check and a synthetic run on synthetic bars; N unchanged.")
    state = _results().get(sha)
    if st.button("Check", key=CHECK_KEY):
        state = _run_check(source)
    if state is None:
        return None
    for item in state.checks:
        _item(item)
    st.caption(f"Saved as {state.rel_path}. {'Passed' if state.passed else 'Failed'}.")
    return state
