"""Step 3 of the Backtest page: the SPEC root's windows (open, closed greyed with the reason, on disk,
missing). Read-only; no network call."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from pfb.dash.views import MUTED
from pfb.dash.windows import (
    STATUS_CLOSED,
    STATUS_UNREADABLE,
    Sources,
    WindowRow,
    open_windows,
    window_rows,
)

OPEN_ENDED = "open-ended"
COLUMNS = ("first", "last", "status", "reason", "on disk", "files")
# a checkout without runs/catalog.json is told the command that builds it.
CATALOG_COMMAND = "uv run pfb data catalog"


def _root_of(spec: Mapping[str, Any] | Any) -> str:
    """spec["root"] (a SPEC dict, or StrategySpec's dict-style access), else spec.root."""
    try:
        root = spec["root"]
    except (TypeError, KeyError):
        root = getattr(spec, "root", None)
    if not isinstance(root, str) or not root:
        raise ValueError(f"the SPEC root {root!r} is not a name")
    return root


def _table(rows: tuple[WindowRow, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "first": r.first.isoformat(),
                "last": OPEN_ENDED if r.last is None else r.last.isoformat(),
                "status": r.status,
                "reason": r.reason,
                "on disk": "yes" if r.on_disk else "no",
                "files": ", ".join(r.files),
            }
            for r in rows
        ],
        columns=list(COLUMNS),
    )


def _grey(row: pd.Series) -> list[str]:
    greyed = row["status"] in (STATUS_CLOSED, STATUS_UNREADABLE)
    return [f"color: {MUTED}" if greyed else "" for _ in row]


def missing_catalog_text(root: str, sources: Sources) -> str | None:
    """One plain line when the catalog file is absent (it is built, not shipped), else None."""
    if sources.catalog_json.is_file():
        return None
    return (
        f"windows of {root} unavailable: no runs/catalog.json yet; build it with `{CATALOG_COMMAND}` "
        "(reads config/paths.toml)"
    )


def render(
    spec: Mapping[str, Any] | Any,
    *,
    sources: Sources | None = None,
    today: date | None = None,
) -> list[dict[str, object]]:
    """Draw step 3 for the SPEC's root; the open windows (WindowRow.as_dict) for step 4."""
    root = _root_of(spec)
    used = sources or Sources()
    missing_catalog = missing_catalog_text(root, used)
    if missing_catalog is not None:
        st.error(missing_catalog)
        return []
    try:
        table, catalog = used.load()
    except Exception as exc:  # noqa: BLE001 - the table or catalog is unreadable; say so, offer nothing
        st.error(f"windows of {root} unavailable: {type(exc).__name__}: {exc}")
        return []
    rows = window_rows(root, table, catalog, today or date.today())
    st.dataframe(_table(rows).style.apply(_grey, axis=1), hide_index=True, width="stretch")
    opened = open_windows(rows)
    if not opened:
        st.info(f"{root}: no open window on disk")
    return [r.as_dict() for r in opened]
