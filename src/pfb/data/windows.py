"""The closed-window table (config/closed_windows.toml).

A window closes a row of root R (products ["*"] or listing R) when EITHER its stored trade_date label lies in
[first_trade_date, last_trade_date] OR utc_start <= ts_event < utc_end. A request is refused when its
trade-date range touches a label range of the root. The table's bytes must hash to CLOSED_WINDOWS_SHA256.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from pfb.data.errors import ClosedWindowRefused, WindowTableMismatch
from pfb.data.paths import REPO_ROOT

CLOSED_WINDOWS_SHA256 = "212c4ccb50d8559631528901b6eb6ad0fd803922c4e6cd44fb3340f024929583"
DEFAULT_WINDOWS_TOML = REPO_ROOT / "config" / "closed_windows.toml"
WINDOW_SCHEMA = "pfb.closed_windows/1"
WINDOW_KINDS = frozenset({"sealed holdout", "embargo", "sealed test window", "closed by decision"})
ALL_PRODUCTS = "*"
INT64_MAX = 2**63 - 1
NS_PER_SECOND = 1_000_000_000
NS_PER_MICROSECOND = 1_000
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_ONE_DAY = timedelta(days=1)


def datetime_to_ns(moment: datetime) -> int:
    """Exact integer UTC nanoseconds of an aware datetime (no float on the way)."""
    if moment.tzinfo is None:
        raise ValueError(f"{moment!r} is naive; an aware datetime is required")
    delta = moment.astimezone(UTC) - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * NS_PER_SECOND + delta.microseconds * NS_PER_MICROSECOND


@dataclass(frozen=True)
class ClosedWindow:
    """One closed window. utc_*_ns are Python ints (an open end lies past the int64 range)."""

    id: str
    products: tuple[str, ...]
    first_trade_date: date
    last_trade_date: date
    utc_start_ns: int
    utc_end_ns: int
    kind: str

    def applies_to(self, root: str) -> bool:
        return ALL_PRODUCTS in self.products or root in self.products

    def touches_labels(self, first: date, last: date) -> bool:
        return first <= self.last_trade_date and self.first_trade_date <= last

    def row_expr(self, label_col: str, ts_col: str) -> pl.Expr:
        """True for a row closed by this window, by label (a Date column) or by time (Int64 ns)."""
        start = min(self.utc_start_ns, INT64_MAX)
        end = min(self.utc_end_ns, INT64_MAX)
        by_label = pl.col(label_col).is_between(pl.lit(self.first_trade_date), pl.lit(self.last_trade_date))
        by_time = (pl.col(ts_col) >= pl.lit(start, dtype=pl.Int64)) & (
            pl.col(ts_col) < pl.lit(end, dtype=pl.Int64)
        )
        return by_label | by_time


def _field(raw: dict[str, object], key: str, kind: type) -> object:
    value = raw.get(key)
    if not isinstance(value, kind):
        raise WindowTableMismatch(f"window {raw.get('id')!r}: field {key!r} missing or not {kind.__name__}")
    return value


def _date(raw: dict[str, object], key: str) -> date:
    text = str(_field(raw, key, str))
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise WindowTableMismatch(f"window {raw.get('id')!r}: {key} {text!r} is not an ISO date") from exc


def _utc_ns(raw: dict[str, object], key: str) -> int:
    text = str(_field(raw, key, str))
    if not text.endswith("Z"):
        raise WindowTableMismatch(f"window {raw.get('id')!r}: {key} {text!r} must be UTC with a Z suffix")
    try:
        return datetime_to_ns(datetime.fromisoformat(text))
    except ValueError as exc:
        raise WindowTableMismatch(f"window {raw.get('id')!r}: {key} {text!r} is not an ISO instant") from exc


def _parse_window(raw: dict[str, object]) -> ClosedWindow:
    products = raw.get("products")
    if not isinstance(products, list) or not products or not all(isinstance(p, str) and p for p in products):
        raise WindowTableMismatch(f"window {raw.get('id')!r}: products must be a non-empty list of names")
    window = ClosedWindow(
        id=str(_field(raw, "id", str)),
        products=tuple(str(p) for p in products),
        first_trade_date=_date(raw, "first_trade_date"),
        last_trade_date=_date(raw, "last_trade_date"),
        utc_start_ns=_utc_ns(raw, "utc_start"),
        utc_end_ns=_utc_ns(raw, "utc_end"),
        kind=str(_field(raw, "kind", str)),
    )
    if window.kind not in WINDOW_KINDS:
        raise WindowTableMismatch(f"window {window.id!r}: unknown kind {window.kind!r}")
    if window.first_trade_date > window.last_trade_date or window.utc_start_ns >= window.utc_end_ns:
        raise WindowTableMismatch(f"window {window.id!r}: an empty or reversed range")
    return window


def _subtract(spans: list[tuple[date, date]], cut_first: date, cut_last: date) -> list[tuple[date, date]]:
    kept: list[tuple[date, date]] = []
    for lo, hi in spans:
        if cut_last < lo or cut_first > hi:
            kept.append((lo, hi))
            continue
        if lo < cut_first:
            kept.append((lo, cut_first - _ONE_DAY))
        if cut_last < hi:
            kept.append((cut_last + _ONE_DAY, hi))
    return kept


@dataclass(frozen=True)
class WindowTable:
    """The loaded, hash-checked table."""

    windows: tuple[ClosedWindow, ...]
    sha256: str

    def closed_windows_for(self, root: str) -> tuple[ClosedWindow, ...]:
        return tuple(w for w in self.windows if w.applies_to(root))

    def request_refusal(self, root: str, first: date, last: date) -> ClosedWindowRefused | None:
        """The refusal for a request whose trade-date range touches a closed label range, else None."""
        hits = [w for w in self.closed_windows_for(root) if w.touches_labels(first, last)]
        if not hits:
            return None
        ids = ", ".join(w.id for w in hits)
        return ClosedWindowRefused(
            hits[0].id, f"{root} {first}..{last} touches closed window(s) {ids} ({hits[0].kind})"
        )

    def open_ranges(self, root: str, first: date, last: date) -> tuple[tuple[date, date], ...]:
        """The open sub-ranges of [first, last] by trade-date label, in order."""
        if first > last:
            return ()
        spans = [(first, last)]
        for window in self.closed_windows_for(root):
            spans = _subtract(spans, window.first_trade_date, window.last_trade_date)
        return tuple(sorted(spans))

    def first_closed_row(
        self, root: str, frame: pl.DataFrame, label_col: str, ts_col: str
    ) -> tuple[str, int] | None:
        """(window id, rows) of the first window closing any row of frame, else None."""
        for window in self.closed_windows_for(root):
            hits = frame.select(window.row_expr(label_col, ts_col).sum()).item()
            if hits:
                return window.id, int(hits)
        return None


def load_windows(
    path: Path = DEFAULT_WINDOWS_TOML, expected_sha256: str = CLOSED_WINDOWS_SHA256
) -> WindowTable:
    """Load the table; raise WindowTableMismatch when its bytes hash differently or it is malformed."""
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise WindowTableMismatch(f"{path}: sha256 {actual} != pinned {expected_sha256}")
    try:
        table = tomllib.loads(data.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise WindowTableMismatch(f"{path}: not valid TOML: {exc}") from exc
    if table.get("schema") != WINDOW_SCHEMA:
        raise WindowTableMismatch(f"{path}: schema {table.get('schema')!r} != {WINDOW_SCHEMA!r}")
    raws = table.get("window")
    if not isinstance(raws, list) or not raws:
        raise WindowTableMismatch(f"{path}: no [[window]] entries")
    windows = tuple(_parse_window(raw) for raw in raws)
    ids = [w.id for w in windows]
    if len(set(ids)) != len(ids):
        raise WindowTableMismatch(f"{path}: duplicate window ids")
    return WindowTable(windows=windows, sha256=actual)
