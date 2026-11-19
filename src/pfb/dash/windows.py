"""Step 3: which trade-date windows of a root are open, closed or missing, and which store files hold them.

Read-only and metadata only. The sources are the runner's: the closed-window table through its own parser
(pfb.data.windows.load_windows, hash-checked) and the store catalog (pfb.data.catalog.load_catalog, the
footer facts of runs/catalog.json). No bar store, no parquet footer and no bar row is opened here; the
DataGateway is not imported (tests/dash/test_read_only.py).

The rows of a root, ordered by first trade date:
- closed: one per closed window of the root (its label range, the kind and id as the reason), and whether a
  catalogued store file holds labels inside it (never read);
- open: the open part of a catalogued file's label range, one row per file and open span (a hypothesis
  window inside it can be registered and run, the runner still checks every row);
- unreadable: the open part of a catalogued file the gateway refuses from the catalog alone (an era not read
  in this version, or a sha256 unlike upstream's pin);
- missing: an open span of [REFERENCE_FIRST, today] that no catalogued file covers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from pfb.data.catalog import (
    DEFAULT_CATALOG_JSON,
    PIN_MISMATCH,
    Catalog,
    CatalogEntry,
    load_catalog,
    parse_label,
    pin_status,
)
from pfb.data.paths import ERA_MES_CONFIRMATION
from pfb.data.windows import (
    CLOSED_WINDOWS_SHA256,
    DEFAULT_WINDOWS_TOML,
    ClosedWindow,
    WindowTable,
    load_windows,
)

STATUS_OPEN = "open"
STATUS_CLOSED = "closed"
STATUS_MISSING = "missing"
STATUS_UNREADABLE = "unreadable"
# The earliest trade date the step reports on (an early bound for one-minute CME history); nothing earlier
# is reported as missing.
REFERENCE_FIRST = date(2010, 6, 7)
# The gateway's own UNSUPPORTED_ERAS (pfb.data.gateway, not importable here; a test checks they agree).
UNSUPPORTED_ERAS = frozenset({ERA_MES_CONFIRMATION})
# A closed window's last trade date at or past this year is open-ended (9999-12-31 in the table).
OPEN_ENDED_YEAR = 9999
MISSING_REASON = "not on disk"
_ONE_DAY = timedelta(days=1)


@dataclass(frozen=True)
class WindowRow:
    """One row of step 3. last is None for an open-ended closed window."""

    root: str
    first: date
    last: date | None
    status: str
    reason: str
    on_disk: bool
    files: tuple[str, ...]
    era: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "first": self.first.isoformat(),
            "last": None if self.last is None else self.last.isoformat(),
            "status": self.status,
            "reason": self.reason,
            "on_disk": self.on_disk,
            "files": list(self.files),
            "era": self.era,
        }


@dataclass(frozen=True)
class Sources:
    """Where the closed-window table and the catalog are; the defaults are the runner's own."""

    windows_toml: Path = DEFAULT_WINDOWS_TOML
    windows_sha256: str = CLOSED_WINDOWS_SHA256
    catalog_json: Path = DEFAULT_CATALOG_JSON

    def load(self) -> tuple[WindowTable, Catalog]:
        """The hash-checked table and the catalog; raises WindowTableMismatch or CatalogError."""
        return load_windows(self.windows_toml, self.windows_sha256), load_catalog(self.catalog_json)


def _span(entry: CatalogEntry) -> tuple[date, date] | None:
    if entry.first_trade_date is None or entry.last_trade_date is None:
        return None
    return parse_label(entry.first_trade_date, entry.path), parse_label(entry.last_trade_date, entry.path)


def _name(entry: CatalogEntry) -> str:
    return Path(entry.path).name


def _refusal(entry: CatalogEntry) -> str | None:
    """Why the gateway refuses this file from the catalog alone, else None."""
    if entry.era in UNSUPPORTED_ERAS:
        return f"era {entry.era} is not read in this version"
    if pin_status(entry.pe_pin, entry.sha256) == PIN_MISMATCH:
        return "sha256 differs from the producer pin"
    return None


def _closed_row(root: str, window: ClosedWindow, files: tuple[CatalogEntry, ...]) -> WindowRow:
    holders = tuple(
        _name(e) for e in files if (s := _span(e)) is not None and window.touches_labels(s[0], s[1])
    )
    open_ended = window.last_trade_date.year >= OPEN_ENDED_YEAR
    return WindowRow(
        root=root,
        first=window.first_trade_date,
        last=None if open_ended else window.last_trade_date,
        status=STATUS_CLOSED,
        reason=f"{window.kind} ({window.id})",
        on_disk=bool(holders),
        files=holders,
        era=None,
    )


def _file_rows(root: str, table: WindowTable, entry: CatalogEntry) -> list[WindowRow]:
    span = _span(entry)
    if span is None:
        return []
    refusal = _refusal(entry)
    status = STATUS_OPEN if refusal is None else STATUS_UNREADABLE
    return [
        WindowRow(root, lo, hi, status, refusal or "", True, (_name(entry),), entry.era)
        for lo, hi in table.open_ranges(root, *span)
    ]


def _uncovered(spans: list[tuple[date, date]], covers: list[tuple[date, date]]) -> list[tuple[date, date]]:
    """The parts of spans that no cover touches (inclusive day ranges)."""
    left = list(spans)
    for cut_lo, cut_hi in covers:
        kept: list[tuple[date, date]] = []
        for lo, hi in left:
            if cut_hi < lo or cut_lo > hi:
                kept.append((lo, hi))
                continue
            if lo < cut_lo:
                kept.append((lo, cut_lo - _ONE_DAY))
            if cut_hi < hi:
                kept.append((cut_hi + _ONE_DAY, hi))
        left = kept
    return sorted(left)


def _missing_rows(
    root: str, table: WindowTable, files: tuple[CatalogEntry, ...], today: date
) -> list[WindowRow]:
    covers = [s for e in files if (s := _span(e)) is not None]
    open_spans = list(table.open_ranges(root, REFERENCE_FIRST, today))
    return [
        WindowRow(root, lo, hi, STATUS_MISSING, MISSING_REASON, False, (), None)
        for lo, hi in _uncovered(open_spans, covers)
    ]


def window_rows(root: str, table: WindowTable, catalog: Catalog, today: date) -> tuple[WindowRow, ...]:
    """Every row of root (closed, open, unreadable, missing), ordered by first trade date then status."""
    files = catalog.files_for(root)
    rows = [_closed_row(root, w, files) for w in table.closed_windows_for(root)]
    for entry in files:
        rows.extend(_file_rows(root, table, entry))
    rows.extend(_missing_rows(root, table, files, today))
    return tuple(sorted(rows, key=lambda r: (r.first, r.status, r.files)))


def open_windows(rows: tuple[WindowRow, ...]) -> tuple[WindowRow, ...]:
    return tuple(r for r in rows if r.status == STATUS_OPEN)


def missing_windows(rows: tuple[WindowRow, ...]) -> tuple[WindowRow, ...]:
    return tuple(r for r in rows if r.status == STATUS_MISSING)
