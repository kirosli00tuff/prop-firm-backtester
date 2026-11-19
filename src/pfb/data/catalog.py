"""The store catalog: every bar file the gateway may open, with its sha256 and footer facts.

Only parquet footers are read (row counts, schema, trade_date row-group statistics); the trade_date column
alone is read when its statistics are missing. No price column and no price statistic is ever read here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from pfb.data.errors import CatalogError, PathRefused
from pfb.data.paths import (
    ERA_HIST,
    ERA_RESEARCH,
    ERA_STEP2,
    FORBIDDEN_PARTS,
    REPO_ROOT,
    StoreFileId,
    StorePaths,
    check_path,
)

CATALOG_SCHEMA = "pfb.catalog/1"
DEFAULT_CATALOG_JSON = REPO_ROOT / "runs" / "catalog.json"
LABEL_COLUMN = "trade_date"
PIN_MATCH, PIN_MISMATCH, PIN_NONE = "match", "mismatch", "none"
NOT_LISTED_DIRS = FORBIDDEN_PARTS - {"superseded"}
HASH_BLOCK = 1 << 20
PINS_SCHEMA = "pfb.pins/1"

PinKey = tuple[str, str, str | None]  # (root, era, plan)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ISO_LABEL_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def pin_status(pe_pin: str | None, sha256: str) -> str:
    """The producer pin status, always recomputed from the pin and the file's sha256."""
    if pe_pin is None:
        return PIN_NONE
    return PIN_MATCH if pe_pin == sha256 else PIN_MISMATCH


def parse_label(text: object, where: str) -> date:
    """A trade-date label in strict YYYY-MM-DD form, else CatalogError (a GatewayRefusal)."""
    if not isinstance(text, str) or not _ISO_LABEL_RE.fullmatch(text):
        raise CatalogError(f"{where}: trade-date label {text!r} is not an ISO YYYY-MM-DD date")
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise CatalogError(f"{where}: trade-date label {text!r} is not an ISO YYYY-MM-DD date") from exc


@dataclass(frozen=True)
class CatalogEntry:
    root: str
    era: str
    plan: str | None
    path: str
    size: int
    sha256: str
    num_rows: int
    first_trade_date: str | None
    last_trade_date: str | None
    schema: tuple[tuple[str, str], ...]
    pe_pin: str | None
    pe_pin_status: str

    def overlaps(self, first: date, last: date) -> bool:
        if self.first_trade_date is None or self.last_trade_date is None:
            return False
        lo = parse_label(self.first_trade_date, self.path)
        hi = parse_label(self.last_trade_date, self.path)
        return lo <= last and first <= hi


@dataclass(frozen=True)
class IgnoredFile:
    path: str
    reason: str


@dataclass(frozen=True)
class Catalog:
    created_utc: str
    closed_windows_sha256: str
    files: tuple[CatalogEntry, ...]
    ignored: tuple[IgnoredFile, ...]

    def roots(self) -> frozenset[str]:
        return frozenset(e.root for e in self.files)

    def files_for(self, root: str) -> tuple[CatalogEntry, ...]:
        return tuple(e for e in self.files if e.root == root)

    def pin_mismatches(self) -> tuple[CatalogEntry, ...]:
        return tuple(e for e in self.files if e.pe_pin_status == PIN_MISMATCH)

    def to_json(self) -> dict[str, object]:
        return {
            "schema": CATALOG_SCHEMA,
            "created_utc": self.created_utc,
            "closed_windows_sha256": self.closed_windows_sha256,
            "files": [_entry_json(e) for e in self.files],
            "ignored": [asdict(i) for i in self.ignored],
        }


def _entry_json(entry: CatalogEntry) -> dict[str, object]:
    out = asdict(entry)
    out["schema"] = [{"name": n, "dtype": t} for n, t in entry.schema]
    return out


# ------------------------------------------------------------------ scanning the roots ----
def scan_roots(paths: StorePaths) -> tuple[list[tuple[Path, StoreFileId]], list[IgnoredFile]]:
    """Walk the three roots without opening any file. Forbidden directories are not listed."""
    accepted: list[tuple[Path, StoreFileId]] = []
    ignored: list[IgnoredFile] = []
    seen: set[Path] = set()
    for root in paths.store_roots().values():
        if not root.is_dir():
            raise CatalogError(f"store root {root} is not a directory")
        for current, dirnames, filenames in os.walk(root, followlinks=False):
            here = Path(current)
            for name in sorted(dirnames):
                if name in NOT_LISTED_DIRS:
                    ignored.append(
                        IgnoredFile(str(here / name), f"directory {name!r}: forbidden, not listed")
                    )
                elif (here / name).is_symlink():
                    ignored.append(IgnoredFile(str(here / name), "symlinked directory: not followed"))
            dirnames[:] = sorted(
                d for d in dirnames if d not in NOT_LISTED_DIRS and not (here / d).is_symlink()
            )
            for name in sorted(filenames):
                _classify_file(here / name, paths, accepted, ignored, seen)
    return accepted, ignored


def _classify_file(
    path: Path,
    paths: StorePaths,
    accepted: list[tuple[Path, StoreFileId]],
    ignored: list[IgnoredFile],
    seen: set[Path],
) -> None:
    try:
        file_id = check_path(path, paths)
    except PathRefused as exc:
        ignored.append(IgnoredFile(str(path), str(exc)))
        return
    if file_id.resolved in seen:
        ignored.append(IgnoredFile(str(path), f"duplicate of {file_id.resolved}"))
        return
    seen.add(file_id.resolved)
    accepted.append((path, file_id))


# ------------------------------------------------------------------ footer facts ----
def _as_text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _label_range_from_stats(meta: pq.FileMetaData, col: int) -> tuple[str | None, str | None] | None:
    lows: list[str] = []
    highs: list[str] = []
    for i in range(meta.num_row_groups):
        group = meta.row_group(i)
        if group.num_rows == 0:
            continue
        stats = group.column(col).statistics
        if stats is None or not stats.has_min_max:
            return None
        lows.append(_as_text(stats.min))
        highs.append(_as_text(stats.max))
    if not lows:
        return (None, None)
    return min(lows), max(highs)


def _label_range_from_column(path: Path) -> tuple[str | None, str | None]:
    labels = pq.read_table(path, columns=[LABEL_COLUMN]).column(LABEL_COLUMN)
    bounds = pc.min_max(labels).as_py()  # pyright: ignore[reportAttributeAccessIssue]
    return bounds["min"], bounds["max"]


def footer_facts(path: Path) -> tuple[int, tuple[tuple[str, str], ...], str | None, str | None]:
    """(num_rows, schema, first label, last label) from the footer; trade_date alone when stats are absent."""
    handle = pq.ParquetFile(path)
    meta = handle.metadata
    schema = tuple((f.name, str(f.type)) for f in handle.schema_arrow)
    names = [n for n, _ in schema]
    if LABEL_COLUMN not in names:
        raise CatalogError(f"{path}: no {LABEL_COLUMN} column")
    col = handle.schema_arrow.get_field_index(LABEL_COLUMN)
    found = _label_range_from_stats(meta, col)
    first, last = found if found is not None else _label_range_from_column(path)
    return meta.num_rows, schema, first, last


# ------------------------------------------------------------------ producer pins ----
def _read_json(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"{path}: cannot read JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise CatalogError(f"{path}: not a JSON object")
    return data


def _pin_entry(raw: object, where: str) -> tuple[PinKey, str]:
    if not isinstance(raw, dict):
        raise CatalogError(f"{where}: a pin is not an object")
    root, era, plan, sha = raw.get("root"), raw.get("era"), raw.get("plan"), raw.get("sha256")
    if not isinstance(root, str) or era not in (ERA_RESEARCH, ERA_STEP2, ERA_HIST):
        raise CatalogError(f"{where}: a pin needs a root and an era of research, step2 or hist")
    if plan is not None and not isinstance(plan, str):
        raise CatalogError(f"{where}: a pin's plan is a string or null")
    if not isinstance(sha, str) or not _SHA256_RE.fullmatch(sha):
        raise CatalogError(f"{where}: pin sha256 of {root} {era} is malformed")
    return (root, str(era), plan), sha


def load_pins(path: Path) -> dict[PinKey, str]:
    """The sha256 the data producer recorded per store file: {"schema": "pfb.pins/1", "pins": [{"root",
    "era", "plan", "sha256"}, ...]}. A file the catalog finds with a pin must hash to it (PinMismatch)."""
    data = _read_json(path)
    if data.get("schema") != PINS_SCHEMA:
        raise CatalogError(f"{path}: schema {data.get('schema')!r} != {PINS_SCHEMA!r}")
    raw = data.get("pins")
    if not isinstance(raw, list):
        raise CatalogError(f"{path}: pins must be a list")
    pins: dict[PinKey, str] = {}
    for entry in raw:
        key, sha = _pin_entry(entry, str(path))
        if key in pins:
            raise CatalogError(f"{path}: {key} is pinned twice")
        pins[key] = sha
    return pins


def pins_json(pins: dict[PinKey, str]) -> dict[str, object]:
    """The JSON form load_pins reads, in a stable order."""
    ordered = sorted(pins.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or ""))
    rows = [{"root": root, "era": era, "plan": plan, "sha256": sha} for (root, era, plan), sha in ordered]
    return {"schema": PINS_SCHEMA, "pins": rows}


def _default_pins(paths: StorePaths) -> dict[PinKey, str]:
    return {} if paths.pins_json is None else load_pins(paths.pins_json)


# ------------------------------------------------------------------ build / write / load ----
def _entry(path: Path, file_id: StoreFileId, pins: dict[PinKey, str]) -> CatalogEntry:
    num_rows, schema, first, last = footer_facts(file_id.resolved)
    sha = file_sha256(file_id.resolved)
    pin = pins.get((file_id.root, file_id.era, file_id.plan))
    status = pin_status(pin, sha)
    return CatalogEntry(
        root=file_id.root,
        era=file_id.era,
        plan=file_id.plan,
        path=str(path),
        size=file_id.resolved.stat().st_size,
        sha256=sha,
        num_rows=num_rows,
        first_trade_date=first,
        last_trade_date=last,
        schema=schema,
        pe_pin=pin,
        pe_pin_status=status,
    )


def build_catalog(
    paths: StorePaths, closed_windows_sha256: str, created_utc: str, pins: dict[PinKey, str] | None = None
) -> Catalog:
    """Scan the roots, hash every accepted file and read its footer. pins default to paths.pins_json's
    (none when it is not set)."""
    use_pins = _default_pins(paths) if pins is None else pins
    accepted, ignored = scan_roots(paths)
    files = tuple(_entry(path, file_id, use_pins) for path, file_id in accepted)
    return Catalog(created_utc, closed_windows_sha256, files, tuple(ignored))


def write_catalog(catalog: Catalog, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(catalog.to_json(), indent=1) + "\n", encoding="utf-8")


def _entry_from_json(raw: object) -> CatalogEntry:
    if not isinstance(raw, dict):
        raise CatalogError("catalog file entry is not an object")
    try:
        schema = tuple((str(c["name"]), str(c["dtype"])) for c in raw["schema"])
        entry = CatalogEntry(**{**raw, "schema": schema})
    except (KeyError, TypeError) as exc:
        raise CatalogError(f"malformed catalog entry: {exc}") from exc
    if not isinstance(entry.sha256, str) or not _SHA256_RE.fullmatch(entry.sha256):
        raise CatalogError(f"{entry.path}: catalog sha256 malformed")
    pin = entry.pe_pin
    if pin is not None and not (isinstance(pin, str) and _SHA256_RE.fullmatch(pin)):
        raise CatalogError(f"{entry.path}: catalog pe_pin malformed")
    check_pin_status(entry)
    return entry


def check_pin_status(entry: CatalogEntry) -> str:
    """The recomputed pin status; CatalogError when the stored pe_pin_status disagrees with it."""
    expected = pin_status(entry.pe_pin, entry.sha256)
    if entry.pe_pin_status != expected:
        raise CatalogError(
            f"{entry.path}: pe_pin_status {entry.pe_pin_status!r} disagrees with {expected!r} recomputed "
            "from pe_pin and sha256"
        )
    return expected


def load_catalog(path: Path = DEFAULT_CATALOG_JSON) -> Catalog:
    data = _read_json(path)
    if data.get("schema") != CATALOG_SCHEMA:
        raise CatalogError(f"{path}: schema {data.get('schema')!r} != {CATALOG_SCHEMA!r}")
    files = data.get("files")
    ignored = data.get("ignored")
    if not isinstance(files, list) or not isinstance(ignored, list):
        raise CatalogError(f"{path}: files and ignored must be lists")
    return Catalog(
        created_utc=str(data.get("created_utc")),
        closed_windows_sha256=str(data.get("closed_windows_sha256")),
        files=tuple(_entry_from_json(raw) for raw in files),
        ignored=tuple(IgnoredFile(str(i["path"]), str(i["reason"])) for i in ignored),
    )
