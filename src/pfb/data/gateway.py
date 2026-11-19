"""The DataGateway: the ONLY code path in this repository that reads bars.

bars() refuses (raises) instead of filtering: a request touching a closed window raises before any file is
opened; every returned row is then checked again against the bound, the closed windows (label and time), the
store order and the tick grid. Prices come back as int64 vendor ticks. No upstream code is imported.
files_for() names, without reading a bar, exactly the catalog entries bars() reads, in read order (ADR A14 d).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from pfb.data.catalog import (
    PIN_MATCH,
    PIN_MISMATCH,
    PIN_NONE,
    Catalog,
    CatalogEntry,
    check_pin_status,
    file_sha256,
)
from pfb.data.errors import (
    BadRequest,
    BoundBreached,
    CatalogError,
    ClosedWindowRefused,
    EraNotSupported,
    HashMismatch,
    NoBarsFound,
    OffGridPrice,
    PathRefused,
    PinMismatch,
    StoreRefused,
    TickMismatch,
    WindowTableMismatch,
)
from pfb.data.guards import (  # CT and bound_ns now live in guards; imported here for the gateway's users
    CT,
    bound_ns,
    check_catalog_windows,
    check_ts_range,
    pinned_window_table,
    refuse_real_store_roots,
)
from pfb.data.paths import (
    DEFAULT_PATHS_TOML,
    ERA_HIST,
    ERA_MES_CONFIRMATION,
    ERA_RESEARCH,
    ERA_STEP2,
    StorePaths,
    check_path,
)
from pfb.data.windows import CLOSED_WINDOWS_SHA256, WindowTable

PRICE_SCALE = 1_000_000_000
GRID_TOLERANCE = 1e-6
MAX_ABS_PRICE = 9.0e9  # rint(price * 1e9) must fit in int64
MINUTES_PER_DAY = 1440
# The store metadata tick (tick.tick_fixed_1e9) is the EXCHANGE tick; the stored prices use the vendor tick =
# exchange tick * vendor_price_factor, checked exactly against the injected price_factor_of (ADR A5 d).
PRICE_COLUMNS = ("open", "high", "low", "close")
ERA_ORDER = {ERA_HIST: 0, ERA_STEP2: 1, ERA_MES_CONFIRMATION: 1, ERA_RESEARCH: 2}
# ADR A5 b: Upstream re-books the MES confirmation file's scheduled-closure prints with rules not ported
# here.
UNSUPPORTED_ERAS = frozenset({ERA_MES_CONFIRMATION})
READABLE_PIN_STATUSES = frozenset({PIN_MATCH, PIN_NONE})
# A file whose metadata has no tick is read, flagged and warned about.
TICK_EXACT, TICK_ABSENT = "exact", "absent"
# Only the two MES files of ADR A14 (h) may lack a metadata tick; any other is refused.
TICK_ABSENT_ALLOWED = frozenset({("MES", ERA_RESEARCH), ("MES", ERA_MES_CONFIRMATION)})
STORE_METADATA_KEY = b"pfb.store"  # footer key of the store metadata JSON (holds the tick)
_FILE_IDX = "__file_idx"
_LABEL = "trade_date"

STORED_SCHEMA: dict[str, pl.DataType] = {
    "ts_event": pl.Int64(),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Int64(),
    "instrument_id": pl.Int64(),
    "raw_symbol": pl.String(),
    "in_flatten_window": pl.Boolean(),
    "in_no_new_positions_window": pl.Boolean(),
    "early_halt_ct": pl.String(),
    "in_scheduled_closure": pl.Boolean(),
    "trade_date": pl.String(),
    "is_roll_session": pl.Boolean(),
    "gap_before_minutes": pl.Int64(),
    "vendor_degraded_day": pl.Boolean(),
}
OUTPUT_COLUMNS = (
    "ts_event", "open", "high", "low", "close", "volume", "trade_date", "ct_minute", "is_roll_session",
    "in_flatten_window", "in_no_new_positions_window", "in_scheduled_closure", "vendor_degraded_day",
    "early_halt_ct", "gap_before_minutes", "instrument_id", "raw_symbol", "era",
)  # fmt: skip


@dataclass(frozen=True)
class AllSessions:
    """Every bar of the requested trade dates."""


ALL = AllSessions()


@dataclass(frozen=True)
class ClockWindow:
    """Bars with start_minute <= ct_minute < end_minute; start > end wraps past CT midnight."""

    start_minute: int
    end_minute: int

    def __post_init__(self) -> None:
        if not 0 <= self.start_minute < MINUTES_PER_DAY or not 0 <= self.end_minute <= MINUTES_PER_DAY:
            raise BadRequest(f"clock window {self.start_minute}..{self.end_minute} outside 0..1440")
        if self.start_minute == self.end_minute:
            raise BadRequest("clock window with start == end is ambiguous")

    def expr(self) -> pl.Expr:
        minute = pl.col("ct_minute")
        if self.start_minute < self.end_minute:
            return (minute >= self.start_minute) & (minute < self.end_minute)
        return (minute >= self.start_minute) | (minute < self.end_minute)


Session = AllSessions | ClockWindow


@dataclass(frozen=True)
class ProvenanceRecord:
    path: str
    era: str
    sha256: str
    rows: int
    pe_pin_status: str
    tick_metadata: str


def provenance_warnings(records: tuple[ProvenanceRecord, ...]) -> tuple[str, ...]:
    """One warning per file and cause: no producer pin (ADR A5 a), no metadata tick."""
    unpinned = tuple(
        f"{r.era} file {Path(r.path).name} has no producer pin (pe_pin_status none)"
        for r in records
        if r.pe_pin_status == PIN_NONE
    )
    no_tick = tuple(
        f"{r.era} file {Path(r.path).name} has no metadata tick (tick_metadata absent; grid check only)"
        for r in records
        if r.tick_metadata == TICK_ABSENT
    )
    return unpinned + no_tick


_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def cached_sha256(path: Path) -> str:
    """sha256 of a file's bytes, cached per (path, size, mtime_ns) inside the process."""
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _HASH_CACHE:
        _HASH_CACHE[key] = file_sha256(path)
    return _HASH_CACHE[key]


def grid_violations(prices: np.ndarray, fixed: np.ndarray, tick_fixed: int) -> np.ndarray:
    """Both price guards: fixed off the tick grid, or |fixed / 1e9 - price| > 1e-6."""
    return (fixed % tick_fixed != 0) | (np.abs(fixed / PRICE_SCALE - prices) > GRID_TOLERANCE)


def to_ticks(prices: np.ndarray, tick_fixed: int, root: str, column: str) -> np.ndarray:
    """fixed = rint(price * 1e9); off grid raises OffGridPrice (no snapping); returns fixed // tick_fixed."""
    if not np.all(np.isfinite(prices)) or np.any(np.abs(prices) > MAX_ABS_PRICE):
        raise OffGridPrice(f"{root} {column}: a non-finite or out-of-range price")
    fixed = np.rint(prices * PRICE_SCALE).astype(np.int64)
    off = grid_violations(prices, fixed, tick_fixed)
    if np.any(off):
        first_row = int(np.argmax(off))
        raise OffGridPrice(
            f"{root} {column}: {int(off.sum())} price(s) off the vendor tick grid "
            f"(tick_fixed {tick_fixed}; first at returned row {first_row})"
        )
    return fixed // tick_fixed


def metadata_tick_fixed(path: Path) -> int | None:
    """The store's exchange tick (tick.tick_fixed_1e9) from the parquet footer; None when absent."""
    meta = pq.read_schema(path).metadata or {}
    raw = meta.get(STORE_METADATA_KEY)
    if raw is None:
        return None
    try:
        tick = json.loads(raw).get("tick")
    except (json.JSONDecodeError, AttributeError) as exc:
        raise StoreRefused(f"{path}: store metadata is not a JSON object") from exc
    if not isinstance(tick, dict) or "tick_fixed_1e9" not in tick:
        return None
    value = tick["tick_fixed_1e9"]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TickMismatch(f"{path}: metadata tick_fixed_1e9 is not an integer")
    return value


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


class DataGateway:
    """Reads bars of one root over a trade-date range, refusing every closed or suspect row."""

    def __init__(
        self,
        paths: StorePaths,
        windows: WindowTable,
        catalog: Catalog,
        tick_fixed_of: Callable[[str], int],
        price_factor_of: Callable[[str], int],
    ) -> None:
        """Accepts only the pinned closed-window table (by content) and a catalog built against it.

        The gateway then checks with its OWN freshly loaded table, never the caller's object.
        """
        pinned = pinned_window_table(windows)
        self._setup(paths, pinned, check_catalog_windows(catalog, CLOSED_WINDOWS_SHA256), tick_fixed_of,
                    price_factor_of)  # fmt: skip

    @classmethod
    def for_synthetic_stores(
        cls,
        paths: StorePaths,
        windows: WindowTable,
        catalog: Catalog,
        tick_fixed_of: Callable[[str], int],
        price_factor_of: Callable[[str], int],
        *,
        paths_toml: Path = DEFAULT_PATHS_TOML,
    ) -> DataGateway:
        """TEST FACTORY: any window table, but every store root must lie away from the real store locations
        (the roots of config/paths.toml when it exists); the catalog must be built against that table."""
        refuse_real_store_roots(paths, paths_toml)
        if not isinstance(windows, WindowTable):
            raise WindowTableMismatch(f"windows is a {type(windows).__name__}, not a WindowTable")
        gateway = cls.__new__(cls)
        gateway._setup(paths, windows, check_catalog_windows(catalog, windows.sha256), tick_fixed_of,
                       price_factor_of)  # fmt: skip
        return gateway

    def _setup(
        self,
        paths: StorePaths,
        windows: WindowTable,
        catalog: Catalog,
        tick_fixed_of: Callable[[str], int],
        price_factor_of: Callable[[str], int],
    ) -> None:
        self._paths = paths
        self._windows = windows
        self._catalog = catalog
        self._tick_fixed_of = tick_fixed_of
        self._price_factor_of = price_factor_of
        self._provenance: tuple[ProvenanceRecord, ...] = ()

    @property
    def catalog(self) -> Catalog:
        """The catalog this gateway reads from."""
        return self._catalog

    @property
    def provenance(self) -> tuple[ProvenanceRecord, ...]:
        """Per file of the last successful read: path, era, sha256, rows returned."""
        return self._provenance

    def files_for(self, root: str, first: date, last: date) -> tuple[CatalogEntry, ...]:
        """Exactly the entries bars() reads, in read order (era, first trade date, path); no file is opened.

        Raises what bars() raises before opening a file: a bad request, a request touching a closed label
        range, no covering file, an unsupported era, or a producer pin mismatch.
        """
        self._check_dates(root, first, last)
        refusal = self._windows.request_refusal(root, first, last)
        if refusal is not None:
            raise refusal
        entries = self._select(root, first, last)
        for entry in entries:
            _check_entry_admissible(entry)
        return entries

    def bars(self, root: str, first: date, last: date, session: Session = ALL) -> pl.DataFrame:
        self._provenance = ()
        _check_session(session)
        entries = self.files_for(root, first, last)
        tick_fixed = self._tick_fixed(root)
        exchange_factor = self._price_factor(root)
        checked = [self._check_file(entry, root, tick_fixed, exchange_factor) for entry in entries]
        frame = pl.concat(
            [self._scan(path, entry, idx, first, last) for idx, (path, entry, _) in enumerate(checked)]
        )
        self._check_rows(root, first, last, frame)
        frame = self._with_ticks(frame, tick_fixed, root).with_columns(ct_minute_expr())
        if isinstance(session, ClockWindow):
            frame = frame.filter(session.expr())
        self._provenance = self._record(frame, entries, tuple(tick for _, _, tick in checked))
        return frame.select(OUTPUT_COLUMNS)

    # ---------------------------------------------------------------- steps ----
    def _check_dates(self, root: str, first: date, last: date) -> None:
        for name, value in (("first", first), ("last", last)):
            if not isinstance(value, date) or isinstance(value, datetime):
                raise BadRequest(f"{name} must be a datetime.date, got {type(value).__name__}")
        if root not in self._catalog.roots():
            raise BadRequest(f"unknown root {root!r}: no catalogued store file")
        if first > last:
            raise BadRequest(f"{root}: first {first} is after last {last}")

    def _select(self, root: str, first: date, last: date) -> tuple[CatalogEntry, ...]:
        entries = [e for e in self._catalog.files_for(root) if e.overlaps(first, last)]
        if not entries:
            raise NoBarsFound(f"{root}: no store file covers {first}..{last}")
        return tuple(sorted(entries, key=lambda e: (ERA_ORDER[e.era], e.first_trade_date or "", e.path)))

    def _tick_fixed(self, root: str) -> int:
        value = self._tick_fixed_of(root)
        if not _positive_int(value):
            raise TickMismatch(f"{root}: tick_fixed_of returned {value!r}, not a positive int")
        return value

    def _price_factor(self, root: str) -> int:
        value = self._price_factor_of(root)
        if not _positive_int(value):
            raise TickMismatch(f"{root}: price_factor_of returned {value!r}, not a positive int")
        return value

    def _check_file(
        self, entry: CatalogEntry, root: str, tick_fixed: int, price_factor: int
    ) -> tuple[Path, CatalogEntry, str]:
        file_id = check_path(Path(entry.path), self._paths)
        if (file_id.root, file_id.era, file_id.plan) != (root, entry.era, entry.plan) or entry.root != root:
            raise PathRefused(
                f"{entry.path}: path says {file_id.root}/{file_id.era}, catalog says {entry.root}"
            )
        actual = cached_sha256(file_id.resolved)
        if actual != entry.sha256:
            raise HashMismatch(f"{entry.path}: sha256 {actual} != catalog {entry.sha256}")
        stored_tick = metadata_tick_fixed(file_id.resolved)
        if stored_tick is None:
            if (entry.root, entry.era) not in TICK_ABSENT_ALLOWED:
                raise TickMismatch(
                    f"{entry.path}: no metadata tick; only {sorted(TICK_ABSENT_ALLOWED)} may lack one (A14 h)"
                )
            return file_id.resolved, entry, TICK_ABSENT
        if stored_tick * price_factor != tick_fixed:
            raise TickMismatch(
                f"{root}: vendor tick {tick_fixed} is not store metadata tick {stored_tick} "
                f"times the vendor price factor {price_factor}"
            )
        return file_id.resolved, entry, TICK_EXACT

    def _scan(self, path: Path, entry: CatalogEntry, idx: int, first: date, last: date) -> pl.DataFrame:
        lazy = pl.scan_parquet(path)
        schema = dict(lazy.collect_schema())
        if schema != STORED_SCHEMA:
            raise StoreRefused(f"{entry.path}: schema differs from the 16-column bar schema")
        label = pl.col(_LABEL)
        picked = label.is_null() | label.is_between(pl.lit(first.isoformat()), pl.lit(last.isoformat()))
        frame = lazy.filter(picked).collect()
        if frame[_LABEL].null_count() or frame["ts_event"].null_count():
            raise StoreRefused(f"{entry.path}: a null trade_date label or ts_event")
        try:
            frame = frame.with_columns(label.str.to_date("%Y-%m-%d", strict=True))
        except pl.exceptions.PolarsError as exc:
            raise StoreRefused(f"{entry.path}: a trade_date label is not YYYY-MM-DD") from exc
        check_ts_range(frame, entry, "ts_event")
        return frame.with_columns(
            pl.lit(entry.era).alias("era"), pl.lit(idx, dtype=pl.Int32).alias(_FILE_IDX)
        )

    def _check_rows(self, root: str, first: date, last: date, frame: pl.DataFrame) -> None:
        label, ts = pl.col(_LABEL), pl.col("ts_event")
        if frame.select((~label.is_between(pl.lit(first), pl.lit(last))).sum()).item():
            raise BoundBreached(f"{root}: a returned label lies outside {first}..{last}")
        bound = bound_ns(last)
        if frame.select((ts >= pl.lit(bound, dtype=pl.Int64)).sum()).item():
            raise BoundBreached(f"{root}: a row stamped at or after 17:00 CT on {last}")
        hit = self._windows.first_closed_row(root, frame, _LABEL, "ts_event")
        if hit is not None:
            raise ClosedWindowRefused(
                hit[0], f"{root}: {hit[1]} returned row(s) inside closed window {hit[0]}"
            )
        if frame.height > 1 and not frame.select((ts.diff().drop_nulls() > 0).all()).item():
            raise StoreRefused(f"{root}: ts_event is not strictly increasing")
        shared = frame.group_by(_LABEL).agg(pl.col(_FILE_IDX).n_unique().alias("n")).filter(pl.col("n") > 1)
        if shared.height:
            raise StoreRefused(f"{root}: {shared.height} trade date(s) present in two store files")

    def _with_ticks(self, frame: pl.DataFrame, tick_fixed: int, root: str) -> pl.DataFrame:
        for column in PRICE_COLUMNS:
            if frame[column].null_count():
                raise OffGridPrice(f"{root} {column}: a null price")
        ticks = {c: to_ticks(frame[c].to_numpy(), tick_fixed, root, c) for c in PRICE_COLUMNS}
        return frame.with_columns(pl.Series(c, ticks[c], dtype=pl.Int64) for c in PRICE_COLUMNS)

    def _record(
        self, frame: pl.DataFrame, entries: tuple[CatalogEntry, ...], ticks: tuple[str, ...]
    ) -> tuple[ProvenanceRecord, ...]:
        counts = dict(frame.group_by(_FILE_IDX).len().iter_rows())
        return tuple(
            ProvenanceRecord(
                path=e.path,
                era=e.era,
                sha256=e.sha256,
                rows=int(counts.get(i, 0)),
                pe_pin_status=check_pin_status(e),
                tick_metadata=tick,
            )
            for i, (e, tick) in enumerate(zip(entries, ticks, strict=True))
        )


def _check_session(session: object) -> None:
    """ADR A5 g: a plain CT clock filter. SPEC names ("rth", "globex") are resolved in pfb.engine.run."""
    if not isinstance(session, AllSessions | ClockWindow):
        raise BadRequest(f"session must be ALL or a ClockWindow, got {type(session).__name__}")


def _check_entry_admissible(entry: CatalogEntry) -> None:
    """ADR A5 a and b, decided from the catalog alone; the pin status is recomputed."""
    if entry.era in UNSUPPORTED_ERAS:
        raise EraNotSupported(f"{entry.root}: era {entry.era} is not read in this version ({entry.path})")
    status = check_pin_status(entry)  # CatalogError when the stored status disagrees with pe_pin and sha256
    if status == PIN_MISMATCH:
        raise PinMismatch(
            f"{entry.path}: sha256 {entry.sha256} differs from the producer pin {entry.pe_pin}"
        )
    if status not in READABLE_PIN_STATUSES:
        raise CatalogError(f"{entry.path}: pe_pin_status {status!r} is not match or none")


def ct_minute_expr() -> pl.Expr:
    ct = (
        pl.from_epoch(pl.col("ts_event"), time_unit="ns")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone(str(CT))
    )
    return (
        (ct.dt.hour().cast(pl.Int16) * 60 + ct.dt.minute().cast(pl.Int16)).cast(pl.Int16).alias("ct_minute")
    )
