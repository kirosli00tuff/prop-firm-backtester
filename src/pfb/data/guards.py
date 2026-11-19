"""Fail-closed guards the DataGateway applies at construction and per store file.

- The public constructor binds the PINNED closed-window table: it reloads config/closed_windows.toml (hash
  checked) and compares content, never trusting the caller's table or its sha256 field (G1).
- A synthetic window table is only accepted through the explicit test factory, which refuses store roots that
  overlap a real store location: the roots of config/paths.toml when it exists (G1).
- A catalog built against another window table is refused (G2 i).
- Timestamps must be UTC nanoseconds inside the file's label range, with a margin of a few days (G4).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from pfb.data.catalog import Catalog, CatalogEntry, parse_label
from pfb.data.errors import CatalogError, PathRefused, StoreRefused, WindowTableMismatch
from pfb.data.paths import StorePaths, load_paths
from pfb.data.windows import CLOSED_WINDOWS_SHA256, NS_PER_SECOND, WindowTable, datetime_to_ns, load_windows

CT = ZoneInfo("America/Chicago")
BOUND_CT = time(17, 0)
TS_MARGIN = timedelta(days=4)
TS_MARGIN_NS = TS_MARGIN.days * 86_400 * NS_PER_SECOND
_ONE_DAY = timedelta(days=1)
# Any bar stamped in UTC nanoseconds lies in this span; microseconds or milliseconds of today land in 1970.
EARLIEST_NS = datetime_to_ns(datetime(2000, 1, 1, tzinfo=UTC))
LATEST_NS = datetime_to_ns(datetime(2100, 1, 1, tzinfo=UTC))
CATALOG_HINT = "run `pfb data catalog` to rebuild runs/catalog.json"


def bound_ns(last: date) -> int:
    """17:00 America/Chicago on the calendar day `last`, as integer UTC nanoseconds."""
    return datetime_to_ns(datetime.combine(last, BOUND_CT, tzinfo=CT))


# ---------------------------------------------------------------- G1: window table ----
def pinned_window_table(windows: object) -> WindowTable:
    """The gateway's own freshly loaded pinned table, once the caller's table equals it in content."""
    pinned = load_windows()
    if not isinstance(windows, WindowTable):
        raise WindowTableMismatch(f"windows is a {type(windows).__name__}, not a WindowTable")
    if windows.windows != pinned.windows or windows.sha256 != CLOSED_WINDOWS_SHA256:
        raise WindowTableMismatch(
            "the window table is not the pinned closed-window table (content differs from "
            f"config/closed_windows.toml at sha256 {CLOSED_WINDOWS_SHA256}); a synthetic table needs "
            "DataGateway.for_synthetic_stores"
        )
    return pinned


def real_store_locations(paths_toml: Path) -> tuple[Path, ...]:
    """When the paths file exists: its store roots and its upstream root (if any); else nothing."""
    if not (paths_toml.exists() or paths_toml.is_symlink()):
        return ()
    real = load_paths(paths_toml)  # a present but broken file raises: fail closed
    upstream = () if real.upstream_root is None else (real.upstream_root,)
    return (*real.store_roots().values(), *upstream)


def refuse_real_store_roots(paths: object, paths_toml: Path) -> StorePaths:
    """Refuse synthetic store roots inside (or containing) a real store location. Opens no store file."""
    if not isinstance(paths, StorePaths):
        raise PathRefused(f"paths is a {type(paths).__name__}, not StorePaths")
    real = [location.resolve() for location in real_store_locations(paths_toml)]
    for name, root in paths.store_roots().items():
        mine = Path(root).resolve()
        if any(mine.is_relative_to(r) or r.is_relative_to(mine) for r in real):
            raise PathRefused(
                f"synthetic {name} store root overlaps a real store location; the test factory reads "
                "synthetic stores only"
            )
    return paths


# ---------------------------------------------------------------- G2 (i): catalog ----
def check_catalog_windows(catalog: object, expected_sha256: str) -> Catalog:
    if not isinstance(catalog, Catalog):
        raise CatalogError(f"catalog is a {type(catalog).__name__}, not a Catalog")
    if catalog.closed_windows_sha256 != expected_sha256:
        raise CatalogError(
            f"the catalog was built against closed-window table {catalog.closed_windows_sha256!r}, "
            f"not the pinned {expected_sha256}; {CATALOG_HINT}"
        )
    return catalog


# ---------------------------------------------------------------- G4: time unit and range ----
def check_ts_range(frame: pl.DataFrame, entry: CatalogEntry, ts_col: str) -> None:
    """Every ts is UTC nanoseconds and lies in [17:00 CT of first label - 1 day, 17:00 CT of last label],
    widened by TS_MARGIN on both sides; first and last are the file's catalogued label range."""
    if frame.height == 0:
        return
    low, high = frame.select(pl.col(ts_col).min().alias("low"), pl.col(ts_col).max().alias("high")).row(0)
    if low < EARLIEST_NS or high >= LATEST_NS:
        raise StoreRefused(f"{entry.path}: {ts_col} is not UTC nanoseconds of 2000..2099 (unit slip?)")
    if entry.first_trade_date is None or entry.last_trade_date is None:
        raise StoreRefused(f"{entry.path}: the catalog has no label range for this file")
    first = parse_label(entry.first_trade_date, entry.path)
    last = parse_label(entry.last_trade_date, entry.path)
    lower = bound_ns(first - _ONE_DAY) - TS_MARGIN_NS
    upper = bound_ns(last) + TS_MARGIN_NS
    if low < lower or high > upper:
        raise StoreRefused(
            f"{entry.path}: a {ts_col} lies outside the file's label range {first}..{last} "
            f"(margin {TS_MARGIN.days} days)"
        )
