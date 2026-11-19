"""The synthetic demo store: seeded bars written in the layout the DataGateway reads. No market data.

Each demo root gets one research-era file under data/demo/research/<ROOT>/ (research_file_name) holding
pfb.engine.synthetic bars over DEMO_FIRST..DEMO_LAST in the stored schema (vendor float prices, the trade
date as text, the exchange tick in the footer metadata). The producer's sha256 of every file goes to
data/demo/pins.json, and the catalog is built with those pins, so a read through the public
DataGateway constructor passes every guard: path rule, sha256, pin, tick, closed windows. The demo range
reaches into the embargo of config/closed_windows.toml, so the store holds rows that the gateway refuses.

The bytes are deterministic for a given seed; the store is rebuilt only when a pinned file is missing or
differs from its pin.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from pfb.data.catalog import (
    Catalog,
    PinKey,
    build_catalog,
    file_sha256,
    load_pins,
    pins_json,
    write_catalog,
)
from pfb.data.errors import CatalogError
from pfb.data.gateway import PRICE_COLUMNS, STORE_METADATA_KEY, STORED_SCHEMA
from pfb.data.paths import (
    DEMO_STORE_ROOT,
    ERA_RESEARCH,
    RESEARCH_FIRST,
    RESEARCH_LAST,
    StorePaths,
    demo_paths,
)
from pfb.engine.synthetic import synthetic_frame
from pfb.instruments import InstrumentTable, load_instruments

DEMO_ROOTS = ("MNQ", "NQ")
DEMO_FIRST = date.fromisoformat(RESEARCH_FIRST)
DEMO_LAST = date.fromisoformat(RESEARCH_LAST)
DEMO_SEED = 20250303
PRICE_SCALE = 1_000_000_000
CREATED_UTC = "1970-01-01T00:00:00Z"  # fixed, so a rebuilt catalog is byte-identical


def research_file_name(root: str) -> str:
    """The research-era file name the path rule accepts (pfb.data.paths)."""
    return f"ohlcv-1m_{root}_v_0_{RESEARCH_FIRST}_{RESEARCH_LAST}_research.parquet"


@dataclass(frozen=True)
class DemoStore:
    """The demo store's paths, its producer pins and whether this call built it."""

    paths: StorePaths
    pins: dict[PinKey, str]
    built: bool


def stored_frame(
    root: str, first: date, last: date, *, seed: int, instruments: InstrumentTable
) -> pl.DataFrame:
    """Synthetic bars of ``root`` in the 16-column stored schema: vendor float prices on the vendor tick grid
    (rint(price x 1e9) is exactly ticks x vendor tick), the trade date as YYYY-MM-DD text."""
    bars = synthetic_frame(root, first, last, seed=seed)
    tick_fixed = instruments.tick_fixed_of(root)
    prices = [
        (pl.col(c).cast(pl.Int64) * tick_fixed).cast(pl.Float64).truediv(PRICE_SCALE).alias(c)
        for c in PRICE_COLUMNS
    ]
    frame = bars.with_columns(*prices, pl.col("trade_date").dt.strftime("%Y-%m-%d").alias("trade_date"))
    return frame.select([pl.col(name).cast(dtype) for name, dtype in STORED_SCHEMA.items()])


def write_store_file(path: Path, frame: pl.DataFrame, exchange_tick_fixed: int) -> str:
    """Write one store file with the exchange tick in its footer; returns the file's sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = frame.to_arrow()
    meta = {STORE_METADATA_KEY: json.dumps({"tick": {"tick_fixed_1e9": exchange_tick_fixed}}).encode()}
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **meta})
    pq.write_table(table, path, compression="zstd")
    return file_sha256(path)


def _pins_current(paths: StorePaths, roots: tuple[str, ...]) -> dict[PinKey, str] | None:
    """The pins when every demo file exists and hashes to its pin, else None."""
    if paths.pins_json is None or not paths.pins_json.is_file():
        return None
    try:
        pins = load_pins(paths.pins_json)
    except CatalogError:
        return None
    for root in roots:
        sha = pins.get((root, ERA_RESEARCH, None))
        path = paths.research / root / research_file_name(root)
        if sha is None or not path.is_file() or file_sha256(path) != sha:
            return None
    return pins


def build_demo_store(
    root_dir: Path = DEMO_STORE_ROOT,
    *,
    roots: tuple[str, ...] = DEMO_ROOTS,
    first: date = DEMO_FIRST,
    last: date = DEMO_LAST,
    seed: int = DEMO_SEED,
    instruments: InstrumentTable | None = None,
    force: bool = False,
) -> DemoStore:
    """Write (or reuse) the demo store under ``root_dir`` and its pins file."""
    paths = demo_paths(root_dir)
    for store_root in paths.store_roots().values():
        store_root.mkdir(parents=True, exist_ok=True)
    current = None if force else _pins_current(paths, roots)
    if current is not None:
        return DemoStore(paths, current, built=False)
    table = instruments or load_instruments()
    pins: dict[PinKey, str] = {}
    for offset, root in enumerate(roots):
        instrument = table.get(root)
        frame = stored_frame(root, first, last, seed=seed + offset, instruments=table)
        exchange_tick = instrument.vendor_tick_fixed // instrument.vendor_price_factor
        path = paths.research / root / research_file_name(root)
        pins[(root, ERA_RESEARCH, None)] = write_store_file(path, frame, exchange_tick)
    assert paths.pins_json is not None  # demo_paths always names one
    paths.pins_json.write_text(json.dumps(pins_json(pins), indent=1) + "\n", encoding="utf-8")
    return DemoStore(paths, pins, built=True)


def demo_catalog(store: DemoStore, windows_sha256: str, out: Path | None = None) -> Catalog:
    """The catalog of the demo store, built with its pins; written to ``out`` when given."""
    catalog = build_catalog(store.paths, windows_sha256, CREATED_UTC, pins=store.pins)
    if out is not None:
        write_catalog(catalog, out)
    return catalog


__all__ = [
    "DEMO_FIRST",
    "DEMO_LAST",
    "DEMO_ROOTS",
    "DEMO_SEED",
    "DemoStore",
    "build_demo_store",
    "demo_catalog",
    "research_file_name",
    "stored_frame",
    "write_store_file",
]
