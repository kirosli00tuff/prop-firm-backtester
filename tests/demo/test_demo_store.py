"""The synthetic bar generator and the demo store: deterministic bytes, the stored schema, and reads through
the public DataGateway constructor with the pinned closed-window table."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from pfb.data.catalog import PIN_MATCH, file_sha256, load_pins
from pfb.data.demo import build_demo_store, demo_catalog, research_file_name, stored_frame
from pfb.data.errors import ClosedWindowRefused, HashMismatch
from pfb.data.gateway import STORED_SCHEMA, DataGateway, provenance_warnings
from pfb.data.paths import DEFAULT_PATHS_TOML, demo_paths, load_paths
from pfb.data.windows import load_windows
from pfb.engine.synthetic import synthetic_frame
from pfb.instruments import load_instruments

FIRST, LAST = date(2026, 4, 1), date(2026, 5, 15)
ROOTS = ("MNQ",)


def _store(base: Path, **kw: object):  # noqa: ANN202 - the DemoStore of build_demo_store
    return build_demo_store(base, roots=ROOTS, first=FIRST, last=LAST, **kw)  # type: ignore[arg-type]


def test_synthetic_frame_is_the_same_for_a_seed_and_differs_across_seeds() -> None:
    a = synthetic_frame("MNQ", date(2025, 4, 1), date(2025, 4, 10), seed=3)
    b = synthetic_frame("MNQ", date(2025, 4, 1), date(2025, 4, 10), seed=3)
    c = synthetic_frame("MNQ", date(2025, 4, 1), date(2025, 4, 10), seed=4)

    assert a.equals(b)
    assert not a.equals(c)
    assert a["ts_event"].is_sorted() and a["ts_event"].n_unique() == a.height


def test_stored_frame_has_the_store_schema_and_prices_on_the_vendor_grid() -> None:
    instruments = load_instruments()
    bars = synthetic_frame("MNQ", FIRST, FIRST, seed=1)

    stored = stored_frame("MNQ", FIRST, FIRST, seed=1, instruments=instruments)

    assert dict(stored.schema) == STORED_SCHEMA
    tick = instruments.tick_fixed_of("MNQ")
    fixed = (stored["close"] * 1_000_000_000).round().cast(pl.Int64)
    assert (fixed // tick).to_list() == bars["close"].to_list()
    assert (fixed % tick == 0).all()
    assert stored["trade_date"].to_list() == [d.isoformat() for d in bars["trade_date"].to_list()]


def test_the_demo_store_is_byte_identical_when_rebuilt(tmp_path: Path) -> None:
    one = _store(tmp_path / "one")
    two = _store(tmp_path / "two")

    rel = Path("research") / "MNQ" / research_file_name("MNQ")
    assert one.built and two.built
    assert file_sha256(tmp_path / "one" / rel) == file_sha256(tmp_path / "two" / rel)
    assert one.pins == two.pins
    assert load_pins(tmp_path / "one" / "pins.json") == one.pins


def test_an_unchanged_store_is_reused_and_a_changed_file_is_rebuilt(tmp_path: Path) -> None:
    first = _store(tmp_path)
    again = _store(tmp_path)
    path = tmp_path / "research" / "MNQ" / research_file_name("MNQ")
    path.write_bytes(path.read_bytes()[:-1])

    rebuilt = _store(tmp_path)

    assert first.built and not again.built and rebuilt.built
    assert file_sha256(path) == first.pins[("MNQ", "research", None)]


def test_the_public_gateway_reads_the_open_range_and_refuses_the_closed_one(tmp_path: Path) -> None:
    store = _store(tmp_path)
    windows = load_windows()
    instruments = load_instruments()
    catalog = demo_catalog(store, windows.sha256, tmp_path / "catalog.json")
    gateway = DataGateway(
        store.paths, windows, catalog, instruments.tick_fixed_of, instruments.price_factor_of
    )

    frame = gateway.bars("MNQ", date(2026, 4, 27), date(2026, 5, 8))

    assert frame.height > 0 and frame["era"].unique().to_list() == ["research"]
    assert [p.pe_pin_status for p in gateway.provenance] == [PIN_MATCH]
    assert provenance_warnings(gateway.provenance) == ()
    with pytest.raises(ClosedWindowRefused, match="demo_embargo_tail"):
        gateway.bars("MNQ", date(2026, 5, 4), date(2026, 5, 15))


def test_a_demo_file_changed_after_cataloguing_is_refused(tmp_path: Path) -> None:
    store = _store(tmp_path)
    windows = load_windows()
    instruments = load_instruments()
    catalog = demo_catalog(store, windows.sha256)
    path = tmp_path / "research" / "MNQ" / research_file_name("MNQ")
    path.write_bytes(path.read_bytes() + b"\0")
    gateway = DataGateway(
        store.paths, windows, catalog, instruments.tick_fixed_of, instruments.price_factor_of
    )

    with pytest.raises(HashMismatch):
        gateway.bars("MNQ", date(2026, 4, 27), date(2026, 5, 8))


@pytest.mark.skipif(DEFAULT_PATHS_TOML.is_file(), reason="a local config/paths.toml names other stores")
def test_without_a_paths_file_the_default_paths_are_the_demo_store() -> None:
    assert load_paths() == demo_paths()
    assert load_paths().upstream_root is None
