"""Fail-closed hardening of the DataGateway (construction guards, pin status, tick metadata).

Synthetic stores under tmp_path only; the pinned closed-window table is read from config (no bar).
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, time
from pathlib import Path

import pytest
from gateway_synth import PRICE_FACTORS, TICKS_FIXED, Row, SynthStores, ct_ns, session_rows

from pfb.data.catalog import PIN_MATCH, PIN_MISMATCH, PIN_NONE, Catalog, load_catalog, write_catalog
from pfb.data.errors import (
    CatalogError,
    ClosedWindowRefused,
    GatewayRefusal,
    PathRefused,
    PathsConfigError,
    StoreRefused,
    TickMismatch,
    WindowTableMismatch,
)
from pfb.data.gateway import TICK_ABSENT, DataGateway
from pfb.data.windows import CLOSED_WINDOWS_SHA256, WindowTable, datetime_to_ns, load_windows

JAN4, JAN8, JAN29 = date(2021, 1, 4), date(2021, 1, 8), date(2021, 1, 29)
EMBARGO_DAY, HOLDOUT_DAY = date(2026, 5, 13), date(2026, 6, 1)  # demo_embargo_tail, demo_holdout_tail
OPEN_DAY = date(2026, 5, 8)  # the Friday before demo_embargo_tail: open by label


def _z(text: str) -> int:
    return datetime_to_ns(datetime.fromisoformat(text).replace(tzinfo=UTC))


def _public(stores: SynthStores, windows: WindowTable, catalog: Catalog) -> DataGateway:
    return DataGateway(stores.paths, windows, catalog, TICKS_FIXED.__getitem__, PRICE_FACTORS.__getitem__)


@pytest.fixture
def closed_rows(stores: SynthStores) -> SynthStores:
    """An NQ file holding one embargo row and one holdout row."""
    stores.research("NQ", [Row(_z("2026-05-13T15:00:00"), "2026-05-13", 40_000),
                           Row(_z("2026-06-01T15:00:00"), "2026-06-01", 40_001)])  # fmt: skip
    return stores


# ---------------------------------------------------------------- G1: the pin binds the gateway ----
def test_g1_the_pinned_table_refuses_the_embargo_and_holdout_rows(closed_rows: SynthStores) -> None:
    gateway = _public(closed_rows, load_windows(), closed_rows.catalog(windows_sha256=CLOSED_WINDOWS_SHA256))

    with pytest.raises(ClosedWindowRefused):
        gateway.bars("NQ", EMBARGO_DAY, HOLDOUT_DAY)


def test_g1_an_empty_window_table_is_refused_by_the_constructor(closed_rows: SynthStores) -> None:
    catalog = closed_rows.catalog(windows_sha256=CLOSED_WINDOWS_SHA256)

    with pytest.raises(WindowTableMismatch, match="pinned"):
        _public(closed_rows, WindowTable(windows=(), sha256="not-the-pin"), catalog)
    with pytest.raises(WindowTableMismatch, match="pinned"):
        _public(closed_rows, WindowTable(windows=(), sha256=CLOSED_WINDOWS_SHA256), catalog)


def test_g1_another_table_carrying_the_pinned_sha256_field_is_refused(closed_rows: SynthStores) -> None:
    forged = WindowTable(windows=closed_rows.windows.windows, sha256=CLOSED_WINDOWS_SHA256)

    with pytest.raises(WindowTableMismatch):
        _public(closed_rows, forged, closed_rows.catalog(windows_sha256=CLOSED_WINDOWS_SHA256))


def test_g1_a_pinned_table_dropping_one_window_is_refused(closed_rows: SynthStores) -> None:
    pinned = load_windows()
    short = WindowTable(windows=pinned.windows[1:], sha256=pinned.sha256)

    with pytest.raises(WindowTableMismatch):
        _public(closed_rows, short, closed_rows.catalog(windows_sha256=CLOSED_WINDOWS_SHA256))


class _BlindTable(WindowTable):
    """Same content as the pinned table, but its checks never refuse."""

    def request_refusal(self, root: str, first: date, last: date) -> ClosedWindowRefused | None:
        return None

    def first_closed_row(self, root: str, frame: object, label_col: str, ts_col: str) -> None:  # type: ignore[override]
        return None


def test_g1_the_gateway_checks_with_its_own_loaded_table_not_the_callers_object(
    closed_rows: SynthStores,
) -> None:
    pinned = load_windows()
    blind = _BlindTable(windows=pinned.windows, sha256=pinned.sha256)
    gateway = _public(closed_rows, blind, closed_rows.catalog(windows_sha256=CLOSED_WINDOWS_SHA256))

    with pytest.raises(ClosedWindowRefused):
        gateway.bars("NQ", EMBARGO_DAY, HOLDOUT_DAY)


def _paths_toml(tmp_path: Path, research: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    toml = tmp_path / "paths.toml"
    toml.write_text(
        f'[stores]\nresearch = "{research}"\nstep2 = "{tmp_path / "s2"}"\nhist = "{tmp_path / "h"}"\n'
        f'[upstream]\nroot = "{tmp_path / "pe_real"}"\n[costs]\nbook_table = "{tmp_path / "t.json"}"\n',
        encoding="utf-8",
    )
    return toml


@pytest.mark.parametrize("inside", [True, False])
def test_g1_the_test_factory_refuses_roots_overlapping_a_configured_store_root(
    tmp_path: Path, inside: bool
) -> None:
    if inside:  # the synthetic roots lie inside the configured research root
        real_research = tmp_path / "real"
        synth = SynthStores(real_research / "synthetic")
    else:  # the synthetic research root contains the configured one
        synth = SynthStores(tmp_path / "synth")
        real_research = synth.paths.research / "nested_real"
    toml = _paths_toml(tmp_path / "cfg", real_research)

    with pytest.raises(PathRefused, match="real store location"):
        DataGateway.for_synthetic_stores(
            synth.paths, synth.windows, synth.catalog(), TICKS_FIXED.__getitem__, PRICE_FACTORS.__getitem__,
            paths_toml=toml,
        )  # fmt: skip


def test_g1_the_test_factory_refuses_when_the_paths_file_is_malformed(
    stores: SynthStores, tmp_path: Path
) -> None:
    toml = tmp_path / "bad_paths.toml"
    toml.write_text("this is = = not toml", encoding="utf-8")

    catalog = stores.catalog()

    with pytest.raises(PathsConfigError):
        DataGateway.for_synthetic_stores(
            stores.paths, stores.windows, catalog, TICKS_FIXED.__getitem__, PRICE_FACTORS.__getitem__,
            paths_toml=toml,
        )  # fmt: skip


def test_g1_the_test_factory_reads_synthetic_stores_away_from_real_roots(
    stores: SynthStores, tmp_path: Path
) -> None:
    stores.research("NQ", session_rows(JAN4, JAN8))
    toml = _paths_toml(tmp_path / "cfg", tmp_path / "elsewhere")

    gateway = DataGateway.for_synthetic_stores(
        stores.paths, stores.windows, stores.catalog(), TICKS_FIXED.__getitem__, PRICE_FACTORS.__getitem__,
        paths_toml=toml,
    )  # fmt: skip

    assert gateway.bars("NQ", JAN4, JAN8).height == 15


# ---------------------------------------------------------------- G2: catalog trust ----
def test_g2i_a_catalog_built_against_another_window_table_is_refused(stores: SynthStores) -> None:
    stores.research("NQ", session_rows(JAN4, JAN8))
    stale = stores.catalog(windows_sha256="e" * 64)

    with pytest.raises(CatalogError, match="pfb data catalog"):
        _public(stores, load_windows(), stale)
    with pytest.raises(CatalogError, match="pfb data catalog"):
        stores.gateway(stale)


def test_g2i_a_catalog_built_against_the_pin_is_accepted_by_the_constructor(stores: SynthStores) -> None:
    stores.research("NQ", session_rows(JAN4, JAN8))

    gateway = _public(stores, load_windows(), stores.catalog(windows_sha256=CLOSED_WINDOWS_SHA256))

    assert gateway.bars("NQ", JAN4, JAN8).height == 15


def test_g2ii_a_status_claiming_match_against_a_different_pin_is_refused(stores: SynthStores) -> None:
    stores.research("NQ", session_rows(JAN4, JAN8))
    catalog = stores.catalog()
    lying = replace(catalog.files[0], pe_pin="0" * 64, pe_pin_status=PIN_MATCH)
    gateway = stores.gateway(Catalog(catalog.created_utc, catalog.closed_windows_sha256, (lying,), ()))

    with pytest.raises(GatewayRefusal, match="pe_pin_status"):
        gateway.files_for("NQ", JAN4, JAN8)
    with pytest.raises(GatewayRefusal, match="pe_pin_status"):
        gateway.bars("NQ", JAN4, JAN8)


def test_g2ii_a_status_claiming_none_beside_a_mismatching_pin_is_refused(stores: SynthStores) -> None:
    stores.research("NQ", session_rows(JAN4, JAN8))
    catalog = stores.catalog()
    lying = replace(catalog.files[0], pe_pin="0" * 64, pe_pin_status=PIN_NONE)

    with pytest.raises(GatewayRefusal):
        stores.gateway(Catalog(catalog.created_utc, catalog.closed_windows_sha256, (lying,), ())).bars(
            "NQ", JAN4, JAN8
        )


def test_g2ii_load_catalog_refuses_a_pin_status_that_disagrees_with_pin_and_sha256(
    stores: SynthStores, tmp_path: Path
) -> None:
    stores.research("NQ", session_rows(JAN4, JAN8))
    out = tmp_path / "catalog.json"
    write_catalog(stores.catalog(pins={("NQ", "research", None): "0" * 64}), out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["files"][0]["pe_pin_status"] == PIN_MISMATCH
    data["files"][0]["pe_pin_status"] = PIN_MATCH
    out.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(CatalogError, match="pe_pin_status"):
        load_catalog(out)


@pytest.mark.parametrize("label", ["2025-5-6", "2025-05-6", "20210106", "2021-W01-3", "not a date"])
def test_g2iii_a_non_iso_catalog_label_is_a_gateway_refusal(stores: SynthStores, label: str) -> None:
    stores.research("NQ", session_rows(JAN4, JAN8))
    catalog = stores.catalog()
    odd = replace(catalog.files[0], last_trade_date=label)
    gateway = stores.gateway(Catalog(catalog.created_utc, catalog.closed_windows_sha256, (odd,), ()))

    with pytest.raises(CatalogError, match="ISO"):
        gateway.files_for("NQ", JAN4, JAN8)


def test_g2iii_a_store_label_without_zero_padding_is_a_gateway_refusal(stores: SynthStores) -> None:
    stores.research("NQ", [Row(ct_ns(JAN4, time(8, 30)), "2021-01-04", 40_000),
                           Row(ct_ns(JAN8, time(8, 30)), "2021-1-8", 40_001)])  # fmt: skip

    with pytest.raises(GatewayRefusal):
        stores.gateway().bars("NQ", JAN4, JAN8)


# ---------------------------------------------------------------- G3: absent tick allowlist ----
def test_g3_a_non_mes_file_without_a_metadata_tick_is_refused(stores: SynthStores) -> None:
    stores.research("NQ", session_rows(JAN4, JAN8), meta_tick_fixed=None)

    with pytest.raises(TickMismatch, match="no metadata tick"):
        stores.gateway().bars("NQ", JAN4, JAN8)


def test_g3_an_mes_step2_file_without_a_metadata_tick_is_refused(stores: SynthStores) -> None:
    stores.step2("MES", session_rows(JAN4, JAN8), meta_tick_fixed=None)

    with pytest.raises(TickMismatch, match="no metadata tick"):
        stores.gateway().bars("MES", JAN4, JAN8)


def test_g3_the_mes_research_file_without_a_metadata_tick_reads_flagged(stores: SynthStores) -> None:
    stores.research("MES", session_rows(JAN4, JAN8), meta_tick_fixed=None)
    gateway = stores.gateway()

    gateway.bars("MES", JAN4, JAN8)

    assert [p.tick_metadata for p in gateway.provenance] == [TICK_ABSENT]


# ---------------------------------------------------------------- G4: time unit and file range ----
def test_g4_probe7_a_microsecond_timestamp_under_an_open_label_is_refused(stores: SynthStores) -> None:
    pinned_catalog_sha = CLOSED_WINDOWS_SHA256
    stores.research("NQ", [Row(_z("2026-05-13T15:00:00") // 1000, OPEN_DAY.isoformat(), 40_000)])
    gateway = _public(stores, load_windows(), stores.catalog(windows_sha256=pinned_catalog_sha))

    with pytest.raises(StoreRefused, match="nanoseconds"):
        gateway.bars("NQ", OPEN_DAY, OPEN_DAY)


def test_g4_a_millisecond_timestamp_is_refused(stores: SynthStores) -> None:
    rows = session_rows(JAN4, JAN8)
    rows[0] = Row(rows[0].ts_ns // 1_000_000, rows[0].label, rows[0].price_ticks)
    stores.research("NQ", rows)

    with pytest.raises(StoreRefused, match="nanoseconds"):
        stores.gateway().bars("NQ", JAN4, JAN8)


def test_g4_a_timestamp_days_before_the_files_first_label_is_refused(stores: SynthStores) -> None:
    rows = session_rows(JAN4, JAN8)
    rows.insert(0, Row(_z("2020-12-24T15:00:00"), JAN4.isoformat(), 39_999))
    stores.research("NQ", rows)

    with pytest.raises(StoreRefused, match="label range"):
        stores.gateway().bars("NQ", JAN4, JAN8)


def test_g4_a_timestamp_days_after_the_files_last_label_is_refused(stores: SynthStores) -> None:
    rows = session_rows(JAN4, JAN8)
    rows.append(Row(ct_ns(date(2021, 1, 20), time(8, 30)), JAN8.isoformat(), 50_000))
    stores.research("NQ", rows)

    with pytest.raises(StoreRefused, match="label range"):
        stores.gateway().bars("NQ", JAN4, JAN29)


def test_g4_the_sunday_evening_bar_of_a_monday_label_is_inside_the_margin(stores: SynthStores) -> None:
    stores.research("NQ", session_rows(JAN4, JAN8))  # first bar: Sunday 2021-01-03 17:30 CT, label 01-04

    frame = stores.gateway().bars("NQ", JAN4, JAN8)

    assert frame.height == 15
