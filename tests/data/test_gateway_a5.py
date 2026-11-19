"""ADR-001 A5 (a)-(g) and A14 (d): one test (or more) per letter, on synthetic stores only."""

from datetime import date, time
from pathlib import Path

import numpy as np
import pytest
from gateway_synth import (
    PRICE_FACTORS,
    TICKS_FIXED,
    Row,
    SynthStores,
    ct_ns,
    research_name,
    session_rows,
    step2_name,
    write_store,
)

from pfb.data.catalog import PIN_MATCH, PIN_MISMATCH, PIN_NONE, Catalog, CatalogEntry, file_sha256
from pfb.data.errors import (
    BadRequest,
    CatalogError,
    ClosedWindowRefused,
    EraNotSupported,
    NoBarsFound,
    OffGridPrice,
    PathRefused,
    PinMismatch,
    StoreRefused,
    TickMismatch,
)
from pfb.data.gateway import TICK_ABSENT, TICK_EXACT, DataGateway, grid_violations, provenance_warnings
from pfb.data.paths import ERA_RESEARCH, FORBIDDEN_PARTS

JAN4, JAN8, JAN11, JAN22, JAN29 = (date(2021, 1, d) for d in (4, 8, 11, 22, 29))
FEB8, FEB12, FEB19, MAR31 = date(2021, 2, 8), date(2021, 2, 12), date(2021, 2, 19), date(2021, 3, 31)
NQ_RESEARCH_KEY = ("NQ", ERA_RESEARCH, None)
MES_RESEARCH_KEY = ("MES", ERA_RESEARCH, None)


def _with_entry(catalog: Catalog, replace: CatalogEntry) -> Catalog:
    files = tuple(replace if e.path == replace.path else e for e in catalog.files)
    return Catalog(catalog.created_utc, catalog.closed_windows_sha256, files, catalog.ignored)


# ---------------------------------------------------------------- (a) upstream pin status ----
def test_a_pin_mismatch_entry_is_refused_before_any_bar_is_read(stores: SynthStores) -> None:
    stores.research("NQ", session_rows(JAN4, JAN29))
    catalog = stores.catalog(pins={NQ_RESEARCH_KEY: "0" * 64})
    gateway = stores.gateway(catalog)

    with pytest.raises(PinMismatch):
        gateway.bars("NQ", JAN4, JAN8)
    with pytest.raises(PinMismatch):
        gateway.files_for("NQ", JAN4, JAN8)

    assert catalog.files[0].pe_pin_status == PIN_MISMATCH
    assert gateway.provenance == ()


def test_a_unpinned_entry_is_read_and_flagged_in_provenance(stores: SynthStores) -> None:
    stores.research("NQ", session_rows(JAN4, JAN29), meta_tick_fixed=TICKS_FIXED["NQ"])
    gateway = stores.gateway()

    frame = gateway.bars("NQ", JAN4, JAN8)

    assert frame.height == 15
    assert [p.pe_pin_status for p in gateway.provenance] == [PIN_NONE]
    warnings = provenance_warnings(gateway.provenance)
    assert len(warnings) == 1 and "no producer pin" in warnings[0]


def test_a_pinned_entry_is_read_without_a_warning(stores: SynthStores) -> None:
    path = stores.research("NQ", session_rows(JAN4, JAN29), meta_tick_fixed=TICKS_FIXED["NQ"])
    gateway = stores.gateway(stores.catalog(pins={NQ_RESEARCH_KEY: file_sha256(path)}))

    gateway.bars("NQ", JAN4, JAN8)

    assert [p.pe_pin_status for p in gateway.provenance] == [PIN_MATCH]
    assert provenance_warnings(gateway.provenance) == ()


def test_a_unknown_pin_status_is_refused(stores: SynthStores) -> None:
    stores.research("NQ", session_rows(JAN4, JAN29))
    catalog = stores.catalog()
    odd = CatalogEntry(**{**catalog.files[0].__dict__, "pe_pin_status": "unknown"})

    with pytest.raises(CatalogError, match="pe_pin_status"):
        stores.gateway(_with_entry(catalog, odd)).bars("NQ", JAN4, JAN8)


# ---------------------------------------------------------------- (b) MES confirmation era ----
def test_b_mes_confirmation_era_is_refused_but_stays_in_the_catalog(stores: SynthStores) -> None:
    stores.mes_confirmation(session_rows(JAN4, JAN29))
    gateway = stores.gateway()

    with pytest.raises(EraNotSupported, match="mes_confirmation"):
        gateway.bars("MES", JAN4, JAN8)
    with pytest.raises(EraNotSupported):
        gateway.files_for("MES", JAN4, JAN8)

    assert [e.era for e in gateway.catalog.files_for("MES")] == ["mes_confirmation"]


def test_b_mes_research_era_outside_the_confirmation_range_still_reads(stores: SynthStores) -> None:
    stores.mes_confirmation(session_rows(JAN4, JAN8))
    stores.research("MES", session_rows(JAN11, JAN22))
    gateway = stores.gateway()

    frame = gateway.bars("MES", JAN11, JAN22)

    assert frame["era"].unique().to_list() == ["research"]
    with pytest.raises(EraNotSupported):
        gateway.bars("MES", JAN4, JAN22)


# ---------------------------------------------------------------- (c) order of checks ----
def test_c_request_touching_a_closed_label_range_is_refused_before_files_are_selected(
    stores: SynthStores,
) -> None:
    stores.research("NQ", session_rows(JAN4, MAR31))
    gateway = stores.gateway()
    (stores.paths.research / "NQ" / research_name("NQ")).unlink()  # selection would still succeed

    with pytest.raises(ClosedWindowRefused, match="syn_embargo"):
        gateway.files_for("NQ", FEB8, FEB19)


def test_c_rows_dropped_by_the_label_predicate_are_never_checked(stores: SynthStores) -> None:
    rows = session_rows(JAN4, MAR31)
    embargo_row = next(i for i, r in enumerate(rows) if r.label == "2021-02-16")
    rows[embargo_row] = Row(rows[embargo_row].ts_ns, rows[embargo_row].label, 0, raw_price=1.0001)
    stores.research("NQ", rows)  # an off-grid price on a closed (embargo) date

    frame = stores.gateway().bars("NQ", JAN4, JAN29)

    assert frame.height == 20 * 3
    assert frame["trade_date"].max() == JAN29


def test_c_selected_row_with_a_closed_time_raises_even_with_an_open_label(stores: SynthStores) -> None:
    rows = session_rows(FEB8, FEB12)
    rows.append(Row(ct_ns(FEB12, time(16, 30)), FEB12.isoformat(), 90_000))
    stores.research("NQ", rows)

    with pytest.raises(ClosedWindowRefused, match="syn_embargo"):
        stores.gateway().bars("NQ", FEB8, FEB12)


# ---------------------------------------------------------------- (d) exact tick check ----
def test_d_exact_metadata_tick_times_factor_is_accepted(stores: SynthStores) -> None:
    exchange_tick = TICKS_FIXED["ZC"] // PRICE_FACTORS["ZC"]
    stores.research("ZC", session_rows(JAN4, JAN8), meta_tick_fixed=exchange_tick)

    assert stores.gateway().bars("ZC", JAN4, JAN8).height == 15


def test_d_metadata_tick_right_only_under_another_factor_is_refused(stores: SynthStores) -> None:
    stores.research("ZC", session_rows(JAN4, JAN8), meta_tick_fixed=TICKS_FIXED["ZC"])  # factor 1, not 100

    with pytest.raises(TickMismatch, match="price factor 100"):
        stores.gateway().bars("ZC", JAN4, JAN8)


def test_d_factor_100_metadata_on_a_factor_1_root_is_refused(stores: SynthStores) -> None:
    stores.research("NQ", session_rows(JAN4, JAN8), meta_tick_fixed=TICKS_FIXED["NQ"] // 100)

    with pytest.raises(TickMismatch):
        stores.gateway().bars("NQ", JAN4, JAN8)


@pytest.mark.parametrize("factor", [0, -1, True, 1.0])
def test_d_bad_price_factor_from_the_lookup_is_refused(stores: SynthStores, factor: object) -> None:
    stores.research("NQ", session_rows(JAN4, JAN8), meta_tick_fixed=TICKS_FIXED["NQ"])
    gateway = DataGateway.for_synthetic_stores(  # a synthetic table goes through the test factory
        stores.paths,
        stores.windows,
        stores.catalog(),
        TICKS_FIXED.__getitem__,
        lambda _root: factor,  # type: ignore[arg-type,return-value]
    )

    with pytest.raises(TickMismatch, match="price_factor_of"):
        gateway.bars("NQ", JAN4, JAN8)


def test_d_file_without_a_metadata_tick_is_read_flagged_and_warned(stores: SynthStores) -> None:
    # only the MES research file may lack the tick (A14 h); NQ without one is refused (hardening tests).
    path = stores.research("MES", session_rows(JAN4, JAN8), meta_tick_fixed=None)
    gateway = stores.gateway(stores.catalog(pins={MES_RESEARCH_KEY: file_sha256(path)}))

    frame = gateway.bars("MES", JAN4, JAN8)

    assert frame.height == 15
    assert [p.tick_metadata for p in gateway.provenance] == [TICK_ABSENT]
    warnings = provenance_warnings(gateway.provenance)
    assert len(warnings) == 1 and "no metadata tick" in warnings[0] and Path(path).name in warnings[0]


def test_d_file_without_a_metadata_tick_still_gets_the_grid_check(stores: SynthStores) -> None:
    rows = session_rows(JAN4, JAN8)
    rows[2] = Row(rows[2].ts_ns, rows[2].label, 0, raw_price=10_000.125)
    stores.research("MES", rows, meta_tick_fixed=None)  # the allowed tick-less file

    with pytest.raises(OffGridPrice):
        stores.gateway().bars("MES", JAN4, JAN8)


def test_d_one_tick_warning_per_file_and_exact_files_carry_exact(stores: SynthStores) -> None:
    # the tick-less file is now MES research (allowed); before the fix it was an NQ hist file.
    write_store(
        stores.paths.step2 / "MES" / step2_name("MES"), session_rows(JAN4, JAN8), TICKS_FIXED["MES"],
        meta_tick_fixed=TICKS_FIXED["MES"],
    )  # fmt: skip
    stores.research("MES", session_rows(JAN11, JAN22), meta_tick_fixed=None)
    gateway = stores.gateway()

    gateway.bars("MES", JAN4, JAN22)

    assert [(p.era, p.tick_metadata) for p in gateway.provenance] == [
        ("step2", TICK_EXACT),
        ("research", TICK_ABSENT),
    ]
    assert sum("no metadata tick" in w for w in provenance_warnings(gateway.provenance)) == 1


def test_d_metadata_tick_present_but_not_an_integer_raises(stores: SynthStores) -> None:
    stores.research("NQ", session_rows(JAN4, JAN8), meta_tick_fixed=2.5e8)  # type: ignore[arg-type]

    with pytest.raises(TickMismatch, match="not an integer"):
        stores.gateway().bars("NQ", JAN4, JAN8)


# ---------------------------------------------------------------- (e) both upstream guards ----
def test_e_remainder_guard_flags_a_fixed_value_off_the_grid() -> None:
    prices = np.array([10_000.25, 10_000.125])
    fixed = np.array([10_000_250_000_000, 10_000_125_000_000], dtype=np.int64)

    assert grid_violations(prices, fixed, TICKS_FIXED["NQ"]).tolist() == [False, True]


def test_e_tolerance_guard_flags_a_price_far_from_its_on_grid_fixed_value() -> None:
    prices = np.array([10_000.25, 10_000.25 + 2e-6])
    fixed = np.array([10_000_250_000_000, 10_000_250_000_000], dtype=np.int64)  # both on the grid

    assert grid_violations(prices, fixed, TICKS_FIXED["NQ"]).tolist() == [False, True]


def test_e_price_within_tolerance_converts(stores: SynthStores) -> None:
    rows = session_rows(JAN4, JAN8)
    rows[1] = Row(rows[1].ts_ns, rows[1].label, 0, raw_price=10_000.25 + 4e-10)
    stores.research("NQ", rows)

    frame = stores.gateway().bars("NQ", JAN4, JAN8)

    assert frame["open"][1] == 40_001


# ---------------------------------------------------------------- (f) symlinks and forbidden parts ----
@pytest.mark.parametrize("part", sorted(FORBIDDEN_PARTS))
def test_f_catalog_path_resolving_into_a_forbidden_part_is_refused(stores: SynthStores, part: str) -> None:
    stores.research("NQ", session_rows(JAN4, JAN8))
    catalog = stores.catalog()
    target = write_store(
        stores.paths.research / part / "NQ" / research_name("NQ"), session_rows(JAN4, JAN8), TICKS_FIXED["NQ"]
    )
    link = stores.paths.research / "NQ" / research_name("NQ")
    link.unlink()
    link.symlink_to(target)

    with pytest.raises(PathRefused, match=part):
        stores.gateway(catalog).bars("NQ", JAN4, JAN8)


@pytest.mark.parametrize("part", sorted(FORBIDDEN_PARTS))
def test_f_catalog_path_naming_a_forbidden_part_is_refused(stores: SynthStores, part: str) -> None:
    stores.research("NQ", session_rows(JAN4, JAN8))
    catalog = stores.catalog()
    entry = catalog.files[0]
    forged = Path(entry.path).parent / part / ".." / research_name("NQ")
    tampered = Catalog(
        catalog.created_utc,
        catalog.closed_windows_sha256,
        (CatalogEntry(**{**entry.__dict__, "path": str(forged)}),),
        (),
    )

    with pytest.raises(PathRefused, match=part):
        stores.gateway(tampered).bars("NQ", JAN4, JAN8)


# ---------------------------------------------------------------- (g) session is a clock filter ----
@pytest.mark.parametrize("session", ["rth", "globex", None, (510, 900)])
def test_g_spec_session_names_are_not_resolved_by_the_gateway(stores: SynthStores, session: object) -> None:
    stores.research("NQ", session_rows(JAN4, JAN8))

    with pytest.raises(BadRequest, match="ALL or a ClockWindow"):
        stores.gateway().bars("NQ", JAN4, JAN8, session)  # type: ignore[arg-type]


# ---------------------------------------------------------------- A14 (d) eras and files_for ----
def test_a14_files_for_returns_read_order_without_reading_a_bar(stores: SynthStores) -> None:
    stores.research("NQ", session_rows(JAN22, JAN29))
    stores.step2("NQ", session_rows(JAN11, JAN22)[3:])
    stores.hist("NQ", session_rows(JAN4, JAN8))
    gateway = stores.gateway()

    entries = gateway.files_for("NQ", JAN4, JAN29)
    for entry in entries:
        Path(entry.path).unlink()  # files_for must not need the bytes

    assert [e.era for e in entries] == ["hist", "step2", "research"]
    assert isinstance(entries, tuple)


def test_a14_provenance_equals_files_for_in_path_and_sha256(stores: SynthStores) -> None:
    stores.hist("NQ", session_rows(JAN4, JAN8))
    stores.step2("NQ", session_rows(JAN11, JAN22))
    gateway = stores.gateway()

    planned = gateway.files_for("NQ", JAN4, JAN22)
    gateway.bars("NQ", JAN4, JAN22)

    assert [(p.path, p.sha256) for p in gateway.provenance] == [(e.path, e.sha256) for e in planned]


def test_a14_files_for_only_lists_overlapping_files(stores: SynthStores) -> None:
    stores.hist("NQ", session_rows(JAN4, JAN8))
    stores.step2("NQ", session_rows(JAN11, JAN22))
    gateway = stores.gateway()

    assert [e.era for e in gateway.files_for("NQ", JAN11, JAN22)] == ["step2"]
    with pytest.raises(NoBarsFound):
        gateway.files_for("NQ", date(2020, 6, 1), date(2020, 6, 5))


def test_a14_files_for_refuses_bad_requests_like_bars(stores: SynthStores) -> None:
    stores.research("NQ", session_rows(JAN4, JAN8))
    gateway = stores.gateway()

    with pytest.raises(BadRequest):
        gateway.files_for("NQ", JAN8, JAN4)
    with pytest.raises(BadRequest):
        gateway.files_for("ES", JAN4, JAN8)


def test_a14_ts_going_back_across_eras_is_refused(stores: SynthStores) -> None:
    stores.hist("NQ", session_rows(JAN4, JAN8))
    late_hist = ct_ns(JAN8, time(15, 59))
    step2 = session_rows(JAN11, JAN22)
    step2[0] = Row(late_hist - 60_000_000_000, step2[0].label, step2[0].price_ticks)  # labelled 01-11
    stores.step2("NQ", step2)

    with pytest.raises(StoreRefused, match="strictly increasing"):
        stores.gateway().bars("NQ", JAN4, JAN22)


def test_a14_equal_ts_across_eras_is_refused_not_deduplicated(stores: SynthStores) -> None:
    stores.hist("NQ", session_rows(JAN4, JAN8))
    step2 = session_rows(JAN11, JAN22)
    step2[0] = Row(ct_ns(JAN8, time(15, 59)), step2[0].label, step2[0].price_ticks)
    stores.step2("NQ", step2)

    with pytest.raises(StoreRefused, match="strictly increasing"):
        stores.gateway().bars("NQ", JAN4, JAN22)


def test_a14_repeated_ts_inside_one_file_is_refused(stores: SynthStores) -> None:
    rows = session_rows(JAN4, JAN8)
    rows[7] = Row(rows[6].ts_ns, rows[7].label, rows[7].price_ticks + 5)  # same instant, other price
    stores.research("NQ", rows)

    with pytest.raises(StoreRefused, match="strictly increasing"):
        stores.gateway().bars("NQ", JAN4, JAN8)
