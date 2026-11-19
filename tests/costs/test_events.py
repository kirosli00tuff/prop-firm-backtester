"""Release calendar and event-window costs on synthetic calendars and books (no real row or cost).

Window edges [r, r + 30 min): the minute before, the first minute, the last minute and the minute after;
a root no release concerns; "off" unchanged; the configuration identity of "on".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from book_synth import product

from pfb.costs import (
    MINUTES_PER_DAY,
    CostModelError,
    EventCostModel,
    EventCostsOff,
    EventCoverageError,
    ReleaseCalendarError,
    load_cost_model,
    load_cost_model_for,
)
from pfb.costs import events as events_mod
from pfb.costs.events import (
    EVENT_WINDOW_NS,
    NS_PER_MINUTE,
    ReleaseCalendar,
    instant_ns,
    load_events_config,
    load_pinned_calendar,
    load_release_calendar,
    release_calendar_from_dict,
)
from pfb.costs.model import DEFAULT_COSTS_TOML, StressCostModel, book_slippage_cents
from pfb.instruments import InstrumentTable

CT = ZoneInfo("America/Chicago")
BUY, SELL = 0, 1
NQ_TV = Fraction(500)  # cents per NQ tick ($5.00)
NFP = "2020-03-06T13:30:00Z"  # 07:30 CT (CST)
FOMC = "2020-03-18T18:00:00Z"  # 13:00 CT (CDT, after the 2020-03-08 change)
CPI = "2020-03-11T12:30:00Z"  # 07:30 CT (CDT)


@dataclass(frozen=True)
class Paths:
    """Satisfies pfb.costs.events.EventPaths (root: the folder holding costs.toml)."""

    book_table: Path
    release_calendar: Path
    root: Path


def _bucket(start: int, end: int, half: float, buy_depth: float, sell_depth: float) -> dict[str, Any]:
    return {
        "key": f"{start // 60:02d}:{start % 60:02d}", "start_min": start, "end_min": end,
        "half_spread_ticks": half, "depth_ticks": {"buy": buy_depth, "sell": sell_depth},
        "side_ticks": {"buy": half + buy_depth, "sell": half + sell_depth}, "fallback": False,
    }  # fmt: skip


def _calendar_raw(releases: list[tuple[str, list[str], bool]]) -> dict[str, Any]:
    rows = [
        {"id": f"R{i}", "release": "X", "instant_utc": t, "products": p, "cpi": c}
        for i, (t, p, c) in enumerate(releases)
    ]
    return {"schema": "pfb.release_calendar/1", "coverage": {"first": "2020-01-01", "last": "2020-12-31"},
            "releases": rows}  # fmt: skip


RELEASES = [(NFP, ["NQ", "MNQ", "ZN"], False), (FOMC, ["NQ", "ZN"], False), (CPI, ["NQ"], True)]


def _write_world(tmp: Path, instruments: InstrumentTable, *, calendar_bytes: bytes | None = None) -> Paths:
    """A folder with a synthetic book and calendar, and a costs.toml pinning both."""
    nq = [
        _bucket(0, 420, 0.25, 0.5, 0.75),
        _bucket(420, 480, 1.0, 0.25, 0.5),  # 07:00-08:00 CT, the largest half-spread
        # 480..539 is in no bucket: R-B2 fallback (largest side ticks: buy 1.25, sell 1.5)
        _bucket(540, MINUTES_PER_DAY, 0.5, 0.125, 0.25),
    ]
    zn = [_bucket(0, MINUTES_PER_DAY, 0.5, 0.5, 0.5)]
    book = {"depth_term": "applied", "products": {"NQ": product("NQ", instruments, nq),
                                                    "ZN": product("ZN", instruments, zn)}}  # fmt: skip
    (tmp / "reports").mkdir(parents=True, exist_ok=True)
    book_bytes = json.dumps(book).encode()
    (tmp / "reports" / "book.json").write_bytes(book_bytes)
    cal = calendar_bytes if calendar_bytes is not None else json.dumps(_calendar_raw(RELEASES)).encode()
    (tmp / "reports" / "cal.json").write_bytes(cal)
    (tmp / "costs.toml").write_text(
        'schema = "pfb.costs/1"\n'
        f'[book]\nsha256 = "{hashlib.sha256(book_bytes).hexdigest()}"\nproducts = 2\n'
        f'[events]\ncalendar = "reports/cal.json"\nsha256 = "{hashlib.sha256(cal).hexdigest()}"\n'
        'schema = "pfb.release_calendar/1"\nreleases = 3\ncoverage_first = 2020-01-01\n'
        "coverage_last = 2020-12-31\nwindow_minutes = 30\n",
        encoding="utf-8",
    )
    return Paths(tmp / "reports" / "book.json", tmp / "reports" / "cal.json", tmp)


@pytest.fixture
def world(tmp_path: Path, instruments: InstrumentTable) -> Paths:
    return _write_world(tmp_path, instruments)


def _on(spec: str, world: Paths, instruments: InstrumentTable) -> EventCostModel:
    toml = world.root / "costs.toml"
    model = load_cost_model_for(spec, "on", instruments, world, costs_toml=toml)
    assert isinstance(model, EventCostModel)
    return model


def _ct_minute(ts: int) -> int:
    local = datetime.fromtimestamp(ts / 1e9, tz=CT)
    return local.hour * 60 + local.minute


def _bars(model: EventCostModel, root: str, ts: list[int]) -> np.ndarray:
    arr = np.asarray(ts, dtype=np.int64)
    minute = np.asarray([_ct_minute(t) for t in ts], dtype=np.int64)
    days = np.full(arr.shape, (date(2020, 3, 6) - date(1970, 1, 1)).days, dtype=np.int64)
    return model.bar_slippage_cents(root, 1, arr, minute, days)


# ------------------------------------------------------------------ calendar ----
def test_window_edges_minute_before_first_last_after() -> None:
    cal = release_calendar_from_dict(_calendar_raw(RELEASES), "x" * 64, "synthetic")
    r = instant_ns(NFP)
    assert not cal.in_event_window("NQ", r - NS_PER_MINUTE)
    assert cal.in_event_window("NQ", r)
    assert cal.in_event_window("NQ", r + 29 * NS_PER_MINUTE)
    assert cal.in_event_window("NQ", r + EVENT_WINDOW_NS - 1)
    assert not cal.in_event_window("NQ", r + 30 * NS_PER_MINUTE)


def test_a_root_no_release_concerns_is_never_in_a_window() -> None:
    cal = release_calendar_from_dict(_calendar_raw(RELEASES), "x" * 64, "synthetic")
    r = instant_ns(CPI)
    assert cal.in_event_window("NQ", r)
    assert not cal.in_event_window("ZN", r)  # CPI concerns NQ only here
    assert not cal.in_event_window("CL", r)  # no release at all
    assert cal.instants("CL").size == 0


def test_mask_equals_scalar_and_handles_overlap() -> None:
    raw = _calendar_raw([("2020-03-06T13:30:00Z", ["NQ"], False), ("2020-03-06T13:40:00Z", ["NQ"], False)])
    cal = release_calendar_from_dict(raw, "x" * 64, "synthetic")
    start = instant_ns("2020-03-06T13:00:00Z")
    ts = np.asarray([start + k * NS_PER_MINUTE for k in range(90)], dtype=np.int64)
    mask = cal.event_mask("NQ", ts)
    assert [bool(m) for m in mask] == [cal.in_event_window("NQ", int(t)) for t in ts]
    on = [k for k in range(90) if mask[k]]
    assert on == list(range(30, 70))  # 13:30 .. 14:09 inclusive (the second window ends 14:10)


def test_an_offset_instant_and_a_z_instant_are_the_same_ns() -> None:
    assert instant_ns("2020-03-11T08:30:00-04:00") == instant_ns(CPI)  # EDT, after the US change
    assert instant_ns("2020-03-06T08:30:00-05:00") == instant_ns(NFP)  # EST, before it


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"schema": "other/1"}, "schema"),
        ({"instant_utc": "2020-03-06T13:30:00"}, "timezone"),
        ({"instant_utc": "2020-03-06T13:30:30Z"}, "minute"),
        ({"products": "NQ"}, "products"),
    ],
)
def test_malformed_calendars_are_refused(change: dict[str, Any], match: str) -> None:
    raw = _calendar_raw(RELEASES)
    if "schema" in change:
        raw["schema"] = change["schema"]
    else:
        raw["releases"][0].update(change)
    with pytest.raises(ReleaseCalendarError, match=match):
        release_calendar_from_dict(raw, "x" * 64, "synthetic")


def test_cpi_instants_are_kept() -> None:
    cal = release_calendar_from_dict(_calendar_raw(RELEASES), "x" * 64, "synthetic")
    assert cal.cpi == (instant_ns(CPI),)


def test_coverage_is_by_trade_date() -> None:
    cal = release_calendar_from_dict(_calendar_raw(RELEASES), "x" * 64, "synthetic")
    assert cal.covers(date(2020, 1, 1), date(2020, 12, 31))
    assert not cal.covers(date(2019, 12, 31), date(2020, 6, 1))
    with pytest.raises(EventCoverageError):
        cal.require_covers(date(2020, 6, 1), date(2021, 1, 1))


# ------------------------------------------------------------------ pin ----
def test_a_calendar_unlike_its_pin_is_refused(tmp_path: Path, instruments: InstrumentTable) -> None:
    world = _write_world(tmp_path, instruments)
    toml = tmp_path / "costs.toml"
    (tmp_path / "reports" / "cal.json").write_bytes(json.dumps(_calendar_raw(RELEASES[:2])).encode())
    with pytest.raises(ReleaseCalendarError, match="sha256"):
        load_pinned_calendar(world.release_calendar, toml)


def test_a_count_unlike_the_pin_is_refused(tmp_path: Path, instruments: InstrumentTable) -> None:
    world = _write_world(tmp_path, instruments)
    toml = tmp_path / "costs.toml"
    toml.write_text(
        toml.read_text(encoding="utf-8").replace("releases = 3", "releases = 4"), encoding="utf-8"
    )
    config = load_events_config(toml)
    with pytest.raises(ReleaseCalendarError, match="releases"):
        load_release_calendar(world.root / config.calendar, config)


def test_the_tracked_pin_reads() -> None:
    config = load_events_config(DEFAULT_COSTS_TOML)
    assert config.calendar == "config/demo/release_calendar.json"
    assert (config.coverage_first, config.coverage_last) == (date(2025, 1, 1), date(2026, 6, 30))
    assert config.window_minutes == 30 and config.releases == 486


# ------------------------------------------------------------------ cost tables ----
def test_event_ticks_are_largest_half_spread_plus_own_depth(
    world: Paths, instruments: InstrumentTable
) -> None:
    model = _on("book", world, instruments)
    event = model.event_slippage_cents("NQ", 1)
    normal = model.slippage_cents("NQ", 1)
    # minute 0 (bucket half 0.25, depth 0.5/0.75): event = 1.0 + depth, normal = 0.25 + depth
    assert event[0, BUY] == book_slippage_cents(1, 1.0 + 0.5, NQ_TV) == 750
    assert event[0, SELL] == book_slippage_cents(1, 1.0 + 0.75, NQ_TV) == 875
    assert normal[0, BUY] == 375
    # the largest-half-spread bucket itself: event == normal
    assert (event[420:480] == normal[420:480]).all()
    # an uncalibrated minute pays the R-B2 fallback, event or not
    assert (event[480:540] == normal[480:540]).all()
    assert event[500, BUY] == book_slippage_cents(1, 1.25, NQ_TV)


def test_stress_event_adds_its_ticks(world: Paths, instruments: InstrumentTable) -> None:
    book = _on("book", world, instruments)
    stress = _on("stress:book:1", world, instruments)
    assert (stress.event_slippage_cents("NQ", 2) == book.event_slippage_cents("NQ", 2) + int(2 * NQ_TV)).all()


def test_fixed_cannot_have_event_costs(world: Paths, instruments: InstrumentTable) -> None:
    with pytest.raises(CostModelError, match="book-based"):
        _on("fixed:1", world, instruments)
    fixed = load_cost_model("fixed:1", instruments, None)
    with pytest.raises(EventCostsOff):
        fixed.event_slippage_cents("NQ", 1)


def test_bar_costs_charge_event_only_inside_the_window(world: Paths, instruments: InstrumentTable) -> None:
    model = _on("book", world, instruments)
    r = instant_ns(NFP)  # 07:30 CT, bucket 420..480
    ts = [r - NS_PER_MINUTE, r, r + 29 * NS_PER_MINUTE, r + 30 * NS_PER_MINUTE]
    got = _bars(model, "NQ", ts)
    normal, event = model.slippage_cents("NQ", 1), model.event_slippage_cents("NQ", 1)
    minutes = [_ct_minute(t) for t in ts]
    assert minutes == [449, 450, 479, 480]
    assert (got[0] == normal[449]).all() and (got[3] == normal[480]).all()
    assert (got[1] == event[450]).all() and (got[2] == event[479]).all()


def test_bar_costs_use_the_dst_correct_bucket(world: Paths, instruments: InstrumentTable) -> None:
    model = _on("book", world, instruments)
    r = instant_ns(FOMC)  # 13:00 CDT = minute 780 (bucket 540..1440), not 720
    got = _bars(model, "NQ", [r])
    assert _ct_minute(r) == 780
    assert (got[0] == model.event_slippage_cents("NQ", 1)[780]).all()
    assert got[0, BUY] == book_slippage_cents(1, 1.0 + 0.125, NQ_TV)


def test_a_root_with_no_release_pays_normal(world: Paths, instruments: InstrumentTable) -> None:
    model = _on("book", world, instruments)
    r = instant_ns(CPI)  # concerns NQ only
    got = _bars(model, "ZN", [r])
    assert (got[0] == model.slippage_cents("ZN", 1)[_ct_minute(r)]).all()


def test_a_trade_date_outside_coverage_is_refused(world: Paths, instruments: InstrumentTable) -> None:
    model = _on("book", world, instruments)
    ts = np.asarray([instant_ns(NFP)], dtype=np.int64)
    days = np.asarray([(date(2021, 1, 4) - date(1970, 1, 1)).days], dtype=np.int64)
    with pytest.raises(EventCoverageError):
        model.bar_slippage_cents("NQ", 1, ts, np.asarray([450], dtype=np.int64), days)
    with pytest.raises(EventCoverageError):
        model.bar_slippage_cents(
            "NQ", 1, ts, np.asarray([450], dtype=np.int64), np.asarray(["2019-12-31"], dtype="datetime64[D]")
        )


# ------------------------------------------------------------------ identity and off ----
def test_on_is_a_different_configuration(world: Paths, instruments: InstrumentTable, tmp_path: Path) -> None:
    toml = world.root / "costs.toml"
    off = load_cost_model_for("book", "off", instruments, world, costs_toml=toml)
    on = _on("book", world, instruments)
    assert on.model_id == "book+events" and off.model_id == "book"
    assert on.config_sha256 != off.config_sha256
    other = _write_world(tmp_path / "other", instruments,
                         calendar_bytes=json.dumps(_calendar_raw(RELEASES), indent=1).encode())  # fmt: skip
    other_toml = other.root / "costs.toml"
    on2 = load_cost_model_for("book", "on", instruments, other, costs_toml=other_toml)
    assert on2.config_sha256 != on.config_sha256  # same rows, other bytes: the sha256 changes the identity


def test_off_is_exactly_load_cost_model(world: Paths, instruments: InstrumentTable) -> None:
    toml = world.root / "costs.toml"
    for spec in ("book", "stress:book:1", "fixed:2"):
        a = load_cost_model_for(spec, "off", instruments, world, costs_toml=toml)
        b = load_cost_model(spec, instruments, world, costs_toml=toml)
        assert type(a) is type(b) and (a.model_id, a.config_sha256) == (b.model_id, b.config_sha256)
        assert a.slippage_cents("NQ", 3).tobytes() == b.slippage_cents("NQ", 3).tobytes()
    with pytest.raises(CostModelError):
        load_cost_model_for("book", "maybe", instruments, world, costs_toml=toml)


def test_on_normal_tables_equal_the_base(world: Paths, instruments: InstrumentTable) -> None:
    on = _on("book", world, instruments)
    assert on.slippage_cents("NQ", 2).tobytes() == on.base.slippage_cents("NQ", 2).tobytes()
    assert on.calibrated("NQ").tobytes() == on.base.calibrated("NQ").tobytes()


BOOK_CONFIG_SHA256 = "5fc3657dae2cdaa7e9cca683a1be14181c9c610c77b042405f666fedf3e6907a"  # the shipped table
FIXED1_CONFIG_SHA256 = "5e79c7a588a996b9cb9fb0dc35797de680f7d5a9bb825d46168c5371a1d59020"


def test_off_identity_values_are_unchanged_by_this_stage(instruments: InstrumentTable) -> None:
    # The values the earlier code gave (and ledger records carry); they must not move.
    from pfb.costs.model import COSTS_SCHEMA, _config_sha256, load_costs_config

    pin = load_costs_config(DEFAULT_COSTS_TOML)["sha256"]
    book = {"schema": COSTS_SCHEMA, "model_id": "book", "book_table_sha256": pin}
    assert _config_sha256(book) == BOOK_CONFIG_SHA256
    assert load_cost_model("fixed:1", instruments, None).config_sha256 == FIXED1_CONFIG_SHA256


def test_an_event_model_refuses_a_root_the_calendar_does_not_name(
    world: Paths, instruments: InstrumentTable
) -> None:
    # Arrange: the book prices NQ and ZN; this calendar names NQ and MNQ only
    toml = world.root / "costs.toml"
    book = load_cost_model("book", instruments, world, costs_toml=toml)
    raw = _calendar_raw([(NFP, ["NQ", "MNQ"], False)])
    calendar = release_calendar_from_dict(raw, "x" * 64, "synthetic")
    # Act / Assert
    with pytest.raises(ReleaseCalendarError, match=r"\['ZN'\]"):
        EventCostModel(book, calendar)


def test_a_micro_is_charged_by_the_union_with_its_sibling() -> None:
    # Arrange: NFP concerns NQ and MNQ, FOMC and CPI NQ only
    cal = release_calendar_from_dict(_calendar_raw(RELEASES), "x" * 64, "synthetic")
    # Act
    union = events_mod._union_instants(cal, "MNQ")
    # Assert
    assert cal.instants("MNQ").tolist() == [instant_ns(NFP)]
    assert union.tolist() == cal.instants("NQ").tolist() == sorted(instant_ns(t) for t in (NFP, FOMC, CPI))
    assert events_mod._union_instants(cal, "NQ").tolist() == cal.instants("NQ").tolist()
    assert not union.flags.writeable


def test_qm_and_qg_are_in_the_union_with_cl_and_ng() -> None:
    """: the mini-sized QM and QG are charged like CL and NG."""
    # Arrange: one release concerns CL and QM, a second CL only, a third NG only
    raw = _calendar_raw([(NFP, ["CL", "QM"], False), (FOMC, ["CL"], False), (CPI, ["NG"], False)])
    cal = release_calendar_from_dict(raw, "x" * 64, "synthetic")
    # Act / Assert
    assert (events_mod.EVENT_SIBLINGS["QM"], events_mod.EVENT_SIBLINGS["QG"]) == ("CL", "NG")
    assert len(events_mod.EVENT_SIBLINGS) == 14
    assert cal.instants("QM").tolist() == [instant_ns(NFP)] and cal.instants("QG").tolist() == []
    assert events_mod._union_instants(cal, "QM").tolist() == sorted(instant_ns(t) for t in (NFP, FOMC))
    assert events_mod._union_instants(cal, "QG").tolist() == [instant_ns(CPI)]
    assert events_mod._union_instants(cal, "CL").tolist() == cal.instants("CL").tolist()
    assert events_mod._union_instants(cal, "NG").tolist() == cal.instants("NG").tolist()


def test_stress_over_an_event_model_is_impossible(world: Paths, instruments: InstrumentTable) -> None:
    on = _on("book", world, instruments)
    with pytest.raises(CostModelError, match="never stress over an event model"):
        StressCostModel(on, Fraction(1), "1", instruments)
    # the right nesting keeps the calendar and adds the ticks to both tables
    stressed = _on("stress:book:1", world, instruments)
    assert isinstance(stressed.base, StressCostModel) and stressed.calendar.sha256 == on.calendar.sha256
    assert stressed.model_id == "stress:book:1+events"


def test_calendar_is_frozen() -> None:
    cal: ReleaseCalendar = release_calendar_from_dict(_calendar_raw(RELEASES), "x" * 64, "synthetic")
    with pytest.raises(ValueError):
        cal.instants("NQ")[0] = 0
    with pytest.raises(TypeError):
        cal.by_root["NQ"] = np.zeros(1, dtype=np.int64)  # type: ignore[index]
