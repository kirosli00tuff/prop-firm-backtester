"""Calendar files: loading and validation, the root map against config/instruments.toml, equality with a fresh
read-only extraction from upstream, and the real-store label test of ADR A3 and A14(e).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from pfb.instruments import sessions
from pfb.instruments.calendar import (
    CALENDAR_DIR,
    GROUPS,
    CalendarFileError,
    load_calendar,
    parse_calendar,
)
from pfb.instruments.table import load_instruments

LABEL_ROOTS = ("NQ", "ZN", "6E", "CL", "GC", "ZC", "HE", "MBT")
SKIPPED_ERAS = ("mes_confirmation",)  # refused by the gateway (ADR A5(b))


# ------------------------------------------------------------------ files ----
def test_eight_files_load_with_eras_in_order() -> None:
    cals = [load_calendar(g) for g in GROUPS]

    assert [c.group for c in cals] == list(GROUPS)
    assert all(len(c.sha256) == 64 for c in cals)
    assert [len(c.eras) for c in cals] == [2, 2, 2, 2, 2, 2, 2, 1]
    assert {c.eras[-1].first for c in cals} == {date(2019, 5, 1)}


def test_root_map_equals_instrument_table_groups() -> None:
    table = load_instruments()

    mapping = {root: sessions.group_of(root) for g in GROUPS for root in load_calendar(g).roots}

    assert len(mapping) == 47
    assert mapping == {root: table.get(root).group for root in table.roots()}


def _raw(group: str = "rates") -> dict:
    return json.loads((CALENDAR_DIR / f"{group}.json").read_text())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(schema="other/1"), "schema"),
        (lambda d: d.pop("eras"), "malformed"),
        (lambda d: d["eras"][0].update(first="2010-13-01"), "not a date"),
        (lambda d: d["eras"][0]["early_halts"].append(["2015-01-02", "25:00"]), "HH:MM"),
        (lambda d: d["eras"][1].update(first="2018-01-01"), "overlap"),
        (lambda d: d["eras"][0]["sessions"][0].update(segments=[]), "no segment"),
        (lambda d: d["eras"][0]["sessions"][0]["segments"][0].__setitem__(0, 3), "day offset"),
    ],
)
def test_malformed_file_is_refused(tmp_path: Path, mutate: object, message: str) -> None:
    # Arrange
    payload = _raw()
    mutate(payload)  # type: ignore[operator]
    path = tmp_path / "rates.json"
    path.write_text(json.dumps(payload))

    # Act / Assert
    with pytest.raises(CalendarFileError, match=message):
        parse_calendar(path.read_bytes(), path, "rates")


def _set_lead(d: dict, value: object) -> None:
    d["flat_by"]["early_close_lead_min"] = value


def _set_year_lead(d: dict, value: object) -> None:
    year = next(iter(d["flat_by"]["topstep_lead_min_by_year"]))
    d["flat_by"]["topstep_lead_min_by_year"][year] = value


def _set_year_key(d: dict, value: object) -> None:
    leads = d["flat_by"]["topstep_lead_min_by_year"]
    leads[str(value)] = leads.pop(next(iter(leads)))


@pytest.mark.parametrize("bad", [5.9, True, "5", -1, None], ids=repr)
@pytest.mark.parametrize("setter", [_set_lead, _set_year_lead], ids=["early_close_lead", "year_lead"])
def test_integer_fields_refuse_floats_bools_and_strings(tmp_path: Path, setter: object, bad: object) -> None:
    # A item 17 N2: int() let 5.9, True and "5" through where the file must hold an integer.
    payload = _raw()
    setter(payload, bad)  # type: ignore[operator]
    path = tmp_path / "rates.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(CalendarFileError, match="integer"):
        parse_calendar(path.read_bytes(), path, "rates")


@pytest.mark.parametrize("bad", ["20x9", "019", "+2019", " 2019"])
def test_year_keys_must_be_four_digit_years(tmp_path: Path, bad: str) -> None:
    payload = _raw()
    _set_year_key(payload, bad)
    path = tmp_path / "rates.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(CalendarFileError, match="year"):
        parse_calendar(path.read_bytes(), path, "rates")


def test_wrong_group_and_missing_file_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "fx.json"
    path.write_text(json.dumps(_raw("rates")))

    with pytest.raises(CalendarFileError, match="group"):
        parse_calendar(path.read_bytes(), path, "fx")
    with pytest.raises(CalendarFileError, match="does not exist"):
        load_calendar("fx", tmp_path / "missing")
    with pytest.raises(CalendarFileError, match="unknown group"):
        load_calendar("softs")


def test_outage_late_open_is_kept_in_file_but_not_applied() -> None:
    # 2025-11-28: the unscheduled CME outage (upstream) is ported as data and never shapes
    # a
    # session.
    era = load_calendar("rates").eras[-1]

    assert date(2025, 11, 28) not in era.late_opens
    assert _raw("rates")["eras"][-1]["late_opens"] == [["2025-11-28", "07:30", False]]
