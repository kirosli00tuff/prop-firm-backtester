"""The windows of a root (open, closed with the reason, on disk, missing) from a temporary
closed-window table and catalog, the gateway's era rule, and the real table and catalog read the same way
(metadata only)."""

from __future__ import annotations

import hashlib
import json
import tomllib
from datetime import date
from pathlib import Path

import pytest

from pfb.dash.windows import (
    REFERENCE_FIRST,
    STATUS_CLOSED,
    STATUS_MISSING,
    STATUS_OPEN,
    STATUS_UNREADABLE,
    UNSUPPORTED_ERAS,
    Sources,
    WindowRow,
    missing_windows,
    open_windows,
    window_rows,
)
from pfb.data.catalog import CATALOG_SCHEMA, DEFAULT_CATALOG_JSON, PIN_MATCH, PIN_MISMATCH, PIN_NONE
from pfb.data.errors import WindowTableMismatch
from pfb.data.windows import DEFAULT_WINDOWS_TOML

TODAY = date(2026, 11, 17)
SHA_A, SHA_B = "a" * 64, "b" * 64
TABLE = """
schema = "pfb.closed_windows/1"

[[window]]
id = "holdout_x"
products = ["*"]
first_trade_date = "2026-05-18"
last_trade_date = "9999-12-31"
utc_start = "2026-05-15T21:01:00Z"
utc_end = "9999-12-31T00:00:00Z"
kind = "sealed holdout"

[[window]]
id = "embargo_x"
products = ["*"]
first_trade_date = "2026-05-09"
last_trade_date = "2026-05-17"
utc_start = "2026-05-08T21:01:00Z"
utc_end = "2026-05-17T22:00:00Z"
kind = "embargo"

[[window]]
id = "gap_x"
products = ["*"]
first_trade_date = "2022-02-01"
last_trade_date = "2022-11-30"
utc_start = "2022-01-31T22:01:00Z"
utc_end = "2022-11-30T23:00:00Z"
kind = "sealed holdout"

[[window]]
id = "only_bbb"
products = ["BBB"]
first_trade_date = "2015-01-01"
last_trade_date = "2015-12-31"
utc_start = "2014-12-31T21:00:00Z"
utc_end = "2015-12-31T22:00:00Z"
kind = "closed by decision"
"""


def _entry(root: str, era: str, first: str, last: str, *, pin: str | None = None, sha: str = SHA_A) -> dict:
    status = PIN_NONE if pin is None else (PIN_MATCH if pin == sha else PIN_MISMATCH)
    return {
        "root": root,
        "era": era,
        "plan": None,
        "path": f"/stores/{era}/{root}/ohlcv-1m_{root}_{era}.parquet",
        "size": 1,
        "sha256": sha,
        "num_rows": 1,
        "first_trade_date": first,
        "last_trade_date": last,
        "schema": [{"name": "trade_date", "dtype": "date32[day]"}],
        "pe_pin": pin,
        "pe_pin_status": status,
    }


def _sources(tmp_path: Path, files: list[dict]) -> Sources:
    toml = tmp_path / "closed_windows.toml"
    toml.write_text(TABLE)
    catalog = tmp_path / "catalog.json"
    body = {"schema": CATALOG_SCHEMA, "created_utc": "x", "closed_windows_sha256": "x", "files": files}
    catalog.write_text(json.dumps({**body, "ignored": []}))
    return Sources(toml, hashlib.sha256(toml.read_bytes()).hexdigest(), catalog)


def _rows(tmp_path: Path, files: list[dict], root: str = "AAA") -> tuple[WindowRow, ...]:
    table, catalog = _sources(tmp_path, files).load()
    return window_rows(root, table, catalog, TODAY)


def _spans(rows: tuple[WindowRow, ...], status: str) -> list[tuple[date, date | None]]:
    return [(r.first, r.last) for r in rows if r.status == status]


AAA_FILES = [
    _entry("AAA", "step2", "2019-01-02", "2022-01-31", pin=SHA_A),
    _entry("AAA", "research", "2022-12-01", "2026-05-15"),
]


def test_open_rows_are_the_open_part_of_each_catalogued_file(tmp_path: Path) -> None:
    # Act
    rows = _rows(tmp_path, AAA_FILES)
    # Assert
    assert _spans(rows, STATUS_OPEN) == [
        (date(2019, 1, 2), date(2022, 1, 31)),
        (date(2022, 12, 1), date(2026, 5, 8)),
    ]
    research = [r for r in open_windows(rows) if r.era == "research"][0]
    assert research.on_disk and research.files == ("ohlcv-1m_AAA_research.parquet",)


def test_closed_rows_carry_the_kind_and_id_and_whether_a_file_holds_them(tmp_path: Path) -> None:
    # Act
    closed = {r.reason: r for r in _rows(tmp_path, AAA_FILES) if r.status == STATUS_CLOSED}
    # Assert
    assert set(closed) == {"sealed holdout (holdout_x)", "embargo (embargo_x)", "sealed holdout (gap_x)"}
    assert closed["sealed holdout (holdout_x)"].last is None
    assert closed["embargo (embargo_x)"].on_disk
    assert closed["embargo (embargo_x)"].files == ("ohlcv-1m_AAA_research.parquet",)
    assert not closed["sealed holdout (gap_x)"].on_disk


def test_open_ranges_no_file_covers_are_missing(tmp_path: Path) -> None:
    # Act
    rows = _rows(tmp_path, AAA_FILES)
    # Assert
    assert _spans(rows, STATUS_MISSING) == [(REFERENCE_FIRST, date(2019, 1, 1))]
    assert all(not r.on_disk and r.files == () for r in missing_windows(rows))


def test_a_window_of_another_root_does_not_close_this_one(tmp_path: Path) -> None:
    # Arrange
    files = [
        _entry("AAA", "hist", "2010-06-07", "2018-12-31"),
        _entry("BBB", "hist", "2010-06-07", "2018-12-31"),
    ]
    # Act
    aaa = _rows(tmp_path, files, "AAA")
    bbb = _rows(tmp_path, files, "BBB")
    # Assert
    assert _spans(aaa, STATUS_OPEN) == [(date(2010, 6, 7), date(2018, 12, 31))]
    assert _spans(bbb, STATUS_OPEN) == [
        (date(2010, 6, 7), date(2014, 12, 31)),
        (date(2016, 1, 1), date(2018, 12, 31)),
    ]
    assert "closed by decision (only_bbb)" in {r.reason for r in bbb if r.status == STATUS_CLOSED}


def test_files_the_gateway_refuses_from_the_catalog_are_unreadable_not_open(tmp_path: Path) -> None:
    # Arrange
    files = [
        _entry("AAA", "mes_confirmation", "2019-01-02", "2022-01-31"),
        _entry("AAA", "research", "2022-12-01", "2026-05-15", pin=SHA_B),
    ]
    # Act
    rows = _rows(tmp_path, files)
    # Assert
    assert open_windows(rows) == ()
    unreadable = {r.era: r.reason for r in rows if r.status == STATUS_UNREADABLE}
    assert unreadable == {
        "mes_confirmation": "era mes_confirmation is not read in this version",
        "research": "sha256 differs from the producer pin",
    }
    assert _spans(rows, STATUS_MISSING) == [(REFERENCE_FIRST, date(2019, 1, 1))]


def test_a_root_with_no_file_is_missing_everywhere_open(tmp_path: Path) -> None:
    # Act
    rows = _rows(tmp_path, AAA_FILES, "ZZZ")
    # Assert
    assert _spans(rows, STATUS_MISSING) == [
        (REFERENCE_FIRST, date(2022, 1, 31)),
        (date(2022, 12, 1), date(2026, 5, 8)),
    ]
    assert open_windows(rows) == ()


def test_every_open_row_passes_the_table_request_check(tmp_path: Path) -> None:
    # Arrange
    table, catalog = _sources(tmp_path, AAA_FILES).load()
    # Act
    rows = window_rows("AAA", table, catalog, TODAY)
    # Assert
    for row in open_windows(rows) + missing_windows(rows):
        assert row.last is not None
        assert table.request_refusal("AAA", row.first, row.last) is None
    for row in rows:
        if row.status == STATUS_CLOSED:
            assert table.request_refusal("AAA", row.first, row.last or TODAY) is not None


def test_a_changed_table_is_refused_by_its_pin(tmp_path: Path) -> None:
    # Arrange
    sources = _sources(tmp_path, AAA_FILES)
    bad = Sources(sources.windows_toml, "0" * 64, sources.catalog_json)
    # Act / Assert
    with pytest.raises(WindowTableMismatch):
        bad.load()


def test_as_dict_is_plain_json() -> None:
    # Arrange
    row = WindowRow("AAA", date(2020, 1, 2), None, STATUS_CLOSED, "embargo (e)", False, (), None)
    # Act
    out = json.loads(json.dumps(row.as_dict()))
    # Assert
    assert out == {
        "root": "AAA",
        "first": "2020-01-02",
        "last": None,
        "status": "closed",
        "reason": "embargo (e)",
        "on_disk": False,
        "files": [],
        "era": None,
    }


def test_the_unsupported_eras_are_the_gateways() -> None:
    from pfb.data.gateway import UNSUPPORTED_ERAS as GATEWAY_UNSUPPORTED

    assert UNSUPPORTED_ERAS == GATEWAY_UNSUPPORTED


def test_the_real_table_and_catalog_agree_with_the_tables_rows_on_disk() -> None:
    if not DEFAULT_CATALOG_JSON.is_file():
        pytest.skip("runs/catalog.json not built")
    # Arrange
    table, catalog = Sources().load()
    by_id = {w["id"]: w for w in tomllib.loads(DEFAULT_WINDOWS_TOML.read_text())["window"]}
    # Act
    rows = [r for root in sorted(catalog.roots()) for r in window_rows(root, table, catalog, TODAY)]
    # Assert: a closed window holding catalogued labels is one the table records as on disk (by label)
    for row in rows:
        if row.status == STATUS_CLOSED and row.on_disk:
            window_id = row.reason.rsplit("(", 1)[1].rstrip(")")
            assert by_id[window_id]["rows_on_disk"] is True, (row.root, window_id)
        if row.status == STATUS_OPEN:
            assert row.last is not None
            assert table.request_refusal(row.root, row.first, row.last) is None
