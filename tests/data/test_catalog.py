"""The store catalog and `pfb data catalog`: footer facts, ignored files, producer pins."""

import json
from datetime import date
from pathlib import Path

import pytest
from gateway_synth import SynthStores, research_name, session_rows, step2_name
from typer.testing import CliRunner

from pfb.cli.main import app
from pfb.data.catalog import (
    CATALOG_SCHEMA,
    PIN_MATCH,
    PIN_MISMATCH,
    PIN_NONE,
    PinKey,
    build_catalog,
    file_sha256,
    load_catalog,
    load_pins,
    pins_json,
    write_catalog,
)
from pfb.data.errors import CatalogError

FIRST, LAST = "2021-01-04", "2021-01-29"


@pytest.fixture
def populated(stores: SynthStores) -> SynthStores:
    rows = session_rows(date(2021, 1, 4), date(2021, 1, 29))
    stores.research("NQ", rows)
    stores.step2("NQ", rows)
    stores.hist("NQ", rows)
    junk = stores.paths.research / "NQ" / "superseded" / research_name("NQ")
    junk.parent.mkdir(parents=True)
    junk.write_bytes(b"never opened")
    sealed = stores.paths.step2 / "sealed" / "NQ" / step2_name("NQ")
    sealed.parent.mkdir(parents=True)
    sealed.write_bytes(b"never listed")
    (stores.paths.research / "NQ" / "notes.txt").write_text("x", encoding="utf-8")
    return stores


def _write_pins(stores: SynthStores, research_pin: str, step2_pin: str) -> Path:
    """A producer pins file: NQ research and step2 pinned, MES research pinned to an absent file."""
    pins: dict[PinKey, str] = {
        ("NQ", "research", None): research_pin,
        ("NQ", "step2", None): step2_pin,
        ("MES", "research", None): "a" * 64,
    }
    path = stores.base / "pins.json"
    path.write_text(json.dumps(pins_json(pins)), encoding="utf-8")
    return path


def test_catalog_lists_each_store_file_with_footer_facts(populated: SynthStores) -> None:
    catalog = populated.catalog()

    assert sorted(e.era for e in catalog.files) == ["hist", "research", "step2"]
    for entry in catalog.files:
        assert entry.num_rows == 60
        assert (entry.first_trade_date, entry.last_trade_date) == (FIRST, LAST)
        assert entry.sha256 == file_sha256(Path(entry.path))
        assert entry.schema[0] == ("ts_event", "int64")
        assert entry.pe_pin_status == PIN_NONE
    assert next(e for e in catalog.files if e.era == "hist").plan == "plan_a"


def test_catalog_ignores_other_files_with_reasons_and_never_lists_sealed(populated: SynthStores) -> None:
    catalog = populated.catalog()

    reasons = {Path(i.path).name: i.reason for i in catalog.ignored}
    assert "superseded" in reasons[research_name("NQ")]
    assert "notes.txt" in reasons
    assert "forbidden, not listed" in reasons["sealed"]
    assert not any("sealed/" in i.path for i in catalog.ignored)


def test_catalog_round_trips_through_json(populated: SynthStores, tmp_path: Path) -> None:
    catalog = populated.catalog()
    out = tmp_path / "runs" / "catalog.json"

    write_catalog(catalog, out)
    again = load_catalog(out)

    assert again == catalog
    assert json.loads(out.read_text(encoding="utf-8"))["schema"] == CATALOG_SCHEMA


def test_load_catalog_refuses_wrong_schema(tmp_path: Path) -> None:
    bad = tmp_path / "catalog.json"
    bad.write_text(json.dumps({"schema": "other", "files": [], "ignored": []}), encoding="utf-8")

    with pytest.raises(CatalogError):
        load_catalog(bad)


def test_pins_match_and_mismatch(populated: SynthStores) -> None:
    research = file_sha256(populated.paths.research / "NQ" / research_name("NQ"))
    path = _write_pins(populated, research_pin=research, step2_pin="0" * 64)

    pins = load_pins(path)
    catalog = build_catalog(populated.paths, populated.windows.sha256, "t", pins)

    status = {e.era: e.pe_pin_status for e in catalog.files}
    assert status == {"research": PIN_MATCH, "step2": PIN_MISMATCH, "hist": PIN_NONE}
    assert pins[("MES", "research", None)] == "a" * 64


@pytest.mark.parametrize(
    "bad",
    [
        {"schema": "other", "pins": []},
        {"schema": "pfb.pins/1", "pins": {}},
        {"schema": "pfb.pins/1", "pins": [{"root": "NQ", "era": "sealed", "plan": None, "sha256": "0" * 64}]},
        {"schema": "pfb.pins/1", "pins": [{"root": "NQ", "era": "research", "plan": None, "sha256": "x"}]},
        {"schema": "pfb.pins/1", "pins": [{"root": "NQ", "era": "research", "plan": None, "sha256": "0" * 64}]
         * 2},
    ],
)
def test_a_malformed_pins_file_is_refused(tmp_path: Path, bad: dict) -> None:
    path = tmp_path / "pins.json"
    path.write_text(json.dumps(bad), encoding="utf-8")

    with pytest.raises(CatalogError):
        load_pins(path)


def _paths_toml(stores: SynthStores) -> Path:
    p = stores.paths
    toml = stores.base / "paths.toml"
    toml.write_text(
        f'[stores]\nresearch = "{p.research}"\nstep2 = "{p.step2}"\nhist = "{p.hist}"\n'
        f'pins = "{stores.base / "pins.json"}"\n',
        encoding="utf-8",
    )
    return toml


@pytest.mark.parametrize(("step2_ok", "exit_code"), [(True, 0), (False, 1)])
def test_cli_catalog_writes_json_and_exits_non_zero_on_pin_mismatch(
    populated: SynthStores, step2_ok: bool, exit_code: int
) -> None:
    research = file_sha256(populated.paths.research / "NQ" / research_name("NQ"))
    step2 = file_sha256(populated.paths.step2 / "NQ" / step2_name("NQ"))
    _write_pins(populated, research_pin=research, step2_pin=step2 if step2_ok else "0" * 64)
    out = populated.base / "runs" / "catalog.json"

    result = CliRunner().invoke(
        app, ["data", "catalog", "--paths", str(_paths_toml(populated)), "--out", str(out)]
    )

    assert result.exit_code == exit_code, result.output
    assert "era hist: 1" in result.output
    assert ("PIN MISMATCH" in result.output) is (not step2_ok)
    assert len(load_catalog(out).files) == 3
