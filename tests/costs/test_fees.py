"""Fee schedules: the five tracked files against their A2 source cells, and the loader's refusals."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from pfb.costs import (
    FeeConfigError,
    UnknownFeeRoot,
    load_fees,
    side_from_round_turn,
    usd_to_cents,
)
from pfb.instruments import InstrumentTable

FIRMS = ("topstep", "bulenox", "lucid", "mffu", "tradeify")
HEADER = (
    'schema = "pfb.fees/1"\nfirm = "{firm}"\nlookup_date = 2026-10-21\n'
    'source_file = "x.json"\nsource_sha256 = "ab"\n'
)
ROW = (
    '\n[[fee]]\nroot = "{root}"\ncommission_side_cents = {side}\ncommission_rt_cents = {rt}\n'
    'source_file = "p.txt"\nlookup_date = {lk}\n'
)


def _write(tmp_path: Path, firm: str, rows: list[dict[str, object]]) -> Path:
    text = HEADER.format(firm=firm)
    for row in rows:
        text += ROW.format(**({"lk": "2026-10-21"} | row))
    (tmp_path / f"{firm}.toml").write_text(text, encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("firm", FIRMS)
def test_tracked_fee_file_loads_with_instrument_roots(firm: str, instruments: InstrumentTable) -> None:
    schedule = load_fees(firm, instruments=instruments)

    assert schedule.firm == firm and schedule.lookup_date == date(2026, 10, 21)
    assert schedule.roots() and len(schedule.sha256) == 64
    for root in schedule.roots():
        entry = schedule.entry(root)
        assert entry.commission_side_cents == side_from_round_turn(entry.commission_rt_cents)




def test_default_firm_is_topstep() -> None:
    assert load_fees().firm == "topstep"


def test_unlisted_root_is_refused() -> None:
    schedule = load_fees("topstep")
    with pytest.raises(UnknownFeeRoot):
        schedule.commission_side_cents("ES")


@pytest.mark.parametrize(
    ("usd", "cents"), [("3.78", 378), (3.78, 378), ("0.5", 50), (2, 200), ("1.890", 189)]
)
def test_usd_to_cents_exact(usd: str | float, cents: int) -> None:
    assert usd_to_cents(usd) == cents


@pytest.mark.parametrize("usd", ["1.005", 0.001, "abc", True, "nan"])
def test_usd_to_cents_refuses_fractional_or_bad(usd: object) -> None:
    with pytest.raises(FeeConfigError):
        usd_to_cents(usd)  # type: ignore[arg-type]


def test_side_from_round_turn_rounds_odd_cent_up() -> None:
    assert side_from_round_turn(378) == 189
    assert side_from_round_turn(379) == 190


def test_synthetic_file_loads(tmp_path: Path) -> None:
    fees_dir = _write(tmp_path, "acme", [{"root": "NQ", "side": 190, "rt": 379}])
    schedule = load_fees("acme", fees_dir=fees_dir)
    assert schedule.commission_side_cents("NQ") == 190
    assert schedule.entry("NQ").commission_rt_cents == 379


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"root": "NQ", "side": 188, "rt": 378}], "ceil"),
        ([{"root": "NQ", "side": 189, "rt": 378}, {"root": "NQ", "side": 189, "rt": 378}], "twice"),
        ([{"root": "NQ", "side": "true", "rt": 378}], "int"),
        ([{"root": "NQ", "side": 1.5, "rt": 3}], "int"),
        ([{"root": "NQ", "side": -1, "rt": -2}], ">= 0"),
        ([{"root": "NQ", "side": 189, "rt": 378, "lk": "2026-10-21T00:00:00"}], "date"),
        ([{"root": "XYZ", "side": 189, "rt": 378}], "instrument table"),
    ],
)
def test_bad_rows_are_refused(
    rows: list[dict[str, object]], message: str, tmp_path: Path, instruments: InstrumentTable
) -> None:
    fees_dir = _write(tmp_path, "acme", rows)
    with pytest.raises(FeeConfigError, match=message):
        load_fees("acme", fees_dir=fees_dir, instruments=instruments)


@pytest.mark.parametrize("firm", ["../topstep", "Topstep", "", "a/b", "x" * 40])
def test_bad_firm_names_are_refused(firm: str) -> None:
    with pytest.raises(FeeConfigError):
        load_fees(firm)


def test_missing_file_and_wrong_header_are_refused(tmp_path: Path) -> None:
    with pytest.raises(FeeConfigError, match="no fee file"):
        load_fees("nobody", fees_dir=tmp_path)
    other = HEADER.format(firm="other") + ROW.format(root="NQ", side=1, rt=2, lk="2026-10-21")
    (tmp_path / "acme.toml").write_text(other)
    with pytest.raises(FeeConfigError, match="firm"):
        load_fees("acme", fees_dir=tmp_path)
    (tmp_path / "acme.toml").write_text('schema = "pfb.fees/0"\n')
    with pytest.raises(FeeConfigError, match="schema"):
        load_fees("acme", fees_dir=tmp_path)
    (tmp_path / "acme.toml").write_text("not = [valid")
    with pytest.raises(FeeConfigError, match="TOML"):
        load_fees("acme", fees_dir=tmp_path)
