"""Tests of the instrument table (config/instruments.toml, ADR-001 sections 3, 4, A5(d), A9)."""

import hashlib
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from pfb.instruments import (
    DEFAULT_INSTRUMENTS_TOML,
    Instrument,
    InstrumentTable,
    InstrumentTableError,
    UnknownRoot,
    load_instruments,
)

EXPECTED_ROOT_COUNT = 47
EXPECTED_ADMISSIBLE_COUNT = 45
FRACTIONAL_CENT_ROOTS = ("ZT", "ZF", "ZN", "TN", "M6B")
FACTOR_100_ROOTS = ("ZC", "ZW", "ZS", "ZL", "HE", "LE")

# root -> (vendor_tick_fixed, tick_value_cc, tick_value_cents), worked by hand.
HAND_CHECKED: dict[str, tuple[int, int, Fraction]] = {
    # five roots whose tick value is not a whole number of cents
    "ZT": (3_906_250, 78_125, Fraction(3125, 4)),  # 0.00390625 -> 3906250; $7.8125 = 781.25 c
    "ZF": (7_812_500, 78_125, Fraction(3125, 4)),  # 0.0078125 -> 7812500; $7.8125 = 781.25 c
    "ZN": (15_625_000, 156_250, Fraction(3125, 2)),  # 0.015625 -> 15625000; $15.625 = 1562.5 c
    "TN": (15_625_000, 156_250, Fraction(3125, 2)),  # as ZN
    "M6B": (100_000, 6_250, Fraction(125, 2)),  # 0.0001 -> 100000; $0.625 = 62.5 c
    # six roots the vendor quotes in cents (factor 100): vendor tick = 100 x exchange tick
    "ZC": (250_000_000, 125_000, Fraction(1250)),  # 0.0025 x 100 = 0.25; $12.50
    "ZW": (250_000_000, 125_000, Fraction(1250)),
    "ZS": (250_000_000, 125_000, Fraction(1250)),
    "ZL": (10_000_000, 60_000, Fraction(600)),  # 0.0001 x 100 = 0.01; $6.00
    "HE": (25_000_000, 100_000, Fraction(1000)),  # 0.00025 x 100 = 0.025; $10.00
    "LE": (25_000_000, 100_000, Fraction(1000)),
    # a few whole-cent, factor-1 roots
    "NQ": (250_000_000, 50_000, Fraction(500)),
    "6J": (500, 62_500, Fraction(625)),  # 0.0000005 -> 500
    "MBT": (5_000_000_000, 5_000, Fraction(50)),
}


@pytest.fixture(scope="module")
def table() -> InstrumentTable:
    return load_instruments()


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


_ONE_ROW = """
schema = "pfb.instruments/1"
[[instrument]]
root = "NQ"
group = "equity"
subgroup = "equity"
exposure = "nasdaq100"
contract_type = "mini"
tick_size = "{tick_size}"
tick_value_usd = "{tick_value_usd}"
multiplier_usd = "20"
vendor_price_factor = {factor}
admissible = true
tick_source = "t"
weight_source = "w"
"""


def _one_row(
    tmp_path: Path, *, tick_size: str = "0.25", tick_value_usd: str = "5.00", factor: int = 1
) -> Path:
    body = _ONE_ROW.format(tick_size=tick_size, tick_value_usd=tick_value_usd, factor=factor)
    return _write(tmp_path / "instruments.toml", body)


def test_loads_all_47_roots_in_file_order(table: InstrumentTable) -> None:
    roots = table.roots()

    assert len(roots) == EXPECTED_ROOT_COUNT
    assert roots[:2] == ("NQ", "MNQ")
    assert roots[-1] == "MBT"
    assert len(set(roots)) == EXPECTED_ROOT_COUNT
    assert "ES" in roots and "MES" in roots


def test_admissible_flag_marks_45_roots_and_not_es_or_mes(table: InstrumentTable) -> None:
    admissible = [r for r in table.roots() if table.get(r).admissible]

    assert len(admissible) == EXPECTED_ADMISSIBLE_COUNT
    assert not table.get("ES").admissible
    assert not table.get("MES").admissible


def test_sha256_is_the_hash_of_the_file_bytes(table: InstrumentTable) -> None:
    expected = hashlib.sha256(DEFAULT_INSTRUMENTS_TOML.read_bytes()).hexdigest()

    assert table.sha256 == expected


def test_fields_have_the_adr_types(table: InstrumentTable) -> None:
    nq = table.get("NQ")

    assert isinstance(nq, Instrument)
    assert isinstance(nq.tick_size, Decimal)
    assert isinstance(nq.tick_value_usd, Decimal)
    assert isinstance(nq.multiplier_usd, Decimal)
    assert isinstance(nq.vendor_price_factor, int)
    assert isinstance(nq.vendor_tick_fixed, int)
    assert isinstance(nq.tick_value_cc, int)
    assert isinstance(nq.tick_value_cents, Fraction)


@pytest.mark.parametrize(("root", "expected"), sorted(HAND_CHECKED.items()))
def test_derived_values_match_hand_worked_numbers(
    table: InstrumentTable, root: str, expected: tuple[int, int, Fraction]
) -> None:
    fixed, cc, cents = expected
    instrument = table.get(root)

    assert instrument.vendor_tick_fixed == fixed
    assert instrument.tick_value_cc == cc
    assert instrument.tick_value_cents == cents
    assert table.tick_fixed_of(root) == fixed


def test_only_the_five_named_roots_have_a_fractional_cent_tick_value(table: InstrumentTable) -> None:
    fractional = tuple(r for r in table.roots() if table.get(r).tick_value_cents.denominator != 1)

    assert set(fractional) == set(FRACTIONAL_CENT_ROOTS)


def test_only_the_six_named_roots_have_price_factor_100(table: InstrumentTable) -> None:
    factors = {r: table.price_factor_of(r) for r in table.roots()}

    assert {r for r, f in factors.items() if f == 100} == set(FACTOR_100_ROOTS)
    assert all(f in (1, 100) for f in factors.values())


def test_every_tick_value_equals_tick_times_multiplier(table: InstrumentTable) -> None:
    for root in table.roots():
        instrument = table.get(root)
        assert instrument.tick_size * instrument.multiplier_usd == instrument.tick_value_usd, root


def test_vendor_tick_fixed_is_vendor_tick_in_1e9_units(table: InstrumentTable) -> None:
    for root in table.roots():
        instrument = table.get(root)
        exact = instrument.tick_size * instrument.vendor_price_factor * 1_000_000_000
        assert exact == instrument.vendor_tick_fixed, root


def test_unknown_root_raises_unknown_root(table: InstrumentTable) -> None:
    with pytest.raises(UnknownRoot):
        table.get("XX")
    with pytest.raises(UnknownRoot):
        table.tick_fixed_of("XX")
    with pytest.raises(UnknownRoot):
        table.price_factor_of("XX")


def test_roots_are_case_sensitive(table: InstrumentTable) -> None:
    with pytest.raises(UnknownRoot):
        table.get("nq")


def test_instrument_is_frozen(table: InstrumentTable) -> None:
    nq = table.get("NQ")

    with pytest.raises(AttributeError):
        nq.root = "ES"  # type: ignore[misc]


def test_loads_a_one_row_table_from_an_explicit_path(tmp_path: Path) -> None:
    path = _one_row(tmp_path)

    loaded = load_instruments(path)

    assert loaded.roots() == ("NQ",)
    assert loaded.tick_fixed_of("NQ") == 250_000_000


def test_rejects_a_vendor_tick_that_is_not_whole_1e9_units(tmp_path: Path) -> None:
    path = _one_row(tmp_path, tick_size="0.0000000005", tick_value_usd="0.00000001")

    with pytest.raises(InstrumentTableError, match="vendor tick"):
        load_instruments(path)


def test_rejects_a_tick_value_that_is_not_whole_1e4_usd(tmp_path: Path) -> None:
    path = _one_row(tmp_path, tick_value_usd="5.00001")

    with pytest.raises(InstrumentTableError, match="tick value"):
        load_instruments(path)


def test_rejects_a_wrong_schema(tmp_path: Path) -> None:
    path = _one_row(tmp_path)
    path.write_text(path.read_text().replace("pfb.instruments/1", "pfb.instruments/2"), encoding="utf-8")

    with pytest.raises(InstrumentTableError, match="schema"):
        load_instruments(path)


def test_rejects_a_duplicate_root(tmp_path: Path) -> None:
    path = _one_row(tmp_path)
    row = path.read_text().split('schema = "pfb.instruments/1"\n', 1)[1]
    path.write_text(path.read_text() + row, encoding="utf-8")

    with pytest.raises(InstrumentTableError, match="duplicate"):
        load_instruments(path)


def test_rejects_a_missing_field(tmp_path: Path) -> None:
    path = _one_row(tmp_path)
    path.write_text(path.read_text().replace('exposure = "nasdaq100"\n', ""), encoding="utf-8")

    with pytest.raises(InstrumentTableError, match="exposure"):
        load_instruments(path)


def test_rejects_a_float_where_a_decimal_string_belongs(tmp_path: Path) -> None:
    path = _one_row(tmp_path)
    path.write_text(path.read_text().replace('tick_size = "0.25"', "tick_size = 0.25"), encoding="utf-8")

    with pytest.raises(InstrumentTableError, match="tick_size"):
        load_instruments(path)


def test_rejects_a_zero_or_negative_price_factor(tmp_path: Path) -> None:
    path = _one_row(tmp_path, factor=0)

    with pytest.raises(InstrumentTableError, match="vendor_price_factor"):
        load_instruments(path)


def test_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(InstrumentTableError, match="cannot read"):
        load_instruments(tmp_path / "absent.toml")


def test_rejects_malformed_toml(tmp_path: Path) -> None:
    path = _write(tmp_path / "bad.toml", "schema = [unclosed")

    with pytest.raises(InstrumentTableError, match="not valid TOML"):
        load_instruments(path)
