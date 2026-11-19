"""Cost models on synthetic book tables and hand-computed cents (no real cost value appears here)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from book_synth import SynthBook, bucket, product

from pfb.costs import (
    MINUTES_PER_DAY,
    CostModelError,
    CostSpecError,
    CostTableError,
    EventCostsOff,
    UnknownCostRoot,
    load_cost_model,
)
from pfb.costs.model import book_slippage_cents
from pfb.instruments import InstrumentTable

BUY, SELL = 0, 1
NQ_TV_CENTS = 500  # $5.00 per tick
ZN_TV_CENTS = 1562.5  # documented exact value, used only in comments


@pytest.fixture
def book(make_book: Callable[..., SynthBook], instruments: InstrumentTable) -> SynthBook:
    nq = [
        bucket(0, 30, 0.3, 0.301),
        bucket(30, 45, 0.1, 0.5),  # a 15-minute bucket
        # 45..59 is in no bucket: R-B2 fallback
        bucket(60, MINUTES_PER_DAY, 0.2, 0.2),
    ]
    zn = [bucket(0, MINUTES_PER_DAY, 0.5, 1.0)]
    return make_book({"NQ": product("NQ", instruments, nq), "ZN": product("ZN", instruments, zn)})


def _load(spec: str, book: SynthBook, instruments: InstrumentTable) -> Any:
    return load_cost_model(spec, instruments, book.paths, costs_toml=book.costs_toml)


def test_book_cents_follow_pe_expression_per_bucket(book: SynthBook, instruments: InstrumentTable) -> None:
    # Arrange
    model = _load("book", book, instruments)

    # Act
    one = model.slippage_cents("NQ", 1)
    three = model.slippage_cents("NQ", 3)

    # Assert: 0.3 x 500 = 150; 0.301 x 500 = 150.5 -> 151
    assert one[0, BUY] == 150 and one[29, SELL] == 151
    # 3 x 0.1 x 500 is 150.00000000000003 in float; upstream's round(, 6) makes it 150, not 151
    assert three[30, BUY] == 150
    assert one[44, SELL] == 250


def test_book_uses_bucket_minutes_not_ct_minute_div_30(book: SynthBook, instruments: InstrumentTable) -> None:
    model = _load("book", book, instruments)

    cents = model.slippage_cents("NQ", 1)
    calibrated = model.calibrated("NQ")

    # minute 44 is in the 15-minute bucket; 45..59 are in none and take the per-side maximum
    assert cents[44, BUY] == 50
    assert cents[45, BUY] == 150 and cents[59, SELL] == 250
    assert calibrated[44] and not calibrated[45] and not calibrated[59] and calibrated[60]
    assert int((~calibrated).sum()) == 15


def test_book_fractional_cent_tick_ceils_once_on_whole_quantity(
    book: SynthBook, instruments: InstrumentTable
) -> None:
    model = _load("book", book, instruments)

    # ZN 1562.5 c: 0.5 tick -> 781.25 -> 782; 1 tick -> 1563; qty 2 x 0.5 -> 1562.5 -> 1563 (not 2 x 782)
    assert model.slippage_cents("ZN", 1)[0, BUY] == 782
    assert model.slippage_cents("ZN", 1)[0, SELL] == 1563
    assert model.slippage_cents("ZN", 2)[100, BUY] == 1563


def test_book_expression_matches_pe_formula_directly() -> None:
    from fractions import Fraction

    assert book_slippage_cents(3, 0.1, Fraction(NQ_TV_CENTS)) == 150
    assert book_slippage_cents(1, 0.5, Fraction(3125, 2)) == 782


def test_book_root_without_row_is_refused(book: SynthBook, instruments: InstrumentTable) -> None:
    model = _load("book", book, instruments)

    with pytest.raises(UnknownCostRoot):
        model.slippage_cents("MES", 1)
    with pytest.raises(UnknownCostRoot):
        model.calibrated("ES")
    assert model.roots() == ("NQ", "ZN")


def test_tables_are_read_only_and_shaped(book: SynthBook, instruments: InstrumentTable) -> None:
    model = _load("book", book, instruments)

    cents = model.slippage_cents("NQ", 2)

    assert cents.shape == (MINUTES_PER_DAY, 2) and cents.dtype == np.int64
    assert model.calibrated("NQ").shape == (MINUTES_PER_DAY,)
    with pytest.raises(ValueError):
        cents[0, 0] = 0


@pytest.mark.parametrize("contracts", [0, -1, True, 1.0, "1"])
def test_bad_contracts_are_refused(contracts: Any, book: SynthBook, instruments: InstrumentTable) -> None:
    model = _load("book", book, instruments)
    with pytest.raises(CostModelError):
        model.slippage_cents("NQ", contracts)


def test_fixed_is_exact_rational(instruments: InstrumentTable) -> None:
    fixed_tenth = load_cost_model("fixed:0.1", instruments, None)
    fixed_one = load_cost_model("fixed:1", instruments, None)
    fixed_half = load_cost_model("fixed:0.5", instruments, None)

    # float would give ceil(150.00000000000003) = 151; exact 3 x 1/10 x 500 = 150
    assert int(fixed_tenth.slippage_cents("NQ", 3)[0, BUY]) == 150
    assert set(np.unique(fixed_one.slippage_cents("ZN", 1)).tolist()) == {1563}  # ceil(1562.5)
    assert int(fixed_one.slippage_cents("ZN", 2)[700, SELL]) == 3125
    assert int(fixed_half.slippage_cents("M6B", 1)[0, BUY]) == 32  # ceil(31.25)
    assert int(fixed_half.slippage_cents("M6B", 3)[0, BUY]) == 94  # ceil(93.75)
    assert int(load_cost_model("fixed:0", instruments, None).slippage_cents("NQ", 5).max()) == 0
    assert bool(fixed_one.calibrated("NQ").all())


def test_fixed_covers_every_instrument_root_and_refuses_others(instruments: InstrumentTable) -> None:
    model = load_cost_model("fixed:1", instruments, None)
    assert model.roots() == instruments.roots()
    with pytest.raises(UnknownCostRoot):
        model.slippage_cents("XYZ", 1)


def test_stress_adds_its_own_ceil_per_side(book: SynthBook, instruments: InstrumentTable) -> None:
    base = _load("book", book, instruments)
    stress = _load("stress:book:1", book, instruments)
    nested = load_cost_model("stress:fixed:0.5:0.5", instruments, None)

    diff = stress.slippage_cents("NQ", 2) - base.slippage_cents("NQ", 2)

    assert set(np.unique(diff).tolist()) == {2 * NQ_TV_CENTS}
    assert int(nested.slippage_cents("ZN", 1)[0, BUY]) == 782 + 782  # two ceils of 781.25
    assert np.array_equal(stress.calibrated("NQ"), base.calibrated("NQ"))
    assert stress.roots() == base.roots()
    with pytest.raises(UnknownCostRoot):
        stress.slippage_cents("MES", 1)


def test_model_ids_are_canonical_and_hashes_distinct(book: SynthBook, instruments: InstrumentTable) -> None:
    a = load_cost_model("fixed:1.50", instruments, None)
    b = load_cost_model("fixed:1.5", instruments, None)
    c = load_cost_model("fixed:2", instruments, None)
    s = load_cost_model("stress:stress:fixed:1:1:0.25", instruments, None)
    bk = _load("stress:book:1", book, instruments)

    assert a.model_id == "fixed:1.5" and a.config_sha256 == b.config_sha256
    assert a.config_sha256 != c.config_sha256
    assert s.model_id == "stress:stress:fixed:1:1:0.25"
    assert bk.model_id == "stress:book:1"
    assert len({a.config_sha256, c.config_sha256, s.config_sha256, bk.config_sha256}) == 4


def test_book_config_hash_follows_the_table(
    make_book: Callable[..., SynthBook], instruments: InstrumentTable
) -> None:
    one = make_book({"NQ": product("NQ", instruments, [bucket(0, 30, 0.1, 0.1)])})
    first = _load("book", one, instruments).config_sha256
    two = make_book({"NQ": product("NQ", instruments, [bucket(0, 30, 0.2, 0.1)])})
    second = _load("book", two, instruments).config_sha256
    assert first != second


@pytest.mark.parametrize(
    "spec",
    ["", "books", "book:1", "fixed:", "fixed:-1", "fixed:abc", "fixed:nan", "fixed:inf", "fixed: 1",
     "stress:book", "stress::1", "stress:book:", "stress:book:-0.5", "stress:nope:1", "slip150"],
)  # fmt: skip
def test_bad_specs_are_refused(spec: str, book: SynthBook, instruments: InstrumentTable) -> None:
    with pytest.raises(CostSpecError):
        _load(spec, book, instruments)


def test_book_needs_paths(instruments: InstrumentTable) -> None:
    with pytest.raises(CostSpecError):
        load_cost_model("book", instruments, None)


def test_a_fixed_model_has_no_event_cost(book: SynthBook, instruments: InstrumentTable) -> None:
    # book and stress over book carry an event table (tests/costs/test_events.py); fixed does not.
    for spec in ("fixed:1", "stress:fixed:1:1"):
        model = _load(spec, book, instruments)
        with pytest.raises(EventCostsOff):
            model.event_slippage_cents("NQ", 1)


def test_table_hash_differing_from_pin_is_refused(
    make_book: Callable[..., SynthBook], instruments: InstrumentTable
) -> None:
    book = make_book({"NQ": product("NQ", instruments, [bucket(0, 30, 0.1, 0.1)])}, pin="0" * 64)
    with pytest.raises(CostTableError, match="sha256"):
        _load("book", book, instruments)


def _bad_tables(instruments: InstrumentTable) -> list[tuple[str, dict[str, Any], str]]:
    nq_tv = float(instruments.get("NQ").tick_value_usd)
    bad_sum = bucket(0, 30, 0.1, 0.1)
    bad_sum["half_spread_ticks"] = 0.05
    return [
        ("fallback", {"NQ": product("NQ", instruments, [bucket(0, 30, 0.1, 0.1, fallback=True)])},
         "fallback"),
        ("overlap", {"NQ": product("NQ", instruments, [bucket(0, 30, 0.1, 0.1), bucket(20, 40, 0.1, 0.1)])},
         "overlaps"),
        ("outside", {"NQ": product("NQ", instruments, [bucket(1430, 1450, 0.1, 0.1)])}, "CT day"),
        ("negative", {"NQ": product("NQ", instruments, [bucket(0, 30, -0.1, 0.1)])}, ">= 0"),
        ("sum", {"NQ": product("NQ", instruments, [bad_sum])}, "half-spread"),
        ("tick", {"NQ": {"product": "NQ", "calibrated": True, "tick_value_usd": nq_tv * 2,
                         "buckets": [bucket(0, 30, 0.1, 0.1)]}}, "tick value"),
        ("uncal", {"NQ": {"product": "NQ", "calibrated": False, "tick_value_usd": nq_tv,
                          "buckets": [bucket(0, 30, 0.1, 0.1)]}}, "calibrated"),
        ("unknown", {"XYZ": {"product": "XYZ", "calibrated": True, "tick_value_usd": 1.0,
                             "buckets": [bucket(0, 30, 0.1, 0.1)]}}, "instrument table"),
    ]  # fmt: skip


@pytest.mark.parametrize("case", range(8))
def test_bad_book_tables_are_refused(
    case: int, make_book: Callable[..., SynthBook], instruments: InstrumentTable
) -> None:
    name, products, message = _bad_tables(instruments)[case]
    book = make_book(products)
    with pytest.raises(CostTableError, match=message):
        _load("book", book, instruments)


def test_provisional_table_is_refused(
    make_book: Callable[..., SynthBook], instruments: InstrumentTable
) -> None:
    book = make_book({"NQ": product("NQ", instruments, [bucket(0, 30, 0.1, 0.1)])}, depth_term="omitted")
    with pytest.raises(CostTableError, match="provisional"):
        _load("book", book, instruments)
