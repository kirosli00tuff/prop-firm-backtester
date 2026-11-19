"""The book file (schema "pfb.hypothesis/3"), its parse refusals and the
derived pooled power inputs (5, 6, 7). Pure: no ledger, no repository."""

from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from book_support import LATE_WINDOW, WINDOW, book_text, member_toml  # pyright: ignore[reportMissingImports]

from pfb.registry.bookfile import (
    SQRT_SCALE,
    BookHypothesis,
    PoolMember,
    derived_power,
    member_groups,
    parse_book,
    pool_members,
    root_family,
    sqrt_up,
    union_days,
)
from pfb.registry.errors import BookInvalid, PowerFlattered
from pfb.registry.hypfile import Hypothesis, parse_hypothesis

SHA = "f" * 64
PLUMBING_FILE = Path(__file__).resolve().parents[2] / "hypotheses" / "h_example_plumbing.toml"


def _parse(text: str) -> BookHypothesis:
    parsed = parse_hypothesis(text.encode(), "b.toml")
    assert isinstance(parsed, BookHypothesis)
    return parsed


def test_a_book_file_parses_through_the_schema_dispatch() -> None:
    book = _parse(book_text(SHA))
    assert (book.id, book.purpose, book.budget, book.account, book.cap, book.statistic) == (
        "B1", "research", 2, "topstep/combine_xfa_standard_50k", "plan", "daily_net",
    )  # fmt: skip
    assert [dict(m.params) for m in book.members] == [{"a": 1}, {"a": 2}]
    assert book.windows == (WINDOW,) and book.roots == ("MNQ",) and dict(book.grid) == {}


def test_the_schema_two_plumbing_file_still_parses_to_the_same_object() -> None:
    if not PLUMBING_FILE.is_file():
        pytest.skip("hypotheses/h_example_plumbing.toml not present")
    data = PLUMBING_FILE.read_bytes()
    first, second = parse_hypothesis(data, "plumbing"), parse_hypothesis(data, "plumbing")
    assert isinstance(first, Hypothesis) and first == second and first.id == "h_example_plumbing"


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"purpose": '"test"'}, "purpose"),
        ({"budget": "3"}, "budget 3 must equal"),
        ({"budget": "1"}, "budget 1 must equal"),
        ({"suite__neighbourhood": "true"}, "neighbourhood"),
        ({"power__sd_source": '"run:synthetic-x"'}, "sd_source"),
        ({"book__cap": '"none"'}, "cap"),
        ({"book__statistic": '"sharpe"'}, "statistic"),
        ({"book__account": '"topstep"'}, "account"),
        ({"strategy": '"strategies/s.py"'}, "extra"),
        ({"grid": "1"}, "extra"),
        ({"book__window": "1"}, r"\[book\] has exactly"),
    ],
)
def test_a_book_breaking_clause_one_or_six_is_refused(override: dict[str, str], match: str) -> None:
    with pytest.raises(BookInvalid, match=match):
        parse_book(book_text(SHA, **override).encode(), "b.toml")


def test_a_book_of_one_member_is_refused() -> None:
    with pytest.raises(BookInvalid, match="at least 2"):
        parse_book(book_text(SHA, members=[member_toml(SHA)]).encode(), "b.toml")


def test_two_equal_members_are_refused_even_at_other_contracts() -> None:
    rows = [member_toml(SHA, a=1), member_toml(SHA, a=1, contracts=2)]
    with pytest.raises(BookInvalid, match="share strategy, root, params and window"):
        parse_book(book_text(SHA, members=rows).encode(), "b.toml")


def test_a_member_with_an_extra_or_missing_key_is_refused() -> None:
    row = member_toml(SHA).replace("contracts = 1\n", "")
    with pytest.raises(BookInvalid, match="exactly the keys"):
        parse_book(book_text(SHA, members=[row, member_toml(SHA, a=2)]).encode(), "b.toml")


def test_the_union_of_member_windows_counts_each_day_once() -> None:
    assert union_days([WINDOW, LATE_WINDOW]) == 45
    assert union_days([WINDOW, WINDOW]) == 30
    assert union_days([("2025-04-01", "2025-04-10"), ("2025-04-11", "2025-04-20")]) == 20
    assert union_days([("2025-04-01", "2025-04-10"), ("2025-04-12", "2025-04-20")]) == 19


def _pool(i: int, family: str, window: tuple[str, str] = WINDOW, sd: int = 1000) -> PoolMember:
    """Member i on its own strategy file and sha256 (so only the root family or a strategy can tie it)."""
    return PoolMember(5000.0, sd, window, family, f"strategies/s{i}.py", f"{i:x}" * 64)


def _between(value: Fraction, exact_square: Fraction, steps: int = 10) -> bool:
    """value is sqrt(exact_square) rounded up: never below it, and less than `steps` / SQRT_SCALE above (the
    a_i and the final root are each rounded up, so a derived figure may sit a few steps above)."""
    return value * value >= exact_square and (value - Fraction(steps, SQRT_SCALE)) ** 2 < exact_square


def test_the_pooled_rate_is_derived_from_the_members() -> None:
    derived = derived_power([_pool(1, "a"), _pool(2, "b", LATE_WINDOW)], Fraction(0))
    # m = sum(m_i Y_i) / Y = (5000 * 30 + 5000 * 30) / 45 over the union of the two windows
    assert derived.union_days == 45 and derived.trades_per_year == Fraction(5000 * 30 * 2, 45)


@pytest.mark.parametrize("k", [2, 3, 5])
def test_k_tied_members_give_the_linear_sum_sqrt_k_above_the_independent_figure(k: int) -> None:
    tied_book = derived_power([_pool(i + 1, "nasdaq100") for i in range(k)], Fraction(0))
    independent = derived_power([_pool(i + 1, f"family{i}") for i in range(k)], Fraction(0))
    assert tied_book.groups == (tuple(range(k)),)
    assert independent.groups == tuple((i,) for i in range(k))
    assert _between(tied_book.trade_sd_cents, Fraction(k * 1000**2))  # sqrt(K) * 1000: the linear sum
    assert _between(independent.trade_sd_cents, Fraction(1000**2))  # clause 5's old figure


def test_members_on_different_roots_with_rho_zero_give_the_old_figure() -> None:
    derived = derived_power([_pool(1, "a", sd=1000), _pool(2, "b", sd=2000)], Fraction(0))
    assert _between(derived.trade_sd_cents, Fraction(1000**2 + 2000**2, 2))


def test_rho_one_across_groups_equals_one_big_group() -> None:
    apart = [_pool(1, "a", sd=1000), _pool(2, "b", sd=2000), _pool(3, "c", sd=3000)]
    together = [_pool(1, "x", sd=1000), _pool(2, "x", sd=2000), _pool(3, "x", sd=3000)]
    assert (
        derived_power(apart, Fraction(1)).trade_sd_cents
        == derived_power(together, Fraction(0)).trade_sd_cents
    )
    half = derived_power(apart, Fraction(1, 2)).trade_sd_cents
    assert (
        derived_power(apart, Fraction(0)).trade_sd_cents
        < half
        < derived_power(apart, Fraction(1)).trade_sd_cents
    )


def test_a_shared_strategy_file_or_sha_ties_members_on_other_roots() -> None:
    same_path = replace(_pool(2, "b"), strategy_path="strategies/s1.py")
    same_sha = replace(_pool(3, "c"), strategy_sha256=_pool(1, "a").strategy_sha256)
    assert member_groups([_pool(1, "a"), same_path]) == ((0, 1),)
    assert member_groups([_pool(1, "a"), same_sha]) == ((0, 1),)
    assert member_groups([_pool(1, "a"), _pool(2, "b")]) == ((0,), (1,))


def test_a_micro_and_its_full_size_sibling_count_as_one_root() -> None:
    assert root_family("MNQ") == root_family("NQ") and root_family("MES") == root_family("ES")
    assert root_family("MNQ") != root_family("MES")
    rows = [
        member_toml("1" * 64, a=1, root="MNQ", strategy="strategies/one.py"),
        member_toml("2" * 64, a=1, root="NQ", strategy="strategies/two.py"),
    ]
    book = _parse(book_text(SHA, members=rows))
    assert member_groups(pool_members(book.members)) == ((0, 1),)


def test_disjoint_windows_on_one_root_are_not_tied() -> None:
    early, late = ("2025-04-01", "2025-04-10"), ("2025-04-11", "2025-04-20")
    assert member_groups([_pool(1, "x", early), _pool(2, "x", late)]) == ((0,), (1,))
    shared_day = ("2025-04-10", "2025-04-20")  # shares 2025-04-10 with early
    assert member_groups([_pool(1, "x", early), _pool(2, "x", shared_day)]) == ((0, 1),)
    # ties are transitive: early - bridge - late form one group although early and late share no day
    bridge = ("2025-04-10", "2025-04-11")
    assert member_groups([_pool(1, "x", early), _pool(2, "x", late), _pool(3, "x", bridge)]) == ((0, 1, 2),)


@pytest.mark.parametrize(
    "override",
    [
        {"power__member_correlation": "<drop>"},
        {"power__member_correlation": "-0.1"},
        {"power__member_correlation": "1.1"},
        {"power__member_correlation": "nan"},
        {"power__member_correlation": '"0.5"'},
        {"power__member_correlation": "true"},
        {"power__member_correlation_reason": "<drop>"},
        {"power__member_correlation_reason": '""'},
        {"power__member_correlation_reason": '"   "'},
    ],
)
def test_a_missing_or_out_of_range_rho_or_an_empty_reason_is_refused(override: dict[str, str]) -> None:
    with pytest.raises(BookInvalid, match="member_correlation"):
        parse_book(book_text(SHA, **override).encode(), "b.toml")


def test_rho_at_its_bounds_is_accepted_and_kept() -> None:
    for rho in ("0", "1", "0.25"):
        assert _parse(book_text(SHA, power__member_correlation=rho)).member_correlation == float(rho)


def test_the_stated_power_at_exactly_the_derived_inputs_is_accepted() -> None:
    book = _parse(book_text(SHA))
    assert (book.power.trades_per_year, book.power.trade_sd_cents) == (10000.0, 1415)


def test_square_roots_are_rounded_up_never_down() -> None:
    assert sqrt_up(Fraction(4)) == 2 and sqrt_up(Fraction(0)) == 0
    for value in (Fraction(2), Fraction(10**12 + 1), Fraction(1, 3), Fraction(2_500_000)):
        assert _between(sqrt_up(value), value, steps=1)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"power__trades_per_year": "10000.5"}, "trades_per_year"),
        ({"power__trade_sd_cents": "999"}, "trade_sd_cents"),
    ],
)
def test_a_flattered_power_table_is_refused(override: dict[str, str], match: str) -> None:
    with pytest.raises(PowerFlattered, match=match):
        parse_book(book_text(SHA, **override).encode(), "b.toml")


def test_overlapping_member_windows_lower_the_derived_rate_so_the_old_figure_flatters() -> None:
    rows = [member_toml(SHA, a=1), member_toml(SHA, a=2, window=LATE_WINDOW)]
    # derived m = 300000 / 45 = 6666.67 trades a year: stating 10000 (the naive sum) is refused
    with pytest.raises(PowerFlattered, match="trades_per_year"):
        parse_book(book_text(SHA, members=rows).encode(), "b.toml")
    book = _parse(book_text(SHA, members=rows, power__trades_per_year="6666.0"))
    assert book.windows == (WINDOW, LATE_WINDOW)


def test_the_stated_sd_one_cent_below_the_derived_is_refused_and_at_it_accepted() -> None:
    # one tie group of two equal members: derived sd sqrt(2) * 1000 = 1414.21, rounded up 1415
    derived = derived_power(pool_members(_parse(book_text(SHA)).members), Fraction(0))
    at = math.ceil(derived.trade_sd_cents)
    assert at == 1415
    with pytest.raises(PowerFlattered, match="trade_sd_cents"):
        parse_book(book_text(SHA, power__trade_sd_cents=str(at - 1)).encode(), "b.toml")
    assert _parse(book_text(SHA, power__trade_sd_cents=str(at))).power.trade_sd_cents == at
    # members on two strategies and two root families, rho 0: clause 5's sqrt(2_500_000) = 1581.14
    rows = [
        member_toml("1" * 64, a=1, root="MNQ", sd="1000", strategy="strategies/one.py"),
        member_toml("2" * 64, a=1, root="MES", sd="2000", strategy="strategies/two.py"),
    ]
    with pytest.raises(PowerFlattered):
        parse_book(book_text(SHA, members=rows, power__trade_sd_cents="1581").encode(), "b.toml")
    assert _parse(book_text(SHA, members=rows, power__trade_sd_cents="1582")).power.trade_sd_cents == 1582
    # the same two members with rho 1: one big group, (1000 + 2000) / sqrt(2) = 2121.32
    with pytest.raises(PowerFlattered):
        parse_book(
            book_text(
                SHA, members=rows, power__trade_sd_cents="2121", power__member_correlation="1"
            ).encode(),
            "b",
        )
    assert _parse(book_text(SHA, members=rows, power__trade_sd_cents="2122", power__member_correlation="1"))
