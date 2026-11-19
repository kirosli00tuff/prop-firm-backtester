"""A book's registration, its members' begin records and the gate of its pooled statistic, on a
temporary ledger in a temporary git repository (synthetic members; this repository's ledger is never
touched). Each test names the counting-rule clause it closes."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from book_support import (  # pyright: ignore[reportMissingImports]
    BOOK_ID,
    BOOK_REL,
    LATE_WINDOW,
    book_text,
    ledger_bytes,
    member_toml,
    records,
    write_book,
)
from registry_env import HYPOTHESIS_REL, SHA_B, STRATEGY_REL, Env  # pyright: ignore[reportMissingImports]

from pfb.registry.api import begin, finish, read_state, register_hypothesis
from pfb.registry.bookreg import book_refusals, book_runs, register_book
from pfb.registry.errors import (
    BookIncomplete,
    ChainBroken,
    EngineCheckStrategy,
    HypothesisChanged,
    InputNotCommitted,
    MemberAlreadyCounted,
    NotABookMember,
    PowerFlattered,
    PowerInsufficient,
    SpecMismatch,
    StrategyChanged,
    UndeclaredConfiguration,
)
from pfb.registry.powercheck import book_power_at
from pfb.registry.records import replay

K = 2


def _facts(env: Env, **override: object) -> list:
    return [env.facts(spec_contracts=1, **override)] * K


def _register(env: Env, text: str | None = None, rel: str = BOOK_REL) -> None:
    path = write_book(env, text if text is not None else book_text(env.strategy_sha), rel)
    register_book(path, env.config, _facts(env))
    env.checkpoint()


def _member(env: Env, a: int, book_id: str = BOOK_ID, rel: str = BOOK_REL, **override: object):
    request = env.request("trial", hypothesis_id=book_id, hypothesis_path=env.repo / rel, params={"a": a})
    env.checkpoint()
    return begin(replace(request, **override), config=env.config)


def _ok(env: Env, receipt: object) -> None:
    finish(receipt, "ok", {"sharpe_daily": 0.1}, [], config=env.config)  # type: ignore[arg-type]
    env.checkpoint()


# ------------------------------------------------------------------------------ registration ----
def test_registration_adds_a_book_record_and_nothing_to_n(genesis_env: Env) -> None:
    n_before = read_state(genesis_env.config).n
    _register(genesis_env)
    state = read_state(genesis_env.config)  # replay accepts the ledger
    record = state.hypotheses[BOOK_ID]
    assert state.n == n_before and record["n_after"] == record["n_before"] == n_before
    assert record["budget"] == K and record["grid"] is None and record["strategy_path"] is None
    assert [m["params"] for m in record["book"]["members"]] == [{"a": 1}, {"a": 2}]
    derived = record["book"]["derived"]
    assert (derived["union_days"], derived["trades_per_year"], derived["groups"]) == (30, 10000.0, [[0, 1]])
    assert 1414.2135 < derived["trade_sd_cents"] < 1414.2136  # clause 7: one tie group, sqrt(2) * 1000
    assert record["book"]["member_correlation"] == 0.0 and record["book"]["member_correlation_reason"]
    assert [m["root_family"] for m in record["book"]["members"]] == ["nasdaq100", "nasdaq100"]
    assert record["power"]["n_used"] == n_before + K  # clause 5: registration at N + K


def test_the_book_power_figures_are_the_books_over_the_union_of_member_windows(genesis_env: Env) -> None:
    rows = [
        member_toml(genesis_env.strategy_sha, a=1),
        member_toml(genesis_env.strategy_sha, a=2, window=LATE_WINDOW),
    ]
    _register(genesis_env, book_text(genesis_env.strategy_sha, members=rows, power__trades_per_year="6666.0"))
    record = read_state(genesis_env.config).hypotheses[BOOK_ID]
    expected = book_power_at(
        record["power"], [m["window"] for m in record["book"]["members"]], record["n_before"] + K
    )
    assert record["power"]["years"] == pytest.approx(45 / 365.25) == expected.years
    assert record["power"]["min_edge_cents"] == expected.min_edge_cents


def test_a_book_whose_power_check_fails_is_refused_and_nothing_appended(genesis_env: Env) -> None:
    path = write_book(genesis_env, book_text(genesis_env.strategy_sha, power__target_edge_cents="1"))
    before = ledger_bytes(genesis_env)
    with pytest.raises(PowerInsufficient):
        register_book(path, genesis_env.config, _facts(genesis_env))
    assert ledger_bytes(genesis_env) == before


@pytest.mark.parametrize(
    ("member", "error"),
    [
        ({"a": 4}, SpecMismatch),  # outside the SPEC grid
        ({"contracts": 2}, SpecMismatch),  # not the SPEC's contracts
    ],
)
def test_a_member_failing_a_spec_check_is_refused(genesis_env: Env, member: dict, error: type) -> None:
    sha = genesis_env.strategy_sha
    path = write_book(
        genesis_env, book_text(sha, members=[member_toml(sha, **member), member_toml(sha, a=2)])
    )
    with pytest.raises(error):
        register_book(path, genesis_env.config, _facts(genesis_env))


def test_a_member_whose_strategy_hash_differs_is_refused(genesis_env: Env) -> None:
    rows = [member_toml("0" * 64, a=1), member_toml(genesis_env.strategy_sha, a=2)]
    path = write_book(genesis_env, book_text(genesis_env.strategy_sha, members=rows))
    with pytest.raises(StrategyChanged):
        register_book(path, genesis_env.config, _facts(genesis_env))


def test_an_uncommitted_book_is_refused_and_the_dry_run_lists_every_refusal(genesis_env: Env) -> None:
    sha = genesis_env.strategy_sha
    path = genesis_env.write(
        BOOK_REL, book_text(sha, members=[member_toml(sha, a=4), member_toml(sha, a=2, contracts=3)])
    )
    before = ledger_bytes(genesis_env)
    with pytest.raises(InputNotCommitted):
        register_book(path, genesis_env.config, _facts(genesis_env))
    found = [type(e).__name__ for e in book_refusals(path, genesis_env.config, _facts(genesis_env))]
    assert found == ["InputNotCommitted", "SpecMismatch", "SpecMismatch"]
    assert ledger_bytes(genesis_env) == before


def test_the_dry_run_of_a_valid_book_finds_nothing_and_appends_nothing(genesis_env: Env) -> None:
    path = write_book(genesis_env, book_text(genesis_env.strategy_sha))
    before = ledger_bytes(genesis_env)
    assert book_refusals(path, genesis_env.config, _facts(genesis_env)) == []
    assert ledger_bytes(genesis_env) == before


def test_a_schema_two_hypothesis_after_a_book_carries_book_null_and_the_tier_never_falls(
    genesis_env: Env,
) -> None:
    _register(genesis_env)
    genesis_env.write(HYPOTHESIS_REL, genesis_env.hypothesis_text())
    genesis_env.commit()
    register_hypothesis(genesis_env.repo / HYPOTHESIS_REL, genesis_env.config, genesis_env.facts())
    recs = records(genesis_env)
    assert recs[-1]["book"] is None and read_state(genesis_env.config).hypotheses["H1"]["book"] is None
    stripped = {k: v for k, v in recs[-1].items() if k != "book"}
    with pytest.raises(ChainBroken, match="tier"):
        replay([*recs[:-1], stripped])


def test_replay_refuses_a_book_record_whose_power_was_flattered(genesis_env: Env) -> None:
    _register(genesis_env)
    genesis, book = records(genesis_env)[:2]
    flattered = {**book, "power": {**book["power"], "trades_per_year": 20000.0}}
    with pytest.raises(PowerFlattered):
        replay([genesis, flattered])
    tampered = {**book, "book": {**book["book"], "derived": {**book["book"]["derived"], "union_days": 60}}}
    with pytest.raises(ChainBroken, match="derived"):
        replay([genesis, tampered])


def test_replay_refuses_a_book_record_with_a_neighbourhood_or_a_wrong_budget(genesis_env: Env) -> None:
    _register(genesis_env)
    genesis, book = records(genesis_env)[:2]
    with pytest.raises(ChainBroken, match="neighbourhood"):
        replay([genesis, {**book, "suite": {**book["suite"], "neighbourhood": True}}])
    with pytest.raises(ChainBroken, match="budget"):
        replay([genesis, {**book, "budget": 3}])


# ------------------------------------------------------------------------------ member begins ----
def test_running_the_k_members_raises_n_by_exactly_k(genesis_env: Env) -> None:
    _register(genesis_env)
    n0 = read_state(genesis_env.config).n
    first = _member(genesis_env, 1)
    second = _member(genesis_env, 2)
    assert (first.n_after - first.n_before, second.n_after - second.n_before) == (1, 1)
    assert read_state(genesis_env.config).n == n0 + K
    # clause 5: each begin's power is the BOOK's at n_before + K - members already counted
    windows = [m["window"] for m in read_state(genesis_env.config).hypotheses[BOOK_ID]["book"]["members"]]
    for receipt, counted in ((first, 0), (second, 1)):
        n_used = receipt.n_before + K - counted
        assert receipt.record["power"]["n_used"] == n_used
        hyp = read_state(genesis_env.config).hypotheses[BOOK_ID]
        assert (
            receipt.record["power"]["min_edge_cents"]
            == book_power_at(hyp["power"], windows, n_used).min_edge_cents
        )


def test_an_exact_replay_of_a_member_adds_nothing(genesis_env: Env) -> None:
    _register(genesis_env)
    first = _member(genesis_env, 1)
    again = _member(genesis_env, 1)
    assert again.n_after == again.n_before == first.n_after


def test_a_non_member_configuration_is_refused(genesis_env: Env) -> None:
    _register(genesis_env)
    before = ledger_bytes(genesis_env)
    with pytest.raises(NotABookMember):
        _member(genesis_env, 3)  # inside the SPEC grid, but not a member
    with pytest.raises(NotABookMember):
        _member(genesis_env, 1, root="MES")
    with pytest.raises(UndeclaredConfiguration):
        _member(genesis_env, 1, first="2025-04-02")  # a window no member declares
    assert ledger_bytes(genesis_env) == before


def test_a_k_plus_first_configuration_of_a_counted_member_is_refused(genesis_env: Env) -> None:
    _register(genesis_env)
    _member(genesis_env, 1)
    _member(genesis_env, 2)
    n = read_state(genesis_env.config).n
    with pytest.raises(MemberAlreadyCounted):  # same member, other data hash: a new config_id
        _member(genesis_env, 1, data_files=(("research/g.parquet", SHA_B),))
    assert read_state(genesis_env.config).n == n


def test_a_member_recounted_before_the_others_ran_is_refused_too(genesis_env: Env) -> None:
    _register(genesis_env)
    _member(genesis_env, 1)
    with pytest.raises(MemberAlreadyCounted):
        _member(genesis_env, 1, data_files=(("research/g.parquet", SHA_B),))


def test_a_changed_book_file_is_refused_and_counts_again_as_a_new_hypothesis(genesis_env: Env) -> None:
    _register(genesis_env)
    _member(genesis_env, 1)
    write_book(genesis_env, book_text(genesis_env.strategy_sha, statement='"changed"'))
    with pytest.raises(HypothesisChanged):
        _member(genesis_env, 2)
    # the same members under a new id are a new hypothesis: each counts again
    _register(genesis_env, book_text(genesis_env.strategy_sha, book_id="B2"), rel="hypotheses/b2.toml")
    n = read_state(genesis_env.config).n
    receipt = _member(genesis_env, 1, book_id="B2", rel="hypotheses/b2.toml")
    assert receipt.n_after == n + 1


def test_a_member_strategy_file_changed_after_registration_is_refused(genesis_env: Env) -> None:
    _register(genesis_env)
    genesis_env.write(STRATEGY_REL, "x = 2\n")
    genesis_env.commit()
    with pytest.raises(NotABookMember):
        _member(genesis_env, 1, strategy_sha256=genesis_env.strategy_sha)


# ------------------------------------------------------------------------- the pooled statistic ----
def test_the_pooled_statistic_is_refused_for_a_partial_book_and_adds_nothing(genesis_env: Env) -> None:
    _register(genesis_env)
    with pytest.raises(BookIncomplete, match=r"\[1, 2\]"):
        book_runs(genesis_env.config, BOOK_ID)
    first = _member(genesis_env, 1)
    _ok(genesis_env, first)
    second = _member(genesis_env, 2)
    with pytest.raises(BookIncomplete, match=r"\[2\]"):  # begun, no result
        book_runs(genesis_env.config, BOOK_ID)
    finish(second, "failed", None, [], config=genesis_env.config, error="ZeroDivisionError")
    with pytest.raises(BookIncomplete, match=r"\[2\]"):  # failed is not ok
        book_runs(genesis_env.config, BOOK_ID)


def test_the_pooled_statistic_reads_the_k_runs_at_the_live_n_without_appending(genesis_env: Env) -> None:
    _register(genesis_env)
    receipts = [_member(genesis_env, a) for a in (1, 2)]
    for receipt in receipts:
        _ok(genesis_env, receipt)
    before = ledger_bytes(genesis_env)
    runs = book_runs(genesis_env.config, BOOK_ID)
    assert runs.run_ids == tuple(r.run_id for r in receipts)
    assert runs.n == read_state(genesis_env.config).n == receipts[-1].n_after
    assert ledger_bytes(genesis_env) == before  # clause 3: no record, nothing added to N


def test_the_real_ledger_is_not_touched(genesis_env: Env) -> None:
    real = Path(__file__).resolve().parents[2] / "ledger" / "registry.jsonl"
    before = real.read_bytes() if real.is_file() else None
    _register(genesis_env)
    _ok(genesis_env, _member(genesis_env, 1))
    assert (real.read_bytes() if real.is_file() else None) == before


def test_a_member_strategy_cannot_run_uncounted_as_an_engine_check(genesis_env: Env) -> None:
    # rule C5 per member: the book pins each member's strategy, so it cannot run as an uncounted check
    _register(genesis_env)
    genesis_env.write("config/engine_checks.toml", genesis_env.engine_checks_text())
    genesis_env.commit()
    before = ledger_bytes(genesis_env)
    with pytest.raises(EngineCheckStrategy):
        begin(genesis_env.request("engine_check"), config=genesis_env.config)
    assert ledger_bytes(genesis_env) == before
    path = write_book(genesis_env, book_text(genesis_env.strategy_sha, book_id="B3"), "hypotheses/b3.toml")
    with pytest.raises(EngineCheckStrategy):  # and an allowlisted strategy cannot be a member
        register_book(path, genesis_env.config, _facts(genesis_env))


def test_replay_refuses_a_book_record_whose_groups_or_rho_do_not_follow(genesis_env: Env) -> None:
    # clause 7e: the derived figure must follow from the recorded members, root families and rho
    env = genesis_env
    env.write("strategies/s2.py", "x = 3\n")  # a second strategy file on another root: two tie groups
    env.commit()
    from pfb.registry.chain import sha256_hex

    sha2 = sha256_hex((env.repo / "strategies/s2.py").read_bytes())
    rows = [
        member_toml(env.strategy_sha, a=1),
        member_toml(sha2, a=1, root="MES", strategy="strategies/s2.py"),
    ]
    _register(env, book_text(env.strategy_sha, members=rows, power__trade_sd_cents="1001"))
    genesis, book = records(env)[:2]
    assert book["book"]["derived"]["groups"] == [[0], [1]]

    def edited(**fields: object) -> dict:
        return {**book, "book": {**book["book"], **fields}}

    replay([genesis, book])
    with pytest.raises(ChainBroken, match="derived"):
        replay([genesis, edited(member_correlation=1.0)])  # rho 1 would give sqrt(2) * 1000
    with pytest.raises(ChainBroken, match="derived"):  # one family for both roots: one group
        replay([genesis, edited(members=[{**m, "root_family": "x"} for m in book["book"]["members"]])])
    members = [dict(m) for m in book["book"]["members"]]
    one_root = [members[0], {**members[1], "root": "MNQ", "root_family": "other"}]  # MNQ in two families
    with pytest.raises(ChainBroken, match="two root families"):
        replay([genesis, {**edited(members=one_root), "roots": ["MNQ"]}])
    for rho, reason in ((1.5, "x"), (-0.1, "x"), (0.5, ""), (0.5, "  ")):
        with pytest.raises(ChainBroken, match="member_correlation"):
            replay([genesis, edited(member_correlation=rho, member_correlation_reason=reason)])
    split = {**book["book"]["derived"], "groups": [[0, 1]]}
    with pytest.raises(ChainBroken, match="derived"):
        replay([genesis, edited(derived=split)])
