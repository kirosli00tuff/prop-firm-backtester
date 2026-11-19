"""Trial rulings: at most one open TEST hypothesis, artifact
sha256s in the result record, and one program-wide look-ahead seed. Temporary ledgers in temporary git
repositories only (tests/trials/conftest.py), never the real one."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from book_support import BOOK_REL, book_text, write_book  # pyright: ignore[reportMissingImports]
from registry_env import Env  # pyright: ignore[reportMissingImports]
from trial_support import FakeTrialEngine, kinds, n_of  # pyright: ignore[reportMissingImports]

from pfb.registry.api import begin, finish, read_state, register_hypothesis
from pfb.registry.artifacts import require_folder_matches, verify_folders
from pfb.registry.bookfile import parse_book
from pfb.registry.bookreg import register_book
from pfb.registry.chain import read_ledger
from pfb.registry.errors import (
    ArtifactMismatch,
    ChainBroken,
    LookaheadSeedUnexplained,
    OpenTestHypothesis,
    SeedMismatch,
)
from pfb.registry.hypfile import PROGRAM_LOOKAHEAD_SEED, parse_hypothesis
from pfb.registry.records import (
    ARTIFACT_SHA_KEY,
    BOOK_KEY,
    KEYS,
    SEED_KEY,
    SEED_REASON_KEY,
    replay,
)

T1_REL, T2_REL, R1_REL = "hypotheses/t1.toml", "hypotheses/t2.toml", "hypotheses/r1.toml"
REASON_TEXT = "step 1 sees no entry at seed 0 on these synthetic bars"
REASON = f'"{REASON_TEXT}"'


def _register(env: Env, rel: str, text: str) -> Any:
    env.write(rel, text)
    env.commit()
    return register_hypothesis(env.repo / rel, env.config, env.facts())


def _records(env: Env) -> list[dict[str, Any]]:
    return [dict(line.record) for line in read_ledger(env.config.registry_path)]


def _older_tier(record: dict[str, Any]) -> dict[str, Any]:
    """A hypothesis record as an older checkout wrote it (tier 1: no book, no seed keys)."""
    return {k: v for k, v in record.items() if k not in (BOOK_KEY, SEED_KEY, SEED_REASON_KEY)}


# ------------------------------------------------------------------------- one open TEST ----
def test_a_second_test_hypothesis_is_refused_until_the_first_is_spent(genesis_env: Env) -> None:
    # Arrange: T1, a TEST hypothesis (budget 1), registered and not yet run
    _register(genesis_env, T1_REL, genesis_env.hypothesis_text(budget=1, hyp_id="T1"))
    before = genesis_env.config.registry_path.read_bytes()
    # Act / Assert: T2 is refused while T1 is open; nothing is appended
    with pytest.raises(OpenTestHypothesis, match="'T1'"):
        _register(genesis_env, T2_REL, genesis_env.hypothesis_text(budget=1, hyp_id="T2"))
    assert genesis_env.config.registry_path.read_bytes() == before
    # a research hypothesis is never held back by an open TEST one
    _register(genesis_env, R1_REL, genesis_env.hypothesis_text(budget=2, hyp_id="R1"))
    # T1's one trial spends its budget (a crash after begin would spend it too): T2 now registers
    genesis_env.begin_trial(hypothesis_id="T1", hypothesis_path=genesis_env.repo / T1_REL)
    line = _register(genesis_env, T2_REL, genesis_env.hypothesis_text(budget=1, hyp_id="T2"))
    assert line.record["purpose"] == "TEST" and line.record["n_after"] == n_of(genesis_env)
    assert kinds(genesis_env) == ["genesis", "hypothesis", "hypothesis", "trial", "hypothesis"]


def test_replay_applies_the_one_open_test_rule_to_new_records_only(genesis_env: Env) -> None:
    _register(genesis_env, T1_REL, genesis_env.hypothesis_text(budget=1, hyp_id="T1"))
    genesis, t1 = _records(genesis_env)
    t2 = {**t1, "hypothesis_id": "T2"}
    # two TEST records written before (tier 1): replay accepts them, as it accepts the real ledger
    assert set(replay([genesis, _older_tier(t1), _older_tier(t2)]).hypotheses) == {"T1", "T2"}
    # a new (tier 3) TEST record while an older open TEST record exists: refused
    with pytest.raises(OpenTestHypothesis):
        replay([genesis, _older_tier(t1), t2])


# --------------------------------------------------------------------- the program seed ----
def test_the_program_seed_is_zero_and_the_default_everywhere() -> None:
    from pfb.app import validate_strategy
    from pfb.registry.hypfile import DEFAULT_LOOKAHEAD_SEED
    from pfb.runner.api import DEFAULT_SEED

    defaults = validate_strategy.__kwdefaults__ or {}
    assert PROGRAM_LOOKAHEAD_SEED == 0 == DEFAULT_LOOKAHEAD_SEED == DEFAULT_SEED == defaults["seed"]


def test_another_seed_needs_a_reason_in_a_schema_2_file(genesis_env: Env) -> None:
    text = genesis_env.hypothesis_text(lookahead_seed="7")
    with pytest.raises(LookaheadSeedUnexplained, match="lookahead_seed_reason"):
        parse_hypothesis(text.encode(), "h.toml")
    blank = genesis_env.hypothesis_text(lookahead_seed="7", lookahead_seed_reason='"  "')
    with pytest.raises(LookaheadSeedUnexplained):
        parse_hypothesis(blank.encode(), "h.toml")
    text = genesis_env.hypothesis_text(hyp_id="R1", lookahead_seed="7", lookahead_seed_reason=REASON)
    line = _register(genesis_env, R1_REL, text)
    assert (line.record[SEED_KEY], line.record[SEED_REASON_KEY]) == (7, REASON_TEXT)
    assert set(line.record) == KEYS["hypothesis"]


def test_the_program_seed_needs_no_reason_and_is_recorded(trial_env: Env) -> None:
    record = read_state(trial_env.config).hypotheses["H1"]
    assert (record[SEED_KEY], record[SEED_REASON_KEY]) == (PROGRAM_LOOKAHEAD_SEED, "")


def test_another_seed_needs_a_reason_in_a_book_and_the_book_record_carries_both(genesis_env: Env) -> None:
    sha = genesis_env.strategy_sha
    with pytest.raises(LookaheadSeedUnexplained):
        parse_book(book_text(sha, lookahead_seed="5").encode(), "b1.toml")
    path = write_book(genesis_env, book_text(sha, lookahead_seed="5", lookahead_seed_reason=REASON), BOOK_REL)
    line = register_book(path, genesis_env.config, [genesis_env.facts(spec_contracts=1)] * 2)
    assert line.record[SEED_KEY] == 5 and line.record[SEED_REASON_KEY]
    assert line.record[BOOK_KEY] is not None and set(line.record) == KEYS["hypothesis"]


def test_replay_refuses_an_unexplained_seed_and_a_trial_at_another_seed(genesis_env: Env) -> None:
    text = genesis_env.hypothesis_text(lookahead_seed="7", lookahead_seed_reason=REASON)
    _register(genesis_env, "hypotheses/h1.toml", text)  # H1, at seed 7
    receipt = genesis_env.begin_trial(seed=7)
    assert receipt.record["seed"] == 7
    genesis, hyp, trial = _records(genesis_env)
    with pytest.raises(LookaheadSeedUnexplained):
        replay([genesis, {**hyp, SEED_REASON_KEY: ""}])
    with pytest.raises(SeedMismatch):
        replay([genesis, hyp, {**trial, "seed": 0}])
    # a trial under an older hypothesis record (no stored seed) is not checked against one
    assert replay([genesis, _older_tier(hyp), {**trial, "seed": 0}]).n == trial["n_after"]


def test_pfb_check_at_another_seed_is_marked_as_not_the_program_seed() -> None:
    from pfb.app import not_program_seed_text

    text = not_program_seed_text(3)
    assert "not the program look-ahead seed 0" in text and "cannot satisfy a trial" in text


# ---------------------------------------------------------------- artifact sha256s ----
def test_finish_records_artifact_sha256_and_replay_binds_it_to_the_artifacts(genesis_env: Env) -> None:
    receipt = begin(genesis_env.request("synthetic"), config=genesis_env.config)
    names = ["runs/x/summary.json"]
    shas = {names[0]: "d" * 64}
    line = finish(receipt, "ok", {}, names, config=genesis_env.config, artifact_sha256=shas)
    assert line.record[ARTIFACT_SHA_KEY] == shas and set(line.record) == KEYS["result"]
    records = _records(genesis_env)
    for bad in ({}, {**shas, "runs/x/other.json": "d" * 64}, {names[0]: "D" * 64}, {names[0]: None}):
        with pytest.raises(ChainBroken, match=ARTIFACT_SHA_KEY):
            replay([*records[:-1], {**records[-1], ARTIFACT_SHA_KEY: bad}])
    # a result record written before the key (tier 2) still replays, as a prefix
    older = {k: v for k, v in records[-1].items() if k != ARTIFACT_SHA_KEY}
    assert replay([*records[:-1], older]).records == 3
    # a failed run names no artifact and records {}
    failed = begin(genesis_env.request("synthetic"), config=genesis_env.config)
    done = finish(failed, "failed", None, [], config=genesis_env.config, error="ZeroDivisionError")
    assert done.record[ARTIFACT_SHA_KEY] == {}


def _synthetic_run(env: Env, runs: Path) -> Any:
    from pfb.runner.api import run

    fake = FakeTrialEngine(extras=True)
    return run(env.repo / "strategies/s.py", kind="synthetic", deps=fake.deps(env), runs_dir=runs)


def test_the_runner_records_the_sha256_of_exactly_the_bytes_it_writes(genesis_env: Env) -> None:
    # Arrange
    runs = genesis_env.repo / "runs"
    # Act
    outcome = _synthetic_run(genesis_env, runs)
    # Assert: every artifact named, each file hashing to its recorded sha256; N unchanged
    record = outcome.result_record
    run_dir = runs / record["run_id"]
    assert set(record[ARTIFACT_SHA_KEY]) == set(record["artifacts"]) and len(record["artifacts"]) == 5
    for name, sha in record[ARTIFACT_SHA_KEY].items():
        assert hashlib.sha256((run_dir / name.rsplit("/", 1)[1]).read_bytes()).hexdigest() == sha
    assert require_folder_matches(run_dir, record["run_id"], record) is True
    assert n_of(genesis_env) == 302


def test_a_folder_that_does_not_match_its_record_is_refused_and_reported(genesis_env: Env) -> None:
    # Arrange: two runs; the second's frames copied into the first's folder
    runs = genesis_env.repo / "runs"
    first, second = _synthetic_run(genesis_env, runs), _synthetic_run(genesis_env, runs)
    a, b = first.result_record["run_id"], second.result_record["run_id"]
    (runs / a / "daily.parquet").write_bytes(b"not the recorded frame")
    (runs / b / "trades.parquet").unlink()
    state = read_state(genesis_env.config)
    # Act
    report = verify_folders(state, runs)
    # Assert
    assert not report.ok and set(report.mismatched) == {a, b}
    assert report.mismatched[a] == ("daily.parquet: sha256 differs from the result record",)
    assert report.mismatched[b] == ("trades.parquet: missing",)
    with pytest.raises(ArtifactMismatch, match="refused as an input"):
        require_folder_matches(runs / a, a, state.results[a])


def test_records_without_hashes_are_unchecked_never_matching(genesis_env: Env) -> None:
    runs = genesis_env.repo / "runs"
    outcome = _synthetic_run(genesis_env, runs)
    record = {k: v for k, v in outcome.result_record.items() if k != ARTIFACT_SHA_KEY}
    run_id = record["run_id"]
    assert require_folder_matches(runs / run_id, run_id, record) is False
    report = verify_folders(read_state(genesis_env.config), genesis_env.repo / "elsewhere")
    assert report.ok and report.absent == (run_id,) and report.checked == ()
