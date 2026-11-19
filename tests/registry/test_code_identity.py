"""Every begin record carries code_sha256, the identity of the code under src/pfb; for a trial it
is part of config_id, so "an exact replay adds nothing" means same inputs AND same code. Begin records written
before the field existed (the real ledger's line 2) stay valid as a legacy prefix only."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from registry_env import ENGINE_REL, HYPOTHESIS_REL, Env, git

from pfb.registry.api import REPO_ROOT, begin, register_hypothesis
from pfb.registry.chain import ZERO_HASH, canonical_bytes, parse_ledger, read_ledger, sha256_hex
from pfb.registry.codehash import code_canonical_bytes, code_sha256
from pfb.registry.errors import BadRequest, ChainBroken
from pfb.registry.records import CODE_RUN_KEYS, KEYS, LEGACY_RUN_KEYS, config_id, replay

ENGINE_CHECKS_REL = "config/engine_checks.toml"
REAL_LEDGER = REPO_ROOT / "ledger" / "registry.jsonl"
REAL_LEDGER_LINES = 1
REAL_N = 0


def _tree(root: Path, files: dict[str, bytes]) -> Path:
    for rel, data in files.items():
        path = root / "pfb" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root / "pfb"


# ------------------------------------------------------------------------------------- canonical form ----


def test_code_sha256_hashes_sorted_path_names_and_lengths_with_the_bytes(tmp_path: Path) -> None:
    # Arrange: written out of order, plus files that are not *.py
    code = _tree(
        tmp_path, {"z.py": b"z = 1\n", "a/b.py": b"", "a.py": b"a\n", "notes.txt": b"x", "c.pyc": b"x"}
    )
    expected = b"".join(
        rel.encode() + b"\0" + str(len(data)).encode() + b"\0" + data
        for rel, data in (("pfb/a.py", b"a\n"), ("pfb/a/b.py", b""), ("pfb/z.py", b"z = 1\n"))
    )
    # Act / Assert
    assert code_canonical_bytes(code) == expected
    assert code_sha256(code) == hashlib.sha256(expected).hexdigest()


@pytest.mark.parametrize(
    "change",
    [
        lambda c: (c / "z.py").write_bytes(b"z = 2\n"),
        lambda c: (c / "z.py").rename(c / "y.py"),
        lambda c: (c / "new.py").write_bytes(b""),
    ],
    ids=["content", "rename", "new-empty-file"],
)
def test_any_content_or_name_change_changes_code_sha256(
    tmp_path: Path, change: Callable[[Path], Any]
) -> None:
    code = _tree(tmp_path, {"z.py": b"z = 1\n", "a.py": b"a\n"})
    before = code_sha256(code)
    change(code)
    assert code_sha256(code) != before


def test_a_code_root_without_python_files_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BadRequest):
        code_sha256(tmp_path)


# ------------------------------------------------------------------------------------- begin records ----


@pytest.mark.parametrize("kind", ["trial", "engine_check", "synthetic"])
def test_every_begin_record_carries_the_code_sha256_read_from_disk(genesis_env: Env, kind: str) -> None:
    # Arrange: a trial needs its hypothesis; an engine_check needs a strategy no hypothesis pins (rule C5)
    if kind == "trial":
        genesis_env.write(HYPOTHESIS_REL, genesis_env.hypothesis_text())
        genesis_env.commit()
        register_hypothesis(genesis_env.repo / HYPOTHESIS_REL, genesis_env.config, genesis_env.facts())
    genesis_env.write(ENGINE_CHECKS_REL, genesis_env.engine_checks_text())
    genesis_env.commit()
    genesis_env.checkpoint()
    # Act
    receipt = begin(genesis_env.request(kind), config=genesis_env.config)
    # Assert
    assert receipt.record["code_sha256"] == code_sha256(genesis_env.repo / "src" / "pfb")
    assert set(receipt.record) == KEYS[kind]


def test_a_trial_replayed_after_a_committed_engine_change_is_a_new_configuration(trial_env: Env) -> None:
    # Before the fix the second run gave (303, 303) with the same config_id.
    first = trial_env.begin_trial()
    trial_env.write(ENGINE_REL, "v = 2  # a different fill rule\n")
    trial_env.commit()
    second = trial_env.begin_trial()
    assert (first.n_before, first.n_after, second.n_before, second.n_after) == (302, 303, 303, 304)
    assert first.record["configurations"] != second.record["configurations"]


def test_an_edit_hidden_by_assume_unchanged_still_changes_the_configuration(trial_env: Env) -> None:
    # git status does not show the edit, the bytes on disk do.
    first = trial_env.begin_trial()
    git(trial_env.repo, "update-index", "--assume-unchanged", ENGINE_REL)
    trial_env.write(ENGINE_REL, "v = 3  # edited, hidden from git status\n")
    second = trial_env.begin_trial()
    assert second.record["git_dirty"] is False
    assert second.n_after == first.n_after + 1


def test_an_exact_replay_with_the_same_code_still_adds_nothing(trial_env: Env) -> None:
    trial_env.begin_trial()
    assert trial_env.begin_trial().n_after == 303


def test_config_id_covers_code_sha256(trial_env: Env) -> None:
    record = dict(begin(trial_env.request("trial"), config=trial_env.config).record)
    assert config_id({**record, "code_sha256": "f" * 64}) != config_id(record)


# ------------------------------------------------------------------------------ legacy records (schema) ----


def _rechain(records: list[dict[str, Any]]) -> bytes:
    out: list[bytes] = []
    previous = ZERO_HASH
    for record in records:
        raw = canonical_bytes({**record, "prev_sha256": previous})
        out.append(raw)
        previous = sha256_hex(raw)
    return b"".join(x + b"\n" for x in out)


def _as_legacy(record: dict[str, Any]) -> dict[str, Any]:
    """The same begin record as written before R3/R4: no code_sha256, a check without session/seed/params
    (and none tier-3 fields)."""
    legacy = {k: v for k, v in _as_tier2(record).items() if k != "code_sha256"}
    if isinstance(legacy.get("check"), dict):
        legacy["check"] = {k: v for k, v in legacy["check"].items() if k not in ("session", "seed", "params")}
    legacy["configurations"] = [config_id(legacy)]
    return legacy


def _as_tier2(record: dict[str, Any]) -> dict[str, Any]:
    """The same begin record as written: no calendars_sha256, lookahead_signal or power."""
    tier2 = {k: v for k, v in record.items() if k not in ("calendars_sha256", "lookahead_signal", "power")}
    tier2["configurations"] = [config_id(tier2)]
    return tier2


def _records(env: Env) -> list[dict[str, Any]]:
    return [dict(line.record) for line in read_ledger(env.config.registry_path)]


def _engine_check_ledger(env: Env) -> list[dict[str, Any]]:
    env.write(ENGINE_CHECKS_REL, env.engine_checks_text())
    env.commit()
    begin(env.request("engine_check"), config=env.config)
    begin(env.request("engine_check"), config=env.config)
    return _records(env)


def test_a_legacy_begin_record_before_any_current_one_replays(genesis_env: Env) -> None:
    genesis, first, second = _engine_check_ledger(genesis_env)
    records = [genesis, _as_legacy(first), second]
    assert set(records[1]) == LEGACY_RUN_KEYS
    lines = parse_ledger(_rechain(records))
    assert replay([line.record for line in lines]).n == 302


def test_a_legacy_begin_record_after_a_current_one_is_refused(genesis_env: Env) -> None:
    genesis, first, second = _engine_check_ledger(genesis_env)
    with pytest.raises(ChainBroken, match="older begin or result tier"):
        replay([genesis, first, _as_legacy(second)])


def test_a_tier2_begin_record_before_any_tier3_one_replays(genesis_env: Env) -> None:
    genesis, first, second = _engine_check_ledger(genesis_env)
    records = [genesis, _as_tier2(first), second]
    assert set(records[1]) == CODE_RUN_KEYS
    lines = parse_ledger(_rechain(records))
    assert replay([line.record for line in lines]).n == 302


def test_a_tier2_begin_record_after_a_tier3_one_is_refused(genesis_env: Env) -> None:
    genesis, first, second = _engine_check_ledger(genesis_env)
    with pytest.raises(ChainBroken, match="older begin or result tier"):
        replay([genesis, first, _as_tier2(second)])


def test_config_id_covers_calendars_sha256(genesis_env: Env) -> None:
    _, first, _ = _engine_check_ledger(genesis_env)
    assert config_id({**first, "calendars_sha256": "f" * 64}) != config_id(first)
    assert first["configurations"] != [config_id(_as_tier2(first))]


def test_a_malformed_code_sha256_is_refused(genesis_env: Env) -> None:
    genesis, first, _ = _engine_check_ledger(genesis_env)
    broken = {**first, "code_sha256": "not-a-sha"}
    broken["configurations"] = [config_id(broken)]
    with pytest.raises(ChainBroken, match="code_sha256"):
        replay([genesis, broken])


@pytest.mark.skipif(not REAL_LEDGER.is_file(), reason="the repository's ledger is absent")
def test_the_committed_ledger_starts_with_a_genesis_that_imports_nothing() -> None:
    # Read-only: the repository's first line is a genesis without an upstream import, at N = 0.
    data = REAL_LEDGER.read_bytes()
    first = b"".join(x + b"\n" for x in data.split(b"\n")[:REAL_LEDGER_LINES])
    lines = parse_ledger(first)
    state = replay([line.record for line in lines])
    assert (state.n, state.n_pe, state.records) == (REAL_N, 0, REAL_LEDGER_LINES)
    assert lines[0].record["kind"] == "genesis" and lines[0].record["pe_files"] == {}
