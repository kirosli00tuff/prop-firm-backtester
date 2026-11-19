"""The ledger is anchored in git. When HEAD tracks the ledger, its bytes at HEAD must be a byte
prefix of the working file before every append and in status; genesis refuses when HEAD tracks a ledger.
Limit: lines appended after the last commit are not anchored (the last test pins that limit)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from registry_env import Env, git

from pfb.registry.api import (
    LEDGER_REL,
    RegistryConfig,
    begin,
    create_genesis,
    finish,
    read_state,
    status,
)
from pfb.registry.chain import canonical_bytes, read_ledger
from pfb.registry.errors import GenesisExists, LedgerAnchorBroken


def status_of_chain(path: Path) -> int:
    """Lines of a chain-valid file (no anchor, no replay)."""
    return len(read_ledger(path))


def _commit_ledger(env: Env) -> None:
    git(env.repo, "add", "--", LEDGER_REL)
    git(env.repo, "commit", "-q", "--allow-empty", "-m", "ledger")


def _cut_tail(path: Path, lines: int) -> None:
    kept = path.read_bytes().rstrip(b"\n").split(b"\n")[:-lines]
    path.write_bytes(b"".join(x + b"\n" for x in kept))


def test_cutting_committed_lines_off_the_tail_is_refused_on_append_and_status(trial_env: Env) -> None:
    # Arrange: the ledger committed after the trial's two lines
    receipt = begin(trial_env.request("trial"), config=trial_env.config)
    finish(receipt, "ok", {"net_cents": -5}, [], config=trial_env.config)
    _commit_ledger(trial_env)
    path = trial_env.config.registry_path
    _cut_tail(path, 2)
    cut = path.read_bytes()
    # Act / Assert
    with pytest.raises(LedgerAnchorBroken):
        begin(trial_env.request("synthetic"), config=trial_env.config)
    with pytest.raises(LedgerAnchorBroken):
        status(trial_env.config)
    with pytest.raises(LedgerAnchorBroken):
        read_state(trial_env.config)
    assert path.read_bytes() == cut


def test_a_deleted_tracked_ledger_takes_no_new_genesis(genesis_env: Env) -> None:
    _commit_ledger(genesis_env)
    genesis_env.config.registry_path.unlink()
    with pytest.raises(GenesisExists, match="HEAD tracks"):
        create_genesis(genesis_env.config)
    assert not genesis_env.config.registry_path.exists()


def test_an_emptied_tracked_ledger_takes_no_new_genesis(genesis_env: Env) -> None:
    _commit_ledger(genesis_env)
    genesis_env.config.registry_path.write_bytes(b"")
    with pytest.raises(GenesisExists):
        create_genesis(genesis_env.config)
    assert genesis_env.config.registry_path.read_bytes() == b""


def test_lines_appended_after_the_committed_prefix_are_accepted(trial_env: Env) -> None:
    _commit_ledger(trial_env)
    receipt = begin(trial_env.request("trial"), config=trial_env.config)
    finish(receipt, "ok", {}, [], config=trial_env.config)
    assert (status(trial_env.config).n, status(trial_env.config).records) == (303, 4)


def test_a_committed_line_edited_in_place_is_refused_even_when_rehashed(genesis_env: Env) -> None:
    begin(genesis_env.request("synthetic"), config=genesis_env.config)
    _commit_ledger(genesis_env)
    path = genesis_env.config.registry_path
    lines = path.read_bytes().rstrip(b"\n").split(b"\n")
    edited = {**json.loads(lines[-1]), "seed": 5}  # the last line: its chain link stays valid
    path.write_bytes(lines[0] + b"\n" + canonical_bytes(edited) + b"\n")
    assert status_of_chain(path) == 2
    with pytest.raises(LedgerAnchorBroken):
        begin(genesis_env.request("synthetic"), config=genesis_env.config)


def test_outside_a_git_repository_there_is_no_anchor(env: Env, tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    config = replace(env.config, registry_path=plain / "registry.jsonl", repo_root=plain)
    create_genesis(config)
    path = config.registry_path
    path.unlink()
    create_genesis(config)  # no git, no anchor: the stated limit
    assert status(config).records == 1


def test_a_ledger_outside_the_repository_is_not_compared_with_the_tracked_one(
    genesis_env: Env, tmp_path: Path
) -> None:
    _commit_ledger(genesis_env)
    other = tmp_path / "elsewhere" / "registry.jsonl"
    other.parent.mkdir()
    config: RegistryConfig = replace(genesis_env.config, registry_path=other)
    create_genesis(config)
    assert status(config).records == 1


def test_uncommitted_lines_are_not_anchored(trial_env: Env) -> None:
    # The stated limit: a cut of lines appended after the last commit is not detected.
    _commit_ledger(trial_env)
    begin(trial_env.request("trial"), config=trial_env.config)
    _cut_tail(trial_env.config.registry_path, 1)
    assert status(trial_env.config).records == 2
