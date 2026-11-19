"""Genesis (upstream's lines imported unchanged, N = upstream's current_n) and sync."""

from __future__ import annotations

import json

import pytest
from registry_env import HYPOTHESIS_REL, PE_AMD, PE_AMENDMENT, PE_BASE_LINES, PE_REG, Env, pe_line

from pfb.registry.api import begin, create_genesis, current_n, finish, read_state, register_hypothesis, sync
from pfb.registry.chain import read_ledger, sha256_hex
from pfb.registry.errors import GenesisExists, PeLedgerInvalid, SyncPrefixChanged, SyncRequired
from pfb.registry.genesis import check_pe_amendments, join_lines, pe_current_n, split_lines

NEW_PE_LINE = pe_line("r002-T2", "T2", ["T2-a", "T2-b", "T2-c"], 302, 305)


def test_genesis_sets_n_to_pe_current_n_and_embeds_lines_unchanged(env: Env) -> None:
    # Arrange
    before = {name: (env.pe / name).read_bytes() for name in (PE_REG, PE_AMD)}
    # Act
    line = create_genesis(env.config)
    # Assert
    assert line.record["n_after"] == 302 == current_n(env.config)
    for name, data in before.items():
        entry = line.record["pe_files"][name]
        assert join_lines(entry["lines"]) == data
        assert entry["sha256"] == sha256_hex(data)
        assert (env.pe / name).read_bytes() == data


def test_genesis_is_written_once(genesis_env: Env) -> None:
    with pytest.raises(GenesisExists):
        create_genesis(genesis_env.config)
    assert len(read_ledger(genesis_env.config.registry_path)) == 1


@pytest.mark.parametrize(
    "lines",
    [
        [],
        [pe_line("baseline", None, [], None, 299)],
        [PE_BASE_LINES[0], pe_line("r001-T1", "T1", ["T1-a"], 300, 302)],
        [PE_BASE_LINES[0], pe_line("r001-T1", "T1", ["T1-a"], 299, 300)],
        [PE_BASE_LINES[0], pe_line("r001-T1", "T1", [], 300, 300)],
        [*PE_BASE_LINES, pe_line("r002-T1", "T1", ["T1-c"], 302, 303)],
        [*PE_BASE_LINES, pe_line("r002-T2", "T2", ["T1-a"], 302, 303)],
        [PE_BASE_LINES[0], "", PE_BASE_LINES[1]],
        [PE_BASE_LINES[0], "{not json"],
    ],
    ids=["empty", "baseline-n", "n-sum", "n-link", "no-ids", "repeat-test", "repeat-id", "blank", "not-json"],
)
def test_pe_registration_checks_are_ported(lines: list[str]) -> None:
    with pytest.raises(PeLedgerInvalid):
        pe_current_n(lines)


def test_pe_current_n_is_the_last_n_after() -> None:
    assert pe_current_n([*PE_BASE_LINES, NEW_PE_LINE]) == 305


def test_an_amendment_that_changes_n_is_refused() -> None:
    with pytest.raises(PeLedgerInvalid):
        check_pe_amendments([json.dumps({"n_before": 302, "n_after": 303})])


def test_a_pe_file_with_crlf_cannot_be_imported() -> None:
    with pytest.raises(PeLedgerInvalid):
        split_lines(b"a\r\n", PE_REG)


def test_begin_and_registration_are_refused_after_pe_changes_until_sync(genesis_env: Env) -> None:
    # Arrange
    begin(genesis_env.request("synthetic"), config=genesis_env.config)
    hypothesis = genesis_env.write(HYPOTHESIS_REL, genesis_env.hypothesis_text())
    genesis_env.commit()
    genesis_env.write_pe([*PE_BASE_LINES, NEW_PE_LINE], [PE_AMENDMENT])
    # Act / Assert
    with pytest.raises(SyncRequired):
        begin(genesis_env.request("synthetic"), config=genesis_env.config)
    with pytest.raises(SyncRequired):
        register_hypothesis(hypothesis, genesis_env.config, genesis_env.facts())
    assert len(read_ledger(genesis_env.config.registry_path)) == 2


def test_a_result_after_a_pe_ledger_change_is_accepted_and_moves_no_n(genesis_env: Env) -> None:
    # Arrange: a run begun, then upstream's ledger changes before its result (TRIALS 4)
    receipt = begin(genesis_env.request("synthetic"), config=genesis_env.config)
    genesis_env.write_pe([*PE_BASE_LINES, NEW_PE_LINE], [PE_AMENDMENT])
    # Act
    line = finish(receipt, "ok", {}, [], config=genesis_env.config)
    # Assert
    assert line.record["kind"] == "result" and line.record["n_after"] == line.record["n_before"] == 302
    with pytest.raises(SyncRequired):
        begin(genesis_env.request("synthetic"), config=genesis_env.config)


def test_sync_imports_new_lines_and_adds_pe_n_increase(genesis_env: Env) -> None:
    # Arrange
    genesis_env.write_pe([*PE_BASE_LINES, NEW_PE_LINE], [PE_AMENDMENT])
    # Act
    line = sync(genesis_env.config)
    receipt = begin(genesis_env.request("synthetic"), config=genesis_env.config)
    # Assert
    assert line is not None
    assert line.record["pe_files"][PE_REG]["lines_added"] == [NEW_PE_LINE]
    assert line.record["pe_files"][PE_AMD]["lines_added"] == []
    assert (line.record["n_before"], line.record["n_after"]) == (302, 305)
    assert receipt.n_after == 305 == read_state(genesis_env.config).n_pe


def test_sync_with_pe_unchanged_appends_nothing(genesis_env: Env) -> None:
    assert sync(genesis_env.config) is None
    assert len(read_ledger(genesis_env.config.registry_path)) == 1


def test_sync_refuses_when_imported_lines_are_no_longer_a_prefix(genesis_env: Env) -> None:
    edited = PE_BASE_LINES[1].replace("synthetic", "edited")
    genesis_env.write_pe([PE_BASE_LINES[0], edited, NEW_PE_LINE], [PE_AMENDMENT])
    with pytest.raises(SyncPrefixChanged):
        sync(genesis_env.config)
    assert len(read_ledger(genesis_env.config.registry_path)) == 1


def test_sync_refuses_when_the_amendments_file_lost_a_line(genesis_env: Env) -> None:
    genesis_env.write_pe([*PE_BASE_LINES, NEW_PE_LINE], [])
    with pytest.raises(SyncPrefixChanged):
        sync(genesis_env.config)
