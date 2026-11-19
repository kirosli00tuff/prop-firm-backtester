"""Hash chain: canonical lines, tamper detection and every n_before/n_after link."""

from __future__ import annotations

import json

import pytest
from registry_env import Env

from pfb.registry.api import begin, finish, read_state, status
from pfb.registry.chain import ZERO_HASH, canonical_bytes, parse_ledger, read_ledger, sha256_hex
from pfb.registry.errors import BadRequest, ChainBroken, RegistryMissing


def _lines(env: Env) -> list[bytes]:
    return env.config.registry_path.read_bytes().rstrip(b"\n").split(b"\n")


def _rewrite(env: Env, lines: list[bytes]) -> None:
    env.config.registry_path.write_bytes(b"".join(x + b"\n" for x in lines))


def _three_records(env: Env) -> None:
    receipt = begin(env.request("synthetic"), config=env.config)
    finish(receipt, "ok", {"net_cents": 5}, ["runs/x/summary.json"], config=env.config,
           artifact_sha256={"runs/x/summary.json": "e" * 64})  # fmt: skip


def test_canonical_bytes_sorts_keys_without_spaces_in_ascii() -> None:
    assert canonical_bytes({"b": 1, "a": "é"}) == b'{"a":"\\u00e9","b":1}'


def test_canonical_bytes_refuses_nan() -> None:
    with pytest.raises(BadRequest):
        canonical_bytes({"x": float("nan")})


def test_genesis_line_links_to_zero_hash_and_each_line_to_the_previous(genesis_env: Env) -> None:
    _three_records(genesis_env)
    lines = read_ledger(genesis_env.config.registry_path)
    assert lines[0].record["prev_sha256"] == ZERO_HASH
    assert all(lines[i].record["prev_sha256"] == sha256_hex(lines[i - 1].raw) for i in range(1, len(lines)))


def test_reading_a_missing_ledger_raises_registry_missing(env: Env) -> None:
    with pytest.raises(RegistryMissing):
        read_state(env.config)


def test_editing_one_value_breaks_the_chain(genesis_env: Env) -> None:
    _three_records(genesis_env)
    lines = _lines(genesis_env)
    lines[1] = lines[1].replace(b'"seed":0', b'"seed":7')
    _rewrite(genesis_env, lines)
    with pytest.raises(ChainBroken):
        read_state(genesis_env.config)


def test_deleting_a_middle_line_breaks_the_chain(genesis_env: Env) -> None:
    _three_records(genesis_env)
    lines = _lines(genesis_env)
    _rewrite(genesis_env, [lines[0], lines[2]])
    with pytest.raises(ChainBroken):
        read_state(genesis_env.config)


def test_a_non_canonical_line_is_refused_even_with_the_same_content(genesis_env: Env) -> None:
    raw = _lines(genesis_env)[0]
    spaced = json.dumps(json.loads(raw), sort_keys=True).encode()
    with pytest.raises(ChainBroken):
        parse_ledger(spaced + b"\n")


def test_a_torn_last_line_is_refused(genesis_env: Env) -> None:
    data = genesis_env.config.registry_path.read_bytes()
    with pytest.raises(ChainBroken):
        parse_ledger(data + b'{"kind":')


def test_a_blank_line_is_refused(genesis_env: Env) -> None:
    data = genesis_env.config.registry_path.read_bytes()
    with pytest.raises(ChainBroken):
        parse_ledger(data + b"\n")


@pytest.mark.parametrize("field,value", [("n_before", 1), ("n_after", 999)])
def test_a_rehashed_record_with_a_wrong_n_link_is_refused(genesis_env: Env, field: str, value: int) -> None:
    # Arrange: forge a run record with a broken N link and a correct hash chain.
    _three_records(genesis_env)
    lines = _lines(genesis_env)
    record = json.loads(lines[1])
    record[field] = record[field] + value
    lines[1] = canonical_bytes(record)
    third = json.loads(lines[2])
    third["prev_sha256"] = sha256_hex(lines[1])
    lines[2] = canonical_bytes(third)
    _rewrite(genesis_env, lines)
    # Act / Assert
    with pytest.raises(ChainBroken, match="n_"):
        read_state(genesis_env.config)


def test_a_rehashed_genesis_with_a_different_n_is_refused(genesis_env: Env) -> None:
    lines = _lines(genesis_env)
    record = json.loads(lines[0])
    record["n_after"] += 1
    record["n_pe"] += 1
    _rewrite(genesis_env, [canonical_bytes(record)])
    with pytest.raises(ChainBroken):
        read_state(genesis_env.config)


def test_a_rehashed_genesis_with_an_edited_pe_line_is_refused(genesis_env: Env) -> None:
    lines = _lines(genesis_env)
    record = json.loads(lines[0])
    record["pe_files"]["ledger/trial_amendments.jsonl"]["lines"][0] += " "
    _rewrite(genesis_env, [canonical_bytes(record)])
    with pytest.raises(ChainBroken, match="hash"):
        read_state(genesis_env.config)


def test_status_reports_n_records_and_last_hash(genesis_env: Env) -> None:
    _three_records(genesis_env)
    report = status(genesis_env.config)
    assert (report.n, report.n_pe, report.n_local, report.records) == (302, 302, 0, 3)
    assert report.last_sha256 == sha256_hex(_lines(genesis_env)[-1])
    assert report.pe_in_sync and report.open_runs == ()
