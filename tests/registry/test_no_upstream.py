"""A ledger without an upstream checkout: genesis at N = 0, nothing to sync, and the guard that a ledger
which imported an upstream ledger keeps needing it."""

from __future__ import annotations

from dataclasses import replace

import pytest
from registry_env import Env

from pfb.registry.api import begin, create_genesis, read_state, status, sync
from pfb.registry.chain import read_ledger
from pfb.registry.errors import SyncRequired


def test_genesis_without_an_upstream_imports_nothing_and_n_is_zero(env: Env) -> None:
    config = replace(env.config, pe_root=None)

    line = create_genesis(config)

    assert line.record["n_pe"] == 0 and line.record["pe_files"] == {}
    assert (line.record["n_before"], line.record["n_after"]) == (0, 0)
    report = status(config)
    assert (report.n, report.n_pe, report.records, report.pe_changed) == (0, 0, 1, ())


def test_sync_without_an_upstream_appends_nothing(env: Env) -> None:
    config = replace(env.config, pe_root=None)
    create_genesis(config)

    assert sync(config) is None
    assert len(read_ledger(config.registry_path)) == 1


def test_a_synthetic_run_on_a_ledger_without_upstream_adds_nothing(env: Env) -> None:
    config = replace(env.config, pe_root=None)
    create_genesis(config)

    receipt = begin(env.request("synthetic"), config=config)

    assert (receipt.n_before, receipt.n_after) == (0, 0)
    assert read_state(config).n == 0


def test_a_ledger_that_imported_an_upstream_refuses_to_run_without_it(genesis_env: Env) -> None:
    config = replace(genesis_env.config, pe_root=None)

    with pytest.raises(SyncRequired, match="upstream root"):
        status(config)
    with pytest.raises(SyncRequired):
        sync(config)


def test_a_later_sync_imports_an_upstream_into_a_ledger_that_had_none(env: Env) -> None:
    create_genesis(replace(env.config, pe_root=None))

    line = sync(env.config)

    assert line is not None and line.record["n_pe_before"] == 0 and line.record["n_pe"] == 302
    assert read_state(env.config).n == 302
