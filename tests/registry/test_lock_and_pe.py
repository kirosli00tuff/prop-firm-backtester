"""The append lock under two concurrent processes (A14 b), and genesis N against upstream's own current_n."""

from __future__ import annotations

import multiprocessing

from registry_env import Env

from pfb.registry.api import RegistryConfig, RunRequest, begin, read_state
from pfb.registry.chain import read_ledger

APPENDS_PER_PROCESS = 15
PROCESSES = 2
START_TIMEOUT_S = 30
JOIN_TIMEOUT_S = 120


def _append_many(config: RegistryConfig, request: RunRequest, start: object) -> None:
    start.wait(START_TIMEOUT_S)  # type: ignore[attr-defined]
    for _ in range(APPENDS_PER_PROCESS):
        begin(request, config=config)


def test_two_processes_appending_at_once_keep_one_valid_chain(genesis_env: Env) -> None:
    # Arrange
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    request = genesis_env.request("synthetic")
    workers = [
        ctx.Process(target=_append_many, args=(genesis_env.config, request, start)) for _ in range(PROCESSES)
    ]
    for worker in workers:
        worker.start()
    # Act
    start.set()
    for worker in workers:
        worker.join(JOIN_TIMEOUT_S)
    # Assert
    assert [w.exitcode for w in workers] == [0] * PROCESSES
    lines = read_ledger(genesis_env.config.registry_path)
    state = read_state(genesis_env.config)
    assert len(lines) == 1 + PROCESSES * APPENDS_PER_PROCESS
    assert len(state.begun) == PROCESSES * APPENDS_PER_PROCESS
    assert state.n == 302


