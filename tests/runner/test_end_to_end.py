"""One synthetic run through the real engine and a temporary registry (skipped until run.py exists)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from registry_env import Env  # pyright: ignore[reportMissingImports]

from pfb.registry.api import create_genesis, status
from pfb.registry.chain import read_ledger

STRATEGY = """import numpy as np

SPEC = {
    "hypothesis_id": "synthetic-e2e",
    "root": "MNQ",
    "bar_minutes": 1,
    "session": "globex",
    "contracts": 1,
    "params": {"k": {"default": 1, "grid": [1, 2]}},
}


def generate(bars, params):
    n = len(bars["close"])
    flat = np.zeros(n, dtype=bool)
    ticks = np.zeros(n, dtype=np.int64)
    return {"entry_long": flat, "entry_short": flat, "exit": flat, "stop_ticks": ticks, "target_ticks": ticks}
"""


def test_a_synthetic_run_logs_begin_and_result_through_the_real_engine(env: Env, tmp_path: Path) -> None:
    # Arrange
    engine_run = pytest.importorskip("pfb.engine.run")
    if not hasattr(engine_run, "run_strategy"):
        pytest.skip("pfb.engine.run.run_strategy is not written yet")
    from pfb.runner.api import run
    from pfb.runner.deps import default_deps

    create_genesis(env.config)
    strategy = env.write("strategies/e2e.py", STRATEGY)
    deps = replace(default_deps(), registry_config=lambda: env.config)
    # Act
    runs = env.repo / "runs"  # inside the repository: repo-relative artifact names
    outcome = run(strategy, kind="synthetic", cost_model="fixed:1", deps=deps, runs_dir=runs)
    # Assert
    kinds = [line.record["kind"] for line in read_ledger(env.config.registry_path)]
    assert kinds == ["genesis", "synthetic", "result"]
    assert outcome.result_record["status"] == "ok"
    assert status(env.config).open_runs == () and status(env.config).n == 302
    assert all((env.repo / a).is_file() for a in outcome.result_record["artifacts"])
