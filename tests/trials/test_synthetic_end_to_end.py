"""A synthetic run and a synthetic neighbourhood through the real engine leave N as is (temp ledger).
Runs without config/paths.toml: the real deps on a temporary paths file, as
test_suite_error_ledger.py."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from registry_env import Env  # pyright: ignore[reportMissingImports]
from trial_support import (  # pyright: ignore[reportMissingImports]
    kinds,
    n_of,
    temp_paths_toml,
    without_repo_paths,
)

STRATEGY = """import numpy as np

SPEC = {
    "hypothesis_id": "synthetic-e2e",
    "root": "MNQ",
    "bar_minutes": 1,
    "session": "globex",
    "contracts": 1,
    "params": {"k": {"default": 2, "grid": [1, 2, 3]}},
}


def generate(bars, params):
    n = len(bars["close"])
    entry = np.zeros(n, dtype=bool)
    entry[:: 97 * params["k"]] = True
    flat = np.zeros(n, dtype=bool)
    ticks = np.full(n, 8, dtype=np.int64)
    return {"entry_long": entry, "entry_short": flat, "exit": flat,
            "stop_ticks": ticks, "target_ticks": ticks}
"""


@pytest.fixture
def real_deps(genesis_env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The real runner deps on a temporary paths file: config/paths.toml is not needed."""
    from pfb.runner.deps import default_deps

    without_repo_paths(monkeypatch, tmp_path)
    toml = temp_paths_toml(tmp_path / "paths")
    return replace(default_deps(toml), registry_config=lambda: genesis_env.config)


def test_a_synthetic_run_and_neighbourhood_through_the_real_engine_leave_n(
    genesis_env: Env, real_deps: Any
) -> None:
    # Arrange
    from pfb.registry.artifacts import require_folder_matches
    from pfb.runner.api import run
    from pfb.runner.neighbours import run_with_neighbours

    strategy = genesis_env.write("strategies/e2e.py", STRATEGY)
    runs = genesis_env.repo / "runs"
    # Act
    single = run(strategy, kind="synthetic", cost_model="fixed:1", deps=real_deps, runs_dir=runs)
    found = run_with_neighbours(
        strategy, kind="synthetic", cost_model="fixed:1", deps=real_deps, runs_dir=runs
    )
    # Assert
    assert single.result_record["status"] == "ok" and single.begin_record["power"] is None
    assert len(single.begin_record["calendars_sha256"]) == 64
    assert [o.begin_record["params"]["k"] for o in (found.centre, *found.neighbours)] == [2, 1, 3]
    assert kinds(genesis_env) == ["genesis"] + ["synthetic", "result"] * 4 and n_of(genesis_env) == 302
    # each folder written by the real engine hashes to its result record
    record = single.result_record
    assert require_folder_matches(runs / record["run_id"], record["run_id"], record) is True
    doc = json.loads((runs / found.centre.begin_record["run_id"] / "neighbourhood.json").read_text())
    assert doc["verdict"]["verdict"] in ("plateau", "spike", "mixed", "n/a")
