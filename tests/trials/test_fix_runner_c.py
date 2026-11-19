"""Runner side of the registration checks. Temporary ledgers.

F4: step 1 resolves the parameters against the SPEC grid (the engine's resolve_params), so a value outside it
is refused before begin for every kind. F5: a neighbourhood set with a planned "none" look-ahead signal and no
no_signal_reason is refused whole before any begin. F7: the neighbours are begun and run first, the centre
last.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from registry_env import HYPOTHESIS_REL, STRATEGY_REL, Env  # pyright: ignore[reportMissingImports]
from trial_support import FakeTrialEngine, kinds, n_of  # pyright: ignore[reportMissingImports]

from pfb.registry.api import register_hypothesis
from pfb.registry.chain import read_ledger
from pfb.registry.errors import NoSignalUnexplained
from pfb.runner.api import RunRefused, run
from pfb.runner.neighbours import run_with_neighbours

NO_SIGNAL = "look-ahead check saw no signal"


@dataclass
class NarrowSpecEngine(FakeTrialEngine):
    """SPEC grid of `a` (1, 2): a = 3 lies inside the hypothesis grid [1, 2, 3], outside the SPEC's."""

    def strategy(self, path: Path) -> Any:
        found = super().strategy(path)
        params = {"a": SimpleNamespace(default=1, grid=(1, 2))}
        return SimpleNamespace(
            **{**vars(found), "spec": SimpleNamespace(**{**vars(found.spec), "params": params})}
        )


@dataclass
class NoSignalAtOne(FakeTrialEngine):
    """Step 1 sees no signal for a = 1 only (a neighbour of the centre a = 2)."""

    def lookahead(self, strategy: Any, params: Mapping[str, Any], seed: int) -> tuple[str, ...]:
        super().lookahead(strategy, params, seed)
        return (NO_SIGNAL,) if params.get("a") == 1 else ()


def _hypothesis(env: Env, **override: str) -> None:
    env.write(HYPOTHESIS_REL, env.hypothesis_text(**override))
    env.commit()
    register_hypothesis(env.repo / HYPOTHESIS_REL, env.config, env.facts())
    env.checkpoint()


def _neighbourhood(env: Env, engine: FakeTrialEngine, params: Mapping[str, Any]) -> Any:
    return run_with_neighbours(env.repo / STRATEGY_REL, kind="trial", params=params,
                               hypothesis_path=Path(HYPOTHESIS_REL), deps=engine.deps(env),
                               runs_dir=env.repo / "runs")  # fmt: skip


# ------------------------------------------------------------------------- F4: SPEC grid in step 1 ----


@pytest.mark.parametrize("params", [{"a": 5}, {"a": 2.0}], ids=["off-grid", "wrong-type"])
def test_a_synthetic_value_outside_the_spec_grid_is_refused_in_step_one(
    genesis_env: Env, params: dict
) -> None:
    # Arrange
    engine = FakeTrialEngine()
    # Act
    with pytest.raises(RunRefused, match="outside the declared grid"):
        run(genesis_env.repo / STRATEGY_REL, kind="synthetic", params=params, deps=engine.deps(genesis_env),
            runs_dir=genesis_env.repo / "runs")  # fmt: skip
    # Assert: refused before the look-ahead check and before begin
    assert kinds(genesis_env) == ["genesis"]
    assert engine.called("lookahead") == [] and engine.called("run_strategy") == []


def test_a_trial_value_in_the_hypothesis_grid_but_outside_the_spec_grid_counts_nothing(
    genesis_env: Env,
) -> None:
    # Arrange: hypothesis grid a in [1, 2, 3], SPEC grid (1, 2)
    _hypothesis(genesis_env)
    engine = NarrowSpecEngine()
    # Act
    with pytest.raises(RunRefused, match="outside the declared grid"):
        run(genesis_env.repo / STRATEGY_REL, kind="trial", params={"a": 3},
            hypothesis_path=Path(HYPOTHESIS_REL), deps=engine.deps(genesis_env),
            runs_dir=genesis_env.repo / "runs")  # fmt: skip
    # Assert
    assert kinds(genesis_env) == ["genesis", "hypothesis"] and n_of(genesis_env) == 302
    assert engine.called("data_files") == [] and engine.called("run_strategy") == []


def test_a_spec_grid_value_still_runs(genesis_env: Env) -> None:
    engine = FakeTrialEngine()
    outcome = run(genesis_env.repo / STRATEGY_REL, kind="synthetic", params={"a": 3},
                  deps=engine.deps(genesis_env), runs_dir=genesis_env.repo / "runs")  # fmt: skip
    assert outcome.begin_record["params"] == {"a": 3} and outcome.result_record["status"] == "ok"


# --------------------------------------------------------------- F5: no-signal planned before begin ----


def test_a_neighbour_without_signal_refuses_the_whole_set_before_any_begin(genesis_env: Env) -> None:
    # Arrange: neighbourhood = true, empty no_signal_reason; the neighbour a = 1 sees no signal
    _hypothesis(genesis_env, budget="3", suite__neighbourhood="true")
    engine = NoSignalAtOne()
    # Act
    with pytest.raises(NoSignalUnexplained, match="nothing is begun"):
        _neighbourhood(genesis_env, engine, {"a": 2})
    # Assert
    assert kinds(genesis_env) == ["genesis", "hypothesis"] and n_of(genesis_env) == 302
    assert engine.called("run_strategy") == []


def test_a_neighbour_without_signal_runs_when_the_hypothesis_gives_a_reason(genesis_env: Env) -> None:
    _hypothesis(genesis_env, budget="3", suite__neighbourhood="true", no_signal_reason='"rare entries"')
    found = _neighbourhood(genesis_env, NoSignalAtOne(), {"a": 2})
    assert n_of(genesis_env) == 305
    signals = {o.begin_record["params"]["a"]: o.begin_record["lookahead_signal"]
               for o in (found.centre, *found.neighbours)}  # fmt: skip
    assert signals == {1: "none", 2: "seen", 3: "seen"}


# ------------------------------------------------------------------------- F7: the centre runs last ----


def test_the_neighbours_are_begun_first_and_the_centre_last(genesis_env: Env) -> None:
    # Arrange: distinct daily Sharpes per configuration
    _hypothesis(genesis_env, budget="3", suite__neighbourhood="true")
    summaries = {a: {"net_cents": 100, "sharpe_annual": 1.0, "sharpe_daily": s, "warnings": []}
                 for a, s in ((1, 0.1), (2, 0.2), (3, 0.3))}  # fmt: skip
    engine = FakeTrialEngine(summaries=summaries)
    # Act
    found = _neighbourhood(genesis_env, engine, {"a": 2})
    # Assert: ledger order of the begins is a = 1, a = 3, then the centre a = 2
    begun = [
        line.record
        for line in read_ledger(genesis_env.config.registry_path)
        if line.record["kind"] == "trial"
    ]
    assert [r["params"]["a"] for r in begun] == [1, 3, 2]
    assert [r["n_after"] for r in begun] == [303, 304, 305]
    # the centre's N and DSR grid include its own neighbours
    assert found.centre.begin_record["params"] == {"a": 2} and found.centre.begin_record["n_after"] == 305
    assert engine.called("grid_dsr")[-1] == (305, [0.1, 0.3, 0.2])
    assert found.centre.result_record["metrics"]["dsr_n_grid"] == 3
    # neighbourhood.json keeps its shape: centre, neighbours in run order, N before and after the whole set
    doc = json.loads((genesis_env.repo / "runs" / found.centre.begin_record["run_id"] / "neighbourhood.json")
                     .read_text())  # fmt: skip
    assert doc["centre"]["run_id"] == begun[-1]["run_id"]
    assert [n["run_id"] for n in doc["neighbours"]] == [begun[0]["run_id"], begun[1]["run_id"]]
    assert (doc["n_before"], doc["n_after"]) == (302, 305)


def test_a_synthetic_neighbourhood_also_runs_the_centre_last(genesis_env: Env) -> None:
    found = run_with_neighbours(genesis_env.repo / STRATEGY_REL, kind="synthetic", params={"a": 2},
                                deps=FakeTrialEngine().deps(genesis_env),
                                runs_dir=genesis_env.repo / "runs")  # fmt: skip
    begun = [line.record for line in read_ledger(genesis_env.config.registry_path)
             if line.record["kind"] == "synthetic"]  # fmt: skip
    assert [r["params"]["a"] for r in begun] == [1, 3, 2]
    assert found.centre.begin_record["run_id"] == begun[-1]["run_id"]
