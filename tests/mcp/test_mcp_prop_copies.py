"""The MCP tool run_prop_mc's copies argument (follow-up) against pfb.app for real: temporary git
repositories and ledgers, the template on synthetic bars, never the real ledger.

On a synthetic run the tool returns exactly what pfb.app returns for the same copies request. On a trial run
copies is refused by the tool with the app's own refusal (TrialNotDeclared: a hypothesis cannot declare
copies), and nothing is written; with plan "*" (the matrix) it is refused too.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import anyio
import pytest
from app_env import FIRM, PLAN, TEMPLATE, AppEnv  # pyright: ignore[reportMissingImports]
from mcp import Client
from mcp_types import CallToolResult
from registry_env import HYPOTHESIS_REL, Env  # pyright: ignore[reportMissingImports]
from trial_support import DATA, FakeTrialEngine  # pyright: ignore[reportMissingImports]

from pfb import app
from pfb.contract import SCHEMA
from pfb.mcp.server import build_server
from pfb.registry.api import register_hypothesis
from pfb.runner.secondary import grid_dsr, prop_secondary

MC, SEED, COPIES = 300, 11, 2


def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    async def go() -> CallToolResult:
        async with Client(build_server()) as client:
            return await client.call_tool(name, arguments)

    return anyio.run(go)


def _engine_on_template() -> Any:
    def run_strategy(path: Path, **kwargs: Any) -> Any:
        from pfb.engine.run import run_strategy as real

        return real(
            TEMPLATE,
            source="synthetic",
            root=kwargs["root"],
            fees=kwargs["fees"],
            cost_model="fixed:1",
            seed=kwargs["seed"],
            n_trials=kwargs["n_trials"],
            suite=kwargs.get("suite"),
        )

    return run_strategy


@pytest.fixture
def trial(genesis_env: Env) -> Iterator[Env]:
    """A hypothesis declaring one prop plan at mc 300, seed 11, registered and checkpointed (as in
    tests/contract/test_trial_contract.py); the engine is the real one on synthetic bars."""
    env = genesis_env
    env.write(
        HYPOTHESIS_REL,
        env.hypothesis_text(
            secondary__prop_plans=f'["{FIRM}/{PLAN}"]',
            secondary__prop_mc=str(MC),
            secondary__prop_seed=str(SEED),
        ),
    )
    env.commit()
    register_hypothesis(env.repo / HYPOTHESIS_REL, env.config, env.facts())
    env.checkpoint()
    deps = replace(
        FakeTrialEngine().deps(env),
        run_strategy=_engine_on_template(),
        provenance_files=lambda r: DATA,
        grid_dsr=grid_dsr,
        prop_secondary=prop_secondary,
    )
    with app.using(deps=deps, runs_dir=env.repo / "runs", ledger=None):
        yield env


def test_copies_through_the_tool_equals_the_app_result_on_a_synthetic_run(app_env: AppEnv) -> None:
    # Arrange
    base = app.run_backtest(str(app_env.strategy), kind="synthetic", cost_model="fixed:1", baseline=False)
    request = {"firm": FIRM, "plan": PLAN, "mc": 200, "seed": 4, "copies": 3}
    expected = app.run_prop_mc(base["run_id"], **request)
    # Act
    result = call_tool("run_prop_mc", {"run_id": base["run_id"], **request})
    # Assert
    assert not result.is_error, result.content
    assert result.structured_content == expected
    [row] = expected["prop"]
    assert row["copies"]["n_accounts"] == 3
    assert (app_env.runs / base["run_id"] / "prop" / f"{FIRM}_{PLAN}_4_copies3.json").is_file()


def test_copies_on_a_trial_is_refused_by_the_tool_as_by_the_app(trial: Env) -> None:
    # Arrange: the declared plan, mc and seed; only copies is added
    doc = app.run_backtest("strategies/s.py", kind="trial", hypothesis=HYPOTHESIS_REL)
    run_id = doc["run_id"]
    request = {"firm": FIRM, "plan": PLAN, "mc": MC, "seed": SEED, "copies": COPIES}
    prop_dir = trial.repo / "runs" / run_id / "prop"
    files_before = sorted(p.name for p in prop_dir.iterdir())
    with pytest.raises(app.Refused) as caught:
        app.run_prop_mc(run_id, **request)
    # Act
    result = call_tool("run_prop_mc", {"run_id": run_id, **request})
    # Assert
    assert caught.value.code == "TrialNotDeclared"
    assert not result.is_error, result.content
    assert result.structured_content == {
        "schema": SCHEMA,
        "status": "refused",
        "error": {"code": caught.value.code, "reason": caught.value.reason},
    }
    assert sorted(p.name for p in prop_dir.iterdir()) == files_before
    assert not any("copies" in name for name in files_before)


def test_copies_with_every_plan_is_refused_by_the_tool_as_by_the_app(app_env: AppEnv) -> None:
    # Arrange
    base = app.run_backtest(str(app_env.strategy), kind="synthetic", cost_model="fixed:1", baseline=False)
    request = {"firm": "*", "plan": "*", "mc": 200, "seed": 4, "copies": 3}
    with pytest.raises(app.Refused) as caught:
        app.run_prop_mc(base["run_id"], **request)
    # Act
    result = call_tool("run_prop_mc", {"run_id": base["run_id"], **request})
    # Assert
    assert result.structured_content == {
        "schema": SCHEMA,
        "status": "refused",
        "error": {"code": caught.value.code, "reason": caught.value.reason},
    }
