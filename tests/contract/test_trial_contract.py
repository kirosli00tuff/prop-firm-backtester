"""J2 on a trial (temporary git repository and ledger; no real data): the declared [secondary] prop plan runs
after the result record (pfb.prop.mc KINDS now holds "trial"), lands under runs/<run_id>/prop/ and in the
contract; a declared prop request on the trial run gives the same numbers; an undeclared plan, mc or seed is
refused. The engine is the real one on synthetic bars (the template), standing in for the gateway read."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from app_env import FIRM, PLAN, TEMPLATE
from registry_env import HYPOTHESIS_REL, Env  # pyright: ignore[reportMissingImports]
from trial_support import DATA, FakeTrialEngine, clean  # pyright: ignore[reportMissingImports]

from pfb import app
from pfb.contract import validate
from pfb.registry.api import register_hypothesis
from pfb.registry.chain import read_ledger
from pfb.registry.powercheck import power_at
from pfb.runner.secondary import grid_dsr, prop_secondary

MC, SEED = 300, 11


def _engine_on_template(calls: list[Any]) -> Any:
    def run_strategy(path: Path, **kwargs: Any) -> Any:
        from pfb.engine.run import run_strategy as real

        calls.append(kwargs)
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
def trial(genesis_env: Env) -> Iterator[tuple[Env, Any]]:
    """H1 declaring one prop plan at mc 300, seed 11, registered and checkpointed; deps whose engine is the
    real one on synthetic bars and whose secondary output is the real prop_secondary."""
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
    fake = FakeTrialEngine()
    calls: list[Any] = []
    deps = replace(fake.deps(env), run_strategy=_engine_on_template(calls), provenance_files=lambda r: DATA,
                   grid_dsr=grid_dsr, prop_secondary=prop_secondary)  # fmt: skip
    with app.using(deps=deps, runs_dir=env.repo / "runs", ledger=None):
        yield env, deps


def test_a_trial_contract_carries_its_declared_prop_rows(trial: tuple[Env, Any]) -> None:
    env, _ = trial
    # Act
    doc = app.run_backtest("strategies/s.py", kind="trial", hypothesis=HYPOTHESIS_REL, params={"a": 2})
    # Assert
    validate(doc)
    assert (doc["kind"], doc["status"], doc["n_added"], doc["purpose"]) == ("trial", "ok", 1, "research")
    assert doc["window"] == ["2025-04-01", "2025-04-30"] and doc["params"] == {"a": 2}
    run_dir = env.repo / "runs" / doc["run_id"]
    index = json.loads((run_dir / "prop" / "index.json").read_text())
    assert doc["prop"] == index["rows"] and len(doc["prop"]) == 1
    [row] = doc["prop"]
    record = json.loads((run_dir / "prop" / f"{FIRM}_{PLAN}_{SEED}.json").read_text())
    assert (row["status"], row["n_paths"], row["seed"]) == ("ok", MC, SEED) and record["kind"] == "trial"
    assert row["p_pass"] == record["outputs"]["p_pass"]["value"]
    assert any(a.endswith("/prop/index.json") for a in doc["artifacts"])
    # the stated power: the begin record's, and alpha recomputed from the hypothesis record at its n_used
    lines = read_ledger(env.config.registry_path)
    begin = next(ln.record for ln in lines if ln.record.get("run_id") == doc["run_id"])
    hyp = next(ln.record for ln in lines if ln.record["kind"] == "hypothesis")
    assert doc["power"]["min_edge_cents"] == begin["power"]["min_edge_cents"]
    assert doc["power"]["target_edge_cents"] == begin["power"]["target_edge_cents"] == 1000
    alpha = power_at(hyp["power"], hyp["windows"], begin["power"]["n_used"]).min_edge_cents_alpha
    assert doc["power"]["min_edge_cents_alpha"] == alpha
    assert not any(w.startswith("secondary prop outputs failed") for w in doc["warnings"])
    assert clean(env)


def test_a_declared_prop_request_on_the_trial_repeats_the_secondary_numbers(trial: tuple[Env, Any]) -> None:
    doc = app.run_backtest("strategies/s.py", kind="trial", hypothesis=HYPOTHESIS_REL)
    again = app.run_prop_mc(doc["run_id"], firm=FIRM, plan=PLAN, mc=MC, seed=SEED)
    validate(again)
    assert again["prop"] == doc["prop"]


@pytest.mark.parametrize(
    "change",
    [{"plan": "combine_xfa_consistency_50k"}, {"mc": MC + 1}, {"seed": SEED + 1}, {"stress_ticks": 1},
     {"soft_daily_loss": "minute_low"}, {"block": 6}],
    ids=["plan", "mc", "seed", "stress", "soft", "block"],
)  # fmt: skip
def test_an_undeclared_prop_request_on_a_trial_is_refused(
    trial: tuple[Env, Any], change: dict[str, Any]
) -> None:
    doc = app.run_backtest("strategies/s.py", kind="trial", hypothesis=HYPOTHESIS_REL)
    args: dict[str, Any] = {"firm": FIRM, "plan": PLAN, "mc": MC, "seed": SEED, **change}
    with pytest.raises(app.Refused) as caught:
        app.run_prop_mc(doc["run_id"], **args)
    assert caught.value.code == "TrialNotDeclared"
    validate(caught.value.to_json())


def test_a_trial_window_given_as_dates_maps_to_its_declared_index(trial: tuple[Env, Any]) -> None:
    doc = app.run_backtest("strategies/s.py", kind="trial", hypothesis=HYPOTHESIS_REL,
                           window=["2025-04-01", "2025-04-30"])  # fmt: skip
    assert doc["window"] == ["2025-04-01", "2025-04-30"]
    with pytest.raises(app.Refused) as caught:
        app.run_backtest(
            "strategies/s.py", kind="trial", hypothesis=HYPOTHESIS_REL, window=["2025-04-01", "2025-05-30"]
        )
    assert caught.value.code == "UndeclaredConfiguration"


def test_template_copy_is_independent_of_the_repository_file(tmp_path: Path) -> None:
    # pfb new copies the template with its id set and never overwrites
    with app.using(strategies_dir=tmp_path):
        out = app.new_strategy("my_orb")
        text = (tmp_path / "my_orb.py").read_text()
        assert out == {"schema": "pfb.new/1", "path": "strategies/my_orb.py", "hypothesis_id": "h_my_orb"}
        assert (
            '"hypothesis_id": "h_my_orb",' in text
            and text.replace("h_my_orb", "h_template") == TEMPLATE.read_text()
        )
        with pytest.raises(app.Refused) as caught:
            app.new_strategy("my_orb")
        assert caught.value.code == "Exists"
        for bad in ("", "9x", "a-b", "../x", "é"):
            with pytest.raises(app.Refused):
                app.new_strategy(bad)
