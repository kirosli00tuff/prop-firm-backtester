"""Temporary git repositories and ledgers only.

R1: registry begin refuses a trial whose request.seed differs from its hypothesis's lookahead_seed, with the
same SeedMismatch class the runner raises (pfb.registry.errors; pfb.runner.api re-exports it).
R4: run_with_neighbours takes `seed: int | None = None`, so a trial left without a seed runs at the
hypothesis's lookahead_seed.
R5: replay (_check_hypothesis_shape) refuses a hypothesis record with purpose TEST and a budget other than 1,
or with two windows sharing a day, as the parser does (a record written by an older checkout).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from registry_env import HYPOTHESIS_REL, STRATEGY_REL, Env  # pyright: ignore[reportMissingImports]
from trial_support import FakeTrialEngine, kinds, n_of  # pyright: ignore[reportMissingImports]

import pfb.registry.records as records
import pfb.runner.api as runner_api
from pfb.registry import hypfile
from pfb.registry.api import RegistryConfig, RunRequest, begin, read_state, register_hypothesis
from pfb.registry.errors import ChainBroken, RegistryError, SeedMismatch
from pfb.runner.api import Plan, execute_planned, prepare, run_args
from pfb.runner.deps import RunnerDeps
from pfb.runner.neighbours import run_with_neighbours

Prepared = tuple[RunnerDeps, RegistryConfig, Plan, RunRequest]
APRIL = "[[window]]\nfirst = 2025-04-01\nlast = 2025-04-30\n"
MAY = "[[window]]\nfirst = 2025-05-01\nlast = 2025-05-30\n"
APRIL_TO_MAY = "[[window]]\nfirst = 2025-04-15\nlast = 2025-05-30\n"  # shares 04-15..04-30 with APRIL


def _register(env: Env, text: str) -> None:
    env.write(HYPOTHESIS_REL, text)
    env.commit()
    register_hypothesis(env.repo / HYPOTHESIS_REL, env.config, env.facts())
    env.checkpoint()


def _seeded(text: str, seed: str) -> str:
    head, sep, rest = text.partition("[[window]]")
    reason = 'lookahead_seed_reason = "a test of the seed rules"\n'
    return f"{head}lookahead_seed = {seed}\n{reason}{sep}{rest}"


def _prepared(env: Env) -> tuple[Path, Prepared]:
    path = (env.repo / STRATEGY_REL).resolve()
    args = run_args(hypothesis_path=Path(HYPOTHESIS_REL))
    return path, prepare("trial", path, args, {"a": 2}, None, FakeTrialEngine().deps(env))


# ------------------------------------------------------------------------ R1: begin checks the seed ----


def test_seed_mismatch_is_one_class_for_runner_and_registry() -> None:
    assert runner_api.SeedMismatch is SeedMismatch
    assert issubclass(SeedMismatch, RegistryError)


def test_begin_refuses_a_trial_at_a_seed_its_hypothesis_did_not_declare(genesis_env: Env) -> None:
    _register(genesis_env, _seeded(genesis_env.hypothesis_text(), "5"))
    _, (_, config, _, request) = _prepared(genesis_env)
    assert request.seed == 5
    n0 = n_of(genesis_env)
    with pytest.raises(SeedMismatch, match="lookahead_seed 5"):
        begin(replace(request, seed=7), config=config)
    assert n_of(genesis_env) == n0 and kinds(genesis_env) == ["genesis", "hypothesis"]


def test_begin_accepts_a_trial_at_its_hypothesis_seed(genesis_env: Env) -> None:
    _register(genesis_env, _seeded(genesis_env.hypothesis_text(), "5"))
    _, (_, config, _, request) = _prepared(genesis_env)
    receipt = begin(request, config=config)
    assert receipt.record["seed"] == 5
    assert kinds(genesis_env) == ["genesis", "hypothesis", "trial"]


def test_a_plan_and_request_both_moved_to_another_seed_are_refused_at_begin(genesis_env: Env) -> None:
    _register(genesis_env, _seeded(genesis_env.hypothesis_text(), "5"))
    path, (deps, config, plan, request) = _prepared(genesis_env)
    tampered = (deps, config, replace(plan, seed=7), replace(request, seed=7))
    with pytest.raises(SeedMismatch):
        execute_planned("trial", path, tampered, None, genesis_env.repo / "runs")
    assert kinds(genesis_env) == ["genesis", "hypothesis"]


# ------------------------------------------------------------- R4: neighbourhood seed default is None ----


def test_a_neighbourhood_trial_with_the_seed_left_out_runs_at_the_hypothesis_seed(genesis_env: Env) -> None:
    _register(genesis_env, _seeded(genesis_env.hypothesis_text(budget=3, suite__neighbourhood="true"), "5"))
    engine = FakeTrialEngine()
    found = run_with_neighbours(genesis_env.repo / STRATEGY_REL, kind="trial", params={"a": 2},
                                hypothesis_path=Path(HYPOTHESIS_REL), deps=engine.deps(genesis_env),
                                runs_dir=genesis_env.repo / "runs")  # fmt: skip
    assert {o.begin_record["seed"] for o in (found.centre, *found.neighbours)} == {5}


def test_a_synthetic_neighbourhood_with_the_seed_left_out_runs_at_seed_zero(genesis_env: Env) -> None:
    engine = FakeTrialEngine()
    found = run_with_neighbours(genesis_env.repo / STRATEGY_REL, kind="synthetic", params={"a": 2},
                                deps=engine.deps(genesis_env),
                                runs_dir=genesis_env.repo / "runs")  # fmt: skip
    assert {o.begin_record["seed"] for o in (found.centre, *found.neighbours)} == {0}


# ------------------------------------------------------------------- R5: replay applies F1 and F5 ----


def _register_under_older_checkout(env: Env, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    """Register `text` with the C.1 parser rules and the new replay rules switched off (an older checkout)."""
    with monkeypatch.context() as mp:
        mp.setattr(hypfile, "_purpose_budget", lambda purpose, budget, where: budget)
        mp.setattr(hypfile, "_refuse_overlap", lambda windows, where: None)
        mp.setattr(records, "_check_hypothesis_rules", lambda record, where: None)
        _register(env, text)


def test_replay_refuses_a_test_hypothesis_record_with_budget_two(
    genesis_env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = genesis_env.hypothesis_text(budget=2).replace('purpose = "research"', 'purpose = "test"')
    _register_under_older_checkout(genesis_env, monkeypatch, text)
    with pytest.raises(ChainBroken, match="budget"):
        read_state(genesis_env.config)


def test_replay_refuses_a_hypothesis_record_with_overlapping_windows(
    genesis_env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = genesis_env.hypothesis_text(budget=2, windows=APRIL + APRIL_TO_MAY)
    _register_under_older_checkout(genesis_env, monkeypatch, text)
    with pytest.raises(ChainBroken, match="overlap"):
        read_state(genesis_env.config)


def test_appending_a_test_hypothesis_record_with_budget_two_is_refused(
    genesis_env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = genesis_env.hypothesis_text(budget=2).replace('purpose = "research"', 'purpose = "test"')
    with monkeypatch.context() as mp:  # only the parser is old: the append's replay check refuses it
        mp.setattr(hypfile, "_purpose_budget", lambda purpose, budget, where: budget)
        with pytest.raises(ChainBroken):
            _register(genesis_env, text)
    assert kinds(genesis_env) == ["genesis"]


@pytest.mark.parametrize(
    ("budget", "windows"), [(1, APRIL), (2, APRIL + MAY)], ids=["test-one-window", "research-adjacent"]
)
def test_replay_accepts_records_that_keep_the_rules(genesis_env: Env, budget: int, windows: str) -> None:
    _register(genesis_env, genesis_env.hypothesis_text(budget=budget, windows=windows))
    assert read_state(genesis_env.config).n == n_of(genesis_env)
