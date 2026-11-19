"""TRIALS.md C2-C6 through pfb.runner with a fake engine: order, binding, counting, neighbours, secondary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from registry_env import HYPOTHESIS_REL, STRATEGY_REL, Env, git  # pyright: ignore[reportMissingImports]
from trial_support import FakeTrialEngine, clean, kinds, n_of  # pyright: ignore[reportMissingImports]

from pfb.analysis.spec import SuiteSpec
from pfb.registry.api import register_hypothesis, status
from pfb.registry.errors import BudgetExceeded, SuiteMismatch, UndeclaredConfiguration
from pfb.runner.api import RunOutcome, RunRefused, run
from pfb.runner.neighbours import run_with_neighbours

NO_SIGNAL = "look-ahead check saw no signal"


def _trial(env: Env, engine: FakeTrialEngine, **kwargs: Any) -> RunOutcome:
    return run(env.repo / STRATEGY_REL, kind="trial", hypothesis_path=Path(HYPOTHESIS_REL),
               deps=engine.deps(env), runs_dir=env.repo / "runs", **kwargs)  # fmt: skip


def _hypothesis(env: Env, **override: str) -> None:
    env.write(HYPOTHESIS_REL, env.hypothesis_text(**override))
    env.commit()
    register_hypothesis(env.repo / HYPOTHESIS_REL, env.config, env.facts())


# ------------------------------------------------------------------------------------------- order, C6 ----


def test_a_trial_logs_begin_then_result_and_leaves_the_ledger_committed(trial_env: Env) -> None:
    # Arrange
    engine = FakeTrialEngine()
    # Act
    outcome = _trial(trial_env, engine)
    # Assert
    assert kinds(trial_env) == ["genesis", "hypothesis", "trial", "result"]
    assert outcome.begin_record["n_after"] == 303 and outcome.result_record["status"] == "ok"
    assert clean(trial_env)  # checkpointed after begin and at the end
    log = git(trial_env.repo, "log", "--format=%s", "-3")
    assert log.splitlines()[0] == "ledger checkpoint: N 303, records 4"
    assert log.splitlines()[1] == "ledger checkpoint: N 303, records 3"
    kwargs = engine.called("run_strategy")[0]
    assert kwargs["source"] == "real" and kwargs["fees"] == "topstep" and kwargs["cost_model"] == "book"
    assert kwargs["suite"] == SuiteSpec.from_mapping(json.loads(json.dumps(
        {"cost_stress_ticks": [1, 2], "baseline_draws": 200, "baseline_seed": 7, "walk_forward_folds": 5,
         "neighbourhood": False, "sizing_plan": ""})))  # fmt: skip
    assert kwargs["permit"].run_id == outcome.begin_record["run_id"] and kwargs["n_trials"] == 303


def test_the_runner_checkpoints_uncommitted_ledger_lines_before_begin(trial_env: Env) -> None:
    run(trial_env.repo / STRATEGY_REL, kind="synthetic", deps=FakeTrialEngine().deps(trial_env),
        runs_dir=trial_env.repo / "runs")  # fmt: skip
    assert not clean(trial_env)
    _trial(trial_env, FakeTrialEngine())
    assert clean(trial_env) and kinds(trial_env)[-2:] == ["trial", "result"]


def test_suite_json_extras_and_declared_secondary_outputs_are_written(trial_env: Env) -> None:
    engine = FakeTrialEngine(extras=True)
    outcome = _trial(trial_env, engine)
    run_id = outcome.begin_record["run_id"]
    out = trial_env.repo / "runs" / run_id
    assert (out / "suite.json").is_file() and (out / "rolling_sharpe.parquet").is_file()
    assert f"runs/{run_id}/suite.json" in outcome.result_record["artifacts"]
    assert f"runs/{run_id}/rolling_sharpe.parquet" in outcome.result_record["artifacts"]
    assert engine.called("prop_secondary") == [(run_id, "MNQ", {"prop_plans": [], "prop_mc": 20000,
                                                                "prop_seed": 20261116})]  # fmt: skip
    assert outcome.secondary == {"rows": [], "declared": []}


def test_a_failed_secondary_output_is_a_warning_not_a_failed_run(trial_env: Env) -> None:
    outcome = _trial(trial_env, FakeTrialEngine(prop_error=ValueError("kind must be one of")))
    assert outcome.result_record["status"] == "ok" and outcome.secondary is None
    assert outcome.warnings and outcome.warnings[0].startswith("secondary prop outputs failed: ValueError")


def test_step_one_no_signal_goes_into_the_begin_record_and_is_refused_without_a_reason(
    trial_env: Env,
) -> None:
    with pytest.raises(Exception, match="no_signal_reason"):
        _trial(trial_env, FakeTrialEngine(lookahead_warnings=(NO_SIGNAL,)))
    assert kinds(trial_env) == ["genesis", "hypothesis"]


def test_step_one_no_signal_with_a_reason_is_recorded(genesis_env: Env) -> None:
    _hypothesis(genesis_env, no_signal_reason='"rare entries"')
    outcome = _trial(genesis_env, FakeTrialEngine(lookahead_warnings=(NO_SIGNAL,)))
    assert outcome.begin_record["lookahead_signal"] == "none"
    assert outcome.begin_record["warnings"] == [f"step 1: {NO_SIGNAL}"]
    assert outcome.result_record["metrics"]["warnings"][-1] == f"step 1: {NO_SIGNAL}"


# --------------------------------------------------------------------------------- binding, C4, s4 ----


@pytest.mark.parametrize(
    "kwargs",
    [{"fees": "lucid"}, {"cost_model": "fixed:2"}, {"suite": SuiteSpec.off()}],
    ids=["fees", "cost-model", "suite"],
)
def test_a_command_line_value_unlike_the_hypothesis_is_refused(trial_env: Env, kwargs: dict) -> None:
    with pytest.raises(SuiteMismatch):
        _trial(trial_env, FakeTrialEngine(), **kwargs)
    assert kinds(trial_env) == ["genesis", "hypothesis"]


def test_the_same_values_given_explicitly_are_accepted(trial_env: Env) -> None:
    outcome = _trial(trial_env, FakeTrialEngine(), fees="topstep", cost_model="book")
    assert outcome.result_record["status"] == "ok"


@pytest.mark.parametrize(
    "kwargs", [{"root": "MES"}, {"window": 2}, {"params": {"a": 9}}], ids=["root", "window", "params"]
)
def test_a_root_window_or_params_not_declared_are_refused_before_begin(trial_env: Env, kwargs: dict) -> None:
    with pytest.raises((UndeclaredConfiguration, RunRefused)):
        _trial(trial_env, FakeTrialEngine(), **kwargs)
    assert kinds(trial_env) == ["genesis", "hypothesis"]


def test_with_two_windows_the_window_index_chooses_one(genesis_env: Env) -> None:
    windows = (
        "[[window]]\nfirst = 2025-04-01\nlast = 2025-04-30\n"
        "[[window]]\nfirst = 2025-05-01\nlast = 2025-05-30\n"
    )
    _hypothesis(genesis_env, windows=windows)
    with pytest.raises(RunRefused, match="--window"):
        _trial(genesis_env, FakeTrialEngine())
    outcome = _trial(genesis_env, FakeTrialEngine(), window=2)
    assert (outcome.begin_record["first"], outcome.begin_record["last"]) == ("2025-05-01", "2025-05-30")


def test_a_hypothesis_naming_another_strategy_is_refused(trial_env: Env) -> None:
    other = trial_env.write("strategies/other.py", "y = 1\n")
    trial_env.commit()
    with pytest.raises(RunRefused, match="pins"):
        run(other, kind="trial", hypothesis_path=Path(HYPOTHESIS_REL), deps=FakeTrialEngine().deps(trial_env),
            runs_dir=trial_env.repo / "runs")  # fmt: skip


# ------------------------------------------------------------------------------------------ counting ----


def test_n_moves_by_one_for_a_new_configuration_and_zero_for_an_exact_replay(trial_env: Env) -> None:
    _trial(trial_env, FakeTrialEngine())
    assert n_of(trial_env) == 303
    _trial(trial_env, FakeTrialEngine())
    assert n_of(trial_env) == 303
    _trial(trial_env, FakeTrialEngine(), params={"a": 2})
    assert n_of(trial_env) == 304


def test_a_crash_after_begin_still_counts_and_leaves_a_failed_result(trial_env: Env) -> None:
    with pytest.raises(ZeroDivisionError):
        _trial(trial_env, FakeTrialEngine(crash=True))
    assert kinds(trial_env)[-2:] == ["trial", "result"] and n_of(trial_env) == 303
    assert clean(trial_env)  # the final checkpoint runs on failure too


def test_a_refused_trial_adds_nothing(trial_env: Env) -> None:
    _trial(trial_env, FakeTrialEngine(), params={"a": 1})
    _trial(trial_env, FakeTrialEngine(), params={"a": 2})
    with pytest.raises(BudgetExceeded):
        _trial(trial_env, FakeTrialEngine(), params={"a": 3})
    assert n_of(trial_env) == 304 and status(trial_env.config).open_runs == ()


def test_the_dsr_grid_holds_the_ok_sharpes_of_the_hypothesis_this_run_last(trial_env: Env) -> None:
    summaries = {1: {"net_cents": 5, "sharpe_daily": 0.1}, 2: {"net_cents": 6, "sharpe_daily": 0.2}}
    first = FakeTrialEngine(summaries=summaries)
    _trial(trial_env, first, params={"a": 1})
    assert first.called("grid_dsr") == []  # one configuration: the floor variance only
    second = FakeTrialEngine(summaries=summaries)
    outcome = _trial(trial_env, second, params={"a": 2})
    assert second.called("grid_dsr") == [(304, [0.1, 0.2])]
    assert outcome.result_record["metrics"]["dsr_n_grid"] == 2


# ---------------------------------------------------------------------------------------- neighbours ----


def _neighbourhood_hypothesis(env: Env, budget: int = 3) -> None:
    _hypothesis(env, budget=str(budget), suite__neighbourhood="true")


def test_a_trial_whose_hypothesis_declares_neighbours_cannot_run_alone(genesis_env: Env) -> None:
    _neighbourhood_hypothesis(genesis_env)
    with pytest.raises(SuiteMismatch):
        _trial(genesis_env, FakeTrialEngine())


def test_a_neighbourhood_of_k_new_configurations_adds_k(genesis_env: Env) -> None:
    # Arrange: centre a = 2, neighbours a = 1 and a = 3
    _neighbourhood_hypothesis(genesis_env)
    engine = FakeTrialEngine()
    # Act
    found = run_with_neighbours(genesis_env.repo / STRATEGY_REL, kind="trial", params={"a": 2},
                                hypothesis_path=Path(HYPOTHESIS_REL), deps=engine.deps(genesis_env),
                                runs_dir=genesis_env.repo / "runs")  # fmt: skip
    # Assert
    assert n_of(genesis_env) == 302 + 3
    assert [o.begin_record["params"]["a"] for o in (found.centre, *found.neighbours)] == [2, 1, 3]
    doc = json.loads((genesis_env.repo / "runs" / found.centre.begin_record["run_id"] / "neighbourhood.json")
                     .read_text())  # fmt: skip
    assert (
        doc["n_before"] == 302
        and doc["n_after"] == 305
        and doc["verdict"]["verdict"] == found.verdict["verdict"]
    )
    assert clean(genesis_env)


def test_a_neighbourhood_the_budget_cannot_cover_is_refused_whole(genesis_env: Env) -> None:
    _neighbourhood_hypothesis(genesis_env, budget=2)
    engine = FakeTrialEngine()
    with pytest.raises(BudgetExceeded, match="whole set"):
        run_with_neighbours(genesis_env.repo / STRATEGY_REL, kind="trial", params={"a": 2},
                            hypothesis_path=Path(HYPOTHESIS_REL), deps=engine.deps(genesis_env),
                            runs_dir=genesis_env.repo / "runs")  # fmt: skip
    assert kinds(genesis_env) == ["genesis", "hypothesis"] and n_of(genesis_env) == 302
    assert engine.called("run_strategy") == []


def test_a_neighbourhood_counts_only_configurations_not_yet_counted(genesis_env: Env) -> None:
    _neighbourhood_hypothesis(genesis_env)
    deps = FakeTrialEngine().deps(genesis_env)
    run_with_neighbours(genesis_env.repo / STRATEGY_REL, kind="trial", params={"a": 1},
                        hypothesis_path=Path(HYPOTHESIS_REL), deps=deps,
                        runs_dir=genesis_env.repo / "runs")  # a = 1 and its neighbour a = 2  # fmt: skip
    assert n_of(genesis_env) == 304
    run_with_neighbours(genesis_env.repo / STRATEGY_REL, kind="trial", params={"a": 2},
                        hypothesis_path=Path(HYPOTHESIS_REL), deps=deps,
                        runs_dir=genesis_env.repo / "runs")  # only a = 3 is new  # fmt: skip
    assert n_of(genesis_env) == 305


def test_a_synthetic_neighbourhood_leaves_n_as_is(genesis_env: Env) -> None:
    found = run_with_neighbours(genesis_env.repo / STRATEGY_REL, kind="synthetic", params={"a": 2},
                                deps=FakeTrialEngine().deps(genesis_env),
                                runs_dir=genesis_env.repo / "runs")  # fmt: skip
    assert len(found.neighbours) == 2 and n_of(genesis_env) == 302
    assert kinds(genesis_env).count("synthetic") == 3


def test_an_engine_check_has_no_neighbours(genesis_env: Env) -> None:
    with pytest.raises(RunRefused):
        run_with_neighbours(genesis_env.repo / STRATEGY_REL, kind="engine_check",
                            deps=FakeTrialEngine().deps(genesis_env))  # fmt: skip
