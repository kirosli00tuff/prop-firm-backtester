"""Begin and finish of trial, engine_check and synthetic runs: N, refusals, committed inputs (A1-A14)."""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from registry_env import DECLARED_IN, HYPOTHESIS_REL, STRATEGY_REL, Env, git

from pfb.registry.api import Receipt, begin, finish, read_state, register_hypothesis, status
from pfb.registry.chain import canonical_bytes, read_ledger, sha256_hex
from pfb.registry.errors import (
    BadRequest,
    BudgetExceeded,
    ChecksNotPassed,
    DirtyTree,
    EngineCheckNotDeclared,
    HypothesisChanged,
    HypothesisInvalid,
    HypothesisNotRegistered,
    InputNotCommitted,
    ResultExists,
    RunNotBegun,
    SeedMismatch,
    StrategyChanged,
    UndeclaredConfiguration,
)
from pfb.registry.inputs import load_engine_checks
from pfb.registry.records import KEYS, config_id, replay

ENGINE_CHECKS_REL = "config/engine_checks.toml"


def _records(env: Env) -> int:
    return len(read_ledger(env.config.registry_path))


# ------------------------------------------------------------------------------------------------ trial ----


def test_trial_raises_n_by_one_for_a_new_configuration(trial_env: Env) -> None:
    receipt = begin(trial_env.request("trial"), config=trial_env.config)
    assert (receipt.n_before, receipt.n_after) == (302, 303)
    assert receipt.record["configurations"] == [config_id(receipt.record)]
    assert set(receipt.record) == KEYS["trial"]


def test_an_exact_replay_adds_nothing_and_is_still_logged(trial_env: Env) -> None:
    trial_env.begin_trial()
    receipt = trial_env.begin_trial()
    assert (receipt.n_before, receipt.n_after) == (303, 303)
    assert read_state(trial_env.config).n == 303


def test_a_different_seed_is_the_same_configuration(trial_env: Env) -> None:
    # the seed is not part of config_id; since C.1 R1 a trial can only begin at its hypothesis's seed
    first = trial_env.begin_trial()
    assert config_id({**first.record, "seed": 9}) == config_id(first.record)


def test_a_trial_at_another_seed_than_its_hypothesis_is_refused_at_begin(trial_env: Env) -> None:
    trial_env.begin_trial()
    with pytest.raises(SeedMismatch, match="lookahead_seed 0"):
        trial_env.begin_trial(seed=9)
    assert read_state(trial_env.config).n == 303


def test_budget_is_cumulative_over_the_hypothesis(trial_env: Env) -> None:
    trial_env.begin_trial(params={"a": 1})
    trial_env.begin_trial(params={"a": 2})
    with pytest.raises(BudgetExceeded):
        trial_env.begin_trial(params={"a": 3})
    assert read_state(trial_env.config).n == 304


@pytest.mark.parametrize(
    "override",
    [
        {"params": {"a": 4}},
        {"params": {"a": 1, "b": 1}},
        {"params": {}},
        {"params": {"a": "1"}},
        {"params": {"a": True}},
        {"root": "NQ"},
        {"first": "2025-03-31"},
        {"last": "2025-05-01"},
    ],
    ids=["value", "extra-key", "missing-key", "string", "bool", "root", "early", "late"],
)
def test_an_undeclared_configuration_is_refused(trial_env: Env, override: dict[str, object]) -> None:
    with pytest.raises(UndeclaredConfiguration):
        begin(trial_env.request("trial", **override), config=trial_env.config)
    assert _records(trial_env) == 2


def test_a_trial_without_a_registered_hypothesis_is_refused(genesis_env: Env) -> None:
    genesis_env.write(HYPOTHESIS_REL, genesis_env.hypothesis_text())
    genesis_env.commit()
    genesis_env.checkpoint()
    with pytest.raises(HypothesisNotRegistered):
        begin(genesis_env.request("trial"), config=genesis_env.config)


def test_a_changed_and_committed_hypothesis_file_is_refused(trial_env: Env) -> None:
    trial_env.write(HYPOTHESIS_REL, trial_env.hypothesis_text(budget=3))
    trial_env.commit()
    with pytest.raises(HypothesisChanged):
        begin(trial_env.request("trial"), config=trial_env.config)


def test_a_changed_and_committed_strategy_is_refused(trial_env: Env) -> None:
    trial_env.write(STRATEGY_REL, "x = 2\n")
    trial_env.commit()
    with pytest.raises(StrategyChanged):
        begin(trial_env.request("trial"), config=trial_env.config)


def test_a_request_hash_that_differs_from_the_strategy_file_is_refused(trial_env: Env) -> None:
    with pytest.raises(StrategyChanged):
        begin(trial_env.request("trial", strategy_sha256="c" * 64), config=trial_env.config)


def test_a_trial_on_a_dirty_tree_is_refused(trial_env: Env) -> None:
    trial_env.write("notes.txt", "untracked\n")
    with pytest.raises(DirtyTree):
        begin(trial_env.request("trial"), config=trial_env.config)


def test_a_trial_with_ledger_lines_not_at_head_is_refused(trial_env: Env) -> None:
    # TRIALS 4: nothing may differ from HEAD, the ledger included (the runner checkpoints first, C6)
    begin(trial_env.request("synthetic"), config=trial_env.config)
    assert "registry.jsonl" in git(trial_env.repo, "status", "--porcelain", "--untracked-files=all")
    with pytest.raises(DirtyTree):
        begin(trial_env.request("trial"), config=trial_env.config)
    assert trial_env.begin_trial().record["git_dirty"] is False


def test_a_trial_with_an_uncommitted_strategy_change_is_refused(trial_env: Env) -> None:
    trial_env.write(STRATEGY_REL, "x = 1  # edited\n")
    request = trial_env.request("trial")
    with pytest.raises(InputNotCommitted):
        begin(replace(request, strategy_sha256=trial_env.strategy_sha), config=trial_env.config)


def test_an_untracked_hypothesis_is_refused_at_registration(genesis_env: Env) -> None:
    genesis_env.write(HYPOTHESIS_REL, genesis_env.hypothesis_text())
    with pytest.raises(InputNotCommitted):
        register_hypothesis(genesis_env.repo / HYPOTHESIS_REL, genesis_env.config, genesis_env.facts())


def test_a_trial_with_an_uncommitted_hypothesis_edit_is_refused(trial_env: Env) -> None:
    trial_env.write(HYPOTHESIS_REL, trial_env.hypothesis_text(budget=3))
    with pytest.raises(InputNotCommitted):
        begin(trial_env.request("trial"), config=trial_env.config)


def test_a_failed_lint_or_lookahead_check_is_refused(trial_env: Env) -> None:
    for override in ({"lint_passed": False}, {"lookahead_passed": False}):
        with pytest.raises(ChecksNotPassed):
            begin(trial_env.request("trial", **override), config=trial_env.config)
    assert _records(trial_env) == 2


def test_a_hypothesis_id_is_registered_once(trial_env: Env) -> None:
    with pytest.raises(HypothesisInvalid):
        register_hypothesis(trial_env.repo / HYPOTHESIS_REL, trial_env.config, trial_env.facts())


@pytest.mark.parametrize(
    "text_change",
    [
        ("budget = 2", "budget = 0"),
        # "on" registers (tests/trials/test_event_costs_wiring.py); "yes" does not
        ('event_costs = "off"', 'event_costs = "yes"'),
        ("a = [1, 2, 3]", "a = []"),
        ('id = "H1"', 'id = "H1"\nextra = 1'),
    ],
    ids=["budget", "event-costs", "empty-grid", "extra-key"],
)
def test_a_malformed_hypothesis_is_refused(genesis_env: Env, text_change: tuple[str, str]) -> None:
    path = genesis_env.write("hypotheses/bad.toml", genesis_env.hypothesis_text().replace(*text_change))
    genesis_env.commit()
    with pytest.raises(HypothesisInvalid):
        register_hypothesis(path, genesis_env.config, genesis_env.facts())


def test_hypothesis_registration_leaves_n_unchanged(trial_env: Env) -> None:
    record = read_ledger(trial_env.config.registry_path)[-1].record
    assert (record["kind"], record["n_before"], record["n_after"]) == ("hypothesis", 302, 302)


# ----------------------------------------------------------------------------------------- engine_check ----


def _with_allowlist(env: Env, *, commit: bool = True, **override: Any) -> None:
    env.write(ENGINE_CHECKS_REL, env.engine_checks_text(**override))
    if commit:
        env.commit()


def test_a_declared_engine_check_is_logged_with_n_unchanged(genesis_env: Env) -> None:
    _with_allowlist(genesis_env)
    receipt = begin(genesis_env.request("engine_check"), config=genesis_env.config)
    assert (receipt.n_before, receipt.n_after) == (302, 302)
    assert receipt.record["check"]["id"] == "E3"
    assert receipt.record["declared_in"] == DECLARED_IN
    assert receipt.record["engine_checks_sha256"] is not None


def test_every_engine_check_is_refused_without_the_allowlist(genesis_env: Env) -> None:
    with pytest.raises(EngineCheckNotDeclared):
        begin(genesis_env.request("engine_check"), config=genesis_env.config)


def test_an_unknown_check_id_is_refused(genesis_env: Env) -> None:
    _with_allowlist(genesis_env)
    with pytest.raises(EngineCheckNotDeclared):
        begin(genesis_env.request("engine_check", check_id="E9"), config=genesis_env.config)


@pytest.mark.parametrize(
    "override",
    [
        {"root": "NQ"},
        {"first": "2025-04-02"},
        {"last": "2025-04-29"},
        {"fees": "bulenox"},
        {"cost_model": "fixed:1"},
    ],
    ids=["root", "first", "last", "fees", "cost-model"],
)
def test_an_engine_check_differing_from_its_entry_is_refused(
    genesis_env: Env, override: dict[str, str]
) -> None:
    _with_allowlist(genesis_env)
    with pytest.raises(EngineCheckNotDeclared):
        begin(genesis_env.request("engine_check", **override), config=genesis_env.config)


def test_an_engine_check_with_an_uncommitted_allowlist_is_refused(genesis_env: Env) -> None:
    _with_allowlist(genesis_env, commit=False)
    with pytest.raises(InputNotCommitted):
        begin(genesis_env.request("engine_check"), config=genesis_env.config)


def test_an_engine_check_whose_declared_in_is_absent_at_head_is_refused(genesis_env: Env) -> None:
    _with_allowlist(genesis_env, declared_in="docs/checks/NOT_THERE.md")
    with pytest.raises(InputNotCommitted):
        begin(genesis_env.request("engine_check"), config=genesis_env.config)


def test_an_engine_check_whose_strategy_has_uncommitted_changes_is_refused(genesis_env: Env) -> None:
    _with_allowlist(genesis_env)
    genesis_env.write(STRATEGY_REL, "x = 1  # edited\n")
    request = genesis_env.request("engine_check", strategy_sha256=genesis_env.strategy_sha)
    with pytest.raises(InputNotCommitted):
        begin(request, config=genesis_env.config)


# -------------------------------------------------------------------------------------------- synthetic ----


def test_a_synthetic_run_needs_no_commit_and_records_git_dirty(genesis_env: Env) -> None:
    outside = genesis_env.repo.parent / "outside.py"
    outside.write_text("y = 1\n")
    request = genesis_env.request("synthetic", strategy_path=outside, strategy_sha256=sha256_hex(b"y = 1\n"))
    receipt = begin(request, config=genesis_env.config)
    assert receipt.record["git_dirty"] is False
    assert receipt.record["strategy_path"] == "<outside repo>/outside.py"
    assert receipt.n_after == 302


def test_a_synthetic_run_with_data_files_is_refused(genesis_env: Env) -> None:
    with pytest.raises(BadRequest, match="synthetic"):
        begin(
            genesis_env.request("synthetic", data_files=(("research/f.parquet", "a" * 64),)),
            config=genesis_env.config,
        )


def test_an_unknown_kind_is_refused(genesis_env: Env) -> None:
    with pytest.raises(BadRequest):
        begin(genesis_env.request("trial_run"), config=genesis_env.config)


# --------------------------------------------------------------------------------------- finish and G7 ----


def test_a_begun_run_without_result_is_listed_as_open(genesis_env: Env) -> None:
    receipt = begin(genesis_env.request("synthetic"), config=genesis_env.config)
    assert status(genesis_env.config).open_runs == (receipt.run_id,)
    finish(receipt, "ok", {"x": 1}, ["runs/a/summary.json"], config=genesis_env.config,
           artifact_sha256={"runs/a/summary.json": "e" * 64})  # fmt: skip
    assert status(genesis_env.config).open_runs == ()


def test_a_crashed_trial_still_counts(trial_env: Env) -> None:
    receipt = begin(trial_env.request("trial"), config=trial_env.config)
    assert status(trial_env.config).open_runs == (receipt.run_id,)
    assert status(trial_env.config).n == 303


def test_finish_records_status_metrics_and_planned_artifacts_with_n_unchanged(genesis_env: Env) -> None:
    receipt = begin(genesis_env.request("synthetic"), config=genesis_env.config)
    line = finish(
        receipt,
        "failed",
        None,
        ["runs/a/trades.parquet"],
        config=genesis_env.config,
        error="LookAheadDetected",
        artifact_sha256={"runs/a/trades.parquet": "e" * 64},
    )
    assert line.record["kind"] == "result" and line.record["error"] == "LookAheadDetected"
    assert line.record["artifacts"] == ["runs/a/trades.parquet"]
    assert line.record["n_before"] == line.record["n_after"] == 302


def test_a_second_result_for_one_run_is_refused(genesis_env: Env) -> None:
    receipt = begin(genesis_env.request("synthetic"), config=genesis_env.config)
    finish(receipt, "ok", {}, [], config=genesis_env.config)
    with pytest.raises(ResultExists):
        finish(receipt, "ok", {}, [], config=genesis_env.config)


def test_a_result_without_begin_is_refused(genesis_env: Env) -> None:
    foreign = Receipt("synthetic-unknown", "synthetic", "0" * 64, 302, 302, {})
    with pytest.raises(RunNotBegun):
        finish(foreign, "ok", {}, [], config=genesis_env.config)


@pytest.mark.parametrize("args", [("done", None), ("failed", None), ("ok", "Boom")])
def test_a_result_status_and_error_must_agree(genesis_env: Env, args: tuple[str, str | None]) -> None:
    receipt = begin(genesis_env.request("synthetic"), config=genesis_env.config)
    with pytest.raises(BadRequest):
        finish(receipt, args[0], {}, [], config=genesis_env.config, error=args[1])
    assert status(genesis_env.config).open_runs == (receipt.run_id,)


def test_metrics_with_nan_are_refused_before_writing(genesis_env: Env) -> None:
    receipt = begin(genesis_env.request("synthetic"), config=genesis_env.config)
    with pytest.raises(BadRequest):
        finish(receipt, "ok", {"sharpe": float("nan")}, [], config=genesis_env.config)
    assert _records(genesis_env) == 2


def test_a_tracked_record_never_holds_an_absolute_home_path(trial_env: Env) -> None:
    begin(trial_env.request("trial"), config=trial_env.config)
    text = trial_env.config.registry_path.read_text()
    assert str(Path.home()) not in text and str(trial_env.repo) not in text
    assert all(json.loads(x)["schema"] == "pfb.registry/1" for x in text.splitlines())


# -------------------------------------------- session, seed and params bound to the entry ----


@pytest.mark.parametrize(
    "override",
    [
        {"params": {"a": 999}},
        {"params": {"a": 999, "b": "anything"}},
        {"params": {"a": 1.0}},
        {"session": "rth"},
        {"seed": 7},
    ],
    ids=["param-value", "extra-param", "param-type", "session", "seed"],
)
def test_an_engine_check_whose_session_seed_or_params_differ_from_its_entry_is_refused(
    genesis_env: Env, override: dict[str, object]
) -> None:
    # Before the fix these gave (302, 302) straight through registry.api.begin.
    _with_allowlist(genesis_env)
    with pytest.raises(EngineCheckNotDeclared):
        begin(genesis_env.request("engine_check", **override), config=genesis_env.config)
    assert _records(genesis_env) == 1


def test_the_entry_session_seed_and_params_are_recorded_in_the_check(genesis_env: Env) -> None:
    _with_allowlist(genesis_env)
    record = begin(genesis_env.request("engine_check"), config=genesis_env.config).record
    check = record["check"]
    assert (check["session"], check["seed"], check["params"]) == ("globex", 0, {"a": 1})
    assert (record["session"], record["seed"], record["params"]) == ("globex", 0, {"a": 1})


@pytest.mark.parametrize(
    "field_value", [("params", {"a": 2}), ("session", "rth"), ("seed", 3)], ids=["params", "session", "seed"]
)
def test_replay_refuses_an_engine_check_record_that_differs_from_its_recorded_check(
    genesis_env: Env, field_value: tuple[str, object]
) -> None:
    _with_allowlist(genesis_env)
    begin(genesis_env.request("engine_check"), config=genesis_env.config)
    genesis, record = (dict(line.record) for line in read_ledger(genesis_env.config.registry_path))
    key, value = field_value
    edited = {**record, key: value}
    edited["configurations"] = [config_id(edited)]
    with pytest.raises(EngineCheckNotDeclared):
        replay([genesis, edited])


@pytest.mark.parametrize(
    "bad", ['seed = "0"', "seed = true", "params = 3"], ids=["seed-str", "seed-bool", "params"]
)
def test_an_allowlist_entry_with_a_malformed_seed_or_params_is_refused(tmp_path: Path, bad: str) -> None:
    text = (
        'schema = "pfb.engine_checks/1"\n[[check]]\nid = "E3"\ndeclared_in = "d.md"\nstrategy = "s.py"\n'
        f'strategy_sha256 = "{"a" * 64}"\nroot = "MNQ"\nfirst = "2025-04-01"\nlast = "2025-04-30"\n'
        'fees = "topstep"\ncost_model = "book"\nsession = "globex"\n'
    )
    key = bad.split(" ")[0]
    good = {"seed": "seed = 0\n", "params": "params = { a = 1 }\n"}
    body = text + "".join(bad + "\n" if k == key else line for k, line in good.items())
    path = tmp_path / "engine_checks.toml"
    path.write_text(body)
    with pytest.raises(EngineCheckNotDeclared):
        load_engine_checks(path)


REAL_ALLOWLIST = Path(__file__).resolve().parents[2] / ENGINE_CHECKS_REL
REFERENCE = Path(__file__).resolve().parents[2] / "strategies" / "reference_breakout.py"


def test_every_real_entry_declares_globex_seed_0_and_the_reference_spec_defaults() -> None:
    checks = load_engine_checks(REAL_ALLOWLIST)
    assert checks is not None and sorted(checks.entries) == ["E3", "E6_10Y", "E6_1Y", "E6_5Y"]
    tree = ast.parse(REFERENCE.read_text())
    spec = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SPEC" for t in n.targets)
    )
    defaults = {name: p["default"] for name, p in spec["params"].items()}
    for entry in checks.entries.values():
        assert (entry.session, entry.seed) == ("globex", 0)
        assert canonical_bytes(dict(entry.params)) == canonical_bytes(defaults)
