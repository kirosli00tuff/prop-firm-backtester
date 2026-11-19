"""TRIALS.md section 2: schema-2 hypothesis files and every refusal of registration (N never moves)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from registry_env import (  # pyright: ignore[reportMissingImports]
    HYPOTHESIS_REL,
    PE_AMENDMENT,
    PE_BASE_LINES,
    STRATEGY_REL,
    Env,
    pe_line,
)
from trial_support import FakeTrialEngine, kinds, n_of  # pyright: ignore[reportMissingImports]

from pfb.analysis.power import min_edge, years_of
from pfb.data.errors import ClosedWindowRefused, NoBarsFound
from pfb.registry.api import begin, finish, hypothesis_record, register_hypothesis
from pfb.registry.errors import (
    BudgetExceeded,
    ClosedWindowTouched,
    EngineCheckStrategy,
    HypothesisInvalid,
    InputNotCommitted,
    NoDataFile,
    PowerInsufficient,
    SpecMismatch,
    StrategyChanged,
    SyncRequired,
    ZeroSlippage,
)
from pfb.registry.records import KEYS
from pfb.runner.register import register

ENGINE_CHECKS_REL = "config/engine_checks.toml"


def _register(env: Env, text: str, **facts: object) -> None:
    path = env.write(HYPOTHESIS_REL, text)
    env.commit()
    register_hypothesis(path, env.config, env.facts(**facts))


# ------------------------------------------------------------------------------------------ the record ----


def test_registration_writes_the_section_3_record_and_adds_nothing_to_n(genesis_env: Env) -> None:
    # Arrange / Act
    _register(genesis_env, genesis_env.hypothesis_text())
    record = hypothesis_record(genesis_env.config, "H1")
    # Assert
    assert set(record) == KEYS["hypothesis"]
    assert (record["n_before"], record["n_after"]) == (302, 302)
    assert record["purpose"] == "research" and record["windows"] == [["2025-04-01", "2025-04-30"]]
    assert record["cost"]["model_id"] == "book" and record["cost"]["event_costs"] == "off"
    years = years_of([(date(2025, 4, 1), date(2025, 4, 30))])
    expected = min_edge(trades_per_year=5000.0, trade_sd_cents=1000, years=years, n_trials=304,
                        target_edge_cents=1000, trade_skew=0.0, trade_kurtosis=3.0)  # fmt: skip
    power = record["power"]
    assert power["n_used"] == 302 + 2 and power["min_edge_cents"] == expected.min_edge_cents
    assert power["min_edge_cents_alpha"] == expected.min_edge_cents_alpha
    assert (power["trade_skew"], power["trade_kurtosis"]) == (0.0, 3.0)
    assert record["suite"]["baseline_draws"] == 200 and record["secondary"]["prop_mc"] == 20000


def test_a_research_purpose_is_stored_as_research(genesis_env: Env) -> None:
    _register(genesis_env, genesis_env.hypothesis_text(purpose='"research"'))
    assert hypothesis_record(genesis_env.config, "H1")["purpose"] == "research"


def test_roots_may_add_to_the_spec_root(genesis_env: Env) -> None:
    _register(genesis_env, genesis_env.hypothesis_text(roots='["MNQ", "MES"]'))
    assert hypothesis_record(genesis_env.config, "H1")["roots"] == ["MNQ", "MES"]


# --------------------------------------------------------------------------------- refusals: the file ----


@pytest.mark.parametrize(
    "override",
    [
        {"schema": '"pfb.hypothesis/1"'},
        {"purpose": "<drop>"},
        {"extra_key": "1"},
        {"cost__extra": "1"},
        {"power__trade_skew": "<drop>"},
        {"secondary__prop_mc": "<drop>"},
        {"suite__neighbourhood": "<drop>"},
        {"purpose": '"exploration"'},
        {"budget": "0"},
        {"bar_minutes": "0"},
        {"roots": "[]"},
        {"roots": '["MNQ", "MNQ"]'},
        {"cost__event_costs": '"maybe"'},  # "on" registers (test_event_costs_wiring.py)
        {"cost__event_stance": '""'},
        {"cost__event_stance": '"   "'},
        {"power__trade_kurtosis": "0.9"},
        {"power__trades_per_year": "0.0"},
        {"power__sd_source": '"guess"'},
        {"suite__baseline_draws": "199"},
        {"suite__walk_forward_folds": "1"},
        {"secondary__prop_plans": '["*", "topstep/x_50k"]'},
        {"secondary__prop_plans": '["topstep"]'},
        {"windows": "[[window]]\nfirst = 2025-04-30\nlast = 2025-04-01\n"},
        {"windows": ""},
        {"windows": "[[window]]\nfirst = 2025-04-01\nlast = 2025-04-30\n" * 2},
    ],
    ids=[
        "schema-1", "missing-key", "extra-key", "extra-cost-key", "missing-skew", "missing-prop-mc",
        "missing-suite-key", "purpose", "budget-0", "bar-minutes-0", "no-roots", "repeated-root",
        "event-costs-unknown", "empty-event-stance", "blank-event-stance", "kurtosis-below-1", "zero-trades",
        "sd-source", "baseline-draws", "folds", "star-not-alone", "plan-name", "reversed-window", "no-window",
        "repeated-window",
    ],
)  # fmt: skip
def test_a_malformed_schema_2_file_is_refused(genesis_env: Env, override: dict[str, str]) -> None:
    with pytest.raises(HypothesisInvalid):
        _register(genesis_env, genesis_env.hypothesis_text(**override))
    assert kinds(genesis_env) == ["genesis"]


@pytest.mark.parametrize(
    "grid", ["a = [3, 2, 1]", "a = [1, 1, 2]", 'a = [1, "x"]', 'a = ["x", "x"]', "a = []"],
    ids=["decreasing", "repeat", "mixed", "repeated-string", "empty"],
)  # fmt: skip
def test_a_grid_that_is_not_strictly_increasing_or_repeats_is_refused(genesis_env: Env, grid: str) -> None:
    with pytest.raises(HypothesisInvalid):
        _register(genesis_env, genesis_env.hypothesis_text(grid=grid))


@pytest.mark.parametrize("model", ['"fixed:0"', '"stress:fixed:0:0"', '"fixed:0.0"'])
def test_a_cost_model_with_zero_slippage_is_refused(genesis_env: Env, model: str) -> None:
    with pytest.raises(ZeroSlippage):
        _register(genesis_env, genesis_env.hypothesis_text(cost__model=model))


def test_a_stress_of_a_zero_model_by_ticks_has_slippage_and_is_accepted(genesis_env: Env) -> None:
    _register(genesis_env, genesis_env.hypothesis_text(cost__model='"stress:fixed:0:1"'))
    assert hypothesis_record(genesis_env.config, "H1")["cost"]["model"] == "stress:fixed:0:1"


def test_a_hypothesis_file_not_committed_is_refused(genesis_env: Env) -> None:
    path = genesis_env.write(HYPOTHESIS_REL, genesis_env.hypothesis_text())
    with pytest.raises(InputNotCommitted):
        register_hypothesis(path, genesis_env.config, genesis_env.facts())


def test_a_strategy_not_committed_is_refused(genesis_env: Env) -> None:
    genesis_env.write(STRATEGY_REL, "x = 2  # not committed\n")
    path = genesis_env.write(HYPOTHESIS_REL, genesis_env.hypothesis_text())  # names the new sha256
    git_add_only(genesis_env, HYPOTHESIS_REL)
    with pytest.raises(InputNotCommitted):
        register_hypothesis(path, genesis_env.config, genesis_env.facts())


def git_add_only(env: Env, rel: str) -> None:
    from registry_env import git  # pyright: ignore[reportMissingImports]

    git(env.repo, "add", "--", rel)
    git(env.repo, "commit", "-q", "-m", "hypothesis only", "--", rel)


def test_a_strategy_hash_mismatch_is_refused(genesis_env: Env) -> None:
    text = genesis_env.hypothesis_text().replace(genesis_env.strategy_sha, "f" * 64)
    with pytest.raises(StrategyChanged):
        _register(genesis_env, text)


# --------------------------------------------------------------------------------- refusals: the SPEC ----


@pytest.mark.parametrize(
    ("override", "facts"),
    [
        ({"roots": '["MES"]'}, {}),
        ({"bar_minutes": "5"}, {}),
        ({"session": '"rth"'}, {}),
        ({}, {"spec_params": ("a", "b")}),
        ({"grid": "a = [1, 2]\nb = [1]"}, {}),
    ],
    ids=["roots-without-spec-root", "bar-minutes", "session", "grid-misses-a-param", "grid-extra-param"],
)
def test_a_hypothesis_unlike_its_spec_is_refused(
    genesis_env: Env, override: dict[str, str], facts: dict[str, object]
) -> None:
    grid = override.pop("grid", "a = [1, 2, 3]")
    with pytest.raises(SpecMismatch):
        _register(genesis_env, genesis_env.hypothesis_text(grid=grid, **override), **facts)
    assert kinds(genesis_env) == ["genesis"]


def test_a_budget_above_roots_windows_and_grid_points_is_refused(genesis_env: Env) -> None:
    # 1 root x 1 window x 3 points = 3
    _register(genesis_env, genesis_env.hypothesis_text(budget=3, hyp_id="H_ok"))
    genesis_env.write("hypotheses/h2.toml", genesis_env.hypothesis_text(budget=4, hyp_id="H_big"))
    genesis_env.commit()
    with pytest.raises(BudgetExceeded):
        register_hypothesis(genesis_env.repo / "hypotheses/h2.toml", genesis_env.config, genesis_env.facts())


# -------------------------------------------------------------------------------- refusals: the power ----


def test_a_target_below_the_minimum_detectable_edge_is_refused(genesis_env: Env) -> None:
    with pytest.raises(PowerInsufficient):
        _register(genesis_env, genesis_env.hypothesis_text(power__target_edge_cents="300"))
    assert kinds(genesis_env) == ["genesis"]


def test_an_undetectable_edge_is_refused(genesis_env: Env) -> None:
    with pytest.raises(PowerInsufficient):
        _register(genesis_env, genesis_env.hypothesis_text(power__trades_per_year="5.0"))


def test_a_negative_skew_raises_the_bar(genesis_env: Env) -> None:
    # Arrange: a target that passes with normal trades but not with skew -2 and kurtosis 9
    text = genesis_env.hypothesis_text(power__target_edge_cents="345", power__trade_skew="-2.0",
                                       power__trade_kurtosis="9.0")  # fmt: skip
    # Act / Assert
    with pytest.raises(PowerInsufficient):
        _register(genesis_env, text)


def test_sd_source_must_name_a_finished_ok_synthetic_or_engine_check_run(genesis_env: Env) -> None:
    with pytest.raises(HypothesisInvalid):
        _register(genesis_env, genesis_env.hypothesis_text(power__sd_source='"run:synthetic-nope"'))
    receipt = begin(genesis_env.request("synthetic"), config=genesis_env.config)
    realized = {"trades": 500, "days": 20, "trade_sd_cents": 900}  # binds
    finish(receipt, "ok", {"suite": {"power": realized}}, [], config=genesis_env.config)
    genesis_env.checkpoint()
    _register(genesis_env, genesis_env.hypothesis_text(power__sd_source=f'"run:{receipt.run_id}"'))
    assert hypothesis_record(genesis_env.config, "H1")["power"]["sd_source"] == f"run:{receipt.run_id}"


# ------------------------------------------------------------------------------ refusals: C5 and upstream
# ----


def test_a_strategy_on_the_allowlist_is_refused(genesis_env: Env) -> None:
    genesis_env.write(ENGINE_CHECKS_REL, genesis_env.engine_checks_text())
    with pytest.raises(EngineCheckStrategy):
        _register(genesis_env, genesis_env.hypothesis_text())


def test_a_strategy_run_as_an_engine_check_in_the_ledger_is_refused(genesis_env: Env) -> None:
    # Arrange: an engine_check of strategies/s.py, then the allowlist names another strategy
    genesis_env.write(ENGINE_CHECKS_REL, genesis_env.engine_checks_text())
    genesis_env.commit()
    begin(genesis_env.request("engine_check"), config=genesis_env.config)
    genesis_env.write(ENGINE_CHECKS_REL, genesis_env.engine_checks_text(strategy_sha256="9" * 64))
    genesis_env.checkpoint()
    # Act / Assert: the ledger half of C5, checked by replay
    with pytest.raises(EngineCheckStrategy):
        _register(genesis_env, genesis_env.hypothesis_text())


def test_registration_with_pe_out_of_sync_is_refused(genesis_env: Env) -> None:
    genesis_env.write_pe([*PE_BASE_LINES, pe_line("r002-T2", "T2", ["T2-a"], 302, 303)], [PE_AMENDMENT])
    with pytest.raises(SyncRequired):
        _register(genesis_env, genesis_env.hypothesis_text())


def test_a_registered_id_is_refused_a_second_time(trial_env: Env) -> None:
    with pytest.raises(HypothesisInvalid):
        register_hypothesis(trial_env.repo / HYPOTHESIS_REL, trial_env.config, trial_env.facts())


# --------------------------------------------------------------- refusals needing the gateway (runner) ----


def _runner_register(env: Env, engine: FakeTrialEngine, text: str, firms: Path | None = None) -> None:
    path = env.write(HYPOTHESIS_REL, text)
    env.commit()
    register(path, engine.deps(env), firms or env.repo / "firms")


def test_a_window_touching_a_closed_window_is_refused(genesis_env: Env) -> None:
    engine = FakeTrialEngine(data_error=ClosedWindowRefused("W1", "MNQ 2026-05-11 is sealed"))
    with pytest.raises(ClosedWindowTouched):
        _runner_register(genesis_env, engine, genesis_env.hypothesis_text())
    assert kinds(genesis_env) == ["genesis"]


@pytest.mark.parametrize(
    "engine",
    [FakeTrialEngine(data_error=NoBarsFound("none")), FakeTrialEngine(no_files=True)],
    ids=["no-bars", "empty"],
)
def test_a_window_without_a_data_file_is_refused(genesis_env: Env, engine: FakeTrialEngine) -> None:
    with pytest.raises(NoDataFile):
        _runner_register(genesis_env, engine, genesis_env.hypothesis_text())


def test_every_root_and_window_is_asked_of_the_gateway(genesis_env: Env) -> None:
    engine = FakeTrialEngine()
    windows = (
        "[[window]]\nfirst = 2025-04-01\nlast = 2025-04-30\n"
        "[[window]]\nfirst = 2025-05-01\nlast = 2025-05-30\n"
    )
    _runner_register(
        genesis_env, engine, genesis_env.hypothesis_text(roots='["MNQ", "MES"]', windows=windows)
    )
    assert len(engine.called("data_files")) == 4
    assert n_of(genesis_env) == 302


def test_a_declared_prop_plan_without_a_file_is_refused(genesis_env: Env) -> None:
    firms = genesis_env.repo.parent / "firms"
    (firms / "topstep").mkdir(parents=True)
    (firms / "topstep" / "combine_xfa_standard_50k.toml").write_text("x = 1\n")
    text = genesis_env.hypothesis_text(secondary__prop_plans='["topstep/combine_xfa_standard_50k"]')
    _runner_register(genesis_env, FakeTrialEngine(), text, firms)  # the file exists: accepted
    other = genesis_env.hypothesis_text(hyp_id="H2", secondary__prop_plans='["topstep/nope_50k"]')
    genesis_env.write("hypotheses/h2.toml", other)
    genesis_env.commit()
    with pytest.raises(HypothesisInvalid):
        register(genesis_env.repo / "hypotheses/h2.toml", FakeTrialEngine().deps(genesis_env), firms)
