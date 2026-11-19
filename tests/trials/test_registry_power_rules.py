"""Registry power rules: power years of ONE window, sd_source bound to its run, and
a hypothesis grid value outside the SPEC parameter's own grid refused at registration (TRIALS.md s2, s5)."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from registry_env import HYPOTHESIS_REL, Env  # pyright: ignore[reportMissingImports]
from trial_support import FakeTrialEngine, kinds  # pyright: ignore[reportMissingImports]

from pfb.analysis.power import min_edge, years_of
from pfb.registry.api import begin, finish, hypothesis_record, register_hypothesis
from pfb.registry.chain import read_ledger, sha256_hex
from pfb.registry.errors import HypothesisInvalid, PowerInsufficient, SpecMismatch
from pfb.registry.powercheck import window_years
from pfb.registry.records import replay
from pfb.runner.register import register

APRIL = "[[window]]\nfirst = 2025-04-01\nlast = 2025-04-30\n"  # 30 days: min edge 332 at N 304
MAY = "[[window]]\nfirst = 2025-05-01\nlast = 2025-05-30\n"  # 30 days
MAY_JUNE = "[[window]]\nfirst = 2025-05-01\nlast = 2025-06-30\n"  # 61 days: min edge 209 at N 304
ENGINE_CHECKS_REL = "config/engine_checks.toml"
OTHER_STRATEGY_REL = "strategies/other.py"


def _register(env: Env, text: str, **facts: object) -> None:
    path = env.write(HYPOTHESIS_REL, text)
    env.commit()
    register_hypothesis(path, env.config, env.facts(**facts))


def _edge(first: date, last: date, n_used: int) -> int | None:
    return min_edge(trades_per_year=5000.0, trade_sd_cents=1000, years=years_of([(first, last)]),
                    n_trials=n_used, trade_skew=0.0, trade_kurtosis=3.0).min_edge_cents  # fmt: skip


# ------------------------------------------------------------------------ one window's years ----


def test_window_years_refuses_more_than_one_window() -> None:
    assert window_years([["2025-04-01", "2025-04-30"]]) == pytest.approx(30 / 365.25)
    with pytest.raises(ValueError, match="one window"):
        window_years([["2025-04-01", "2025-04-30"], ["2025-05-01", "2025-05-30"]])


def test_registration_refuses_windows_that_pass_only_when_their_years_are_summed(genesis_env: Env) -> None:
    # Arrange: each 30-day window needs 342 cents, the two summed would need 217; the target is 300
    text = genesis_env.hypothesis_text(windows=APRIL + MAY, power__target_edge_cents="300")
    # Act / Assert
    with pytest.raises(PowerInsufficient, match="2025-0"):
        _register(genesis_env, text)
    assert kinds(genesis_env) == ["genesis"]


def test_registration_refuses_when_one_declared_window_fails_on_its_own(genesis_env: Env) -> None:
    # Arrange: the 61-day window passes a target of 300 (215), the 30-day one does not (342)
    text = genesis_env.hypothesis_text(windows=MAY_JUNE + APRIL, power__target_edge_cents="300")
    # Act / Assert
    with pytest.raises(PowerInsufficient, match="2025-04-01"):
        _register(genesis_env, text)


def test_the_hypothesis_record_stores_the_shortest_windows_figures(genesis_env: Env) -> None:
    # Arrange / Act: a 61-day window first, the 30-day one second
    _register(genesis_env, genesis_env.hypothesis_text(windows=MAY_JUNE + APRIL))
    power = hypothesis_record(genesis_env.config, "H1")["power"]
    # Assert: the 30-day window's figures (the largest minimum edge), never the sum of both
    assert power["years"] == pytest.approx(30 / 365.25)
    assert power["min_edge_cents"] == _edge(date(2025, 4, 1), date(2025, 4, 30), 304) == 332


def test_begin_and_replay_use_the_trials_own_window(genesis_env: Env) -> None:
    # Arrange
    _register(genesis_env, genesis_env.hypothesis_text(windows=APRIL + MAY_JUNE))
    # Act: a trial on the 61-day window (begin replays the record before it is written)
    receipt = genesis_env.begin_trial(first="2025-05-01", last="2025-06-30")
    # Assert: that window's own minimum (209), not the 30-day one (332) nor the sum of both (165)
    assert receipt.record["power"]["min_edge_cents"] == _edge(date(2025, 5, 1), date(2025, 6, 30), 304) == 209
    records = [line.record for line in read_ledger(genesis_env.config.registry_path, None)]
    forged = {**records[-1], "power": {**records[-1]["power"], "min_edge_cents": 165}}
    with pytest.raises(Exception, match="power figures differ"):
        replay([*records[:-1], forged])


# ----------------------------------------------------------------------- sd_source "run:<id>" ----


def _power_metrics(*, sd: int = 900, trades: int = 500, days: int = 20) -> dict[str, Any]:
    """A result's metrics with the suite's realized power figures (pfb.analysis.inrun.realized_power)."""
    power = {"realized_edge_cents": 50, "trades": trades, "days": days, "years": days / 252,
             "trade_sd_cents": sd, "trade_skew": 0.0, "trade_kurtosis": 3.0}  # fmt: skip
    return {"net_cents": 100, "suite": {"power": power}}


def _synthetic_source(env: Env, metrics: dict[str, Any], **request: Any) -> str:
    receipt = begin(env.request("synthetic", **request), config=env.config)
    finish(receipt, "ok", metrics, [], config=env.config)
    env.checkpoint()
    return receipt.run_id


def _register_from(env: Env, run_id: str) -> None:
    _register(env, env.hypothesis_text(power__sd_source=f'"run:{run_id}"'))


def test_a_synthetic_source_with_realized_power_figures_that_bind_is_accepted(genesis_env: Env) -> None:
    # Arrange: realized sd 900 <= stated 1000; realized 500 trades over 20 days = 6300 a year >= 5000
    run_id = _synthetic_source(genesis_env, _power_metrics())
    # Act
    _register_from(genesis_env, run_id)
    # Assert
    assert hypothesis_record(genesis_env.config, "H1")["power"]["sd_source"] == f"run:{run_id}"


@pytest.mark.parametrize(
    "metrics",
    [{}, {"suite": None}, {"suite": {"power": None}}, _power_metrics(trades=1), _power_metrics(sd=0)],
    ids=["no-metrics", "no-suite", "no-power", "one-trade", "zero-sd"],
)
def test_a_source_run_without_realized_power_figures_is_refused(
    genesis_env: Env, metrics: dict[str, Any]
) -> None:
    run_id = _synthetic_source(genesis_env, metrics)
    with pytest.raises(HypothesisInvalid, match="realized power"):
        _register_from(genesis_env, run_id)
    assert kinds(genesis_env) == ["genesis", "synthetic", "result"]


def test_a_source_run_whose_realized_sd_is_above_the_stated_one_is_refused(genesis_env: Env) -> None:
    run_id = _synthetic_source(genesis_env, _power_metrics(sd=1001))
    with pytest.raises(HypothesisInvalid, match="trade_sd_cents"):
        _register_from(genesis_env, run_id)


def test_a_source_run_whose_realized_trades_per_year_are_below_the_stated_ones_is_refused(
    genesis_env: Env,
) -> None:
    # 396 trades over 20 days = 4989.6 a year < 5000; 397 would give 5002.2
    run_id = _synthetic_source(genesis_env, _power_metrics(trades=396))
    with pytest.raises(HypothesisInvalid, match="trades_per_year"):
        _register_from(genesis_env, run_id)


def test_a_synthetic_source_of_another_strategy_is_refused(genesis_env: Env) -> None:
    other = genesis_env.write(OTHER_STRATEGY_REL, "y = 1\n")
    genesis_env.commit()
    run_id = _synthetic_source(
        genesis_env, _power_metrics(), strategy_path=other, strategy_sha256=sha256_hex(other.read_bytes())
    )
    with pytest.raises(HypothesisInvalid, match="strategy"):
        _register_from(genesis_env, run_id)


def test_an_engine_check_source_of_another_strategy_binds_by_its_figures(genesis_env: Env) -> None:
    # Arrange: rule C5 means an engine_check never runs the hypothesis's own strategy
    other = genesis_env.write(OTHER_STRATEGY_REL, "y = 1\n")
    sha = sha256_hex(other.read_bytes())
    text = genesis_env.engine_checks_text(strategy=OTHER_STRATEGY_REL, strategy_sha256=sha)
    genesis_env.write(ENGINE_CHECKS_REL, text)
    genesis_env.commit()

    def source(metrics: dict[str, Any]) -> str:
        receipt = begin(genesis_env.request("engine_check", strategy_path=other, strategy_sha256=sha),
                        config=genesis_env.config)  # fmt: skip
        finish(receipt, "ok", metrics, [], config=genesis_env.config)
        genesis_env.checkpoint()
        return receipt.run_id

    too_wide = source(_power_metrics(sd=2000))
    fine = source(_power_metrics())
    # Act / Assert
    with pytest.raises(HypothesisInvalid, match="trade_sd_cents"):
        _register_from(genesis_env, too_wide)
    _register_from(genesis_env, fine)
    assert hypothesis_record(genesis_env.config, "H1")["power"]["sd_source"] == f"run:{fine}"


def test_replay_refuses_a_source_run_whose_figures_no_longer_bind(genesis_env: Env) -> None:
    # Arrange: a valid ledger whose hypothesis names a synthetic run
    run_id = _synthetic_source(genesis_env, _power_metrics())
    _register_from(genesis_env, run_id)
    records = [line.record for line in read_ledger(genesis_env.config.registry_path, None)]
    index = next(i for i, r in enumerate(records) if r["kind"] == "result" and r["run_id"] == run_id)
    altered = {**records[index], "metrics": _power_metrics(sd=5000)}
    # Act / Assert: replay alone (no hash chain) decides it
    replay(records)
    with pytest.raises(HypothesisInvalid, match="trade_sd_cents"):
        replay([*records[:index], altered, *records[index + 1 :]])


# ------------------------------------------------------------- grid values in the SPEC grid ----


@pytest.mark.parametrize(
    "grid", ["a = [1, 2, 5]", "a = [0]", "a = [1.0, 2.0]", 'a = ["1"]'],
    ids=["one-value-outside", "only-outside", "float-for-int", "string-for-int"],
)  # fmt: skip
def test_a_grid_value_outside_the_spec_parameters_grid_is_refused(genesis_env: Env, grid: str) -> None:
    with pytest.raises(SpecMismatch, match="outside"):
        _register(genesis_env, genesis_env.hypothesis_text(grid=grid, budget=1))
    assert kinds(genesis_env) == ["genesis"]


def test_a_grid_that_is_a_subset_of_the_spec_grid_is_accepted(genesis_env: Env) -> None:
    _register(genesis_env, genesis_env.hypothesis_text(grid="a = [1, 3]"))
    assert hypothesis_record(genesis_env.config, "H1")["grid"] == {"a": [1, 3]}


def test_the_runner_passes_the_spec_grids_so_registration_refuses_an_outside_value(genesis_env: Env) -> None:
    # Arrange: the fake engine's SPEC grid of a is (1, 2, 3)
    path = genesis_env.write(HYPOTHESIS_REL, genesis_env.hypothesis_text(grid="a = [1, 2, 4]"))
    genesis_env.commit()
    # Act / Assert
    with pytest.raises(SpecMismatch, match="outside"):
        register(path, FakeTrialEngine().deps(genesis_env), genesis_env.repo / "firms")
    assert kinds(genesis_env) == ["genesis"]
