"""Event_costs "on" through registration, the runner, begin, the records and replay (fake
engine and gateway; temporary ledgers in temporary git repositories; nothing real is read or written).

Rules: registration and each begin need the calendar's coverage; replay ties event_windows to
the hypothesis; begin and replay refuse "on" without a "+events" id and
event_windows "on", and the reverse; every run of an "on" hypothesis runs "on" (the trial,
its neighbours, each book member); LockedExitThroughSessionEnd after begin.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from book_support import BOOK_ID, BOOK_REL, LATE_WINDOW, book_text, member_toml, write_book  # pyright: ignore[reportMissingImports]
from registry_env import HYPOTHESIS_REL, STRATEGY_REL, Env  # pyright: ignore[reportMissingImports]
from trial_support import EVENTS_SHA, FakeTrialEngine, kinds, n_of  # pyright: ignore[reportMissingImports]

from pfb.costs.events import EVENT_MODEL_SUFFIX, EventCoverageError
from pfb.engine.run import LockedExitThroughSessionEnd
from pfb.registry.api import begin, hypothesis_record, read_state, register_hypothesis, status
from pfb.registry.chain import read_ledger
from pfb.registry.errors import (
    BadRequest,
    ChainBroken,
    HypothesisInvalid,
    MemberAlreadyCounted,
    UndeclaredConfiguration,
)
from pfb.registry.inputs import EVENT_COSTS_ENGINE_WIRED
from pfb.registry.records import EVENTS_MODEL_SUFFIX, config_id, replay
from pfb.runner.api import RunOutcome, RunRefused, run
from pfb.runner.neighbours import run_with_neighbours
from pfb.runner.register import dry_run, register

ON = {"cost__event_costs": '"on"'}
WINDOW_2 = (
    "[[window]]\nfirst = 2025-04-01\nlast = 2025-04-30\n[[window]]\nfirst = 2025-06-02\nlast = 2025-06-30\n"
)
LOCKED = "LockedExitThroughSessionEnd"


def _register_on(env: Env, engine: FakeTrialEngine, **override: str) -> None:
    """Register H1 with event_costs "on" through the runner (cost ids from the fake: book+events)."""
    env.write(HYPOTHESIS_REL, env.hypothesis_text(**{**ON, **override}))
    env.commit()
    register(env.repo / HYPOTHESIS_REL, engine.deps(env))
    env.checkpoint()


def _trial(env: Env, engine: FakeTrialEngine, **kwargs: Any) -> RunOutcome:
    return run(env.repo / STRATEGY_REL, kind="trial", hypothesis_path=Path(HYPOTHESIS_REL),
               deps=engine.deps(env), runs_dir=env.repo / "runs", **kwargs)  # fmt: skip


def _records(env: Env) -> list[dict[str, Any]]:
    return [dict(line.record) for line in read_ledger(env.config.registry_path)]


# ------------------------------------------------------------------ the switch and the suffix ----
def test_the_switch_is_on_and_the_registry_suffix_is_the_cost_packages() -> None:
    assert EVENT_COSTS_ENGINE_WIRED is True
    assert EVENTS_MODEL_SUFFIX == EVENT_MODEL_SUFFIX == "+events"


# ------------------------------------------------------------------ registration (task 1) ----
def test_an_on_hypothesis_registers_with_the_events_id_after_every_window_is_covered(
    genesis_env: Env,
) -> None:
    # Arrange
    engine = FakeTrialEngine()
    # Act
    _register_on(genesis_env, engine, windows=WINDOW_2, budget="3")
    # Assert
    record = hypothesis_record(genesis_env.config, "H1")
    assert record["cost"]["event_costs"] == "on" and record["cost"]["model"] == "book"
    assert record["cost"]["model_id"] == "book+events" and record["cost"]["config_sha256"] == EVENTS_SHA
    assert engine.called("event_coverage") == [
        (date(2025, 4, 1), date(2025, 4, 30)),
        (date(2025, 6, 2), date(2025, 6, 30)),
    ]
    assert n_of(genesis_env) == 302


def test_an_off_hypothesis_never_asks_the_calendar(genesis_env: Env) -> None:
    engine = FakeTrialEngine()
    genesis_env.write(HYPOTHESIS_REL, genesis_env.hypothesis_text())
    genesis_env.commit()
    register(genesis_env.repo / HYPOTHESIS_REL, engine.deps(genesis_env))
    assert engine.called("event_coverage") == []
    assert hypothesis_record(genesis_env.config, "H1")["cost"]["model_id"] == "book"


def test_an_on_window_the_calendar_does_not_cover_is_refused_at_registration(genesis_env: Env) -> None:
    # Arrange
    engine = FakeTrialEngine(coverage_error=EventCoverageError("window outside the calendar"))
    genesis_env.write(HYPOTHESIS_REL, genesis_env.hypothesis_text(**ON))
    genesis_env.commit()
    before = genesis_env.config.registry_path.read_bytes()
    # Act / Assert
    with pytest.raises(RunRefused, match="EventCoverageError"):
        register(genesis_env.repo / HYPOTHESIS_REL, engine.deps(genesis_env))
    assert genesis_env.config.registry_path.read_bytes() == before
    found = dry_run(genesis_env.repo / HYPOTHESIS_REL, engine.deps(genesis_env))
    assert any("EventCoverageError" in str(e) for e in found)


def test_the_registry_refuses_an_id_that_does_not_carry_the_declared_event_costs(genesis_env: Env) -> None:
    # Arrange: "on" with the plain id, and "off" with the events id
    path = genesis_env.write(HYPOTHESIS_REL, genesis_env.hypothesis_text(**ON))
    genesis_env.commit()
    # Act / Assert
    with pytest.raises(BadRequest, match=r"\+events"):
        register_hypothesis(path, genesis_env.config, genesis_env.facts())
    path = genesis_env.write(HYPOTHESIS_REL, genesis_env.hypothesis_text())
    genesis_env.commit()
    with pytest.raises(BadRequest, match=r"\+events"):
        register_hypothesis(path, genesis_env.config, genesis_env.facts(cost_model_id="book+events"))
    assert kinds(genesis_env) == ["genesis"]


def test_on_over_a_fixed_model_is_still_refused_by_the_parser(genesis_env: Env) -> None:
    path = genesis_env.write(HYPOTHESIS_REL, genesis_env.hypothesis_text(cost__model='"fixed:1"', **ON))
    genesis_env.commit()
    with pytest.raises(HypothesisInvalid, match="book-based"):
        register_hypothesis(path, genesis_env.config, genesis_env.facts(cost_model_id="fixed:1+events"))


def test_replay_refuses_a_hypothesis_record_whose_id_contradicts_its_event_costs(genesis_env: Env) -> None:
    # Arrange
    _register_on(genesis_env, FakeTrialEngine())
    records = _records(genesis_env)
    # Act / Assert
    forged = {**records[-1], "cost": {**records[-1]["cost"], "model_id": "book"}}
    with pytest.raises(ChainBroken, match="event_costs"):
        replay([*records[:-1], forged])
    forged = {**records[-1], "cost": {**records[-1]["cost"], "event_costs": "off"}}
    with pytest.raises(ChainBroken, match="event_costs"):
        replay([*records[:-1], forged])


# ------------------------------------------------------------------ a trial "on" (tasks 1 to 3) ----
def test_an_on_trial_is_covered_begun_run_permitted_recorded_and_replayed_on(genesis_env: Env) -> None:
    # Arrange
    engine = FakeTrialEngine()
    _register_on(genesis_env, engine)
    # Act
    outcome = _trial(genesis_env, engine)
    # Assert
    begun = outcome.begin_record
    assert begun["event_windows"] == "on" and begun["cost_model"] == "book"
    assert begun["cost_model_id"] == "book+events" and begun["cost_config_sha256"] == EVENTS_SHA
    assert begun["configurations"] == [config_id(begun)] and begun["n_after"] == 303
    kwargs = engine.called("run_strategy")[0]
    assert kwargs["event_costs"] == "on" and kwargs["cost_model"] == "book"  # the base spec, never +events
    assert kwargs["permit"].event_costs == "on" and kwargs["permit"].cost_model == "book"
    assert engine.called("event_coverage")[-1] == (date(2025, 4, 1), date(2025, 4, 30))
    assert outcome.result_record["status"] == "ok"
    assert read_state(genesis_env.config).n == 303  # the whole ledger replays


def test_an_off_trial_passes_off_everywhere_and_never_asks_the_calendar(trial_env: Env) -> None:
    engine = FakeTrialEngine()
    outcome = _trial(trial_env, engine)
    kwargs = engine.called("run_strategy")[0]
    assert outcome.begin_record["event_windows"] == "off" and outcome.begin_record["cost_model_id"] == "book"
    assert kwargs["event_costs"] == "off" and kwargs["permit"].event_costs == "off"
    assert engine.called("event_coverage") == []


def test_an_on_trial_outside_the_calendar_is_refused_before_begin(genesis_env: Env) -> None:
    # Arrange
    engine = FakeTrialEngine()
    _register_on(genesis_env, engine)
    engine.coverage_error = EventCoverageError("window outside the calendar")
    before = genesis_env.config.registry_path.read_bytes()
    # Act / Assert
    with pytest.raises(RunRefused, match="EventCoverageError"):
        _trial(genesis_env, engine)
    assert genesis_env.config.registry_path.read_bytes() == before and engine.called("run_strategy") == []


def _on_env(genesis_env: Env) -> Env:
    """H1 "on" registered directly (facts with the events id), ledger at HEAD."""
    path = genesis_env.write(HYPOTHESIS_REL, genesis_env.hypothesis_text(**ON))
    genesis_env.commit()
    facts = genesis_env.facts(cost_model_id="book+events", cost_config_sha256=EVENTS_SHA)
    register_hypothesis(path, genesis_env.config, facts)
    genesis_env.checkpoint()
    return genesis_env


@pytest.mark.parametrize(
    "override",
    [
        {"event_windows": "off"},  # "on" hypothesis, "off" run under the events id
        {"cost_model_id": "book"},  # "on" run under the plain id (and the hypothesis's id differs)
        {"event_windows": "off", "cost_model_id": "book"},  # an "off" run of an "on" hypothesis
    ],
    ids=["windows-off", "plain-id", "both-off"],
)
def test_begin_refuses_an_on_hypothesis_run_without_on_and_the_events_id(
    genesis_env: Env, override: dict[str, str]
) -> None:
    env = _on_env(genesis_env)
    good = {"event_windows": "on", "cost_model_id": "book+events", "cost_config_sha256": EVENTS_SHA}
    with pytest.raises(UndeclaredConfiguration):
        env.begin_trial(**{**good, **override})
    assert kinds(env) == ["genesis", "hypothesis"]


@pytest.mark.parametrize(
    "override",
    [
        {"event_windows": "on"},
        {"event_windows": "on", "cost_model_id": "book+events"},
        {"cost_model_id": "book+events"},
    ],
    ids=["windows-on", "both-on", "events-id"],
)
def test_begin_refuses_an_off_hypothesis_run_that_says_on(trial_env: Env, override: dict[str, str]) -> None:
    with pytest.raises(UndeclaredConfiguration):
        trial_env.begin_trial(**override)
    assert kinds(trial_env) == ["genesis", "hypothesis"]


@pytest.mark.parametrize("kind", ["synthetic", "engine_check"])
def test_only_a_trial_begins_on(genesis_env: Env, kind: str) -> None:
    with pytest.raises(BadRequest, match="event_windows"):
        begin(genesis_env.request(kind, event_windows="on", cost_model_id="book+events"),
              config=genesis_env.config)  # fmt: skip


def _forge(records: list[dict[str, Any]], index: int, **change: Any) -> list[dict[str, Any]]:
    forged = {**records[index], **change}
    forged["configurations"] = [config_id(forged)]
    return [*records[:index], forged, *records[index + 1 :]]


def test_replay_ties_a_trial_begin_to_its_hypothesis_event_costs(genesis_env: Env) -> None:
    # Arrange: an "on" trial and its result, then each one-field forgery of the begin
    engine = FakeTrialEngine()
    _register_on(genesis_env, engine)
    _trial(genesis_env, engine)
    records = _records(genesis_env)
    begin_at = next(i for i, r in enumerate(records) if r["kind"] == "trial")
    assert replay(records).n == 303
    # Act / Assert
    with pytest.raises(UndeclaredConfiguration):
        replay(_forge(records, begin_at, event_windows="off", cost_model_id="book"))  # bound id differs too
    with pytest.raises(ChainBroken, match="event_windows"):
        replay(_forge(records, begin_at, event_windows="off"))  # off under the events id
    with pytest.raises(ChainBroken, match="event_windows"):
        replay(_forge(records, begin_at, event_windows="maybe"))


def test_replay_refuses_an_on_begin_of_an_off_hypothesis_and_an_on_synthetic_run(trial_env: Env) -> None:
    # Arrange
    _trial(trial_env, FakeTrialEngine())
    run(trial_env.repo / STRATEGY_REL, kind="synthetic", deps=FakeTrialEngine().deps(trial_env),
        runs_dir=trial_env.repo / "runs")  # fmt: skip
    records = _records(trial_env)
    trial_at = next(i for i, r in enumerate(records) if r["kind"] == "trial")
    synth_at = next(i for i, r in enumerate(records) if r["kind"] == "synthetic")
    # Act / Assert
    with pytest.raises(UndeclaredConfiguration):
        replay(_forge(records, trial_at, event_windows="on", cost_model_id="book+events"))
    with pytest.raises(ChainBroken, match="trial"):
        replay(_forge(records, synth_at, event_windows="on", cost_model_id="book+events"))


# ------------------------------------------------------------------ every run of "on" runs "on" (task 4) ----
def test_every_neighbourhood_run_of_an_on_hypothesis_runs_on(genesis_env: Env) -> None:
    # Arrange
    engine = FakeTrialEngine()
    _register_on(genesis_env, engine, budget="3", suite__neighbourhood="true")
    # Act
    found = run_with_neighbours(genesis_env.repo / STRATEGY_REL, kind="trial", params={"a": 2},
                                hypothesis_path=Path(HYPOTHESIS_REL), deps=engine.deps(genesis_env),
                                runs_dir=genesis_env.repo / "runs")  # fmt: skip
    # Assert
    outcomes = (*found.neighbours, found.centre)
    assert len(outcomes) == 3 and n_of(genesis_env) == 305
    assert all(o.begin_record["event_windows"] == "on" for o in outcomes)
    assert all(o.begin_record["cost_model_id"] == "book+events" for o in outcomes)
    calls = engine.called("run_strategy")
    assert len(calls) == 3 and all(
        k["event_costs"] == "on" and k["permit"].event_costs == "on" for k in calls
    )


def _register_book_on(env: Env, engine: FakeTrialEngine, text: str | None = None) -> None:
    body = text if text is not None else book_text(env.strategy_sha, cost__event_costs='"on"')
    register(write_book(env, body), engine.deps(env))
    env.checkpoint()


def _member(env: Env, engine: FakeTrialEngine, a: int, **kwargs: Any) -> RunOutcome:
    return run(env.repo / STRATEGY_REL, kind="trial", hypothesis_path=Path(BOOK_REL), deps=engine.deps(env),
               runs_dir=env.repo / "runs", params={"a": a}, **kwargs)  # fmt: skip


def test_each_member_of_an_on_book_runs_on(genesis_env: Env) -> None:
    # Arrange
    engine = FakeTrialEngine()
    _register_book_on(genesis_env, engine)
    record = hypothesis_record(genesis_env.config, BOOK_ID)
    assert record["cost"]["event_costs"] == "on" and record["cost"]["model_id"] == "book+events"
    # Act
    outcomes = [_member(genesis_env, engine, a) for a in (1, 2)]
    # Assert
    assert all(o.begin_record["event_windows"] == "on" for o in outcomes)
    assert all(
        k["event_costs"] == "on" and k["permit"].event_costs == "on" for k in engine.called("run_strategy")
    )
    assert read_state(genesis_env.config).n == 304


def test_an_on_book_checks_each_member_window_at_registration(genesis_env: Env) -> None:
    # Arrange: two members on different windows
    sha = genesis_env.strategy_sha
    members = [member_toml(sha, a=1), member_toml(sha, a=2, window=LATE_WINDOW)]
    text = book_text(sha, members=members, cost__event_costs='"on"', power__trades_per_year="6000.0",
                     power__trade_sd_cents="2000")  # fmt: skip
    engine = FakeTrialEngine(coverage_error=EventCoverageError("outside"))
    # Act / Assert
    with pytest.raises(RunRefused, match="EventCoverageError"):
        register(write_book(genesis_env, text), engine.deps(genesis_env))
    assert kinds(genesis_env) == ["genesis"]
    assert engine.called("event_coverage") == [(date(2025, 4, 1), date(2025, 4, 30))]  # the first refuses


def test_a_book_registered_with_a_plain_id_for_on_is_refused(genesis_env: Env) -> None:
    from pfb.registry.bookfile import BookHypothesis
    from pfb.registry.bookreg import register_book
    from pfb.registry.hypfile import load_hypothesis
    from pfb.runner.register import member_facts

    path = write_book(genesis_env, book_text(genesis_env.strategy_sha, cost__event_costs='"on"'))
    book = load_hypothesis(path)
    assert isinstance(book, BookHypothesis)
    facts = [replace(f, cost_model_id="book") for f in member_facts(book, FakeTrialEngine().deps(genesis_env),
                                                                       genesis_env.repo)]  # fmt: skip
    with pytest.raises(BadRequest, match=r"\+events"):
        register_book(path, genesis_env.config, facts)


# ------------------------------------------------------------------ LockedExitThroughSessionEnd (task 5) ----
def _locked() -> LockedExitThroughSessionEnd:
    return LockedExitThroughSessionEnd(
        "locked_exit_through_session_end: an exit still waited at the session end"
    )


def _failed_and_closed(env: Env, run_id: str) -> dict[str, Any]:
    result = next(r for r in _records(env) if r["kind"] == "result" and r["run_id"] == run_id)
    assert result["status"] == "failed" and result["error"] == LOCKED and result["metrics"] is None
    assert status(env.config).open_runs == ()
    return result


def test_a_trial_that_raises_locked_exit_is_counted_failed_and_not_left_open(trial_env: Env) -> None:
    # Arrange
    engine = FakeTrialEngine(run_error=_locked())
    # Act
    with pytest.raises(LockedExitThroughSessionEnd):
        _trial(trial_env, engine)
    # Assert
    begun = next(r for r in _records(trial_env) if r["kind"] == "trial")
    assert begun["n_after"] == begun["n_before"] + 1 == 303
    _failed_and_closed(trial_env, begun["run_id"])
    assert kinds(trial_env) == ["genesis", "hypothesis", "trial", "result"]


def test_a_book_member_that_raises_locked_exit_is_counted_and_cannot_rerun_changed(genesis_env: Env) -> None:
    # Arrange
    engine = FakeTrialEngine()
    _register_book_on(genesis_env, engine, book_text(genesis_env.strategy_sha))
    engine.run_error = _locked()
    # Act
    with pytest.raises(LockedExitThroughSessionEnd):
        _member(genesis_env, engine, 1)
    # Assert: counted, failed, closed. The book's rule (records.py require_member_uncounted, unchanged): the
    # member's config_id is recorded at its begin; the same configuration may rerun at N + 0 (and fails the
    # same way on the same code and data), a changed one (a code, engine or data fix) is MemberAlreadyCounted,
    # so the book cannot complete
    begun = next(r for r in _records(genesis_env) if r["kind"] == "trial")
    assert begun["n_after"] == begun["n_before"] + 1
    _failed_and_closed(genesis_env, begun["run_id"])
    engine.run_error = None
    fixed = replace(engine.deps(genesis_env), engine_version=lambda: "pfb-engine/fixed")
    with pytest.raises(MemberAlreadyCounted):
        run(genesis_env.repo / STRATEGY_REL, kind="trial", hypothesis_path=Path(BOOK_REL), deps=fixed,
            runs_dir=genesis_env.repo / "runs", params={"a": 1})  # fmt: skip
    assert kinds(genesis_env) == ["genesis", "hypothesis", "trial", "result"]


def test_a_neighbour_that_raises_locked_exit_is_counted_failed_and_stops_the_set(genesis_env: Env) -> None:
    # Arrange: centre a = 2; the first neighbour (a = 1) runs first and raises
    genesis_env.write(HYPOTHESIS_REL, genesis_env.hypothesis_text(budget=3, suite__neighbourhood="true"))
    genesis_env.commit()
    register_hypothesis(genesis_env.repo / HYPOTHESIS_REL, genesis_env.config, genesis_env.facts())
    genesis_env.checkpoint()
    engine = FakeTrialEngine(run_error=_locked())
    # Act
    with pytest.raises(LockedExitThroughSessionEnd):
        run_with_neighbours(genesis_env.repo / STRATEGY_REL, kind="trial", params={"a": 2},
                            hypothesis_path=Path(HYPOTHESIS_REL), deps=engine.deps(genesis_env),
                            runs_dir=genesis_env.repo / "runs")  # fmt: skip
    # Assert: one begin (the neighbour), counted, failed and closed; the rest of the set was never begun
    begins = [r for r in _records(genesis_env) if r["kind"] == "trial"]
    assert len(begins) == 1 and begins[0]["params"] == {"a": 1} and n_of(genesis_env) == 303
    _failed_and_closed(genesis_env, begins[0]["run_id"])
