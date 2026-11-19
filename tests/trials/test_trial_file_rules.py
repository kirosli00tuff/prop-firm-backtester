"""Hypothesis-file rules for trials. Temporary ledgers only.

- purpose "test" with a budget other than 1 is refused when the file is parsed (PurposeTestBudget).
- two declared windows sharing at least one day are refused when the file is parsed (WindowsOverlap);
windows apply to every root, so one window over two roots is fine and adjacent windows are allowed.
- the step-1 look-ahead seed is the hypothesis's `lookahead_seed` (int >= 0, default 0). A trial runs at it;
an explicit seed that differs is refused (SeedMismatch) before step 1 and before any begin, on every path
(runner, neighbourhood runner, app, CLI, MCP). Other kinds keep the switch (default 0).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from registry_env import HYPOTHESIS_REL, STRATEGY_REL, Env  # pyright: ignore[reportMissingImports]
from trial_support import FakeTrialEngine, kinds  # pyright: ignore[reportMissingImports]
from typer.testing import CliRunner

import pfb.cli.registry_cmd as registry_cmd
import pfb.cli.run_cmd as run_cmd
import pfb.runner.api as runner_api
from pfb import app as service
from pfb.cli.main import app as cli_app
from pfb.registry.api import register_hypothesis
from pfb.registry.chain import sha256_hex
from pfb.registry.errors import HypothesisInvalid
from pfb.registry.hypfile import (
    DEFAULT_LOOKAHEAD_SEED,
    PurposeTestBudget,
    WindowsOverlap,
    load_hypothesis,
    parse_hypothesis,
)
from pfb.runner.api import SeedMismatch, run
from pfb.runner.neighbours import run_with_neighbours
from pfb.runner.register import register as runner_register

REPO = Path(__file__).resolve().parents[2]
PLUMBING_FILE = REPO / "hypotheses" / "h_example_plumbing.toml"
APRIL = "[[window]]\nfirst = 2025-04-01\nlast = 2025-04-30\n"
MAY = "[[window]]\nfirst = 2025-05-01\nlast = 2025-05-30\n"
APRIL_TO_MAY = "[[window]]\nfirst = 2025-04-30\nlast = 2025-05-30\n"  # shares 2025-04-30 with APRIL
MID_APRIL = "[[window]]\nfirst = 2025-04-10\nlast = 2025-04-12\n"  # inside APRIL
SPRING = "[[window]]\nfirst = 2025-03-01\nlast = 2025-05-30\n"  # holds APRIL and MAY
cli = CliRunner()


@dataclass
class SeedEngine(FakeTrialEngine):
    """Records the seed of each step-1 check and each engine call."""

    seeds: list[tuple[str, int]] = field(default_factory=list)

    def lookahead(self, strategy: Any, params: Mapping[str, Any], seed: int) -> tuple[str, ...]:
        self.seeds.append(("lookahead", seed))
        return super().lookahead(strategy, params, seed)

    def run_strategy(self, path: Path, **kwargs: Any) -> Any:
        self.seeds.append(("run_strategy", kwargs["seed"]))
        return super().run_strategy(path, **kwargs)


def _parse(env: Env, **override: Any) -> Any:
    return parse_hypothesis(env.hypothesis_text(**override).encode(), "h.toml")


def _register(env: Env, text: str) -> None:
    env.write(HYPOTHESIS_REL, text)
    env.commit()
    register_hypothesis(env.repo / HYPOTHESIS_REL, env.config, env.facts())
    env.checkpoint()


def _seeded(text: str, seed: str) -> str:
    """A hypothesis text with `lookahead_seed = <seed>` among the top-level keys."""
    head, sep, rest = text.partition("[[window]]")
    reason = 'lookahead_seed_reason = "a test of the seed rules"\n'
    return f"{head}lookahead_seed = {seed}\n{reason}{sep}{rest}"


def _trial(env: Env, engine: FakeTrialEngine, **kwargs: Any) -> Any:
    return run(env.repo / STRATEGY_REL, kind="trial", hypothesis_path=Path(HYPOTHESIS_REL),
               deps=engine.deps(env), runs_dir=env.repo / "runs", params={"a": 2}, **kwargs)  # fmt: skip


# ------------------------------------------------------------------------------- test budget is 1 ----


def test_purpose_test_with_budget_two_is_refused_at_parse(env: Env) -> None:
    with pytest.raises(PurposeTestBudget, match="budget"):
        _parse(env, purpose='"test"', budget=2)


def test_purpose_test_budget_refusal_is_a_hypothesis_invalid(env: Env) -> None:
    assert issubclass(PurposeTestBudget, HypothesisInvalid)


def test_purpose_test_with_budget_one_parses(env: Env) -> None:
    hyp = _parse(env, purpose='"test"', budget=1)
    assert (hyp.purpose, hyp.budget) == ("test", 1)


def test_purpose_research_with_budget_two_parses(env: Env) -> None:
    hyp = _parse(env, purpose='"research"', budget=2)
    assert (hyp.purpose, hyp.budget) == ("research", 2)


def test_registration_of_a_test_hypothesis_with_budget_two_is_refused(genesis_env: Env) -> None:
    with pytest.raises(PurposeTestBudget):
        _register(genesis_env, genesis_env.hypothesis_text(purpose='"test"', budget=2))
    assert kinds(genesis_env) == ["genesis"]


def test_a_test_hypothesis_is_stored_as_test(genesis_env: Env) -> None:
    _register(genesis_env, genesis_env.hypothesis_text(purpose='"test"', budget=1))
    record = json.loads((genesis_env.config.registry_path).read_text().splitlines()[-1])
    assert (record["purpose"], record["budget"]) == ("TEST", 1)


# ---------------------------------------------------------------------------- windows must not overlap ----


@pytest.mark.parametrize(
    "windows",
    [APRIL + APRIL_TO_MAY, APRIL + MID_APRIL, MAY + SPRING, SPRING + APRIL],
    ids=["one-shared-day", "nested", "out-of-order", "containing-first"],
)
def test_overlapping_windows_are_refused_at_parse(env: Env, windows: str) -> None:
    with pytest.raises(WindowsOverlap, match="overlap"):
        _parse(env, windows=windows)


def test_a_window_overlap_is_a_hypothesis_invalid() -> None:
    assert issubclass(WindowsOverlap, HypothesisInvalid)


def test_overlap_far_apart_in_declared_order_is_found(env: Env) -> None:
    # APRIL, MAY, then a window holding the end of April: the overlap is with the first window, not the last
    late_april = "[[window]]\nfirst = 2025-04-20\nlast = 2025-04-25\n"
    with pytest.raises(WindowsOverlap):
        _parse(env, windows=APRIL + MAY + late_april)


@pytest.mark.parametrize("windows", [APRIL + MAY, MAY + APRIL], ids=["in-order", "reversed"])
def test_adjacent_windows_without_a_shared_day_parse(env: Env, windows: str) -> None:
    hyp = _parse(env, windows=windows)
    assert sorted(hyp.windows) == [("2025-04-01", "2025-04-30"), ("2025-05-01", "2025-05-30")]


def test_the_same_window_on_two_roots_parses(env: Env) -> None:
    hyp = _parse(env, roots='["MNQ", "MES"]', windows=APRIL)
    assert hyp.roots == ("MNQ", "MES") and hyp.windows == (("2025-04-01", "2025-04-30"),)


def test_registration_of_overlapping_windows_is_refused_before_the_gateway(genesis_env: Env) -> None:
    engine = FakeTrialEngine()
    genesis_env.write(HYPOTHESIS_REL, genesis_env.hypothesis_text(windows=APRIL + APRIL_TO_MAY))
    genesis_env.commit()
    with pytest.raises(WindowsOverlap):
        runner_register(genesis_env.repo / HYPOTHESIS_REL, engine.deps(genesis_env))
    assert engine.called("data_files") == [] and kinds(genesis_env) == ["genesis"]


# -------------------------------------------------------------------------- lookahead_seed in the file ----


def test_lookahead_seed_defaults_to_zero(env: Env) -> None:
    hyp = _parse(env)
    assert hyp.lookahead_seed == DEFAULT_LOOKAHEAD_SEED == 0


def test_lookahead_seed_is_read(env: Env) -> None:
    text = _seeded(env.hypothesis_text(), "5")
    assert parse_hypothesis(text.encode(), "h.toml").lookahead_seed == 5


@pytest.mark.parametrize("value", ["-1", "true", '"5"', "1.5"], ids=["negative", "bool", "string", "float"])
def test_a_bad_lookahead_seed_is_refused(env: Env, value: str) -> None:
    with pytest.raises(HypothesisInvalid, match="lookahead_seed"):
        parse_hypothesis(_seeded(env.hypothesis_text(), value).encode(), "h.toml")


def test_an_unknown_extra_key_is_still_refused(env: Env) -> None:
    text = env.hypothesis_text().replace("[[window]]", "look_seed = 1\n[[window]]", 1)
    with pytest.raises(HypothesisInvalid, match="extra"):
        parse_hypothesis(text.encode(), "h.toml")


def test_the_registered_plumbing_hypothesis_still_parses_with_seed_zero() -> None:
    if not PLUMBING_FILE.is_file():
        pytest.skip("hypotheses/h_example_plumbing.toml not present")
    hyp = load_hypothesis(PLUMBING_FILE)
    assert (hyp.id, hyp.purpose, hyp.budget, hyp.lookahead_seed) == ("h_example_plumbing", "test", 1, 0)
    assert hyp.sha256 == sha256_hex(PLUMBING_FILE.read_bytes())


def test_a_trial_runs_at_the_hypothesis_seed(genesis_env: Env) -> None:
    # Arrange
    _register(genesis_env, _seeded(genesis_env.hypothesis_text(), "5"))
    engine = SeedEngine()
    # Act
    outcome = _trial(genesis_env, engine)
    # Assert: step 1, the engine and the begin record all at 5
    assert engine.seeds == [("lookahead", 5), ("run_strategy", 5)]
    assert outcome.begin_record["seed"] == 5


def test_a_trial_with_the_equal_explicit_seed_runs(genesis_env: Env) -> None:
    _register(genesis_env, _seeded(genesis_env.hypothesis_text(), "5"))
    outcome = _trial(genesis_env, SeedEngine(), seed=5)
    assert outcome.begin_record["seed"] == 5


@pytest.mark.parametrize("given", [0, 3], ids=["zero", "three"])
def test_a_trial_with_another_seed_is_refused_before_step_one_and_begin(genesis_env: Env, given: int) -> None:
    # Arrange
    _register(genesis_env, _seeded(genesis_env.hypothesis_text(), "5"))
    engine = SeedEngine()
    # Act
    with pytest.raises(SeedMismatch, match="lookahead_seed"):
        _trial(genesis_env, engine, seed=given)
    # Assert
    assert engine.seeds == [] and kinds(genesis_env) == ["genesis", "hypothesis"]


def test_a_neighbourhood_trial_with_another_seed_is_refused_before_any_begin(genesis_env: Env) -> None:
    _register(genesis_env, _seeded(genesis_env.hypothesis_text(budget=3, suite__neighbourhood="true"), "5"))
    engine = SeedEngine()
    with pytest.raises(SeedMismatch):
        run_with_neighbours(genesis_env.repo / STRATEGY_REL, kind="trial", params={"a": 2}, seed=3,
                            hypothesis_path=Path(HYPOTHESIS_REL), deps=engine.deps(genesis_env),
                            runs_dir=genesis_env.repo / "runs")  # fmt: skip
    assert engine.seeds == [] and kinds(genesis_env) == ["genesis", "hypothesis"]


def test_a_neighbourhood_trial_without_a_seed_runs_every_member_at_the_hypothesis_seed(
    genesis_env: Env,
) -> None:
    _register(genesis_env, _seeded(genesis_env.hypothesis_text(budget=3, suite__neighbourhood="true"), "5"))
    engine = SeedEngine()
    found = run_with_neighbours(genesis_env.repo / STRATEGY_REL, kind="trial", params={"a": 2}, seed=None,
                                hypothesis_path=Path(HYPOTHESIS_REL), deps=engine.deps(genesis_env),
                                runs_dir=genesis_env.repo / "runs")  # fmt: skip
    assert {seed for _, seed in engine.seeds} == {5}
    assert {o.begin_record["seed"] for o in (found.centre, *found.neighbours)} == {5}


@pytest.mark.parametrize(("given", "expected"), [(None, 0), (3, 3)], ids=["default", "explicit"])
def test_a_synthetic_run_keeps_the_seed_switch(genesis_env: Env, given: int | None, expected: int) -> None:
    engine = SeedEngine()
    outcome = run(genesis_env.repo / STRATEGY_REL, kind="synthetic", deps=engine.deps(genesis_env),
                  runs_dir=genesis_env.repo / "runs", seed=given)  # fmt: skip
    assert engine.seeds == [("lookahead", expected), ("run_strategy", expected)]
    assert outcome.begin_record["seed"] == expected


# ----------------------------------------------------------------------- lookahead_seed on app, CLI, MCP ----


@pytest.fixture
def seeded_app(genesis_env: Env, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Env, SeedEngine]]:
    """H1 with lookahead_seed 5 registered; the app and the CLI on the temporary repository."""
    _register(genesis_env, _seeded(genesis_env.hypothesis_text(), "5"))
    engine = SeedEngine()
    deps = engine.deps(genesis_env)
    monkeypatch.setattr(registry_cmd, "registry_config", lambda _: genesis_env.config)
    monkeypatch.setattr(registry_cmd, "default_deps", lambda *_: deps)
    monkeypatch.setattr(runner_api, "default_deps", lambda *_: deps)
    monkeypatch.setattr(run_cmd, "default_deps", lambda *_: deps)
    monkeypatch.chdir(genesis_env.repo)
    with service.using(deps=deps, runs_dir=genesis_env.repo / "runs"):
        yield genesis_env, engine


def test_app_trial_with_another_seed_is_refused(seeded_app: tuple[Env, SeedEngine]) -> None:
    env, engine = seeded_app
    with pytest.raises(service.Refused) as caught:
        service.run_backtest(STRATEGY_REL, kind="trial", hypothesis=HYPOTHESIS_REL, params={"a": 2}, seed=0)
    assert caught.value.code == "SeedMismatch"
    assert engine.seeds == [] and kinds(env) == ["genesis", "hypothesis"]


def test_app_trial_without_a_seed_runs_at_the_hypothesis_seed(seeded_app: tuple[Env, SeedEngine]) -> None:
    _, engine = seeded_app
    doc = service.run_backtest(STRATEGY_REL, kind="trial", hypothesis=HYPOTHESIS_REL, params={"a": 2})
    assert doc["status"] == "ok" and {seed for _, seed in engine.seeds} == {5}


def test_cli_trial_with_another_seed_exits_2(seeded_app: tuple[Env, SeedEngine]) -> None:
    env, engine = seeded_app
    res = cli.invoke(cli_app, ["run", STRATEGY_REL, "--kind", "trial", "--hypothesis", HYPOTHESIS_REL,
                               "--param", "a=2", "--seed", "0", "--json"])  # fmt: skip
    assert res.exit_code == 2, res.output
    assert json.loads(res.stdout)["error"]["code"] == "SeedMismatch"
    assert engine.seeds == [] and kinds(env) == ["genesis", "hypothesis"]


def test_cli_trial_without_a_seed_runs_at_the_hypothesis_seed(seeded_app: tuple[Env, SeedEngine]) -> None:
    _, engine = seeded_app
    res = cli.invoke(cli_app, ["run", STRATEGY_REL, "--kind", "trial", "--hypothesis", HYPOTHESIS_REL,
                               "--param", "a=2", "--json"])  # fmt: skip
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["status"] == "ok" and {seed for _, seed in engine.seeds} == {5}


def test_cli_synthetic_seed_switch_still_works(seeded_app: tuple[Env, SeedEngine]) -> None:
    _, engine = seeded_app
    res = cli.invoke(cli_app, ["run", STRATEGY_REL, "--kind", "synthetic", "--seed", "3", "--json"])
    assert res.exit_code == 0, res.output
    assert {seed for _, seed in engine.seeds} == {3}


def test_mcp_run_backtest_leaves_the_seed_unset_by_default() -> None:
    import inspect

    from pfb.mcp import server

    assert inspect.signature(server.run_backtest).parameters["seed"].default is None
    assert inspect.signature(service.run_backtest).parameters["seed"].default is None


def test_mcp_trial_with_another_seed_is_refused(
    seeded_app: tuple[Env, SeedEngine],
) -> None:
    from pfb.mcp import server

    env, engine = seeded_app
    out = server.run_backtest(STRATEGY_REL, "trial", hypothesis=HYPOTHESIS_REL, params={"a": 2}, seed=1)
    assert out["status"] == "refused" and out["error"]["code"] == "SeedMismatch"
    assert engine.seeds == [] and kinds(env) == ["genesis", "hypothesis"]
