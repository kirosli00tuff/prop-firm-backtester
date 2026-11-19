"""pfb.runner.run against a fake engine: record order, permits, refusals and artifacts (ADR A1, A2, A14)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest
from registry_env import STRATEGY_REL, Env  # pyright: ignore[reportMissingImports]

import pfb.runner.api as runner_api
from pfb.data.errors import NoBarsFound, PathsConfigError
from pfb.registry.api import create_genesis, status
from pfb.registry.chain import read_ledger, sha256_hex
from pfb.registry.errors import EngineCheckNotDeclared
from pfb.runner.api import ProvenanceMismatch, RunRefused, planned_artifacts, run
from pfb.runner.deps import RunnerDeps

DATA = (("research/ohlcv-1m_MNQ_research.parquet", "d" * 64),)


@dataclass
class FakeEngine:
    """Records every call; returns a fixed result."""

    calls: list[tuple[str, Any]] = field(default_factory=list)
    lint: list[str] = field(default_factory=list)
    leak: bool = False
    crash: bool = False
    provenance: tuple[tuple[str, str], ...] = DATA
    summary: dict[str, Any] = field(default_factory=lambda: {"net_cents": 125, "trades": 2})
    lookahead_warnings: tuple[str, ...] = ()
    roots: Mapping[str, Path] | None = None  # None: synthetic stores beside the temporary repository

    def strategy(self, path: Path) -> Any:
        params = {"a": SimpleNamespace(default=1, grid=(1, 2))}
        spec = SimpleNamespace(root="MNQ", session="globex", hypothesis_id="engine-check", params=params)
        return SimpleNamespace(spec=spec, sha256=sha256_hex(path.read_bytes()), generate=None, path=path)

    def lookahead(self, strategy: Any, params: Mapping[str, Any], seed: int) -> tuple[str, ...]:
        self.calls.append(("lookahead", dict(params)))
        if self.leak:
            raise ValueError("LookAheadDetected")
        return self.lookahead_warnings

    def run_strategy(self, path: Path, **kwargs: Any) -> Any:
        self.calls.append(("run_strategy", kwargs))
        if self.crash:
            raise ZeroDivisionError("engine crashed")
        frame = pl.DataFrame({"x": [1]})
        return SimpleNamespace(
            trades=frame, equity_minute=frame, daily=frame, summary=self.summary, provenance=self.provenance
        )

    def deps(self, env: Env) -> RunnerDeps:
        return RunnerDeps(
            load_strategy=self.strategy,
            lint_strategy=lambda p: list(self.lint),
            lookahead_check=self.lookahead,
            run_strategy=self.run_strategy,
            make_permit=lambda **k: SimpleNamespace(**k),
            engine_version=lambda: "pfb-engine/test",
            instruments_sha256=lambda: "1" * 64,
            cost_model_ids=lambda spec, event_costs: (spec, "2" * 64),
            fees_sha256=lambda firm: "3" * 64,
            closed_windows_sha256=lambda: "4" * 64,
            data_files=lambda root, first, last: DATA,
            provenance_files=lambda result: result.provenance,
            registry_config=lambda: env.config,
            store_roots=lambda: self.roots if self.roots is not None else synthetic_roots(env.repo.parent),
        )


def synthetic_roots(base: Path) -> dict[str, Path]:
    return {name: base / "synthetic_stores" / name for name in ("research", "step2", "hist")}


@pytest.fixture
def ready(env: Env) -> Env:
    create_genesis(env.config)
    env.write("config/engine_checks.toml", env.engine_checks_text())
    env.commit()
    return env


def _kinds(env: Env) -> list[str]:
    return [line.record["kind"] for line in read_ledger(env.config.registry_path)]


def test_synthetic_run_logs_begin_then_result_and_writes_planned_artifacts(
    ready: Env, tmp_path: Path
) -> None:
    # Arrange
    engine = FakeEngine()
    runs = ready.repo / "runs"  # inside the repository (ignored by its .gitignore): repo-relative names
    # Act
    outcome = run(ready.repo / STRATEGY_REL, kind="synthetic", deps=engine.deps(ready), runs_dir=runs)
    # Assert
    assert _kinds(ready) == ["genesis", "synthetic", "result"]
    assert outcome.result_record["metrics"] == engine.summary
    run_id = outcome.begin_record["run_id"]
    assert outcome.result_record["artifacts"] == [
        f"runs/{run_id}/{n}"
        for n in ("trades.parquet", "equity_minute.parquet", "daily.parquet", "summary.json")
    ]
    assert all((ready.repo / a).is_file() for a in outcome.result_record["artifacts"])
    kwargs = engine.calls[-1][1]
    assert kwargs["source"] == "synthetic" and kwargs["n_trials"] == 302 and "permit" not in kwargs


def test_engine_check_runs_real_with_a_permit_from_the_receipt(ready: Env, tmp_path: Path) -> None:
    engine = FakeEngine()
    outcome = run(
        ready.repo / STRATEGY_REL,
        kind="engine_check",
        check_id="E3",
        deps=engine.deps(ready),
        runs_dir=tmp_path / "runs",
    )
    kwargs = engine.calls[-1][1]
    permit = kwargs["permit"]
    lines = read_ledger(ready.config.registry_path)
    assert kwargs["source"] == "real" and str(kwargs["first"]) == "2025-04-01"
    assert permit.run_id == outcome.begin_record["run_id"] and permit.record_sha256 == lines[1].sha256
    assert permit.params_json == '{"a":1}'
    assert outcome.begin_record["data_files"] == [{"path": DATA[0][0], "sha256": DATA[0][1]}]
    assert outcome.begin_record["n_after"] == 302


def test_engine_check_with_non_default_params_is_refused_before_begin(ready: Env, tmp_path: Path) -> None:
    with pytest.raises(RunRefused):
        run(
            ready.repo / STRATEGY_REL,
            kind="engine_check",
            check_id="E3",
            params={"a": 2},
            deps=FakeEngine().deps(ready),
            runs_dir=tmp_path,
        )
    assert _kinds(ready) == ["genesis"]


def test_engine_check_with_another_window_is_refused_before_begin(ready: Env, tmp_path: Path) -> None:
    with pytest.raises(RunRefused):
        run(
            ready.repo / STRATEGY_REL,
            kind="engine_check",
            check_id="E3",
            first=date(2025, 4, 2),
            deps=FakeEngine().deps(ready),
            runs_dir=tmp_path,
        )
    assert _kinds(ready) == ["genesis"]


def test_an_undeclared_engine_check_is_refused(ready: Env, tmp_path: Path) -> None:
    with pytest.raises(EngineCheckNotDeclared):
        run(
            ready.repo / STRATEGY_REL,
            kind="engine_check",
            check_id="NOPE",
            deps=FakeEngine().deps(ready),
            runs_dir=tmp_path,
        )
    assert _kinds(ready) == ["genesis"]


@pytest.mark.parametrize(
    "engine", [FakeEngine(lint=["banned open"]), FakeEngine(leak=True)], ids=["lint", "leak"]
)
def test_lint_or_lookahead_failure_is_refused_before_begin(
    ready: Env, tmp_path: Path, engine: FakeEngine
) -> None:
    with pytest.raises(RunRefused):
        run(ready.repo / STRATEGY_REL, kind="synthetic", deps=engine.deps(ready), runs_dir=tmp_path)
    assert _kinds(ready) == ["genesis"]


def test_a_trial_without_a_hypothesis_file_is_refused_before_begin(ready: Env, tmp_path: Path) -> None:
    with pytest.raises(RunRefused, match="hypothesis"):
        run(ready.repo / STRATEGY_REL, kind="trial", deps=FakeEngine().deps(ready), runs_dir=tmp_path)
    assert _kinds(ready) == ["genesis"]


def test_an_engine_crash_logs_a_failed_result_and_re_raises(ready: Env, tmp_path: Path) -> None:
    with pytest.raises(ZeroDivisionError):
        run(
            ready.repo / STRATEGY_REL,
            kind="synthetic",
            deps=FakeEngine(crash=True).deps(ready),
            runs_dir=tmp_path / "runs",
        )
    last = read_ledger(ready.config.registry_path)[-1].record
    assert (last["kind"], last["status"], last["error"]) == ("result", "failed", "ZeroDivisionError")
    assert status(ready.config).open_runs == ()
    assert not (tmp_path / "runs").exists()


def test_a_provenance_different_from_files_for_fails_the_run(ready: Env, tmp_path: Path) -> None:
    engine = FakeEngine(provenance=(("research/other.parquet", "e" * 64),))
    with pytest.raises(ProvenanceMismatch):
        run(
            ready.repo / STRATEGY_REL,
            kind="engine_check",
            check_id="E3",
            deps=engine.deps(ready),
            runs_dir=tmp_path,
        )
    assert read_ledger(ready.config.registry_path)[-1].record["error"] == "ProvenanceMismatch"


def test_metrics_that_are_not_json_log_a_failed_result(ready: Env, tmp_path: Path) -> None:
    engine = FakeEngine(summary={"sharpe": float("nan")})
    with pytest.raises(Exception, match="JSON"):
        run(ready.repo / STRATEGY_REL, kind="synthetic", deps=engine.deps(ready), runs_dir=tmp_path)
    last = read_ledger(ready.config.registry_path)[-1].record
    assert (last["status"], last["error"]) == ("failed", "MetricsNotJson")


def test_unknown_params_are_refused(ready: Env, tmp_path: Path) -> None:
    with pytest.raises(RunRefused):
        run(
            ready.repo / STRATEGY_REL,
            kind="synthetic",
            params={"zz": 1},
            deps=FakeEngine().deps(ready),
            runs_dir=tmp_path,
        )


def test_params_reach_the_lookahead_check_resolved(ready: Env, tmp_path: Path) -> None:
    engine = FakeEngine()
    run(
        ready.repo / STRATEGY_REL,
        kind="synthetic",
        params={"a": 2},
        deps=engine.deps(ready),
        runs_dir=tmp_path,
    )
    assert engine.calls[0] == ("lookahead", {"a": 2})


# ------------------------------------------------------------------------- registry and path refusals ----


def test_the_outcome_carries_the_sha256_of_its_two_ledger_lines(ready: Env, tmp_path: Path) -> None:
    outcome = run(
        ready.repo / STRATEGY_REL,
        kind="engine_check",
        check_id="E3",
        deps=FakeEngine().deps(ready),
        runs_dir=tmp_path / "runs",
    )
    lines = read_ledger(ready.config.registry_path)
    assert (outcome.begin_sha256, outcome.result_sha256) == (lines[1].sha256, lines[2].sha256)
    assert lines[2].record["prev_sha256"] == outcome.begin_sha256


def _refusing_data_files(root: str, first: date, last: date) -> Any:
    raise NoBarsFound("no catalogued file of MNQ overlaps 2025-04")


def test_a_gateway_refusal_before_begin_is_a_run_refusal_and_logs_nothing(ready: Env, tmp_path: Path) -> None:
    # files_for refusing (a closed window, no catalog) is a refusal, not a traceback
    deps = replace(FakeEngine().deps(ready), data_files=_refusing_data_files)
    with pytest.raises(RunRefused, match="NoBarsFound"):
        run(ready.repo / STRATEGY_REL, kind="engine_check", check_id="E3", deps=deps, runs_dir=tmp_path)
    assert _kinds(ready) == ["genesis"]


def test_a_paths_config_error_in_default_deps_is_a_run_refusal(
    ready: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_paths() -> RunnerDeps:
        raise PathsConfigError("config/paths.toml: not found")

    monkeypatch.setattr(runner_api, "default_deps", no_paths)
    with pytest.raises(RunRefused, match="PathsConfigError"):
        run(ready.repo / STRATEGY_REL, kind="synthetic", runs_dir=tmp_path)
    assert _kinds(ready) == ["genesis"]


def test_a_gateway_refusal_after_begin_is_logged_failed_and_re_raised(ready: Env, tmp_path: Path) -> None:
    engine = FakeEngine()

    def refuse(path: Path, **kwargs: Any) -> Any:
        raise NoBarsFound("the store file vanished")

    deps = replace(engine.deps(ready), run_strategy=refuse)
    with pytest.raises(NoBarsFound):
        run(ready.repo / STRATEGY_REL, kind="engine_check", check_id="E3", deps=deps, runs_dir=tmp_path)
    last = read_ledger(ready.config.registry_path)[-1].record
    assert (last["status"], last["error"]) == ("failed", "NoBarsFound")


def test_artifacts_outside_the_repository_are_named_external_without_an_absolute_path(
    ready: Env, tmp_path: Path
) -> None:
    runs = tmp_path / "elsewhere" / "runs"
    outcome = run(ready.repo / STRATEGY_REL, kind="synthetic", deps=FakeEngine().deps(ready), runs_dir=runs)
    run_id = outcome.begin_record["run_id"]
    artifacts = outcome.result_record["artifacts"]
    assert artifacts[0] == f"external:runs/{run_id}/trades.parquet"
    assert all(a.startswith("external:runs/") and str(tmp_path) not in a for a in artifacts)
    assert all((runs / a.removeprefix("external:runs/")).is_file() for a in artifacts)


def test_planned_artifacts_are_repo_relative_inside_the_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    names = planned_artifacts(repo / "out" / "runs", "r1", repo)
    assert names[-1] == "out/runs/r1/summary.json"
    assert planned_artifacts(tmp_path / "x", "r1", repo)[-1] == "external:x/r1/summary.json"


# ------------------------------------------------------------------------- ledger and store-root checks ----


def _real_paths_toml(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    """A paths.toml standing in for config/paths.toml: its store roots are 'the configured real ones'."""
    roots = {name: tmp_path / "real_stores" / name for name in ("research", "step2", "hist")}
    toml = tmp_path / "paths.toml"
    toml.write_text(
        "[stores]\n"
        + "".join(f'{name} = "{root}"\n' for name, root in roots.items())
        + f'[upstream]\nroot = "{tmp_path / "pe_checkout"}"\n'
        + f'[costs]\nbook_table = "{tmp_path / "e2a.parquet"}"\n'
    )
    return toml, roots


def test_a_real_data_run_on_the_real_store_roots_with_a_throwaway_ledger_is_refused_before_begin(
    ready: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the registry swapped for a throwaway one while the engine reads the real stores
    toml, real = _real_paths_toml(tmp_path)
    monkeypatch.setattr(runner_api, "PATHS_TOML", toml)
    engine = FakeEngine(roots={"research": real["research"] / ".." / "research"})
    with pytest.raises(RunRefused, match="ledger"):
        run(ready.repo / STRATEGY_REL, kind="engine_check", check_id="E3", deps=engine.deps(ready),
            runs_dir=tmp_path / "runs")  # fmt: skip
    assert _kinds(ready) == ["genesis"]
    assert [c[0] for c in engine.calls] == ["lookahead"]


def test_a_real_data_run_on_synthetic_store_roots_with_a_temp_ledger_is_allowed(
    ready: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    toml, _ = _real_paths_toml(tmp_path)
    monkeypatch.setattr(runner_api, "PATHS_TOML", toml)
    outcome = run(ready.repo / STRATEGY_REL, kind="engine_check", check_id="E3",
                  deps=FakeEngine().deps(ready), runs_dir=tmp_path / "runs")  # fmt: skip
    assert outcome.result_record["status"] == "ok"
    assert _kinds(ready) == ["genesis", "engine_check", "result"]


def test_a_real_data_run_on_the_real_store_roots_with_the_repo_ledger_is_allowed(
    ready: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "the repo's ledger" is redirected to the temporary one, written under another spelling (paths resolved)
    toml, real = _real_paths_toml(tmp_path)
    monkeypatch.setattr(runner_api, "PATHS_TOML", toml)
    ledger = ready.config.registry_path
    monkeypatch.setattr(runner_api, "REPO_LEDGER", ledger.parent / ".." / ledger.parent.name / ledger.name)
    outcome = run(ready.repo / STRATEGY_REL, kind="engine_check", check_id="E3",
                  deps=FakeEngine(roots=real).deps(ready), runs_dir=tmp_path / "runs")  # fmt: skip
    assert outcome.result_record["status"] == "ok"
    assert _kinds(ready) == ["genesis", "engine_check", "result"]


def test_a_synthetic_run_reads_no_store_so_real_roots_with_a_temp_ledger_stay_allowed(
    ready: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # kind synthetic never opens a store (tests/runner/test_end_to_end.py runs default_deps on a temp ledger)
    toml, real = _real_paths_toml(tmp_path)
    monkeypatch.setattr(runner_api, "PATHS_TOML", toml)
    outcome = run(ready.repo / STRATEGY_REL, kind="synthetic", deps=FakeEngine(roots=real).deps(ready),
                  runs_dir=tmp_path / "runs")  # fmt: skip
    assert outcome.result_record["status"] == "ok"


def test_step_one_lookahead_warnings_are_appended_to_the_result_warnings(ready: Env, tmp_path: Path) -> None:
    # step 1's warning reaches the begin record, result record and summary.json with the
    # prefix "step 1: "; the engine's in-run look-ahead warning gets "in-run: "
    engine = FakeEngine(
        lookahead_warnings=("look-ahead check saw no signal",),
        summary={"trades": 0, "warnings": ["engine warning", "look-ahead check saw no signal"]},
    )
    runs = tmp_path / "r"
    outcome = run(ready.repo / STRATEGY_REL, kind="synthetic", deps=engine.deps(ready), runs_dir=runs)
    expected = [
        "engine warning",
        "in-run: look-ahead check saw no signal",
        "step 1: look-ahead check saw no signal",
    ]
    assert outcome.begin_record["warnings"] == ["step 1: look-ahead check saw no signal"]
    assert outcome.begin_record["lookahead_signal"] == "none"
    assert outcome.result_record["metrics"]["warnings"] == expected
    assert outcome.result.summary["warnings"] == expected
    run_id = outcome.begin_record["run_id"]
    assert json.loads((runs / run_id / "summary.json").read_text())["warnings"] == expected
    assert engine.summary["warnings"] == ["engine warning", "look-ahead check saw no signal"]  # not mutated
