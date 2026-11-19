"""Pfb.runner.run refuses a leaky strategy file with the REAL lint, loader and truncation check BEFORE the
gateway is asked for anything and before any registry record (ADR A1 step 1). Test doubles stand in for the
gateway and the engine run; the ledger is a temporary one."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest
from lookahead_mutants import LEAKY_STRATEGY_SOURCE, peeking_source  # pyright: ignore[reportMissingImports]
from registry_env import STRATEGY_REL, Env  # pyright: ignore[reportMissingImports]

from pfb.engine.lint import lint_strategy
from pfb.engine.strategy import load_strategy
from pfb.registry.api import create_genesis
from pfb.registry.chain import read_ledger
from pfb.runner.api import RunRefused, run
from pfb.runner.deps import RunnerDeps, _lookahead, default_deps

REFERENCE = Path(__file__).resolve().parents[2] / "strategies" / "reference_breakout.py"
DATA = (("research/ohlcv-1m_MNQ_research.parquet", "d" * 64),)


@dataclass
class RecordingGateway:
    """Stands in for the DataGateway and the engine's bar read: records every request."""

    calls: list[str] = field(default_factory=list)

    def files_for(self, root: str, first: Any, last: Any) -> tuple[tuple[str, str], ...]:
        self.calls.append("files_for")
        return DATA

    def run_strategy(self, path: Path, **kwargs: Any) -> Any:
        self.calls.append("bars")
        frame = pl.DataFrame({"x": [1]})
        return SimpleNamespace(trades=frame, equity_minute=frame, daily=frame,
                               summary={"trades": 0}, provenance=DATA)  # fmt: skip

    def deps(self, env: Env) -> RunnerDeps:
        return RunnerDeps(
            load_strategy=load_strategy,
            lint_strategy=lint_strategy,
            lookahead_check=_lookahead,  # the check default_deps wires: synthetic bars, 20 seeded cuts
            run_strategy=self.run_strategy,
            make_permit=lambda **k: SimpleNamespace(**k),
            engine_version=lambda: "pfb-engine/test",
            instruments_sha256=lambda: "1" * 64,
            cost_model_ids=lambda spec, event_costs: (spec, "2" * 64),  # (spec, event_costs) since
            fees_sha256=lambda firm: "3" * 64,
            closed_windows_sha256=lambda: "4" * 64,
            data_files=self.files_for,
            provenance_files=lambda result: result.provenance,
            registry_config=lambda: env.config,
            store_roots=lambda: {"research": env.repo.parent / "synthetic_stores" / "research"},
        )


def _ready(env: Env, source: str) -> Env:
    create_genesis(env.config)
    env.write(STRATEGY_REL, source)
    env.commit()
    env.write("config/engine_checks.toml", env.engine_checks_text())
    env.commit()
    return env


def _kinds(env: Env) -> list[str]:
    return [line.record["kind"] for line in read_ledger(env.config.registry_path)]


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


@pytest.fixture
def no_real_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a real DataGateway was constructed")

    monkeypatch.setattr("pfb.data.gateway.DataGateway.__init__", refuse)


SOURCES = {
    "next_session_open": LEAKY_STRATEGY_SOURCE,
    "sparse_one_bar_peek": peeking_source(REFERENCE.read_text()),
}


@pytest.mark.parametrize("leak", sorted(SOURCES))
@pytest.mark.parametrize("kind", ["engine_check", "synthetic"])
def test_a_leaky_strategy_is_refused_before_the_gateway_and_before_begin(
    env: Env, tmp_path: Path, kind: str, leak: str, no_real_gateway: None
) -> None:
    # Arrange
    ready = _ready(env, SOURCES[leak])
    gateway = RecordingGateway()
    check_id = "E3" if kind == "engine_check" else None
    # Act
    with pytest.raises(RunRefused, match="look-ahead check failed: LookAheadDetected"):
        run(ready.repo / STRATEGY_REL, kind=kind, check_id=check_id, deps=gateway.deps(ready),
            runs_dir=tmp_path / "runs")  # fmt: skip
    # Assert
    assert gateway.calls == []
    assert _kinds(ready) == ["genesis"]
    assert not (tmp_path / "runs").exists()


def test_the_honest_reference_strategy_passes_and_reaches_the_gateway_double(
    env: Env, tmp_path: Path, no_real_gateway: None
) -> None:
    # Arrange: positive control, the same doubles DO see requests once the check passes
    ready = _ready(env, REFERENCE.read_text())
    gateway = RecordingGateway()
    # Act
    outcome = run(ready.repo / STRATEGY_REL, kind="engine_check", check_id="E3", deps=gateway.deps(ready),
                  runs_dir=tmp_path / "runs")  # fmt: skip
    # Assert
    assert gateway.calls == ["files_for", "bars"]
    assert _kinds(ready) == ["genesis", "engine_check", "result"]
    assert outcome.begin_record["static_checks"] == {"lint": "pass", "lookahead": "pass"}


def test_default_deps_wire_the_same_truncation_check(paths_toml: Path) -> None:
    assert default_deps(paths_toml).lookahead_check is _lookahead
