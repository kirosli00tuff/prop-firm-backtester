"""Dashboard tests: one temporary repository, ledger and runs directory, built once per module.

The fixture runs a synthetic neighbourhood (centre and two neighbours, each with a suite) through the real
runner and engine on synthetic bars, the prop Monte Carlo of one plan and the matrix on the centre, begins one
run that never finishes (open), and removes two frames from one neighbour (missing artifacts). Nothing touches
this repository's ledger or runs directory, and N in the temporary ledger is the synthetic upstream's 302.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "registry"))

from registry_env import Env  # noqa: E402  # pyright: ignore[reportMissingImports]

from pfb.analysis.spec import SuiteSpec  # noqa: E402
from pfb.dash.data import DashConfig  # noqa: E402
from pfb.registry.api import begin, create_genesis  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_main_module() -> Iterator[None]:
    """Streamlit's AppTest runs its script as a new __main__ module and leaves it in sys.modules; a later
    test that starts processes with the "spawn" method would then re-run that script in each child. Put the
    process's own __main__ back after every dashboard test."""
    main = sys.modules.get("__main__")
    yield
    if main is not None:
        sys.modules["__main__"] = main


STRATEGY_SRC = Path(__file__).resolve().parent / "strategies" / "dash_random.py"
PATHS_TOML = Path(__file__).resolve().parents[2] / "config" / "paths.toml"
SIZING_PLAN = "topstep/combine_xfa_standard_50k"
PROP_MC = 400
PROP_SEED = 11
REMOVED = ("daily.parquet", "equity_minute.parquet")


@dataclass(frozen=True)
class DashEnv:
    config: DashConfig
    env: Env
    centre: str
    neighbours: tuple[str, ...]
    open_run: str
    prop_file: Path


def _run_prop(args: list[str]) -> None:
    from pfb.cli.prop_cmd import prop_app

    done = CliRunner().invoke(prop_app, args)
    assert done.exit_code == 0, done.output


def build(base: Path) -> DashEnv:
    from pfb.runner.deps import default_deps
    from pfb.runner.neighbours import run_with_neighbours

    env = Env(base)
    create_genesis(env.config)
    strategy = env.write("strategies/dash_random.py", STRATEGY_SRC.read_text())
    deps = replace(default_deps(), registry_config=lambda: env.config)
    runs = env.repo / "runs"
    suite = SuiteSpec(walk_forward_folds=3, sizing_plan=SIZING_PLAN)
    outcome = run_with_neighbours(
        strategy, kind="synthetic", cost_model="fixed:1", deps=deps, runs_dir=runs, suite=suite
    )
    centre = outcome.centre.begin_record["run_id"]
    neighbours = tuple(o.begin_record["run_id"] for o in outcome.neighbours)
    ledger = str(env.config.registry_path)
    common = ["--runs-dir", str(runs), "--ledger", ledger, "--mc", str(PROP_MC), "--seed", str(PROP_SEED)]
    _run_prop([centre, "--firm", "topstep", "--plan", "combine_xfa_standard", "--size", "50k", *common])
    _run_prop(["matrix", centre, *common])
    open_run = begin(env.request("synthetic"), config=env.config).run_id
    for name in REMOVED:
        (runs / neighbours[1] / name).unlink()
    prop_file = next((runs / centre / "prop").glob("*.json"))
    config = DashConfig(
        repo_root=env.repo, ledger_path=env.config.registry_path, runs_dir=runs, pe_root=env.pe
    )
    return DashEnv(config, env, centre, neighbours, open_run, prop_file)


@pytest.fixture(scope="session")
def dash_env(tmp_path_factory: pytest.TempPathFactory) -> DashEnv:
    return build(tmp_path_factory.mktemp("dash"))


@pytest.fixture
def broken_ledger(tmp_path: Path, dash_env: DashEnv) -> DashConfig:
    """A copy of the ledger whose last line is cut: the chain check fails on every read."""
    ledger = tmp_path / "registry.jsonl"
    shutil.copyfile(dash_env.config.ledger_path, ledger)
    lines = ledger.read_bytes().splitlines(keepends=True)
    ledger.write_bytes(b"".join(lines[:-1]) + lines[-1][:-20])
    return replace(dash_env.config, ledger_path=ledger, repo_root=tmp_path)
