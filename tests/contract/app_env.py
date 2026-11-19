"""Shared fixtures of tests/contract and tests/cli: temporary git repositories and ledgers,
synthetic bars, never the real ledger. The strategy is strategies/_template.py copied into the temporary
repository. Imported by both folders' conftest.py."""

from __future__ import annotations

import shutil
import sys
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS / "registry"))
sys.path.insert(0, str(TESTS / "trials"))

from registry_env import Env  # noqa: E402  # pyright: ignore[reportMissingImports]

from pfb import app  # noqa: E402
from pfb.registry.api import create_genesis  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "strategies" / "_template.py"
PATHS_TOML = REPO / "config" / "paths.toml"
STRATEGY_REL = "strategies/tpl.py"
PLAN = "combine_xfa_standard_50k"
FIRM = "topstep"


@dataclass(frozen=True)
class AppEnv:
    env: Env
    deps: Any
    strategy: Path
    runs: Path

    @property
    def ledger(self) -> Path:
        return self.env.config.registry_path


@pytest.fixture
def genesis_env(tmp_path: Path) -> Env:
    env = Env(tmp_path)
    create_genesis(env.config)
    env.checkpoint()
    return env


@pytest.fixture
def real_deps(genesis_env: Env) -> Any:
    from pfb.runner.deps import default_deps

    return replace(default_deps(), registry_config=lambda: genesis_env.config)


@pytest.fixture
def app_env(genesis_env: Env, real_deps: Any) -> Iterator[AppEnv]:
    """The template in a temporary repository; pfb.app pointed at its ledger and runs directory."""
    strategy = genesis_env.repo / STRATEGY_REL
    shutil.copyfile(TEMPLATE, strategy)
    runs = genesis_env.repo / "runs"
    with app.using(deps=real_deps, runs_dir=runs, ledger=None):
        yield AppEnv(genesis_env, real_deps, strategy, runs)
