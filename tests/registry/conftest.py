"""Fixtures for the registry tests (helpers in registry_env.py)."""

from __future__ import annotations

from pathlib import Path

import pytest
from registry_env import HYPOTHESIS_REL, Env

from pfb.registry.api import create_genesis, register_hypothesis


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


@pytest.fixture
def genesis_env(env: Env) -> Env:
    create_genesis(env.config)
    return env


@pytest.fixture
def trial_env(genesis_env: Env) -> Env:
    """Genesis plus a committed, registered hypothesis H1 (grid a in 1, 2, 3; budget 2), ledger at HEAD."""
    genesis_env.write(HYPOTHESIS_REL, genesis_env.hypothesis_text())
    genesis_env.commit()
    register_hypothesis(genesis_env.repo / HYPOTHESIS_REL, genesis_env.config, genesis_env.facts())
    genesis_env.checkpoint()
    return genesis_env
