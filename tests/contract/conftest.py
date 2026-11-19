"""Fixtures for the result-contract tests (defined in app_env.py, shared with tests/cli)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app_env import app_env, genesis_env, real_deps  # noqa: E402

__all__ = ["app_env", "genesis_env", "real_deps"]
