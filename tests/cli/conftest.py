"""Fixtures for the CLI tests (defined in tests/contract/app_env.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "contract"))

from app_env import app_env, genesis_env, real_deps  # noqa: E402  # pyright: ignore[reportMissingImports]

__all__ = ["app_env", "genesis_env", "real_deps"]
