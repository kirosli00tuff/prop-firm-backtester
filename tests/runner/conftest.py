"""Runner tests reuse the registry test helpers (temporary git repository, synthetic upstream ledger)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "registry"))

from registry_env import Env  # noqa: E402  # pyright: ignore[reportMissingImports]


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)
