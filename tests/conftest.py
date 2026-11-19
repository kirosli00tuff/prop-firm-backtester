"""Shared pytest constants and fixtures."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def paths_toml() -> Path:
    """config/paths.toml; without it, pfb.data.paths.load_paths gives the demo store's paths."""
    return REPO_ROOT / "config" / "paths.toml"
