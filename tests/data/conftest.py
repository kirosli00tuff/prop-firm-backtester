"""Fixtures for the data-layer tests: synthetic stores under tmp_path."""

from pathlib import Path

import pytest
from gateway_synth import SynthStores


@pytest.fixture
def stores(tmp_path: Path) -> SynthStores:
    return SynthStores(tmp_path)
