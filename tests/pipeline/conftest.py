"""The slow planted-edge cases stay out of the default suite.

pyproject.toml registers the `slow` marker but its addopts deselects nothing, and the usual suite command
(`pytest -q -m "not real_data"`) runs slow tests. So the slow tests of THIS directory are skipped unless the
mark expression names `slow` (`-m slow`, `-m "slow and not real_data"`) or PFB_RUN_SLOW=1 is set. Tests of
other directories are not touched.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
RUN_SLOW_ENV = "PFB_RUN_SLOW"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if "slow" in str(config.getoption("markexpr") or "") or os.environ.get(RUN_SLOW_ENV) == "1":
        return
    skip = pytest.mark.skip(reason=f"slow planted-edge case: run with -m slow or {RUN_SLOW_ENV}=1")
    for item in items:
        if item.get_closest_marker("slow") and HERE in Path(str(item.path)).resolve().parents:
            item.add_marker(skip)
