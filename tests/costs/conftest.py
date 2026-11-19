"""Fixtures for the cost tests: the tracked instrument table and synthetic book tables under tmp_path."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from book_synth import SynthBook, TablePaths, write_costs_toml

from pfb.instruments import InstrumentTable, load_instruments


@pytest.fixture(scope="session")
def instruments() -> InstrumentTable:
    return load_instruments()


@pytest.fixture
def make_book(tmp_path: Path) -> Callable[..., SynthBook]:
    """Write a synthetic table and a costs.toml pinning its sha256 (pin=None: pin the real hash)."""

    def _make(products: dict[str, Any], *, depth_term: str = "applied", pin: str | None = None) -> SynthBook:
        table = tmp_path / "costs_table.json"
        data = json.dumps({"depth_term": depth_term, "products": products}).encode("utf-8")
        table.write_bytes(data)
        costs_toml = tmp_path / "costs.toml"
        write_costs_toml(costs_toml, pin or hashlib.sha256(data).hexdigest(), len(products))
        return SynthBook(TablePaths(table), costs_toml)

    return _make
