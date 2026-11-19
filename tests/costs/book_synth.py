"""Synthetic book cost tables for the cost tests (no real values)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pfb.instruments import InstrumentTable

PE_TABLE_REL = "reports/costs_table.json"


@dataclass(frozen=True)
class TablePaths:
    """Satisfies pfb.costs.CostPaths."""

    book_table: Path


@dataclass(frozen=True)
class SynthBook:
    paths: TablePaths
    costs_toml: Path


def bucket(start: int, end: int, buy: float, sell: float, *, fallback: bool = False) -> dict[str, Any]:
    """A synthetic bucket: half-spread 0, depth = side ticks, so side == half + depth exactly."""
    return {
        "key": f"{start // 60:02d}:{start % 60:02d}",
        "start_min": start,
        "end_min": end,
        "half_spread_ticks": 0.0,
        "depth_ticks": {"buy": buy, "sell": sell},
        "side_ticks": {"buy": buy, "sell": sell},
        "fallback": fallback,
    }


def product(root: str, instruments: InstrumentTable, buckets: list[dict[str, Any]]) -> dict[str, Any]:
    tv = float(instruments.get(root).tick_value_usd)
    return {"product": root, "calibrated": True, "tick_value_usd": tv, "buckets": buckets}


def write_costs_toml(path: Path, sha256: str, products: int) -> None:
    path.write_text(
        f'schema = "pfb.costs/1"\n[book]\nsha256 = "{sha256}"\nproducts = {products}\n', encoding="utf-8"
    )
