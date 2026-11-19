"""Synthetic bar stores for the gateway tests. Every row is generated here; no real row is used.

The synthetic calendar runs in early 2021. Each trade date D gets three bars: 17:30 CT on the evening before
(Sunday for a Monday), 08:30 CT and 15:59 CT on D. Prices sit on the root's tick grid.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq

from pfb.data.catalog import Catalog, PinKey, build_catalog
from pfb.data.gateway import STORE_METADATA_KEY, DataGateway
from pfb.data.paths import StorePaths
from pfb.data.windows import WindowTable, datetime_to_ns, load_windows

CT = ZoneInfo("America/Chicago")
TICKS_FIXED = {
    "NQ": 250_000_000, "ZL": 10_000_000, "ZM": 100_000_000, "ZN": 15_625_000, "ZC": 250_000_000,
    "MES": 250_000_000,
}  # fmt: skip
# Vendor price factor per root: ZC is quoted in cents (vendor tick 0.25 = exchange tick 0.0025 x 100).
PRICE_FACTORS = {"NQ": 1, "ZL": 1, "ZM": 1, "ZN": 1, "ZC": 100, "MES": 1}
BASE_TICKS = 40_000
BAR_TIMES = ((-1, time(17, 30)), (0, time(8, 30)), (0, time(15, 59)))

WINDOW_TOML = """schema = "pfb.closed_windows/1"

[[window]]
id = "syn_holdout"
products = ["*"]
first_trade_date = "2021-03-01"
last_trade_date = "9999-12-31"
utc_start = "2021-02-26T22:01:00Z"
utc_end = "9999-12-31T00:00:00Z"
kind = "sealed holdout"

[[window]]
id = "syn_embargo"
products = ["*"]
first_trade_date = "2021-02-15"
last_trade_date = "2021-02-19"
utc_start = "2021-02-12T22:01:00Z"
utc_end = "2021-02-19T23:00:00Z"
kind = "embargo"

[[window]]
id = "syn_test"
products = ["ZL"]
first_trade_date = "2021-01-11"
last_trade_date = "2021-01-15"
utc_start = "2021-01-08T22:01:00Z"
utc_end = "2021-01-15T23:00:00Z"
kind = "sealed test window"

[[window]]
id = "syn_decision"
products = ["ZM"]
first_trade_date = "2021-02-01"
last_trade_date = "2021-02-05"
utc_start = "2021-01-29T22:01:00Z"
utc_end = "2021-02-05T23:00:00Z"
kind = "closed by decision"
"""

STORED_FIELDS = [
    ("ts_event", pa.int64()),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.int64()),
    ("instrument_id", pa.int64()),
    ("raw_symbol", pa.large_string()),
    ("in_flatten_window", pa.bool_()),
    ("in_no_new_positions_window", pa.bool_()),
    ("early_halt_ct", pa.large_string()),
    ("in_scheduled_closure", pa.bool_()),
    ("trade_date", pa.large_string()),
    ("is_roll_session", pa.bool_()),
    ("gap_before_minutes", pa.int64()),
    ("vendor_degraded_day", pa.bool_()),
]


@dataclass(frozen=True)
class Row:
    """One synthetic bar: open instant (UTC ns), label, price in ticks (or an explicit raw price)."""

    ts_ns: int
    label: str | None
    price_ticks: int
    raw_price: float | None = None


def ct_ns(day: date, clock: time) -> int:
    return datetime_to_ns(datetime.combine(day, clock, tzinfo=CT))


def weekdays(first: date, last: date) -> list[date]:
    days = (first + timedelta(days=i) for i in range((last - first).days + 1))
    return [d for d in days if d.weekday() < 5]


def session_rows(first: date, last: date, start_ticks: int = BASE_TICKS) -> list[Row]:
    rows: list[Row] = []
    for n, day in enumerate(weekdays(first, last)):
        for k, (offset, clock) in enumerate(BAR_TIMES):
            when = day + timedelta(days=offset)
            rows.append(Row(ct_ns(when, clock), day.isoformat(), start_ticks + 3 * n + k))
    return rows


def write_store(path: Path, rows: list[Row], tick_fixed: int, meta_tick_fixed: int | None = None) -> Path:
    """Write a 16-column bar file; prices are price_ticks * tick_fixed / 1e9 (float64, as upstream stores
    them)."""
    prices = [r.price_ticks * tick_fixed / 1e9 if r.raw_price is None else r.raw_price for r in rows]
    n = len(rows)
    columns = {
        "ts_event": [r.ts_ns for r in rows],
        "open": prices,
        "high": prices,
        "low": prices,
        "close": prices,
        "volume": [1] * n,
        "instrument_id": [7] * n,
        "raw_symbol": ["SYNH1"] * n,
        "in_flatten_window": [False] * n,
        "in_no_new_positions_window": [False] * n,
        "early_halt_ct": [""] * n,
        "in_scheduled_closure": [False] * n,
        "trade_date": [r.label for r in rows],
        "is_roll_session": [False] * n,
        "gap_before_minutes": [0] * n,
        "vendor_degraded_day": [False] * n,
    }
    metadata = None
    if meta_tick_fixed is not None:
        metadata = {STORE_METADATA_KEY: json.dumps({"tick": {"tick_fixed_1e9": meta_tick_fixed}}).encode()}
    schema = pa.schema(STORED_FIELDS, metadata=metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns, schema=schema), path)
    return path


def research_name(root: str) -> str:
    return f"ohlcv-1m_{root}_v_0_2025-03-03_2026-05-15_research.parquet"


def step2_name(root: str) -> str:
    return f"ohlcv-1m_{root}_v_0_2019-01-02_2022-01-31_step2.parquet"


def hist_name(root: str, plan: str) -> str:
    return f"ohlcv-1m_{root}_v_0_2010-06-07_2018-12-31_{plan}.parquet"


MES_CONFIRMATION_NAME = "ohlcv-1m_MES_v_0_2019-01-02_2022-01-31_confirmation.parquet"


class _ExactTick:
    """Sentinel: write the root's exchange tick (vendor tick / price factor) into the file's metadata."""


EXACT_TICK = _ExactTick()
MetaTick = int | None | _ExactTick


def exchange_tick(root: str) -> int:
    return TICKS_FIXED[root] // PRICE_FACTORS[root]


def _meta_tick(root: str, meta: MetaTick) -> int | None:
    """Every synthetic file carries the exact metadata tick unless a test asks for None or a value."""
    return exchange_tick(root) if isinstance(meta, _ExactTick) else meta


@dataclass
class SynthStores:
    """Three temporary store roots, a temporary window table, and helpers to build a gateway.

    gateway() goes through DataGateway.for_synthetic_stores, the explicit test factory: the public
    constructor accepts only the pinned closed-window table.
    """

    base: Path
    paths: StorePaths = field(init=False)
    windows: WindowTable = field(init=False)

    def __post_init__(self) -> None:
        self.paths = StorePaths(
            research=self.base / "research",
            step2=self.base / "step2",
            hist=self.base / "hist",
            upstream_root=self.base / "pe",
            book_table=self.base / "pe" / "costs.json",
        )
        for root in self.paths.store_roots().values():
            root.mkdir(parents=True, exist_ok=True)
        table = self.base / "closed_windows.toml"
        table.write_text(WINDOW_TOML, encoding="utf-8")
        self.windows = load_windows(table, hashlib.sha256(table.read_bytes()).hexdigest())

    def research(self, root: str, rows: list[Row], meta_tick_fixed: MetaTick = EXACT_TICK) -> Path:
        path = self.paths.research / root / research_name(root)
        return write_store(path, rows, TICKS_FIXED[root], _meta_tick(root, meta_tick_fixed))

    def step2(self, root: str, rows: list[Row], meta_tick_fixed: MetaTick = EXACT_TICK) -> Path:
        path = self.paths.step2 / root / step2_name(root)
        return write_store(path, rows, TICKS_FIXED[root], _meta_tick(root, meta_tick_fixed))

    def hist(
        self, root: str, rows: list[Row], plan: str = "plan_a", meta_tick_fixed: MetaTick = EXACT_TICK
    ) -> Path:
        path = self.paths.hist / plan / root / hist_name(root, plan)
        return write_store(path, rows, TICKS_FIXED[root], _meta_tick(root, meta_tick_fixed))

    def mes_confirmation(self, rows: list[Row]) -> Path:
        path = self.paths.research / "MES" / MES_CONFIRMATION_NAME
        return write_store(path, rows, TICKS_FIXED["MES"])  # no metadata tick, as on disk (ADR A14 h)

    def catalog(self, pins: dict[PinKey, str] | None = None, windows_sha256: str | None = None) -> Catalog:
        use = {} if pins is None else pins
        sha = self.windows.sha256 if windows_sha256 is None else windows_sha256
        return build_catalog(self.paths, sha, "2021-01-01T00:00:00Z", pins=use)

    def gateway(self, catalog: Catalog | None = None) -> DataGateway:
        use = self.catalog() if catalog is None else catalog
        return DataGateway.for_synthetic_stores(
            self.paths, self.windows, use, TICKS_FIXED.__getitem__, PRICE_FACTORS.__getitem__
        )
