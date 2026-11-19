"""Fictional cost tables for the demo: a per-bucket book table and a release calendar.

Nothing here is calibrated on any market. The book table gives every admissible root of
config/instruments.toml the same made-up buckets of the CT day (half-spread plus depth, in ticks, exact
binary fractions so that side = half-spread + depth holds exactly); the minutes 16:00-17:00 CT lie in no
bucket, so the R-B2 fallback (the root's largest side ticks) is exercised.

The release calendar is a made-up schedule over CALENDAR_FIRST..CALENDAR_LAST: a daily report every weekday
at 13:30 UTC and a mid-month report (flagged cpi) on the second Tuesday at 12:30 UTC that concern every root,
and a weekly Wednesday report at 14:30 UTC that concerns the energy roots.

Both are deterministic: `python -m pfb.costs.demo_tables` rewrites config/demo/book_costs.json and
config/demo/release_calendar.json byte for byte; config/costs.toml pins their sha256.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from pfb.instruments import InstrumentTable, load_instruments

REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = REPO_ROOT / "config" / "demo"
BOOK_FILE = "book_costs.json"
CALENDAR_FILE = "release_calendar.json"
NOTE = "fictional values for the demo; not calibrated on any market"
RELEASE_CALENDAR_SCHEMA = "pfb.release_calendar/1"
CALENDAR_FIRST = date(2025, 1, 1)
CALENDAR_LAST = date(2026, 6, 30)
ENERGY_GROUP = "energy"
TUESDAY, WEDNESDAY, SATURDAY = 1, 2, 5

# (start minute, end minute, half-spread ticks, depth ticks buy, depth ticks sell), CT clock, [start, end)
BOOK_BUCKETS: tuple[tuple[int, int, float, float, float], ...] = (
    (0, 480, 0.5, 0.25, 0.25),  # overnight
    (480, 510, 0.5, 0.125, 0.25),  # before the day open
    (510, 540, 0.5, 0.5, 0.5),  # the first half hour
    (540, 900, 0.5, 0.0, 0.0),  # the day session
    (900, 960, 0.5, 0.25, 0.125),  # the close
    (1020, 1440, 0.75, 0.25, 0.25),  # the evening; 960..1019 lies in no bucket (R-B2 fallback)
)


def _bucket(start: int, end: int, half: float, buy: float, sell: float) -> dict[str, Any]:
    return {
        "key": f"{start // 60:02d}:{start % 60:02d}",
        "start_min": start,
        "end_min": end,
        "half_spread_ticks": half,
        "depth_ticks": {"buy": buy, "sell": sell},
        "side_ticks": {"buy": half + buy, "sell": half + sell},
        "fallback": False,
    }


def book_table(instruments: InstrumentTable) -> dict[str, Any]:
    """The fictional book table: every admissible root, the same buckets."""
    buckets = [_bucket(*row) for row in BOOK_BUCKETS]
    products = {
        root: {
            "product": root,
            "calibrated": True,
            "tick_value_usd": float(instruments.get(root).tick_value_usd),
            "buckets": buckets,
        }
        for root in instruments.roots()
        if instruments.get(root).admissible
    }
    return {"note": NOTE, "depth_term": "applied", "products": products}


def _nth_weekday(year: int, month: int, weekday: int, nth: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (nth - 1))


def _months(first: date, last: date) -> list[tuple[int, int]]:
    out = []
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        out.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def _release(kind: str, day: date, clock: str, products: Sequence[str], cpi: bool) -> dict[str, Any]:
    return {
        "id": f"{kind}-{day.isoformat()}",
        "release": kind,
        "instant_utc": f"{day.isoformat()}T{clock}:00Z",
        "products": list(products),
        "cpi": cpi,
    }


def release_calendar(
    instruments: InstrumentTable, first: date = CALENDAR_FIRST, last: date = CALENDAR_LAST
) -> dict[str, Any]:
    """The fictional release calendar over first..last (coverage), sorted by instant."""
    every = [r for r in instruments.roots() if instruments.get(r).admissible]
    energy = [r for r in every if instruments.get(r).group == ENERGY_GROUP]
    rows: list[dict[str, Any]] = []
    for year, month in _months(first, last):
        rows.append(_release("midmonth", _nth_weekday(year, month, TUESDAY, 2), "12:30", every, True))
    day = first
    while day <= last:
        if day.weekday() < SATURDAY:
            rows.append(_release("daily", day, "13:30", every, False))
        day += timedelta(days=1)
    day = first + timedelta(days=(WEDNESDAY - first.weekday()) % 7)
    while day <= last:
        rows.append(_release("weekly", day, "14:30", energy, False))
        day += timedelta(days=7)
    rows = sorted((r for r in rows if first <= date.fromisoformat(r["instant_utc"][:10]) <= last),
                  key=lambda r: (r["instant_utc"], r["id"]))  # fmt: skip
    coverage = {"first": first.isoformat(), "last": last.isoformat()}
    return {"schema": RELEASE_CALENDAR_SCHEMA, "note": NOTE, "coverage": coverage, "releases": rows}


def _compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(", ", ": "))


def _member(key: str, value: Any) -> str:
    """One top-level member; a list or a mapping of objects gets one element per line."""
    if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
        inner = ",\n".join(f"  {_compact(v)}" for v in value)
        return f"{json.dumps(key)}: [\n{inner}\n ]"
    if isinstance(value, dict) and value and all(isinstance(v, dict) for v in value.values()):
        inner = ",\n".join(f"  {json.dumps(k)}: {_compact(v)}" for k, v in sorted(value.items()))
        return f"{json.dumps(key)}: {{\n{inner}\n }}"
    return f"{json.dumps(key)}: {_compact(value)}"


def dumps(obj: dict[str, Any]) -> bytes:
    """The exact bytes written: sorted keys, one release or product per line, one trailing newline."""
    body = ",\n".join(f" {_member(k, obj[k])}" for k in sorted(obj))
    return ("{\n" + body + "\n}\n").encode("utf-8")


def write(out_dir: Path = DEMO_DIR, instruments: InstrumentTable | None = None) -> dict[str, Path]:
    """Write both tables; returns {file name: path}."""
    table = instruments or load_instruments()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, obj in ((BOOK_FILE, book_table(table)), (CALENDAR_FILE, release_calendar(table))):
        path = out_dir / name
        path.write_bytes(dumps(obj))
        written[name] = path
    return written


if __name__ == "__main__":
    for path in write(Path(sys.argv[1]) if len(sys.argv) > 1 else DEMO_DIR).values():
        print(path)
