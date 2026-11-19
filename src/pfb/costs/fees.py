"""Commissions per firm and root: config/fees/<firm>.toml (ADR-001 section 6, A7, A14 (f) G9).

One [[fee]] per root the firm allows, integer cents: commission_side_cents and commission_rt_cents, with the
source file and lookup date of the value. USD values convert as cents = Decimal(str(usd)) x 100, which
must be whole; a side derived from a round turn alone is ceil(rt_cents / 2). A root the file does not list
cannot run with that firm (UnknownFeeRoot).
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Any

from pfb.instruments import InstrumentTable

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FEES_DIR = REPO_ROOT / "config" / "fees"
FEES_SCHEMA = "pfb.fees/1"
DEFAULT_FIRM = "topstep"
CENTS_PER_USD = 100
_FIRM_RE = re.compile(r"[a-z][a-z0-9_]{0,31}")


class FeeConfigError(ValueError):
    """A fee file is missing or malformed, or a value breaks the cents rules."""


class UnknownFeeRoot(LookupError):
    """The firm's fee file does not list this root; the run is refused, never defaulted."""


def usd_to_cents(usd: str | int | float | Decimal) -> int:
    """Decimal(str(usd)) x 100, which must be whole (A14 G9)."""
    if isinstance(usd, bool):
        raise FeeConfigError("a fee value cannot be a bool")
    try:
        cents = Decimal(str(usd)) * CENTS_PER_USD
    except InvalidOperation:
        raise FeeConfigError(f"fee value {usd!r} is not a decimal number") from None
    if not cents.is_finite() or cents != cents.to_integral_value():
        raise FeeConfigError(f"fee value {usd!r} USD is not a whole number of cents")
    return int(cents)


def side_from_round_turn(rt_cents: int) -> int:
    """The per-side commission of a round turn quoted alone: ceil(rt / 2), so an odd cent goes UP."""
    return -(-rt_cents // 2)


@dataclass(frozen=True)
class FeeEntry:
    root: str
    commission_side_cents: int
    commission_rt_cents: int
    source_file: str
    lookup_date: date


@dataclass(frozen=True)
class FeeSchedule:
    firm: str
    lookup_date: date
    source_file: str
    source_sha256: str
    sha256: str
    entries: Mapping[str, FeeEntry]

    def roots(self) -> tuple[str, ...]:
        return tuple(self.entries)

    def entry(self, root: str) -> FeeEntry:
        try:
            return self.entries[root]
        except KeyError:
            raise UnknownFeeRoot(f"firm {self.firm!r} lists no commission for {root!r}") from None

    def commission_side_cents(self, root: str) -> int:
        return self.entry(root).commission_side_cents


def _cents_field(row: Mapping[str, Any], name: str, where: str) -> int:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise FeeConfigError(f"{where}: {name} must be an int (cents)")
    if value < 0:
        raise FeeConfigError(f"{where}: {name} must be >= 0")
    return value


def _date_field(row: Mapping[str, Any], name: str, where: str) -> date:
    value = row.get(name)
    if not isinstance(value, date) or hasattr(value, "hour"):
        raise FeeConfigError(f"{where}: {name} must be a TOML date (YYYY-MM-DD)")
    return value


def _text_field(row: Mapping[str, Any], name: str, where: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value:
        raise FeeConfigError(f"{where}: {name} must be a non-empty string")
    return value


def _parse_entry(row: Any, where: str, instruments: InstrumentTable | None) -> FeeEntry:
    if not isinstance(row, dict):
        raise FeeConfigError(f"{where}: must be a table")
    root = _text_field(row, "root", where)
    side = _cents_field(row, "commission_side_cents", where)
    rt = _cents_field(row, "commission_rt_cents", where)
    if side != side_from_round_turn(rt):
        raise FeeConfigError(f"{where} {root}: commission_side_cents {side} != ceil(rt {rt} / 2)")
    if instruments is not None and root not in instruments.roots():
        raise FeeConfigError(f"{where}: {root!r} is not in the instrument table")
    source_file = _text_field(row, "source_file", where)
    return FeeEntry(root, side, rt, source_file, _date_field(row, "lookup_date", where))


def _parse_schedule(
    raw: Mapping[str, Any], firm: str, path: Path, digest: str, instruments: InstrumentTable | None
) -> FeeSchedule:
    if raw.get("schema") != FEES_SCHEMA:
        raise FeeConfigError(f"{path}: schema must be {FEES_SCHEMA!r}")
    if raw.get("firm") != firm:
        raise FeeConfigError(f"{path}: firm {raw.get('firm')!r} does not match the file name")
    rows = raw.get("fee")
    if not isinstance(rows, list) or not rows:
        raise FeeConfigError(f"{path}: no [[fee]] rows")
    entries: dict[str, FeeEntry] = {}
    for index, row in enumerate(rows):
        entry = _parse_entry(row, f"{path} [[fee]] {index}", instruments)
        if entry.root in entries:
            raise FeeConfigError(f"{path}: root {entry.root} listed twice")
        entries[entry.root] = entry
    return FeeSchedule(
        firm=firm,
        lookup_date=_date_field(raw, "lookup_date", str(path)),
        source_file=_text_field(raw, "source_file", str(path)),
        source_sha256=_text_field(raw, "source_sha256", str(path)),
        sha256=digest,
        entries=MappingProxyType(entries),
    )


def load_fees(
    firm: str = DEFAULT_FIRM,
    *,
    fees_dir: Path = DEFAULT_FEES_DIR,
    instruments: InstrumentTable | None = None,
) -> FeeSchedule:
    """config/fees/<firm>.toml -> FeeSchedule. With ``instruments`` every root must be in that table."""
    if not isinstance(firm, str) or not _FIRM_RE.fullmatch(firm):
        raise FeeConfigError(f"firm name {firm!r} must match {_FIRM_RE.pattern}")
    path = fees_dir / f"{firm}.toml"
    if not path.is_file():
        raise FeeConfigError(f"{path}: no fee file for firm {firm!r}")
    data = path.read_bytes()
    try:
        raw = tomllib.loads(data.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise FeeConfigError(f"{path}: not valid TOML: {exc}") from exc
    return _parse_schedule(raw, firm, path, hashlib.sha256(data).hexdigest(), instruments)
