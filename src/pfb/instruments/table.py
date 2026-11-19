"""The instrument table: config/instruments.toml, one row per upstream root.

ADR-001 sections 3 and 4, A5(d), A9.

Prices are in exchange units; the bars arrive in vendor units, ``vendor_price_factor`` x exchange units
(factor 100 for the six roots the vendor quotes in US cents). Money fields are exact Decimal USD; the derived
values are whole integers or an exact Fraction, never a float. This module imports nothing from pfb.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INSTRUMENTS_TOML = REPO_ROOT / "config" / "instruments.toml"
INSTRUMENTS_SCHEMA = "pfb.instruments/1"

PRICE_SCALE = 1_000_000_000  # vendor fixed-point prices: 1e-9 units
CENTI_CENTS_PER_USD = 10_000  # tick_value_cc: 1e-4 USD units
CENTS_PER_USD = 100

_REQUIRED_FIELDS = (
    "root", "group", "subgroup", "exposure", "contract_type", "tick_size", "tick_value_usd",
    "multiplier_usd", "vendor_price_factor", "admissible", "tick_source", "weight_source",
)  # fmt: skip


class InstrumentTableError(ValueError):
    """config/instruments.toml is missing, malformed or holds a value that breaks the table's rules."""


class UnknownRoot(LookupError):
    """The root is not in the instrument table."""


@dataclass(frozen=True)
class Instrument:
    """One contract. Decimal fields come from decimal STRINGS in the file; derived values are exact."""

    root: str
    group: str
    subgroup: str
    exposure: str
    contract_type: str
    tick_size: Decimal
    tick_value_usd: Decimal
    multiplier_usd: Decimal
    vendor_price_factor: int
    admissible: bool
    tick_source: str
    weight_source: str

    @property
    def vendor_tick_fixed(self) -> int:
        """The tick in vendor fixed-point units (1e-9): tick_size x vendor_price_factor x 1e9, whole."""
        value = self.tick_size * self.vendor_price_factor * PRICE_SCALE
        if value != value.to_integral_value():
            raise InstrumentTableError(f"{self.root}: vendor tick {value} is not whole 1e-9 units")
        return int(value)

    @property
    def tick_value_cc(self) -> int:
        """The tick value in 1e-4 USD, whole (the kernel's exact gross P&L unit)."""
        value = self.tick_value_usd * CENTI_CENTS_PER_USD
        if value != value.to_integral_value():
            raise InstrumentTableError(f"{self.root}: tick value {self.tick_value_usd} is not whole 1e-4 USD")
        return int(value)

    @property
    def tick_value_cents(self) -> Fraction:
        """The tick value in cents, exact (781.25 for ZT, 62.5 for M6B)."""
        return Fraction(self.tick_value_usd) * CENTS_PER_USD


class InstrumentTable:
    """All instruments in file order, with the file's sha256."""

    def __init__(self, instruments: tuple[Instrument, ...], sha256: str) -> None:
        self._by_root = {i.root: i for i in instruments}
        self._roots = tuple(i.root for i in instruments)
        self.sha256 = sha256

    def get(self, root: str) -> Instrument:
        try:
            return self._by_root[root]
        except KeyError:
            raise UnknownRoot(root) from None

    def roots(self) -> tuple[str, ...]:
        return self._roots

    def tick_fixed_of(self, root: str) -> int:
        return self.get(root).vendor_tick_fixed

    def price_factor_of(self, root: str) -> int:
        return self.get(root).vendor_price_factor


def _decimal(row: dict[str, Any], field: str, where: str) -> Decimal:
    raw = row[field]
    if not isinstance(raw, str):
        raise InstrumentTableError(f"{where}: {field} must be a decimal string, got {type(raw).__name__}")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        raise InstrumentTableError(f"{where}: {field} {raw!r} is not a decimal") from None
    if not value.is_finite() or value <= 0:
        raise InstrumentTableError(f"{where}: {field} {raw!r} must be finite and positive")
    return value


def _string(row: dict[str, Any], field: str, where: str) -> str:
    raw = row[field]
    if not isinstance(raw, str) or not raw:
        raise InstrumentTableError(f"{where}: {field} must be a non-empty string")
    return raw


def _parse_row(row: dict[str, Any], index: int) -> Instrument:
    where = f"instrument #{index}"
    for field in _REQUIRED_FIELDS:
        if field not in row:
            raise InstrumentTableError(f"{where}: missing field {field}")
    factor, admissible = row["vendor_price_factor"], row["admissible"]
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise InstrumentTableError(f"{where}: vendor_price_factor must be an int >= 1, got {factor!r}")
    if not isinstance(admissible, bool):
        raise InstrumentTableError(f"{where}: admissible must be a bool")
    instrument = Instrument(
        root=_string(row, "root", where),
        group=_string(row, "group", where),
        subgroup=_string(row, "subgroup", where),
        exposure=_string(row, "exposure", where),
        contract_type=_string(row, "contract_type", where),
        tick_size=_decimal(row, "tick_size", where),
        tick_value_usd=_decimal(row, "tick_value_usd", where),
        multiplier_usd=_decimal(row, "multiplier_usd", where),
        vendor_price_factor=factor,
        admissible=admissible,
        tick_source=_string(row, "tick_source", where),
        weight_source=_string(row, "weight_source", where),
    )
    _ = instrument.vendor_tick_fixed  # raises when the tick is not whole vendor units
    _ = instrument.tick_value_cc  # raises when the tick value is not whole 1e-4 USD
    return instrument


def load_instruments(path: Path | str | None = None) -> InstrumentTable:
    """Load and validate the table; ``path`` defaults to config/instruments.toml."""
    source = Path(path) if path is not None else DEFAULT_INSTRUMENTS_TOML
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise InstrumentTableError(f"cannot read {source}: {exc}") from exc
    try:
        table = tomllib.loads(data.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise InstrumentTableError(f"{source} is not valid TOML: {exc}") from exc
    if table.get("schema") != INSTRUMENTS_SCHEMA:
        found = table.get("schema")
        raise InstrumentTableError(f"{source}: schema must be {INSTRUMENTS_SCHEMA!r}, got {found!r}")
    rows = table.get("instrument")
    if not isinstance(rows, list) or not rows:
        raise InstrumentTableError(f"{source}: no [[instrument]] rows")
    instruments = tuple(_parse_row(row, i) for i, row in enumerate(rows))
    roots = [i.root for i in instruments]
    if len(set(roots)) != len(roots):
        dupes = sorted({r for r in roots if roots.count(r) > 1})
        raise InstrumentTableError(f"{source}: duplicate roots {dupes}")
    return InstrumentTable(instruments, hashlib.sha256(data).hexdigest())
