"""Slippage cost models: book, fixed and stress (ADR-001 section 6, amended by A7).

Every table is per MINUTE of the CT day: ``slippage_cents(root, contracts) -> int64[1440, 2]`` is the cents
one MARKET fill of ``contracts`` pays at that CT minute, side 0 buy and 1 sell. Commission is in fees.py.

book: a per-bucket table (config/costs.toml [book] table, or config/paths.toml [costs] book_table), checked
against the sha256 pinned in config/costs.toml; the repository ships a fictional one (pfb.costs.demo_tables).
Each bucket covers its own [start_min, end_min); a minute in no bucket takes the root's largest per-side
slippage and is uncalibrated. Cents are ceil(round(qty x ticks x tick value, 6)), the only
float in a money path, evaluated once when a table is built.
fixed:<k>: ceil(qty x k x tick_value_cents), exact rational arithmetic, k a decimal string >= 0.
stress:<base spec>:<k>: the base model plus ceil(qty x k x tick_value_cents) per side.
Event windows: ``event_slippage_cents`` is what a market fill pays inside a
release event window: the product's LARGEST half-spread over its buckets plus the fill minute's own bucket's
depth term, per side; an uncalibrated minute pays the R-B2 fallback, event or not. Stress adds its k ticks.
A fixed model has none and raises EventCostsOff. Which fills are inside a window: pfb.costs.events.
"""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from pfb.instruments import InstrumentTable

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COSTS_TOML = REPO_ROOT / "config" / "costs.toml"
COSTS_SCHEMA = "pfb.costs/1"
EVENT_WINDOWS = "off"

MINUTES_PER_DAY = 1440
SIDES = ("buy", "sell")  # column 0 buy, column 1 sell
EVENT_KEY = "event:"  # cache-key prefix of the event-window tables
ROUND_DIGITS = 6  # round(..., 6) strips float noise before the ceil
SIDE_SUM_TOLERANCE = 1e-12  # side_ticks must equal half_spread + depth to this tolerance
APPLIED_DEPTH_TERM = "applied"

SPEC_BOOK = "book"
PREFIX_FIXED = "fixed:"
PREFIX_STRESS = "stress:"

Int64Table = NDArray[np.int64]
BoolRow = NDArray[np.bool_]


class CostModelError(ValueError):
    """A cost spec, the costs config or the book table is invalid."""


class CostSpecError(CostModelError):
    """The cost model spec cannot be parsed."""


class CostTableError(CostModelError):
    """The book table is missing, differs from its pin or breaks a rule of the lookup."""


class EventCostsOff(CostModelError):
    """Event-window costs are off (the release calendar is not ported)."""


class UnknownCostRoot(LookupError):
    """The cost model has no row for this root; the run is refused, never defaulted."""


class CostPaths(Protocol):
    """What the book model needs from config/paths.toml (pfb.data.paths.StorePaths satisfies it): the book
    table's path, or None for the one config/costs.toml [book] table names inside the repository."""

    @property
    def book_table(self) -> Path | None: ...


def book_slippage_cents(qty: int, slip_ticks: float, tick_value_cents: Fraction) -> int:
    """Slippage of one fill in cents: ceil once per fill, the expression evaluated in this order."""
    return math.ceil(round(qty * slip_ticks * float(tick_value_cents), ROUND_DIGITS))


def exact_tick_cents(qty: int, ticks: Fraction, tick_value_cents: Fraction) -> int:
    """ceil(qty x ticks x tick value in cents), exact."""
    return math.ceil(qty * ticks * tick_value_cents)


def _check_contracts(contracts: int) -> int:
    if isinstance(contracts, bool) or not isinstance(contracts, int | np.integer):
        raise CostModelError(f"contracts must be an int, not {type(contracts).__name__}")
    if contracts < 1:
        raise CostModelError(f"contracts must be >= 1, not {contracts}")
    return int(contracts)


def _frozen(array: NDArray[Any]) -> NDArray[Any]:
    array.setflags(write=False)
    return array


def _config_sha256(payload: Mapping[str, object]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


class CostModel(ABC):
    """The interface the engine prices market fills with."""

    # True only for pfb.costs.events.EventCostModel: the model decides per fill (bar open) from a release
    # calendar, so a per-minute table alone does not describe what it charges
    carries_release_calendar: bool = False

    def __init__(self, model_id: str, config_sha256: str) -> None:
        self._model_id = model_id
        self._config_sha256 = config_sha256
        self._cache: dict[tuple[str, int], Int64Table] = {}

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def config_sha256(self) -> str:
        return self._config_sha256

    @abstractmethod
    def roots(self) -> tuple[str, ...]:
        """The roots this model can price."""

    @abstractmethod
    def calibrated(self, root: str) -> BoolRow:
        """bool[1440]: False where a minute pays the R-B2 fallback."""

    @abstractmethod
    def _build(self, root: str, contracts: int) -> Int64Table:
        """int64[1440, 2] for a root this model covers."""

    def _require(self, root: str) -> None:
        if root not in self.roots():
            raise UnknownCostRoot(f"cost model {self._model_id!r} has no row for {root!r}")

    def slippage_cents(self, root: str, contracts: int) -> Int64Table:
        """int64[1440, 2], read-only: cents one market fill of ``contracts`` pays per CT minute and side."""
        qty = _check_contracts(contracts)
        self._require(root)
        key = (root, qty)
        table = self._cache.get(key)
        if table is None:
            table = _frozen(self._build(root, qty))
            self._cache[key] = table
        return table

    def event_slippage_cents(self, root: str, contracts: int) -> Int64Table:
        """int64[1440, 2], read-only: cents one market fill pays per CT minute and side INSIDE a release
        event window (pfb.costs.events decides which fills are inside). Only a book-based model has one."""
        qty = _check_contracts(contracts)
        self._require(root)
        key = (EVENT_KEY + root, qty)
        table = self._cache.get(key)
        if table is None:
            table = _frozen(self._build_event(root, qty))
            self._cache[key] = table
        return table

    def _build_event(self, root: str, contracts: int) -> Int64Table:
        raise EventCostsOff(
            f"cost model {self._model_id!r} has no event-window cost (the event rule widens the book's "
            f"half-spread; only book and stress:<book spec>:<k> carry one) ({root})"
        )


class BookCostModel(CostModel):
    """A pinned per-bucket slippage table, expanded to minutes."""

    def __init__(
        self,
        slip_ticks: Mapping[str, NDArray[np.float64]],
        calibrated: Mapping[str, BoolRow],
        tick_value_cents: Mapping[str, Fraction],
        table_sha256: str,
        event_slip_ticks: Mapping[str, NDArray[np.float64]] | None = None,
    ) -> None:
        super().__init__(
            SPEC_BOOK,
            _config_sha256(
                {"schema": COSTS_SCHEMA, "model_id": SPEC_BOOK, "book_table_sha256": table_sha256}
            ),
        )
        self._slip = MappingProxyType({r: _frozen(a) for r, a in slip_ticks.items()})
        self._event_slip = MappingProxyType({r: _frozen(a) for r, a in (event_slip_ticks or {}).items()})
        self._calibrated = MappingProxyType({r: _frozen(a) for r, a in calibrated.items()})
        self._tv = MappingProxyType(dict(tick_value_cents))
        self._roots = tuple(sorted(self._slip))
        self.table_sha256 = table_sha256

    def roots(self) -> tuple[str, ...]:
        return self._roots

    def calibrated(self, root: str) -> BoolRow:
        self._require(root)
        return self._calibrated[root]

    def slip_ticks(self, root: str) -> NDArray[np.float64]:
        """float64[1440, 2], read-only: the table's slippage ticks per minute and side (R-B2 filled)."""
        self._require(root)
        return self._slip[root]

    def event_slip_ticks(self, root: str) -> NDArray[np.float64]:
        """float64[1440, 2], read-only: the event-window slippage ticks per minute and side."""
        self._require(root)
        if root not in self._event_slip:
            raise EventCostsOff(f"book model built without event-window ticks for {root!r}")
        return self._event_slip[root]

    def _build(self, root: str, contracts: int) -> Int64Table:
        return _cents_table(self._slip[root], contracts, self._tv[root])

    def _build_event(self, root: str, contracts: int) -> Int64Table:
        return _cents_table(self.event_slip_ticks(root), contracts, self._tv[root])


def _cents_table(slip: NDArray[np.float64], contracts: int, tv: Fraction) -> Int64Table:
    out = np.empty((MINUTES_PER_DAY, len(SIDES)), dtype=np.int64)
    for minute in range(MINUTES_PER_DAY):
        for side in range(len(SIDES)):
            out[minute, side] = book_slippage_cents(contracts, float(slip[minute, side]), tv)
    return out


class FixedCostModel(CostModel):
    """k ticks per side at every minute, every root of the instrument table."""

    def __init__(self, ticks: Fraction, ticks_text: str, instruments: InstrumentTable) -> None:
        model_id = PREFIX_FIXED + ticks_text
        super().__init__(model_id, _config_sha256({"schema": COSTS_SCHEMA, "model_id": model_id}))
        self._ticks = ticks
        self._instruments = instruments

    def roots(self) -> tuple[str, ...]:
        return self._instruments.roots()

    def calibrated(self, root: str) -> BoolRow:
        self._require(root)
        return _frozen(np.ones(MINUTES_PER_DAY, dtype=np.bool_))

    def _build(self, root: str, contracts: int) -> Int64Table:
        tv = self._instruments.get(root).tick_value_cents
        cents = exact_tick_cents(contracts, self._ticks, tv)
        return np.full((MINUTES_PER_DAY, len(SIDES)), cents, dtype=np.int64)


class StressCostModel(CostModel):
    """A base model plus k ticks per side (its own ceil), on the base model's roots."""

    def __init__(
        self, base: CostModel, ticks: Fraction, ticks_text: str, instruments: InstrumentTable
    ) -> None:
        if base.carries_release_calendar:
            # stress over an event model would keep only its minute tables and
            # lose the calendar without an error; the stressed event model is EventCostModel(Stress(base))
            raise CostModelError(
                f"stress over {base.model_id!r} is refused: build the stressed model as "
                "EventCostModel(stress:<base>:<k>, calendar), never stress over an event model"
            )
        model_id = f"{PREFIX_STRESS}{base.model_id}:{ticks_text}"
        super().__init__(
            model_id,
            _config_sha256(
                {"schema": COSTS_SCHEMA, "model_id": model_id, "base_config_sha256": base.config_sha256}
            ),
        )
        self._base = base
        self._ticks = ticks
        self._instruments = instruments

    @property
    def base(self) -> CostModel:
        return self._base

    def roots(self) -> tuple[str, ...]:
        return self._base.roots()

    def calibrated(self, root: str) -> BoolRow:
        return self._base.calibrated(root)

    def _build(self, root: str, contracts: int) -> Int64Table:
        tv = self._instruments.get(root).tick_value_cents
        extra = exact_tick_cents(contracts, self._ticks, tv)
        return self._base.slippage_cents(root, contracts) + np.int64(extra)

    def _build_event(self, root: str, contracts: int) -> Int64Table:
        tv = self._instruments.get(root).tick_value_cents
        extra = exact_tick_cents(contracts, self._ticks, tv)
        return self._base.event_slippage_cents(root, contracts) + np.int64(extra)


# ------------------------------------------------------------------ spec parsing ----
def parse_ticks(text: str) -> tuple[Fraction, str]:
    """A decimal string >= 0 -> (exact Fraction, canonical text)."""
    try:
        value = Decimal(text)
    except InvalidOperation:
        raise CostSpecError(f"ticks {text!r} is not a decimal number") from None
    if text != text.strip() or not value.is_finite():
        raise CostSpecError(f"ticks {text!r} is not a finite decimal number")
    if value < 0:
        raise CostSpecError(f"ticks {text!r} must be >= 0 (a negative cost would flatter a result)")
    return Fraction(value), format(value.normalize(), "f")


def _split_stress(rest: str) -> tuple[str, str]:
    base, sep, ticks = rest.rpartition(":")
    if not sep or not base or not ticks:
        raise CostSpecError(f"stress spec must be 'stress:<base spec>:<k>', not 'stress:{rest}'")
    return base, ticks


# ------------------------------------------------------------------ costs config and book table ----
def load_costs_config(path: Path = DEFAULT_COSTS_TOML) -> dict[str, Any]:
    """config/costs.toml -> its [book] table, checked."""
    if not path.is_file():
        raise CostModelError(f"{path}: not found")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise CostModelError(f"{path}: not valid TOML: {exc}") from exc
    if raw.get("schema") != COSTS_SCHEMA:
        raise CostModelError(f"{path}: schema must be {COSTS_SCHEMA!r}")
    book = raw.get("book")
    if not isinstance(book, dict) or not isinstance(book.get("sha256"), str) or len(book["sha256"]) != 64:
        raise CostModelError(f"{path}: [book] sha256 must be a 64-character hex string")
    if isinstance(book.get("products"), bool) or not isinstance(book.get("products"), int):
        raise CostModelError(f"{path}: [book] products must be an int")
    return book


def _read_pinned_table(table_path: Path, pinned_sha256: str) -> dict[str, Any]:
    if not table_path.is_file():
        raise CostTableError(f"{table_path}: book cost table not found")
    data = table_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != pinned_sha256:
        raise CostTableError(f"{table_path}: sha256 {digest[:12]}... is not the pinned {pinned_sha256[:12]}")
    raw = json.loads(data.decode("utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("products"), dict):
        raise CostTableError(f"{table_path}: no products table")
    if raw.get("depth_term") != APPLIED_DEPTH_TERM:
        raise CostTableError(f"{table_path}: depth_term is not {APPLIED_DEPTH_TERM!r} (a provisional table)")
    return raw


def _minute_field(root: str, bucket: Mapping[str, Any], name: str) -> int:
    value = bucket.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CostTableError(f"{root} bucket {bucket.get('key')}: {name} must be an int")
    return value


def _bucket_minutes(root: str, bucket: Mapping[str, Any]) -> tuple[int, int, tuple[float, float]]:
    key = bucket.get("key")
    if bucket.get("fallback"):
        raise CostTableError(f"{root} bucket {key}: a fallback bucket (not encoded)")
    start, end = _minute_field(root, bucket, "start_min"), _minute_field(root, bucket, "end_min")
    if not 0 <= start < end <= MINUTES_PER_DAY:
        raise CostTableError(f"{root} bucket {key}: [{start}, {end}) is not inside one CT day")
    half = float(bucket["half_spread_ticks"])
    sides: list[float] = []
    for side in SIDES:
        value = float(bucket["side_ticks"][side])
        depth = float(bucket["depth_ticks"][side])
        if not math.isfinite(value) or value < 0:
            raise CostTableError(f"{root} bucket {key}: side_ticks[{side}] {value} is not finite and >= 0")
        if abs(value - (half + depth)) > SIDE_SUM_TOLERANCE:
            raise CostTableError(f"{root} bucket {key}: side_ticks[{side}] != half-spread + depth")
        sides.append(value)
    return start, end, (sides[0], sides[1])


def _expand_root(root: str, entry: Mapping[str, Any]) -> tuple[NDArray[np.float64], BoolRow]:
    """Per-minute slippage ticks and the calibrated mask of one product (R-B2 fallback per side)."""
    if entry.get("product") != root or entry.get("calibrated") is not True:
        raise CostTableError(f"{root}: entry is not a calibrated product of that name")
    buckets = entry.get("buckets")
    if not isinstance(buckets, list) or not buckets:
        raise CostTableError(f"{root}: no buckets")
    slip = np.zeros((MINUTES_PER_DAY, len(SIDES)), dtype=np.float64)
    covered = np.zeros(MINUTES_PER_DAY, dtype=np.bool_)
    fallback = [0.0, 0.0]
    for bucket in buckets:
        start, end, sides = _bucket_minutes(root, bucket)
        if covered[start:end].any():
            raise CostTableError(f"{root} bucket {bucket.get('key')}: overlaps another bucket")
        covered[start:end] = True
        slip[start:end, :] = sides
        fallback = [max(fallback[i], sides[i]) for i in range(len(SIDES))]
    slip[~covered, :] = fallback
    return slip, covered


def _expand_event(root: str, entry: Mapping[str, Any], slip: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-minute EVENT slippage ticks of one product (buckets already checked by _expand_root): the largest
    half-spread of the product + the bucket's depth, summed in that order as floats; an uncalibrated minute
    keeps its R-B2 fallback (``slip``)."""
    buckets: list[Mapping[str, Any]] = entry["buckets"]
    max_half = max(float(b["half_spread_ticks"]) for b in buckets)
    event = slip.copy()
    for bucket in buckets:
        start, end = int(bucket["start_min"]), int(bucket["end_min"])
        for col, side in enumerate(SIDES):
            event[start:end, col] = max_half + float(bucket["depth_ticks"][side])
    return event


def _tick_value_cents(root: str, entry: Mapping[str, Any], instruments: InstrumentTable) -> Fraction:
    try:
        instrument = instruments.get(root)
    except LookupError as exc:
        raise CostTableError(f"{root}: in the book table but not in the instrument table") from exc
    if Decimal(str(entry.get("tick_value_usd"))) != instrument.tick_value_usd:
        raise CostTableError(f"{root}: book tick value differs from config/instruments.toml")
    return instrument.tick_value_cents


def book_table_path(paths: CostPaths, costs_toml: Path = DEFAULT_COSTS_TOML) -> Path:
    """paths.book_table, else the table config/costs.toml [book] names (relative to the repository root)."""
    if paths.book_table is not None:
        return Path(paths.book_table)
    table = load_costs_config(costs_toml).get("table")
    if not isinstance(table, str) or not table:
        raise CostModelError(f"{costs_toml}: [book] table must name the file (or set [costs] book_table)")
    return REPO_ROOT / table


def build_book_model(table_path: Path, instruments: InstrumentTable, costs_toml: Path) -> BookCostModel:
    """Read the pinned table in place and expand every product to minutes."""
    book = load_costs_config(costs_toml)
    raw = _read_pinned_table(table_path, book["sha256"])
    products: Mapping[str, Any] = raw["products"]
    if len(products) != book["products"]:
        pinned = book["products"]
        raise CostTableError(f"{table_path}: {len(products)} products, config/costs.toml pins {pinned}")
    slip: dict[str, NDArray[np.float64]] = {}
    event: dict[str, NDArray[np.float64]] = {}
    calibrated: dict[str, BoolRow] = {}
    tvs: dict[str, Fraction] = {}
    for root, entry in products.items():
        tvs[root] = _tick_value_cents(root, entry, instruments)
        slip[root], calibrated[root] = _expand_root(root, entry)
        event[root] = _expand_event(root, entry, slip[root])
    return BookCostModel(slip, calibrated, tvs, book["sha256"], event)


def load_cost_model(
    spec: str,
    instruments: InstrumentTable,
    paths: CostPaths | None,
    *,
    costs_toml: Path = DEFAULT_COSTS_TOML,
) -> CostModel:
    """'book', 'fixed:<k>' or 'stress:<base spec>:<k>' -> a CostModel. ``paths`` is needed for the book."""
    if not isinstance(spec, str) or not spec:
        raise CostSpecError("cost model spec must be a non-empty string")
    if spec == SPEC_BOOK:
        if paths is None:
            raise CostSpecError("the book model needs the store paths (pfb.data.paths.load_paths)")
        return build_book_model(book_table_path(paths, costs_toml), instruments, costs_toml)
    if spec.startswith(PREFIX_FIXED):
        ticks, text = parse_ticks(spec[len(PREFIX_FIXED) :])
        return FixedCostModel(ticks, text, instruments)
    if spec.startswith(PREFIX_STRESS):
        base_spec, ticks_text = _split_stress(spec[len(PREFIX_STRESS) :])
        ticks, text = parse_ticks(ticks_text)
        base = load_cost_model(base_spec, instruments, paths, costs_toml=costs_toml)
        return StressCostModel(base, ticks, text, instruments)
    raise CostSpecError(f"unknown cost model spec {spec!r} (book, fixed:<k>, stress:<base spec>:<k>)")
