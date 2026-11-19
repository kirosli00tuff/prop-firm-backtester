"""Release calendar and event-window costs.

The calendar is a JSON file (schema "pfb.release_calendar/1") named by config/costs.toml [events] calendar
(relative to the repository root) or by config/paths.toml [costs] release_calendar, and checked against the
sha256, release count and coverage pinned in config/costs.toml [events]. The repository ships a fictional one
(pfb.costs.demo_tables).

Semantics (ported):
- ReleaseCalendar.in_event_window: a fill at UTC instant t on root r is in an event window when the latest
  release concerning r at or before t is less than 30 minutes before t, i.e. t in [release, release + 30 min).
  Only the fill's own root counts (and, for a micro, its sibling's; below). t is the fill bar's OPEN.
- release_calendar_from_dict / load_release_calendar: schema check, a timezone-aware instant on a whole
  minute, per-root sorted distinct instants, the CPI instants, coverage.
- The cost of a fill inside a window: CostModel.event_slippage_cents (pfb.costs.model).

Not ported (not cost): a fill guard [release, release + 2 min), a CPI opening window, and a clock-time
fallback for dates before the calendar. The price-limit exit's forced event cost is the engine's
(pfb.engine.run, engine 0.3.1: the cost model's event table on every PRICE_LIMIT exit, "off" and "on").
A fill whose trade date lies outside the calendar's coverage is refused, never charged as "no event".

Sibling union: a micro root is charged by the UNION of its own release instants and its full-size sibling's
(EVENT_SIBLINGS), so a micro whose own list is shorter is never charged less than its sibling. The calendar
itself is kept as read (ReleaseCalendar.instants); the union lives in EventCostModel. The mini-sized QM and QG
are in the union too (QM with CL, QG with NG).
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from pfb.costs.model import (
    COSTS_SCHEMA,
    DEFAULT_COSTS_TOML,
    MINUTES_PER_DAY,
    REPO_ROOT,
    SIDES,
    BookCostModel,
    BoolRow,
    CostModel,
    CostModelError,
    Int64Table,
    StressCostModel,
    _config_sha256,
    load_cost_model,
)
from pfb.instruments import InstrumentTable

RELEASE_CALENDAR_SCHEMA = "pfb.release_calendar/1"
EVENT_WINDOW_MINUTES = 30  # [release, release + 30 min)
NS_PER_SECOND = 1_000_000_000
NS_PER_MINUTE = 60 * NS_PER_SECOND
EVENT_WINDOW_NS = EVENT_WINDOW_MINUTES * NS_PER_MINUTE
EVENT_MODEL_SUFFIX = "+events"
EVENT_COSTS_OFF = "off"
EVENT_COSTS_ON = "on"
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
EPOCH_DATE = date(1970, 1, 1)
MICROSECOND = timedelta(microseconds=1)
NS_PER_MICROSECOND = 1000
# Each micro (and SIL) root and its full-size sibling; the micro is charged by the union of both release
# lists. config/instruments.toml's exposure groups these pairs (tested). The mini-sized QM and QG, not
# micros, join their full-size siblings. The pairs are part of the "+events" config sha256: a change here
# moves every "on" cost identity.
EVENT_SIBLINGS: Mapping[str, str] = MappingProxyType({
    "MNQ": "NQ", "MES": "ES", "MYM": "YM", "M2K": "RTY", "MCL": "CL", "MNG": "NG", "MGC": "GC",
    "SIL": "SI", "MHG": "HG", "M6E": "6E", "M6B": "6B", "M6A": "6A",
    "QM": "CL",
    "QG": "NG",
})  # fmt: skip


class ReleaseCalendarError(CostModelError):
    """The release calendar is missing, differs from its pin, or is malformed."""


class EventCoverageError(ReleaseCalendarError):
    """A window or a fill lies outside the calendar's coverage: its event minutes are unknown."""


class EventPaths(Protocol):
    """What an event cost model needs from config/paths.toml (pfb.data.paths.StorePaths satisfies it): the
    book table and the release calendar, each None for the file config/costs.toml names in the repository."""

    @property
    def book_table(self) -> Path | None: ...

    @property
    def release_calendar(self) -> Path | None: ...


# ------------------------------------------------------------------ config ----
@dataclass(frozen=True)
class EventsConfig:
    """config/costs.toml [events]: the pin of the release calendar."""

    calendar: str  # relative to the repository root
    sha256: str
    releases: int
    coverage_first: date
    coverage_last: date
    window_minutes: int


def _config_int(table: Mapping[str, Any], key: str, path: Path) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReleaseCalendarError(f"{path}: [events] {key} must be an int")
    return value


def _config_date(table: Mapping[str, Any], key: str, path: Path) -> date:
    value = table.get(key)
    if isinstance(value, datetime) or not isinstance(value, date):
        raise ReleaseCalendarError(f"{path}: [events] {key} must be a TOML date")
    return value


def load_events_config(path: Path = DEFAULT_COSTS_TOML) -> EventsConfig:
    """config/costs.toml -> its [events] table, checked."""
    if not path.is_file():
        raise ReleaseCalendarError(f"{path}: not found")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ReleaseCalendarError(f"{path}: not valid TOML: {exc}") from exc
    if raw.get("schema") != COSTS_SCHEMA:
        raise ReleaseCalendarError(f"{path}: schema must be {COSTS_SCHEMA!r}")
    table = raw.get("events")
    if not isinstance(table, dict):
        raise ReleaseCalendarError(f"{path}: no [events] table (the release calendar pin)")
    sha = table.get("sha256")
    if not isinstance(sha, str) or len(sha) != 64:
        raise ReleaseCalendarError(f"{path}: [events] sha256 must be a 64-character hex string")
    if table.get("schema") != RELEASE_CALENDAR_SCHEMA:
        raise ReleaseCalendarError(f"{path}: [events] schema must be {RELEASE_CALENDAR_SCHEMA!r}")
    calendar = table.get("calendar")
    if not isinstance(calendar, str) or not calendar:
        raise ReleaseCalendarError(f"{path}: [events] calendar must name the file")
    window = _config_int(table, "window_minutes", path)
    if window != EVENT_WINDOW_MINUTES:
        raise ReleaseCalendarError(f"{path}: [events] window_minutes {window} is not 30")
    return EventsConfig(
        calendar, sha, _config_int(table, "releases", path),
        _config_date(table, "coverage_first", path), _config_date(table, "coverage_last", path), window,
    )  # fmt: skip


# ------------------------------------------------------------------ calendar ----
def _frozen_ints(values: Sequence[int]) -> NDArray[np.int64]:
    array = np.asarray(values, dtype=np.int64)
    array.setflags(write=False)
    return array


def window_mask(instants: NDArray[np.int64], ts_ns: NDArray[np.int64]) -> BoolRow:
    """bool[n]: ts in [r, r + 30 min) of the latest r <= ts among ``instants`` (sorted, distinct)."""
    ts = np.asarray(ts_ns, dtype=np.int64)
    if instants.size == 0:
        return np.zeros(ts.shape, dtype=np.bool_)
    idx = np.searchsorted(instants, ts, side="right") - 1
    latest = instants[np.clip(idx, 0, None)]
    return (idx >= 0) & (ts < latest + np.int64(EVENT_WINDOW_NS))


@dataclass(frozen=True)
class ReleaseCalendar:
    """Scheduled releases concerning each root (UTC ns, sorted, distinct) and the CPI instants."""

    by_root: Mapping[str, NDArray[np.int64]]
    cpi: tuple[int, ...]
    first: date
    last: date
    sha256: str
    source: str
    releases: int

    def instants(self, root: str) -> NDArray[np.int64]:
        """The release instants concerning ``root`` (empty for a root no release concerns)."""
        return self.by_root.get(root, _frozen_ints([]))

    def in_event_window(self, root: str, ts_ns: int) -> bool:
        """ts in [r, r + 30 min) of the latest release r <= ts."""
        return bool(self.event_mask(root, np.asarray([ts_ns], dtype=np.int64))[0])

    def event_mask(self, root: str, ts_ns: NDArray[np.int64]) -> BoolRow:
        """bool[n]: in_event_window for every instant of ``ts_ns`` (UTC ns, any order)."""
        return window_mask(self.instants(root), ts_ns)

    def covers(self, first: date, last: date) -> bool:
        """The calendar covers the trade dates first..last."""
        return self.first <= first and last <= self.last

    def require_covers(self, first: date, last: date, what: str = "window") -> None:
        if first > last:
            raise EventCoverageError(f"{what} {first}..{last} is empty")
        if not self.covers(first, last):
            raise EventCoverageError(
                f"{what} {first}..{last} is outside the release calendar's coverage "
                f"{self.first}..{self.last}: its event minutes are unknown, so event costs cannot price it"
            )


def instant_ns(text: str, source: str = "release calendar") -> int:
    """An ISO instant ('Z' or an offset) -> UTC ns, exact; a naive instant is refused."""
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ReleaseCalendarError(f"{source}: release instant {text!r} is not ISO 8601") from exc
    if dt.tzinfo is None:
        raise ReleaseCalendarError(f"{source}: release instant {text!r} is not timezone-aware")
    return ((dt - EPOCH) // MICROSECOND) * NS_PER_MICROSECOND


def _root_names(entry: Mapping[str, Any], source: str) -> list[str]:
    products = entry.get("products")
    if not isinstance(products, list) or not all(isinstance(p, str) and p for p in products):
        raise ReleaseCalendarError(f"{source}: {entry.get('id')} products must be a list of root names")
    return products


def release_calendar_from_dict(raw: Mapping[str, Any], sha256: str, source: str) -> ReleaseCalendar:
    """The calendar of a parsed JSON object, checked."""
    if raw.get("schema") != RELEASE_CALENDAR_SCHEMA:
        schema = raw.get("schema")
        raise ReleaseCalendarError(f"{source}: schema {schema!r} is not {RELEASE_CALENDAR_SCHEMA!r}")
    entries = raw.get("releases")
    if not isinstance(entries, list):
        raise ReleaseCalendarError(f"{source}: no releases list")
    by_root: dict[str, set[int]] = {}
    cpi: set[int] = set()
    for entry in entries:
        ns = instant_ns(str(entry.get("instant_utc")), source)
        if ns % NS_PER_MINUTE:
            raise ReleaseCalendarError(f"{source}: {entry.get('id')} is not on a minute")
        for root in _root_names(entry, source):
            by_root.setdefault(root, set()).add(ns)
        if entry.get("cpi", False):
            cpi.add(ns)
    cov = raw.get("coverage")
    if not isinstance(cov, dict):
        raise ReleaseCalendarError(f"{source}: no coverage")
    return ReleaseCalendar(
        MappingProxyType({r: _frozen_ints(sorted(v)) for r, v in sorted(by_root.items())}),
        tuple(sorted(cpi)), date.fromisoformat(str(cov["first"])), date.fromisoformat(str(cov["last"])),
        sha256, source, len(entries),
    )  # fmt: skip


def load_release_calendar(path: Path, config: EventsConfig) -> ReleaseCalendar:
    """Read the calendar in place, refuse it unless its bytes, count and coverage equal the pin."""
    if not path.is_file():
        raise ReleaseCalendarError(f"{path}: release calendar not found")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != config.sha256:
        raise ReleaseCalendarError(f"{path}: sha256 {digest[:12]}... is not the pinned {config.sha256[:12]}")
    calendar = release_calendar_from_dict(json.loads(data.decode("utf-8")), digest, config.calendar)
    if calendar.releases != config.releases:
        raise ReleaseCalendarError(f"{path}: {calendar.releases} releases, the pin says {config.releases}")
    if (calendar.first, calendar.last) != (config.coverage_first, config.coverage_last):
        got = f"{calendar.first}..{calendar.last}"
        raise ReleaseCalendarError(f"{path}: coverage {got} is not the pinned one")
    return calendar


def load_pinned_calendar(
    calendar_path: Path | None = None, costs_toml: Path = DEFAULT_COSTS_TOML
) -> ReleaseCalendar:
    """The pinned release calendar: ``calendar_path``, else the file config/costs.toml [events] names
    (relative to the repository root)."""
    config = load_events_config(costs_toml)
    path = Path(calendar_path) if calendar_path is not None else REPO_ROOT / config.calendar
    return load_release_calendar(path, config)


# ------------------------------------------------------------------ event cost model ----
def has_event_costs(model: CostModel) -> bool:
    """True for book and for stress over a book-based model: the models the event rule applies to."""
    if isinstance(model, BookCostModel):
        return True
    if isinstance(model, StressCostModel):
        return has_event_costs(model.base)
    return False


def _days(trade_date: NDArray[Any]) -> NDArray[np.int64]:
    arr = np.asarray(trade_date)
    if arr.dtype.kind == "M":
        return arr.astype("datetime64[D]").astype(np.int64)
    if arr.dtype.kind not in "iu":
        raise CostModelError("trade_date must be integer days since 1970-01-01 or datetime64")
    return arr.astype(np.int64)


def _union_instants(calendar: ReleaseCalendar, root: str) -> NDArray[np.int64]:
    """A micro (or QM, QG) root's own release instants united with its full-size sibling's."""
    own = calendar.instants(root)
    sibling = EVENT_SIBLINGS.get(root)
    if sibling is None:
        return own
    return _frozen_ints(np.union1d(own, calendar.instants(sibling)).tolist())


def _require_calendar_roots(base: CostModel, calendar: ReleaseCalendar) -> None:
    """Every root the model prices (and a micro's sibling) must be a root of the calendar;
    a root the calendar does not name would otherwise be charged as "no event" without a word."""
    wanted = set(base.roots()) | {EVENT_SIBLINGS[r] for r in base.roots() if r in EVENT_SIBLINGS}
    missing = sorted(wanted - set(calendar.by_root))
    if missing:
        raise ReleaseCalendarError(
            f"release calendar {calendar.source} names no release list for root(s) {missing} that cost model "
            f"{base.model_id!r} prices: their event minutes are unknown"
        )


class EventCostModel(CostModel):
    """A book-based model plus the release calendar: a market fill inside an event window pays the base
    model's event table at its CT minute, every other fill the base table. A distinct configuration: its
    model id is the base's plus '+events' and its config_sha256 covers the calendar's sha256, the window and
    the sibling pairs. Under cost stress the nesting is EventCostModel(Stress(book)); a StressCostModel over
    an EventCostModel is refused (pfb.costs.model.StressCostModel)."""

    carries_release_calendar = True

    def __init__(self, base: CostModel, calendar: ReleaseCalendar) -> None:
        if not has_event_costs(base):
            raise CostModelError(
                f'event_costs "on" needs a book-based cost model (book or stress:<book spec>:<k>), '
                f"not {base.model_id!r}: the event rule widens the book's half-spread"
            )
        _require_calendar_roots(base, calendar)
        model_id = base.model_id + EVENT_MODEL_SUFFIX
        super().__init__(
            model_id,
            _config_sha256({
                "schema": COSTS_SCHEMA, "model_id": model_id, "base_config_sha256": base.config_sha256,
                "release_calendar_sha256": calendar.sha256, "event_window_minutes": EVENT_WINDOW_MINUTES,
                "event_siblings": dict(EVENT_SIBLINGS),
            }),
        )  # fmt: skip
        self._base = base
        self._calendar = calendar
        self._instants = MappingProxyType({r: _union_instants(calendar, r) for r in base.roots()})

    @property
    def base(self) -> CostModel:
        return self._base

    @property
    def calendar(self) -> ReleaseCalendar:
        return self._calendar

    def roots(self) -> tuple[str, ...]:
        return self._base.roots()

    def calibrated(self, root: str) -> BoolRow:
        return self._base.calibrated(root)

    def _build(self, root: str, contracts: int) -> Int64Table:
        return self._base.slippage_cents(root, contracts)

    def _build_event(self, root: str, contracts: int) -> Int64Table:
        return self._base.event_slippage_cents(root, contracts)

    def event_instants(self, root: str) -> NDArray[np.int64]:
        """The release instants that charge ``root``: its own, united with its sibling's for a micro."""
        self._require(root)
        return self._instants[root]

    def event_bars(self, root: str, ts_ns: NDArray[np.int64]) -> BoolRow:
        """bool[n]: which fill instants (bar opens, UTC ns) lie in an event window of ``root`` (its own
        releases and, for a micro, its full-size sibling's)."""
        return window_mask(self.event_instants(root), ts_ns)

    def bar_slippage_cents(
        self,
        root: str,
        contracts: int,
        ts_ns: NDArray[np.int64],
        minute: NDArray[np.int64],
        trade_date: NDArray[Any],
    ) -> Int64Table:
        """int64[n, 2]: the cents a market fill of ``contracts`` at each bar pays (bar open ``ts_ns`` UTC ns,
        its CT minute, its trade date), side 0 buy and 1 sell. A trade date outside the calendar's coverage
        is refused."""
        ts = np.asarray(ts_ns, dtype=np.int64)
        mins = np.asarray(minute, dtype=np.int64)
        days = _days(trade_date)
        if ts.ndim != 1 or ts.shape != mins.shape or ts.shape != days.shape:
            raise CostModelError("ts_ns, minute and trade_date must be 1-D arrays of one length")
        if mins.size and (int(mins.min()) < 0 or int(mins.max()) >= MINUTES_PER_DAY):
            raise CostModelError("minute must lie in 0..1439")
        if days.size:
            first = EPOCH_DATE + timedelta(days=int(days.min()))
            last = EPOCH_DATE + timedelta(days=int(days.max()))
            self._calendar.require_covers(first, last, what="trade dates")
        normal = self.slippage_cents(root, contracts)[mins]
        event = self.event_slippage_cents(root, contracts)[mins]
        mask = self.event_bars(root, ts)
        return np.where(mask[:, None], event, normal).astype(np.int64).reshape(ts.size, len(SIDES))


def load_event_cost_model(
    spec: str, instruments: InstrumentTable, paths: EventPaths, *, costs_toml: Path = DEFAULT_COSTS_TOML
) -> EventCostModel:
    """The cost model of ``spec`` with event windows on (the release calendar read in place and pinned)."""
    base = load_cost_model(spec, instruments, paths, costs_toml=costs_toml)
    return EventCostModel(base, load_pinned_calendar(paths.release_calendar, costs_toml))


def load_cost_model_for(
    spec: str,
    event_costs: str,
    instruments: InstrumentTable,
    paths: EventPaths | None,
    *,
    costs_toml: Path = DEFAULT_COSTS_TOML,
) -> CostModel:
    """The model a hypothesis's [cost] declares: "off" is exactly load_cost_model; "on" wraps it."""
    if event_costs == EVENT_COSTS_OFF:
        return load_cost_model(spec, instruments, paths, costs_toml=costs_toml)
    if event_costs != EVENT_COSTS_ON:
        raise CostModelError(f'event_costs must be "off" or "on", not {event_costs!r}')
    if paths is None:
        raise CostModelError('event_costs "on" needs the store paths (the book table and release calendar)')
    return load_event_cost_model(spec, instruments, paths, costs_toml=costs_toml)
