"""Group calendars ported from upstream as data (config/calendars/<group>.json, ADR-001 section 4, A3).

One file per group holds its eras (hist2010 for trade dates before 2019-05-01, cme2019 from it), each with the
CME session regimes, full closures, early halts, late opens and crypto's booked-forward and 24/7 rules, plus
the
group's flat-by rule. This module loads and validates a file and answers the session questions for
one trade date (is it a trade date, its open intervals), in integer nanoseconds. Clock times are minutes of
the
America/Chicago day; a CT wall-clock instant becomes UTC nanoseconds through zoneinfo's offset, never a float.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[3]
CALENDAR_DIR = REPO_ROOT / "config" / "calendars"
CALENDAR_SCHEMA = "pfb.calendar/1"
GROUPS = ("equity", "rates", "fx", "energy", "metals", "grains", "livestock", "crypto")
CT = ZoneInfo("America/Chicago")
EPOCH = date(1970, 1, 1)
NS_PER_S = 1_000_000_000
NS_PER_MIN = 60 * NS_PER_S
MINUTES_PER_DAY = 1440
SATURDAY = 5
PREVIOUS_TRADE_DATE_MAX_BACK = 10  # days searched back for the previous trade date


class CalendarError(Exception):
    """A calendar question that cannot be answered."""


class CalendarFileError(CalendarError):
    """A calendar file is missing or malformed."""


class CalendarCoverageError(CalendarError):
    """A date or instant outside the calendar's coverage."""


class SessionModelError(CalendarError):
    """The session tables give overlapping intervals or no single regime for a trade date."""


class NotATradeDate(CalendarError):
    """The date is covered but the calendar books no session on it."""


def day_ordinal(day: date) -> int:
    """Days since 1970-01-01 (numpy datetime64[D] as an int; ADR A9 trade_day)."""
    return day.toordinal() - EPOCH.toordinal()


def from_day_ordinal(ordinal: int) -> date:
    return date.fromordinal(ordinal + EPOCH.toordinal())


@lru_cache(maxsize=1 << 16)
def ct_ns(day: date, minute: int) -> int:
    """UTC nanoseconds of CT wall-clock ``minute`` (0..1440) of calendar date ``day``."""
    if not 0 <= minute <= MINUTES_PER_DAY:
        raise ValueError(f"minute {minute} outside 0..1440")
    if minute == MINUTES_PER_DAY:
        return ct_ns(day + timedelta(days=1), 0)
    local = datetime(day.year, day.month, day.day, minute // 60, minute % 60, tzinfo=CT)
    offset = local.utcoffset()
    if offset is None:  # pragma: no cover - zoneinfo always gives an offset
        raise CalendarError("no UTC offset for America/Chicago")
    seconds = day_ordinal(day) * 86_400 + minute * 60 - offset.days * 86_400 - offset.seconds
    return seconds * NS_PER_S


def ct_date_of(ts_ns: int) -> date:
    """The CT calendar date of an instant (integer seconds; no float)."""
    return datetime.fromtimestamp(ts_ns // NS_PER_S, tz=CT).date()


@dataclass(frozen=True)
class Segment:
    """[day + start_offset start_min, day + end_offset end_min) in CT wall clock (upstream data/calendars
    Segment)."""

    start_offset: int
    start_min: int
    end_offset: int
    end_min: int


@dataclass(frozen=True)
class SessionSpec:
    valid_from: date
    valid_to: date | None
    segments: tuple[Segment, ...]
    day_session: Mapping[str, tuple[int, int]]

    def holds(self, day: date) -> bool:
        return self.valid_from <= day and (self.valid_to is None or day <= self.valid_to)


@dataclass(frozen=True)
class Era:
    """One frozen calendar of a group (upstream GroupCalendar or HistGroupCalendar) over trade dates
    first..last."""

    name: str
    first: date
    last: date
    sessions: tuple[SessionSpec, ...]
    full_closures: frozenset[date]
    early_halts: Mapping[date, int]
    late_opens: Mapping[date, int]  # scheduled late opens only (upstream); outages are not encoded
    booked_forward: Mapping[date, date]
    after_weekend_segments: tuple[Segment, ...]
    delayed_starts: Mapping[date, tuple[int, int]]
    weekend_from: date | None
    unsourced: frozenset[date]
    subgroup_of_root: Mapping[str, str]

    def spec_for(self, day: date) -> SessionSpec:
        found = [s for s in self.sessions if s.holds(day)]
        if not found:
            booked = [b for b, d in self.booked_forward.items() if d == day]
            if booked:
                return self.spec_for(max(booked))
        if len(found) != 1:
            raise SessionModelError(f"{self.name}: {len(found)} session regimes cover {day}")
        return found[0]

    def is_trade_date(self, day: date) -> bool:
        return day.weekday() < SATURDAY and day not in self.full_closures and day not in self.booked_forward

    def covers(self, day: date) -> bool:
        return self.first <= day <= self.last and any(s.holds(day) for s in self.sessions)

    def booked_target_covered(self, day: date) -> bool:
        return any(d == day and self.covers(b) for b, d in self.booked_forward.items())

    def day_session(self, root: str, day: date) -> tuple[int, int]:
        """(open, close) CT minutes, keys looked up in order: root, subgroup, "*", one value."""
        table = self.spec_for(day).day_session
        sub = self.subgroup_of_root.get(root)
        for key in (root, sub, "*"):
            if key is not None and key in table:
                return table[key]
        if len(set(table.values())) == 1:
            return next(iter(table.values()))
        raise CalendarError(f"{self.name}: no day session for {root} on {day}")


@dataclass(frozen=True)
class FlatByRule:
    """The flat-by rule of one group, as CT minutes; the Topstep rows apply to every group by date."""

    regular_min: int
    reopen_min: int | None  # None: livestock, no evening segment
    day_open_min: int | None  # livestock: nothing before 08:30 CT
    pause: tuple[int, int] | None  # grains: flat from 07:43 to the 08:30 CT open
    early_close_lead_min: int
    topstep_close_by: Mapping[date, int | None]  # None: markets closed
    topstep_lead_by_year: Mapping[int, int]


@dataclass(frozen=True)
class GroupCalendar:
    group: str
    roots: tuple[str, ...]
    eras: tuple[Era, ...]
    flat_by: FlatByRule
    sha256: str
    path: str

    def era_for(self, day: date) -> Era:
        """The era whose first date is the latest one on or before ``day`` (upstream
        GroupCalendars.cal_for)."""
        chosen = self.eras[0]
        for era in self.eras:
            if era.first <= day:
                chosen = era
        return chosen

    @property
    def first(self) -> date:
        return self.eras[0].first

    @property
    def last(self) -> date:
        era = self.eras[-1]
        targets = [d for b, d in era.booked_forward.items() if era.covers(b) and d > era.last]
        return max([era.last, *targets])

    def check_coverage(self, first: date, last: date) -> None:
        if first > last:
            raise CalendarCoverageError(f"{self.group}: first {first} is after last {last}")
        if first < self.first or last > self.last:
            raise CalendarCoverageError(
                f"{self.group}: {first}..{last} outside the calendar coverage {self.first}..{self.last}"
            )

    def books(self, day: date) -> bool:
        """``day`` is a covered trade date (upstream open_intervals: covers or a covered booked-forward
        target)."""
        era = self.era_for(day)
        return era.is_trade_date(day) and (era.covers(day) or era.booked_target_covered(day))

    def session_intervals(self, day: date) -> list[tuple[int, int]]:
        """The session intervals of a booked trade date: segments, delayed start, scheduled late open,
        early-halt cut."""
        era = self.era_for(day)
        if not era.is_trade_date(day):
            return []
        out = _base_intervals(era, day)
        delayed = era.delayed_starts.get(day)
        if delayed is not None:
            start = ct_ns(day + timedelta(days=delayed[0]), delayed[1])
            out = [(max(lo, start), hi) for lo, hi in out if hi > start]
        late = era.late_opens.get(day)
        if late is not None:
            start = ct_ns(day, late)
            out = [(max(lo, start), hi) for lo, hi in out if hi > start]
        halt = era.early_halts.get(day)
        if halt is not None:
            cut = ct_ns(day, halt)
            out = [(lo, min(hi, cut)) for lo, hi in out if lo < cut]
        return out


def _segment_ns(day: date, seg: Segment) -> tuple[int, int]:
    return (
        ct_ns(day + timedelta(days=seg.start_offset), seg.start_min),
        ct_ns(day + timedelta(days=seg.end_offset), seg.end_min),
    )


def _regular_intervals(era: Era, day: date) -> list[tuple[int, int]]:
    return [_segment_ns(day, s) for s in era.spec_for(day).segments]


def _previous_trade_date(era: Era, day: date) -> date | None:
    for n in range(1, PREVIOUS_TRADE_DATE_MAX_BACK + 1):
        if era.is_trade_date(day - timedelta(days=n)):
            return day - timedelta(days=n)
    return None


def _base_intervals(era: Era, day: date) -> list[tuple[int, int]]:
    """Upstream _base_intervals: crypto's weekend assignment from weekend_from, else the regular segments
    preceded
    by
    the regular sessions of the booked-forward days whose trade date is ``day``."""
    if era.weekend_from is not None and day >= era.weekend_from:
        prev = _previous_trade_date(era, day)
        if prev is not None and (day - prev).days > 1:
            return _anchored_span(era, day, prev)
        return _regular_intervals(era, day)
    out: list[tuple[int, int]] = []
    for booked in sorted(b for b, d in era.booked_forward.items() if d == day):
        out.extend(_regular_intervals(era, booked))
    out.extend(_regular_intervals(era, day))
    return out


def _weekly_gaps(era: Era, day: date, prev: date) -> list[tuple[int, int]]:
    tpl = era.after_weekend_segments
    gaps: list[tuple[int, int]] = []
    if len(tpl) < 2:
        return gaps
    span = [prev + timedelta(days=n) for n in range((day - prev).days + 1)]
    for a, b in zip(tpl, tpl[1:], strict=False):
        weekday = a.end_offset % 7  # the template is a Monday trade date
        for d in span:
            if d.weekday() == weekday:
                gaps.append(
                    (
                        ct_ns(d, a.end_min),
                        ct_ns(d + timedelta(days=b.start_offset - a.end_offset), b.start_min),
                    )
                )
    return gaps


def _anchored_span(era: Era, day: date, prev: date) -> list[tuple[int, int]]:
    segs = era.spec_for(day).segments
    if len(segs) != 1:
        raise SessionModelError(f"{era.name}: weekend assignment needs one segment on {day}")
    lo = ct_ns(prev, segs[0].start_min)
    hi = ct_ns(day + timedelta(days=segs[0].end_offset), segs[0].end_min)
    out: list[tuple[int, int]] = []
    cursor = lo
    for g0, g1 in sorted(_weekly_gaps(era, day, prev)):
        if g0 > cursor:
            out.append((cursor, min(g0, hi)))
        cursor = max(cursor, g1)
    if cursor < hi:
        out.append((cursor, hi))
    tpl = era.after_weekend_segments
    if tpl and (day - prev).days == 3 and day.weekday() == 0 and out != [_segment_ns(day, t) for t in tpl]:
        raise SessionModelError(f"{era.name}: the anchored span of {day} differs from the 24/7 template")
    return out


# ------------------------------------------------------------------ parsing ----
def _fail(where: str, message: str) -> CalendarFileError:
    return CalendarFileError(f"{where}: {message}")


def _date(value: Any, where: str) -> date:
    if not isinstance(value, str):
        raise _fail(where, f"{value!r} is not a YYYY-MM-DD string")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise _fail(where, f"{value!r} is not a date") from exc


def _minute(value: Any, where: str) -> int:
    if not isinstance(value, str) or len(value) != 5 or value[2] != ":":
        raise _fail(where, f"{value!r} is not HH:MM")
    hours, minutes = value[:2], value[3:]
    if not (hours.isdigit() and minutes.isdigit()) or int(hours) > 23 or int(minutes) > 59:
        raise _fail(where, f"{value!r} is not a valid HH:MM")
    return int(hours) * 60 + int(minutes)


def _count(value: Any, where: str, name: str) -> int:
    """A non-negative integer as the file holds it: 5.9, True and "5" are refused (A item 17, N2)."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _fail(where, f"{name} {value!r} is not a non-negative integer")
    return value


def _year(value: Any, where: str) -> int:
    if not isinstance(value, str) or len(value) != 4 or not (value.isascii() and value.isdigit()):
        raise _fail(where, f"{value!r} is not a four-digit year")
    return int(value)


def _offset(value: Any, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not -7 <= value <= 0:
        raise _fail(where, f"day offset {value!r} is not an integer in -7..0")
    return value


def _segment(raw: Any, where: str) -> Segment:
    if not isinstance(raw, list) or len(raw) != 4:
        raise _fail(where, "a segment is [start_offset, start HH:MM, end_offset, end HH:MM]")
    return Segment(
        _offset(raw[0], where), _minute(raw[1], where), _offset(raw[2], where), _minute(raw[3], where)
    )


def _session(raw: dict[str, Any], where: str) -> SessionSpec:
    valid_to = raw.get("valid_to")
    segments = tuple(_segment(s, where) for s in raw["segments"])
    if not segments:
        raise _fail(where, "a session regime has no segment")
    table = {str(k): (_minute(v[0], where), _minute(v[1], where)) for k, v in raw["day_session_ct"].items()}
    return SessionSpec(
        _date(raw["valid_from"], where), None if valid_to is None else _date(valid_to, where), segments, table
    )


def _era(raw: dict[str, Any], where: str) -> Era:
    here = f"{where} era {raw.get('name')!r}"
    weekend_from = raw.get("weekend_to_next_trade_date_from")
    return Era(
        name=str(raw["name"]),
        first=_date(raw["first"], here),
        last=_date(raw["last"], here),
        sessions=tuple(_session(s, here) for s in raw["sessions"]),
        full_closures=frozenset(_date(d, here) for d in raw["full_closures"]),
        early_halts={_date(d, here): _minute(t, here) for d, t in raw["early_halts"]},
        late_opens={
            _date(d, here): _minute(t, here) for d, t, scheduled in raw["late_opens"] if scheduled is True
        },
        booked_forward={_date(b, here): _date(d, here) for b, d in raw["booked_forward"]},
        after_weekend_segments=tuple(_segment(s, here) for s in raw["after_weekend_segments"]),
        delayed_starts={
            _date(d, here): (_offset(o, here), _minute(t, here)) for d, o, t in raw["delayed_starts"]
        },
        weekend_from=None if weekend_from is None else _date(weekend_from, here),
        unsourced=frozenset(_date(d, here) for d in raw["unsourced"]),
        subgroup_of_root={str(k): str(v) for k, v in raw["subgroup_of_root"].items()},
    )


def _optional_minute(value: Any, where: str) -> int | None:
    return None if value is None else _minute(value, where)


def _flat_by(raw: dict[str, Any], where: str) -> FlatByRule:
    here = f"{where} flat_by"
    pause = raw["pause_ct"]
    return FlatByRule(
        regular_min=_minute(raw["regular_flatten_ct"], here),
        reopen_min=_optional_minute(raw["reopen_ct"], here),
        day_open_min=_optional_minute(raw["day_open_ct"], here),
        pause=None if pause is None else (_minute(pause[0], here), _minute(pause[1], here)),
        early_close_lead_min=_count(raw["early_close_lead_min"], here, "early_close_lead_min"),
        topstep_close_by={_date(d, here): _optional_minute(t, here) for d, t in raw["topstep_close_by"]},
        topstep_lead_by_year={
            _year(y, here): _count(m, here, f"topstep_lead_min_by_year[{y}]")
            for y, m in raw["topstep_lead_min_by_year"].items()
        },
    )


def _check_eras(eras: tuple[Era, ...], where: str) -> None:
    if not eras:
        raise _fail(where, "no era")
    for era in eras:
        if era.first > era.last:
            raise _fail(where, f"era {era.name} ends before it starts")
    for a, b in zip(eras, eras[1:], strict=False):
        if a.last >= b.first:
            raise _fail(where, f"eras {a.name} and {b.name} overlap or are out of order")


def parse_calendar(raw: bytes, path: Path, group: str | None = None) -> GroupCalendar:
    """Validate one pfb.calendar/1 file. Any missing key or malformed value raises CalendarFileError."""
    where = path.name
    try:
        payload = json.loads(raw)
        if payload.get("schema") != CALENDAR_SCHEMA:
            raise _fail(where, f"schema {payload.get('schema')!r} is not {CALENDAR_SCHEMA!r}")
        if group is not None and payload.get("group") != group:
            raise _fail(where, f"group {payload.get('group')!r} is not {group!r}")
        eras = tuple(_era(e, where) for e in payload["eras"])
        flat_by = _flat_by(payload["flat_by"], where)
        roots = tuple(str(r) for r in payload["roots"])
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, ValueError, AttributeError) as exc:
        if isinstance(exc, CalendarFileError):
            raise
        raise _fail(where, f"malformed calendar file ({type(exc).__name__}: {exc})") from exc
    _check_eras(eras, where)
    if payload["group"] not in GROUPS or not roots:
        raise _fail(where, f"unknown group {payload['group']!r} or no roots")
    return GroupCalendar(
        group=str(payload["group"]), roots=roots, eras=eras, flat_by=flat_by,
        sha256=hashlib.sha256(raw).hexdigest(), path=str(path),
    )  # fmt: skip


@lru_cache(maxsize=64)
def load_calendar(group: str, directory: Path | None = None) -> GroupCalendar:
    """The calendar of ``group`` from ``<directory>/<group>.json`` (default config/calendars)."""
    if group not in GROUPS:
        raise CalendarFileError(f"unknown group {group!r} (groups {list(GROUPS)})")
    path = (directory or CALENDAR_DIR) / f"{group}.json"
    if not path.is_file():
        raise CalendarFileError(f"{path} does not exist")
    return parse_calendar(path.read_bytes(), path, group)


__all__ = [
    "CALENDAR_DIR", "CALENDAR_SCHEMA", "CT", "GROUPS", "NS_PER_MIN", "CalendarCoverageError", "CalendarError",
    "CalendarFileError", "Era", "FlatByRule", "GroupCalendar", "NotATradeDate", "Segment",
    "SessionModelError",
    "SessionSpec", "ct_date_of", "ct_ns", "day_ordinal", "from_day_ordinal", "load_calendar",
    "parse_calendar",
]  # fmt: skip
