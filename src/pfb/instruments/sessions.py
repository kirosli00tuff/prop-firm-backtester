"""Sessions API (ADR-001 section 4 as amended by A3 and A9): trade days, open intervals, session windows,
unsourced days and the must-be-flat intervals, all in integer nanoseconds.

A root maps to its group through the calendar files' own root lists (no instrument table needed). Trade days
are
int32 days since 1970-01-01, -1 where the calendar books nothing. The trade-day rule assigns trade dates
with the close minute going to the previous interval: a bar inside an open
interval takes its trade date; a bar in the first minute after an interval ends takes that interval's trade
date;
any other bar is booked by nothing (-1; upstream gives it the next interval's date). The must-be-flat rule is
upstream
rules/sessions.py session_state(...).must_be_flat, evaluated per CT calendar date; both eras use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pfb.instruments.calendar import (
    GROUPS,
    MINUTES_PER_DAY,
    NS_PER_MIN,
    SATURDAY,
    CalendarCoverageError,
    CalendarError,
    GroupCalendar,
    NotATradeDate,
    SessionModelError,
    ct_date_of,
    ct_ns,
    day_ordinal,
    load_calendar,
)
from pfb.instruments.table import UnknownRoot

FRIDAY, SUNDAY = 4, 6
SESSION_NAMES = ("rth", "globex")
NO_TRADE_DAY = -1
GAP_EXIT_LEAD_MIN = 2  # minutes before a session pause at which a position is exited

I64 = npt.NDArray[np.int64]
I32 = npt.NDArray[np.int32]


@dataclass(frozen=True)
class _Booked:
    """Every open interval of a group's whole coverage, sorted, with its trade day."""

    starts: I64
    ends: I64
    days: I32


@lru_cache(maxsize=4)
def _root_groups(directory: Path | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for group in GROUPS:
        for root in load_calendar(group, directory).roots:
            if root in out:
                raise CalendarError(f"root {root} is listed by two calendars ({out[root]}, {group})")
            out[root] = group
    return out


def group_of(root: str, directory: Path | None = None) -> str:
    try:
        return _root_groups(directory)[root]
    except KeyError:
        raise UnknownRoot(f"no calendar lists root {root!r}") from None


def calendar_of(root: str, directory: Path | None = None) -> GroupCalendar:
    return load_calendar(group_of(root, directory), directory)


def _days(first: date, last: date) -> list[date]:
    return [first + timedelta(days=n) for n in range((last - first).days + 1)]


@lru_cache(maxsize=16)
def _booked_group(group: str, directory: Path | None) -> _Booked:
    cal = load_calendar(group, directory)
    rows: list[tuple[int, int, int]] = []
    for day in _days(cal.first, cal.last):
        if cal.books(day):
            rows.extend((lo, hi, day_ordinal(day)) for lo, hi in cal.session_intervals(day))
    rows.sort(key=lambda r: r[0])
    for (_s0, e0, d0), (s1, _e1, d1) in zip(rows, rows[1:], strict=False):
        if s1 < e0:
            raise SessionModelError(f"{cal.group}: open intervals of trade days {d0} and {d1} overlap")
    return _Booked(
        np.array([r[0] for r in rows], dtype=np.int64),
        np.array([r[1] for r in rows], dtype=np.int64),
        np.array([r[2] for r in rows], dtype=np.int32),
    )


def trade_days_of(root: str, ts_ns: npt.ArrayLike, directory: Path | None = None) -> I32:
    """Trade day of each instant: its open interval's, the closing interval's in the minute after a close,
    else -1.
    An instant before the first open or after the last close minute of the coverage raises."""
    booked = _booked_group(group_of(root, directory), directory)
    ts = np.asarray(ts_ns, dtype=np.int64)
    if ts.size and (
        int(ts.min()) < int(booked.starts[0]) or int(ts.max()) >= int(booked.ends[-1]) + NS_PER_MIN
    ):
        raise CalendarCoverageError(f"{root}: an instant lies outside the calendar coverage")
    idx = np.searchsorted(booked.starts, ts, side="right") - 1
    safe = np.clip(idx, 0, None)
    end = booked.ends[safe]
    booked_here = (idx >= 0) & (ts < end + NS_PER_MIN)
    return np.where(booked_here, booked.days[safe], NO_TRADE_DAY).astype(np.int32)


def open_intervals(root: str, first: date, last: date, directory: Path | None = None) -> tuple[I64, I64, I32]:
    """(starts_ns, ends_ns, trade_days) of every open interval of the trade dates first..last, sorted,
    half-open."""
    cal = calendar_of(root, directory)
    cal.check_coverage(first, last)
    booked = _booked_group(cal.group, directory)
    keep = (booked.days >= day_ordinal(first)) & (booked.days <= day_ordinal(last))
    return booked.starts[keep], booked.ends[keep], booked.days[keep]


def session_window_ns(
    root: str, trade_date: date, name: str, directory: Path | None = None
) -> tuple[int, int]:
    """"globex": from the trade date's first open to its last close. "rth": the group's day session (O, C)
    for the
    root on that trade date, clipped to the globex window."""
    if name not in SESSION_NAMES:
        raise ValueError(f"session name {name!r} is not one of {SESSION_NAMES}")
    cal = calendar_of(root, directory)
    cal.check_coverage(trade_date, trade_date)
    if not cal.books(trade_date):
        raise NotATradeDate(f"{root}: {trade_date} is not a trade date")
    intervals = cal.session_intervals(trade_date)
    start, end = intervals[0][0], intervals[-1][1]
    if name == "globex":
        return start, end
    open_min, close_min = cal.era_for(trade_date).day_session(root, trade_date)
    return max(start, ct_ns(trade_date, open_min)), min(end, ct_ns(trade_date, close_min))


def unsourced_days(root: str, directory: Path | None = None) -> I32:
    """Trade days upstream marks unsourced (hist era; every reader drops them), ascending."""
    cal = calendar_of(root, directory)
    days = sorted(d for era in cal.eras for d in era.unsourced)
    return np.array([day_ordinal(d) for d in days], dtype=np.int32)


# ------------------------------------------------------------------ flat-by ----
def flatten_minute(cal: GroupCalendar, day: date) -> int | None:
    """The author's research repo day_rule: F on CT calendar date ``day`` in CT minutes, None when closed all
    day."""
    era, rule = cal.era_for(day), cal.flat_by
    if day in era.full_closures:
        return None
    has_topstep_row = day in rule.topstep_close_by
    close_by = rule.topstep_close_by.get(day)
    if has_topstep_row and close_by is None:
        return None
    candidates = [rule.regular_min]
    halt = era.early_halts.get(day)
    if halt is not None:
        candidates.append(halt - rule.early_close_lead_min)
        lead = rule.topstep_lead_by_year.get(day.year)
        if lead is not None and not has_topstep_row:
            candidates.append(halt - lead)
    if close_by is not None:
        candidates.append(close_by)
    return min(candidates)


def _next_weekday(day: date) -> date:
    nxt = day + timedelta(days=1)
    while nxt.weekday() >= SATURDAY:
        nxt += timedelta(days=1)
    return nxt


def _evening_closed(cal: GroupCalendar, day: date) -> bool:
    """The evening segment on CT date ``day`` belongs to the next weekday; closed if that is closed or opens
    late."""
    nxt = _next_weekday(day)
    return flatten_minute(cal, nxt) is None or nxt in cal.era_for(nxt).late_opens


def _livestock_closed(cal: GroupCalendar, day: date, open_min: int) -> list[tuple[int, int]]:
    flat = flatten_minute(cal, day)
    if day.weekday() >= SATURDAY or flat is None:
        return [(0, MINUTES_PER_DAY)]
    return [(0, open_min), (flat, MINUTES_PER_DAY)]


def _day_part_closed(cal: GroupCalendar, day: date, reopen: int) -> list[tuple[int, int]]:
    flat = flatten_minute(cal, day)
    if flat is None:
        return [(0, reopen)]
    out = [(flat, reopen)]
    late = cal.era_for(day).late_opens.get(day)
    if late is not None:
        out.append((0, late))
    if cal.flat_by.pause is not None:
        out.append(cal.flat_by.pause)
    return out


def closed_minutes(cal: GroupCalendar, day: date) -> list[tuple[int, int]]:
    """CT minute ranges [a, b) of calendar date ``day`` in which upstream session_state says must_be_flat."""
    rule = cal.flat_by
    if rule.reopen_min is None:
        if rule.day_open_min is None:
            raise CalendarError(f"{cal.group}: a group without a reopen needs a day open")
        return _livestock_closed(cal, day, rule.day_open_min)
    reopen, weekday = rule.reopen_min, day.weekday()
    if weekday == SATURDAY:
        return [(0, MINUTES_PER_DAY)]
    out = [(0, reopen)] if weekday == SUNDAY else _day_part_closed(cal, day, reopen)
    if weekday == FRIDAY or _evening_closed(cal, day):
        out.append((reopen, MINUTES_PER_DAY))
    return [(a, b) for a, b in out if a < b]


def _merge(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for lo, hi in sorted(windows):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def hist_gap_windows_ns(cal: GroupCalendar, first: date, last: date) -> list[tuple[int, int]]:
    """For each hist-era trade date in first..last (before the last era), every
    gap between two of its open intervals is must-be-flat from GAP_EXIT_LEAD_MIN minutes before the gap to the
    next open, in addition to upstream's rule. Upstream's grain pause (07:43 to the 08:30 open) is this rule
    for the
    2019-on sessions; the hist sessions have other gaps (grains 07:15-09:30 CT before 2012-05-21)."""
    lead = GAP_EXIT_LEAD_MIN * NS_PER_MIN
    out: list[tuple[int, int]] = []
    for day in _days(first, last):
        if cal.era_for(day) is cal.eras[-1] or not cal.books(day):
            continue
        intervals = cal.session_intervals(day)
        out.extend(
            (hi - lead, nxt)
            for (_lo, hi), (nxt, _h) in zip(intervals, intervals[1:], strict=False)
            if nxt > hi
        )
    return out


def must_be_flat_intervals_ns(
    root: str, first: date, last: date, directory: Path | None = None
) -> tuple[I64, I64]:
    """Half-open, sorted, merged (starts_ns, ends_ns): every instant at which the author's research repo
    requires
    a flat account, plus (hist era only) every same-date session gap from 2
    minutes before it, over the CT calendar dates from the first session open of trade date ``first`` (or
    the day before it) through ``last``, clipped to that span."""
    cal = calendar_of(root, directory)
    cal.check_coverage(first, last)
    starts, _ends, _days_arr = open_intervals(root, first, last, directory)
    span_first = first - timedelta(days=1)
    if starts.size:
        span_first = min(span_first, ct_date_of(int(starts[0])))
    lo, hi = ct_ns(span_first, 0), ct_ns(last + timedelta(days=1), 0)
    windows: list[tuple[int, int]] = []
    for day in _days(span_first, last):
        windows.extend((ct_ns(day, a), ct_ns(day, b)) for a, b in closed_minutes(cal, day))
    windows.extend(
        hist_gap_windows_ns(cal, span_first, last + timedelta(days=1))
    )  # gaps of the next date too
    clipped = [(max(a, lo), min(b, hi)) for a, b in _merge(windows) if b > lo and a < hi]
    return (
        np.array([w[0] for w in clipped], dtype=np.int64),
        np.array([w[1] for w in clipped], dtype=np.int64),
    )


__all__ = [
    "GAP_EXIT_LEAD_MIN", "NO_TRADE_DAY", "SESSION_NAMES", "calendar_of", "closed_minutes", "flatten_minute",
    "group_of", "hist_gap_windows_ns", "must_be_flat_intervals_ns", "open_intervals", "session_window_ns",
    "trade_days_of", "unsourced_days",
]  # fmt: skip
