"""Sessions API: hand-computed cases (both eras), then upstream's own functions on every booked minute
2019-05-01..2026-06-19.

Hand cases give the expected instants in UTC, worked out from the calendar entries named in each test (CT is
UTC-6 in winter, UTC-5 in summer).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime

import numpy as np
import pytest

from pfb.instruments import calendar as calmod
from pfb.instruments import sessions
from pfb.instruments.calendar import CalendarCoverageError, NotATradeDate, day_ordinal
from pfb.instruments.table import UnknownRoot

ONE_ROOT_PER_GROUP = ("NQ", "ZN", "6E", "CL", "GC", "ZC", "HE", "MBT")
STITCH = date(2019, 5, 1)


def utc(y: int, mo: int, d: int, h: int, mi: int) -> int:
    return int(datetime(y, mo, d, h, mi, tzinfo=UTC).timestamp()) * 1_000_000_000


def intervals(root: str, first: date, last: date) -> list[tuple[int, int]]:
    starts, ends, _ = sessions.open_intervals(root, first, last)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def flat(root: str, first: date, last: date) -> list[tuple[int, int]]:
    starts, ends = sessions.must_be_flat_intervals_ns(root, first, last)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


# ------------------------------------------------------------------ hist era, by hand ----
def test_hist_equity_split_session_has_two_intervals() -> None:
    # Arrange: regime 2012-11-19..2015-09-20, segments [-1 17:00, 0 15:15) and [0 15:30, 0 16:15) CT; CST.
    day = date(2014, 3, 3)

    # Act
    got = intervals("NQ", day, day)
    days = sessions.open_intervals("NQ", day, day)[2]

    # Assert
    assert got == [
        (utc(2014, 3, 2, 23, 0), utc(2014, 3, 3, 21, 15)),
        (utc(2014, 3, 3, 21, 30), utc(2014, 3, 3, 22, 15)),
    ]
    assert days.tolist() == [day_ordinal(day)] * 2


def test_hist_equity_regular_day_flat_from_1508_to_reopen() -> None:
    # Sunday before 17:00 CT is closed; Monday from F 15:08 to the 17:00 reopen (Tuesday's evening is open).
    got = flat("NQ", date(2014, 3, 3), date(2014, 3, 3))

    assert got == [
        (utc(2014, 3, 2, 6, 0), utc(2014, 3, 2, 23, 0)),
        (utc(2014, 3, 3, 21, 8), utc(2014, 3, 3, 23, 0)),
    ]


def test_hist_early_halt_cuts_session_and_moves_flat_by() -> None:
    # 2013-11-29 early halt 12:15 CT: the session ends at the halt (the 15:30 segment is gone); F = 12:15 -
    # 15 min.
    day = date(2013, 11, 29)

    assert intervals("NQ", day, day) == [(utc(2013, 11, 28, 23, 0), utc(2013, 11, 29, 18, 15))]
    assert flat("NQ", day, day)[-1] == (utc(2013, 11, 29, 18, 0), utc(2013, 11, 30, 6, 0))


def test_hist_2019_early_halt_uses_topstep_year_lead() -> None:
    # 2019-01-21 halt 12:00: candidates 15:08, 11:45 (15 min) and 11:30 (Topstep 2019 lead 30 min, no row).
    cal = sessions.calendar_of("NQ")

    assert sessions.flatten_minute(cal, date(2019, 1, 21)) == 11 * 60 + 30


def test_hist_grain_pause_and_flat_by() -> None:
    # Regime from 2015-07-06: [-1 19:00, 0 07:45) and [0 08:30, 0 13:20) CT; F 13:18, pause 07:43-08:30,
    # reopen 19:00.
    day = date(2016, 3, 1)

    assert intervals("ZC", day, day) == [
        (utc(2016, 3, 1, 1, 0), utc(2016, 3, 1, 13, 45)),
        (utc(2016, 3, 1, 14, 30), utc(2016, 3, 1, 19, 20)),
    ]
    assert flat("ZC", day, day) == [
        (utc(2016, 2, 29, 13, 43), utc(2016, 2, 29, 14, 30)),
        (utc(2016, 2, 29, 19, 18), utc(2016, 3, 1, 1, 0)),
        (utc(2016, 3, 1, 13, 43), utc(2016, 3, 1, 14, 30)),
        (utc(2016, 3, 1, 19, 18), utc(2016, 3, 2, 1, 0)),
    ]  # fmt: skip


def test_hist_2011_grain_gap_is_flat_from_0713_to_0930() -> None:
    # Arrange: regime 2010-06-01..2012-05-20, [-1 18:00, 0 07:15) and [0 09:30, 0 13:15) CT (CST, UTC-6).
    # The same-date gap is flat from 07:13 to the 09:30 open; upstream's 07:43-08:30 is
    # inside.
    day = date(2011, 3, 1)

    # Act
    got = flat("ZC", day, day)

    # Assert: the 18:00-19:00 evening stays flat under upstream's 19:00 reopen (inside [13:18, 19:00)).
    assert intervals("ZC", day, day) == [
        (utc(2011, 3, 1, 0, 0), utc(2011, 3, 1, 13, 15)),
        (utc(2011, 3, 1, 15, 30), utc(2011, 3, 1, 19, 15)),
    ]
    assert got == [
        (utc(2011, 2, 28, 13, 13), utc(2011, 2, 28, 15, 30)),
        (utc(2011, 2, 28, 19, 18), utc(2011, 3, 1, 1, 0)),
        (utc(2011, 3, 1, 13, 13), utc(2011, 3, 1, 15, 30)),
        (utc(2011, 3, 1, 19, 18), utc(2011, 3, 2, 1, 0)),
    ]  # fmt: skip


def test_hist_group_without_same_date_gap_is_unchanged() -> None:
    # Rates hist 2011-10-03 on: one segment [-1 17:00, 0 16:00): no gap, so only upstream's rule applies.
    day = date(2014, 3, 3)
    cal = sessions.calendar_of("ZN")

    got = flat("ZN", day, day)

    assert sessions.hist_gap_windows_ns(cal, date(2014, 3, 2), date(2014, 3, 4)) == []
    assert got == [
        (utc(2014, 3, 2, 6, 0), utc(2014, 3, 2, 23, 0)),
        (utc(2014, 3, 3, 21, 8), utc(2014, 3, 3, 23, 0)),
    ]


def test_cme2019_era_has_no_gap_addition() -> None:
    cal = sessions.calendar_of("ZC")

    assert sessions.hist_gap_windows_ns(cal, date(2019, 5, 1), date(2019, 5, 31)) == []
    assert sessions.hist_gap_windows_ns(cal, date(2019, 4, 29), date(2019, 4, 30)) != []


def test_hist_unsourced_livestock_days() -> None:
    got = sessions.unsourced_days("HE")

    assert len(got) == 25
    assert got[0] == day_ordinal(date(2013, 1, 21))
    assert sessions.unsourced_days("NQ").size == 0


# ------------------------------------------------------------------ 2019-on era, by hand ----
def test_grain_late_open_removes_evening_and_morning() -> None:
    # 2019-07-04 full closure; 2019-07-05 scheduled late open 08:30 CT (CDT): only the day segment trades.
    day = date(2019, 7, 5)

    assert intervals("ZC", day, day) == [(utc(2019, 7, 5, 13, 30), utc(2019, 7, 5, 18, 20))]
    assert flat("ZC", day, day) == [
        (utc(2019, 7, 4, 5, 0), utc(2019, 7, 5, 13, 30)),
        (utc(2019, 7, 5, 18, 18), utc(2019, 7, 6, 5, 0)),
    ]


def test_early_halt_with_topstep_row_takes_the_earliest() -> None:
    # 2025-11-27 halt 12:00, Topstep close-by 11:30; 2025-11-28 halt 12:15, Topstep 11:45 (CST).
    cal = sessions.calendar_of("NQ")
    day = date(2025, 11, 28)

    assert sessions.flatten_minute(cal, date(2025, 11, 27)) == 11 * 60 + 30
    assert sessions.flatten_minute(cal, day) == 11 * 60 + 45
    assert intervals("NQ", day, day) == [(utc(2025, 11, 27, 23, 0), utc(2025, 11, 28, 18, 15))]
    assert flat("NQ", day, day) == [
        (utc(2025, 11, 27, 17, 30), utc(2025, 11, 27, 23, 0)),
        (utc(2025, 11, 28, 17, 45), utc(2025, 11, 29, 6, 0)),
    ]


def test_dst_changes_shift_utc_but_not_ct_clock() -> None:
    spring, autumn = date(2025, 3, 10), date(2025, 11, 3)

    assert intervals("NQ", spring, spring) == [(utc(2025, 3, 9, 22, 0), utc(2025, 3, 10, 21, 0))]
    assert intervals("NQ", autumn, autumn) == [(utc(2025, 11, 2, 23, 0), utc(2025, 11, 3, 22, 0))]
    assert (utc(2025, 3, 10, 20, 8), utc(2025, 3, 10, 22, 0)) in flat("NQ", spring, spring)


def test_trade_days_inside_close_minute_and_unbooked() -> None:
    # Arrange: around the 2025-03-10 session (opens Sunday 22:00Z, closes Monday 21:00Z).
    ts = [
        utc(2025, 3, 9, 21, 59),
        utc(2025, 3, 9, 22, 0),
        utc(2025, 3, 10, 20, 59),
        utc(2025, 3, 10, 21, 0),
        utc(2025, 3, 10, 21, 1),
    ]
    want = [
        -1,
        day_ordinal(date(2025, 3, 10)),
        day_ordinal(date(2025, 3, 10)),
        day_ordinal(date(2025, 3, 10)),
        -1,
    ]

    # Act / Assert
    assert sessions.trade_days_of("NQ", np.array(ts, dtype=np.int64)).tolist() == want
    assert sessions.trade_days_of("NQ", np.array(ts, dtype=np.int64)).dtype == np.int32


def test_crypto_weekend_assignment_and_booked_target() -> None:
    # From 2026-06-01 a Monday runs from Friday 16:02 CT, less Saturday 02:00-04:00 CT maintenance (CDT).
    day = date(2026, 6, 8)
    cal = sessions.calendar_of("MBT")

    assert intervals("MBT", day, day) == [
        (utc(2026, 6, 5, 21, 2), utc(2026, 6, 6, 7, 0)),
        (utc(2026, 6, 6, 9, 0), utc(2026, 6, 8, 21, 0)),
    ]
    assert cal.last == date(2026, 6, 22)
    assert not cal.books(date(2026, 6, 19))


def test_session_windows_rth_by_subgroup_and_globex() -> None:
    day = date(2025, 4, 2)

    assert sessions.session_window_ns("SI", day, "rth") == (utc(2025, 4, 2, 12, 20), utc(2025, 4, 2, 17, 25))
    assert sessions.session_window_ns("HG", day, "rth") == (utc(2025, 4, 2, 12, 10), utc(2025, 4, 2, 17, 0))
    assert sessions.session_window_ns("NQ", day, "rth") == (utc(2025, 4, 2, 13, 30), utc(2025, 4, 2, 20, 0))
    assert sessions.session_window_ns("NQ", day, "globex") == (utc(2025, 4, 1, 22, 0), utc(2025, 4, 2, 21, 0))


def test_topstep_markets_closed_row_closes_any_group() -> None:
    # Arrange: a synthetic Topstep "markets closed" row on an ordinary weekday.
    cal = sessions.calendar_of("CL")
    day = date(2025, 4, 2)
    rule = replace(cal.flat_by, topstep_close_by={**cal.flat_by.topstep_close_by, day: None})

    # Act
    got = sessions.flatten_minute(replace(cal, flat_by=rule), day)

    # Assert
    assert sessions.flatten_minute(cal, day) == 15 * 60 + 8
    assert got is None


# ------------------------------------------------------------------ refusals ----
def test_unknown_root_raises() -> None:
    with pytest.raises(UnknownRoot):
        sessions.open_intervals("XX", date(2025, 4, 1), date(2025, 4, 2))


@pytest.mark.parametrize(
    ("root", "first", "last"),
    [
        ("NQ", date(2010, 5, 31), date(2010, 6, 2)),
        ("NQ", date(2026, 6, 19), date(2026, 6, 22)),
        ("MBT", date(2019, 4, 30), date(2019, 5, 2)),
    ],
)
def test_outside_coverage_raises(root: str, first: date, last: date) -> None:
    with pytest.raises(CalendarCoverageError):
        sessions.open_intervals(root, first, last)
    with pytest.raises(CalendarCoverageError):
        sessions.must_be_flat_intervals_ns(root, first, last)


def test_instant_outside_coverage_raises() -> None:
    with pytest.raises(CalendarCoverageError):
        sessions.trade_days_of("NQ", np.array([utc(2026, 6, 20, 12, 0)], dtype=np.int64))


def test_session_window_refuses_non_trade_date_and_bad_name() -> None:
    with pytest.raises(NotATradeDate):
        sessions.session_window_ns("NQ", date(2025, 4, 5), "rth")
    with pytest.raises(ValueError, match="session name"):
        sessions.session_window_ns("NQ", date(2025, 4, 2), "day")


def test_ct_ns_equals_zoneinfo_across_dst() -> None:
    for day in (date(2025, 3, 9), date(2025, 11, 2), date(2016, 3, 13)):
        for minute in (0, 60, 179, 180, 1020, 1439):
            local = datetime(day.year, day.month, day.day, minute // 60, minute % 60, tzinfo=calmod.CT)
            assert calmod.ct_ns(day, minute) == int(local.timestamp()) * 1_000_000_000







