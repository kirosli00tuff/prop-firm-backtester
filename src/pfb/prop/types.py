"""Interface types between the rule engine and the Monte Carlo (docs/PROP_MC.md).

Money in integer cents. A day path is a run of minutes with eq_low, eq_high, eq_close relative to the day's
opening balance (docs/PROP_RULES_SCHEMA.md section 8).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Phase end codes (EvalOutcome.end, FundedOutcome.end and the batch arrays).
END_PASS = 0  # evaluation passed
END_BREACH = 1  # drawdown or hard daily-loss breach
END_TIMEOUT = 2  # horizon or padding (-1) reached while open
END_PAYOUT_LIMIT = 3  # funded account ended after payout_count_limit payouts
END_TIME_LIMIT = 4  # evaluation access window (time_limit_calendar_days) ran out
END_SIZE_LIMIT = 5  # a day above the phase's contract limit (max_contracts_minis or the scaling tier)


class RuleUnknown(LookupError):
    """A run needs a rule value the config marks unknown; the message is the dotted path and the reason."""


@dataclass(frozen=True)
class DayPaths:
    """Per-trading-day minute equity paths of one run (or a pool), ragged by day.

    Minutes of day d are offsets[d]:offsets[d + 1]. low/high/close are cents relative to the day's opening
    equity (a day with no minute rows has one row of zeros). sides[d] counts contract sides filled that day;
    max_contracts[d] is the largest absolute position in mini equivalents; fill_sides[m] counts sides
    filled at minute m (for the cost stress). trade_dates[d] is the CME trade date as int yyyymmdd.
    """

    trade_dates: np.ndarray  # int32 [n_days]
    offsets: np.ndarray  # int64 [n_days + 1]
    low: np.ndarray  # int64 [n_minutes]
    high: np.ndarray  # int64 [n_minutes]
    close: np.ndarray  # int64 [n_minutes]
    fill_sides: np.ndarray  # int64 [n_minutes]
    sides: np.ndarray  # int64 [n_days]
    max_contracts: np.ndarray  # int64 [n_days]
    source: tuple[str, ...] = field(default_factory=tuple)  # run ids (several for a pool)

    @property
    def n_days(self) -> int:
        return int(self.offsets.shape[0] - 1)

    def day_net(self) -> np.ndarray:
        """Close-of-day P&L per day (int64 cents)."""
        return self.close[self.offsets[1:] - 1]


@dataclass(frozen=True)
class EvalOutcome:
    end: int  # END_PASS | END_BREACH | END_TIMEOUT | END_TIME_LIMIT | END_SIZE_LIMIT
    days: int  # trading days used, the ending day included
    balance_cents: int  # balance at the end (at the breach minute's equity for a breach)
    best_day_cents: int
    traded_days: int


@dataclass(frozen=True)
class FundedOutcome:
    end: int  # END_BREACH | END_TIMEOUT | END_PAYOUT_LIMIT | END_SIZE_LIMIT
    days: int  # trading days alive
    payouts_gross_cents: tuple[int, ...]  # each payout as taken from the account
    payouts_trader_cents: tuple[int, ...]  # after the split
    first_payout_day: int  # -1 when none
    balance_cents: int
