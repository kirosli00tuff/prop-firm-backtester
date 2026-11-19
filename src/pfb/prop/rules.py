"""Scalar reference rule engine (docs/PROP_RULES_SCHEMA.md section 8): one evaluation attempt or one funded
life along a sequence of day indices, minute by minute, in plain loops. This module is the specification;
kernel.py (Numba) is pinned to it field by field.

Readings this module fixes where the schema is silent:
- Within one minute: the high moves an intraday-trailing floor first, then the low is tested against the
  floor (breach), then the daily loss on the low, then (evaluation, target_check intraday) the target on the
  high. A breach ends the phase with the balance at that minute's low equity.
- A day whose max_contracts exceeds the allowed size (eval max_contracts_minis; funded: the scaling tier of
  the prior close's balance over start, before a payout taken at that close, as upstream funnel.run_xfa) ends
  the
  phase as END_SIZE_LIMIT on that day with the balance at the open.
- time_limit_calendar_days converts to trading days at 30 calendar = 21 trading (docs/PROP_MC.md section 4);
  an attempt still open after that many days ends END_TIME_LIMIT.
- Funded: the payout is requested at the close of the first eligible day (after the floor trail); with
  payout_day_counts false the NEXT trading day (the day the request is processed) does not count toward the
  new window's day counts, as upstream. dead_days_after_payout days after that count toward no day count.
  first_payout_day is the number of trading days alive at the first payout.
- Window net and best day (net_positive_since_last, net_min_*, window consistency, window cap) cover EVERY
  day since the last payout, the skipped and dead days included: the account's net since the last payout
  (upstream left those days out). Only the traded and winning day counts skip them.
- Calendar (N3 and payout.window_calendar_days): the funded phase keeps a sequence clock: the first
  day of the sequence is its trade date and each later day advances the clock by that day's own gap from the
  previous trading day of the history (sequence_clock). request_weekday reads the clock's weekday;
  window_calendar_days counts clock days from the window start (the first traded day of the life, then the
  close of each payout). Along historical order the clock is the trade date itself.
- Soft daily loss: books exactly -daily_loss by default (as upstream, optimistic);
soft_daily_loss="minute_low"
  books the triggering minute's low, never better than -daily_loss (N1, a sensitivity switch).
- The trader share of a payout is floor division in cents; split_first_full_cents pays the first that many
  cumulative gross cents at 100%.
- The window cap (payout.window_cap_bps, from payout index window_cap_from_payout on) uses the window net
  since the last payout: amount <= window net x bps // 10000.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np

from pfb.prop.config import BPS_ONE, NONE, PlanRules, PlanVersions
from pfb.prop.types import (
    END_BREACH,
    END_PASS,
    END_PAYOUT_LIMIT,
    END_SIZE_LIMIT,
    END_TIME_LIMIT,
    END_TIMEOUT,
    DayPaths,
    EvalOutcome,
    FundedOutcome,
)

TRADING_DAYS_PER_30_CALENDAR = 21
CALENDAR_DAYS_PER_BLOCK = 30
DD_EOD, DD_INTRADAY, DD_STATIC = 0, 1, 2
DD_TYPES = {"eod_trailing": DD_EOD, "intraday_trailing": DD_INTRADAY, "static": DD_STATIC}
CONS_NONE, CONS_PROFIT, CONS_TARGET = 0, 1, 2
CONS_TYPES = {NONE: CONS_NONE, "best_day_frac_of_profit": CONS_PROFIT, "best_day_frac_of_target": CONS_TARGET}
# Soft daily loss sensitivity switch (N1): "limit" books exactly -daily_loss (default, as upstream);
# "minute_low" books the triggering minute's low, never better than -daily_loss (the pessimistic fill).
SOFT_DLL_LIMIT, SOFT_DLL_LOW = "limit", "minute_low"
SOFT_DLL_MODES = (SOFT_DLL_LIMIT, SOFT_DLL_LOW)


# ------------------------------------------------------------------------- resolved parameters ----
@dataclass(frozen=True)
class Drawdown:
    start: int
    drawdown: int
    dd_type: int  # DD_EOD | DD_INTRADAY | DD_STATIC
    breach_intraday: bool
    at_equal: bool
    lock_rel: int | None  # floor never rises above start + lock_rel
    daily_loss: int | None
    daily_loss_hard: bool
    soft_at_low: bool = False  # soft daily loss booked at the minute low (SOFT_DLL_LOW), else -daily_loss


@dataclass(frozen=True)
class EvalParams:
    dd: Drawdown
    target: int
    target_intraday: bool
    cons_type: int
    cons_bps: int
    cons_strict: bool
    min_trading_days: int
    min_winning_days: int
    winning_day: int | None
    max_minis: int
    limit_days: int | None  # trading days


@dataclass(frozen=True)
class FundedParams:
    dd: Drawdown
    floor_after_first_rel: int | None
    scaling: tuple[tuple[int | None, int], ...]
    boundary_lower: bool
    payout_limit: int | None
    win_day: int | None
    win_strict: bool
    win_days_req: int
    traded_req: int
    net_positive: bool
    net_min_first: int | None
    net_min_later: int | None
    min_balance_rel: int | None
    buffer_rel: int | None
    frac_bps: int
    min_request: int
    caps: tuple[int, ...] | None  # None: no fixed cap
    window_cap_bps: int | None
    window_cap_from: int  # 1-based payout index the window cap applies from
    cons_bps: int | None
    cons_cumulative: bool
    cons_strict: bool
    weekday: int | None
    window_days: int | None  # calendar days a payout window needs (payout.window_calendar_days)
    dead_days: int
    day_counts: bool
    split_bps: int
    split_first_full: int | None


def check_soft_daily_loss(mode: str) -> bool:
    """True for SOFT_DLL_LOW, False for SOFT_DLL_LIMIT; anything else raises ValueError."""
    if mode not in SOFT_DLL_MODES:
        raise ValueError(f"soft_daily_loss must be one of {SOFT_DLL_MODES}, got {mode!r}")
    return mode == SOFT_DLL_LOW


def _drawdown(plan: PlanRules, sec: str, soft_daily_loss: str) -> Drawdown:
    soft_at_low = check_soft_daily_loss(soft_daily_loss)
    v = plan.value
    dd_type = DD_TYPES[v(f"{sec}.drawdown_type")]
    kind = v(f"{sec}.daily_loss_kind")
    return Drawdown(
        start=v(f"{sec}.start_balance_cents"),
        drawdown=v(f"{sec}.drawdown_cents"),
        dd_type=dd_type,
        breach_intraday=v(f"{sec}.breach_check") == "intraday",
        at_equal=v(f"{sec}.breach_at_equal"),
        lock_rel=None if dd_type == DD_STATIC else v(f"{sec}.drawdown_lock_rel_cents"),
        daily_loss=None if kind == NONE else v(f"{sec}.daily_loss_cents"),
        daily_loss_hard=kind == "hard",
        soft_at_low=soft_at_low,
    )


def trading_day_limit(calendar_days: int | None) -> int | None:
    """Calendar access window to trading days (30 calendar = 21 trading, floor)."""
    if calendar_days is None:
        return None
    return calendar_days * TRADING_DAYS_PER_30_CALENDAR // CALENDAR_DAYS_PER_BLOCK


def eval_params(plan: PlanRules, soft_daily_loss: str = SOFT_DLL_LIMIT) -> EvalParams:
    """The evaluation rules a run needs; an unknown one raises RuleUnknown (only the ones used are read)."""
    v = plan.value
    cons_type = CONS_TYPES[v("eval.consistency_type")]
    min_win = v("eval.min_winning_days")
    return EvalParams(
        dd=_drawdown(plan, "eval", soft_daily_loss),
        target=v("eval.profit_target_cents"),
        target_intraday=v("eval.target_check") == "intraday",
        cons_type=cons_type,
        cons_bps=0 if cons_type == CONS_NONE else v("eval.consistency_bps"),
        cons_strict=False if cons_type == CONS_NONE else v("eval.consistency_strict"),
        min_trading_days=v("eval.min_trading_days"),
        min_winning_days=min_win,
        winning_day=v("eval.winning_day_cents") if min_win > 0 else None,
        max_minis=v("eval.max_contracts_minis"),
        limit_days=trading_day_limit(v("eval.time_limit_calendar_days")),
    )


def funded_params(plan: PlanRules, soft_daily_loss: str = SOFT_DLL_LIMIT) -> FundedParams:
    """The funded and payout rules a run needs; an unknown one raises RuleUnknown."""
    v = plan.value
    win_req = v("payout.win_days_required")
    cons_bps = v("payout.consistency_bps")
    window_cap = v("payout.window_cap_bps")
    return FundedParams(
        dd=_drawdown(plan, "funded", soft_daily_loss),
        floor_after_first_rel=v("funded.floor_after_first_payout_rel_cents"),
        scaling=v("funded.scaling"),
        boundary_lower=v("funded.scaling_boundary") == "lower",
        payout_limit=v("funded.payout_count_limit"),
        win_day=v("payout.win_day_cents") if win_req > 0 else None,
        win_strict=v("payout.win_day_strict") if win_req > 0 else False,
        win_days_req=win_req,
        traded_req=v("payout.traded_days_required"),
        net_positive=v("payout.net_positive_since_last"),
        net_min_first=v("payout.net_min_first_cents"),
        net_min_later=v("payout.net_min_later_cents"),
        min_balance_rel=v("payout.min_balance_rel_cents"),
        buffer_rel=v("payout.buffer_rel_cents"),
        frac_bps=v("payout.frac_of_profit_bps"),
        min_request=v("payout.min_request_cents"),
        caps=v("payout.caps_cents"),
        window_cap_bps=window_cap,
        window_cap_from=v("payout.window_cap_from_payout") if window_cap is not None else 1,
        cons_bps=cons_bps,
        cons_cumulative=cons_bps is not None and v("payout.consistency_scope") == "cumulative",
        cons_strict=cons_bps is not None and v("payout.consistency_strict"),
        weekday=v("payout.request_weekday"),
        window_days=v("payout.window_calendar_days"),
        dead_days=v("payout.dead_days_after_payout"),
        day_counts=v("payout.payout_day_counts"),
        split_bps=v("payout.split_bps"),
        split_first_full=v("payout.split_first_full_cents"),
    )


# -------------------------------------------------------------------------------- shared rules ----
def floor_from_peak(dd: Drawdown, peak: int) -> int:
    floor = peak - dd.drawdown
    if dd.lock_rel is not None:
        floor = min(floor, dd.start + dd.lock_rel)
    return floor


def below_floor(dd: Drawdown, equity: int, floor: int) -> bool:
    return equity <= floor if dd.at_equal else equity < floor


def frac_ok(best: int, base: int, bps: int, strict: bool) -> bool:
    """best <= (or <) bps/10000 x base, in integers."""
    lhs, rhs = best * BPS_ONE, bps * base
    return lhs < rhs if strict else lhs <= rhs


def tier_minis(scaling: tuple[tuple[int | None, int], ...], rel: int, boundary_lower: bool) -> int:
    """Max minis for a balance over start; exactly on a boundary takes the lower tier when boundary_lower."""
    for below, minis in scaling:
        if below is None or rel < below or (boundary_lower and rel == below):
            return minis
    raise AssertionError("the last scaling tier has no bound")


def iso_weekday(yyyymmdd: int) -> int:
    """ISO weekday (1 = Monday) of an int yyyymmdd date (Sakamoto, integers only)."""
    y, m, d = yyyymmdd // 10000, (yyyymmdd // 100) % 100, yyyymmdd % 100
    offsets = (0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4)
    if m < 3:
        y -= 1
    w = (y + y // 4 - y // 100 + y // 400 + offsets[m - 1] + d) % 7  # 0 = Sunday
    return 7 if w == 0 else w


# ------------------------------------------------------------------------ sequence calendar ----
def date_ordinal(yyyymmdd: int) -> int:
    """Proleptic Gregorian ordinal of an int yyyymmdd date (0001-01-01 = 1, a Monday)."""
    return dt.date(yyyymmdd // 10000, (yyyymmdd // 100) % 100, yyyymmdd % 100).toordinal()


def ordinal_weekday(ordinal: int) -> int:
    """ISO weekday (1 = Monday) of a date ordinal."""
    return (ordinal - 1) % 7 + 1


def calendar_gap(days: DayPaths, d: int) -> int:
    """Calendar days from the history's previous trading day to day d: 1 for the first day of the history,
    never below 1 (check_day_paths refuses dates that do not increase)."""
    if d == 0:
        return 1
    gap = date_ordinal(int(days.trade_dates[d])) - date_ordinal(int(days.trade_dates[d - 1]))
    return max(1, gap)


def sequence_clock(days: DayPaths, d: int, k: int, clock: int) -> int:
    """The sequence's own calendar (N3 and the payout window): the first day of a sequence is its
    trade date; every later day advances the clock by that day's own calendar gap in the history. Along
    historical order this is each day's trade date; under the block bootstrap a block keeps its real
    spacing and a jump advances by the new day's gap."""
    if k == 0:
        return date_ordinal(int(days.trade_dates[d]))
    return clock + calendar_gap(days, d)


@dataclass(frozen=True)
class DayResult:
    """One day's run against the drawdown (a breach, an intraday target touch, or the close)."""

    breach: bool
    equity: int  # breach equity, or the close balance
    net: int  # close P&L (soft daily loss: -daily_loss)
    peak: int
    floor: int
    target_minute_hit: bool = False
    target_high: int = 0


def run_day(
    dd: Drawdown,
    days: DayPaths,
    d: int,
    open_bal: int,
    peak: int,
    floor: int,
    trail_intraday: bool,
    target_abs: int | None,
    target_ok_at: Callable[[int], bool] | None = None,
) -> DayResult:
    """Minutes of day d. ``target_abs`` (eval, intraday target) is the balance that passes when touched;
    ``target_ok_at(high_pnl)`` says whether the other pass conditions hold at that minute."""
    m0, m1 = int(days.offsets[d]), int(days.offsets[d + 1])
    for m in range(m0, m1):
        hi, lo = int(days.high[m]), int(days.low[m])
        if trail_intraday:
            peak = max(peak, open_bal + hi)
            floor = floor_from_peak(dd, peak)
        if dd.breach_intraday and below_floor(dd, open_bal + lo, floor):
            return DayResult(True, open_bal + lo, lo, peak, floor)
        if dd.daily_loss is not None and lo <= -dd.daily_loss:
            if dd.daily_loss_hard:
                return DayResult(True, open_bal + lo, lo, peak, floor)
            booked = min(lo, -dd.daily_loss) if dd.soft_at_low else -dd.daily_loss
            return _close(dd, open_bal, booked, peak, floor)
        if (
            target_abs is not None
            and target_ok_at is not None
            and open_bal + hi >= target_abs
            and target_ok_at(hi)
        ):
            return DayResult(False, open_bal + hi, hi, peak, floor, True, hi)
    return _close(dd, open_bal, int(days.close[m1 - 1]), peak, floor)


def _close(dd: Drawdown, open_bal: int, net: int, peak: int, floor: int) -> DayResult:
    close_bal = open_bal + net
    if not dd.breach_intraday and below_floor(dd, close_bal, floor):
        return DayResult(True, close_bal, net, peak, floor)
    return DayResult(False, close_bal, net, peak, floor)


# -------------------------------------------------------------------------------- evaluation ----
def _eval_pass(p: EvalParams, profit: int, best: int, traded: int, wins: int) -> bool:
    if profit < p.target or traded < p.min_trading_days or wins < p.min_winning_days:
        return False
    if p.cons_type == CONS_PROFIT:
        return frac_ok(best, profit, p.cons_bps, p.cons_strict)
    if p.cons_type == CONS_TARGET:
        return frac_ok(best, p.target, p.cons_bps, p.cons_strict)
    return True


def _intraday_pass(p: EvalParams, open_bal: int, best: int, traded: int, wins: int, hi: int) -> bool:
    """Pass conditions at a minute whose high touches the target (today counted with P&L = high)."""
    win = p.winning_day is not None and hi >= p.winning_day
    return _eval_pass(p, open_bal + hi - p.dd.start, max(best, hi), traded, wins + win)


def simulate_eval(
    plan: PlanRules, days: DayPaths, seq: np.ndarray, *, soft_daily_loss: str = SOFT_DLL_LIMIT
) -> EvalOutcome:
    """One evaluation attempt from a fresh account along ``seq`` (day indices; -1 pads = timeout)."""
    return simulate_eval_params(eval_params(plan, soft_daily_loss), days, seq)


def simulate_eval_params(p: EvalParams, days: DayPaths, seq: np.ndarray) -> EvalOutcome:
    return simulate_eval_at(p, lambda _d: p, days, seq)


def simulate_eval_at(
    p0: EvalParams, at: Callable[[int], EvalParams], days: DayPaths, seq: np.ndarray
) -> EvalOutcome:
    """One evaluation attempt where day index d is charged under ``at(d)`` (the rules version in force on
    that day); ``p0`` opens the account (start balance, first floor). With a constant ``at``
    this is simulate_eval_params exactly. On a version change every rule of the new version applies from that
    day on; the floor carried from the prior close stays until the new version's trailing rule moves it."""
    p = p0
    dd = p.dd
    bal, peak = dd.start, dd.start
    floor = dd.start - dd.drawdown
    best, traded, wins = 0, 0, 0
    horizon = int(seq.shape[0])
    for k in range(horizon):
        d = int(seq[k])
        if d >= 0:
            p = at(d)
            dd = p.dd
        if p.limit_days is not None and k >= p.limit_days:
            return EvalOutcome(END_TIME_LIMIT, k, bal, best, traded)
        if d < 0:
            return EvalOutcome(END_TIMEOUT, k, bal, best, traded)
        is_traded = int(days.sides[d]) > 0
        if int(days.max_contracts[d]) > p.max_minis:
            return EvalOutcome(END_SIZE_LIMIT, k + 1, bal, best, traded + is_traded)

        ok_at = partial(_intraday_pass, p, bal, best, traded + is_traded, wins)
        target_abs = dd.start + p.target if p.target_intraday else None
        res = run_day(dd, days, d, bal, peak, floor, dd.dd_type == DD_INTRADAY, target_abs, ok_at)
        if res.breach:
            return EvalOutcome(END_BREACH, k + 1, res.equity, best, traded + is_traded)
        peak, floor = res.peak, res.floor
        traded += is_traded
        best = max(best, res.net)
        wins += p.winning_day is not None and res.net >= p.winning_day
        bal = res.equity
        if res.target_minute_hit:
            return EvalOutcome(END_PASS, k + 1, bal, best, traded)
        if dd.dd_type == DD_EOD:
            peak = max(peak, bal)
            floor = floor_from_peak(dd, peak)
        if _eval_pass(p, bal - dd.start, best, traded, wins):
            return EvalOutcome(END_PASS, k + 1, bal, best, traded)
    if p.limit_days is not None and horizon >= p.limit_days:
        return EvalOutcome(END_TIME_LIMIT, horizon, bal, best, traded)
    return EvalOutcome(END_TIMEOUT, horizon, bal, best, traded)


# ------------------------------------------------------------------------------------ funded ----
@dataclass
class _Window:
    """Payout bookkeeping (local to one simulate_funded call)."""

    traded: int = 0
    wins: int = 0
    net: int = 0
    best: int = 0
    cum_net: int = 0
    cum_best: int = 0
    start: int | None = None  # sequence-clock ordinal the window runs from (first traded day, then payouts)

    def book(self, p: FundedParams, net: int, traded: bool, counted: bool) -> None:
        """One closed day. Every day's P&L enters the window net and best day (the account's net since the
        last payout); only counted days enter the traded and winning day counts."""
        if counted:
            self.traded += traded
            self.wins += p.win_day is not None and (net > p.win_day if p.win_strict else net >= p.win_day)
        self.net += net
        self.best = max(self.best, net)
        self.cum_net += net
        self.cum_best = max(self.cum_best, net)

    def reset(self, clock: int) -> None:
        self.traded, self.wins, self.net, self.best = 0, 0, 0, 0
        self.start = clock


def window_open(p: FundedParams, w: _Window, clock: int) -> bool:
    """payout.window_calendar_days: at least that many calendar days of the sequence clock since the window
    start (the first traded day, then each payout's close); no start yet means no trade yet."""
    if p.window_days is None:
        return True
    return w.start is not None and clock - w.start >= p.window_days


def payout_eligible(p: FundedParams, w: _Window, npay: int, profit: int, clock: int) -> bool:
    """The window since the last payout qualifies at this close (schema section 6); ``clock`` is the
    sequence-clock ordinal of the day (its weekday is the one the request_weekday gate reads)."""
    if w.wins < p.win_days_req or w.traded < p.traded_req or not window_open(p, w, clock):
        return False
    if p.net_positive and not (w.net > 0 if npay > 0 else profit > 0):
        return False
    if npay == 0 and p.net_min_first is not None and w.net < p.net_min_first:
        return False
    if npay > 0 and p.net_min_later is not None and w.net < p.net_min_later:
        return False
    if p.min_balance_rel is not None and profit < p.min_balance_rel:
        return False
    if p.cons_bps is not None:
        base, best = (w.cum_net, w.cum_best) if p.cons_cumulative else (w.net, w.best)
        if base <= 0 or not frac_ok(best, base, p.cons_bps, p.cons_strict):
            return False
    return p.weekday is None or ordinal_weekday(clock) == p.weekday


def payout_amount(p: FundedParams, npay: int, profit: int, window_net: int) -> int:
    """The largest request allowed: cap of the (npay+1)-th payout, frac of profit, buffer, window cap."""
    amount = p.frac_bps * profit // BPS_ONE
    if p.caps is not None:
        amount = min(amount, p.caps[min(npay, len(p.caps) - 1)])
    if p.buffer_rel is not None:
        amount = min(amount, profit - p.buffer_rel)
    if p.window_cap_bps is not None and npay + 1 >= p.window_cap_from:
        amount = min(amount, window_net * p.window_cap_bps // BPS_ONE)
    return amount


def trader_share(p: FundedParams, gross: int, paid_before: int) -> int:
    full = 0
    if p.split_first_full is not None:
        full = min(gross, max(0, p.split_first_full - paid_before))
    return full + (gross - full) * p.split_bps // BPS_ONE


def simulate_funded(
    plan: PlanRules, days: DayPaths, seq: np.ndarray, *, soft_daily_loss: str = SOFT_DLL_LIMIT
) -> FundedOutcome:
    """One funded life from the funded start balance along ``seq`` (-1 pads = timeout)."""
    return simulate_funded_params(funded_params(plan, soft_daily_loss), days, seq)


def simulate_funded_params(p: FundedParams, days: DayPaths, seq: np.ndarray) -> FundedOutcome:
    return simulate_funded_at(p, lambda _d: p, days, seq)


def simulate_funded_at(  # noqa: C901 - the reference loop, kept in one piece as the specification
    p0: FundedParams, at: Callable[[int], FundedParams], days: DayPaths, seq: np.ndarray
) -> FundedOutcome:
    """One funded life where day index d is charged under ``at(d)`` (see simulate_eval_at); ``p0`` opens
    the account. With a constant ``at`` this is simulate_funded_params exactly."""
    p = p0
    dd = p.dd
    bal, peak = dd.start, dd.start
    close_bal = dd.start  # the prior close's balance before a payout taken at it (sets the scaling tier)
    floor = dd.start - dd.drawdown
    fixed = False
    w = _Window()
    gross: list[int] = []
    trader: list[int] = []
    first = -1
    skip_today, dead = False, 0
    clock = 0
    horizon = int(seq.shape[0])

    def out(end: int, n_days: int, balance: int) -> FundedOutcome:
        return FundedOutcome(end, n_days, tuple(gross), tuple(trader), first, balance)

    for k in range(horizon):
        d = int(seq[k])
        if d < 0:
            return out(END_TIMEOUT, k, bal)
        p = at(d)
        dd = p.dd
        clock = sequence_clock(days, d, k, clock)
        if int(days.max_contracts[d]) > tier_minis(p.scaling, close_bal - dd.start, p.boundary_lower):
            return out(END_SIZE_LIMIT, k + 1, bal)
        res = run_day(dd, days, d, bal, peak, floor, dd.dd_type == DD_INTRADAY and not fixed, None)
        if res.breach:
            return out(END_BREACH, k + 1, res.equity)
        peak, floor, bal, net = res.peak, res.floor, res.equity, res.net
        counted = not skip_today and dead == 0
        if not skip_today and dead > 0:
            dead -= 1
        skip_today = False
        is_traded = int(days.sides[d]) > 0
        if w.start is None and is_traded:
            w.start = clock
        w.book(p, net, is_traded, counted)
        if dd.dd_type == DD_EOD and not fixed:
            peak = max(peak, bal)
            floor = floor_from_peak(dd, peak)
        close_bal = bal
        profit = bal - dd.start
        if not payout_eligible(p, w, len(gross), profit, clock):
            continue
        amount = payout_amount(p, len(gross), profit, w.net)
        if amount < p.min_request or amount <= 0:
            continue
        trader.append(trader_share(p, amount, sum(gross)))
        gross.append(amount)
        bal -= amount
        if len(gross) == 1:
            first = k + 1
            if p.floor_after_first_rel is not None:
                floor, fixed = dd.start + p.floor_after_first_rel, True
        w.reset(clock)
        dead = p.dead_days
        skip_today = not p.day_counts
        if p.payout_limit is not None and len(gross) >= p.payout_limit:
            return out(END_PAYOUT_LIMIT, k + 1, bal)
    return out(END_TIMEOUT, horizon, bal)


# ------------------------------------------------------------ rules in force by date ----
RULES_BY_DAY = "by_day"  # a dated (historical-order) path: each day charged under the version in force
RULES_NEWEST = "newest_undated"  # a resampled path: every day charged under the newest version


@dataclass(frozen=True)
class DatedRun:
    """One evaluation attempt, then (after a pass) one funded life, along the history in date order from
    ``start``; each day charged under the rules version in force on its trade date. ``labels`` are the
    rules_label of every version that charged a day, with the count of days it charged."""

    eval: EvalOutcome
    funded: FundedOutcome | None
    day_version: tuple[int, ...]  # the version index charged on each walked day, in walk order
    days_before_earliest: int
    labels: tuple[dict, ...]
    warnings: tuple[str, ...]
    rules_in_force: str = RULES_BY_DAY


def versions_by_day(versions: PlanVersions, days: DayPaths) -> np.ndarray:
    """The index (into versions.versions) of the rules version in force on each day of the history."""
    return np.asarray([versions.in_force(int(t)) for t in days.trade_dates], dtype=np.int64)


def before_earliest_warning(versions: PlanVersions, n_before: int, n_days: int) -> str | None:
    if not n_before:
        return None
    first = versions.earliest
    name = f"{first.firm}/{first.plan}_{first.size}"
    return (f"rules: {n_before} of {n_days} days predate the earliest rules version of {name} (version "
            f"{first.version}, effective_from {first.effective_from}) and are charged under it")  # fmt: skip


def run_dated(
    versions: PlanVersions, days: DayPaths, *, start: int = 0, soft_daily_loss: str = SOFT_DLL_LIMIT
) -> DatedRun:
    """The dated path from day ``start``. Only the versions in force on a walked day
    are read, so an unknown rule of an unused version raises nothing."""
    if not 0 <= start < days.n_days:
        raise ValueError(f"start {start} outside the history's {days.n_days} days")
    idx = versions_by_day(versions, days)
    cache_e: dict[int, EvalParams] = {}
    cache_f: dict[int, FundedParams] = {}

    def eval_at(d: int) -> EvalParams:
        i = int(idx[d])
        if i not in cache_e:
            cache_e[i] = eval_params(versions.versions[i], soft_daily_loss)
        return cache_e[i]

    def funded_at(d: int) -> FundedParams:
        i = int(idx[d])
        if i not in cache_f:
            cache_f[i] = funded_params(versions.versions[i], soft_daily_loss)
        return cache_f[i]

    seq = np.arange(start, days.n_days, dtype=np.int64)
    ev = simulate_eval_at(eval_at(start), eval_at, days, seq)
    walked = list(seq[: ev.days])
    funded = None
    if ev.end == END_PASS and start + ev.days < days.n_days:
        fseq = np.arange(start + ev.days, days.n_days, dtype=np.int64)
        funded = simulate_funded_at(funded_at(int(fseq[0])), funded_at, days, fseq)
        walked += list(fseq[: funded.days])
    return _dated_result(versions, days, idx, walked, ev, funded)


def _dated_result(versions: PlanVersions, days: DayPaths, idx: np.ndarray, walked: list,
                  ev: EvalOutcome, funded: FundedOutcome | None) -> DatedRun:  # fmt: skip
    used = tuple(int(idx[d]) for d in walked)
    n_before = sum(versions.before_earliest(int(days.trade_dates[d])) for d in walked)
    labels = tuple(
        {**versions.versions[i].rules_label(), "days_charged": used.count(i)} for i in sorted(set(used))
    )
    warning = before_earliest_warning(versions, n_before, len(walked))
    return DatedRun(ev, funded, used, n_before, labels, (warning,) if warning else ())
