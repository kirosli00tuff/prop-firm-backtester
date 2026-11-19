"""Numba batch rule engine: simulate_eval_many and simulate_funded_many over many day-index sequences.

Pinned field by field to the scalar reference src/pfb/prop/rules.py (tests/prop/test_kernel_parity.py). Plan
values reach the kernels as one int64 parameter vector per phase (index constants below; a None value is a
has-flag of 0) plus small arrays (scaling tiers, payout caps; caps "none" is one NO_BOUND entry); the plan is
resolved once by rules.eval_params / rules.funded_params, so an unknown rule raises RuleUnknown before any
kernel runs. The funded kernel keeps the same sequence clock as rules.sequence_clock (date ordinals and
calendar gaps per day, calendar_arrays) for request_weekday and window_calendar_days.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from pfb.prop.config import BPS_ONE, PlanRules
from pfb.prop.rules import (
    CONS_PROFIT,
    CONS_TARGET,
    DD_EOD,
    DD_INTRADAY,
    SOFT_DLL_LIMIT,
    Drawdown,
    EvalParams,
    FundedParams,
    date_ordinal,
    eval_params,
    funded_params,
)
from pfb.prop.types import (
    END_BREACH,
    END_PASS,
    END_PAYOUT_LIMIT,
    END_SIZE_LIMIT,
    END_TIME_LIMIT,
    END_TIMEOUT,
    DayPaths,
)

# drawdown block, shared by both phase vectors
(
    D_START,
    D_DD,
    D_TYPE,
    D_BREACH_INTRA,
    D_AT_EQUAL,
    D_HAS_LOCK,
    D_LOCK,
    D_HAS_DLL,
    D_DLL,
    D_DLL_HARD,
    D_SOFT_LOW,
) = range(11)
# evaluation vector
(
    E_TARGET,
    E_TARGET_INTRA,
    E_CONS_TYPE,
    E_CONS_BPS,
    E_CONS_STRICT,
    E_MIN_TRADED,
    E_MIN_WINS,
    E_HAS_WINDAY,
    E_WINDAY,
    E_MAX_MINIS,
    E_HAS_LIMIT,
    E_LIMIT,
    E_LEN,
) = range(11, 24)
# funded vector
(
    F_HAS_FLOOR_AFTER,
    F_FLOOR_AFTER,
    F_BOUNDARY_LOWER,
    F_HAS_PLIMIT,
    F_PLIMIT,
    F_HAS_WINDAY,
    F_WINDAY,
    F_WIN_STRICT,
    F_WIN_REQ,
    F_TRADED_REQ,
    F_NET_POS,
    F_HAS_NMF,
    F_NMF,
    F_HAS_NML,
    F_NML,
    F_HAS_MINBAL,
    F_MINBAL,
    F_HAS_BUF,
    F_BUF,
    F_FRAC,
    F_MINREQ,
    F_HAS_CONS,
    F_CONS_BPS,
    F_CONS_CUM,
    F_CONS_STRICT,
    F_HAS_WEEKDAY,
    F_WEEKDAY,
    F_DEAD,
    F_DAY_COUNTS,
    F_SPLIT,
    F_HAS_SPLIT_FULL,
    F_SPLIT_FULL,
    F_HAS_WCAP,
    F_WCAP,
    F_WCAP_FROM,
    F_HAS_WINDOW,
    F_WINDOW,
    F_LEN,
) = range(11, 49)
NO_BOUND = np.iinfo(np.int64).max
DAY_CLOSE, DAY_BREACH, DAY_TARGET = 0, 1, 2


@dataclass(frozen=True)
class EvalBatch:
    """simulate_eval_many: the EvalOutcome fields as arrays [n_paths]."""

    end: np.ndarray
    days: np.ndarray
    balance_cents: np.ndarray
    best_day_cents: np.ndarray
    traded_days: np.ndarray


@dataclass(frozen=True)
class FundedBatch:
    """simulate_funded_many: the FundedOutcome fields as arrays. payouts_*_cents are [n_paths, max_payouts],
    zero past n_payouts[i]; max_payouts is a bound no path can exceed (or the caller's choice)."""

    end: np.ndarray
    days: np.ndarray
    n_payouts: np.ndarray
    payouts_gross_cents: np.ndarray
    payouts_trader_cents: np.ndarray
    gross_total_cents: np.ndarray
    trader_total_cents: np.ndarray
    first_payout_day: np.ndarray
    balance_cents: np.ndarray


# ------------------------------------------------------------------------------- param vectors ----
def _opt(x: int | None) -> tuple[int, int]:
    return (0, 0) if x is None else (1, int(x))


def _dd_vector(dd: Drawdown, length: int) -> np.ndarray:
    v = np.zeros(length, dtype=np.int64)
    v[D_START], v[D_DD], v[D_TYPE] = dd.start, dd.drawdown, dd.dd_type
    v[D_BREACH_INTRA], v[D_AT_EQUAL] = dd.breach_intraday, dd.at_equal
    v[D_HAS_LOCK], v[D_LOCK] = _opt(dd.lock_rel)
    v[D_HAS_DLL], v[D_DLL] = _opt(dd.daily_loss)
    v[D_DLL_HARD] = dd.daily_loss_hard
    v[D_SOFT_LOW] = dd.soft_at_low
    return v


def eval_vector(p: EvalParams) -> np.ndarray:
    v = _dd_vector(p.dd, E_LEN)
    v[E_TARGET], v[E_TARGET_INTRA] = p.target, p.target_intraday
    v[E_CONS_TYPE], v[E_CONS_BPS], v[E_CONS_STRICT] = p.cons_type, p.cons_bps, p.cons_strict
    v[E_MIN_TRADED], v[E_MIN_WINS] = p.min_trading_days, p.min_winning_days
    v[E_HAS_WINDAY], v[E_WINDAY] = _opt(p.winning_day)
    v[E_MAX_MINIS] = p.max_minis
    v[E_HAS_LIMIT], v[E_LIMIT] = _opt(p.limit_days)
    return v


def funded_vector(p: FundedParams) -> np.ndarray:
    v = _dd_vector(p.dd, F_LEN)
    v[F_HAS_FLOOR_AFTER], v[F_FLOOR_AFTER] = _opt(p.floor_after_first_rel)
    v[F_BOUNDARY_LOWER] = p.boundary_lower
    v[F_HAS_PLIMIT], v[F_PLIMIT] = _opt(p.payout_limit)
    v[F_HAS_WINDAY], v[F_WINDAY] = _opt(p.win_day)
    v[F_WIN_STRICT], v[F_WIN_REQ], v[F_TRADED_REQ] = p.win_strict, p.win_days_req, p.traded_req
    v[F_NET_POS] = p.net_positive
    v[F_HAS_NMF], v[F_NMF] = _opt(p.net_min_first)
    v[F_HAS_NML], v[F_NML] = _opt(p.net_min_later)
    v[F_HAS_MINBAL], v[F_MINBAL] = _opt(p.min_balance_rel)
    v[F_HAS_BUF], v[F_BUF] = _opt(p.buffer_rel)
    v[F_FRAC], v[F_MINREQ] = p.frac_bps, p.min_request
    v[F_HAS_CONS], v[F_CONS_BPS] = _opt(p.cons_bps)
    v[F_CONS_CUM], v[F_CONS_STRICT] = p.cons_cumulative, p.cons_strict
    v[F_HAS_WEEKDAY], v[F_WEEKDAY] = _opt(p.weekday)
    v[F_DEAD], v[F_DAY_COUNTS], v[F_SPLIT] = p.dead_days, p.day_counts, p.split_bps
    v[F_HAS_SPLIT_FULL], v[F_SPLIT_FULL] = _opt(p.split_first_full)
    v[F_HAS_WCAP], v[F_WCAP] = _opt(p.window_cap_bps)
    v[F_WCAP_FROM] = p.window_cap_from
    v[F_HAS_WINDOW], v[F_WINDOW] = _opt(p.window_days)
    return v


def caps_array(p: FundedParams) -> np.ndarray:
    """Payout caps by index (the last repeats); no fixed cap is one NO_BOUND entry."""
    return np.asarray((NO_BOUND,) if p.caps is None else p.caps, dtype=np.int64)


def scaling_arrays(p: FundedParams) -> tuple[np.ndarray, np.ndarray]:
    below = np.asarray([NO_BOUND if b is None else b for b, _ in p.scaling], dtype=np.int64)
    minis = np.asarray([m for _, m in p.scaling], dtype=np.int64)
    return below, minis


def payout_slots(p: FundedParams, horizon: int) -> int:
    """Most payouts any path can take: each needs max(1, days required) distinct counted days."""
    need = max(1, p.win_days_req, p.traded_req)
    slots = horizon // need
    if p.payout_limit is not None:
        slots = min(slots, p.payout_limit)
    return max(slots, 1)


# ------------------------------------------------------------------------------------ kernels ----
@njit(cache=True)
def _floor_from_peak(P: np.ndarray, peak: int) -> int:
    floor = peak - P[D_DD]
    if P[D_HAS_LOCK] == 1:
        floor = min(floor, P[D_START] + P[D_LOCK])
    return floor


@njit(cache=True)
def _below(P: np.ndarray, equity: int, floor: int) -> bool:
    return equity <= floor if P[D_AT_EQUAL] == 1 else equity < floor


@njit(cache=True)
def _frac_ok(best: int, base: int, bps: int, strict: int) -> bool:
    lhs = best * BPS_ONE
    rhs = bps * base
    return lhs < rhs if strict == 1 else lhs <= rhs


@njit(cache=True)
def _eval_pass(P: np.ndarray, profit: int, best: int, traded: int, wins: int) -> bool:
    if profit < P[E_TARGET] or traded < P[E_MIN_TRADED] or wins < P[E_MIN_WINS]:
        return False
    if P[E_CONS_TYPE] == CONS_PROFIT:
        return _frac_ok(best, profit, P[E_CONS_BPS], P[E_CONS_STRICT])
    if P[E_CONS_TYPE] == CONS_TARGET:
        return _frac_ok(best, P[E_TARGET], P[E_CONS_BPS], P[E_CONS_STRICT])
    return True


@njit(cache=True)
def _close_day(P: np.ndarray, open_bal: int, net: int, floor: int) -> tuple[int, int]:
    close_bal = open_bal + net
    if P[D_BREACH_INTRA] == 0 and _below(P, close_bal, floor):
        return DAY_BREACH, close_bal
    return DAY_CLOSE, close_bal


@njit(cache=True)
def _run_day(P, low, high, close, m0, m1, open_bal, peak, floor, trail, intraday_target, best, traded, wins):
    """rules.run_day: (code, equity, net, peak, floor). intraday_target only for the evaluation vector."""
    for m in range(m0, m1):
        hi = high[m]
        lo = low[m]
        if trail:
            peak = max(peak, open_bal + hi)
            floor = _floor_from_peak(P, peak)
        if P[D_BREACH_INTRA] == 1 and _below(P, open_bal + lo, floor):
            return DAY_BREACH, open_bal + lo, lo, peak, floor
        if P[D_HAS_DLL] == 1 and lo <= -P[D_DLL]:
            if P[D_DLL_HARD] == 1:
                return DAY_BREACH, open_bal + lo, lo, peak, floor
            booked = min(lo, -P[D_DLL]) if P[D_SOFT_LOW] == 1 else -P[D_DLL]
            code, eq = _close_day(P, open_bal, booked, floor)
            return code, eq, booked, peak, floor
        if intraday_target and open_bal + hi >= P[D_START] + P[E_TARGET]:
            win = 1 if (P[E_HAS_WINDAY] == 1 and hi >= P[E_WINDAY]) else 0
            if _eval_pass(P, open_bal + hi - P[D_START], max(best, hi), traded, wins + win):
                return DAY_TARGET, open_bal + hi, hi, peak, floor
    net = close[m1 - 1]
    code, eq = _close_day(P, open_bal, net, floor)
    return code, eq, net, peak, floor


@njit(cache=True)
def _eval_kernel(
    P, offsets, low, high, close, sides, max_contracts, seqs, end, n_days, bal_out, best_out, traded_out
):
    n_paths, horizon = seqs.shape
    start = P[D_START]
    has_limit = P[E_HAS_LIMIT] == 1
    trail = P[D_TYPE] == DD_INTRADAY
    for i in range(n_paths):
        bal, peak = start, start
        floor = start - P[D_DD]
        best, traded, wins = 0, 0, 0
        e, k_end = -1, horizon
        for k in range(horizon):
            if has_limit and k >= P[E_LIMIT]:
                e, k_end = END_TIME_LIMIT, k
                break
            d = seqs[i, k]
            if d < 0:
                e, k_end = END_TIMEOUT, k
                break
            is_traded = 1 if sides[d] > 0 else 0
            if max_contracts[d] > P[E_MAX_MINIS]:
                e, k_end = END_SIZE_LIMIT, k + 1
                traded += is_traded
                break
            code, eq, net, peak, floor = _run_day(
                P,
                low,
                high,
                close,
                offsets[d],
                offsets[d + 1],
                bal,
                peak,
                floor,
                trail,
                P[E_TARGET_INTRA] == 1,
                best,
                traded + is_traded,
                wins,
            )
            if code == DAY_BREACH:
                e, k_end, bal = END_BREACH, k + 1, eq
                traded += is_traded
                break
            traded += is_traded
            best = max(best, net)
            if P[E_HAS_WINDAY] == 1 and net >= P[E_WINDAY]:
                wins += 1
            bal = eq
            if code == DAY_TARGET:
                e, k_end = END_PASS, k + 1
                break
            if P[D_TYPE] == DD_EOD:
                peak = max(peak, bal)
                floor = _floor_from_peak(P, peak)
            if _eval_pass(P, bal - start, best, traded, wins):
                e, k_end = END_PASS, k + 1
                break
        if e == -1:
            e = END_TIME_LIMIT if (has_limit and horizon >= P[E_LIMIT]) else END_TIMEOUT
        end[i], n_days[i], bal_out[i], best_out[i], traded_out[i] = e, k_end, bal, best, traded


@njit(cache=True)
def _tier(below, minis, rel, lower):
    for t in range(below.shape[0]):
        b = below[t]
        if b == NO_BOUND or rel < b or (lower and rel == b):
            return minis[t]
    return minis[minis.shape[0] - 1]


@njit(cache=True)
def _eligible(P, w_traded, w_wins, w_net, w_best, cum_net, cum_best, npay, profit, clock, w_start):
    if w_wins < P[F_WIN_REQ] or w_traded < P[F_TRADED_REQ]:
        return False
    if P[F_HAS_WINDOW] == 1 and (w_start < 0 or clock - w_start < P[F_WINDOW]):
        return False
    if P[F_NET_POS] == 1:
        if npay > 0 and w_net <= 0:
            return False
        if npay == 0 and profit <= 0:
            return False
    if npay == 0 and P[F_HAS_NMF] == 1 and w_net < P[F_NMF]:
        return False
    if npay > 0 and P[F_HAS_NML] == 1 and w_net < P[F_NML]:
        return False
    if P[F_HAS_MINBAL] == 1 and profit < P[F_MINBAL]:
        return False
    if P[F_HAS_CONS] == 1:
        base, best = (cum_net, cum_best) if P[F_CONS_CUM] == 1 else (w_net, w_best)
        if base <= 0 or not _frac_ok(best, base, P[F_CONS_BPS], P[F_CONS_STRICT]):
            return False
    return P[F_HAS_WEEKDAY] == 0 or (clock - 1) % 7 + 1 == P[F_WEEKDAY]


@njit(cache=True)
def _funded_kernel(
    P,
    below,
    minis,
    caps,
    offsets,
    low,
    high,
    close,
    sides,
    max_contracts,
    ordinals,
    gaps,
    seqs,
    end,
    n_days,
    npay_out,
    gross,
    trader,
    gross_tot,
    trader_tot,
    first_out,
    bal_out,
):
    n_paths, horizon = seqs.shape
    start = P[D_START]
    slots = gross.shape[1]
    for i in range(n_paths):
        bal, peak = start, start
        close_bal = start  # prior close before a same-close payout (scaling tier, rules.py)
        floor = start - P[D_DD]
        fixed = False
        w_traded, w_wins, w_net, w_best, cum_net, cum_best = 0, 0, 0, 0, 0, 0
        npay, paid, paid_trader, first = 0, 0, 0, -1
        skip_today, dead = False, 0
        clock, w_start = 0, -1  # sequence clock (rules.sequence_clock); -1: no window start yet
        e, k_end = END_TIMEOUT, horizon
        for k in range(horizon):
            d = seqs[i, k]
            if d < 0:
                e, k_end = END_TIMEOUT, k
                break
            clock = ordinals[d] if k == 0 else clock + gaps[d]
            if max_contracts[d] > _tier(below, minis, close_bal - start, P[F_BOUNDARY_LOWER] == 1):
                e, k_end = END_SIZE_LIMIT, k + 1
                break
            trail = P[D_TYPE] == DD_INTRADAY and not fixed
            code, eq, net, peak, floor = _run_day(
                P, low, high, close, offsets[d], offsets[d + 1], bal, peak, floor, trail, False, 0, 0, 0
            )
            if code == DAY_BREACH:
                e, k_end, bal = END_BREACH, k + 1, eq
                break
            bal = eq
            counted = (not skip_today) and dead == 0
            if (not skip_today) and dead > 0:
                dead -= 1
            skip_today = False
            if w_start < 0 and sides[d] > 0:
                w_start = clock
            if counted:
                if sides[d] > 0:
                    w_traded += 1
                strict = P[F_WIN_STRICT] == 1
                over = net > P[F_WINDAY] if strict else net >= P[F_WINDAY]
                if P[F_HAS_WINDAY] == 1 and over:
                    w_wins += 1
            w_net += net
            w_best = max(w_best, net)
            cum_net += net
            cum_best = max(cum_best, net)
            if P[D_TYPE] == DD_EOD and not fixed:
                peak = max(peak, bal)
                floor = _floor_from_peak(P, peak)
            close_bal = bal
            profit = bal - start
            ok = _eligible(
                P, w_traded, w_wins, w_net, w_best, cum_net, cum_best, npay, profit, clock, w_start
            )
            if not ok:
                continue
            amount = min(caps[min(npay, caps.shape[0] - 1)], P[F_FRAC] * profit // BPS_ONE)
            if P[F_HAS_BUF] == 1:
                amount = min(amount, profit - P[F_BUF])
            if P[F_HAS_WCAP] == 1 and npay + 1 >= P[F_WCAP_FROM]:
                amount = min(amount, w_net * P[F_WCAP] // BPS_ONE)
            if amount < P[F_MINREQ] or amount <= 0:
                continue
            full = 0
            if P[F_HAS_SPLIT_FULL] == 1:
                full = min(amount, max(0, P[F_SPLIT_FULL] - paid))
            share = full + (amount - full) * P[F_SPLIT] // BPS_ONE
            if npay < slots:
                gross[i, npay] = amount
                trader[i, npay] = share
            npay += 1
            paid += amount
            paid_trader += share
            bal -= amount
            if npay == 1:
                first = k + 1
                if P[F_HAS_FLOOR_AFTER] == 1:
                    floor = start + P[F_FLOOR_AFTER]
                    fixed = True
            w_traded, w_wins, w_net, w_best = 0, 0, 0, 0
            w_start = clock
            dead = P[F_DEAD]
            skip_today = P[F_DAY_COUNTS] == 0
            if P[F_HAS_PLIMIT] == 1 and npay >= P[F_PLIMIT]:
                e, k_end = END_PAYOUT_LIMIT, k + 1
                break
        end[i], n_days[i], npay_out[i], first_out[i], bal_out[i] = e, k_end, npay, first, bal
        gross_tot[i], trader_tot[i] = paid, paid_trader


# ------------------------------------------------------------------------------- entry points ----
def calendar_arrays(days: DayPaths) -> tuple[np.ndarray, np.ndarray]:
    """Per day: the trade date's ordinal and rules.calendar_gap (the sequence clock's inputs)."""
    ordinals = np.asarray([date_ordinal(int(t)) for t in days.trade_dates], dtype=np.int64)
    gaps = np.ones(days.n_days, dtype=np.int64)
    if days.n_days > 1:
        gaps[1:] = np.maximum(1, np.diff(ordinals))
    return ordinals, gaps


def _check_seqs(days: DayPaths, seqs: np.ndarray) -> np.ndarray:
    s = np.ascontiguousarray(seqs, dtype=np.int64)
    if s.ndim != 2:
        raise ValueError(f"seqs must be [n_paths, horizon], got shape {s.shape}")
    if s.size and (s.max() >= days.n_days or s.min() < -1):
        raise ValueError(f"seqs hold indices outside -1..{days.n_days - 1}")
    return s


def _day_arrays(days: DayPaths) -> tuple[np.ndarray, ...]:
    as64 = (days.offsets, days.low, days.high, days.close, days.sides, days.max_contracts)
    return tuple(np.ascontiguousarray(a, dtype=np.int64) for a in as64)


def simulate_eval_many(
    plan: PlanRules, days: DayPaths, seqs: np.ndarray, *, soft_daily_loss: str = SOFT_DLL_LIMIT
) -> EvalBatch:
    """One evaluation attempt per row of ``seqs`` (rules.simulate_eval on each row)."""
    P = eval_vector(eval_params(plan, soft_daily_loss))
    s = _check_seqs(days, seqs)
    n = s.shape[0]
    out = [np.zeros(n, dtype=np.int64) for _ in range(5)]
    offsets, low, high, close, sides, max_c = _day_arrays(days)
    _eval_kernel(P, offsets, low, high, close, sides, max_c, s, *out)
    return EvalBatch(*out)


def simulate_funded_many(
    plan: PlanRules,
    days: DayPaths,
    seqs: np.ndarray,
    max_payouts: int | None = None,
    *,
    soft_daily_loss: str = SOFT_DLL_LIMIT,
) -> FundedBatch:
    """One funded life per row of ``seqs`` (rules.simulate_funded on each row)."""
    p = funded_params(plan, soft_daily_loss)
    P = funded_vector(p)
    below, minis = scaling_arrays(p)
    caps = caps_array(p)
    s = _check_seqs(days, seqs)
    n, horizon = s.shape
    slots = payout_slots(p, horizon) if max_payouts is None else max_payouts
    ordinals, gaps = calendar_arrays(days)
    end, n_days, npay, first, bal, gtot, ttot = (np.zeros(n, dtype=np.int64) for _ in range(7))
    gross = np.zeros((n, slots), dtype=np.int64)
    trader = np.zeros((n, slots), dtype=np.int64)
    offsets, low, high, close, sides, max_c = _day_arrays(days)
    _funded_kernel(
        P,
        below,
        minis,
        caps,
        offsets,
        low,
        high,
        close,
        sides,
        max_c,
        ordinals,
        gaps,
        s,
        end,
        n_days,
        npay,
        gross,
        trader,
        gtot,
        ttot,
        first,
        bal,
    )
    return FundedBatch(
        end=end,
        days=n_days,
        n_payouts=npay,
        payouts_gross_cents=gross,
        payouts_trader_cents=trader,
        gross_total_cents=gtot,
        trader_total_cents=ttot,
        first_payout_day=first,
        balance_cents=bal,
    )
