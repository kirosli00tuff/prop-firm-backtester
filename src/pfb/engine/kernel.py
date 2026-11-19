"""Numba bar kernel (ADR-001 section 8 with A6 and A14 c). Integers and booleans only: no float enters.

Money inside the kernel: gross P&L in int64 units of 1e-4 USD (tick_value_cc), floored to cents once per
amount that leaves (a trade's gross, an equity mark). Costs are integer cents charged at each fill.
realized = sum of closed trades' net_cents minus the open position's entry costs;
equity = realized + floor(open gross / 100).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

CC_PER_CENT = 100
BUY, SELL = 0, 1

# exit reasons (PRICE_LIMIT: the price-limit forced exit at the open after a close at or beyond a stop level)
SIGNAL, STOP, TARGET, FLAT_BY, SESSION_END, PRICE_LIMIT = 0, 1, 2, 3, 4, 5
EXIT_REASONS = ("SIGNAL", "STOP", "TARGET", "FLAT_BY", "SESSION_END", "PRICE_LIMIT")
# S_PEND_EXIT values: a SIGNAL or PRICE_LIMIT exit queued at a decision, or a STOP or FLAT_BY exit that met a
# locked market and waits for the open of the first bar that trades through
PEND_SIGNAL, PEND_PRICE_LIMIT, PEND_STOP, PEND_FLAT_BY = 1, 2, 3, 4

# trade matrix columns
(
    T_ENTRY_IDX,
    T_EXIT_IDX,
    T_SIDE,
    T_ENTRY_PX,
    T_EXIT_PX,
    T_GROSS,
    T_COMM,
    T_SLIP,
    T_NET,
    T_MAE,
    T_MFE,
    T_REASON,
    T_AFTER_LOCK,  # 1: the exit waited on a market locked against it and filled at a later open
) = range(13)
N_TRADE_COLS = 13

# equity matrix columns
E_OPEN, E_LOW, E_HIGH, E_CLOSE, E_POS = range(5)
N_EQ_COLS = 5

# counters
COUNTER_NAMES = (
    "entry_signals_in_position",
    "entry_signals_at_session_end",  # a session-end bar with an entry flag, in or out of position
    "exit_signals_at_session_end",  # a session-end bar with an exit flag while in position (closed at close)
    "conflicting_signals",
    "entries_dropped",
    "entries_blocked",
    "uncalibrated_fills",
    "price_limit_entries_refused",  # an entry decided at or beyond a stop level
    "price_limit_entries_no_reference",  # an entry decided on a date without a prior settlement reference
    "price_limit_exits",  # forced price-limit exits queued
    "locked_exit_deferrals",  # bars on which an exit waited on a locked market
    "locked_exit_fills",  # exits filled after waiting on a lock (the flagged trips)
    "locked_exits_at_session_end",  # a lock still holding when the session ended
)
(C_IN_POS, C_ENTRY_AT_END, C_EXIT_AT_END, C_CONFLICT, C_DROPPED, C_BLOCKED, C_UNCALIBRATED, C_PL_ZONE,
 C_PL_NOREF, C_PL_EXITS, C_LOCK_DEFER, C_LOCK_FILLS, C_LOCK_END) = range(13)  # fmt: skip
N_COUNTERS = 13

# state slots
(
    S_POS,
    S_ENTRY_PX,
    S_STOP_ON,
    S_STOP_PX,
    S_TGT_ON,
    S_TGT_PX,
    S_ENTRY_IDX,
    S_ENTRY_COMM,
    S_ENTRY_SLIP,
    S_MIN_EXC,
    S_MAX_EXC,
    S_REALIZED,
    S_PEND_ENTRY,
    S_PEND_STOP,
    S_PEND_TGT,
    S_PEND_EXIT,
    S_NTRADES,
    S_EQ_LO,
    S_EQ_HI,
    S_LOCK_WAIT,  # 1 while an exit waits on a locked market
    S_LOCK_BAR,  # 1 + the last bar whose deferral was counted
) = range(21)
N_STATE = 21

# market inputs: columns of the per-bar int64 matrix `px`; P_SLIP_PL_*: what a PRICE_LIMIT exit fill pays at
# the bar (the cost model's event-window table)
P_OPEN, P_HIGH, P_LOW, P_CLOSE, P_SLIP_BUY, P_SLIP_SELL, P_SLIP_PL_BUY, P_SLIP_PL_SELL = range(8)
N_PX = 8
# per-bar bool matrix `flags`
(F_SESSION_END, F_FLAT_BY, F_ENTRY_FILL_OK, F_ENTRY_ALLOWED, F_CALIBRATED, F_LONG, F_SHORT, F_EXIT,
 F_PL_NO_REF) = range(9)  # fmt: skip
N_FLAGS = 9
# per-bar int64 matrix `lim` (price-limit thresholds in ticks; sentinels where the rule does not act):
# a close >= L_UPPER or <= L_LOWER is at or beyond a stop level; a sell is locked while high <= L_LOCK_DOWN,
# a buy while low >= L_LOCK_UP
L_UPPER, L_LOWER, L_LOCK_UP, L_LOCK_DOWN = range(4)
N_LIM = 4
NO_UPPER = np.iinfo(np.int64).max
NO_LOWER = np.iinfo(np.int64).min
# scalars: commission per fill (cents, whole quantity), tick value (1e-4 USD), contracts
K_COMMISSION, K_TICK_VALUE_CC, K_CONTRACTS = range(3)


@njit(cache=True)
def _realized(st: np.ndarray) -> int:
    if st[S_POS] == 0:
        return st[S_REALIZED]
    return st[S_REALIZED] - st[S_ENTRY_COMM] - st[S_ENTRY_SLIP]


@njit(cache=True)
def _mark(st: np.ndarray, price: int, k: np.ndarray) -> int:
    if st[S_POS] == 0:
        return st[S_REALIZED]
    gross_cc = st[S_POS] * (price - st[S_ENTRY_PX]) * k[K_TICK_VALUE_CC] * k[K_CONTRACTS]
    return _realized(st) + gross_cc // CC_PER_CENT


@njit(cache=True)
def _point(st: np.ndarray, equity: int) -> None:
    st[S_EQ_LO] = min(st[S_EQ_LO], equity)
    st[S_EQ_HI] = max(st[S_EQ_HI], equity)


@njit(cache=True)
def _touch(st: np.ndarray, price: int, k: np.ndarray) -> None:
    """The open position is exposed to `price`: equity bounds and excursions."""
    _point(st, _mark(st, price, k))
    exc = st[S_POS] * (price - st[S_ENTRY_PX])
    st[S_MIN_EXC] = min(st[S_MIN_EXC], exc)
    st[S_MAX_EXC] = max(st[S_MAX_EXC], exc)


@njit(cache=True)
def _market_slip(counters: np.ndarray, px: np.ndarray, flags: np.ndarray, i: int, side_idx: int) -> int:
    if not flags[i, F_CALIBRATED]:
        counters[C_UNCALIBRATED] += 1
    return px[i, P_SLIP_BUY] if side_idx == BUY else px[i, P_SLIP_SELL]


@njit(cache=True)
def _exit_slip(
    counters: np.ndarray, px: np.ndarray, flags: np.ndarray, i: int, side_idx: int, reason: int
) -> int:
    """An exit fill's slippage: none for a TARGET (a resting limit), the forced-exit column for PRICE_LIMIT,
    else the bar's market slippage."""
    if reason == TARGET:
        return 0
    if reason != PRICE_LIMIT:
        return _market_slip(counters, px, flags, i, side_idx)
    if not flags[i, F_CALIBRATED]:
        counters[C_UNCALIBRATED] += 1
    return px[i, P_SLIP_PL_BUY] if side_idx == BUY else px[i, P_SLIP_PL_SELL]


@njit(cache=True)
def _open_position(
    st: np.ndarray, counters: np.ndarray, px: np.ndarray, flags: np.ndarray, k: np.ndarray, i: int
) -> None:
    side = st[S_PEND_ENTRY]
    price = px[i, P_OPEN]
    st[S_POS] = side
    st[S_ENTRY_PX] = price
    st[S_ENTRY_IDX] = i
    st[S_ENTRY_COMM] = k[K_COMMISSION]
    st[S_ENTRY_SLIP] = _market_slip(counters, px, flags, i, BUY if side > 0 else SELL)
    st[S_STOP_ON] = 1 if st[S_PEND_STOP] > 0 else 0
    st[S_STOP_PX] = price - side * st[S_PEND_STOP]
    st[S_TGT_ON] = 1 if st[S_PEND_TGT] > 0 else 0
    st[S_TGT_PX] = price + side * st[S_PEND_TGT]
    st[S_MIN_EXC] = 0
    st[S_MAX_EXC] = 0
    _point(st, _realized(st))


@njit(cache=True)
def _close_position(
    st: np.ndarray,
    counters: np.ndarray,
    trades: np.ndarray,
    px: np.ndarray,
    flags: np.ndarray,
    k: np.ndarray,
    i: int,
    price: int,
    reason: int,
) -> None:
    side = st[S_POS]
    slip = _exit_slip(counters, px, flags, i, SELL if side > 0 else BUY, reason)
    gross = (side * (price - st[S_ENTRY_PX]) * k[K_TICK_VALUE_CC] * k[K_CONTRACTS]) // CC_PER_CENT
    comm = st[S_ENTRY_COMM] + k[K_COMMISSION]
    slippage = st[S_ENTRY_SLIP] + slip
    net = gross - comm - slippage
    row = st[S_NTRADES]
    trades[row, T_ENTRY_IDX] = st[S_ENTRY_IDX]
    trades[row, T_EXIT_IDX] = i
    trades[row, T_SIDE] = side
    trades[row, T_ENTRY_PX] = st[S_ENTRY_PX]
    trades[row, T_EXIT_PX] = price
    trades[row, T_GROSS] = gross
    trades[row, T_COMM] = comm
    trades[row, T_SLIP] = slippage
    trades[row, T_NET] = net
    trades[row, T_MAE] = -st[S_MIN_EXC]
    trades[row, T_MFE] = st[S_MAX_EXC]
    trades[row, T_REASON] = reason
    trades[row, T_AFTER_LOCK] = st[S_LOCK_WAIT]
    st[S_NTRADES] = row + 1
    if st[S_LOCK_WAIT] == 1:
        if reason == SESSION_END:
            counters[C_LOCK_END] += 1
        else:
            counters[C_LOCK_FILLS] += 1
        st[S_LOCK_WAIT] = 0
    st[S_REALIZED] += net
    st[S_POS] = 0
    st[S_PEND_EXIT] = 0
    _point(st, st[S_REALIZED])


@njit(cache=True)
def _locked(st: np.ndarray, px: np.ndarray, lim: np.ndarray, i: int) -> bool:
    """R-08: the open position's exit must trade against a lock on bar i (a long at limit-down, a short at
    limit-up), so it is not filled on this bar."""
    if st[S_POS] > 0:
        return px[i, P_HIGH] <= lim[i, L_LOCK_DOWN]
    if st[S_POS] < 0:
        return px[i, P_LOW] >= lim[i, L_LOCK_UP]
    return False


@njit(cache=True)
def _defer(st: np.ndarray, counters: np.ndarray, i: int) -> None:
    st[S_LOCK_WAIT] = 1
    if st[S_LOCK_BAR] != i + 1:
        counters[C_LOCK_DEFER] += 1
        st[S_LOCK_BAR] = i + 1


@njit(cache=True)
def _pending_reason(pending: int) -> int:
    if pending == PEND_PRICE_LIMIT:
        return PRICE_LIMIT
    if pending == PEND_STOP:
        return STOP
    if pending == PEND_FLAT_BY:
        return FLAT_BY
    return SIGNAL


@njit(cache=True)
def _at_open(
    st: np.ndarray,
    counters: np.ndarray,
    trades: np.ndarray,
    px: np.ndarray,
    flags: np.ndarray,
    k: np.ndarray,
    lim: np.ndarray,
    i: int,
) -> None:
    """Steps 1 and 2: pending exit (SIGNAL, PRICE_LIMIT, or a STOP or FLAT_BY that met a lock), pending entry,
    then flat-by, all at open[i]; an exit against a locked market stays pending for a bar that trades through
    and then fills at that bar's open with its own reason."""
    if st[S_POS] != 0:
        _touch(st, px[i, P_OPEN], k)
        if st[S_PEND_EXIT] != 0:
            if _locked(st, px, lim, i):
                _defer(st, counters, i)
            else:
                reason = _pending_reason(st[S_PEND_EXIT])
                _close_position(st, counters, trades, px, flags, k, i, px[i, P_OPEN], reason)
    if st[S_POS] == 0:
        st[S_PEND_EXIT] = 0
    if st[S_PEND_ENTRY] != 0:
        if st[S_POS] == 0 and flags[i, F_ENTRY_FILL_OK]:
            _open_position(st, counters, px, flags, k, i)
        else:
            counters[C_DROPPED] += 1
        st[S_PEND_ENTRY] = 0
    if flags[i, F_FLAT_BY] and st[S_POS] != 0:
        if _locked(st, px, lim, i):
            _defer(st, counters, i)
            if st[S_PEND_EXIT] == 0:
                st[S_PEND_EXIT] = PEND_FLAT_BY
        else:
            _close_position(st, counters, trades, px, flags, k, i, px[i, P_OPEN], FLAT_BY)


@njit(cache=True)
def _intrabar(
    st: np.ndarray,
    counters: np.ndarray,
    trades: np.ndarray,
    px: np.ndarray,
    flags: np.ndarray,
    k: np.ndarray,
    lim: np.ndarray,
    i: int,
) -> None:
    """Step 3: the stop (touched; a gap fills at the open) before the target (one tick through), else the
    position stays open through the bar and is exposed to its whole range. A stop hit on a bar locked against
    it is queued (PEND_STOP) and fills at the open of the first bar that trades through, reason STOP (a
    pending SIGNAL or PRICE_LIMIT exit fills at that same open; so it keeps its own reason)."""
    side = st[S_POS]
    if side == 0:
        return
    adverse = px[i, P_LOW] if side > 0 else px[i, P_HIGH]
    favourable = px[i, P_HIGH] if side > 0 else px[i, P_LOW]
    stop_hit = st[S_STOP_ON] == 1 and side * (adverse - st[S_STOP_PX]) <= 0
    if stop_hit and _locked(st, px, lim, i):
        _defer(st, counters, i)
        if st[S_PEND_EXIT] == 0:
            st[S_PEND_EXIT] = PEND_STOP
        stop_hit = False
    target_hit = st[S_TGT_ON] == 1 and side * (favourable - st[S_TGT_PX]) >= 1
    if stop_hit:
        fill = min(st[S_STOP_PX], px[i, P_OPEN]) if side > 0 else max(st[S_STOP_PX], px[i, P_OPEN])
        _touch(st, favourable, k)
        _touch(st, fill, k)
        _close_position(st, counters, trades, px, flags, k, i, fill, STOP)
    elif target_hit:
        _touch(st, adverse, k)
        _touch(st, st[S_TGT_PX], k)
        _close_position(st, counters, trades, px, flags, k, i, st[S_TGT_PX], TARGET)
    else:
        _touch(st, adverse, k)
        _touch(st, favourable, k)


@njit(cache=True)
def _at_close(
    st: np.ndarray,
    counters: np.ndarray,
    trades: np.ndarray,
    px: np.ndarray,
    flags: np.ndarray,
    k: np.ndarray,
    stops: np.ndarray,
    targets: np.ndarray,
    lim: np.ndarray,
    i: int,
) -> None:
    """Step 4: session end exits at close[i] and ignores the bar's signals (an entry flag is counted, an exit
    flag only while in position: flat, it asks for nothing); otherwise the decision. price-limit at the
    decision: a
    close at or beyond a stop level forces the exit (PRICE_LIMIT, at the next open) and refuses an entry, and
    no entry is taken on a date without a prior settlement reference. A session-end exit on a bar locked
    against the position is a lock still holding at the session end (counted; the run is refused by name)."""
    long_sig, short_sig, exit_sig = flags[i, F_LONG], flags[i, F_SHORT], flags[i, F_EXIT]
    if flags[i, F_SESSION_END]:
        if long_sig or short_sig:
            counters[C_ENTRY_AT_END] += 1
        if exit_sig and st[S_POS] != 0:
            counters[C_EXIT_AT_END] += 1
        if st[S_POS] != 0:
            if _locked(st, px, lim, i):
                st[S_LOCK_WAIT] = 1  # _close_position counts it in locked_exits_at_session_end
            _close_position(st, counters, trades, px, flags, k, i, px[i, P_CLOSE], SESSION_END)
        return
    beyond = px[i, P_CLOSE] >= lim[i, L_UPPER] or px[i, P_CLOSE] <= lim[i, L_LOWER]
    if st[S_POS] != 0:
        # a forced exit supersedes a queued SIGNAL exit, not a STOP or FLAT_BY already waiting on a lock
        if beyond and (st[S_PEND_EXIT] == 0 or st[S_PEND_EXIT] == PEND_SIGNAL):
            st[S_PEND_EXIT] = PEND_PRICE_LIMIT
            counters[C_PL_EXITS] += 1
        elif exit_sig and st[S_PEND_EXIT] == 0:
            st[S_PEND_EXIT] = PEND_SIGNAL
        if long_sig or short_sig:
            counters[C_IN_POS] += 1
        return
    if long_sig and short_sig:
        counters[C_CONFLICT] += 1
    elif long_sig or short_sig:
        if not flags[i, F_ENTRY_ALLOWED]:
            counters[C_BLOCKED] += 1
        elif flags[i, F_PL_NO_REF]:
            counters[C_PL_NOREF] += 1
        elif beyond:
            counters[C_PL_ZONE] += 1
        else:
            st[S_PEND_ENTRY] = 1 if long_sig else -1
            st[S_PEND_STOP] = stops[i]
            st[S_PEND_TGT] = targets[i]


@njit(cache=True)
def simulate_bars(
    px: np.ndarray, flags: np.ndarray, stops: np.ndarray, targets: np.ndarray, k: np.ndarray, lim: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run every bar; returns (trades[n_trades, 12], equity[n, 5], counters[N_COUNTERS]), all int64."""
    n = px.shape[0]
    st = np.zeros(N_STATE, dtype=np.int64)
    trades = np.zeros((n, N_TRADE_COLS), dtype=np.int64)
    equity = np.zeros((n, N_EQ_COLS), dtype=np.int64)
    counters = np.zeros(N_COUNTERS, dtype=np.int64)
    for i in range(n):
        eq_open = _mark(st, px[i, P_OPEN], k)
        st[S_EQ_LO] = eq_open
        st[S_EQ_HI] = eq_open
        _at_open(st, counters, trades, px, flags, k, lim, i)
        _intrabar(st, counters, trades, px, flags, k, lim, i)
        _at_close(st, counters, trades, px, flags, k, stops, targets, lim, i)
        eq_close = _mark(st, px[i, P_CLOSE], k)
        _point(st, eq_close)
        equity[i, E_OPEN] = eq_open
        equity[i, E_LOW] = st[S_EQ_LO]
        equity[i, E_HIGH] = st[S_EQ_HI]
        equity[i, E_CLOSE] = eq_close
        equity[i, E_POS] = st[S_POS] * k[K_CONTRACTS]
    return trades[: st[S_NTRADES]], equity, counters


# ----------------------------------------------------------------- Python side ----
@dataclass(frozen=True)
class KernelInputs:
    """Per usable bar: prices in ticks, the slippage columns (cents for the whole quantity), flags.
    slip_pl_buy/slip_pl_sell: what a PRICE_LIMIT exit fill pays at the bar (None: slip_buy/slip_sell)."""

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    slip_buy: np.ndarray
    slip_sell: np.ndarray
    session_end: np.ndarray
    flat_by: np.ndarray
    entry_fill_ok: np.ndarray
    entry_allowed: np.ndarray
    calibrated: np.ndarray
    entry_long: np.ndarray
    entry_short: np.ndarray
    exit: np.ndarray
    stop_ticks: np.ndarray
    target_ticks: np.ndarray
    commission_cents: int
    tick_value_cc: int
    contracts: int
    # price-limit per bar (pfb.engine.price_limit_inputs); None: the rule does not act (sentinels)
    pl_upper: np.ndarray | None = None
    pl_lower: np.ndarray | None = None
    pl_lock_up: np.ndarray | None = None
    pl_lock_down: np.ndarray | None = None
    pl_no_reference: np.ndarray | None = None
    slip_pl_buy: np.ndarray | None = None
    slip_pl_sell: np.ndarray | None = None


INT_FIELDS = ("open", "high", "low", "close", "slip_buy", "slip_sell", "stop_ticks", "target_ticks")
LIM_FIELDS = ("pl_upper", "pl_lower", "pl_lock_up", "pl_lock_down")
LIM_SENTINELS = (NO_UPPER, NO_LOWER, NO_UPPER, NO_LOWER)
BOOL_FIELDS = ("session_end", "flat_by", "entry_fill_ok", "entry_allowed", "calibrated", "entry_long",
               "entry_short", "exit")  # fmt: skip


@dataclass(frozen=True)
class KernelOutput:
    trades: np.ndarray
    equity: np.ndarray
    counters: dict[str, int]


class KernelInputError(TypeError):
    """A kernel input is not an integer or boolean array of the bars' length (no float enters the kernel)."""


def _check_inputs(inputs: KernelInputs) -> int:
    n = len(inputs.open)
    for name in INT_FIELDS + BOOL_FIELDS:
        arr = getattr(inputs, name)
        kind = "b" if name in BOOL_FIELDS else "i"
        if not isinstance(arr, np.ndarray) or arr.ndim != 1 or len(arr) != n or arr.dtype.kind != kind:
            raise KernelInputError(f"kernel input {name} must be a 1-d {kind} array of length {n}")
    for name in ("commission_cents", "tick_value_cc", "contracts"):
        value = getattr(inputs, name)
        if type(value) is not int or value < 0:
            raise KernelInputError(f"kernel scalar {name} must be a non-negative int, got {value!r}")
    if inputs.contracts < 1 or inputs.tick_value_cc < 1:
        raise KernelInputError("contracts and tick_value_cc must be >= 1")
    for name in (*LIM_FIELDS, "pl_no_reference", "slip_pl_buy", "slip_pl_sell"):
        arr = getattr(inputs, name)
        kind = "b" if name == "pl_no_reference" else "i"
        if arr is not None and (
            not isinstance(arr, np.ndarray) or arr.ndim != 1 or len(arr) != n or arr.dtype.kind != kind
        ):
            raise KernelInputError(f"kernel input {name} must be None or a 1-d {kind} array of length {n}")
    return n


def run_kernel(inputs: KernelInputs) -> KernelOutput:
    """Validate the inputs (integers and booleans only), pack them and run the Numba kernel."""
    n = _check_inputs(inputs)
    px = np.empty((n, N_PX), dtype=np.int64)
    for col, name in enumerate(("open", "high", "low", "close", "slip_buy", "slip_sell")):
        px[:, col] = getattr(inputs, name)
    px[:, P_SLIP_PL_BUY] = inputs.slip_buy if inputs.slip_pl_buy is None else inputs.slip_pl_buy
    px[:, P_SLIP_PL_SELL] = inputs.slip_sell if inputs.slip_pl_sell is None else inputs.slip_pl_sell
    flags = np.empty((n, N_FLAGS), dtype=np.bool_)
    for col, name in enumerate(BOOL_FIELDS):
        flags[:, col] = getattr(inputs, name)
    flags[:, F_PL_NO_REF] = False if inputs.pl_no_reference is None else inputs.pl_no_reference
    lim = np.empty((n, N_LIM), dtype=np.int64)
    for col, (name, sentinel) in enumerate(zip(LIM_FIELDS, LIM_SENTINELS, strict=True)):
        arr = getattr(inputs, name)
        lim[:, col] = sentinel if arr is None else arr
    k = np.array([inputs.commission_cents, inputs.tick_value_cc, inputs.contracts], dtype=np.int64)
    stops = inputs.stop_ticks.astype(np.int64)
    targets = inputs.target_ticks.astype(np.int64)
    trades, equity, counters = simulate_bars(px, flags, stops, targets, k, lim)
    named: dict[str, int] = {name: int(v) for name, v in zip(COUNTER_NAMES, counters.tolist(), strict=True)}
    return KernelOutput(trades=trades, equity=equity, counters=named)


__all__ = ["COUNTER_NAMES", "EXIT_REASONS", "KernelInputError", "KernelInputs", "KernelOutput", "run_kernel"]
