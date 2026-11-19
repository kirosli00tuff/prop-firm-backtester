"""S2 random-entry baseline: the run's own trades placed again at random entry bars.

Comparability rules. One draw places every real trade again, in ledger order, keeping its side, contracts and
hold, at an entry bar drawn uniformly from the bars where the engine would fill an entry: the decision bar
before it is not a session end (the engine ignores signals there), passes the run's entry_allowed mask and is
not refused by the price-limit rule (price-limit: its trade date has a settlement reference and
its close is not at or beyond a stop level, read from the run's own per-bar arrays, sim.price_limits, with the
kernel's test; price_limit_refusals), the entry bar passes the engine's entry_fill_ok (not inside, and its
next minute not inside, a must-be-flat interval), and the hold ends no later than the last bar of that
session segment (a session end, a flat gap or the bar before a flat-by bar), so no draw crosses a session end
or a flat-by. A placement overlapping an earlier one of the same draw is redrawn, MAX_TRIES attempts in all;
the last attempt is then kept and counted. A drawn hold is not cut short where the kernel would force a
PRICE_LIMIT exit (a close beyond a stop level inside the hold): the real trade's hold is kept, as for every
other exit reason.

Fill convention of a placement, mirroring pfb.engine.kernel: entry by market at open[e] with the entry side's
slippage at bar e's CT minute; exit by market at close[e + h] with the exit side's slippage at that bar's CT
minute (none when the real trade exited at its TARGET, a resting limit, as in the kernel); one commission per
fill; gross = floor(side x (exit - entry) x tick_value_cc x contracts / 100), as in the kernel. The hold h
counts bars of exposure: a real exit at a bar's close (SESSION_END) or inside a bar (STOP, TARGET) at bar x
gives h = x - e; a real exit at the open of bar x (SIGNAL, FLAT_BY, PRICE_LIMIT) gives h = x - e - 1, so the
drawn exit is at the close of the last bar the real trade held through the close (the one-bar difference
between that close and the next open is the approximation). A trade's day is the trade date of its exit bar.
(PRICE_LIMIT, fills at the open as SIGNAL does. A STOP queued on a bar locked
against it and filled at the open of a later bar, engine 0.3.1, is an open exit too when the caller passes
the run's exit_after_lock flags (pfb.engine.run.SimOutput.exit_after_lock); without
them it is read as an intrabar STOP.)

Mode "uniform" draws from every eligible bar; mode "tod" keeps the real trade's CT minute of day (so the
costs by time of day are the strategy's) and draws only the day. Mode "cost_matched" makes
exactly the placements of mode "uniform" (same candidates, same seed, same draw count, so draw d of both is
the same set of entry bars) but charges each placement the strategy's OWN costs for that trade: the
commission_cents + slippage_cents the engine charged the real trade in the ledger row it replaces (per trade
as charged, by the trade's own entry and exit fills, event costs included, never an average or a bar's table
value). So every draw of it pays the strategy's exact total cost, and its net percentile ranks the strategy's
gross among the random grosses: a cheaper or dearer timing of fills earns the strategy nothing there.

Every mode runs whenever the baseline runs; there is no switch. The top-level figures (net_percentile,
sharpe_percentile, p_net, p_sharpe) come from one mode, named in `mode`, fixed in code: the one with the
largest p_net of the three, ties going to "cost_matched", then "tod" (docs/TRIALS.md section 7).
The uniform and tod blocks are what they were before, so the reported p_net is never
below the one the two-mode rule gave. All modes stay in the output.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numba import njit

from pfb.analysis.spec import SuiteIntegrityError
from pfb.engine import kernel as K

MIN_DRAWS = 200
MAX_TRIES = 50
QUANTILES = (0.05, 0.50, 0.95)
MODES = ("uniform", "tod", "cost_matched")
COST_MATCHED = "cost_matched"
# the reported mode: the largest p_net; on a tie the first of these (the cost-matched one, the harder bar)
REPORT_ORDER = ("cost_matched", "tod", "uniform")
TOD_SEED_OFFSET = 1
# draw seed of each mode = the run's baseline seed + this; cost_matched shares uniform's draws exactly
SEED_OFFSET = {"uniform": 0, "tod": TOD_SEED_OFFSET, "cost_matched": 0}
# the bars a mode draws from: cost_matched draws as uniform does
CANDIDATE_MODE = {"uniform": "uniform", "tod": "tod", "cost_matched": "uniform"}
COST_COLUMNS = ("commission_cents", "slippage_cents")
TRADING_DAYS_PER_YEAR = 252
MINUTES_PER_DAY = 1440
# exits filled at the OPEN of their exit bar (kernel step 1); PRICE_LIMIT: the price-limit forced exit
OPEN_EXITS = (K.SIGNAL, K.FLAT_BY, K.PRICE_LIMIT)


class BaselineError(ValueError):
    """The baseline cannot be built as the comparability rules require."""


@dataclass(frozen=True)
class BaselineMarket:
    """The usable bars as the kernel saw them, plus what a placement needs (integers and booleans only)."""

    open: np.ndarray  # int64 ticks
    close: np.ndarray  # int64 ticks
    slip_buy: np.ndarray  # int64 cents, whole quantity, at each bar's CT minute
    slip_sell: np.ndarray
    ct_minute: np.ndarray  # int64
    day_idx: np.ndarray  # int64: index of the bar's trade date in the run's daily frame
    entry_ok: np.ndarray  # bool: the engine would fill an entry here
    room: np.ndarray  # int64: last bar of the bar's session segment minus the bar's index
    n_days: int
    commission_cents: int  # per fill, whole quantity
    tick_value_cc: int
    contracts: int


@dataclass(frozen=True)
class BaselineTrades:
    side: np.ndarray  # int64 +1 / -1
    entry_bar: np.ndarray  # int64, the real entry bar
    hold: np.ndarray  # int64 bars of exposure after the entry bar (exit at close[entry + hold])
    exit_slip: np.ndarray  # bool: the exit pays market slippage (False for a TARGET exit)
    own_cost: np.ndarray  # int64 cents: commission + slippage the engine charged this trade (cost_matched)


def segment_room(session_end: np.ndarray, flat_now: np.ndarray) -> np.ndarray:
    """room[i] = j - i for the first j >= i that is a session end, a flat-by bar or followed by one."""
    n = len(session_end)
    stop = session_end | flat_now
    if n > 1:
        stop[:-1] |= flat_now[1:]
    if n:
        stop[-1] = True
    idx = np.where(stop, np.arange(n, dtype=np.int64), n)
    seg_last = np.minimum.accumulate(idx[::-1])[::-1]
    return seg_last - np.arange(n, dtype=np.int64)


def price_limit_refusals(
    close: np.ndarray, upper: np.ndarray, lower: np.ndarray, no_reference: np.ndarray
) -> np.ndarray:
    """Decision bars where the kernel refuses an entry under the price-limit rule (kernel._decide): the
    bar's trade date has no settlement reference, or its close is at or beyond a stop level (close >= upper or
    close <= lower). The levels are the run's own per-bar arrays (pfb.engine.price_limit_inputs), never
    computed again here; where the rule does not act they are the kernel's sentinels: nothing is refused."""
    n = len(close)
    for name, arr in (("upper", upper), ("lower", lower), ("no_reference", no_reference)):
        if len(arr) != n:
            raise SuiteIntegrityError(f"price-limit array {name} has {len(arr)} bars, the run has {n}")
    return no_reference.astype(np.bool_) | (close >= upper) | (close <= lower)


def entry_ok_mask(
    session_end: np.ndarray,
    entry_fill_ok: np.ndarray,
    entry_allowed: np.ndarray,
    refused: np.ndarray | None = None,
) -> np.ndarray:
    """Bars where a pending entry fills: decision bar e - 1 not a session end, allowed and not refused by the
    price-limit rule (price_limit_refusals; None: the rule refuses nothing); e fill-ok."""
    ok = np.zeros(len(session_end), dtype=np.bool_)
    if len(ok) > 1:
        ok[1:] = ~session_end[:-1] & entry_allowed[:-1] & entry_fill_ok[1:]
        if refused is not None:
            ok[1:] &= ~refused[:-1]
    return ok


def own_costs(trades: pl.DataFrame) -> np.ndarray:
    """Per ledger row, the integer cents the engine charged it: commission_cents + slippage_cents, as charged
    (not recomputed from a table, not averaged). A missing, non-integer, null or negative cost is refused."""
    missing = [c for c in COST_COLUMNS if c not in trades.columns]
    if missing:
        raise SuiteIntegrityError(f"the trade ledger lacks {missing}: the cost-matched baseline needs them")
    total = np.zeros(trades.height, dtype=np.int64)
    for col in COST_COLUMNS:
        series = trades[col]
        if not series.dtype.is_integer() or series.null_count():
            raise SuiteIntegrityError(f"the trade ledger's {col} must be integer cents with no nulls")
        values = series.to_numpy().astype(np.int64)
        if bool(np.any(values < 0)):
            raise SuiteIntegrityError(f"the trade ledger's {col} has a negative charge")
        total += values
    return total


def trades_of(trades: pl.DataFrame, after_lock: np.ndarray | None = None) -> BaselineTrades:
    """Side, real entry bar, hold in bars, exit-slippage flag and own cost of each ledger row. after_lock
    (bool per row, SimOutput.exit_after_lock): a row whose exit waited on a lock filled at a bar's open, so
    a STOP among them is an open exit."""
    reasons = [K.EXIT_REASONS.index(r) for r in trades["exit_reason"].to_list()]
    reason = np.asarray(reasons, dtype=np.int64)
    entry = trades["entry_bar"].to_numpy().astype(np.int64)
    exit_ = trades["exit_bar"].to_numpy().astype(np.int64)
    open_exit = np.isin(reason, np.asarray(OPEN_EXITS, dtype=np.int64))
    if after_lock is not None:
        flags = np.asarray(after_lock)
        if flags.dtype != np.bool_ or flags.shape != reason.shape:
            raise SuiteIntegrityError("after_lock must be one bool per trade ledger row")
        open_exit = open_exit | (flags & (reason == K.STOP))
    hold = exit_ - entry - open_exit.astype(np.int64)
    if bool(np.any(hold < 0)):
        raise SuiteIntegrityError("a trade exits at the open of its own entry bar")
    return BaselineTrades(
        side=trades["side"].to_numpy().astype(np.int64),
        entry_bar=entry,
        hold=hold,
        exit_slip=reason != K.TARGET,
        own_cost=own_costs(trades),
    )


# ------------------------------------------------------------------------------------ kernels ----
@njit(cache=True)
def _placement_net(
    e: int, h: int, side: int, exit_slip: bool, open_: np.ndarray, close: np.ndarray,
    slip_buy: np.ndarray, slip_sell: np.ndarray, commission: int, tick_value_cc: int, contracts: int,
) -> int:  # fmt: skip
    x = e + h
    gross = _placement_gross(e, h, side, open_, close, tick_value_cc, contracts)
    slip = slip_buy[e] if side > 0 else slip_sell[e]
    if exit_slip:
        slip += slip_sell[x] if side > 0 else slip_buy[x]
    return gross - 2 * commission - slip


@njit(cache=True)
def _placement_gross(
    e: int, h: int, side: int, open_: np.ndarray, close: np.ndarray, tick_value_cc: int, contracts: int
) -> int:
    return (side * (close[e + h] - open_[e]) * tick_value_cc * contracts) // K.CC_PER_CENT


@njit(cache=True)
def _draws(
    cand: np.ndarray, lo: np.ndarray, hi: np.ndarray, side: np.ndarray, hold: np.ndarray,
    exit_slip: np.ndarray, open_: np.ndarray, close: np.ndarray, slip_buy: np.ndarray,
    slip_sell: np.ndarray, day_idx: np.ndarray, n_days: int, commission: int, tick_value_cc: int,
    contracts: int, draws: int, seed: int, max_tries: int, own_cost: np.ndarray, use_own_cost: bool,
) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """daily[d, day] net cents of draw d; overlaps[d] placements kept after max_tries overlapping attempts.
    The placements depend on the seed and candidates only, never on costs; use_own_cost charges trade t its
    own_cost[t] instead of the drawn bars' slippage and two commissions."""
    np.random.seed(seed)
    n_bars = open_.shape[0]
    occupied = np.full(n_bars, -1, dtype=np.int64)
    daily = np.zeros((draws, n_days), dtype=np.int64)
    overlaps = np.zeros(draws, dtype=np.int64)
    for d in range(draws):
        for t in range(side.shape[0]):
            h = hold[t]
            e = -1
            free = False
            for _ in range(max_tries):
                e = int(cand[lo[t] + np.random.randint(0, hi[t] - lo[t])])
                free = True
                for j in range(e, e + h + 1):
                    if occupied[j] == d:
                        free = False
                        break
                if free:
                    break
            if not free:
                overlaps[d] += 1
            for j in range(e, e + h + 1):
                occupied[j] = d
            if use_own_cost:
                net = _placement_gross(e, h, side[t], open_, close, tick_value_cc, contracts) - own_cost[t]
            else:
                net = _placement_net(e, h, side[t], exit_slip[t], open_, close, slip_buy, slip_sell,
                                     commission, tick_value_cc, contracts)  # fmt: skip
            daily[d, day_idx[e + h]] += net
    return daily, overlaps


# ------------------------------------------------------------------------------- candidates ----
def _candidates(
    market: BaselineMarket, trades: BaselineTrades, mode: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eligible bars sorted by (group, room) and, per trade, the slice [lo, hi) it draws from."""
    if mode not in CANDIDATE_MODE:
        raise BaselineError(f"mode must be one of {MODES}, got {mode!r}")
    mode = CANDIDATE_MODE[mode]
    bars = np.flatnonzero(market.entry_ok).astype(np.int64)
    if mode == "uniform":
        group = np.zeros(len(bars), dtype=np.int64)
        trade_group = np.zeros(len(trades.side), dtype=np.int64)
        n_groups = 1
    elif mode == "tod":
        group = market.ct_minute[bars]
        trade_group = market.ct_minute[trades.entry_bar]
        n_groups = MINUTES_PER_DAY
    else:
        raise BaselineError(f"mode must be one of {MODES}, got {mode!r}")
    room = market.room[bars]
    order = np.lexsort((room, group))
    bars, group, room = bars[order], group[order], room[order]
    starts = np.searchsorted(group, np.arange(n_groups + 1, dtype=np.int64), side="left")
    width = int(room.max(initial=0)) + 2  # a hold above every room maps past its group's block
    key = group * width + room
    lo = np.searchsorted(key, trade_group * width + np.minimum(trades.hold, width - 1), side="left")
    hi = starts[trade_group + 1]
    if bool(np.any(lo >= hi)):
        raise BaselineError(f"{int((lo >= hi).sum())} trade(s) have no eligible entry bar in mode {mode}")
    return bars, lo.astype(np.int64), hi.astype(np.int64)


def check_own_placements(market: BaselineMarket, trades: BaselineTrades) -> None:
    """Every real trade is itself an eligible placement; otherwise the rules disagree with the engine."""
    if len(trades.side) == 0:
        return
    ok = market.entry_ok[trades.entry_bar] & (market.room[trades.entry_bar] >= trades.hold)
    if not bool(np.all(ok)):
        bad = int(np.flatnonzero(~ok)[0])
        raise SuiteIntegrityError(
            f"real trade {bad} (entry bar {int(trades.entry_bar[bad])}) is not an eligible placement: the "
            "baseline's rules disagree with the engine"
        )


def placement_nets(market: BaselineMarket, trades: BaselineTrades, entry_bars: np.ndarray) -> np.ndarray:
    """Net cents of each trade placed at the given entry bars (the fill convention above)."""
    out = np.empty(len(entry_bars), dtype=np.int64)
    for t, e in enumerate(entry_bars.tolist()):
        out[t] = _placement_net(int(e), int(trades.hold[t]), int(trades.side[t]), bool(trades.exit_slip[t]),
                                market.open, market.close, market.slip_buy, market.slip_sell,
                                market.commission_cents, market.tick_value_cc, market.contracts)  # fmt: skip
    return out


# ---------------------------------------------------------------------------------- summaries ----
def sharpe_rows(daily: np.ndarray) -> np.ndarray:
    """Annual Sharpe of each row: mean / population sd x sqrt(252), every day of the row included (the
    summary's convention, pfb.metrics.core); a row with sd 0 gives NaN."""
    x = daily.astype(np.float64)
    mean = x.mean(axis=1)
    sd = x.std(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(sd > 0, mean / sd, np.nan)
    return out * math.sqrt(TRADING_DAYS_PER_YEAR)


def _position(value: float, draws: np.ndarray) -> dict[str, object]:
    """Percentile (share strictly below plus half the ties) and one-sided p = (1 + #draws >= value) / (n + 1).
    NaN draws (sd 0) count as below a finite value."""
    n = len(draws)
    finite = np.where(np.isnan(draws), -np.inf, draws)
    below = int((finite < value).sum())
    ties = int((finite == value).sum())
    at_or_above = int((finite >= value).sum())
    return {"percentile": (below + 0.5 * ties) / n, "p": (1 + at_or_above) / (n + 1)}


def _describe(values: np.ndarray, *, money: bool) -> dict[str, object]:
    clean = values[~np.isnan(values)] if values.dtype.kind == "f" else values
    if len(clean) == 0:
        return {"mean": None, "sd": None, "q05": None, "q50": None, "q95": None}
    qs = np.quantile(clean, QUANTILES, method="inverted_cdf")
    conv = (lambda v: int(v)) if money else (lambda v: float(v))
    return {"mean": float(clean.mean()), "sd": float(clean.std()), "q05": conv(qs[0]), "q50": conv(qs[1]),
            "q95": conv(qs[2])}  # fmt: skip


@dataclass(frozen=True)
class ModeResult:
    mode: str
    seed: int
    nets: np.ndarray  # int64 per draw
    sharpes: np.ndarray  # float per draw (NaN when sd 0)
    overlaps: np.ndarray  # int64 per draw


def run_mode(market: BaselineMarket, trades: BaselineTrades, mode: str, draws: int, seed: int) -> ModeResult:
    """All draws of one mode (vectorised in Numba: one pass over the trades per draw)."""
    if len(trades.side) == 0:
        zeros = np.zeros((draws, market.n_days), dtype=np.int64)
        return ModeResult(mode, seed, zeros.sum(axis=1), sharpe_rows(zeros), np.zeros(draws, dtype=np.int64))
    cand, lo, hi = _candidates(market, trades, mode)
    own_cost = np.ascontiguousarray(trades.own_cost)
    if own_cost.dtype.kind not in "iu" or own_cost.shape != trades.side.shape:
        raise BaselineError("own_cost must be integer cents, one per trade")
    daily, overlaps = _draws(
        cand, lo, hi, trades.side, trades.hold, trades.exit_slip, market.open, market.close, market.slip_buy,
        market.slip_sell, market.day_idx, market.n_days, market.commission_cents, market.tick_value_cc,
        market.contracts, draws, seed, MAX_TRIES, own_cost.astype(np.int64), mode == COST_MATCHED,
    )  # fmt: skip
    return ModeResult(mode, seed, daily.sum(axis=1), sharpe_rows(daily), overlaps)


def mode_json(result: ModeResult, strategy_net: int, strategy_sharpe: float) -> dict[str, object]:
    net_pos = _position(float(strategy_net), result.nets.astype(np.float64))
    sharpe_pos = _position(strategy_sharpe, result.sharpes) if not math.isnan(strategy_sharpe) else None
    return {
        "mode": result.mode,
        "seed": result.seed,
        "draws": len(result.nets),
        "overlaps": int(result.overlaps.sum()),
        "draws_with_overlap": int((result.overlaps > 0).sum()),
        "net": {"strategy": strategy_net, "percentile": net_pos["percentile"], "p": net_pos["p"],
                **_describe(result.nets, money=True)},
        "sharpe": {"strategy": None if math.isnan(strategy_sharpe) else strategy_sharpe,
                   "percentile": None if sharpe_pos is None else sharpe_pos["percentile"],
                   "p": None if sharpe_pos is None else sharpe_pos["p"],
                   **_describe(result.sharpes, money=False)},
    }  # fmt: skip


def reported_mode(modes: Mapping[str, Any]) -> str:
    """The mode whose figures the top level reports, fixed in code (no file key, no flag): the largest net p
    of the three modes, ties to "cost_matched", then "tod" (TRIALS.md section 7). The largest
    of the p-values is conservative while any is valid, and it is never below the two-mode rule's p (the
    uniform and tod blocks are unchanged), so adding the cost-matched mode cannot make the bar easier."""
    best = max(modes[m]["net"]["p"] for m in REPORT_ORDER)
    return next(m for m in REPORT_ORDER if modes[m]["net"]["p"] == best)


def random_baseline(
    market: BaselineMarket, trades: BaselineTrades, strategy_daily: np.ndarray, *, draws: int, seed: int
) -> tuple[dict[str, object], pl.DataFrame]:
    """Every mode (always all three, same draw count; cost_matched with uniform's seed), the reported mode's
    four figures copied to the top level with its name in `mode` (docs/TRIALS.md section 7), and the draws
    frame."""
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < MIN_DRAWS:
        raise BaselineError(f"the random baseline needs at least {MIN_DRAWS} draws, got {draws!r}")
    if len(strategy_daily) != market.n_days:
        raise SuiteIntegrityError("the strategy's daily nets do not cover the run's days")
    check_own_placements(market, trades)
    strategy_net = int(strategy_daily.sum())
    strategy_sharpe = float(sharpe_rows(strategy_daily.reshape(1, -1))[0])
    out: dict[str, object] = {"draws": draws, "seed": seed, "trades": len(trades.side)}
    frames: list[pl.DataFrame] = []
    for mode in MODES:
        result = run_mode(market, trades, mode, draws, seed + SEED_OFFSET[mode])
        out[mode] = mode_json(result, strategy_net, strategy_sharpe)
        frames.append(pl.DataFrame({
            "mode": [mode] * draws, "draw": np.arange(draws, dtype=np.int64), "net_cents": result.nets,
            "sharpe_annual": result.sharpes, "overlaps": result.overlaps,
        }))  # fmt: skip
    mode = reported_mode(out)
    head = out[mode]
    assert isinstance(head, dict)
    net, sharpe = head["net"], head["sharpe"]
    assert isinstance(net, dict) and isinstance(sharpe, dict)
    out |= {"mode": mode, "net_percentile": net["percentile"], "sharpe_percentile": sharpe["percentile"],
            "p_net": net["p"], "p_sharpe": sharpe["p"]}  # fmt: skip
    return out, pl.concat(frames)
