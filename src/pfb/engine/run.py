"""Run a strategy over one root's bars (ADR-001 sections 8 and 9 with A1, A3, A5 g, A6, A7, A9, A14 c d).

`simulate` is the engine proper and takes every market input as data (tests drive it with hand-built frames);
`run_strategy` connects it to the gateway or the synthetic bars, the sessions, the costs, the fees and the
metrics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import polars as pl

from pfb.analysis.spec import SuiteSpec
from pfb.costs.events import EVENT_COSTS_OFF, EVENT_COSTS_ON, EventCostModel, load_cost_model_for
from pfb.costs.fees import FeeSchedule, load_fees
from pfb.costs.model import CostModel, EventCostsOff
from pfb.data.catalog import load_catalog
from pfb.data.gateway import DataGateway, provenance_warnings
from pfb.data.paths import load_paths
from pfb.data.windows import load_windows
from pfb.engine import ENGINE_VERSION
from pfb.engine import kernel as K
from pfb.engine.lookahead import lookahead_check
from pfb.engine.permit import PermitRequest, PermitRequired, RunPermit, check_permit
from pfb.engine.price_limit_inputs import (
    STATUS_DCB,
    STATUS_NO_REFERENCE,
    STATUS_PENDING,
    PriceLimitArrays,
    build_price_limit_arrays,
)
from pfb.engine.price_limits import DCB_READING_DEFAULT, TOPSTEP_BUFFER
from pfb.engine.strategy import (
    ParamValue,
    Strategy,
    StrategySpec,
    load_strategy,
    make_bars,
    params_json,
    resolve_params,
)
from pfb.engine.synthetic import DEFAULT_SYNTHETIC_FIRST, DEFAULT_SYNTHETIC_LAST, synthetic_frame
from pfb.instruments import sessions
from pfb.instruments.table import InstrumentTable, load_instruments
from pfb.metrics.core import TRADE_COLUMNS, summarize

NS_PER_MINUTE = 60_000_000_000
MINUTES_PER_DAY = 1440
REQUIRED_COLUMNS = ("ts_event", "open", "high", "low", "close", "volume", "trade_date", "ct_minute",
                    "in_scheduled_closure", "instrument_id", "era")  # fmt: skip
EQUITY_COLUMNS = ("ts_event", "trade_date", "position", "eq_close", "eq_low", "eq_high")
DAILY_COLUMNS = ("trade_date", "eq_open", "eq_low", "eq_high", "eq_close", "net_cents")
IGNORED_SIGNAL_COUNTERS = ("entry_signals_in_position", "entry_signals_at_session_end",
                           "exit_signals_at_session_end")  # fmt: skip


class EngineInputError(ValueError):
    """The bars or the market inputs handed to the engine are malformed."""


class LockedExitThroughSessionEnd(RuntimeError):
    """R-08: an exit was still waiting on a locked market when the session ended (upstream refuses the case by
    name, locked_exit_through_session_end: a question for the user, not a fill the engine invents)."""


# internal switch: True runs the price-limit rule. Only the parity script and engine tests may pass
# price_limit_rule=False to simulate; run_strategy (the trial path) never passes it.
PRICE_LIMIT_RULE_DEFAULT = True


@dataclass(frozen=True)
class MarketInputs:
    """Everything about the root the engine needs, as plain integer arrays (no float)."""

    must_be_flat_starts: np.ndarray  # int64 ns, half-open, sorted, merged (A3)
    must_be_flat_ends: np.ndarray
    session_days: np.ndarray  # int32 trade days with a session window, ascending
    session_starts: np.ndarray  # int64 ns: the SPEC session's window on each of those days (A5 g)
    session_ends: np.ndarray
    unsourced_days: np.ndarray  # int32 trade days whose bars are dropped and counted (A3)
    slippage_cents: np.ndarray  # int64[1440, 2]: one market fill of the whole quantity, by CT minute and side
    calibrated: np.ndarray  # bool[1440]
    commission_cents: int  # per fill, whole quantity
    tick_value_cc: int  # 1e-4 USD per tick per contract
    # int64[1440, 2]: what a PRICE_LIMIT exit fill pays, by CT minute and side: the cost model's event-window
    # table (upstream charges it on every price-limit exit); None: the model has no
    # event table (fixed:<k>, stress over fixed) and the forced exit pays slippage_cents (a run warning)
    event_slippage_cents: np.ndarray | None = None


@dataclass(frozen=True)
class PreparedBars:
    frame: pl.DataFrame  # the usable bars
    session_end: np.ndarray
    counters: dict[str, int]


@dataclass(frozen=True)
class KernelReplay:
    """What a kernel-only rerun needs (pfb.analysis.stress): the usable bars, the kernel inputs (signals
    included) and the counters that do not come from the kernel. Never enters a summary or an artifact."""

    prepared: PreparedBars
    kernel_in: K.KernelInputs
    minute: np.ndarray  # int64 CT minute of each usable bar (indexes the cost tables)
    flat_gap_count: int
    event_slippage_cents: np.ndarray | None = None  # MarketInputs' forced-exit table (None: none)
    # event_costs "on": the model that priced each bar from the release calendar, the root, and
    # each usable bar's open (UTC ns) and trade date (int days), so a rerun can price the bars again
    event_model: EventCostModel | None = None
    root: str | None = None
    ts: np.ndarray | None = None
    days: np.ndarray | None = None


@dataclass(frozen=True)
class SimOutput:
    trades: pl.DataFrame  # exactly pfb.metrics.core.TRADE_COLUMNS
    equity_minute: pl.DataFrame
    daily: pl.DataFrame
    counters: dict[str, int]
    warnings: tuple[str, ...] = ()  # the in-run look-ahead check's
    replay: KernelReplay | None = field(default=None, repr=False, compare=False)
    price_limits: PriceLimitArrays | None = field(default=None, repr=False, compare=False)
    # bool per trade row: the exit waited on a market locked against it and filled at a later bar's open
    # (kernel T_AFTER_LOCK); not a trade column, so artifacts and hashes do not move
    exit_after_lock: np.ndarray | None = field(default=None, repr=False, compare=False)


# ------------------------------------------------------------------ bars ----
def in_intervals(ts: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """ts[i] lies in some half-open [starts[j], ends[j]) (sorted, non-overlapping intervals)."""
    idx = np.searchsorted(starts, ts, side="right") - 1
    inside = idx >= 0
    inside[inside] = ts[inside] < ends[idx[inside]]
    return inside


def flat_gap_ends(ts: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Bar i lies before a must-be-flat interval's start and bar i + 1 at or after its end, so the interval
    holds no bar: bar i ends the session (ADR A16; sorted; non-overlapping intervals)."""
    gap = np.zeros(len(ts), dtype=bool)
    if len(ts) < 2 or len(starts) == 0:
        return gap
    nxt = np.searchsorted(starts, ts[:-1], side="right")  # the first interval starting after ts[i]
    exists = nxt < len(starts)
    gap[:-1] = exists & (ends[np.minimum(nxt, len(starts) - 1)] <= ts[1:])
    return gap


def _check_frame(frame: pl.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise EngineInputError(f"bars lack columns {missing}")
    ts = frame["ts_event"].to_numpy()
    if len(ts) > 1 and not bool(np.all(np.diff(ts) > 0)):
        raise EngineInputError("ts_event is not strictly increasing (nothing is de-duplicated silently)")


def _session_mask(ts: np.ndarray, days: np.ndarray, inputs: MarketInputs) -> np.ndarray:
    known_days = inputs.session_days
    pos = np.minimum(np.searchsorted(known_days, days), max(len(known_days) - 1, 0))
    known = known_days[pos] == days if len(known_days) else np.zeros(len(days), dtype=bool)
    if not bool(np.all(known)):
        raise EngineInputError(f"{int((~known).sum())} bar(s) on a trade day without a session window")
    return (ts >= inputs.session_starts[pos]) & (ts < inputs.session_ends[pos])


def _session_end(
    days: np.ndarray, instrument: np.ndarray, era: np.ndarray, next_closure: np.ndarray
) -> np.ndarray:
    """Last usable bar before a trade-day, instrument_id or era change, a removed closure bar, or the end."""
    end = np.ones(len(days), dtype=bool)
    if len(days) > 1:
        end[:-1] = (days[1:] != days[:-1]) | (instrument[1:] != instrument[:-1]) | (era[1:] != era[:-1])
    return end | next_closure


def prepare_bars(frame: pl.DataFrame, inputs: MarketInputs) -> PreparedBars:
    """Drop unsourced days, scheduled-closure bars and bars outside the SPEC session; derive session_end."""
    _check_frame(frame)
    days = frame["trade_date"].cast(pl.Int32).to_numpy()
    closure = frame["in_scheduled_closure"].to_numpy().astype(bool)
    next_closure = np.zeros(len(closure), dtype=bool)
    next_closure[:-1] = closure[1:]
    unsourced = np.isin(days, inputs.unsourced_days)
    usable = ~unsourced & ~closure
    in_session = np.zeros(len(days), dtype=bool)
    in_session[usable] = _session_mask(frame["ts_event"].to_numpy()[usable], days[usable], inputs)
    keep = usable & in_session
    kept = frame.filter(pl.Series(keep))
    counters = {
        "bars_dropped_unsourced": int(unsourced.sum()),
        "days_dropped_unsourced": len(np.unique(days[unsourced])),
        "bars_dropped_closure": int((closure & ~unsourced).sum()),
        "bars_dropped_outside_session": int((usable & ~in_session).sum()),
    }
    era = kept["era"].to_numpy().astype(str)
    session_end = _session_end(days[keep], kept["instrument_id"].to_numpy(), era, next_closure[keep])
    return PreparedBars(frame=kept, session_end=session_end, counters=counters)


# ---------------------------------------------------------------- frames ----
def _trade_frame(frame: pl.DataFrame, trades: np.ndarray, contracts: int) -> pl.DataFrame:
    """The ledger with exactly pfb.metrics.core.TRADE_COLUMNS, one row per round trip."""
    ts = frame["ts_event"].to_numpy()
    entry_idx, exit_idx, reason = trades[:, K.T_ENTRY_IDX], trades[:, K.T_EXIT_IDX], trades[:, K.T_REASON]
    close_fill = np.where(reason == K.SESSION_END, NS_PER_MINUTE, 0)  # a close fill is stamped at the close
    ints = {
        "side": trades[:, K.T_SIDE], "contracts": np.full(len(trades), contracts, dtype=np.int64),
        "entry_ts": ts[entry_idx], "exit_ts": ts[exit_idx] + close_fill,
        "entry_price": trades[:, K.T_ENTRY_PX], "exit_price": trades[:, K.T_EXIT_PX],
        "gross_cents": trades[:, K.T_GROSS], "commission_cents": trades[:, K.T_COMM],
        "slippage_cents": trades[:, K.T_SLIP], "net_cents": trades[:, K.T_NET],
        "mae_ticks": trades[:, K.T_MAE], "mfe_ticks": trades[:, K.T_MFE],
        "entry_bar": entry_idx, "exit_bar": exit_idx,
    }  # fmt: skip
    available = {c: pl.Series(c, v, dtype=pl.Int64) for c, v in ints.items()}
    labels = [K.EXIT_REASONS[r] for r in reason.tolist()]
    available["exit_reason"] = pl.Series("exit_reason", labels, dtype=pl.String)
    available["trade_date"] = frame["trade_date"].gather(pl.Series(exit_idx, dtype=pl.UInt32))
    missing = [c for c in TRADE_COLUMNS if c not in available]
    if missing:
        raise EngineInputError(f"pfb.metrics.core.TRADE_COLUMNS names column(s) the engine lacks: {missing}")
    return pl.DataFrame([available[c].alias(c) for c in TRADE_COLUMNS])


def _equity_frames(frame: pl.DataFrame, equity: np.ndarray) -> tuple[pl.DataFrame, pl.DataFrame]:
    columns = {"eq_open": K.E_OPEN, "eq_low": K.E_LOW, "eq_high": K.E_HIGH, "eq_close": K.E_CLOSE,
               "position": K.E_POS}  # fmt: skip
    full = frame.select("ts_event", "trade_date").with_columns(
        pl.Series(name, equity[:, col], dtype=pl.Int64) for name, col in columns.items()
    )
    daily = (
        full.group_by("trade_date", maintain_order=True)
        .agg(
            pl.col("eq_open").first(),
            pl.col("eq_low").min(),
            pl.col("eq_high").max(),
            pl.col("eq_close").last(),
        )
        .with_columns((pl.col("eq_close") - pl.col("eq_open")).alias("net_cents"))
    )
    return full.select(EQUITY_COLUMNS), daily.select(DAILY_COLUMNS)


# -------------------------------------------------------------- simulate ----
def _check_cost_tables(slippage_cents: np.ndarray, calibrated: np.ndarray) -> None:
    if slippage_cents.shape != (MINUTES_PER_DAY, 2) or slippage_cents.dtype.kind != "i":
        raise EngineInputError("slippage_cents must be int64[1440, 2]")
    if calibrated.shape != (MINUTES_PER_DAY,) or calibrated.dtype != np.bool_:
        raise EngineInputError("calibrated must be bool[1440]")
    if bool(np.any(slippage_cents < 0)):
        raise EngineInputError("a negative slippage")


def _check_event_table(table: np.ndarray | None) -> None:
    if table is None:
        return
    if not isinstance(table, np.ndarray) or table.shape != (MINUTES_PER_DAY, 2) or table.dtype.kind != "i":
        raise EngineInputError("event_slippage_cents must be None or int64[1440, 2]")
    if bool(np.any(table < 0)):
        raise EngineInputError("a negative event slippage")


def _check_inputs(inputs: MarketInputs) -> None:
    _check_cost_tables(inputs.slippage_cents, inputs.calibrated)
    _check_event_table(inputs.event_slippage_cents)


def _forced_exit_slip(
    slippage_cents: np.ndarray, event_slippage_cents: np.ndarray | None, minute: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(buy, sell) per bar: what a PRICE_LIMIT exit fill pays (the event table, else the ordinary one)."""
    table = slippage_cents if event_slippage_cents is None else event_slippage_cents
    return table[minute, 0], table[minute, 1]


def _check_event_model(
    model: EventCostModel,
    root: str,
    contracts: int,
    slippage_cents: np.ndarray,
    event_table_: np.ndarray | None,
) -> None:
    """The event model must price the very tables the run carries: its ordinary table is slippage_cents and
    its event table the forced-exit table (a mismatch is a wiring error, never a silent mix)."""
    if not isinstance(model, EventCostModel):
        raise EngineInputError("event_costs must be a pfb.costs.events.EventCostModel or None")
    if not np.array_equal(np.asarray(model.slippage_cents(root, contracts)), slippage_cents):
        raise EngineInputError(f"the event model {model.model_id!r} prices {root} unlike slippage_cents")
    if event_table_ is None or not np.array_equal(
        np.asarray(model.event_slippage_cents(root, contracts)), event_table_
    ):
        raise EngineInputError(
            f"the event model {model.model_id!r}: the forced-exit table must be its event table for {root}"
        )


def _bar_slip(
    replay_ts: np.ndarray, minute: np.ndarray, days: np.ndarray, slippage_cents: np.ndarray,
    model: EventCostModel | None, root: str, contracts: int,
) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """(buy, sell) per usable bar: the minute table at the bar's CT minute ("off"), or under "on" the event
    model's per-bar price (event table inside a release window of the root, at the bar open; else the
    minute table)."""
    if model is None:
        return slippage_cents[minute, 0], slippage_cents[minute, 1]
    table = np.asarray(model.bar_slippage_cents(root, contracts, replay_ts, minute, days), dtype=np.int64)
    return np.ascontiguousarray(table[:, 0]), np.ascontiguousarray(table[:, 1])


def _entry_allowed(mask: np.ndarray | None, n: int) -> np.ndarray:
    if mask is None:
        return np.ones(n, dtype=bool)
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != (n,):
        raise EngineInputError(f"entry_allowed must be a bool array over the {n} usable bars")
    return mask


def simulate(
    strategy: Strategy,
    frame: pl.DataFrame,
    params: Mapping[str, ParamValue],
    inputs: MarketInputs,
    *,
    seed: int,
    entry_allowed: np.ndarray | None = None,
    root: str | None = None,
    price_limit_buffer: Decimal = TOPSTEP_BUFFER,
    price_limit_rule: bool = PRICE_LIMIT_RULE_DEFAULT,
    event_costs: EventCostModel | None = None,
) -> SimOutput:
    """Usable bars -> look-ahead check (whose last call is the signals) -> kernel -> frames and counters.

    entry_allowed (the firm rules) is a bool mask over the USABLE bars, read at the entry decision bar.
    root defaults to the strategy's. The price-limit rule uses upstream's buffer by default; trials always
    get the default; price_limit_rule False is the internal off switch (parity script and engine tests only).
    event_costs: None is event costs "off" (each fill pays inputs.slippage_cents at its CT
    minute, exactly as before); an EventCostModel is "on": each bar's fill pays the model's per-bar price
    (EventCostModel.bar_slippage_cents at the bar open), and the model must price inputs' two tables. The
    PRICE_LIMIT exit pair stays the forced-exit table under both (upstream force_event).
    """
    _check_inputs(inputs)
    rule_root = root or strategy.spec.root
    if event_costs is not None:
        _check_event_model(event_costs, rule_root, strategy.spec.contracts, inputs.slippage_cents,
                           inputs.event_slippage_cents)  # fmt: skip
    prepared = prepare_bars(frame, inputs)
    bars = make_bars(prepared.frame)
    n = len(bars["ts_event"])
    # E1, E4: random, entry and exit cuts on the run's own bars, the full call last; its Signals is the one
    # the kernel gets (no further generate call), each call on its own copies and params
    check = lookahead_check(strategy.generate, bars, params, seed)
    signals = check.signals
    ts, minute = bars["ts_event"], bars["ct_minute"].astype(np.int64)
    starts, ends = inputs.must_be_flat_starts, inputs.must_be_flat_ends
    flat_now = in_intervals(ts, starts, ends)
    gap = flat_gap_ends(ts, starts, ends)
    limits = build_price_limit_arrays(
        rule_root, frame, prepared.frame, buffer=price_limit_buffer, enabled=price_limit_rule
    )
    pl_buy, pl_sell = _forced_exit_slip(inputs.slippage_cents, inputs.event_slippage_cents, minute)
    days = prepared.frame["trade_date"].cast(pl.Int32).to_numpy().astype(np.int64)
    slip_buy, slip_sell = _bar_slip(ts, minute, days, inputs.slippage_cents, event_costs, rule_root,
                                    strategy.spec.contracts)  # fmt: skip
    kernel_in = K.KernelInputs(
        open=bars["open"], high=bars["high"], low=bars["low"], close=bars["close"],
        slip_buy=slip_buy, slip_sell=slip_sell,
        session_end=prepared.session_end | gap, flat_by=flat_now,
        entry_fill_ok=~flat_now & ~in_intervals(ts + NS_PER_MINUTE, starts, ends),
        entry_allowed=_entry_allowed(entry_allowed, n), calibrated=inputs.calibrated[minute],
        entry_long=signals.entry_long, entry_short=signals.entry_short, exit=signals.exit,
        stop_ticks=signals.stop_ticks, target_ticks=signals.target_ticks,
        commission_cents=inputs.commission_cents, tick_value_cc=inputs.tick_value_cc,
        contracts=strategy.spec.contracts,
        pl_upper=limits.upper, pl_lower=limits.lower, pl_lock_up=limits.lock_up,
        pl_lock_down=limits.lock_down, pl_no_reference=limits.no_reference,
        slip_pl_buy=pl_buy, slip_pl_sell=pl_sell,
    )  # fmt: skip
    replay = KernelReplay(prepared, kernel_in, minute, int((gap & ~prepared.session_end).sum()),
                          inputs.event_slippage_cents, event_costs, rule_root, ts, days)  # fmt: skip
    return _kernel_output(replay, check.warnings, limits)


def _kernel_output(
    replay: KernelReplay, warnings: tuple[str, ...], limits: PriceLimitArrays | None = None
) -> SimOutput:
    out = K.run_kernel(replay.kernel_in)
    if out.counters["locked_exits_at_session_end"]:
        raise LockedExitThroughSessionEnd(
            f"locked_exit_through_session_end: {out.counters['locked_exits_at_session_end']} exit(s) still "
            "locked when the session ended"
        )
    frame = replay.prepared.frame
    trades = _trade_frame(frame, out.trades, replay.kernel_in.contracts)
    equity_minute, daily = _equity_frames(frame, out.equity)
    counters = replay.prepared.counters | out.counters
    counters["session_ends_at_flat_gap"] = replay.flat_gap_count
    counters["signals_ignored"] = sum(counters[k] for k in IGNORED_SIGNAL_COUNTERS)
    after_lock = out.trades[:, K.T_AFTER_LOCK] == 1
    return SimOutput(trades, equity_minute, daily, counters, warnings, replay, limits, after_lock)


def with_event_table(sim: SimOutput, event_slippage_cents: np.ndarray | None) -> SimOutput:
    """The same SimOutput whose replay carries another forced-exit (event-window) table, int64[1440, 2] or
    None (that cost model has none): a caller of rerun_kernel with another cost model sets it first."""
    if sim.replay is None:
        raise EngineInputError("this SimOutput carries no kernel replay")
    _check_event_table(event_slippage_cents)
    return replace(sim, replay=replace(sim.replay, event_slippage_cents=event_slippage_cents))


def _check_rerun_events(replay: KernelReplay, event_costs: EventCostModel | None) -> None:
    """A run simulated "on" is re-run "on" with the same calendar, an "off" run "off" (the
    per-fill event charge must survive the cost-stress re-run, never be dropped by a per-minute table)."""
    before = replay.event_model
    if before is None and event_costs is None:
        return
    if before is None:
        raise EngineInputError("this run was simulated with event costs off; a rerun cannot turn them on")
    if event_costs is None:
        raise EngineInputError(
            f"this run was simulated with event costs on ({before.model_id}); its rerun needs the event "
            "model of the new costs (EventCostModel over the stressed base), or the per-fill event charge "
            "is lost"
        )
    if event_costs.calendar.sha256 != before.calendar.sha256:
        raise EngineInputError("the rerun's event model reads another release calendar than the run's")


def rerun_kernel(
    sim: SimOutput,
    slippage_cents: np.ndarray,
    calibrated: np.ndarray,
    *,
    event_costs: EventCostModel | None = None,
) -> SimOutput:
    """The kernel alone again on the same bars and signals with other cost tables (int64[1440, 2] and
    bool[1440]): no generate call and no look-ahead check. Exactly what simulate gives with those tables and
    the forced-exit table the replay carries (with_event_table; None: a PRICE_LIMIT exit pays
    slippage_cents, as simulate does without an event table). A run simulated with event costs "on" must be
    re-run with ``event_costs``, the EventCostModel of the new costs (whose ordinary table is slippage_cents):
    each bar is priced again from the calendar at its open, as simulate does."""
    if sim.replay is None:
        raise EngineInputError("this SimOutput carries no kernel replay")
    _check_cost_tables(slippage_cents, calibrated)
    replay = sim.replay
    _check_rerun_events(replay, event_costs)
    minute = replay.minute
    contracts = replay.kernel_in.contracts
    if event_costs is not None:
        if replay.root is None or replay.ts is None or replay.days is None:
            raise EngineInputError("this replay lacks the root, bar opens or trade dates of an event run")
        _check_event_model(event_costs, replay.root, contracts, slippage_cents, replay.event_slippage_cents)
        slip_buy, slip_sell = _bar_slip(replay.ts, minute, replay.days, slippage_cents, event_costs,
                                        replay.root, contracts)  # fmt: skip
    else:
        slip_buy, slip_sell = slippage_cents[minute, 0], slippage_cents[minute, 1]
    pl_buy, pl_sell = _forced_exit_slip(slippage_cents, replay.event_slippage_cents, minute)
    kernel_in = replace(replay.kernel_in, slip_buy=slip_buy, slip_sell=slip_sell,
                        calibrated=calibrated[minute], slip_pl_buy=pl_buy, slip_pl_sell=pl_sell)  # fmt: skip
    rerun = replace(replay, kernel_in=kernel_in, event_model=event_costs)
    return _kernel_output(rerun, sim.warnings, sim.price_limits)


# ---------------------------------------------------------------- wiring ----
SOURCES = ("real", "synthetic")
EVENT_WINDOWS = "off"
EVENT_WARNING = "event windows off: release-minute slippage is not charged, so costs are understated (A7)"
# under "on" the warning names what is still not modelled of the event rules
EVENT_ON_WARNING = (
    "event windows on: release-window slippage is charged, but a 2-minute fill guard after a release "
    "and a CPI opening rule are not modelled"
)
BOOK_TABLE_FIRST, BOOK_TABLE_LAST = date(2025, 4, 1), date(2026, 6, 30)
BOOK_WARNING = (
    "book cost table has no date axis (sampled 2025-2026); this run lies outside 2025-04..2026-06 (A7)"
)
EPOCH = date(1970, 1, 1)
# only where the rule did not run on every trade date
STANDING_WARNING = "price-limit and locked-market rules are not modelled"
DCB_WARNING = (
    "dynamic-circuit-breaker halts are not modelled (no price lock limit on this product)"
)
NO_SLIPPAGE_WARNING = "cost model without slippage"
# a model without an event table prices a PRICE_LIMIT exit with its own
# slippage; said whenever at least one such exit occurred
FORCED_EXIT_FIXED_SLIPPAGE_WARNING = (
    "price-limit exit charged the fixed slippage: this cost model has no event table"
)


class RunRequestError(ValueError):
    """run_strategy was called with a missing or contradictory argument."""


@dataclass(frozen=True)
class RunResult:
    trades: pl.DataFrame
    equity_minute: pl.DataFrame
    daily: pl.DataFrame
    summary: dict[str, object]
    counters: dict[str, int]
    provenance: dict[str, object]
    spec: StrategySpec
    params: dict[str, ParamValue]
    extras: dict[str, pl.DataFrame] = field(default_factory=dict)  # suite frames (runs/<id>/<name>.parquet)


@dataclass(frozen=True)
class _Request:
    strategy: Strategy
    root: str
    first: date
    last: date
    params: dict[str, ParamValue]


def _resolve(
    strategy_path: Path | str, source: str, root: str | None, first: date | None, last: date | None,
    params: Mapping[str, ParamValue] | None,
) -> _Request:  # fmt: skip
    if source not in SOURCES:
        raise RunRequestError(f"source must be one of {SOURCES}, got {source!r}")
    strategy = load_strategy(strategy_path)
    if source == "real" and (first is None or last is None):
        raise RunRequestError("a real-data run needs first and last")
    run_first = first or DEFAULT_SYNTHETIC_FIRST
    run_last = last or DEFAULT_SYNTHETIC_LAST
    if run_first > run_last:
        raise RunRequestError(f"first {run_first} is after last {run_last}")
    resolved = resolve_params(strategy.spec, params)
    return _Request(strategy, root or strategy.spec.root, run_first, run_last, resolved)


def _check_permit(
    req: _Request, source: str, permit: RunPermit | None, fees: str, cost_model: str,
    event_costs: str = EVENT_COSTS_OFF,
) -> None:  # fmt: skip
    if source == "synthetic" and permit is None:
        return
    wanted = PermitRequest(req.strategy.sha256, req.root, req.first, req.last, params_json(req.params), fees,
                           cost_model, event_costs)  # fmt: skip
    checked = check_permit(permit, wanted)
    if source == "real" and checked.kind == "synthetic":
        raise PermitRequired("a synthetic permit cannot run on real data")


def _load_bars(
    req: _Request, source: str, seed: int, instruments: InstrumentTable
) -> tuple[pl.DataFrame, list[dict[str, object]], list[str]]:
    if source == "synthetic":
        return synthetic_frame(req.root, req.first, req.last, seed=seed), [], []
    gateway = DataGateway(
        load_paths(), load_windows(), load_catalog(), instruments.tick_fixed_of, instruments.price_factor_of
    )
    frame = gateway.bars(req.root, req.first, req.last)
    files = [
        {"path": r.path, "era": r.era, "sha256": r.sha256, "rows": r.rows, "pe_pin_status": r.pe_pin_status,
         "tick_metadata": r.tick_metadata}
        for r in gateway.provenance
    ]  # fmt: skip
    return frame, files, list(provenance_warnings(gateway.provenance))


def _session_windows(root: str, name: str, days: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique = np.unique(days).astype(np.int32)
    windows = [sessions.session_window_ns(root, EPOCH + timedelta(days=int(d)), name) for d in unique]
    starts = np.array([w[0] for w in windows], dtype=np.int64)
    ends = np.array([w[1] for w in windows], dtype=np.int64)
    return unique, starts, ends


@dataclass(frozen=True)
class _Costs:
    slippage_cents: np.ndarray
    calibrated: np.ndarray
    commission_cents: int
    tick_value_cc: int
    event_slippage_cents: np.ndarray | None = None  # MarketInputs.event_slippage_cents


def event_table(cost: CostModel, root: str, contracts: int) -> np.ndarray | None:
    """The cost model's event-window table (int64[1440, 2]), what a PRICE_LIMIT exit pays under event costs
    "off" and "on"; None when the model has none (fixed:<k>, stress over fixed: EventCostsOff)."""
    try:
        return np.asarray(cost.event_slippage_cents(root, contracts), dtype=np.int64)
    except EventCostsOff:
        return None


def _costs(req: _Request, instruments: InstrumentTable, cost: CostModel, fees: FeeSchedule) -> _Costs:
    """The root's cost rows, looked up BEFORE any bar is read: a root without a row is refused (A7)."""
    contracts = req.strategy.spec.contracts
    return _Costs(
        slippage_cents=np.asarray(cost.slippage_cents(req.root, contracts), dtype=np.int64),
        calibrated=np.asarray(cost.calibrated(req.root), dtype=np.bool_),
        commission_cents=int(fees.commission_side_cents(req.root)) * contracts,
        tick_value_cc=int(instruments.get(req.root).tick_value_cc),
        event_slippage_cents=event_table(cost, req.root, contracts),
    )


def _market_inputs(req: _Request, frame: pl.DataFrame, costs: _Costs) -> MarketInputs:
    flat_starts, flat_ends = sessions.must_be_flat_intervals_ns(req.root, req.first, req.last)
    days, starts, ends = _session_windows(
        req.root, req.strategy.spec.session, frame["trade_date"].cast(pl.Int32).to_numpy()
    )
    return MarketInputs(
        must_be_flat_starts=np.asarray(flat_starts, dtype=np.int64),
        must_be_flat_ends=np.asarray(flat_ends, dtype=np.int64),
        session_days=days, session_starts=starts, session_ends=ends,
        unsourced_days=np.asarray(sessions.unsourced_days(req.root), dtype=np.int32),
        slippage_cents=costs.slippage_cents, calibrated=costs.calibrated,
        commission_cents=costs.commission_cents, tick_value_cc=costs.tick_value_cc,
        event_slippage_cents=costs.event_slippage_cents,
    )  # fmt: skip


def _uses_book(cost_model: str) -> bool:
    return "book" in cost_model.split(":")


def held_dates(sim: SimOutput) -> tuple[tuple[str, str], ...]:
    """(entry, exit) ISO trade dates of each trade: a position was open on every trade date between them
    (from the trade frame's entry and exit bars)."""
    if sim.trades.is_empty():
        return ()
    days = sim.equity_minute["trade_date"]
    entry = days.gather(sim.trades["entry_bar"].cast(pl.UInt32)).to_list()
    exit_ = days.gather(sim.trades["exit_bar"].cast(pl.UInt32)).to_list()
    return tuple((a.isoformat(), b.isoformat()) for a, b in zip(entry, exit_, strict=True))


def price_limit_warnings(
    limits: PriceLimitArrays | None, held: tuple[tuple[str, str], ...] = ()
) -> list[str]:
    """A date without a prior settlement
    reference on which no position was open counts as a date the rule ran (entries were refused there,
    upstream's
    engine_price_limit_reference_unavailable): a run whose dates are all applied or of that kind carries only
    the narrower count. STANDING_WARNING stays when a date lies outside the sourced limit table or a position
    (``held``: (entry, exit) ISO trade dates per trade) was open on a no-reference date, where no forced exit
    could be evaluated. A DCB-only product gets the narrower DCB warning."""
    if limits is None or not limits.enabled:
        return [STANDING_WARNING]
    if limits.kind == "dcb_only":
        return [DCB_WARNING]
    n_dates = len(limits.date_status)
    pending = limits.dates_with(STATUS_PENDING)
    no_ref = sorted(d for d, s in limits.date_status.items() if s == STATUS_NO_REFERENCE)
    open_there = [d for d in no_ref if any(a <= d <= b for a, b in held)]
    if pending == 0 and not open_there:
        if not no_ref:
            return []
        return [
            f"price-limit rule: entries refused on {len(no_ref)} of {n_dates} trade date(s) without a prior "
            "settlement reference in the run's bars"
        ]
    held_note = (
        f"; a position was open on {len(open_there)} of them (no forced exit could be evaluated there)"
        if open_there
        else ""
    )
    return [STANDING_WARNING, (
        f"price-limit rule did not run on {pending + len(no_ref)} of {n_dates} trade date(s): "
        f"{pending} outside the sourced limit table, {len(no_ref)} without a prior settlement reference in "
        f"the run's bars (entries refused there){held_note}"
    )]  # fmt: skip


def price_limit_summary(limits: PriceLimitArrays | None, counters: Mapping[str, int]) -> dict[str, object]:
    """The result summary's "price_limits" block."""
    if limits is None:
        return {"enabled": False}
    return {
        "enabled": limits.enabled, "buffer": str(limits.buffer), "dcb_reading": DCB_READING_DEFAULT.value,
        "kind": {limits.root: limits.kind},
        "dates": {s: limits.dates_with(s) for s in sorted(set(limits.date_status.values()))},
        "entries_refused": counters.get("price_limit_entries_refused", 0),
        "entries_no_reference": counters.get("price_limit_entries_no_reference", 0),
        "forced_exits": counters.get("price_limit_exits", 0),
        "locked_fills": counters.get("locked_exit_fills", 0),
        "locked_deferrals": counters.get("locked_exit_deferrals", 0),
        "dates_without_reference": limits.dates_with(STATUS_NO_REFERENCE),
        "dates_dcb": limits.dates_with(STATUS_DCB),
    }  # fmt: skip


def forced_exit_cost_warnings(event_slippage_cents: np.ndarray | None, trades: pl.DataFrame) -> list[str]:
    """: the run used a cost model without an event table and holds a PRICE_LIMIT exit."""
    forced = K.EXIT_REASONS[K.PRICE_LIMIT]
    if event_slippage_cents is None and bool((trades["exit_reason"] == forced).any()):
        return [FORCED_EXIT_FIXED_SLIPPAGE_WARNING]
    return []


def event_warning(event_costs: str) -> str:
    """The run's event-window warning: costs understated under "off", what is not modelled under "on"."""
    return EVENT_ON_WARNING if event_costs == EVENT_COSTS_ON else EVENT_WARNING


def _warnings(
    req: _Request, cost_model: str, costs: _Costs, sim: SimOutput, data: list[str],
    event_costs: str = EVENT_COSTS_OFF,
) -> list[str]:  # fmt: skip
    out = [*price_limit_warnings(sim.price_limits, held_dates(sim)), event_warning(event_costs)]
    if _uses_book(cost_model) and (req.first < BOOK_TABLE_FIRST or req.last > BOOK_TABLE_LAST):
        out.append(BOOK_WARNING)
    if not bool(np.any(costs.slippage_cents)):
        out.append(NO_SLIPPAGE_WARNING)
    out += forced_exit_cost_warnings(costs.event_slippage_cents, sim.trades)
    return out + list(sim.warnings) + data


def run_strategy(
    strategy_path: Path | str,
    *,
    source: str,
    root: str | None = None,
    first: date | None = None,
    last: date | None = None,
    params: Mapping[str, ParamValue] | None = None,
    fees: str = "topstep",
    cost_model: str = "book",
    seed: int = 0,
    entry_allowed: np.ndarray | None = None,
    permit: RunPermit | None = None,
    n_trials: int | None = None,
    suite: SuiteSpec | None = None,
    event_costs: str = EVENT_COSTS_OFF,
) -> RunResult:
    """One run (ADR section 9, A1). source "real" reads through the gateway and needs a matching permit.
    With a suite, summary["suite"] holds the in-run analyses (pfb.analysis.inrun) on the same bars and
    RunResult.extras their frames; without one nothing changes. event_costs "off" (the default) is the run
    as before, byte for byte; "on" prices each fill with the release calendar (EventCostModel over
    cost_model, id <cost_model>+events), refuses a window the calendar does not cover before any bar is
    read, and records event_windows "on"."""
    req = _resolve(strategy_path, source, root, first, last, params)
    _check_permit(req, source, permit, fees, cost_model, event_costs)  # the permit names event_costs
    instruments = load_instruments()
    fee_schedule = load_fees(fees, instruments=instruments)
    needs_paths = _uses_book(cost_model) or event_costs != EVENT_COSTS_OFF  # "on" reads the upstream calendar
    cost = load_cost_model_for(cost_model, event_costs, instruments, load_paths() if needs_paths else None)
    event_model = cost if isinstance(cost, EventCostModel) else None
    if event_model is not None:
        event_model.calendar.require_covers(req.first, req.last)
    costs = _costs(req, instruments, cost, fee_schedule)
    frame, files, data_warnings = _load_bars(req, source, seed, instruments)
    inputs = _market_inputs(req, frame, costs)
    sim = simulate(
        req.strategy,
        frame,
        req.params,
        inputs,
        seed=seed,
        entry_allowed=entry_allowed,
        root=req.root,
        event_costs=event_model,
    )
    summary = summarize(sim.trades, sim.equity_minute, sim.daily, n_trials=n_trials)
    warnings = _warnings(req, cost_model, costs, sim, data_warnings, event_costs)
    summary |= {"event_windows": event_costs, "warnings": warnings,
                "price_limits": price_limit_summary(sim.price_limits, sim.counters)}  # fmt: skip
    extras: dict[str, pl.DataFrame] = {}
    if suite is not None:
        from pfb.analysis.inrun import RunContext, compute_suite  # imports this module: loaded on use

        ctx = RunContext(req.strategy, frame, req.params, inputs, seed, entry_allowed, cost_model, req.root,
                         instruments, sim, n_trials)  # fmt: skip
        summary["suite"], extras, suite_warnings = compute_suite(ctx, suite)
        if suite_warnings:  # a part a known suite error left None: the run stays ok
            summary["warnings"] = [*warnings, *suite_warnings]
    provenance: dict[str, object] = {
        "source": source, "root": req.root, "first": req.first.isoformat(), "last": req.last.isoformat(),
        "files": files, "strategy_path": str(req.strategy.path), "strategy_sha256": req.strategy.sha256,
        "params_json": params_json(req.params), "seed": seed, "instruments_sha256": instruments.sha256,
        "cost_model_id": cost.model_id, "cost_config_sha256": cost.config_sha256, "fees_firm": fees,
        "fees_sha256": fee_schedule.sha256, "event_windows": event_costs, "engine_version": ENGINE_VERSION,
        "permit_run_id": None if permit is None else permit.run_id,
    }  # fmt: skip
    return RunResult(sim.trades, sim.equity_minute, sim.daily, summary, sim.counters, provenance,
                     req.strategy.spec, req.params, extras)  # fmt: skip


__all__ = [
    "DAILY_COLUMNS", "EQUITY_COLUMNS", "EngineInputError", "MarketInputs",
    "KernelReplay", "NO_SLIPPAGE_WARNING", "PreparedBars", "RunRequestError", "RunResult", "STANDING_WARNING",
    "DCB_WARNING", "LockedExitThroughSessionEnd", "price_limit_summary", "price_limit_warnings",
    "FORCED_EXIT_FIXED_SLIPPAGE_WARNING", "SimOutput", "event_table", "forced_exit_cost_warnings",
    "EVENT_ON_WARNING", "EVENT_WARNING", "event_warning",
    "held_dates", "rerun_kernel", "with_event_table",
    "flat_gap_ends", "in_intervals", "prepare_bars", "run_strategy", "simulate",
]  # fmt: skip
