"""Deliberately leaky mutants (positive controls, test suite only; the kernel has no switch for them).

Engine mutants (the strategy is honest, the engine leaks), run through the same kernel as the honest engine:
- same_bar_fill: the decision taken at bar i's close fills at bar i's OWN open (upstream _SameBarFillMutant);
- peek_next_bar: while the engine is on bar i the strategy is shown bar i + 1 (upstream _PeekNextBarMutant).
Strategy mutants (the engine is honest, generate reads the future):
- peek_next_close: compares the next bar's close with this one;
- next_session_open: compares the NEXT session's first open with this bar's close (a later session's value);
- session_last_close: compares this session's final close with this bar's close (a later value, same session).
"""

from __future__ import annotations

import types
from collections.abc import Callable, Mapping

import numpy as np
import polars as pl

from pfb.engine import kernel as K
from pfb.engine.price_limit_inputs import build_price_limit_arrays
from pfb.engine.run import NS_PER_MINUTE, MarketInputs, in_intervals, prepare_bars
from pfb.engine.strategy import (
    BAR_DTYPES,
    SIGNAL_KEYS,
    Bars,
    ParamValue,
    Strategy,
    bar_count,
    bars_from_arrays,
    make_bars,
    normalize_signals,
)

GenerateFn = Callable[[Bars, Mapping[str, ParamValue]], object]
CanaryEngine = Callable[[Strategy, pl.DataFrame, MarketInputs], pl.DataFrame]


# ------------------------------------------------------------ engine mutants ----
def next_bar_view(bars: Bars) -> Bars:
    """Bar i of the view is bar i + 1 (the last bar repeats), as upstream zip(bars, bars[1:] + [bars[-1]])."""
    return bars_from_arrays({k: np.concatenate((bars[k][1:], bars[k][-1:])) for k in BAR_DTYPES})


def one_bar_back(values: np.ndarray) -> np.ndarray:
    """out[i] = values[i + 1]: the decision of bar i + 1 is handed to the kernel at bar i, so it fills at
    open[i + 1], the decision bar's own open."""
    out = np.zeros_like(values)
    out[:-1] = values[1:]
    return out


def kernel_trades(
    strategy: Strategy,
    frame: pl.DataFrame,
    inputs: MarketInputs,
    params: Mapping[str, ParamValue] | None = None,
    *,
    peek: bool = False,
    same_bar_fill: bool = False,
) -> pl.DataFrame:
    """simulate's kernel path (no truncation check) with at most the named bug; four trade columns."""
    prepared = prepare_bars(frame, inputs)
    bars = make_bars(prepared.frame)
    n = bar_count(bars)
    frozen = types.MappingProxyType(dict(params or {}))
    signals = normalize_signals(strategy.generate(next_bar_view(bars) if peek else bars, frozen), n)
    sig = {k: signals.get(k) for k in SIGNAL_KEYS}
    if same_bar_fill:
        sig = {k: one_bar_back(v) for k, v in sig.items()}
    ts, minute = bars["ts_event"], bars["ct_minute"].astype(np.int64)
    starts, ends = inputs.must_be_flat_starts, inputs.must_be_flat_ends
    flat_now = in_intervals(ts, starts, ends)
    limits = build_price_limit_arrays(strategy.spec.root, frame, prepared.frame)  # simulate's
    out = K.run_kernel(
        K.KernelInputs(
            open=bars["open"], high=bars["high"], low=bars["low"], close=bars["close"],
            slip_buy=inputs.slippage_cents[minute, 0], slip_sell=inputs.slippage_cents[minute, 1],
            session_end=prepared.session_end, flat_by=flat_now,
            entry_fill_ok=~flat_now & ~in_intervals(ts + NS_PER_MINUTE, starts, ends),
            entry_allowed=np.ones(n, dtype=np.bool_), calibrated=inputs.calibrated[minute],
            entry_long=sig["entry_long"], entry_short=sig["entry_short"], exit=sig["exit"],
            stop_ticks=sig["stop_ticks"], target_ticks=sig["target_ticks"],
            commission_cents=inputs.commission_cents, tick_value_cc=inputs.tick_value_cc,
            contracts=strategy.spec.contracts,
            pl_upper=limits.upper, pl_lower=limits.lower, pl_lock_up=limits.lock_up,
            pl_lock_down=limits.lock_down, pl_no_reference=limits.no_reference,
        )
    )  # fmt: skip
    t = out.trades
    return pl.DataFrame(
        {
            "side": t[:, K.T_SIDE],
            "entry_ts": ts[t[:, K.T_ENTRY_IDX]],
            "entry_price": t[:, K.T_ENTRY_PX],
            "exit_price": t[:, K.T_EXIT_PX],
        }
    )


def honest_kernel_engine(strategy: Strategy, frame: pl.DataFrame, inputs: MarketInputs) -> pl.DataFrame:
    return kernel_trades(strategy, frame, inputs)


def same_bar_fill_engine(strategy: Strategy, frame: pl.DataFrame, inputs: MarketInputs) -> pl.DataFrame:
    return kernel_trades(strategy, frame, inputs, same_bar_fill=True)


def peek_next_bar_engine(strategy: Strategy, frame: pl.DataFrame, inputs: MarketInputs) -> pl.DataFrame:
    return kernel_trades(strategy, frame, inputs, peek=True)


ENGINE_MUTANTS: dict[str, CanaryEngine] = {
    "same_bar_fill": same_bar_fill_engine,
    "peek_next_bar": peek_next_bar_engine,
}


# ---------------------------------------------------------- strategy mutants ----
def signals_from(long: np.ndarray, short: np.ndarray) -> dict[str, np.ndarray]:
    n = len(long)
    zeros = np.zeros(n, dtype=np.int64)
    return {"entry_long": long, "entry_short": short & ~long, "exit": np.zeros(n, dtype=np.bool_),
            "stop_ticks": zeros, "target_ticks": zeros.copy()}  # fmt: skip


def _session_index(day: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(session number of each bar, first bar index of each session)."""
    change = np.zeros(len(day), dtype=np.int64)
    change[1:] = (np.diff(day) != 0).astype(np.int64)
    starts = np.concatenate((np.zeros(1, dtype=np.int64), np.flatnonzero(change).astype(np.int64)))
    return np.cumsum(change), starts


def peek_next_close(bars: Bars, params: Mapping[str, ParamValue]) -> dict[str, np.ndarray]:
    close = bars["close"]
    nxt = np.concatenate((close[1:], close[-1:]))
    return signals_from(nxt > close, nxt < close)


def next_session_open(bars: Bars, params: Mapping[str, ParamValue]) -> dict[str, np.ndarray]:
    close, n = bars["close"], bar_count(bars)
    if n == 0:
        return signals_from(np.zeros(0, dtype=np.bool_), np.zeros(0, dtype=np.bool_))
    session, starts = _session_index(bars["trade_day"])
    next_start = np.append(starts[1:], -1)[session]
    has_next = next_start >= 0
    nxt_open = bars["open"][np.maximum(next_start, 0)]
    return signals_from(has_next & (nxt_open > close), has_next & (nxt_open < close))


def session_last_close(bars: Bars, params: Mapping[str, ParamValue]) -> dict[str, np.ndarray]:
    close, n = bars["close"], bar_count(bars)
    if n == 0:
        return signals_from(np.zeros(0, dtype=np.bool_), np.zeros(0, dtype=np.bool_))
    session, starts = _session_index(bars["trade_day"])
    last_idx = np.append(starts[1:], n) - 1
    final = close[last_idx[session]]
    return signals_from(final > close, final < close)


def honest_momentum(bars: Bars, params: Mapping[str, ParamValue]) -> dict[str, np.ndarray]:
    close = bars["close"]
    prev = np.concatenate((close[:1], close[:-1]))
    return signals_from(close > prev, close < prev)


STRATEGY_MUTANTS: dict[str, GenerateFn] = {
    "peek_next_close": peek_next_close,
    "next_session_open": next_session_open,
    "session_last_close": session_last_close,
}

# The next-session mutant as a strategy FILE that passes the static lint (for the runner test).
LEAKY_STRATEGY_SOURCE = '''"""Leaky on purpose (test only): reads the NEXT session's first open."""

from __future__ import annotations

import numpy as np

SPEC = {
    "hypothesis_id": "test:leaky_next_session_open",
    "root": "MNQ",
    "bar_minutes": 1,
    "session": "globex",
    "contracts": 1,
    "params": {"edge_ticks": {"default": 0, "grid": [0]}},
}


def generate(bars, params):
    day = bars["trade_day"]
    close = bars["close"]
    n = len(day)
    zeros = np.zeros(n, dtype=np.int64)
    change = np.zeros(n, dtype=np.int64)
    change[1:] = (np.diff(day) != 0).astype(np.int64)
    starts = np.concatenate((np.zeros(1, dtype=np.int64), np.flatnonzero(change)))
    session = np.cumsum(change)
    next_start = np.append(starts[1:], -1)[session] if n else zeros
    has_next = next_start >= 0
    nxt = bars["open"][np.maximum(next_start, 0)] if n else zeros
    edge = params["edge_ticks"]
    long = has_next & (nxt > close + edge)
    short = has_next & (nxt < close - edge)
    return {
        "entry_long": long,
        "entry_short": short,
        "exit": np.zeros(n, dtype=np.bool_),
        "stop_ticks": zeros,
        "target_ticks": zeros.copy(),
    }
'''


def suppress_with_next_bar(base: GenerateFn) -> GenerateFn:
    """Mutant: `base`'s signals, minus every entry whose NEXT bar (same trade day) moves against it.

    The next bar only SUPPRESSES: the full-data run has no signal at bar i where a run cut at i + 1 has one,
    so a cut after a full-data signal bar never sees it."""

    def generate(bars: Bars, params: Mapping[str, ParamValue]) -> dict[str, np.ndarray]:
        raw = base(bars, params)
        assert isinstance(raw, Mapping)
        out = {k: np.array(v, copy=True) for k, v in raw.items()}
        close, day, n = bars["close"], bars["trade_day"], bar_count(bars)
        if n < 2:
            return out
        same_day = np.zeros(n, dtype=np.bool_)
        same_day[:-1] = day[1:] == day[:-1]
        step = np.zeros(n, dtype=np.int64)
        step[:-1] = close[1:] - close[:-1]
        out["entry_long"] &= ~(same_day & (step < 0))
        out["entry_short"] &= ~(same_day & (step > 0))
        return out

    return generate


def peeking_source(reference_source: str) -> str:
    """A lint-passing strategy FILE: the reference's own generate, fed the next bar (sparse one-bar peek)."""
    honest = reference_source.replace("def generate(bars, params):", "def _honest(bars, params):", 1)
    return honest + (
        "\n\ndef generate(bars, params):\n"
        "    view = {k: np.concatenate((bars[k][1:], bars[k][-1:])) for k in bars}\n"
        "    return _honest(view, params)\n"
    )
