"""Strategy template: copy it with `pfb new <name>`, then edit SPEC and generate.

One file, two names: SPEC (a dict literal the engine reads without running anything else) and
generate(bars, params) (called on bars; it returns five numpy arrays of the bars' length). The lint allows
numpy, polars and math only, no I/O, no clock, no randomness, no global state; `pfb check <file>` runs the
lint and the look-ahead check on synthetic bars before anything is logged.

The rule this example trades (an opening-range breakout with a stop and a target):
- range: the bars of the trade date whose CT minute lies in [day_open_minute, day_open_minute +
  range_minutes); its high and low are known only once each bar has closed (np.maximum.accumulate);
- entry decision: the first later bar, before last_entry_minute, whose close is at least buffer_ticks above
  the range high (long) or below the range low (short); one decision a day;
- exit: the stop (stop_ticks) or the target (target_ticks) from the fill, or an exit decision on the first
  bar at or after exit_minute; the engine also closes at the session end.

Look-ahead: a signal on bar i is a decision made at the close of bar i and filled by the engine at the open of
bar i + 1. generate may use bars 0..i to decide bar i and nothing later; every array below is built from
running (cumulative) values, so cutting the bars at any point leaves the earlier signals unchanged, which is
exactly what the look-ahead check tests. Prices are integer ticks; distances (stop, target) are ticks >= 0,
0 meaning none.
"""

from __future__ import annotations

import numpy as np

SPEC = {
    # The id of the hypothesis this strategy serves; `pfb new <name>` sets "h_<name>". A trial's hypothesis
    # file (hypotheses/<id>.toml, schema "pfb.hypothesis/2") must use the same id and pin this file's sha256.
    "hypothesis_id": "h_template",
    "root": "MNQ",  # a root of config/instruments.toml; a hypothesis may add roots
    "bar_minutes": 1,  # must equal the hypothesis's bar_minutes
    "session": "globex",  # must equal the hypothesis's session
    "contracts": 1,  # contracts per entry
    # Every parameter: a default and the grid of values the hypothesis may test (numeric grids strictly
    # increasing). A trial's [grid] must name exactly these parameters.
    "params": {
        "day_open_minute": {"default": 510, "grid": [510]},  # 08:30 CT
        "range_minutes": {"default": 15, "grid": [15, 30]},
        "buffer_ticks": {"default": 4, "grid": [2, 4, 6]},
        "last_entry_minute": {"default": 840, "grid": [840]},  # 14:00 CT
        "exit_minute": {"default": 895, "grid": [895]},  # 14:55 CT
        "stop_ticks": {"default": 40, "grid": [40]},
        "target_ticks": {"default": 80, "grid": [80]},
    },
}


def _day_bounds(trade_day):
    """[start, end) bar indices of each trade date (bars are in time order)."""
    n = len(trade_day)
    starts = np.flatnonzero(trade_day[1:] != trade_day[:-1]) + 1
    return np.concatenate((np.zeros(1, dtype=np.int64), starts)).tolist(), np.append(starts, n).tolist()


def _entry_of_day(high, low, close, minute, params):
    """(index of the day's entry decision or -1, True if long), from running values only."""
    start = params["day_open_minute"]
    range_end = start + params["range_minutes"]
    buffer = params["buffer_ticks"]
    in_range = (minute >= start) & (minute < range_end)
    seen = np.cumsum(in_range) > 0
    range_high = np.maximum.accumulate(np.where(in_range, high, high.min()))
    range_low = np.minimum.accumulate(np.where(in_range, low, low.max()))
    window = seen & (minute >= range_end) & (minute < params["last_entry_minute"])
    up = window & (close >= range_high + buffer)
    down = window & ~up & (close <= range_low - buffer)
    hits = np.flatnonzero(up | down)
    if len(hits) == 0:
        return -1, True
    first = int(hits[0])
    return first, bool(up[first])


def generate(bars, params):
    """The five signal arrays: entry_long, entry_short, exit (bool) and stop_ticks, target_ticks (int64)."""
    minute = bars["ct_minute"].astype(np.int64)
    n = len(bars["ts_event"])
    entry_long = np.zeros(n, dtype=np.bool_)
    entry_short = np.zeros(n, dtype=np.bool_)
    exit_ = minute >= params["exit_minute"]
    starts, ends = ([], []) if n == 0 else _day_bounds(bars["trade_day"])
    for s, e in zip(starts, ends, strict=True):
        k, is_long = _entry_of_day(
            bars["high"][s:e], bars["low"][s:e], bars["close"][s:e], minute[s:e], params
        )
        if k >= 0:
            entry_long[s + k] = is_long
            entry_short[s + k] = not is_long
    return {
        "entry_long": entry_long,
        "entry_short": entry_short,
        "exit": exit_,
        "stop_ticks": np.full(n, params["stop_ticks"], dtype=np.int64),
        "target_ticks": np.full(n, params["target_ticks"], dtype=np.int64),
    }
