"""The in-run suite's random baseline reads
(1) the per-bar slippage the kernel ran with (ctx.sim.replay.kernel_in), so an event_costs "on" baseline pays
the event prices and an "off" one exactly the minute table, and (2) the trade frame's after-lock flags, so a
STOP queued on a locked bar and filled at a later open is an open exit. Synthetic bars; the book and
the release calendar read in place (nothing registered, no store opened)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import pfb.analysis.inrun as I
import pfb.engine.run as R
from pfb.analysis.spec import SuiteSpec
from pfb.costs.events import EventCostModel, load_cost_model_for
from pfb.costs.fees import load_fees
from pfb.data.paths import load_paths
from pfb.engine.synthetic import synthetic_frame
from pfb.instruments.table import load_instruments

REPO = Path(__file__).resolve().parents[2]
HASH_RANDOM = REPO / "tests" / "analysis" / "strategies" / "hash_random.py"

def _ctx(on: bool) -> tuple[I.RunContext, R.PreparedBars]:
    """run_strategy's wiring up to the suite (as tests/engine/test_event_costs_engine.py _wired does)."""
    instruments = load_instruments()
    req = R._resolve(HASH_RANDOM, "synthetic", None, None, None, None)  # pyright: ignore[reportPrivateUsage]
    cost = load_cost_model_for("book", "on" if on else "off", instruments, load_paths())
    fees = load_fees("topstep", instruments=instruments)
    costs = R._costs(req, instruments, cost, fees)  # pyright: ignore[reportPrivateUsage]
    frame = synthetic_frame(req.root, req.first, req.last, seed=0)
    inputs = R._market_inputs(req, frame, costs)  # pyright: ignore[reportPrivateUsage]
    event_model = cost if isinstance(cost, EventCostModel) else None
    sim = R.simulate(req.strategy, frame, req.params, inputs, seed=0, root=req.root, event_costs=event_model)
    ctx = I.RunContext(
        req.strategy, frame, req.params, inputs, 0, None, "book", req.root, instruments, sim, 5
    )
    return ctx, R.prepare_bars(frame, inputs)


def _minute_table(ctx: I.RunContext, prepared: R.PreparedBars) -> tuple[np.ndarray, np.ndarray]:
    minute = prepared.frame["ct_minute"].to_numpy().astype(np.int64)
    table = ctx.inputs.slippage_cents
    return table[minute, 0].astype(np.int64), table[minute, 1].astype(np.int64)


def test_an_off_baseline_pays_exactly_the_minute_table() -> None:
    # Arrange
    ctx, prepared = _ctx(on=False)
    # Act
    market = I.baseline_market(ctx, prepared)
    # Assert
    buy, sell = _minute_table(ctx, prepared)
    assert np.array_equal(market.slip_buy, buy) and np.array_equal(market.slip_sell, sell)


def test_an_on_baseline_pays_the_per_bar_prices_the_kernel_ran_with() -> None:
    # Arrange
    ctx, prepared = _ctx(on=True)
    assert ctx.sim.replay is not None
    # Act
    market = I.baseline_market(ctx, prepared)
    # Assert
    kernel_in = ctx.sim.replay.kernel_in
    assert np.array_equal(market.slip_buy, kernel_in.slip_buy)
    assert np.array_equal(market.slip_sell, kernel_in.slip_sell)
    buy, _ = _minute_table(ctx, prepared)
    assert bool(np.any(market.slip_buy != buy))  # some bar of the run lies in a release window
    assert bool(np.all(market.slip_buy >= buy))  # the event price is never below the ordinary one


def test_a_sim_without_a_replay_falls_back_to_the_minute_table() -> None:
    ctx, prepared = _ctx(on=False)
    bare = replace(ctx, sim=replace(ctx.sim, replay=None))
    buy, sell = _minute_table(ctx, prepared)
    got_buy, got_sell = I.bar_slippage(bare, prepared.frame["ct_minute"].to_numpy().astype(np.int64))
    assert np.array_equal(got_buy, buy) and np.array_equal(got_sell, sell)


def test_the_baseline_passes_the_after_lock_flags_to_trades_of(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    ctx, prepared = _ctx(on=False)
    seen: list[Any] = []
    real = I.B.trades_of

    def spy(trades: Any, after_lock: Any = None) -> Any:
        seen.append(after_lock)
        return real(trades, after_lock=after_lock)

    monkeypatch.setattr(I.B, "trades_of", spy)
    # Act
    I._baseline(ctx, prepared, SuiteSpec(baseline_draws=200))  # pyright: ignore[reportPrivateUsage]
    # Assert
    assert len(seen) == 1 and seen[0] is ctx.sim.exit_after_lock and seen[0] is not None
