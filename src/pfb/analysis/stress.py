"""S1 cost stress: the run simulated again on the same bars with `stress:<the run's cost model>:<k>`.

Each level re-runs only the kernel (pfb.engine.run.rerun_kernel) on the bars and signals the base simulation
already holds, with the stressed slippage table in place of the run's, and the stressed event-window table
as what a PRICE_LIMIT exit pays (StressCostModel adds the k ticks to it; a model without one, stress over
fixed:<k>, charges the forced exit its own stressed slippage, as a separate run would): no bar read, no
generate call, no second look-ahead check. The same signals give the same trades, so a trade count unlike the
base run's is an error, never a silent difference. The figures equal those of a separate run made with that
cost model (tests/analysis/test_suite_stress.py).

Event costs "on": when the base run was simulated with an
EventCostModel (its replay carries it), each level's model is EventCostModel(stress:<spec>:<k>, the run's
calendar), never stress over the event model (refused by StressCostModel), and the kernel re-run prices
every bar again from the calendar at its open (rerun_kernel's event_costs), so the per-fill event charge
survives the stress: inside a window a fill pays the stressed event table, outside the stressed one.
ctx.cost_model is the base spec ("book"), not the "+events" id.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pfb.analysis.spec import SuiteIntegrityError
from pfb.costs.events import EventCostModel
from pfb.costs.model import CostModel, load_cost_model
from pfb.data.paths import load_paths
from pfb.engine.run import SimOutput, event_table, rerun_kernel, with_event_table
from pfb.metrics.core import summarize

if TYPE_CHECKING:
    from pfb.analysis.inrun import RunContext


def stress_spec(cost_model: str, ticks: int) -> str:
    return f"stress:{cost_model}:{ticks}"


def _uses_book(cost_model: str) -> bool:
    return "book" in cost_model.split(":")


def stressed_model(ctx: RunContext, spec: str) -> CostModel:
    """The level's cost model: the stress spec, wrapped in the run's release calendar when the run was
    simulated with event costs on (EventCostModel(Stress(base)))."""
    model = load_cost_model(spec, ctx.instruments, load_paths() if _uses_book(spec) else None)
    run_events = ctx.sim.replay.event_model if ctx.sim.replay is not None else None
    if run_events is None:
        return model
    return EventCostModel(model, run_events.calendar)


def _rerun(base: SimOutput, model: CostModel, root: str, contracts: int) -> SimOutput:
    slip = np.asarray(model.slippage_cents(root, contracts), dtype=np.int64)
    cal = np.asarray(model.calibrated(root), dtype=np.bool_)
    if isinstance(model, EventCostModel):
        return rerun_kernel(base, slip, cal, event_costs=model)
    return rerun_kernel(base, slip, cal)  # the 3-argument call "off" runs always made


def stress_level(ctx: RunContext, ticks: int) -> dict[str, object]:
    """One level: the run with k extra ticks per side; net_cents, sharpe_annual and trades."""
    spec = stress_spec(ctx.cost_model, ticks)
    model = stressed_model(ctx, spec)
    contracts = ctx.strategy.spec.contracts
    base = with_event_table(ctx.sim, event_table(model, ctx.root, contracts))
    sim = _rerun(base, model, ctx.root, contracts)
    if sim.trades.height != ctx.sim.trades.height:
        raise SuiteIntegrityError(f"cost stress {spec}: {sim.trades.height} trades, the base run has "
                                  f"{ctx.sim.trades.height}")  # fmt: skip
    summary = summarize(sim.trades, sim.equity_minute, sim.daily, n_trials=None)
    return {
        "cost_model": model.model_id,
        "net_cents": summary["net_cents"],
        "sharpe_annual": summary["sharpe_annual"],
        "trades": summary["n_trades"],
        "slippage_cents": summary["slippage_cents"],
    }


def cost_stress(ctx: RunContext, ticks: tuple[int, ...]) -> dict[str, object] | None:
    """{"<k>": level} for each k in order; None when no level is asked for."""
    if not ticks:
        return None
    return {str(k): stress_level(ctx, k) for k in ticks}
