"""Event costs "on" in the engine (part 1): synthetic bars, the repository's book table and release calendar
(no bar store, nothing registered).

An "off" run is byte-identical to the pinned earlier output (pins below). "On": each fill pays the event
model's price of its bar (event table inside a release window at the bar open, ordinary table outside);
the forced-exit pair stays the event table. The narrower warning. The cost-stress re-run keeps the
per-fill event charge, with EventCostModel(Stress(book)).
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from fractions import Fraction
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import pytest

import pfb.engine.run as R
from pfb.analysis.spec import SuiteSpec
from pfb.costs.events import EventCostModel, EventCoverageError, load_cost_model_for
from pfb.costs.fees import load_fees
from pfb.costs.model import CostModel, CostModelError, StressCostModel
from pfb.data.paths import load_paths
from pfb.engine.synthetic import synthetic_frame
from pfb.instruments.table import load_instruments

REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / "strategies" / "reference_breakout.py"
HASH_RANDOM = REPO / "tests" / "analysis" / "strategies" / "hash_random.py"

# Fingerprints of "off" runs taken from the engine at ba4543d (pfb-engine/0.3.1, before) by
# a local scratch folder on an extract of that commit (18 runs, all identical after):
# sha256 of each frame's columns (name + values as text) and of json.dumps(summary, sort_keys, default=str).
OFF_PINS = {
    (REFERENCE, "fixed:1", 0): (
        "888ad0d8e1bde7fa4e0d821020b7f5bc0a4125774900b55b022eaf71d7c895ba",
        "83f3d922df09f289aaa0bdc376109884c2d5068c2170cf72ce42cfe475267564",
        "d8dd18e3aa53a28e9ee62d2c91f649aa052f219d3d34208f30760f691db37cc3",
        "5e79c7a588a996b9cb9fb0dc35797de680f7d5a9bb825d46168c5371a1d59020",
    ),
}
STRESS_SUITE = SuiteSpec(cost_stress_ticks=(1, 2), baseline_draws=0, walk_forward_folds=0)

def _frame_sha(df: pl.DataFrame) -> str:
    h = hashlib.sha256()
    for name in df.columns:
        h.update(name.encode())
        h.update(json.dumps(df[name].cast(str).to_list()).encode())
    return h.hexdigest()


def _summary_sha(summary: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(summary, sort_keys=True, default=str).encode()).hexdigest()


def _wired(
    strategy: Path, spec: str, on: bool, seed: int = 0
) -> tuple[R.SimOutput, CostModel, R._Request, R.MarketInputs]:
    """run_strategy's wiring up to simulate, synthetic bars."""
    instruments = load_instruments()
    req = R._resolve(strategy, "synthetic", None, None, None, None)
    cost = load_cost_model_for(spec, "on" if on else "off", instruments, load_paths())
    costs = R._costs(req, instruments, cost, load_fees("topstep", instruments=instruments))
    frame = synthetic_frame(req.root, req.first, req.last, seed=seed)
    inputs = R._market_inputs(req, frame, costs)
    event_model = cost if isinstance(cost, EventCostModel) else None
    sim = R.simulate(
        req.strategy, frame, req.params, inputs, seed=seed, root=req.root, event_costs=event_model
    )
    return sim, cost, req, inputs


# ------------------------------------------------------------------ "off" is the pinned earlier output ----
@pytest.mark.parametrize(("strategy", "spec", "seed"), list(OFF_PINS))
def test_an_off_run_is_byte_identical_to_ba4543d(strategy: Path, spec: str, seed: int) -> None:
    # Act
    suite = STRESS_SUITE if seed == 3 else None
    result = R.run_strategy(strategy, source="synthetic", cost_model=spec, seed=seed, n_trials=5, suite=suite)
    # Assert
    trades, daily, summary, config = OFF_PINS[(strategy, spec, seed)]
    assert _frame_sha(result.trades) == trades
    assert _frame_sha(result.daily) == daily
    assert _summary_sha(result.summary) == summary
    assert result.provenance["cost_config_sha256"] == config
    assert result.provenance["cost_model_id"] == spec and result.provenance["event_windows"] == "off"
    assert result.summary["event_windows"] == "off" and R.EVENT_WARNING in _warnings(result)


# ------------------------------------------------------------------ "on": per-bar charge ----
def test_on_charges_each_fill_bar_by_the_calendar_and_nothing_else_moves() -> None:
    # Arrange
    off, _, _, _ = _wired(HASH_RANDOM, "book", on=False)
    on, model, req, inputs = _wired(HASH_RANDOM, "book", on=True)
    assert isinstance(model, EventCostModel) and on.replay is not None and on.replay.ts is not None
    assert on.replay.days is not None
    # Act
    bar = model.bar_slippage_cents(req.root, req.strategy.spec.contracts, on.replay.ts, on.replay.minute,
                                   on.replay.days)  # fmt: skip
    mask = model.event_bars(req.root, on.replay.ts)
    # Assert: the kernel was handed exactly the per-bar prices
    assert np.array_equal(on.replay.kernel_in.slip_buy, bar[:, 0])
    assert np.array_equal(on.replay.kernel_in.slip_sell, bar[:, 1])
    assert np.array_equal(bar[~mask], inputs.slippage_cents[on.replay.minute[~mask]])
    assert np.array_equal(bar[mask], np.asarray(inputs.event_slippage_cents)[on.replay.minute[mask]])
    # a trade pays its entry fill bar's side price plus its exit fill bar's (no PRICE_LIMIT, no TARGET here)
    t = on.trades
    assert set(t["exit_reason"].to_list()) <= {"SIGNAL", "STOP", "SESSION_END", "FLAT_BY"}
    side = t["side"].to_numpy()
    e, x = t["entry_bar"].to_numpy(), t["exit_bar"].to_numpy()
    entry_col, exit_col = np.where(side > 0, 0, 1), np.where(side > 0, 1, 0)
    assert np.array_equal(t["slippage_cents"].to_numpy(), bar[e, entry_col] + bar[x, exit_col])
    # same trades, prices and bars; only the fills inside a window pay more, and net by exactly that
    in_window = mask[e] | mask[x]
    assert int(in_window.sum()) > 0
    same = [c for c in t.columns if c not in ("slippage_cents", "net_cents")]
    assert t.select(same).equals(off.trades.select(same))
    delta = t["slippage_cents"].to_numpy() - off.trades["slippage_cents"].to_numpy()
    assert bool(np.all(delta[~in_window] == 0)) and bool(np.all(delta[in_window] >= 0)) and delta.sum() > 0
    assert np.array_equal(off.trades["net_cents"].to_numpy() - t["net_cents"].to_numpy(), delta)
    # the forced-exit pair is the event table at the minute under "on" as under "off"
    assert np.array_equal(on.replay.kernel_in.slip_pl_buy, off.replay.kernel_in.slip_pl_buy)  # type: ignore[union-attr]


def test_simulate_refuses_an_event_model_unlike_the_runs_tables() -> None:
    # Arrange: the inputs of an "off" book run, the event model over stress:book:1
    _, _, req, inputs = _wired(REFERENCE, "book", on=False)
    other = load_cost_model_for("stress:book:1", "on", load_instruments(), load_paths())
    frame = synthetic_frame(req.root, req.first, req.last, seed=0)
    # Act / Assert
    with pytest.raises(R.EngineInputError, match="unlike slippage_cents"):
        R.simulate(req.strategy, frame, req.params, inputs, seed=0, root=req.root,
                   event_costs=other)  # type: ignore[arg-type]  # fmt: skip


# ------------------------------------------------------------------ the stress re-run ----
def _stressed(ticks: int) -> EventCostModel:
    model = load_cost_model_for(f"stress:book:{ticks}", "on", load_instruments(), load_paths())
    assert isinstance(model, EventCostModel)
    return model


def test_the_stress_rerun_of_an_on_run_equals_a_separate_on_run_with_the_stressed_model() -> None:
    # Arrange
    on, _, req, _ = _wired(HASH_RANDOM, "book", on=True)
    stressed = _stressed(2)
    contracts = req.strategy.spec.contracts
    # Act
    base = R.with_event_table(on, R.event_table(stressed, req.root, contracts))
    rerun = R.rerun_kernel(base, np.asarray(stressed.slippage_cents(req.root, contracts)),
                           np.asarray(stressed.calibrated(req.root)), event_costs=stressed)  # fmt: skip
    alone, _, _, _ = _wired(HASH_RANDOM, "stress:book:2", on=True)
    off_stress, _, _, _ = _wired(HASH_RANDOM, "stress:book:2", on=False)
    # Assert: exactly the separate run, and dearer than stress without the calendar
    assert rerun.trades.equals(alone.trades) and rerun.daily.equals(alone.daily)
    assert rerun.trades["slippage_cents"].sum() > off_stress.trades["slippage_cents"].sum()


def test_a_rerun_of_an_on_run_without_the_event_model_is_refused_and_the_reverse() -> None:
    on, _, req, _ = _wired(REFERENCE, "book", on=True)
    off, _, _, _ = _wired(REFERENCE, "book", on=False)
    stressed = _stressed(1)
    slip = np.asarray(stressed.slippage_cents(req.root, req.strategy.spec.contracts))
    cal = np.asarray(stressed.calibrated(req.root))
    with pytest.raises(R.EngineInputError, match="per-fill event charge is lost"):
        R.rerun_kernel(on, slip, cal)
    with pytest.raises(R.EngineInputError, match="event costs off"):
        R.rerun_kernel(off, slip, cal, event_costs=stressed)


def test_stress_over_an_event_model_is_refused() -> None:
    on = load_cost_model_for("book", "on", load_instruments(), load_paths())
    with pytest.raises(CostModelError, match="never stress over an event model"):
        StressCostModel(on, Fraction(1), "1", load_instruments())


def test_the_suite_stress_levels_keep_the_event_costs() -> None:
    # Act
    result = R.run_strategy(HASH_RANDOM, source="synthetic", cost_model="book", seed=0, suite=STRESS_SUITE,
                            event_costs="on")  # fmt: skip
    # Assert: each level is EventCostModel(stress:book:k) and equals a separate "on" run with that model
    levels = result.summary["suite"]["cost_stress"]  # type: ignore[index]
    assert levels is not None
    for k in (1, 2):
        alone, _, _, _ = _wired(HASH_RANDOM, f"stress:book:{k}", on=True)
        level = levels[str(k)]
        assert level["cost_model"] == f"stress:book:{k}+events"
        assert level["slippage_cents"] == int(alone.trades["slippage_cents"].sum())
        assert level["net_cents"] == int(alone.trades["net_cents"].sum())


# ---------------------------------------------------------------- the narrower warning and run_strategy ----
def test_run_strategy_on_records_on_and_the_narrower_warning() -> None:
    # Act
    on = R.run_strategy(REFERENCE, source="synthetic", cost_model="book", seed=0, event_costs="on")
    off = R.run_strategy(REFERENCE, source="synthetic", cost_model="book", seed=0)
    # Assert
    assert on.summary["event_windows"] == "on" and on.provenance["event_windows"] == "on"
    assert on.provenance["cost_model_id"] == "book+events"
    assert on.provenance["cost_config_sha256"] != off.provenance["cost_config_sha256"]
    assert R.EVENT_ON_WARNING in _warnings(on) and R.EVENT_WARNING not in _warnings(on)
    assert "2-minute fill guard" in R.EVENT_ON_WARNING and "CPI opening rule" in R.EVENT_ON_WARNING
    assert R.EVENT_WARNING in _warnings(off)


def _warnings(result: R.RunResult) -> list[str]:
    return cast(list[str], result.summary["warnings"])


def test_run_strategy_on_refuses_a_window_the_calendar_does_not_cover() -> None:
    with pytest.raises(EventCoverageError):
        R.run_strategy(REFERENCE, source="synthetic", cost_model="book", first=date(2019, 4, 1),
                       last=date(2019, 4, 30), event_costs="on")  # fmt: skip


def test_run_strategy_refuses_on_over_fixed_and_an_unknown_switch() -> None:
    with pytest.raises(CostModelError, match="book-based"):
        R.run_strategy(REFERENCE, source="synthetic", cost_model="fixed:1", event_costs="on")
    with pytest.raises(CostModelError, match='"off" or "on"'):
        R.run_strategy(REFERENCE, source="synthetic", cost_model="book", event_costs="maybe")
