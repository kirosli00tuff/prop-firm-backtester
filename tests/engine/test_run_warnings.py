"""Synthetic runs: the in-run check is step 1's check without the sweep,
a check that saw no signal warns, every result carries the standing warning, a cost model without slippage
warns, and the engine version rose with the fill convention."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

from pfb.engine import ENGINE_VERSION
from pfb.engine.canaries import zero_cost_inputs
from pfb.engine.lookahead import NO_SIGNAL_WARNING, LookAheadDetected
from pfb.engine.run import NO_SLIPPAGE_WARNING, STANDING_WARNING, run_strategy, simulate
from pfb.engine.strategy import Strategy, load_strategy, resolve_params
from pfb.engine.synthetic import synthetic_frame
from pfb.runner.deps import _lookahead

REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / "strategies" / "reference_breakout.py"
sys.path.insert(0, str(REPO / "tests" / "lookahead"))
from lookahead_mutants import next_bar_view  # noqa: E402  # pyright: ignore[reportMissingImports]

NEVER = '''"""Never signals (test only)."""

import numpy as np

SPEC = {"hypothesis_id": "test:never", "root": "MNQ", "bar_minutes": 1, "session": "globex", "contracts": 1,
        "params": {}}


def generate(bars, params):
    n = len(bars["close"])
    flags = np.zeros(n, dtype=np.bool_)
    zeros = np.zeros(n, dtype=np.int64)
    return {"entry_long": flags, "entry_short": flags.copy(), "exit": flags.copy(), "stop_ticks": zeros,
            "target_ticks": zeros.copy()}
'''


def test_e4_simulate_cuts_after_entry_bars_on_the_runs_own_bars() -> None:
    # Arrange: the reference fed the next bar (sparse: about one entry a day); before E4 simulate ran the
    # 20 random cuts only
    ref = load_strategy(REFERENCE)
    frame = synthetic_frame("MNQ", date(2025, 4, 1), date(2025, 4, 11), seed=3)

    def peeking(bars, p):  # type: ignore[no-untyped-def]
        return ref.generate(next_bar_view(bars), p)

    leaky = Strategy(ref.spec, peeking, ref.sha256, ref.path)
    # Act / Assert
    with pytest.raises(LookAheadDetected, match="cut after a signal bar"):
        simulate(leaky, frame, resolve_params(ref.spec, None), zero_cost_inputs("MNQ", frame), seed=3)


def test_e4_a_check_that_saw_no_signal_warns_at_step_1_and_in_the_result(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "never.py"
    path.write_text(NEVER, encoding="utf-8")
    strategy = load_strategy(path)
    # Act
    step1 = _lookahead(strategy, {}, 3)
    result = run_strategy(path, source="synthetic", cost_model="fixed:1", seed=3)
    # Assert
    assert step1 == (NO_SIGNAL_WARNING,)
    assert NO_SIGNAL_WARNING in result.summary["warnings"]  # type: ignore[operator]


def test_e4_the_reference_sees_signals_and_does_not_warn() -> None:
    strategy = load_strategy(REFERENCE)
    assert _lookahead(strategy, resolve_params(strategy.spec, None), 3) == ()


def test_e7_e9_every_result_warns_of_price_limits_and_a_zero_slippage_model_warns() -> None:
    # Act
    zero = run_strategy(REFERENCE, source="synthetic", cost_model="fixed:0", seed=3)
    one = run_strategy(REFERENCE, source="synthetic", cost_model="fixed:1", seed=3)
    # Assert:: the rule ran on every date but the first, where entries were refused and
    # no position was open, so the standing warning drops and the narrower count stays
    for run in (zero, one):
        warnings = run.summary["warnings"]
        assert STANDING_WARNING not in warnings  # type: ignore[operator]
        assert any(w.startswith("price-limit rule: entries refused on 1 of ") for w in warnings)  # type: ignore[union-attr]
    assert NO_SLIPPAGE_WARNING in zero.summary["warnings"]  # type: ignore[operator]
    assert NO_SLIPPAGE_WARNING not in one.summary["warnings"]  # type: ignore[operator]


def test_e10_the_engine_version_rose_with_the_e5_fill_convention() -> None:
    # 0.2.0 fill convention; 0.3.0 the price-limit rule on; 0.3.1 the lock and forced-exit cost
    # fixes (no trial ran at 0.3.0)
    assert ENGINE_VERSION == "pfb-engine/0.3.1"
