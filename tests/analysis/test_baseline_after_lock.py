"""A STOP queued on a locked bar and filled at the open of a later bar is an open
exit for the baseline's hold, told apart from an intrabar STOP by the kernel's T_AFTER_LOCK marker
(SimOutput.exit_after_lock), which is not a trade column (artifacts do not move)."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import pytest

from pfb.analysis import baseline as B
from pfb.analysis.spec import SuiteIntegrityError
from pfb.engine import kernel as K


def _engine_cases_kernel_inputs() -> Callable[..., K.KernelInputs]:
    """tests/engine/engine_cases.py's kernel_inputs (the builder 's R1 tests use), loaded by path."""
    path = Path(__file__).resolve().parents[1] / "engine" / "engine_cases.py"
    spec = importlib.util.spec_from_file_location("p3b_engine_cases", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return cast(Callable[..., K.KernelInputs], cast(Any, module).kernel_inputs)


kernel_inputs = _engine_cases_kernel_inputs()

# the reproducer R1 (tests/engine/test_locked_exits_and_forced_exit_cost.py): long from bar 1, stop 950,
# bars 2 and 3 wholly at the limit-down price 900, bar 4 opens at 960: STOP at the open of bar 4
R1_BARS = [(1000, 1001, 999, 1000), (1000, 1001, 999, 1000), (900, 900, 900, 900), (900, 900, 900, 900),
           (960, 970, 955, 965), (965, 970, 960, 968), (968, 970, 966, 969)]  # fmt: skip
LOCK = 900


def _ledger(out: K.KernelOutput) -> pl.DataFrame:
    t = out.trades
    return pl.DataFrame({
        "side": t[:, K.T_SIDE], "entry_bar": t[:, K.T_ENTRY_IDX], "exit_bar": t[:, K.T_EXIT_IDX],
        "exit_reason": [K.EXIT_REASONS[int(r)] for r in t[:, K.T_REASON]],
        "commission_cents": t[:, K.T_COMM], "slippage_cents": t[:, K.T_SLIP],
    })  # fmt: skip


def test_the_kernel_marks_a_stop_filled_after_a_lock_and_not_an_intrabar_one() -> None:
    inputs = kernel_inputs(R1_BARS, long=[0], stops={0: 50})
    n = len(R1_BARS)
    locked = K.run_kernel(dataclasses.replace(inputs, pl_lock_down=np.full(n, LOCK, dtype=np.int64)))
    plain = K.run_kernel(inputs)
    assert locked.trades[:, K.T_AFTER_LOCK].tolist() == [1]
    assert plain.trades[:, K.T_AFTER_LOCK].tolist() == [0]


def test_trades_of_reads_a_queued_stop_as_an_open_exit_only_with_the_marker() -> None:
    # Arrange
    inputs = kernel_inputs(R1_BARS, long=[0], stops={0: 50})
    n = len(R1_BARS)
    out = K.run_kernel(dataclasses.replace(inputs, pl_lock_down=np.full(n, LOCK, dtype=np.int64)))
    ledger = _ledger(out)
    marker = out.trades[:, K.T_AFTER_LOCK] == 1
    # Act
    with_marker = B.trades_of(ledger, after_lock=marker)
    without = B.trades_of(ledger)
    # Assert: entry bar 1, exit at the open of bar 4: held through the closes of bars 1..3, h = 4 - 1 - 1
    assert with_marker.hold.tolist() == [2]
    assert without.hold.tolist() == [3]  # the old reading (intrabar STOP)
    assert with_marker.exit_slip.tolist() == [True]


def test_an_intrabar_stop_keeps_its_hold_with_the_marker() -> None:
    out = K.run_kernel(kernel_inputs(R1_BARS, long=[0], stops={0: 50}))
    ledger = _ledger(out)
    marker = out.trades[:, K.T_AFTER_LOCK] == 1
    assert B.trades_of(ledger, after_lock=marker).hold.tolist() == B.trades_of(ledger).hold.tolist() == [1]


def test_a_malformed_marker_is_refused() -> None:
    ledger = _ledger(K.run_kernel(kernel_inputs(R1_BARS, long=[0], stops={0: 50})))
    with pytest.raises(SuiteIntegrityError, match="one bool per trade"):
        B.trades_of(ledger, after_lock=np.zeros(2, dtype=np.bool_))
