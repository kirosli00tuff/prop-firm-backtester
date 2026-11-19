"""The suite's cross-checks of the base run's own trades and
days raise SuiteIntegrityError, which is not one of SUITE_PART_ERRORS, so the run still fails. A part that
cannot be computed (FoldError, sizing plan, PowerInputError, a trade with no eligible placement) stays null
with a warning. Synthetic bars only; nothing is registered."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import polars as pl
import pytest
from suite_planting import REFERENCE

import pfb.analysis.stress as stress
from pfb.analysis import baseline as B
from pfb.analysis import breakdowns as BD
from pfb.analysis.inrun import SUITE_PART_ERRORS
from pfb.analysis.spec import SuiteIntegrityError, SuiteSpec
from pfb.engine import kernel as K
from pfb.engine import run as engine_run
from pfb.engine.run import RunResult, run_strategy

DAYS = [date(2025, 4, 1), date(2025, 4, 2)]


def _run(spec: SuiteSpec) -> RunResult:
    return run_strategy(REFERENCE, source="synthetic", cost_model="fixed:1", seed=3, n_trials=1000,
                        suite=spec)  # fmt: skip


def test_the_integrity_error_is_not_a_suite_part_error() -> None:
    assert not issubclass(SuiteIntegrityError, SUITE_PART_ERRORS)
    assert not issubclass(SuiteIntegrityError, ValueError)


# ------------------------------------------------------------------------------- each site ----


def test_breakdowns_refuse_a_ledger_net_unlike_the_daily_net() -> None:
    trades = pl.DataFrame({"net_cents": [5], "trade_date": [DAYS[0]], "entry_bar": [0]})
    daily = pl.DataFrame({"trade_date": DAYS, "net_cents": [5, 1]})
    with pytest.raises(SuiteIntegrityError, match="differs from the daily net"):
        BD.breakdowns(trades, daily, pl.DataFrame({"ct_minute": [500]}))


def test_breakdowns_refuse_a_run_day_without_a_usable_bar() -> None:
    bars = pl.DataFrame({"trade_date": [DAYS[0]], "high": [10], "low": [5]})
    with pytest.raises(SuiteIntegrityError, match="no usable bar"):
        BD.day_ranges(bars, pl.Series("trade_date", DAYS))


def test_breakdown_buckets_that_do_not_sum_to_the_run_net_are_refused() -> None:
    with pytest.raises(SuiteIntegrityError, match="bucket nets sum"):
        BD._check_sum("by_year", [{"net_cents": 3}], 4)  # pyright: ignore[reportPrivateUsage]


def test_baseline_refuses_a_trade_that_exits_at_the_open_of_its_entry_bar() -> None:
    trades = pl.DataFrame({"exit_reason": [K.EXIT_REASONS[K.SIGNAL]], "entry_bar": [4], "exit_bar": [4],
                           "side": [1]})  # fmt: skip
    with pytest.raises(SuiteIntegrityError, match="open of its own entry bar"):
        B.trades_of(trades)


def _market(n_days: int = 1) -> Any:
    return SimpleNamespace(entry_ok=np.array([False, True]), room=np.array([5, 5], dtype=np.int64),
                           n_days=n_days)  # fmt: skip


def _trades() -> B.BaselineTrades:
    one = np.array([0], dtype=np.int64)
    return B.BaselineTrades(side=one + 1, entry_bar=one, hold=one, exit_slip=np.array([True]), own_cost=one)


def test_baseline_refuses_a_real_trade_that_is_not_an_eligible_placement() -> None:
    with pytest.raises(SuiteIntegrityError, match="not an eligible placement"):
        B.check_own_placements(_market(), _trades())


def test_baseline_refuses_daily_nets_that_do_not_cover_the_run_days() -> None:
    with pytest.raises(SuiteIntegrityError, match="do not cover the run's days"):
        B.random_baseline(
            _market(n_days=3), _trades(), np.zeros(2, dtype=np.int64), draws=B.MIN_DRAWS, seed=1
        )


def test_a_baseline_trade_without_an_eligible_entry_bar_stays_a_part_error() -> None:
    assert issubclass(B.BaselineError, SUITE_PART_ERRORS)


# --------------------------------------------------------------- the run fails, end to end ----


def test_a_stress_trade_count_unlike_the_base_run_fails_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    real = engine_run.rerun_kernel

    def one_trade_less(sim: Any, slippage: Any, calibrated: Any) -> Any:
        out = real(sim, slippage, calibrated)
        return replace(out, trades=out.trades.head(out.trades.height - 1))

    monkeypatch.setattr(stress, "rerun_kernel", one_trade_less)
    with pytest.raises(SuiteIntegrityError, match="trades, the base run has"):
        _run(SuiteSpec())


def test_a_daily_net_unlike_the_ledger_fails_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    import pfb.analysis.inrun as inrun

    real = inrun.breakdowns

    def drifted(trades: pl.DataFrame, daily: pl.DataFrame, bars: pl.DataFrame) -> Any:
        moved = daily.with_columns(pl.col("net_cents") + pl.lit(1, dtype=pl.Int64))
        return real(trades, moved, bars)

    monkeypatch.setattr(inrun, "breakdowns", drifted)
    with pytest.raises(SuiteIntegrityError, match="differs from the daily net"):
        _run(SuiteSpec())


def test_a_part_error_still_leaves_the_run_ok_with_a_warning() -> None:
    result = _run(SuiteSpec(walk_forward_folds=10**6))
    suite = cast(dict[str, Any], result.summary["suite"])
    assert suite["walk_forward"] is None and suite["breakdowns"] is not None
    warnings = cast(list[str], result.summary["warnings"])
    assert [w for w in warnings if w.startswith("suite: ")][0].startswith(
        "suite: walk_forward not computed: FoldError: "
    )
