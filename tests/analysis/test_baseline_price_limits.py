"""The random baseline under the price-limit rule (synthetic bars only).

A PRICE_LIMIT exit fills at the open of its exit bar, so its hold is SIGNAL's. The
baseline's entry mask refuses what the kernel refuses, a decision bar on a date without a settlement reference
and one whose close is at or beyond a stop level, read from sim.price_limits; where the rule never acts the
draws are byte-identical to the baseline without the rule.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import date

import numpy as np
import polars as pl
import pytest
from suite_planting import HASH_RANDOM

import pfb.analysis.inrun as inrun
from pfb.analysis import baseline as B
from pfb.analysis.spec import SuiteIntegrityError, SuiteSpec
from pfb.engine import kernel as K
from pfb.engine.run import RunResult, prepare_bars, run_strategy

BASELINE_ONLY = SuiteSpec(cost_stress_ticks=(), baseline_draws=200, baseline_seed=7, walk_forward_folds=0)
TWO_MONTHS = (date(2025, 4, 1), date(2025, 5, 30))
# HASH_RANDOM on synthetic MGC (DCB only: the rule never acts), seed 3, TWO_MONTHS, BASELINE_ONLY, taken from
# the code at HEAD 48beb4e (git archive) before: sha256 of the draws frame's CSV and of the baseline block
HEAD_MGC_DRAWS_SHA256 = "ac7f1772673e09f59937fb8103703c1c93c1c2bac3b8cee53e997a2d745dd2bf"
HEAD_MGC_BASELINE_SHA256 = "3e5531bcf56a0902285acb06582d87b1ece46e0011187a52fc544133dcaf12ad"


def _run(root: str, suite: SuiteSpec = BASELINE_ONLY) -> RunResult:
    return run_strategy(HASH_RANDOM, source="synthetic", cost_model="fixed:1", seed=3, first=TWO_MONTHS[0],
                        last=TWO_MONTHS[1], suite=suite, root=root)  # fmt: skip


def _capture(monkeypatch: pytest.MonkeyPatch) -> list[inrun.RunContext]:
    seen: list[inrun.RunContext] = []
    real = inrun.compute_suite

    def spy(ctx: inrun.RunContext, spec: SuiteSpec):  # type: ignore[no-untyped-def]
        seen.append(ctx)
        return real(ctx, spec)

    monkeypatch.setattr(inrun, "compute_suite", spy)
    return seen


def _market(ctx: inrun.RunContext) -> B.BaselineMarket:
    return inrun.baseline_market(ctx, prepare_bars(ctx.frame, ctx.inputs))


def _drawable(market: B.BaselineMarket, trades: B.BaselineTrades, mode: str) -> np.ndarray:
    """Every bar any trade of the mode can be drawn at (the union of its candidate slices)."""
    cand, lo, hi = B._candidates(market, trades, mode)  # pyright: ignore[reportPrivateUsage]
    return np.unique(np.concatenate([cand[a:b] for a, b in zip(lo.tolist(), hi.tolist(), strict=True)]))


# ---------------------------------------------------------------------------------- the exit fill ----
def test_a_price_limit_exit_holds_as_long_as_a_signal_exit() -> None:
    rows = {"exit_reason": ["SIGNAL", "FLAT_BY", "PRICE_LIMIT", "SESSION_END", "STOP"],
            "entry_bar": [10] * 5, "exit_bar": [15] * 5, "side": [1] * 5, "commission_cents": [100] * 5,
            "slippage_cents": [50] * 5}  # fmt: skip
    trades = B.trades_of(pl.DataFrame(rows))
    assert K.PRICE_LIMIT in B.OPEN_EXITS
    assert trades.hold.tolist() == [4, 4, 4, 5, 5]
    assert trades.exit_slip.tolist() == [True] * 5  # a forced exit is a market fill


# ---------------------------------------------------------------------------------- the entry mask ----
def test_price_limit_refusals_follow_the_kernel_test() -> None:
    close = np.array([100, 100, 120, 80, 100], dtype=np.int64)
    upper = np.array([K.NO_UPPER, 120, 120, K.NO_UPPER, 101], dtype=np.int64)
    lower = np.array([K.NO_LOWER, 80, 80, 80, K.NO_LOWER], dtype=np.int64)
    no_ref = np.array([True, False, False, False, False])
    refused = B.price_limit_refusals(close, upper, lower, no_ref)
    # no reference; inside; at the upper level; at the lower level; inside (sentinels never refuse)
    assert refused.tolist() == [True, False, True, True, False]
    with pytest.raises(SuiteIntegrityError, match="upper has 4 bars"):
        B.price_limit_refusals(close, upper[:4], lower, no_ref)


def test_the_entry_mask_drops_a_refused_decision_bar() -> None:
    n = 5
    ok = B.entry_ok_mask(
        np.zeros(n, bool), np.ones(n, bool), np.ones(n, bool), np.array([0, 0, 1, 0, 0], bool)
    )
    assert ok.tolist() == [False, True, True, False, True]  # bar 3's decision bar 2 is refused
    assert B.entry_ok_mask(np.zeros(n, bool), np.ones(n, bool), np.ones(n, bool), None).tolist() == [
        False, True, True, True, True,
    ]  # fmt: skip


def test_on_a_hard_limit_root_no_draw_enters_on_the_first_date_or_in_the_stop_zone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MNQ (equity bands): the window's first date has no settlement reference. A stop zone is planted in the
    run's own arrays on a later date (the close of every decision bar there at the upper level): no mode can
    draw an entry whose decision bar is on either."""
    seen = _capture(monkeypatch)
    _run("MNQ", suite=SuiteSpec.off())
    ctx = seen[0]
    limits = ctx.sim.price_limits
    assert limits is not None and limits.no_reference.any()
    assert ctx.sim.counters["price_limit_entries_no_reference"] > 0
    prepared = prepare_bars(ctx.frame, ctx.inputs)
    days = prepared.frame["trade_date"].to_numpy()
    zone_day = np.unique(days[~limits.no_reference])[5]
    zone = days == zone_day
    close = prepared.frame["close"].to_numpy().astype(np.int64)
    upper = np.where(zone, close, limits.upper)  # close >= upper: at the stop level
    planted = replace(ctx, sim=replace(ctx.sim, price_limits=replace(limits, upper=upper)))
    market = _market(planted)
    trades = B.trades_of(ctx.sim.trades.filter(pl.col("trade_date") != zone_day))
    refused = limits.no_reference | zone
    for mode in B.MODES:
        drawn = _drawable(market, trades, mode)
        assert len(drawn) > 0
        assert not refused[drawn - 1].any(), mode
    unruled = _market(replace(ctx, sim=replace(ctx.sim, price_limits=None)))  # the mask as at HEAD
    assert unruled.entry_ok[1:][refused[:-1]].any()  # without the refusal these bars were eligible


def test_the_real_trades_stay_eligible_and_the_first_date_leaves_the_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _capture(monkeypatch)
    result = _run("MNQ")
    market = _market(seen[0])
    limits = seen[0].sim.price_limits
    assert limits is not None
    assert not market.entry_ok[1:][limits.no_reference[:-1]].any()
    assert result.summary["suite"]["baseline"] is not None  # type: ignore[index]  # own placements checked


def test_where_the_rule_never_acts_the_draws_are_byte_identical_to_head() -> None:
    result = _run("MGC")
    draws = result.extras["baseline_draws"]
    base = result.summary["suite"]["baseline"]  # type: ignore[index]
    assert hashlib.sha256(draws.write_csv().encode()).hexdigest() == HEAD_MGC_DRAWS_SHA256
    assert hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest() == HEAD_MGC_BASELINE_SHA256
