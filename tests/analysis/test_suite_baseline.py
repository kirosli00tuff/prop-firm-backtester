"""S2 random-entry baseline: comparability rules and known answers, on synthetic bars only."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from suite_planting import (
    EVEN_DAY_PLANTED,
    HASH_RANDOM,
    HOURLY_PLANTED,
    REFERENCE,
    SESSION_HOLD,
    TOD_ZERO_EDGE,
    drifting_frame,
    even_day_planted_frame,
    planted_frame,
    use_frame,
)

import pfb.analysis.inrun as inrun
from pfb.analysis import baseline as B
from pfb.analysis.spec import SuiteSpec
from pfb.engine.run import RunResult, run_strategy

BASELINE_ONLY = SuiteSpec(cost_stress_ticks=(), baseline_draws=200, baseline_seed=7, walk_forward_folds=0)
TWO_MONTHS = (date(2025, 4, 1), date(2025, 5, 30))


def _run(strategy: Path, *, seed: int = 3, params: dict[str, object] | None = None,
         suite: SuiteSpec = BASELINE_ONLY, window: tuple[date, date] = TWO_MONTHS) -> RunResult:  # fmt: skip
    return run_strategy(strategy, source="synthetic", cost_model="fixed:1", seed=seed, first=window[0],
                        last=window[1], params=params, suite=suite)  # type: ignore[arg-type]  # fmt: skip


def _baseline(result: RunResult) -> dict[str, object]:
    out = result.summary["suite"]["baseline"]  # type: ignore[index]
    assert isinstance(out, dict)
    return out


# ------------------------------------------------------------------------------ rules ----
def test_segment_room_stops_at_session_ends_and_before_flat_bars() -> None:
    session_end = np.array([0, 0, 1, 0, 0, 0, 0, 1], dtype=bool)
    flat_now = np.array([0, 0, 0, 0, 0, 1, 0, 0], dtype=bool)
    room = B.segment_room(session_end, flat_now)
    # segment lasts: bar 2 (session end), bar 4 (followed by the flat bar 5), bar 5, bar 7
    assert room.tolist() == [2, 1, 0, 1, 0, 0, 1, 0]


def test_entry_needs_a_non_session_end_decision_bar_an_allowed_one_and_a_fill_ok_bar() -> None:
    session_end = np.array([0, 0, 1, 0, 0], dtype=bool)
    fill_ok = np.array([1, 1, 1, 1, 0], dtype=bool)
    allowed = np.array([1, 0, 1, 1, 1], dtype=bool)
    ok = B.entry_ok_mask(session_end, fill_ok, allowed)
    # bar 0: no decision bar; bar 1: decision 0 ok; bar 2: decision 1 not allowed; bar 3: decision 2 is a
    # session end; bar 4: not fill-ok
    assert ok.tolist() == [False, True, False, False, False]


def _capture(monkeypatch: pytest.MonkeyPatch) -> list[inrun.RunContext]:
    seen: list[inrun.RunContext] = []
    real = inrun.compute_suite

    def spy(ctx: inrun.RunContext, spec: SuiteSpec):  # type: ignore[no-untyped-def]
        seen.append(ctx)
        return real(ctx, spec)

    monkeypatch.setattr(inrun, "compute_suite", spy)
    return seen


def _market(ctx: inrun.RunContext) -> B.BaselineMarket:
    from pfb.engine.run import prepare_bars

    return inrun.baseline_market(ctx, prepare_bars(ctx.frame, ctx.inputs))


def test_session_end_trades_placed_at_their_own_bars_reproduce_the_engine_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _capture(monkeypatch)
    result = _run(SESSION_HOLD)
    ledger = result.trades.filter(pl.col("exit_reason") == "SESSION_END")
    assert ledger.height >= 0.8 * result.trades.height  # early-close days end at a flat-by instead
    trades = B.trades_of(ledger)
    nets = B.placement_nets(_market(seen[0]), trades, trades.entry_bar)
    assert nets.tolist() == ledger["net_cents"].to_list()


@pytest.mark.parametrize("strategy", [REFERENCE, HASH_RANDOM, HOURLY_PLANTED, SESSION_HOLD])
def test_every_real_trade_is_an_eligible_placement(monkeypatch: pytest.MonkeyPatch, strategy: Path) -> None:
    seen = _capture(monkeypatch)
    result = _run(strategy, suite=SuiteSpec.off())
    B.check_own_placements(_market(seen[0]), B.trades_of(result.trades))


def test_draws_keep_trade_count_side_mix_and_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Placed at random bars, the per-draw gross can differ, but the count of fills and their commissions
    cannot: with zero prices moves (open == close everywhere) every draw nets exactly the real costs."""
    seen = _capture(monkeypatch)
    result = _run(HASH_RANDOM, suite=SuiteSpec.off())
    market = _market(seen[0])
    flat = B.BaselineMarket(**{**market.__dict__, "open": np.zeros_like(market.open),
                               "close": np.zeros_like(market.close)})  # fmt: skip
    trades = B.trades_of(result.trades)
    out = B.run_mode(flat, trades, "uniform", 200, 1)
    commission = 2 * market.commission_cents * len(trades.side)
    # fixed:1 slippage is the same at every minute and side, so cost per placement is constant
    slip = int(market.slip_buy[0]) * (len(trades.side) + int(trades.exit_slip.sum()))
    assert set(out.nets.tolist()) == {-(commission + slip)}


# ------------------------------------------------------------------------- known answers ----
def test_a_zero_edge_strategy_sits_inside_the_distribution_for_most_seeds() -> None:
    inside = []
    for salt in range(20):
        base = _baseline(_run(HASH_RANDOM, seed=100 + salt, params={"salt": salt}))
        inside.append(0.05 <= float(base["net_percentile"]) <= 0.95)  # type: ignore[arg-type]
    assert sum(inside) >= 16, inside  # at least 80 % of 20 seeds


def test_a_pure_time_of_day_edge_is_above_uniform_but_reported_in_the_tod_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_frame(monkeypatch, planted_frame)
    result = _run(HOURLY_PLANTED)
    base = _baseline(result)
    assert result.summary["net_cents"] > 0  # type: ignore[operator]
    # uniform draws do not earn the move; the time-of-day mode keeps each entry minute, so it earns it too
    uniform, tod = base["uniform"], base["tod"]
    assert isinstance(uniform, dict) and isinstance(tod, dict)
    assert uniform["net"]["q95"] < 0
    assert uniform["net"]["percentile"] > 0.95 and uniform["net"]["p"] == pytest.approx(1 / 201)
    assert tod["net"]["q05"] > 0.95 * result.summary["net_cents"]  # type: ignore[operator]
    # the headline is the mode with the larger p_net (TRIALS.md section 7): an edge that is only the minute
    # of day is not distinguished from random entry at the same minutes
    assert base["mode"] == "tod" and base["p_net"] == tod["net"]["p"] > uniform["net"]["p"]


def test_a_planted_edge_on_some_days_sits_above_the_reported_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_frame(monkeypatch, even_day_planted_frame)
    result = _run(EVEN_DAY_PLANTED)
    base = _baseline(result)
    assert result.summary["net_cents"] > 0  # type: ignore[operator]
    assert base["mode"] == "cost_matched"  # every mode has every draw below: a tie goes to cost_matched
    assert base["net_percentile"] > 0.95 and base["sharpe_percentile"] > 0.95  # type: ignore[operator]
    assert base["p_net"] == pytest.approx(1 / 201) and base["p_sharpe"] == pytest.approx(1 / 201)
    for mode in B.MODES:
        part = base[mode]
        assert isinstance(part, dict) and part["net"]["p"] == pytest.approx(1 / 201)


ZERO_EDGE_SEEDS = range(10)
ZERO_EDGE_EVERY = (1, 4)  # 1: fills every minute-day cell it uses; 4: a quarter of them
APRIL = (date(2025, 4, 1), date(2025, 4, 30))


def _inside(value: object) -> bool:
    return 0.05 <= float(value) <= 0.95  # type: ignore[arg-type]


def test_a_zero_edge_time_of_day_rule_under_the_book_costs_sits_inside_the_reported_distribution() -> None:
    # uniform draws enter at any minute and pay overnight slippage, so this rule sat at the top
    # of the uniform distribution with no edge; the reported mode must not place it there (S8: at least 80 %)
    suite = SuiteSpec(cost_stress_ticks=(), baseline_draws=1000, baseline_seed=7, walk_forward_folds=0)
    reported, uniform = [], []
    for every in ZERO_EDGE_EVERY:
        for seed in ZERO_EDGE_SEEDS:
            result = run_strategy(
                TOD_ZERO_EDGE, source="synthetic", cost_model="book", seed=seed, first=APRIL[0],
                last=APRIL[1], params={"every": every}, suite=suite,
            )  # fmt: skip
            base = _baseline(result)
            assert base["mode"] in B.MODES
            reported.append(_inside(base["net_percentile"]))
            uniform.append(_inside(base["uniform"]["net"]["percentile"]))  # type: ignore[index]
    runs = len(ZERO_EDGE_EVERY) * len(ZERO_EDGE_SEEDS)
    assert sum(reported) >= 0.8 * runs, reported
    assert sum(uniform) < 0.8 * runs, uniform  # the test separates: the uniform mode alone fails it


def test_the_reported_mode_has_the_largest_p_net_and_cost_matched_then_tod_win_a_tie() -> None:
    def modes(p_uniform: float, p_tod: float, p_cm: float) -> dict[str, object]:
        return {"uniform": {"net": {"p": p_uniform}}, "tod": {"net": {"p": p_tod}},
                "cost_matched": {"net": {"p": p_cm}}}  # fmt: skip

    assert B.reported_mode(modes(0.4, 0.1, 0.3)) == "uniform"
    assert B.reported_mode(modes(0.1, 0.4, 0.3)) == "tod"
    assert B.reported_mode(modes(0.2, 0.2, 0.1)) == "tod"  # the two-mode rule's tie, unchanged
    assert B.reported_mode(modes(0.1, 0.2, 0.3)) == "cost_matched"
    assert B.reported_mode(modes(0.3, 0.2, 0.3)) == "cost_matched"
    assert B.reported_mode(modes(0.3, 0.3, 0.3)) == "cost_matched"


def test_the_four_top_level_figures_come_from_the_named_mode() -> None:
    base = _baseline(_run(HASH_RANDOM))
    mode = base["mode"]
    assert mode in B.MODES and "headline" not in base
    part = base[mode]  # type: ignore[index]
    assert isinstance(part, dict)
    assert (base["net_percentile"], base["p_net"]) == (part["net"]["percentile"], part["net"]["p"])
    assert (base["sharpe_percentile"], base["p_sharpe"]) == (
        part["sharpe"]["percentile"],
        part["sharpe"]["p"],
    )
    assert mode == B.reported_mode(base)  # type: ignore[arg-type]


def test_a_long_only_strategy_on_drifting_bars_is_not_above_the_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_frame(monkeypatch, drifting_frame)
    result = _run(HASH_RANDOM, params={"long_only": 1})
    base = _baseline(result)
    assert set(result.trades["side"].to_list()) == {1}
    assert result.summary["net_cents"] > 0  # type: ignore[operator]  # it earns the drift ...
    uniform = base["uniform"]
    assert isinstance(uniform, dict) and uniform["net"]["q50"] > 0  # ... and so does the baseline
    assert base["net_percentile"] < 0.95  # type: ignore[operator]


def test_fewer_than_200_draws_is_refused() -> None:
    market = B.BaselineMarket(
        open=np.zeros(3, np.int64), close=np.zeros(3, np.int64), slip_buy=np.zeros(3, np.int64),
        slip_sell=np.zeros(3, np.int64), ct_minute=np.zeros(3, np.int64), day_idx=np.zeros(3, np.int64),
        entry_ok=np.ones(3, bool), room=np.zeros(3, np.int64), n_days=1, commission_cents=0,
        tick_value_cc=1, contracts=1,
    )  # fmt: skip
    trades = B.BaselineTrades(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64),
                              np.zeros(0, bool), np.zeros(0, np.int64))  # fmt: skip
    with pytest.raises(B.BaselineError, match="200"):
        B.random_baseline(market, trades, np.zeros(1, np.int64), draws=199, seed=1)


# ------------------------------------------------------------------------- mechanics ----
def _toy_market(n: int, eligible: list[int]) -> B.BaselineMarket:
    entry_ok = np.zeros(n, dtype=bool)
    entry_ok[eligible] = True
    return B.BaselineMarket(
        open=np.arange(n, dtype=np.int64), close=np.arange(n, dtype=np.int64) + 1,
        slip_buy=np.zeros(n, np.int64), slip_sell=np.zeros(n, np.int64), ct_minute=np.zeros(n, np.int64),
        day_idx=np.zeros(n, np.int64), entry_ok=entry_ok, room=np.arange(n, dtype=np.int64)[::-1].copy(),
        n_days=1, commission_cents=0, tick_value_cc=100, contracts=1,
    )  # fmt: skip


def test_overlaps_are_redrawn_then_kept_and_counted() -> None:
    # two eligible bars, three trades of hold 0: the third always overlaps after 50 tries
    market = _toy_market(10, [2, 5])
    trades = B.BaselineTrades(np.ones(3, np.int64), np.array([2, 5, 2]), np.zeros(3, np.int64),
                              np.ones(3, bool), np.zeros(3, np.int64))  # fmt: skip
    out = B.run_mode(market, trades, "uniform", 200, 3)
    assert out.overlaps.tolist() == [1] * 200
    # one bar long each: open i, close i + 1 -> 1 tick x 100 cc / 100 = 1 cent per long placement
    assert set(out.nets.tolist()) == {3}


def test_holds_longer_than_a_bars_room_are_never_drawn_there() -> None:
    market = _toy_market(10, [0, 7, 8])  # room 9, 2, 1
    trades = B.BaselineTrades(np.ones(1, np.int64), np.array([0]), np.array([5]), np.ones(1, bool),
                              np.zeros(1, np.int64))  # fmt: skip
    cand, lo, hi = B._candidates(market, trades, "uniform")
    assert cand[lo[0] : hi[0]].tolist() == [0]


def test_the_baseline_is_reproducible_per_seed_and_differs_across_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _capture(monkeypatch)
    result = _run(HASH_RANDOM, suite=SuiteSpec.off())
    market, trades = _market(seen[0]), B.trades_of(result.trades)
    a = B.run_mode(market, trades, "uniform", 200, 11)
    b = B.run_mode(market, trades, "uniform", 200, 11)
    c = B.run_mode(market, trades, "uniform", 200, 12)
    assert np.array_equal(a.nets, b.nets) and not np.array_equal(a.nets, c.nets)


def test_percentile_counts_half_the_ties_and_p_counts_draws_at_or_above() -> None:
    draws = np.array([1.0, 2.0, 2.0, 3.0])
    pos = B._position(2.0, draws)
    assert pos["percentile"] == pytest.approx((1 + 0.5 * 2) / 4)
    assert pos["p"] == pytest.approx((1 + 3) / 5)


def test_sharpe_rows_match_the_summary_function() -> None:
    from pfb.metrics.core import summarize  # noqa: F401  (the convention's home)
    from pfb.metrics.stats import moments

    rng = np.random.default_rng(5)
    daily = rng.integers(-5000, 5000, (3, 250)).astype(np.int64)
    got = B.sharpe_rows(daily)
    want = [moments([float(v) for v in row])["sharpe"] * np.sqrt(252) for row in daily]
    assert got == pytest.approx(want, rel=1e-12)


def test_draws_frame_has_one_row_per_draw_and_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run(HASH_RANDOM)
    frame = result.extras["baseline_draws"]
    assert frame.height == 600
    assert frame.group_by("mode").len().sort("mode").to_dicts() == [
        {"mode": "cost_matched", "len": 200},
        {"mode": "tod", "len": 200},
        {"mode": "uniform", "len": 200},
    ]
    base = _baseline(result)
    uni = frame.filter(pl.col("mode") == "uniform")["net_cents"].to_numpy()
    assert base["uniform"]["net"]["mean"] == pytest.approx(float(uni.mean()))  # type: ignore[index]
