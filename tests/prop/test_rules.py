"""Scalar rule engine (src/pfb/prop/rules.py) on hand-built day paths.

Port of the upstream xfa_rules unit and sequence tests that apply to a day-path engine (floor trailing
and lock, real-time breach on the intrabar low, the Combine pass with consistency, both payout paths, the
floor after the first payout, scaling tiers), plus the A3 extensions (intraday trailing, soft and hard daily
loss, cumulative consistency, buffer, weekday, dead days, caps per payout, payout count limit, time limit).
"""

from __future__ import annotations

import datetime as dt

import pytest
from _synth import FIRMS_DIR, TOPSTEP_CONS, TOPSTEP_STD, closes, make_days, seq_all

from pfb.prop.config import load_plan, with_values
from pfb.prop.rules import iso_weekday, simulate_eval, simulate_funded, tier_minis, trading_day_limit
from pfb.prop.types import (
    END_BREACH,
    END_PASS,
    END_PAYOUT_LIMIT,
    END_SIZE_LIMIT,
    END_TIME_LIMIT,
    END_TIMEOUT,
    RuleUnknown,
)

STD = load_plan(TOPSTEP_STD)
CONS = load_plan(TOPSTEP_CONS)
START = 5_000_000


def run_eval(plan, nets, pad=0):
    days = closes(nets)
    return simulate_eval(plan, days, seq_all(days, pad))


def run_funded(plan, nets, pad=0):
    days = closes(nets)
    return simulate_funded(plan, days, seq_all(days, pad))


# --------------------------------------------------------------------- evaluation: floor and breach ----
class TestEvalFloor:
    def test_floor_ratchets_with_the_close_locks_at_start_and_touch_breaches(self) -> None:
        # closes 51,000 / 51,800 / 52,300 / 52,700: floor 49,000 / 49,800 / 50,000 (lock) / 50,000
        days = make_days(
            [
                [(0, 100_000, 100_000)],
                [(0, 80_000, 80_000)],
                [(0, 50_000, 50_000)],
                [(0, 40_000, 40_000)],
                [(-270_000, 0, -100_000)],
            ]
        )
        out = simulate_eval(STD, days, seq_all(days))
        assert (out.end, out.days, out.balance_cents) == (END_BREACH, 5, START)

    def test_wick_one_cent_above_the_locked_floor_survives(self) -> None:
        days = make_days(
            [
                [(0, 100_000, 100_000)],
                [(0, 80_000, 80_000)],
                [(0, 50_000, 50_000)],
                [(0, 40_000, 40_000)],
                [(-269_999, 0, -100_000)],
            ],
        )
        out = simulate_eval(STD, days, seq_all(days))
        assert out.end == END_TIMEOUT and out.balance_cents == START + 170_000

    def test_floor_never_moves_down_on_a_losing_day(self) -> None:
        # +1,000 (floor 49,000), -1,500 (balance 49,500, floor stays), then a wick to 49,000 breaches
        days = make_days([[(0, 100_000, 100_000)], [(-150_000, 0, -150_000)], [(-50_000, 0, 0)]])
        out = simulate_eval(STD, days, seq_all(days))
        assert (out.end, out.days, out.balance_cents) == (END_BREACH, 3, 4_900_000)

    def test_eod_floor_does_not_move_intraday(self) -> None:
        # a +1,500 high then -1,999 low in one day: the floor is still 48,000
        days = make_days([[(0, 150_000, 0), (-199_999, 0, -100_000)]])
        out = simulate_eval(STD, days, seq_all(days))
        assert out.end == END_TIMEOUT and out.balance_cents == START - 100_000

    def test_intrabar_low_breaches_although_the_close_is_positive(self) -> None:
        days = make_days([[(-200_000, 60_000, 50_000)]])
        out = simulate_eval(STD, days, seq_all(days))
        assert (out.end, out.days, out.balance_cents) == (END_BREACH, 1, 4_800_000)

    def test_eod_breach_check_tests_only_the_close(self) -> None:
        plan = with_values(STD, {"eval.breach_check": "eod"})
        wick = make_days([[(-250_000, 0, 50_000)]])
        assert simulate_eval(plan, wick, seq_all(wick)).end == END_TIMEOUT
        close = make_days([[(-250_000, 0, -200_000)]])
        out = simulate_eval(plan, close, seq_all(close))
        assert (out.end, out.balance_cents) == (END_BREACH, 4_800_000)

    def test_breach_at_equal_false_lets_a_touch_survive(self) -> None:
        plan = with_values(STD, {"eval.breach_at_equal": False})
        assert run_eval(plan, [-200_000]).end == END_TIMEOUT
        assert run_eval(plan, [-200_001]).end == END_BREACH


# ------------------------------------------------------------------------- evaluation: pass ----
class TestEvalPass:
    def test_even_days_pass_at_the_target(self) -> None:
        out = run_eval(STD, [100_000, 100_000, 100_000, 50_000])
        got = (out.end, out.days, out.balance_cents, out.best_day_cents)
        assert got == (END_PASS, 3, START + 300_000, 100_000)

    def test_big_day_raises_the_target_and_does_not_fail(self) -> None:
        # best 2,000: needs profit > 2,000 / 0.55 = 3,636.36
        out = run_eval(STD, [200_000, 100_000, 60_000, 4_000])
        assert (out.end, out.days) == (END_PASS, 4)
        assert run_eval(STD, [200_000, 100_000, 63_700]).end == END_PASS

    def test_exactly_55_percent_fails_strict_and_passes_inclusive(self) -> None:
        nets = [165_000, 135_000]  # profit 3,000, best 1,650 = 55.00%
        assert run_eval(STD, nets).end == END_TIMEOUT
        inclusive = with_values(STD, {"eval.consistency_strict": False})
        assert run_eval(inclusive, nets).end == END_PASS

    def test_frac_of_target_compares_the_best_day_with_the_target(self) -> None:
        plan = with_values(STD, {"eval.consistency_type": "best_day_frac_of_target"})
        assert run_eval(plan, [160_000, 140_000]).end == END_PASS
        assert run_eval(plan, [170_000, 140_000, 100_000]).end == END_TIMEOUT  # never passes again

    def test_min_trading_days_counts_only_days_with_trades(self) -> None:
        plan = with_values(STD, {"eval.consistency_type": "none", "eval.consistency_bps": "none"})
        days = make_days([[(0, 300_000, 300_000)], [], [(0, 1_000, 1_000)]])
        out = simulate_eval(plan, days, seq_all(days))
        assert (out.end, out.days, out.traded_days) == (END_PASS, 3, 2)

    def test_min_winning_days(self) -> None:
        plan = with_values(STD, {"eval.min_winning_days": 3, "eval.winning_day_cents": 20_000})
        out = run_eval(plan, [100_000, 100_000, 19_999, 100_000, 50_000])
        assert (out.end, out.days) == (END_PASS, 4)

    def test_intraday_target_passes_at_the_touching_minute(self) -> None:
        plan = with_values(STD, {"eval.target_check": "intraday"})
        days = make_days([[(0, 150_000, 150_000)], [(0, 150_000, 100_000), (-150_000, 0, -140_000)]])
        out = simulate_eval(plan, days, seq_all(days))
        assert (out.end, out.days, out.balance_cents) == (END_PASS, 2, START + 300_000)
        eod = simulate_eval(STD, days, seq_all(days))
        assert eod.end == END_TIMEOUT

    def test_size_above_the_eval_limit_ends_as_size_limit_not_breach(self) -> None:
        days = make_days([[(0, 10_000, 10_000)], [(0, 10_000, 10_000)]], max_contracts=[5, 6])
        out = simulate_eval(STD, days, seq_all(days))
        got = (out.end, out.days, out.balance_cents, out.traded_days)
        assert got == (END_SIZE_LIMIT, 2, START + 10_000, 2)


class TestEvalEnds:
    def test_padding_ends_as_timeout_at_the_pad(self) -> None:
        out = run_eval(STD, [1_000, 1_000], pad=3)
        assert (out.end, out.days) == (END_TIMEOUT, 2)

    def test_time_limit_in_trading_days(self) -> None:
        assert trading_day_limit(30) == 21
        plan = with_values(STD, {"eval.time_limit_calendar_days": 30})
        out = run_eval(plan, [1_000] * 40)
        assert (out.end, out.days) == (END_TIME_LIMIT, 21)
        assert run_eval(plan, [1_000] * 21).end == END_TIME_LIMIT
        assert run_eval(plan, [1_000] * 20).end == END_TIMEOUT

    def test_soft_daily_loss_flattens_the_day_without_a_breach(self) -> None:
        plan = with_values(STD, {"eval.daily_loss_cents": 100_000, "eval.daily_loss_kind": "soft"})
        days = make_days([[(-100_000, 0, -100_000), (-150_000, 50_000, 40_000)], [(0, 1_000, 1_000)]])
        out = simulate_eval(plan, days, seq_all(days))
        assert (out.end, out.balance_cents) == (END_TIMEOUT, START - 99_000)

    def test_hard_daily_loss_is_a_breach(self) -> None:
        plan = with_values(STD, {"eval.daily_loss_cents": 100_000, "eval.daily_loss_kind": "hard"})
        days = make_days([[(-100_000, 0, -50_000)]])
        out = simulate_eval(plan, days, seq_all(days))
        assert (out.end, out.days, out.balance_cents) == (END_BREACH, 1, START - 100_000)

    def test_mll_breach_comes_before_the_daily_loss_on_the_same_minute(self) -> None:
        plan = with_values(STD, {"eval.daily_loss_cents": 100_000, "eval.daily_loss_kind": "soft"})
        days = make_days([[(-200_000, 0, -200_000)]])
        assert simulate_eval(plan, days, seq_all(days)).end == END_BREACH

    def test_intraday_trailing_applies_the_high_before_the_low(self) -> None:
        plan = with_values(STD, {"eval.drawdown_type": "intraday_trailing"})
        ok = make_days([[(-49_999, 150_000, 0)]])  # peak 51,500, floor 49,500; low 49,500.01
        assert simulate_eval(plan, ok, seq_all(ok)).end == END_TIMEOUT
        bad = make_days([[(-50_000, 150_000, 0)]])
        assert simulate_eval(plan, bad, seq_all(bad)).end == END_BREACH
        assert simulate_eval(STD, bad, seq_all(bad)).end == END_TIMEOUT

    def test_static_floor_never_moves(self) -> None:
        plan = with_values(STD, {"eval.drawdown_type": "static"})
        out = run_eval(plan, [150_000, -340_000])
        assert out.end == END_TIMEOUT and out.balance_cents == START - 190_000


# ----------------------------------------------------------------------------- funded: payouts ----
class TestStandardPayout:
    def test_five_winning_days_pay_half_the_balance(self) -> None:
        out = run_funded(STD, [15_000] * 5, pad=1)
        assert out.payouts_gross_cents == (37_500,)
        assert out.payouts_trader_cents == (33_750,)
        assert (out.first_payout_day, out.balance_cents, out.end, out.days) == (5, 37_500, END_TIMEOUT, 5)

    def test_a_149_99_day_does_not_count_and_days_need_not_be_consecutive(self) -> None:
        out = run_funded(STD, [15_000, 14_999, 15_000, -1_000, 15_000, 15_000, 15_000])
        assert out.first_payout_day == 7

    def test_amount_capped_at_the_dollar_cap(self) -> None:
        out = run_funded(STD, [100_000] * 5)
        assert out.payouts_gross_cents == (200_000,)

    def test_below_the_minimum_request_waits(self) -> None:
        # 5 winners by day 6 with a balance of $200: 50% = $100 < $125; day 7 brings it to $300
        out = run_funded(STD, [15_000] * 4 + [-55_000, 15_000, 10_000])
        assert out.payouts_gross_cents == (15_000,) and out.first_payout_day == 7

    def test_floor_after_the_first_payout_and_loss_of_the_remainder_breaches(self) -> None:
        out = run_funded(STD, [15_000] * 5 + [-37_500])
        assert (out.end, out.days, out.balance_cents) == (END_BREACH, 6, 0)
        out = run_funded(STD, [15_000] * 5 + [-37_499])
        assert out.end == END_TIMEOUT

    def test_floor_after_first_payout_none_keeps_the_trailing_floor(self) -> None:
        plan = with_values(STD, {"funded.floor_after_first_payout_rel_cents": "none"})
        out = run_funded(plan, [15_000] * 5 + [-37_500])
        assert out.end == END_TIMEOUT  # the floor is still -1,250

    def test_request_day_does_not_count_toward_the_next_window(self) -> None:
        out = run_funded(STD, [15_000] * 10 + [15_000])
        assert len(out.payouts_gross_cents) == 2  # day 6 skipped: second payout at day 11
        assert len(run_funded(STD, [15_000] * 10).payouts_gross_cents) == 1
        counts = with_values(STD, {"payout.payout_day_counts": True})
        assert len(run_funded(counts, [15_000] * 10).payouts_gross_cents) == 2

    def test_standard_needs_profit_since_the_last_payout(self) -> None:
        nets = [15_000] * 5 + [0] + [15_000] * 4 + [-75_000, 15_000]
        out = run_funded(STD, nets)
        assert len(out.payouts_gross_cents) == 1  # window net exactly 0 after 5 winners
        out = run_funded(STD, [*nets, 1])
        assert len(out.payouts_gross_cents) == 2

    def test_dead_days_count_toward_nothing(self) -> None:
        plan = with_values(STD, {"payout.dead_days_after_payout": 2})
        assert len(run_funded(plan, [15_000] * 12).payouts_gross_cents) == 1
        assert len(run_funded(plan, [15_000] * 13).payouts_gross_cents) == 2


class TestConsistencyPayout:
    def test_exactly_40_percent_passes(self) -> None:
        out = run_funded(CONS, [40_000, 30_000, 30_000])
        assert out.payouts_gross_cents == (50_000,) and out.first_payout_day == 3

    def test_40_01_percent_fails(self) -> None:
        assert run_funded(CONS, [40_010, 29_995, 29_995]).payouts_gross_cents == ()

    def test_needs_three_days_with_a_trade(self) -> None:
        days = make_days([[(0, 30_000, 30_000)], [], [(0, 30_000, 30_000)], [(0, 40_000, 40_000)]])
        out = simulate_funded(CONS, days, seq_all(days))
        assert out.first_payout_day == 4

    def test_cap_is_3000(self) -> None:
        out = run_funded(CONS, [400_000, 400_000, 400_000])
        assert out.payouts_gross_cents == (300_000,)

    def test_cumulative_scope_never_resets(self) -> None:
        plan = with_values(CONS, {"payout.consistency_scope": "cumulative"})
        nets = [40_000, 30_000, 30_000, 0, 90_000, 10_000, 10_000]
        assert len(run_funded(CONS, nets).payouts_gross_cents) == 1  # window: 90k / 110k > 40%
        assert len(run_funded(plan, nets).payouts_gross_cents) == 1
        assert len(run_funded(plan, [*nets, 60_000, 50_000]).payouts_gross_cents) == 2


class TestFundedExtensions:
    def test_payout_count_limit_ends_the_account(self) -> None:
        plan = with_values(STD, {"funded.payout_count_limit": 1})
        out = run_funded(plan, [15_000] * 8)
        assert (out.end, out.days, out.balance_cents) == (END_PAYOUT_LIMIT, 5, 37_500)

    def test_caps_per_payout_and_the_last_repeats(self) -> None:
        plan = with_values(STD, {"payout.caps_cents": [20_000, 30_000], "payout.payout_day_counts": True})
        out = run_funded(plan, [15_000] * 20)
        assert out.payouts_gross_cents[:3] == (20_000, 30_000, 30_000)

    def test_buffer_and_min_balance(self) -> None:
        plan = with_values(STD, {"payout.buffer_rel_cents": 50_000, "payout.frac_of_profit_bps": 10_000})
        out = run_funded(plan, [15_000] * 5)
        assert out.payouts_gross_cents == (25_000,) and out.balance_cents == 50_000
        plan = with_values(STD, {"payout.min_balance_rel_cents": 80_000})
        assert run_funded(plan, [15_000] * 5).payouts_gross_cents == ()
        assert run_funded(plan, [15_000] * 6).payouts_gross_cents == (45_000,)

    def test_net_minimums_first_and_later(self) -> None:
        plan = with_values(STD, {"payout.net_min_first_cents": 90_000})
        assert run_funded(plan, [15_000] * 5).payouts_gross_cents == ()
        assert run_funded(plan, [15_000] * 6).payouts_gross_cents == (45_000,)

    def test_request_weekday(self) -> None:
        plan = with_values(STD, {"payout.request_weekday": 5})  # Friday
        days = closes([15_000] * 10)  # 2024-01-02 is a Tuesday: day 5 is Monday 01-08, day 9 Friday 01-12
        out = simulate_funded(plan, days, seq_all(days))
        assert out.first_payout_day == 9
        assert iso_weekday(int(days.trade_dates[out.first_payout_day - 1])) == 5

    def test_window_cap_applies_from_its_payout_index(self) -> None:
        # payout 1: profit 5,000 -> min(cap 2,000, 50%) = 2,000 (window 2 x 5,000 does not bind);
        # payout 2: 5 x 150 USD -> profit 3,750, 50% = 1,875, window net 750 x 2 = 1,500
        nets = [100_000] * 5 + [15_000] * 5
        base = {"payout.payout_day_counts": True, "payout.window_cap_bps": 20_000}
        out = run_funded(with_values(STD, {**base, "payout.window_cap_from_payout": 2}), nets)
        assert out.payouts_gross_cents == (200_000, 150_000)
        out = run_funded(with_values(STD, {**base, "payout.window_cap_from_payout": 3}), nets)
        assert out.payouts_gross_cents == (200_000, 187_500)
        out = run_funded(with_values(STD, {**base, "payout.window_cap_from_payout": 1}), nets)
        assert out.payouts_gross_cents == (200_000, 150_000)
        tight = with_values(STD, {**base, "payout.window_cap_bps": 1_000, "payout.window_cap_from_payout": 1})
        assert run_funded(tight, nets[:5]).payouts_gross_cents == (50_000,)  # 10% of the 5,000 window

    def test_caps_none_means_no_fixed_cap(self) -> None:
        plan = with_values(STD, {"payout.caps_cents": "none"})
        assert run_funded(plan, [100_000] * 5).payouts_gross_cents == (250_000,)
        assert run_funded(STD, [100_000] * 5).payouts_gross_cents == (200_000,)

    def test_split_first_full(self) -> None:
        plan = with_values(STD, {"payout.split_first_full_cents": 30_000})
        out = run_funded(plan, [15_000] * 5)
        assert out.payouts_trader_cents == (30_000 + 7_500 * 9_000 // 10_000,)

    def test_win_day_strict(self) -> None:
        plan = with_values(STD, {"payout.win_day_strict": True})
        assert run_funded(plan, [15_000] * 5).payouts_gross_cents == ()

    def test_intraday_trailing_and_soft_daily_loss_in_funded(self) -> None:
        plan = with_values(
            STD,
            {
                "funded.drawdown_type": "intraday_trailing",
                "funded.daily_loss_cents": 50_000,
                "funded.daily_loss_kind": "soft",
            },
        )
        days = make_days([[(-100_000, 150_000, 0)]])  # peak 1,500, floor -500: low -1,000 breaches
        assert simulate_funded(plan, days, seq_all(days)).end == END_BREACH
        days = make_days([[(-60_000, 0, -60_000)], [(0, 1_000, 1_000)]])
        out = simulate_funded(plan, days, seq_all(days))
        assert (out.end, out.balance_cents) == (END_TIMEOUT, -49_000)


class TestScaling:
    @pytest.mark.parametrize(
        ("rel", "lower", "upper"),
        [
            (-100_000, 2, 2),
            (149_999, 2, 2),
            (150_000, 2, 3),
            (150_001, 3, 3),
            (200_000, 3, 5),
            (200_001, 5, 5),
        ],
    )
    def test_tiers_and_boundary_readings(self, rel: int, lower: int, upper: int) -> None:
        tiers = STD.value("funded.scaling")
        assert tier_minis(tiers, rel, True) == lower
        assert tier_minis(tiers, rel, False) == upper

    def test_tier_follows_the_prior_close_both_directions(self) -> None:
        nets = [100_000, 60_000, 0, -30_000, 0]  # balances 1,000 / 1,600 / 1,600 / 1,300 / 1,300
        days = make_days([[(min(0, c), max(0, c), c)] for c in nets], max_contracts=[2, 2, 3, 3, 3])
        out = simulate_funded(STD, days, seq_all(days))
        assert (out.end, out.days, out.balance_cents) == (END_SIZE_LIMIT, 5, 130_000)  # 3 minis on days 3-4
        days = make_days([[(min(0, c), max(0, c), c)] for c in nets], max_contracts=[2, 3, 3, 2, 2])
        out = simulate_funded(STD, days, seq_all(days))
        assert (out.end, out.days) == (END_SIZE_LIMIT, 2)

    def test_tier_uses_the_close_before_a_same_close_payout(self) -> None:
        # Upstream funnel.run_xfa: the tier reads the prior close, before the payout taken at that close.
        # 5 x 320 USD: close 1,600 (tier 3), payout 800 at that close, open 800 (tier 2 if read after it)
        days = make_days(
            [[(0, 32_000, 32_000)]] * 5 + [[(0, 0, 0)], [(0, 0, 0)]], max_contracts=[2, 2, 2, 2, 2, 3, 3]
        )
        out = simulate_funded(STD, days, seq_all(days))
        assert out.payouts_gross_cents == (80_000,)
        assert (out.end, out.days) == (END_SIZE_LIMIT, 7)  # day 6 allowed (close 1,600), day 7 not (800)


class TestUnknownAndHelpers:
    def test_unknown_eval_value_raises_only_when_used(self, tmp_path) -> None:
        text = TOPSTEP_STD.read_text().replace(
            'consistency_bps = { v = 5500, src = "R075" }', 'consistency_bps = { unknown = "test reason" }'
        )
        path = tmp_path / TOPSTEP_STD.name
        path.write_text(text)
        plan = load_plan(path)
        days = closes([1_000])
        with pytest.raises(RuleUnknown, match="eval.consistency_bps: test reason"):
            simulate_eval(plan, days, seq_all(days))
        assert simulate_funded(plan, days, seq_all(days)).end == END_TIMEOUT

    def test_iso_weekday_matches_datetime(self) -> None:
        day = dt.date(1999, 12, 25)
        for _ in range(3000):
            assert iso_weekday(day.year * 10000 + day.month * 100 + day.day) == day.isoweekday()
            day += dt.timedelta(days=3)

    def test_outcomes_are_frozen(self) -> None:
        out = run_eval(STD, [1_000])
        with pytest.raises(AttributeError):
            out.end = 0  # type: ignore[misc]
        funded = run_funded(STD, [15_000] * 5)
        with pytest.raises(AttributeError):
            funded.balance_cents = 0  # type: ignore[misc]
        assert funded.payouts_gross_cents == (37_500,)


class TestFirmConfigs:
    def test_growth_buffer_keeps_a_payout_strictly_above_the_floor(self) -> None:
        # caps [3,000]; days 1,000 / 1,000 / 600 / 200 / 250 USD then an untraded day. With buffer
        # 10000 the payout took the balance to the floor (start + 100) and the untraded day breached.
        growth = load_plan(FIRMS_DIR / "tradeify" / "growth_50k.toml")
        plan = with_values(growth, {"payout.caps_cents": [300_000]})
        days = make_days([[(0, c, c)] for c in (100_000, 100_000, 60_000, 20_000, 25_000)] + [[]])
        out = simulate_funded(plan, days, seq_all(days))
        assert out.payouts_gross_cents == (294_999,)
        assert (out.end, out.days, out.balance_cents) == (END_TIMEOUT, 6, START + 10_001)

    def test_bulenox_payout_excludes_three_days_after_a_request(self) -> None:
        # a Friday request leaves Mon (payout_day_counts false), Tue and Wed (2 dead days) uncounted:
        # 3 days, not 4. The weekday gate is lifted here so the count is not hidden by the Friday alignment.
        plan = with_values(
            load_plan(FIRMS_DIR / "bulenox" / "qualification_option1_50k.toml"),
            {
                "payout.consistency_bps": "none",
                "payout.request_weekday": "none",
                "payout.buffer_rel_cents": 0,
            },
        )
        days = closes([20_000] * 30)
        seq = seq_all(days)
        out = simulate_funded(plan, days, seq)
        assert out.first_payout_day == 10  # 10 traded days
        assert len(simulate_funded(plan, days, seq[:22]).payouts_gross_cents) == 1
        assert len(simulate_funded(plan, days, seq[:23]).payouts_gross_cents) == 2  # days 11-13 out, 14-23 in
