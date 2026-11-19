"""Open items in the prop engine: each test failed before its fix.

1. A soft daily loss can be booked at the minute's low (sensitivity switch, default "limit" as upstream).
2. max_contracts takes trade quantities, so a round trip inside one minute shows its size.
3. payout.window_calendar_days, calendar days from the trade dates (MFFU Pro's funded phase runs).
4. Skipped and dead days' P&L in the window net, the weekday gate following the sequence's own
   calendar, and a sensitivity plan carries its own hash, not the file's).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from _synth import FIRMS_DIR, TOPSTEP_STD, closes, make_days, random_days, random_seqs, seq_all

from pfb.prop.config import PlanConfigError, PlanRules, load_plan, with_values
from pfb.prop.days import day_paths_from_run, from_day_nets
from pfb.prop.kernel import simulate_eval_many, simulate_funded_many
from pfb.prop.mc import run_mc
from pfb.prop.rules import SOFT_DLL_LIMIT, SOFT_DLL_LOW, simulate_eval, simulate_funded
from pfb.prop.types import END_TIMEOUT, DayPaths

GROWTH = FIRMS_DIR / "tradeify" / "growth_50k.toml"
SELECT_FLEX = FIRMS_DIR / "tradeify" / "select_flex_50k.toml"
MFFU_PRO = FIRMS_DIR / "mffu" / "pro_50k.toml"
NS = 60_000_000_000


# ------------------------------------------------------------------ 1. N1 soft daily loss ----
def _soft_plan() -> PlanRules:
    return with_values(
        load_plan(TOPSTEP_STD), {"eval.daily_loss_cents": 125_000, "eval.daily_loss_kind": "soft"}
    )


def _deep_minute_day() -> DayPaths:
    # A minute from -900 to -1,500 USD closing -1,400 under a 1,250 USD soft DLL.
    return make_days([[(-90_000, 0, -90_000), (-150_000, -90_000, -140_000), (-140_000, 50_000, 40_000)]])


def test_soft_daily_loss_default_books_exactly_minus_dll() -> None:
    plan, days = _soft_plan(), _deep_minute_day()
    out = simulate_eval(plan, days, seq_all(days))
    assert (out.end, out.balance_cents) == (END_TIMEOUT, 5_000_000 - 125_000)


def test_soft_daily_loss_at_minute_low_books_the_low_in_both_engines() -> None:
    plan, days = _soft_plan(), _deep_minute_day()
    seq = seq_all(days)
    out = simulate_eval(plan, days, seq, soft_daily_loss=SOFT_DLL_LOW)
    assert (out.end, out.balance_cents, out.best_day_cents) == (END_TIMEOUT, 5_000_000 - 150_000, 0)
    batch = simulate_eval_many(plan, days, seq[None, :], soft_daily_loss=SOFT_DLL_LOW)
    assert int(batch.balance_cents[0]) == out.balance_cents


def test_soft_daily_loss_switch_refuses_an_unknown_mode() -> None:
    plan, days = _soft_plan(), _deep_minute_day()
    with pytest.raises(ValueError, match="soft_daily_loss"):
        simulate_eval(plan, days, seq_all(days), soft_daily_loss="close")


@pytest.mark.parametrize("path", [GROWTH, FIRMS_DIR / "lucid" / "lucidflex_50k.toml"], ids=lambda p: p.stem)
def test_soft_daily_loss_at_minute_low_kernel_equals_scalar(path: Path) -> None:
    plan = load_plan(path)
    days = random_days(np.random.default_rng(5), 200, scale=60_000, drift=5_000)
    seqs = random_seqs(np.random.default_rng(6), days.n_days, 600, 80)
    ev = simulate_eval_many(plan, days, seqs, soft_daily_loss=SOFT_DLL_LOW)
    fb = simulate_funded_many(plan, days, seqs, soft_daily_loss=SOFT_DLL_LOW)
    differs = 0
    for i in range(seqs.shape[0]):
        e = simulate_eval(plan, days, seqs[i], soft_daily_loss=SOFT_DLL_LOW)
        assert (ev.end[i], ev.days[i], ev.balance_cents[i], ev.best_day_cents[i]) == (
            e.end, e.days, e.balance_cents, e.best_day_cents), i  # fmt: skip
        f = simulate_funded(plan, days, seqs[i], soft_daily_loss=SOFT_DLL_LOW)
        assert (fb.end[i], fb.days[i], fb.balance_cents[i], fb.gross_total_cents[i]) == (
            f.end, f.days, f.balance_cents, sum(f.payouts_gross_cents)), i  # fmt: skip
        differs += e != simulate_eval(plan, days, seqs[i])
    assert differs > 0  # the switch changes some attempt on this pool


def test_mc_output_names_the_soft_daily_loss_mode() -> None:
    plan = load_plan(GROWTH)
    days = from_day_nets([60_000, -20_000])
    default = run_mc(plan, days, n_paths=20, seed=1, horizon_funded=5, kind="synthetic")
    low = run_mc(
        plan, days, n_paths=20, seed=1, horizon_funded=5, kind="synthetic", soft_daily_loss=SOFT_DLL_LOW
    )
    assert (default.soft_daily_loss, low.soft_daily_loss) == (SOFT_DLL_LIMIT, SOFT_DLL_LOW)
    assert '"soft_daily_loss": "minute_low"' in low.to_json()
    assert any("soft daily loss" in w for w in default.warnings)


# ------------------------------------------------------------------ 2. N9 max contracts ----
def _write_run(tmp: Path, *, equity: dict, daily: dict, trades: dict) -> Path:
    run = tmp / "run-n9"
    run.mkdir()
    for name, frame in (("equity_minute", equity), ("daily", daily), ("trades", trades)):
        pl.DataFrame(frame, schema_overrides={"trade_date": pl.Date}).write_parquet(run / f"{name}.parquet")
    return run


def test_round_trip_inside_one_minute_shows_its_size(tmp_path: Path) -> None:
    import datetime as dt

    d1, d2 = dt.date(2024, 3, 4), dt.date(2024, 3, 5)
    ts = [i * NS for i in range(4)]
    run = _write_run(
        tmp_path,
        equity={"ts_event": ts, "trade_date": [d1, d1, d2, d2], "position": [0, 0, 0, 0],
                "eq_close": [0, 50, 50, 20], "eq_low": [0, -10, 50, 0], "eq_high": [0, 60, 50, 40]},
        daily={"trade_date": [d1, d2], "eq_close": [50, 20]},
        trades={"contracts": [3, 7], "entry_ts": [ts[1], ts[3]], "exit_ts": [ts[1], ts[3]],
                "entry_bar": [1, 3], "exit_bar": [1, 3], "trade_date": [d1, d2]},
    )  # fmt: skip
    assert day_paths_from_run(run).max_contracts.tolist() == [3, 7]
    assert day_paths_from_run(run, contracts_per_mini=10).max_contracts.tolist() == [1, 1]


# ------------------------------------------------------------------ 3. calendar-day window ----
def _windowed(**extra: object) -> PlanRules:
    values = {"payout.window_calendar_days": 14, **extra}
    return with_values(load_plan(TOPSTEP_STD), values)


def _both(plan: PlanRules, days: DayPaths, seq: np.ndarray) -> tuple[tuple[int, ...], int]:
    ref = simulate_funded(plan, days, seq)
    batch = simulate_funded_many(plan, days, seq[None, :])
    n = int(batch.n_payouts[0])
    assert tuple(int(x) for x in batch.payouts_gross_cents[0, :n]) == ref.payouts_gross_cents
    assert int(batch.first_payout_day[0]) == ref.first_payout_day
    return ref.payouts_gross_cents, ref.first_payout_day


def test_window_calendar_days_is_a_schema_field() -> None:
    plan = _windowed()
    assert plan.value("payout.window_calendar_days") == 14
    with pytest.raises(PlanConfigError, match="window_calendar_days"):
        with_values(load_plan(TOPSTEP_STD), {"payout.window_calendar_days": 0})


def test_first_payout_waits_14_calendar_days_from_the_first_trade() -> None:
    # Weekdays from Tue 2024-01-02; 300 USD a day. Without the window: day 5 (Mon Jan 8), 50% of 1,500.
    days = closes([30_000] * 21)
    assert _both(load_plan(TOPSTEP_STD), days, seq_all(days)[:5])[0] == (75_000,)
    # With it: Tue Jan 16 = day 11 (profit 3,300, 50% = 1,650), then 14 days after that payout: Tue Jan 30 =
    # day 21.
    assert _both(_windowed(), days, seq_all(days)[:20]) == ((165_000,), 11)
    assert _both(_windowed(), days, seq_all(days)) == ((165_000, 200_000), 11)


def test_window_starts_at_the_first_traded_day() -> None:
    # Day 1 (Tue Jan 2) untraded: the clock starts Wed Jan 3, so the first payout is Wed Jan 17 = day 12.
    days = make_days([[]] + [[(0, 30_000, 30_000)]] * 12)
    assert _both(_windowed(), days, seq_all(days))[1] == 12


def test_later_windows_run_from_the_last_payout_not_from_the_first_trade() -> None:
    # 8 non-winning days (100 USD), then 300 USD days: 5 winning days by day 13 (Thu Jan 18) -> first payout.
    # The next one needs 14 calendar days from Jan 18: Thu Feb 1 = day 23 (Jan 30, day 21, if anchored at the
    # first trade on Jan 2).
    days = closes([10_000] * 8 + [30_000] * 15)
    assert _both(_windowed(), days, seq_all(days)[:22]) == ((115_000,), 13)
    assert _both(_windowed(), days, seq_all(days)) == ((115_000, 200_000), 13)


def test_mffu_pro_funded_phase_runs() -> None:
    plan = load_plan(MFFU_PRO)
    assert plan.value("payout.window_calendar_days") == 14
    # 1,000 USD a day from Tue Jan 2: buffer 2,100 + 1,000 minimum is met on day 4, the window on Tue Jan 16
    # (day 11): profit 11,000 - buffer 2,100 = 8,900, 80% to the trader.
    days = closes([100_000] * 11)
    ref = simulate_funded(plan, days, seq_all(days))
    assert (ref.payouts_gross_cents, ref.payouts_trader_cents, ref.first_payout_day) == (
        (890_000,),
        (712_000,),
        11,
    )


# ------------------------------------------------------------------ 4. N2, N3, N10 ----
def test_skipped_day_loss_enters_the_window_net() -> None:
    # On Tradeify Select Flex: nets 5 x 600, -1,000, 5 x 150 USD. The first payout (1,500) is
    # on day 5; day 6 is the processing day. The true net since that payout on day 11 is -250: no second
    # payout.
    plan = load_plan(SELECT_FLEX)
    days = closes([60_000] * 5 + [-100_000] + [15_000] * 5)
    assert _both(plan, days, seq_all(days)) == ((150_000,), 5)


def test_weekday_gate_follows_the_sequence_calendar() -> None:
    # Day index 4 is Mon 2024-01-08 (3 calendar days after the Friday before). Drawn three times in a row, the
    # sequence's own clock reads Mon Jan 8, Thu Jan 11, Sun Jan 14: a Thursday gate opens on the 2nd day.
    plan = with_values(load_plan(TOPSTEP_STD), {"payout.request_weekday": 4, "payout.win_days_required": 1})
    days = closes([30_000] * 6)
    assert _both(plan, days, np.array([4, 4, 4], dtype=np.int64)) == ((30_000,), 2)
    # Historical order keeps the real weekdays: Thu Jan 4 is day 3.
    assert _both(plan, days, seq_all(days))[1] == 3


def test_sensitivity_plan_carries_its_own_hash() -> None:
    base = load_plan(TOPSTEP_STD)
    alt = with_values(base, {"payout.split_bps": 8000})
    assert alt.sha256 != base.sha256 and len(alt.sha256) == 64
    assert alt.file_sha256 == base.sha256 == base.file_sha256
    assert with_values(base, {"payout.split_bps": 8000}).sha256 == alt.sha256  # deterministic
    assert with_values(alt, {"payout.split_bps": 9000}).sha256 not in (alt.sha256, base.sha256)
    res = run_mc(alt, from_day_nets([60_000]), n_paths=10, seed=1, horizon_funded=5, kind="synthetic")
    assert res.plan_sha256 == alt.sha256 and res.plan_file_sha256 == base.sha256
    assert res.plan_overrides == (("payout.split_bps", 8000),)


def test_day_paths_refuse_dates_that_do_not_increase() -> None:
    from pfb.prop.days import DayPathsError, check_day_paths

    days = closes([1, 2, 3])
    dates = days.trade_dates.copy()
    dates[2] = dates[1]
    with pytest.raises(DayPathsError, match="strictly increasing"):
        check_day_paths(make_days([[(0, 1, 1)]] * 3, dates=dates))
