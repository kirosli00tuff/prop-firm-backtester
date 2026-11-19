"""Known-answer tests of the Monte Carlo on the real rule engine (docs/PROP_MC.md).

(a) Coin-flip days +x / -x, static drawdown D, target T, nothing else binding: gambler's ruin gives
    P(pass) = D / (D + T) for D, T multiples of x (breach_at_equal true puts the barrier exactly at -D).
(b) An always-winning day passes in ceil(T / x) days with P 1; an always-losing day never passes.
(c) A scripted day peaks at +1,500 USD then troughs at -1,000 USD and recovers to 0: with a 2,000 USD
    intraday_trailing drawdown the floor rises to -500 and the trough breaches; the same plan with
    eod_trailing keeps the floor at -2,000 (set at the previous close) and never breaches.
The plans are the TOML fixtures in tests/prop/fixtures/. Seeds are fixed here and recorded in every result.
(d) Topstep 50K standard, every day +600 USD, H_f 60: closed-form funded_ev and ev_per_purchase (below), on
    the block bootstrap and on random_start with a history long enough for full windows; a short history
    gives padded paths and lower bounds.
(e) Fee readings on the real configs: an unknown reset nulls only cost_to_funded_with_resets
    (MFFU Rapid); a retry after a time-limit end is a new purchase, so with only time-limit failures
    the two cost figures agree (Bulenox option 1).
Each test runs on the Numba batch kernel (the default engine) and on the scalar reference rules.py.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from pfb.prop.config import load_plan
from pfb.prop.days import from_day_nets, from_minutes
from pfb.prop.mc import Engine, default_engine, run_mc, scalar_engine
from pfb.prop.types import DayPaths, RuleUnknown

FIXTURES = Path(__file__).parent / "fixtures"
FIRMS = Path(__file__).resolve().parents[2] / "config" / "firms"
STATIC = FIXTURES / "ka_static_50k.toml"  # D 150,000 cents, T 100,000 cents
INTRADAY = FIXTURES / "ka_intraday_trailing_50k.toml"  # D 200,000 cents, T 300,000 cents
EOD = FIXTURES / "ka_eod_trailing_50k.toml"
X = 50_000
SEED_A, SEED_B, SEED_C = 20261116, 20261117, 20261118
N_A = 20_000


@pytest.fixture(params=["kernel", "scalar"])
def engine(request: pytest.FixtureRequest) -> Engine:
    return default_engine() if request.param == "kernel" else scalar_engine()


def test_coin_flip_pass_probability(engine: Engine) -> None:
    plan = load_plan(STATIC)
    d, t = plan.value("eval.drawdown_cents"), plan.value("eval.profit_target_cents")
    assert d % X == 0 and t % X == 0
    res = run_mc(plan, from_day_nets([X, -X]), n_paths=N_A, seed=SEED_A, generator="block", mean_block=1,
                 horizon_eval=252, horizon_funded=1, kind="synthetic", engine=engine)  # fmt: skip
    closed_form = d / (d + t)
    assert res.p_pass.se is not None and res.p_pass.value is not None
    assert abs(res.p_pass.value - closed_form) < 4 * res.p_pass.se, (res.p_pass, closed_form)
    assert res.eval_ends.get("timeout", 0) == 0  # 252 days is long enough (mean duration D*T/x^2 = 6 days)
    assert res.seed == SEED_A and res.plan_sha256 == plan.sha256


@pytest.mark.parametrize("x", [30_000, 70_000])
def test_always_winning_passes_in_ceil_target_over_x(x: int, engine: Engine) -> None:
    plan = load_plan(STATIC)
    t = plan.value("eval.profit_target_cents")
    res = run_mc(plan, from_day_nets([x]), n_paths=500, seed=SEED_B, horizon_funded=1, kind="synthetic",
                 engine=engine)  # fmt: skip
    assert res.p_pass.value == 1.0 and res.p_pass.se == 0.0
    assert res.days_to_pass_median.value == math.ceil(t / x)
    assert res.days_to_pass_p90.value == math.ceil(t / x)


def test_always_losing_never_passes(engine: Engine) -> None:
    plan = load_plan(STATIC)
    res = run_mc(plan, from_day_nets([-30_000]), n_paths=500, seed=SEED_B, horizon_funded=1, kind="synthetic",
                 engine=engine)  # fmt: skip
    assert res.p_pass.value == 0.0
    assert res.eval_ends == {"breach": 500}
    assert res.days_to_pass_median.value is None and res.cost_to_funded.value is None


def _peak_and_trough() -> DayPaths:
    return from_minutes([([0, -100_000, -100_000], [150_000, 150_000, 0], [150_000, -100_000, 0])])


def test_peak_and_trough_trailing_type(engine: Engine) -> None:
    days = _peak_and_trough()
    intraday = run_mc(load_plan(INTRADAY), days, n_paths=200, seed=SEED_C, horizon_eval=30, horizon_funded=1,
                      kind="synthetic", engine=engine)  # fmt: skip
    eod = run_mc(load_plan(EOD), days, n_paths=200, seed=SEED_C, horizon_eval=30, horizon_funded=1,
                 kind="synthetic", engine=engine)  # fmt: skip
    assert intraday.eval_ends == {"breach": 200}
    assert intraday.p_pass.value == 0.0
    assert eod.eval_ends == {"timeout": 200}  # never breaches, never reaches the target


# (d) Topstep 50K standard (config/firms/topstep/combine_xfa_standard_50k.toml), every day +600 USD, H_f 60.
# Eval: target 3,000 USD in 5 days (8 calendar days at 21 : 30); fee 49 USD (one started month) + platform
# 14.50 x 8 // 30 = 3.86 USD pro rata, so 5,286 cents; activation 149 USD.
# Funded: 60 days, every one a payout close: 10 payouts, 1,500 + 9 x 2,000 = 19,500 USD gross, 90% split =
# 17,550 USD to the trader; platform 14.50 x 86 // 30 = 41.56 USD (60 trading = 86 calendar days).
# funded_ev = 1,755,000 - 4,156 = 1,750,844 cents; ev_per_purchase = 1,750,844 - 5,286 - 14,900 = 1,730,658.
STD_FUNDED_EV = 1_755_000 - 1_450 * 86 // 30
STD_ATTEMPT_FEE = 4_900 + 1_450 * 8 // 30
STD_EV_PER_PURCHASE = STD_FUNDED_EV - STD_ATTEMPT_FEE - 14_900


def test_topstep_always_winning_closed_form(engine: Engine) -> None:
    plan = load_plan(FIRMS / "topstep" / "combine_xfa_standard_50k.toml")
    res = run_mc(plan, from_day_nets([60_000]), n_paths=200, seed=SEED_C, horizon_funded=60, kind="synthetic",
                 engine=engine)  # fmt: skip
    assert (STD_FUNDED_EV, STD_EV_PER_PURCHASE) == (1_750_844, 1_730_658)
    assert res.p_pass.value == 1.0 and res.days_to_pass_median.value == 5
    assert res.cost_per_attempt.value == STD_ATTEMPT_FEE
    assert res.funded_ev.value == STD_FUNDED_EV and res.funded_ev.se == 0.0
    assert res.ev_per_purchase.value == STD_EV_PER_PURCHASE
    assert res.funded_ends == {"timeout": 200} and res.p_breach_before_first_payout.value == 0.0


@pytest.mark.parametrize("n_days", [40, 120, 400])
def test_random_start_short_history_is_flagged(n_days: int) -> None:
    plan = load_plan(FIRMS / "topstep" / "combine_xfa_standard_50k.toml")
    res = run_mc(plan, from_day_nets([60_000] * n_days), n_paths=2000, seed=3, generator="random_start",
                 horizon_funded=60, kind="synthetic")  # fmt: skip
    if n_days >= res.horizon_eval + res.horizon_funded:  # 252 + 60: full windows, the exact closed form
        assert res.counts["eval_padded"] == res.counts["funded_padded"] == 0
        assert res.p_pass.value == 1.0 and res.funded_ev.value == STD_FUNDED_EV
        assert res.ev_per_purchase.value == STD_EV_PER_PURCHASE
        assert not (res.p_pass.lower_bound or res.funded_ev.lower_bound or res.ev_per_purchase.lower_bound)
        return
    assert res.counts["eval_padded"] > 0 and res.counts["funded_padded"] > 0
    assert res.p_pass.value is not None and res.p_pass.value < 1.0
    assert res.funded_ev.value is not None and res.funded_ev.value < STD_FUNDED_EV
    for name in ("p_pass", "funded_ev", "p_breach_before_first_payout", "ev_per_purchase"):
        assert getattr(res, name).lower_bound, name


def test_unknown_reset_nulls_only_cost_with_resets() -> None:
    plan = load_plan(FIRMS / "mffu" / "rapid_50k.toml")
    with pytest.raises(RuleUnknown):
        plan.value("fees.reset_cents")
    res = run_mc(plan, from_day_nets([60_000]), n_paths=50, seed=1, horizon_funded=60, kind="synthetic")
    assert res.cost_to_funded_with_resets.value is None
    for name in ("p_pass", "cost_per_attempt", "cost_to_funded", "funded_ev", "p_breach_before_first_payout",
                 "ev_per_purchase"):  # fmt: skip
        assert getattr(res, name).value is not None, name
    (warning,) = [w for w in res.warnings if w.startswith("rule unknown")]
    assert "fees.reset_cents" in warning and warning.endswith("not computed: cost_to_funded_with_resets")


def test_time_limit_retries_are_new_purchases(engine: Engine) -> None:
    # +150 / 0 USD days, every failure is an expired 30-day window
    plan = load_plan(FIRMS / "bulenox" / "qualification_option1_50k.toml")
    n = 4000 if engine.eval_many.__module__.endswith("kernel") else 600
    res = run_mc(plan, from_day_nets([15_000, 0, 15_000, 15_000]), n_paths=n, seed=2, mean_block=1,
                 horizon_funded=5, kind="synthetic", engine=engine)  # fmt: skip
    assert set(res.eval_ends) == {"pass", "time_limit"} and res.eval_ends["pass"] > 0
    assert res.cost_to_funded.value is not None and res.cost_to_funded_with_resets.value is not None
    assert res.cost_to_funded_with_resets.value == pytest.approx(res.cost_to_funded.value, rel=1e-12)
    assert any("full 30-calendar-day access window" in w for w in res.warnings)
