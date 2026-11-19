"""Estimators and standard errors of run_mc on a fake engine with known outcomes (src/pfb/prop/mc.py).

The fake engine decides each attempt from the first day index of its sequence: pass iff day < K, so with
uniform start days p_pass = K / n_days exactly; fees and payouts are simple functions of the day indices.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pfb.prop.days import from_day_nets
from pfb.prop.mc import (
    STANDING_WARNING,
    Engine,
    calendar_days,
    cost_to_funded,
    cost_with_resets,
    delta_se,
    eval_horizon,
    run_mc,
)
from pfb.prop.types import (
    END_BREACH,
    END_PASS,
    END_SIZE_LIMIT,
    END_TIME_LIMIT,
    END_TIMEOUT,
    RuleUnknown,
)

N_DAYS, K_PASS = 10, 4
PRICE, RESET, ACTIVATION, FUNDED_MONTHLY = 10_000, 3_000, 15_000, 500


@dataclass(frozen=True)
class FakePlan:
    sha256: str = "f" * 64
    eval: dict = field(default_factory=lambda: {"time_limit_calendar_days": None})
    fees: dict = field(default_factory=lambda: {"activation_cents": ACTIVATION})


@dataclass
class Recorder:
    eval_seqs: list[np.ndarray] = field(default_factory=list)
    funded_seqs: list[np.ndarray] = field(default_factory=list)
    funded_unknown: bool = False
    fee_unknown: str = ""  # "reset" | "first" | "funded": that fee raises RuleUnknown
    time_limits: bool = False  # first day 4 or 5: the attempt ends END_TIME_LIMIT
    size_limits: bool = False  # first day 9: the attempt ends END_SIZE_LIMIT; funded first day 1: too


def fake_engine(rec: Recorder) -> Engine:
    def eval_many(plan: Any, days: Any, seqs: np.ndarray, **_: Any) -> SimpleNamespace:
        rec.eval_seqs.append(seqs.copy())
        first = seqs[:, 0]
        passed = first < K_PASS
        used = np.where(passed, first % 30 + 1, 25 + 3 * first)
        end = np.where(passed, END_PASS, END_BREACH)
        if rec.time_limits:
            end = np.where((first == 4) | (first == 5), END_TIME_LIMIT, end)
        if rec.size_limits:
            end = np.where(first == 9, END_SIZE_LIMIT, end)
        return SimpleNamespace(end=end, days=used)

    def funded_many(plan: Any, days: Any, seqs: np.ndarray, **_: Any) -> SimpleNamespace:
        if rec.funded_unknown:
            raise RuleUnknown("payout.consistency_bps: page does not say")
        rec.funded_seqs.append(seqs.copy())
        first = seqs[:, 0]
        pay = np.stack([first * 100, np.where(first % 3 == 0, 50, 0)], axis=1)
        end = np.where(first % 2 == 0, END_BREACH, END_TIMEOUT)
        if rec.size_limits:
            end = np.where(first == 1, END_SIZE_LIMIT, end)
        return SimpleNamespace(
            end=end, days=5 + first,
            payouts_trader_cents=pay, first_payout_day=np.where(first % 4 == 0, -1, 1),
        )  # fmt: skip

    def attempt_fee(plan: Any, d: int, cal: int, *, first: bool) -> int:
        if rec.fee_unknown == ("first" if first else "reset"):
            raise RuleUnknown(f"fees.{'eval_price' if first else 'reset'}_cents: page does not say")
        periods = math.ceil(cal / 30)
        return PRICE * periods if first else RESET + PRICE * (periods - 1)

    def funded_fee(plan: Any, d: int, cal: int) -> int:
        if rec.fee_unknown == "funded":
            raise RuleUnknown("fees.funded_monthly_cents: page does not say")
        return FUNDED_MONTHLY * math.ceil(cal / 30)

    return Engine(eval_many=eval_many, funded_many=funded_many, attempt_fee=attempt_fee,
                  funded_fee=funded_fee)  # fmt: skip


DAYS = from_day_nets(list(range(N_DAYS)))


def _run(n: int = 4000, seed: int = 7, rec: Recorder | None = None, **kw: Any) -> tuple[Any, Recorder]:
    rec = rec or Recorder()
    kw.setdefault("kind", "synthetic")
    kw.setdefault("plan", FakePlan())
    res = run_mc(days=DAYS, n_paths=n, seed=seed, engine=fake_engine(rec), **kw)
    return res, rec


def _per_path(rec: Recorder) -> dict[str, np.ndarray]:
    first = np.concatenate([s[:, 0] for s in rec.eval_seqs])
    passed = (first < K_PASS).astype(np.int64)
    used = np.where(passed == 1, first % 30 + 1, 25 + 3 * first)
    periods = (calendar_days(used) + 29) // 30
    fee, retry = PRICE * periods, RESET + PRICE * (periods - 1)
    return {"first": first, "passed": passed, "used": used, "fee": fee, "retry": retry}


def test_p_pass_and_binomial_se() -> None:
    res, rec = _run()
    pp = _per_path(rec)
    p = pp["passed"].mean()
    assert res.p_pass.value == pytest.approx(p)
    assert res.p_pass.se == pytest.approx(math.sqrt(p * (1 - p) / 4000))
    assert abs(p - K_PASS / N_DAYS) < 4 * res.p_pass.se
    assert res.eval_ends == {"pass": int(pp["passed"].sum()), "breach": 4000 - int(pp["passed"].sum())}


def test_cost_estimators_match_hand_formulas() -> None:
    res, rec = _run()
    pp = _per_path(rec)
    fee, passed = pp["fee"].astype(float), pp["passed"]
    p, c = passed.mean(), fee.mean()
    assert res.cost_per_attempt.value == pytest.approx(c)
    assert res.cost_per_attempt.se == pytest.approx(fee.std(ddof=1) / math.sqrt(fee.size))
    assert res.cost_to_funded.value == pytest.approx(c / p + ACTIVATION)
    # delta method for C/p with the covariance of fee and pass on the same paths
    n = fee.size
    var_c, var_p = fee.var() / n, p * (1 - p) / n
    cov = np.mean((fee - c) * (passed - p)) / n
    se = math.sqrt(var_c / p**2 + c**2 * var_p / p**4 - 2 * c * cov / p**3)
    assert res.cost_to_funded.se == pytest.approx(se, rel=1e-9)
    r = pp["retry"].mean()
    assert res.cost_to_funded_with_resets.value == pytest.approx(c + (1 / p - 1) * r + ACTIVATION)


def test_one_time_resets_formula_and_se() -> None:
    passed = np.array([1, 0, 0, 1, 0] * 200)
    fee = np.full(passed.size, PRICE)
    p = passed.mean()
    no_tl = np.zeros(passed.size, dtype=np.int64)
    with_reset = cost_with_resets(fee, np.full(passed.size, RESET), passed, no_tl, ACTIVATION)
    assert with_reset.value == pytest.approx(PRICE + (1 / p - 1) * RESET + ACTIVATION)
    assert with_reset.se == pytest.approx(RESET / p**2 * math.sqrt(p * (1 - p) / passed.size))
    no_reset = cost_with_resets(fee, fee, passed, no_tl, ACTIVATION)  # a retry is a new purchase
    assert no_reset.value == pytest.approx(PRICE / p + ACTIVATION)
    # with a constant fee and no reset the two cost outputs agree
    assert cost_to_funded(fee, passed, ACTIVATION).value == pytest.approx(no_reset.value)
    assert cost_to_funded(fee, np.zeros(5, dtype=np.int64), 0).value is None
    assert cost_with_resets(fee[:5], fee[:5], np.zeros(5, dtype=np.int64), no_tl[:5], 0).value is None


def test_retry_after_a_time_limit_is_a_new_purchase() -> None:
    # failures by time limit retry at the price, failures by breach at the reset
    passed = np.array([1, 0, 0, 0, 0] * 200)
    tl = np.array([0, 1, 1, 0, 0] * 200)  # 2 time limits, 2 breaches per 5 paths
    fee = np.full(passed.size, PRICE)
    reset = np.full(passed.size, RESET)
    p = 0.2
    est = cost_with_resets(fee, reset, passed, tl, ACTIVATION)
    # 4 retries per pass, half after a time limit (price), half after a breach (reset)
    assert est.value == pytest.approx(PRICE + 2 * PRICE + 2 * RESET + ACTIVATION)
    only_tl = cost_with_resets(fee, reset, passed, 1 - passed, ACTIVATION)
    assert only_tl.value == pytest.approx(cost_to_funded(fee, passed, ACTIVATION).value)
    assert only_tl.value == pytest.approx(PRICE / p + ACTIVATION)
    assert est.se is not None and est.se > 0


@pytest.mark.parametrize("time_limits", [False, True], ids=["breaches", "time_limits_too"])
def test_delta_method_se_matches_spread_across_seeds(time_limits: bool) -> None:
    est_ctf, se_ctf, est_res, se_res = [], [], [], []
    for seed in range(150):
        res, _ = _run(n=600, seed=1000 + seed, rec=Recorder(time_limits=time_limits))
        est_ctf.append(res.cost_to_funded.value)
        se_ctf.append(res.cost_to_funded.se)
        est_res.append(res.cost_to_funded_with_resets.value)
        se_res.append(res.cost_to_funded_with_resets.se)
    for est, se in ((est_ctf, se_ctf), (est_res, se_res)):
        ratio = float(np.std(est, ddof=1)) / float(np.mean(se))
        assert 0.8 < ratio < 1.2, ratio


def test_delta_se_on_known_influence() -> None:
    assert delta_se(np.array([1.0, -1.0, 1.0, -1.0])) == pytest.approx(0.5)


def test_funded_continues_from_day_after_pass() -> None:
    res, rec = _run(mean_block=10_000_000, horizon_funded=20)
    pp = _per_path(rec)
    fseq = np.concatenate(rec.funded_seqs)
    start = pp["first"][pp["passed"] == 1]
    used = pp["used"][pp["passed"] == 1]
    assert fseq.shape == (start.size, 20)
    assert np.array_equal(fseq[:, 0], (start + used) % N_DAYS)  # no jump at mean_block 1e7 in practice
    assert sum(res.funded_ends.values()) == int(pp["passed"].sum())


def test_funded_outputs_and_ev_per_purchase() -> None:
    res, rec = _run()
    pp = _per_path(rec)
    f0 = np.concatenate([s[:, 0] for s in rec.funded_seqs])
    trader = f0 * 100 + np.where(f0 % 3 == 0, 50, 0)
    ffee = FUNDED_MONTHLY * ((calendar_days(5 + f0) + 29) // 30)
    value = trader - ffee
    assert res.funded_ev.value == pytest.approx(value.mean())
    assert res.funded_ev.se == pytest.approx(value.std(ddof=1) / math.sqrt(value.size))
    assert res.funded_ev.n == int(pp["passed"].sum())
    breach_first = ((f0 % 2 == 0) & (f0 % 4 == 0)).mean()
    assert res.p_breach_before_first_payout.value == pytest.approx(breach_first)
    ev = np.zeros(pp["fee"].size)
    ev[pp["passed"] == 1] = value - ACTIVATION
    ev -= pp["fee"]
    assert res.ev_per_purchase.value == pytest.approx(ev.mean())
    assert res.ev_per_purchase.se == pytest.approx(ev.std(ddof=1) / math.sqrt(ev.size))
    assert any("funded lives still open" in w for w in res.warnings)


def test_days_to_pass_quantiles_are_observed_values() -> None:
    res, rec = _run()
    pp = _per_path(rec)
    pass_days = pp["used"][pp["passed"] == 1]
    assert res.days_to_pass_median.value == float(np.quantile(pass_days, 0.5, method="inverted_cdf"))
    assert res.days_to_pass_p90.value == float(np.quantile(pass_days, 0.9, method="inverted_cdf"))
    assert res.days_to_pass_median.value in set(pass_days.tolist())
    assert res.days_to_pass_median.se is not None and res.days_to_pass_median.se >= 0


def test_deterministic_by_seed_and_recorded() -> None:
    a, _ = _run(seed=3)
    b, _ = _run(seed=3)
    c, _ = _run(seed=4)
    assert a.to_json() == b.to_json()
    assert a.to_json() != c.to_json()
    data = json.loads(a.to_json())
    assert (data["seed"], data["generator"], data["mean_block"], data["n_paths"]) == (3, "block", 5, 4000)
    assert data["plan_sha256"] == "f" * 64 and data["kind"] == "synthetic" and data["trial"] is False
    assert STANDING_WARNING in data["warnings"]


def test_chunking_covers_every_path() -> None:
    res, rec = _run(n=2500, chunk_paths=1000)
    assert [s.shape[0] for s in rec.eval_seqs] == [1000, 1000, 500]
    assert res.p_pass.n == 2500


def test_time_limit_shortens_eval_horizon() -> None:
    plan = FakePlan(eval={"time_limit_calendar_days": 30})
    assert eval_horizon(plan, 252) == 21
    assert eval_horizon(FakePlan(), 252) == 252
    res, rec = _run(plan=plan)
    assert rec.eval_seqs[0].shape[1] == 21 and res.horizon_eval == 21


def test_rule_unknown_in_funded_is_a_warning() -> None:
    res, _ = _run(rec=Recorder(funded_unknown=True))
    assert res.funded_ev.value is None and res.ev_per_purchase.value is None
    assert res.p_pass.value is not None
    assert any("rule unknown" in w and "payout.consistency_bps" in w for w in res.warnings)


def test_random_start_generator_and_timeouts_reported() -> None:
    def eval_many(plan: Any, days: Any, seqs: np.ndarray, **_: Any) -> SimpleNamespace:
        n = seqs.shape[0]
        return SimpleNamespace(end=np.full(n, END_TIMEOUT), days=np.full(n, seqs.shape[1]))

    base = fake_engine(Recorder())
    eng = Engine(eval_many=eval_many, funded_many=base.funded_many, attempt_fee=base.attempt_fee,
                 funded_fee=base.funded_fee)  # fmt: skip
    res = run_mc(FakePlan(), DAYS, n_paths=500, seed=1, generator="random_start", kind="engine_check",
                 engine=eng, horizon_eval=5)  # fmt: skip
    assert res.mean_block is None and res.kind == "engine_check"
    assert res.p_pass.value == 0.0 and res.cost_to_funded.value is None
    text = " ".join(res.warnings)
    assert "500 of 500 attempts timed out" in text and "ran into the end of history" in text
    assert "no attempt passed" in text


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"kind": "real"}, "kind"),
        ({"generator": "shuffle"}, "generator"),
        ({"stress_ticks": 1}, "tick_cents"),
        ({"plan": SimpleNamespace(sha256="")}, "sha256"),
    ],
)
def test_run_mc_refuses_bad_arguments(kw: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _run(**kw)


def test_stress_lowers_the_days_passed_to_the_engine() -> None:
    seen: list[np.ndarray] = []
    base = fake_engine(Recorder())

    def eval_many(plan: Any, days: Any, seqs: np.ndarray, **_: Any) -> Any:
        seen.append(days.day_net())
        return base.eval_many(plan, days, seqs)

    eng = Engine(eval_many=eval_many, funded_many=base.funded_many, attempt_fee=base.attempt_fee,
                 funded_fee=base.funded_fee)  # fmt: skip
    res = run_mc(FakePlan(), DAYS, n_paths=10, seed=1, kind="synthetic", engine=eng, stress_ticks=2,
                 tick_cents=125)  # fmt: skip
    assert np.array_equal(seen[0], DAYS.day_net() - 2 * 125 * 2)
    assert (res.stress_ticks, res.tick_cents) == (2, 125)


def test_time_limit_failures_retry_at_the_price() -> None:
    res, rec = _run(rec=Recorder(time_limits=True))
    pp = _per_path(rec)
    tl = (pp["first"] == 4) | (pp["first"] == 5)
    p = pp["passed"].mean()
    f, r = pp["fee"].mean(), pp["retry"].mean()
    want = f + (tl.mean() * f + (1 - p - tl.mean()) * r) / p + ACTIVATION
    assert res.cost_to_funded_with_resets.value == pytest.approx(want)
    assert res.eval_ends["time_limit"] == int(tl.sum())


@pytest.mark.parametrize(
    ("which", "nulled"),
    [
        ("reset", {"cost_to_funded_with_resets"}),
        ("first", {"cost_per_attempt", "cost_to_funded", "cost_to_funded_with_resets", "ev_per_purchase"}),
        ("funded", {"funded_ev", "ev_per_purchase"}),
    ],
)
def test_unknown_fee_nulls_only_the_outputs_that_need_it(which: str, nulled: set[str]) -> None:
    # the run goes on; only the outputs using the unknown fee are null, and the warning names it
    res, _ = _run(rec=Recorder(fee_unknown=which))
    outputs = (
        "p_pass",
        "days_to_pass_median",
        "days_to_pass_p90",
        "cost_per_attempt",
        "cost_to_funded",
        "cost_to_funded_with_resets",
        "funded_ev",
        "p_breach_before_first_payout",
        "ev_per_purchase",
    )
    assert {o for o in outputs if getattr(res, o).value is None} == nulled
    warned = [w for w in res.warnings if w.startswith("rule unknown")]
    assert len(warned) == 1 and "_cents" in warned[0] and "funded outputs not computed" not in warned[0]
    assert all(o in warned[0] for o in nulled)


def test_size_limit_ends_counted_apart_from_breaches() -> None:
    res, rec = _run(rec=Recorder(size_limits=True))
    pp = _per_path(rec)
    f0 = np.concatenate([s[:, 0] for s in rec.funded_seqs])
    assert res.eval_ends["size_limit"] == int((pp["first"] == 9).sum()) > 0
    assert res.counts["eval_size_limit"] == res.eval_ends["size_limit"]
    assert res.funded_ends["size_limit"] == res.counts["funded_size_limit"] == int((f0 == 1).sum()) > 0
    assert res.eval_ends["breach"] == int(((pp["first"] >= K_PASS) & (pp["first"] != 9)).sum())
    size = [w for w in res.warnings if w.startswith("size:")]
    assert len(size) == 1 and "END_SIZE_LIMIT" in size[0]


def test_size_warning_gives_the_share_of_days_over_each_limit() -> None:
    plan = FakePlan(eval={"time_limit_calendar_days": None, "max_contracts_minis": 3},
                    fees={"activation_cents": ACTIVATION})  # fmt: skip
    object.__setattr__(plan, "funded", {"scaling": [[150_000, 2], ["none", 5]]})
    days = from_day_nets(list(range(N_DAYS)), max_contracts=[1, 2, 3, 4, 1, 1, 1, 1, 1, 1])
    res = run_mc(plan, days, n_paths=200, seed=1, kind="synthetic", engine=fake_engine(Recorder()))
    size = [w for w in res.warnings if w.startswith("size:")]
    assert size == ["size: 10.0% of pool days above eval max_contracts_minis (3); 20.0% of pool days above "
                    "the first funded scaling tier (2); END_SIZE_LIMIT ends (counted apart from breaches): "
                    "0 attempts, 0 funded lives"]  # fmt: skip
    clean, _ = _run()
    assert not any(w.startswith("size:") for w in clean.warnings)


def test_degenerate_binomial_se_is_flagged_and_se_scope_stated() -> None:
    res, _ = _run(rec=Recorder())
    assert res.p_pass.note == ""
    data = json.loads(res.to_json())
    assert "Monte Carlo error given the day pool" in data["se_scope"]
    from pfb.prop.mc import binomial

    for k in (0, 50):
        est = binomial(k, 50)
        assert est.se == 0.0 and "degenerate" in est.note and "3/n" in est.note


def test_random_start_short_history_marks_lower_bounds() -> None:
    def eval_many(plan: Any, days: Any, seqs: np.ndarray, **_: Any) -> SimpleNamespace:
        n = seqs.shape[0]
        hit_pad = (seqs == -1).any(axis=1)
        return SimpleNamespace(end=np.where(hit_pad, END_TIMEOUT, END_PASS), days=np.full(n, 2))

    def funded_many(plan: Any, days: Any, seqs: np.ndarray, **_: Any) -> SimpleNamespace:
        n = seqs.shape[0]
        hit_pad = (seqs == -1).any(axis=1)
        return SimpleNamespace(end=np.where(hit_pad, END_TIMEOUT, END_BREACH), days=np.full(n, 3),
                               payouts_trader_cents=np.zeros(n), first_payout_day=np.full(n, -1))  # fmt: skip

    base = fake_engine(Recorder())
    eng = Engine(eval_many=eval_many, funded_many=funded_many, attempt_fee=base.attempt_fee,
                 funded_fee=base.funded_fee)  # fmt: skip
    # H_e + H_f = 10 = the history: full windows only, nothing padded
    full = run_mc(FakePlan(), DAYS, n_paths=400, seed=1, generator="random_start", kind="synthetic",
                  engine=eng, horizon_eval=3, horizon_funded=7)  # fmt: skip
    assert full.counts["eval_padded"] == full.counts["funded_padded"] == 0
    assert not any(getattr(full, o).lower_bound for o in ("p_pass", "funded_ev", "ev_per_purchase"))
    short = run_mc(FakePlan(), DAYS, n_paths=400, seed=1, generator="random_start", kind="synthetic",
                   engine=eng, horizon_eval=3, horizon_funded=9)  # fmt: skip
    c = short.counts
    assert c["eval_padded"] > 0 and c["funded_padded"] > 0
    assert c["eval_padded"] == short.eval_ends["timeout"]
    assert c["funded_padded"] + short.funded_ends.get("breach", 0) == sum(short.funded_ends.values())
    for name in ("p_pass", "funded_ev", "p_breach_before_first_payout", "ev_per_purchase"):
        assert getattr(short, name).lower_bound, name
    assert not short.cost_per_attempt.lower_bound
    text = " ".join(short.warnings)
    assert "ran into the end of history" in text and "lower bounds" in text
    assert c["funded_open_no_payout"] == c["funded_padded"]
    assert "open with no payout" in text
