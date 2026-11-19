"""kernel.py (Numba) pinned field by field to rules.py (scalar reference) on random sequences.

Every config file present in config/firms/ at test time gets 2,000+ random sequences per phase, on two random
day pools (flat and positive drift); synthetic variants of the Topstep plan reach the A3 extension branches
(intraday trailing, static, daily loss soft and hard, intraday target, time limit, cumulative consistency,
buffer, weekday, dead days, caps per payout, payout limit, first-full split, and the calendar-day
payout window with the sequence clock's weekday).
"""

from __future__ import annotations

import numpy as np
import pytest
from _synth import FIRMS_DIR, TOPSTEP_CONS, TOPSTEP_STD, random_days, random_seqs

from pfb.prop.config import PlanRules, load_plan, with_values
from pfb.prop.kernel import simulate_eval_many, simulate_funded_many
from pfb.prop.rules import simulate_eval, simulate_funded
from pfb.prop.types import (
    END_BREACH,
    END_PASS,
    END_PAYOUT_LIMIT,
    END_SIZE_LIMIT,
    END_TIME_LIMIT,
    DayPaths,
    RuleUnknown,
)

N_PATHS = 2_000
HORIZON = 90
CONFIGS = sorted(FIRMS_DIR.glob("*/*.toml"))
VARIANTS: dict[str, dict[str, object]] = {
    "intraday_trailing_soft_dll": {
        "eval.drawdown_type": "intraday_trailing",
        "eval.daily_loss_cents": 60_000,
        "eval.daily_loss_kind": "soft",
        "funded.drawdown_type": "intraday_trailing",
        "funded.daily_loss_cents": 60_000,
        "funded.daily_loss_kind": "soft",
        "funded.floor_after_first_payout_rel_cents": "none",
        "funded.drawdown_lock_rel_cents": 10_000,
    },
    "static_hard_dll_eod_check": {
        "eval.drawdown_type": "static",
        "eval.daily_loss_cents": 80_000,
        "eval.daily_loss_kind": "hard",
        "eval.breach_check": "eod",
        "eval.breach_at_equal": False,
        "funded.drawdown_type": "static",
        "funded.daily_loss_cents": 80_000,
        "funded.daily_loss_kind": "hard",
        "funded.breach_check": "eod",
    },
    "intraday_target_time_limit_wins": {
        "eval.target_check": "intraday",
        "eval.time_limit_calendar_days": 60,
        "eval.min_winning_days": 2,
        "eval.winning_day_cents": 20_000,
        "eval.consistency_type": "best_day_frac_of_target",
        "eval.consistency_strict": False,
        "eval.drawdown_lock_rel_cents": "none",
        "eval.max_contracts_minis": 4,
    },
    "a3_payout_extensions": {
        "payout.consistency_bps": 3_000,
        "payout.consistency_scope": "cumulative",
        "payout.request_weekday": 3,
        "payout.dead_days_after_payout": 1,
        "payout.buffer_rel_cents": 20_000,
        "payout.caps_cents": [50_000, 100_000, 150_000],
        "payout.split_first_full_cents": 40_000,
        "payout.net_min_first_cents": 30_000,
        "payout.net_min_later_cents": 10_000,
        "payout.min_balance_rel_cents": 25_000,
        "payout.win_day_strict": True,
        "payout.payout_day_counts": True,
        "funded.payout_count_limit": 3,
        "funded.scaling_boundary": "upper",
    },
    "consistency_window_no_floor_fix": {
        "payout.traded_days_required": 2,
        "payout.win_days_required": 0,
        "payout.win_day_cents": "none",
        "payout.consistency_bps": 5_000,
        "payout.net_positive_since_last": False,
        "payout.frac_of_profit_bps": 10_000,
        "payout.min_request_cents": 0,
        "funded.floor_after_first_payout_rel_cents": -50_000,
    },
    "calendar_window_weekday": {  # payout.window_calendar_days and the sequence clock's weekday
        "payout.window_calendar_days": 10,
        "payout.request_weekday": 2,
        "payout.dead_days_after_payout": 1,
        "payout.win_days_required": 2,
    },
    "window_cap_no_fixed_cap": {
        "payout.caps_cents": "none",
        "payout.window_cap_bps": 15_000,
        "payout.window_cap_from_payout": 2,
        "payout.payout_day_counts": True,
        "funded.floor_after_first_payout_rel_cents": "none",
    },
}


def _pools() -> list[tuple[str, DayPaths]]:
    rng = np.random.default_rng(20261116)
    return [
        ("flat", random_days(rng, 300, scale=50_000, drift=0)),
        ("drift", random_days(rng, 300, scale=40_000, drift=12_000)),
    ]


POOLS = _pools()


def _plans() -> list[tuple[str, PlanRules]]:
    plans = [(str(p.relative_to(FIRMS_DIR)), load_plan(p)) for p in CONFIGS]
    base = load_plan(TOPSTEP_STD)
    plans += [(f"variant:{k}", with_values(base, v)) for k, v in VARIANTS.items()]
    plans.append(
        (
            "variant:consistency_cumulative",
            with_values(load_plan(TOPSTEP_CONS), {"payout.consistency_scope": "cumulative"}),
        )
    )
    return plans


PLANS = _plans()


def test_config_files_are_present() -> None:
    assert TOPSTEP_STD in CONFIGS and TOPSTEP_CONS in CONFIGS


def _seqs(seed: int, n_days: int) -> np.ndarray:
    return random_seqs(np.random.default_rng(seed), n_days, N_PATHS, HORIZON)


def _unknown(fn) -> RuleUnknown | None:
    try:
        fn()
    except RuleUnknown as exc:
        return exc
    return None


@pytest.mark.parametrize(("name", "plan"), PLANS, ids=[n for n, _ in PLANS])
@pytest.mark.parametrize(("pool_name", "days"), POOLS, ids=[n for n, _ in POOLS])
def test_eval_kernel_equals_scalar(name: str, plan: PlanRules, pool_name: str, days) -> None:
    seqs = _seqs(1, days.n_days)
    exc = _unknown(lambda: simulate_eval(plan, days, seqs[0]))
    if exc is not None:
        with pytest.raises(RuleUnknown):
            simulate_eval_many(plan, days, seqs)
        pytest.skip(f"{name}: evaluation needs an unknown rule ({exc})")
    batch = simulate_eval_many(plan, days, seqs)
    for i in range(N_PATHS):
        ref = simulate_eval(plan, days, seqs[i])
        got = (
            batch.end[i],
            batch.days[i],
            batch.balance_cents[i],
            batch.best_day_cents[i],
            batch.traded_days[i],
        )
        assert got == (ref.end, ref.days, ref.balance_cents, ref.best_day_cents, ref.traded_days), (name, i)


@pytest.mark.parametrize(("name", "plan"), PLANS, ids=[n for n, _ in PLANS])
@pytest.mark.parametrize(("pool_name", "days"), POOLS, ids=[n for n, _ in POOLS])
def test_funded_kernel_equals_scalar(name: str, plan: PlanRules, pool_name: str, days) -> None:
    seqs = _seqs(2, days.n_days)
    exc = _unknown(lambda: simulate_funded(plan, days, seqs[0]))
    if exc is not None:
        with pytest.raises(RuleUnknown):
            simulate_funded_many(plan, days, seqs)
        pytest.skip(f"{name}: funded phase needs an unknown rule ({exc})")
    batch = simulate_funded_many(plan, days, seqs)
    for i in range(N_PATHS):
        ref = simulate_funded(plan, days, seqs[i])
        n = int(batch.n_payouts[i])
        assert n == len(ref.payouts_gross_cents), (name, i)
        got = (
            batch.end[i],
            batch.days[i],
            tuple(int(x) for x in batch.payouts_gross_cents[i, :n]),
            tuple(int(x) for x in batch.payouts_trader_cents[i, :n]),
            batch.first_payout_day[i],
            batch.balance_cents[i],
            batch.gross_total_cents[i],
            batch.trader_total_cents[i],
        )
        want = (
            ref.end,
            ref.days,
            ref.payouts_gross_cents,
            ref.payouts_trader_cents,
            ref.first_payout_day,
            ref.balance_cents,
            sum(ref.payouts_gross_cents),
            sum(ref.payouts_trader_cents),
        )
        assert got == want, (name, i)
        assert not batch.payouts_gross_cents[i, n:].any()


def test_branches_are_reached() -> None:
    """The random pools reach every end code the engine has (guards against a vacuous parity)."""
    ends_eval: set[int] = set()
    ends_funded: set[int] = set()
    paid = 0
    for name, plan in PLANS:
        if not (name.startswith("variant:") or name.startswith("topstep")):
            continue
        for _, days in POOLS:
            seqs = _seqs(1, days.n_days)
            ends_eval |= set(np.unique(simulate_eval_many(plan, days, seqs).end).tolist())
            fb = simulate_funded_many(plan, days, seqs)
            ends_funded |= set(np.unique(fb.end).tolist())
            paid += int((fb.n_payouts > 1).sum())
    assert {END_PASS, END_BREACH, END_TIME_LIMIT, END_SIZE_LIMIT} <= ends_eval
    assert {END_BREACH, END_PAYOUT_LIMIT, END_SIZE_LIMIT} <= ends_funded
    assert paid > 100


def _eval_key(plan: PlanRules, days: DayPaths, seqs: np.ndarray) -> np.ndarray:
    b = simulate_eval_many(plan, days, seqs)
    return np.stack([b.end, b.days, b.balance_cents])


def _funded_key(plan: PlanRules, days: DayPaths, seqs: np.ndarray) -> np.ndarray:
    b = simulate_funded_many(plan, days, seqs)
    return np.stack([b.end, b.days, b.n_payouts, b.gross_total_cents, b.balance_cents])


# (variant, phase, the same variant with the rule switched off): the rule fires on some path iff turning it
# off changes some outcome, so a branch that never fires on the parity pools fails here.
TOGGLES = [
    ("intraday_trailing_soft_dll", "eval", {"eval.daily_loss_cents": "none", "eval.daily_loss_kind": "none"}),
    (
        "intraday_trailing_soft_dll",
        "funded",
        {"funded.daily_loss_cents": "none", "funded.daily_loss_kind": "none"},
    ),
    ("intraday_target_time_limit_wins", "eval", {"eval.target_check": "eod"}),
    ("a3_payout_extensions", "funded", {"payout.request_weekday": "none"}),
    ("a3_payout_extensions", "funded", {"payout.dead_days_after_payout": 0}),
    ("window_cap_no_fixed_cap", "funded", {"payout.window_cap_bps": "none"}),
    ("calendar_window_weekday", "funded", {"payout.window_calendar_days": "none"}),
    ("calendar_window_weekday", "funded", {"payout.request_weekday": "none"}),
]


@pytest.mark.parametrize(
    ("variant", "phase", "off"), TOGGLES, ids=[f"{v}:{p}:{','.join(o)}" for v, p, o in TOGGLES]
)
def test_rule_branches_fire_on_the_parity_pools(variant: str, phase: str, off: dict[str, object]) -> None:
    on_plan = dict(PLANS)[f"variant:{variant}"]
    off_plan = with_values(on_plan, off)
    key = _eval_key if phase == "eval" else _funded_key
    changed = 0
    for _, days in POOLS:
        seqs = _seqs(1 if phase == "eval" else 2, days.n_days)
        changed += int((key(on_plan, days, seqs) != key(off_plan, days, seqs)).any(axis=0).sum())
    assert changed > 0, (variant, phase, off)


def test_bad_sequences_are_refused() -> None:
    plan = load_plan(TOPSTEP_STD)
    days = POOLS[0][1]
    with pytest.raises(ValueError, match="outside"):
        simulate_eval_many(plan, days, np.full((2, 3), days.n_days))
    with pytest.raises(ValueError, match="n_paths, horizon"):
        simulate_eval_many(plan, days, np.zeros(3, dtype=np.int64))
    with pytest.raises(ValueError, match="outside"):
        simulate_funded_many(plan, days, np.full((2, 3), -2))
