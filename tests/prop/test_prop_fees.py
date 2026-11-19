"""fees.py: attempt and funded fees per docs/PROP_MC.md section 4 (Topstep 50K and synthetic variants)."""

from __future__ import annotations

import pytest
from _synth import TOPSTEP_STD

from pfb.prop.config import load_plan, with_values
from pfb.prop.fees import (
    activation_fee_cents,
    attempt_fee_cents,
    billing_periods,
    funded_fee_cents,
    platform_fee_cents,
)

STD = load_plan(TOPSTEP_STD)


@pytest.mark.parametrize(("days", "periods"), [(0, 1), (1, 1), (30, 1), (31, 2), (60, 2), (61, 3)])
def test_started_billing_periods(days: int, periods: int) -> None:
    assert billing_periods(days, 30) == periods


def test_bad_inputs_are_refused() -> None:
    with pytest.raises(ValueError):
        billing_periods(-1, 30)
    with pytest.raises(ValueError):
        attempt_fee_cents(STD, -1, 3, first=True)
    with pytest.raises(ValueError):
        funded_fee_cents(STD, -1, 3)


def test_first_attempt_pays_the_price_per_started_period() -> None:
    assert attempt_fee_cents(STD, 5, 7, first=True, include_platform=False) == 4_900
    assert attempt_fee_cents(STD, 30, 45, first=True, include_platform=False) == 9_800
    assert attempt_fee_cents(STD, 30, 45, first=True) == 9_800 + 1_450 * 45 // 30  # platform pro rata


def test_retry_is_a_reset_that_pushes_the_rebill() -> None:
    assert attempt_fee_cents(STD, 5, 7, first=False, include_platform=False) == 4_900
    assert attempt_fee_cents(STD, 30, 45, first=False, include_platform=False) == 4_900 + 4_900
    no_push = with_values(STD, {"fees.reset_pushes_rebill": False})
    assert attempt_fee_cents(no_push, 30, 45, first=False, include_platform=False) == 4_900 + 2 * 4_900


def test_retry_without_a_reset_is_a_new_purchase() -> None:
    plan = with_values(STD, {"fees.reset_cents": "none"})
    assert attempt_fee_cents(plan, 30, 45, first=False, include_platform=False) == 9_800


def test_one_time_plan_ignores_time() -> None:
    plan = with_values(
        STD,
        {
            "fees.billing": "one_time",
            "fees.billing_period_calendar_days": "none",
            "fees.eval_price_cents": 15_000,
            "fees.reset_cents": 9_000,
        },
    )
    assert attempt_fee_cents(plan, 40, 90, first=True, include_platform=False) == 15_000
    assert attempt_fee_cents(plan, 40, 90, first=False, include_platform=False) == 9_000
    assert attempt_fee_cents(plan, 40, 90, first=True) == 15_000 + 1_450 * 90 // 30


def test_funded_fees() -> None:
    assert funded_fee_cents(STD, 10, 14, include_platform=False) == 0
    assert funded_fee_cents(STD, 10, 14) == 1_450 * 14 // 30
    assert funded_fee_cents(STD, 0, 0) == 0
    plan = with_values(STD, {"fees.funded_monthly_cents": 12_000, "fees.platform_monthly_cents": 0})
    assert funded_fee_cents(plan, 40, 61) == 3 * 12_000
    assert platform_fee_cents(plan, 61) == 0
    assert activation_fee_cents(STD) == 14_900


@pytest.mark.parametrize(("days", "cents"), [(0, 0), (1, 48), (2, 96), (8, 386), (30, 1_450), (31, 1_498),
                                             (86, 4_156)])  # fmt: skip
def test_platform_fee_is_pro_rata_to_calendar_days(days: int, cents: int) -> None:
    assert platform_fee_cents(STD, days) == cents  # 1,450 x days // 30, integer cents


def test_short_attempts_do_not_each_pay_a_platform_month() -> None:
    # seven 3-day attempts in one month pay about one month of platform fee, not seven
    one = attempt_fee_cents(STD, 2, 3, first=True) - attempt_fee_cents(STD, 2, 3, first=True,
                                                                     include_platform=False)  # fmt: skip
    assert one == 1_450 * 3 // 30
    assert 7 * one <= 1_450
    with pytest.raises(ValueError):
        platform_fee_cents(STD, -1)
