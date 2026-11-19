"""Firm fees of one attempt and one funded life (docs/PROP_MC.md section 4), integer cents.

Readings:
- A monthly plan bills its price once per started billing period: periods = max(1, ceil(calendar_days /
  billing_period_calendar_days)), calendar_days counted by the caller from the trade dates.
- A retry (first=False) after a failed attempt costs the reset when the plan has one, else a new purchase
  (mc.py prices a retry after END_TIME_LIMIT as a new purchase, first=True, per the retry rule). On
  a monthly plan a reset that pushes the rebill restarts the billing clock (reset + price x (periods - 1));
  one that does not leaves the subscription rebilling on its own schedule (reset + price x periods, the
  conservative count). Reset credits added by rebills are not credited (conservative: the credit's use depends
  on the previous attempt, which this function does not see).
- The platform fee (API or data a bot needs) is one continuous subscription, so each phase pays it pro rata to
  its calendar days: platform_monthly_cents x calendar_days // 30 (integer cents, floor), in both phases
  (PROP_MC.md); include_platform=False gives the firm's own fees alone.
"""

from __future__ import annotations

from pfb.prop.config import PlanRules

DEFAULT_PERIOD_DAYS = 30  # platform subscriptions and plans without a billing period


def billing_periods(calendar_days: int, period_days: int) -> int:
    """Started billing periods in a span of calendar days (at least one)."""
    if calendar_days < 0 or period_days <= 0:
        raise ValueError(f"calendar_days {calendar_days} and period_days {period_days} must be >= 0 and > 0")
    return max(1, -(-calendar_days // period_days))


def _period(plan: PlanRules) -> int:
    period = plan.value("fees.billing_period_calendar_days")
    return DEFAULT_PERIOD_DAYS if period is None else int(period)


def platform_fee_cents(plan: PlanRules, calendar_days: int) -> int:
    """The platform fee pro rata to calendar days: monthly x calendar_days // 30 (integer cents)."""
    if calendar_days < 0:
        raise ValueError(f"calendar_days {calendar_days} must be >= 0")
    monthly = int(plan.value("fees.platform_monthly_cents"))
    return monthly * calendar_days // DEFAULT_PERIOD_DAYS


def attempt_fee_cents(
    plan: PlanRules, outcome_days: int, calendar_days: int, *, first: bool, include_platform: bool = True
) -> int:
    """Fees of one evaluation attempt: the purchase (first) or the retry (reset if any, else a new purchase),
    plus the monthly fee per started period on monthly plans and the platform fee pro rata."""
    if outcome_days < 0:
        raise ValueError(f"outcome_days {outcome_days} must be >= 0")
    price = int(plan.value("fees.eval_price_cents"))
    reset = None if first else plan.value("fees.reset_cents")
    if plan.value("fees.billing") == "one_time":
        fee = price if reset is None else int(reset)
    else:
        periods = billing_periods(calendar_days, _period(plan))
        if reset is None:
            fee = price * periods
        elif plan.value("fees.reset_pushes_rebill"):
            fee = int(reset) + price * (periods - 1)
        else:
            fee = int(reset) + price * periods
    return fee + (platform_fee_cents(plan, calendar_days) if include_platform else 0)


def funded_fee_cents(
    plan: PlanRules, days_alive: int, calendar_days: int, *, include_platform: bool = True
) -> int:
    """Fees of one funded life: the funded monthly fee per started period and the platform fee pro rata
    (activation is charged separately, at the pass). A life that never traded a day costs nothing."""
    if days_alive < 0:
        raise ValueError(f"days_alive {days_alive} must be >= 0")
    if days_alive == 0:
        return 0
    monthly = int(plan.value("fees.funded_monthly_cents"))
    fee = monthly * billing_periods(calendar_days, _period(plan)) if monthly else 0
    return fee + (platform_fee_cents(plan, calendar_days) if include_platform else 0)


def activation_fee_cents(plan: PlanRules) -> int:
    return int(plan.value("fees.activation_cents"))
