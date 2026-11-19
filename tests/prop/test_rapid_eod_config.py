"""MFFU Rapid EOD 50K : the provisional plan file against the live-rule note, section 2.

Source: the note, section 2 (page dated 2026-08-24, read 2026-11-17).
Every value the note does not give is Unknown (only payout_count_limit comes from rapid_50k.toml, through the
note's "live transition as standard Rapid 50k"). re-read the note against the 22 unknowns: it
states none of them. read the page itself (source P_EOD, 2026-11-17): it states two in words
(funded.start_balance_cents 0 and funded.inactivity_days 7), so 20 remain in the set below.
"""

from __future__ import annotations

import pytest
from _synth import FIRMS_DIR

from pfb.prop.config import load_plan
from pfb.prop.types import RuleUnknown

RAPID_EOD = FIRMS_DIR / "mffu" / "rapid_eod_50k.toml"

UNKNOWN = {
    "eval.target_check",
    "eval.breach_check",
    "eval.breach_at_equal",
    "eval.drawdown_lock_rel_cents",
    "eval.consistency_type",
    "eval.consistency_strict",
    "eval.min_winning_days",
    "eval.winning_day_cents",
    "eval.time_limit_calendar_days",
    "eval.flat_by",
    "funded.breach_check",
    "funded.breach_at_equal",
    "payout.win_day_cents",
    "payout.win_day_strict",
    "payout.win_days_required",
    "payout.dead_days_after_payout",
    "payout.payout_day_counts",
    "fees.reset_cents",
    "fees.funded_monthly_cents",
    "fees.platform_monthly_cents",
}
# added funded.funded_reset_cents to every plan; MFFU's is unknown with no alt (E3: not published
# on the help pages read 2026-11-17); the Monte Carlo does not model a funded reset (test_firm_unknowns.py).
UNKNOWN_FUNDED_RESET = {"funded.funded_reset_cents"}


def test_rapid_eod_loads_as_a_provisional_mffu_plan() -> None:
    p = load_plan(RAPID_EOD)
    assert (p.firm, p.plan, p.size, p.status, p.effective_from) == (
        "mffu",
        "rapid_eod",
        "50k",
        "provisional",
        "2026-08-24",
    )


def test_rapid_eod_evaluation_values_match_the_note() -> None:
    v = load_plan(RAPID_EOD).value
    assert (v("eval.start_balance_cents"), v("eval.profit_target_cents"), v("eval.drawdown_cents")) == (
        5_000_000,
        300_000,
        200_000,
    )
    assert v("eval.drawdown_type") == "eod_trailing"
    assert (v("eval.daily_loss_cents"), v("eval.daily_loss_kind")) == (None, "none")
    assert (v("eval.consistency_bps"), v("eval.min_trading_days"), v("eval.max_contracts_minis")) == (
        3000,
        4,
        3,
    )
    assert v("eval.news_rule") == "unrestricted"


def test_rapid_eod_funded_and_payout_values_match_the_note() -> None:
    v = load_plan(RAPID_EOD).value
    assert (v("funded.drawdown_cents"), v("funded.drawdown_type"), v("funded.drawdown_lock_rel_cents")) == (
        200_000,
        "eod_trailing",
        10_000,
    )
    assert (v("funded.daily_loss_cents"), v("funded.daily_loss_kind")) == (None, "none")
    assert v("funded.scaling") == ((None, 3),)
    assert v("funded.payout_count_limit") is None
    assert (v("funded.start_balance_cents"), v("funded.inactivity_days")) == (0, 7)  # page P_EOD
    assert (v("payout.path"), v("payout.min_balance_rel_cents"), v("payout.buffer_rel_cents")) == (
        "daily",
        210_000,
        210_000,
    )
    assert (v("payout.net_min_first_cents"), v("payout.net_min_later_cents")) == (None, 50_000)
    assert (v("payout.min_request_cents"), v("payout.caps_cents"), v("payout.consistency_bps")) == (
        50_000,
        None,
        None,
    )
    assert (v("payout.split_bps"), v("payout.split_first_full_cents")) == (9000, None)


def test_rapid_eod_fees_match_the_note() -> None:
    p = load_plan(RAPID_EOD)
    v = p.value
    assert (v("fees.billing"), v("fees.eval_price_cents"), v("fees.activation_cents")) == (
        "one_time",
        27_500,
        0,
    )
    assert p.leaf("fees.eval_price_cents").alt == (16_500,)
    assert p.leaf("fees.reset_cents").alt == ("none",)


def test_rapid_eod_unknowns_are_exactly_the_fields_the_note_does_not_give() -> None:
    p = load_plan(RAPID_EOD)
    assert set(p.unknown_paths) == UNKNOWN | UNKNOWN_FUNDED_RESET
    for path in UNKNOWN:
        with pytest.raises(RuleUnknown):
            p.value(path)
        assert p.leaf(path).alt, path  # every unknown carries the reading(s) a sensitivity run may use
