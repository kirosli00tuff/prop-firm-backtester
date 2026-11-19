"""config.py: pfb-rules-1 loading and validation, and the Topstep 50K files against the recorded values."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
from _synth import FIRMS_DIR, TOPSTEP_CONS, TOPSTEP_STD

from pfb.prop.config import SECTION_FIELDS, PlanConfigError, load_plan, with_values
from pfb.prop.types import RuleUnknown

STD_TEXT = TOPSTEP_STD.read_text()


def write(tmp_path: Path, text: str, name: str = TOPSTEP_STD.name) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def problems_of(tmp_path: Path, text: str, name: str = TOPSTEP_STD.name) -> str:
    with pytest.raises(PlanConfigError) as info:
        load_plan(write(tmp_path, text, name))
    return " | ".join(info.value.problems)


# ------------------------------------------------------------------------------ every file ----
ALL_CONFIGS = sorted(FIRMS_DIR.glob("*/*.toml"))


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.parent.name + "/" + p.name)
def test_every_firm_config_loads(path: Path) -> None:
    plan = load_plan(path)
    assert plan.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.parent.name == plan.firm
    expected = sum(len(f) for f in SECTION_FIELDS.values()) + 1
    assert len(plan.leaves) == expected


# --------------------------------------------------------------------------- Topstep values ----
def test_topstep_standard_values() -> None:
    p = load_plan(TOPSTEP_STD)
    v = p.value
    assert (p.firm, p.plan, p.size, p.status, p.effective_from) == (
        "topstep",
        "combine_xfa_standard",
        "50k",
        "sourced",
        "2026-10-10",
    )
    assert (v("eval.start_balance_cents"), v("eval.profit_target_cents"), v("eval.drawdown_cents")) == (
        5_000_000,
        300_000,
        200_000,
    )
    assert (v("eval.drawdown_type"), v("eval.breach_check"), v("eval.drawdown_lock_rel_cents")) == (
        "eod_trailing",
        "intraday",
        0,
    )
    assert (v("eval.consistency_type"), v("eval.consistency_bps"), v("eval.consistency_strict")) == (
        "best_day_frac_of_profit",
        5500,
        True,
    )
    got = (v("eval.min_trading_days"), v("eval.max_contracts_minis"), v("eval.daily_loss_cents"))
    assert got == (2, 5, None)
    assert v("eval.time_limit_calendar_days") is None
    assert (v("funded.start_balance_cents"), v("funded.floor_after_first_payout_rel_cents")) == (0, 0)
    assert v("funded.scaling") == ((150_000, 2), (200_000, 3), (None, 5))
    assert v("funded.scaling_boundary") == "lower" and v("funded.payout_count_limit") is None
    assert (v("payout.win_day_cents"), v("payout.win_days_required"), v("payout.caps_cents")) == (
        15_000,
        5,
        (200_000,),
    )
    assert (v("payout.frac_of_profit_bps"), v("payout.min_request_cents"), v("payout.split_bps")) == (
        5000,
        12_500,
        9000,
    )
    assert v("payout.payout_day_counts") is False and v("payout.net_positive_since_last") is True
    assert (v("fees.eval_price_cents"), v("fees.reset_cents"), v("fees.activation_cents")) == (
        4_900,
        4_900,
        14_900,
    )
    assert v("fees.platform_monthly_cents") == 1_450 and v("fees.billing_period_calendar_days") == 30


def test_topstep_consistency_path_values() -> None:
    v = load_plan(TOPSTEP_CONS).value
    assert (v("payout.path"), v("payout.traded_days_required"), v("payout.consistency_bps")) == (
        "consistency",
        3,
        4000,
    )
    assert v("payout.consistency_strict") is False and v("payout.caps_cents") == (300_000,)
    assert v("payout.win_days_required") == 0 and v("payout.win_day_cents") is None


def test_topstep_sources_are_records() -> None:
    plan = load_plan(TOPSTEP_STD)
    for leaf in plan.leaves.values():
        src = plan.sources[str(leaf.src)]
        assert src.record is not None
    assert plan.leaf("eval.consistency_strict").kind == "conservative"
    assert plan.leaf("eval.consistency_strict").alt == (False,)


# ---------------------------------------------------------------------------- validation ----
def test_missing_field_is_an_error(tmp_path: Path) -> None:
    text = re.sub(r"^min_trading_days = .*\n", "", STD_TEXT, flags=re.M)
    assert "eval.min_trading_days: missing" in problems_of(tmp_path, text)


def test_extra_field_and_extra_leaf_key_are_errors(tmp_path: Path) -> None:
    text = STD_TEXT.replace("[funded]\n", '[funded]\nbogus = { v = 1, src = "R080" }\n')
    assert "funded.bogus: not a pfb-rules-1 field" in problems_of(tmp_path, text)
    text = STD_TEXT.replace(
        'profit_target_cents = { v = 300000, src = "R068" }',
        'profit_target_cents = { v = 300000, src = "R068", extra = 1 }',
    )
    assert "unknown keys ['extra']" in problems_of(tmp_path, text)
    text = STD_TEXT.replace('schema = "pfb-rules-1"', 'schema = "pfb-rules-1"\nrevision = 2')
    assert "unknown top-level keys" in problems_of(tmp_path, text)


def test_src_must_resolve(tmp_path: Path) -> None:
    text = STD_TEXT.replace('{ v = 300000, src = "R068" }', '{ v = 300000, src = "NOPE" }')
    assert "src 'NOPE' is not a key of [sources]" in problems_of(tmp_path, text)
    text = STD_TEXT.replace('{ v = 300000, src = "R068" }', "{ v = 300000 }")
    assert "src is required with v" in problems_of(tmp_path, text)


def test_enum_and_type_checks(tmp_path: Path) -> None:
    text = STD_TEXT.replace(
        'drawdown_type = { v = "eod_trailing", src = "R070" }', 'drawdown_type = { v = "eod", src = "R070" }'
    )
    assert "eval.drawdown_type" in problems_of(tmp_path, text)
    text = STD_TEXT.replace('{ v = 300000, src = "R068" }', '{ v = 3000.5, src = "R068" }')
    assert "eval.profit_target_cents" in problems_of(tmp_path, text)
    text = STD_TEXT.replace('{ v = 300000, src = "R068" }', '{ v = true, src = "R068" }')
    assert "eval.profit_target_cents" in problems_of(tmp_path, text)
    text = STD_TEXT.replace(
        'consistency_bps = { v = 5500, src = "R075" }', 'consistency_bps = { v = 15000, src = "R075" }'
    )
    assert "eval.consistency_bps" in problems_of(tmp_path, text)


def test_none_only_where_allowed(tmp_path: Path) -> None:
    text = STD_TEXT.replace('{ v = 300000, src = "R068" }', '{ v = "none", src = "R068" }')
    assert '"none" is not allowed here' in problems_of(tmp_path, text)
    text = STD_TEXT.replace('path = { v = "standard"', 'path = { v = "none"')
    assert "payout.path" in problems_of(tmp_path, text)


def test_flat_by_and_news_rule_admit_none_and_any_flag(tmp_path: Path) -> None:
    text = re.sub(r'news_rule = \{ v = "[^"]*"', 'news_rule = { v = "none"', STD_TEXT)
    text = re.sub(r'flat_by = \{ v = "[^"]*"', 'flat_by = { v = "none"', text)
    plan = load_plan(write(tmp_path, text))
    assert plan.value("eval.news_rule") is None and plan.value("eval.flat_by") is None
    text = re.sub(r'news_rule = \{ v = "[^"]*"', 'news_rule = { v = "no_news_rule"', STD_TEXT)
    assert load_plan(write(tmp_path, text)).value("eval.news_rule") == "no_news_rule"


def test_unknown_is_kept_and_raises_when_used(tmp_path: Path) -> None:
    text = STD_TEXT.replace('{ v = 300000, src = "R068" }', '{ unknown = "page silent", src = "R068" }')
    plan = load_plan(write(tmp_path, text))
    assert plan.unknown_paths == ("eval.profit_target_cents",)
    with pytest.raises(RuleUnknown, match="eval.profit_target_cents: page silent"):
        plan.value("eval.profit_target_cents")
    empty = STD_TEXT.replace('{ v = 300000, src = "R068" }', '{ unknown = " " }')
    assert "non-empty reason" in problems_of(tmp_path, empty)
    both = STD_TEXT.replace('{ v = 300000, src = "R068" }', '{ v = 1, unknown = "x", src = "R068" }')
    assert "exactly one of v and unknown" in problems_of(tmp_path, both)


def test_scaling_and_caps_shapes(tmp_path: Path) -> None:
    tiers = '[[150000, 2], [200000, 3], ["none", 5]]'
    text = STD_TEXT.replace(tiers, '[[200000, 2], [150000, 3], ["none", 5]]')
    assert "not strictly ascending" in problems_of(tmp_path, text)
    text = STD_TEXT.replace('[[150000, 2], [200000, 3], ["none", 5]]', "[[150000, 2], [200000, 3]]")
    assert 'only it, has "none"' in problems_of(tmp_path, text)
    text = STD_TEXT.replace("caps_cents = { v = [200000]", "caps_cents = { v = []")
    assert "payout.caps_cents" in problems_of(tmp_path, text)


def test_window_cap_fields_and_caps_none(tmp_path: Path) -> None:
    v = load_plan(TOPSTEP_STD).value
    assert (v("payout.window_cap_bps"), v("payout.window_cap_from_payout")) == (None, 1)
    text = STD_TEXT.replace("caps_cents = { v = [200000]", 'caps_cents = { v = "none"')
    assert load_plan(write(tmp_path, text)).value("payout.caps_cents") is None
    text = re.sub(r"^window_cap_bps = .*\n", "", STD_TEXT, flags=re.M)
    assert "payout.window_cap_bps: missing" in problems_of(tmp_path, text)
    with pytest.raises(PlanConfigError, match="must be >= 1"):
        with_values(load_plan(TOPSTEP_STD), {"payout.window_cap_from_payout": 0})
    with pytest.raises(PlanConfigError, match="must be > 0"):
        with_values(load_plan(TOPSTEP_STD), {"payout.window_cap_bps": 0})


def test_fix_round_firm_values() -> None:
    """Select Daily 2x rule, MFFU Rapid with no fixed cap, Growth buffer, Bulenox."""
    daily = load_plan(FIRMS_DIR / "tradeify" / "select_daily_50k.toml").value
    assert (daily("payout.window_cap_bps"), daily("payout.window_cap_from_payout")) == (20_000, 2)
    assert daily("payout.caps_cents") == (125_000,)
    assert load_plan(FIRMS_DIR / "mffu" / "rapid_50k.toml").value("payout.caps_cents") is None
    growth = load_plan(FIRMS_DIR / "tradeify" / "growth_50k.toml")
    assert growth.value("payout.buffer_rel_cents") == 10_001
    assert growth.leaf("payout.buffer_rel_cents").alt == (10_000,)
    for option in (1, 2):
        v = load_plan(FIRMS_DIR / "bulenox" / f"qualification_option{option}_50k.toml").value
        assert (v("payout.dead_days_after_payout"), v("payout.payout_day_counts")) == (2, False)


def test_source_quotes_are_short() -> None:
    for path in ALL_CONFIGS:
        for src in load_plan(path).sources.values():
            if src.quote is not None:
                assert len(src.quote.split()) <= 16, (path.name, src.key)


def test_cross_field_checks(tmp_path: Path) -> None:
    text = STD_TEXT.replace(
        'daily_loss_kind = { v = "none", src = "R072"', 'daily_loss_kind = { v = "soft", src = "R072"'
    )
    assert "eval.daily_loss_kind" in problems_of(tmp_path, text)
    with pytest.raises(PlanConfigError, match="needs win_day_cents"):
        with_values(load_plan(TOPSTEP_STD), {"payout.win_day_cents": "none"})
    with pytest.raises(PlanConfigError, match="ISO weekday"):
        with_values(load_plan(TOPSTEP_STD), {"payout.request_weekday": 8})


def test_file_name_must_match_plan_and_size(tmp_path: Path) -> None:
    assert "plan id + size" in problems_of(tmp_path, STD_TEXT, name="other_50k.toml")


def test_bad_toml_and_sources(tmp_path: Path) -> None:
    assert "not valid UTF-8 TOML" in problems_of(tmp_path, STD_TEXT + "\n[[[")
    text = STD_TEXT.replace('looked_up = "2026-10-10"', 'looked_up = "Oct 10"', 1)
    assert "looked_up must be an ISO date" in problems_of(tmp_path, text)


def test_with_values_is_a_new_frozen_plan() -> None:
    plan = load_plan(TOPSTEP_STD)
    new = with_values(plan, {"eval.profit_target_cents": 250_000})
    assert new.value("eval.profit_target_cents") == 250_000
    assert plan.value("eval.profit_target_cents") == 300_000
    assert new.overrides == (("eval.profit_target_cents", 250_000),)
    with pytest.raises(PlanConfigError, match="not a pfb-rules-1 field"):
        with_values(plan, {"eval.nope": 1})
    with pytest.raises(AttributeError):
        plan.firm = "x"  # type: ignore[misc]
    with pytest.raises(TypeError):
        plan.leaves["eval.profit_target_cents"] = None  # type: ignore[index]
