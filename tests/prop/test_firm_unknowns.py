"""Missing firm details as explicit unknowns with a reason.

Topstep reset credits and Back2Funded, Tradeify funded reset, Lucid funded DLL as a share of peak profit:
each is in the plan files it applies to, the loader refuses a silent default, a prop result that depends on
one names it, and the Monte Carlo figures of the 13 plans are unchanged by them (synthetic day paths).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from _synth import FIRMS_DIR, REPO, random_days

from pfb.prop.config import NOT_MODELLED, PlanConfigError, load_plan, load_versions, with_values
from pfb.prop.matrix import OUTPUTS, McSettings, compact_estimate, evaluate_plan, prop_row
from pfb.prop.mc import run_mc
from pfb.prop.types import RuleUnknown

FIXTURE = Path(__file__).parent / "fixtures" / "ka_static_50k.toml"
LUCIDFLEX = FIRMS_DIR / "lucid" / "lucidflex_50k.toml"
LUCIDFLEX_V1 = FIRMS_DIR / "lucid" / "versions" / "2026-08-31" / "lucidflex_50k.toml"
SELECT = [FIRMS_DIR / "tradeify" / f"{n}_50k.toml" for n in ("select_flex", "select_daily")]
GROWTH = FIRMS_DIR / "tradeify" / "growth_50k.toml"
TOPSTEP = [FIRMS_DIR / "topstep" / f"combine_xfa_{n}_50k.toml" for n in ("standard", "consistency")]
PLAN_FILES = sorted(FIRMS_DIR.glob("*/*.toml"))
RESET = "funded.funded_reset_cents"
DLL = "funded.daily_loss_cents"
CREDIT = "fees.rebill_adds_reset_credit"


def _key(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"


def _days():  # the synthetic day path the pinned figures were taken on
    return random_days(np.random.default_rng(2909), 300, scale=90_000, drift=15_000, max_minis=2, p_big=0.0)


# ---------------------------------------------------------------- the four details in the files ----
@pytest.mark.parametrize("path", PLAN_FILES, ids=_key)
def test_every_plan_states_its_funded_reset(path: Path) -> None:
    leaf = load_plan(path).leaf(RESET)
    assert leaf.is_unknown or leaf.value is None or leaf.value > 0
    assert not leaf.assume  # a funded reset is never assumed available


@pytest.mark.parametrize("path", [LUCIDFLEX, LUCIDFLEX_V1], ids=_key)
def test_lucidflex_funded_dll_is_an_assumed_unknown(path: Path) -> None:
    plan = load_plan(path)
    leaf = plan.leaf(DLL)
    assert leaf.is_unknown and leaf.assume and leaf.alt == (120_000,)
    assert leaf.unknown is not None and leaf.unknown.startswith("unpublished on the pages read, 2026-11-16")
    assert plan.value(DLL) == 120_000  # the assumption the run reads: the value the file held before
    assert plan.value("funded.daily_loss_kind") == "soft"


def test_lucidpro_keeps_its_published_dll() -> None:
    leaf = load_plan(FIRMS_DIR / "lucid" / "lucidpro_50k.toml").leaf(DLL)
    assert not leaf.is_unknown and leaf.value == 120_000 and leaf.kind == "conservative"


@pytest.mark.parametrize("path", SELECT, ids=_key)
def test_select_funded_reset_is_unknown_without_alt(path: Path) -> None:
    plan = load_plan(path)
    leaf = plan.leaf(RESET)
    assert leaf.unknown is not None and leaf.unknown.startswith("unpublished on the pages read, 2026-11-17")
    assert leaf.alt == () and not leaf.assume
    with pytest.raises(RuleUnknown, match=RESET):
        plan.value(RESET)


def test_growth_funded_reset_is_unknown_with_the_a2_reading() -> None:
    leaf = load_plan(GROWTH).leaf(RESET)
    assert leaf.unknown is not None and leaf.unknown.startswith("unpublished on the pages read, 2026-11-17")
    assert leaf.alt == ("none",) and not leaf.assume


@pytest.mark.parametrize("path", TOPSTEP, ids=_key)
def test_topstep_back2funded_and_credits_are_published_and_not_modelled(path: Path) -> None:
    plan = load_plan(path)
    assert plan.value(RESET) == 59_900 and plan.leaf(RESET).src == "R106"
    assert plan.value(CREDIT) is True
    notes = plan.unknown_notes()
    assert [n.split(" is ")[0] for n in notes] == [
        f"rule not modelled: {RESET}",
        f"rule not modelled: {CREDIT}",
    ]


def test_fixture_plan_has_no_unknown_notes() -> None:
    plan = load_plan(FIXTURE)
    assert plan.value(RESET) is None and plan.value(CREDIT) is False and plan.unknown_notes() == []


# ------------------------------------------------------------------- no silent default ----
def _write(tmp_path: Path, text: str) -> Path:
    out = tmp_path / FIXTURE.name
    out.write_text(text)
    return out


def _with_reset_leaf(leaf: str) -> str:
    text = FIXTURE.read_text()
    old = 'funded_reset_cents = { v = "none", src = "S01" }'
    assert text.count(old) == 1
    return text.replace(old, f"funded_reset_cents = {leaf}")


def test_loader_refuses_a_plan_without_the_field(tmp_path: Path) -> None:
    text = FIXTURE.read_text().replace('funded_reset_cents = { v = "none", src = "S01" }\n', "")
    with pytest.raises(PlanConfigError, match=f"{RESET}: missing"):
        load_plan(_write(tmp_path, text))


@pytest.mark.parametrize(
    ("leaf", "problem"),
    [
        ('{ unknown = "x", assume = true }', "assume needs an alt"),
        ('{ unknown = "x", alt = [59900], assume = true }', "a funded reset is not modelled"),
        ('{ v = 59900, src = "S01", assume = true }', "assume is only for an unknown leaf"),
        ('{ unknown = "x", alt = [59900], assume = "yes" }', "assume must be true or false"),
    ],
)
def test_loader_refuses_an_assumed_default(tmp_path: Path, leaf: str, problem: str) -> None:
    with pytest.raises(PlanConfigError, match=problem):
        load_plan(_write(tmp_path, _with_reset_leaf(leaf)))


def test_loader_refuses_an_assumed_absent_daily_loss_limit(tmp_path: Path) -> None:
    text = FIXTURE.read_text()
    old = '[funded]\nstart_balance_cents = { v = 5000000, src = "S01" }'
    assert text.count(old) == 1
    funded = text.split("[funded]", 1)[1]
    line = 'daily_loss_cents = { v = "none", src = "S01" }'
    assert funded.count(line) == 1
    text = (
        text.split("[funded]", 1)[0]
        + "[funded]"
        + funded.replace(line, 'daily_loss_cents = { unknown = "x", alt = ["none"], assume = true }')
    )
    with pytest.raises(PlanConfigError, match="never assumed absent"):
        load_plan(_write(tmp_path, text))


def test_an_unknown_without_assume_still_raises(tmp_path: Path) -> None:
    plan = load_plan(_write(tmp_path, _with_reset_leaf('{ unknown = "x", alt = ["none"] }')))
    with pytest.raises(RuleUnknown):
        plan.value(RESET)
    assert plan.unknown_notes() == [f"rule not modelled: {RESET} is unknown (x); {NOT_MODELLED[RESET][1]}"]


# --------------------------------------------------------------- results name the unknowns ----
SETTINGS = McSettings(n_paths=200, seed=29, kind="synthetic", horizon_eval=120, horizon_funded=120)


def _record(path: Path) -> dict:
    record, _ = evaluate_plan(path, _days(), SETTINGS, repo_root=REPO, cap_check=lambda d, p: [])
    return record


def test_lucidflex_result_names_the_assumed_dll_and_the_funded_reset() -> None:
    warnings = _record(LUCIDFLEX)["warnings"]
    assert any(w.startswith(f"rule unknown, assumed: {DLL} = 120000 is an assumption") for w in warnings)
    assert any(w.startswith(f"rule not modelled: {RESET} is unknown (") for w in warnings)


def test_select_result_names_the_unknown_funded_reset_and_is_computed() -> None:
    record = _record(SELECT[0])
    assert any(
        w.startswith(f"rule not modelled: {RESET} is unknown (unpublished") for w in record["warnings"]
    )
    assert record["outputs"]["funded_ev"]["value"] is not None  # computed without the feature, and says so
    row = prop_row("tradeify", "select_flex_50k", "ok", None, record)
    # the trial's prop row has no place for it yet
    assert "warnings" not in row


def test_the_assumed_dll_is_the_value_the_run_reads() -> None:
    plan, days = load_versions(LUCIDFLEX).newest, _days()

    def figures(p: object) -> dict:
        r = run_mc(p, days, n_paths=200, seed=29, kind="synthetic", horizon_eval=120, horizon_funded=120)
        return {n: compact_estimate(getattr(r, n)) for n in OUTPUTS}

    assumed = figures(plan)
    assert assumed == figures(with_values(plan, {DLL: 120_000}))
    assert assumed != figures(with_values(plan, {DLL: 60_000}))  # the DLL binds on this path


# ------------------------------------------------- figures identical to the earlier pin ----
# sha256 of {outputs, counts, eval_ends, funded_ends} per plan on _days(), n_paths 300, seed 29, horizons 120,
# pinned before the unknowns were added to any plan file.
HEAD_FIGURES = {
    "bulenox/qualification_option1_50k.toml": "ec6d99ecdd2d7fb138de20f2073b17ad8c822699b092e0c12ef0b4b06f4f3ddd",  # noqa: E501
    "bulenox/qualification_option2_50k.toml": "5832c48be56b59b923fd6b8f75c663ede397cee4dfe6317111cec0e7205d48b7",  # noqa: E501
    "lucid/lucidflex_50k.toml": "f284d840105e48d6639ea7c2a4deba89e3d107299d24f6113934b40dbc5a3ece",
    "lucid/lucidpro_50k.toml": "643fbecf980b503bf1aff67378638e67abcfbff176fb577c3ca397339c9881be",
    "mffu/builder_50k.toml": "be23cadd889f26149a35bb0fafcac51c8fd4a0d480f56a02e83d260b4dee72fd",
    "mffu/pro_50k.toml": "faca01ddb09de41c6479a1ad5333e3454ba6e4e1c91536cdd0fa250a988abd76",
    "mffu/rapid_50k.toml": "1626ce480c996a4940578e33e8b4b167f1634f5fb8dc4fe727c3b0488e771b76",
    "mffu/rapid_eod_50k.toml": "refused RuleUnknown: eval.time_limit_calendar_days",
    "topstep/combine_xfa_consistency_50k.toml": "8cd83a2c8524a420aff5be85c00456b72d873b299e6b0c1a51c0d89bd602d4f8",  # noqa: E501
    "topstep/combine_xfa_standard_50k.toml": "2ecd1172ae0065e4c14d70a8b52b31b2f7375b4543d2ce69c1a8e6f1d9dafd70",  # noqa: E501
    "tradeify/growth_50k.toml": "45069b9fefec7ad89ad3ad25cdbb063201da50f1024a07fb86a4b8e15b2e4ca4",
    "tradeify/select_daily_50k.toml": "e04f63c01727f9cfc288ff26698d2fa043231e1425c92a83e2304bca139e547c",
    "tradeify/select_flex_50k.toml": "3becb3b161054b8fb270300a48bc1824581900625d1b13de5fbeddc67f949390",
}


def test_monte_carlo_figures_equal_head() -> None:
    days, got = _days(), {}
    for path in PLAN_FILES:
        plan = load_versions(path).newest
        try:
            r = run_mc(
                plan, days, n_paths=300, seed=29, kind="synthetic", horizon_eval=120, horizon_funded=120
            )
        except RuleUnknown as exc:
            got[_key(path)] = f"refused RuleUnknown: {str(exc).split(':', 1)[0]}"
            continue
        fig = {"outputs": {n: compact_estimate(getattr(r, n)) for n in OUTPUTS}, "counts": r.counts,
               "eval_ends": r.eval_ends, "funded_ends": r.funded_ends}  # fmt: skip
        got[_key(path)] = hashlib.sha256(json.dumps(fig, sort_keys=True, default=str).encode()).hexdigest()
    assert got == HEAD_FIGURES
