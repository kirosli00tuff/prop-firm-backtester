"""pfb.dash.hypgen builds a schema "pfb.hypothesis/2" file that the registry's own
parser accepts, from the upload, its SPEC, the picked windows and plans and the form. Pure: nothing is read
or run."""

from __future__ import annotations

import ast
import tomllib
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from pfb.dash import hypgen
from pfb.dash.actions import StrategyState, read_spec
from pfb.dash.hypgen import (
    HypForm,
    HypGenError,
    PowerSource,
    build,
    hypothesis_id,
    missing_fields,
    power_preview,
    power_source,
    shape_error,
)
from pfb.registry.hypfile import Hypothesis, parse_hypothesis
from pfb.registry.inputs import ID_RE

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "strategies" / "_template.py"
SHA = "ab" * 32
REL = f"strategies/uploads/{SHA}.py"
WINDOW_A = {"root": "MNQ", "first": "2025-04-01", "last": "2025-09-30", "status": "open"}
WINDOW_B = {"root": "MNQ", "first": "2025-10-01", "last": "2026-05-08", "status": "open"}
PLANS = ["mffu/rapid_50k", "topstep/combine_xfa_standard_50k"]
SOURCE = PowerSource(
    "synthetic-x", trades=21, days=21, trade_sd_cents=2170, trade_skew=0.42, trade_kurtosis=1.62
)
NOW = datetime(2026, 11, 17, 12, 30, 5, tzinfo=UTC)
HUGE_EDGE = 10**7


def _state(spec: dict[str, Any] | None = None) -> StrategyState:
    found = spec if spec is not None else read_spec(TEMPLATE.read_text())
    return StrategyState(rel_path=REL, sha256=SHA, spec=found, checks=(), passed=True)


def _form(**changes: Any) -> HypForm:
    base = HypForm(
        hypothesis_id=hypothesis_id(SHA, NOW),
        statement='Breakout "edge" on MNQ\nsecond line \\ done, café',
        target_edge_cents=3000,
        event_costs="off",
        event_stance="the strategy trades at 08:45 CT, after the 07:30 CT releases",
        full_grid=False,
        power=SOURCE,
    )
    return replace(base, **changes)


def _parsed(text: str) -> Hypothesis:
    hyp = parse_hypothesis(text.encode("utf-8"), "test")
    assert isinstance(hyp, Hypothesis)
    return hyp


def test_build_from_the_template_passes_the_registry_parser_with_spec_defaults() -> None:
    # Arrange
    spec = read_spec(TEMPLATE.read_text())
    assert spec is not None
    # Act
    text = build(_state(), [WINDOW_A], PLANS, _form())
    hyp = _parsed(text)
    # Assert
    assert shape_error(text, "test") is None
    assert hyp.id == "h_abababab_20261117T123005Z"
    assert (hyp.strategy, hyp.strategy_sha256, hyp.purpose) == (REL, SHA, "research")
    assert hyp.roots == ("MNQ",) or list(hyp.roots) == ["MNQ"]
    assert (hyp.bar_minutes, hyp.session, hyp.budget) == (1, "globex", 1)
    assert {k: list(v) for k, v in hyp.grid.items()} == {k: [p["default"]] for k, p in spec["params"].items()}
    raw = tomllib.loads(text)
    assert raw["window"] == [{"first": date(2025, 4, 1), "last": date(2025, 9, 30)}]
    assert raw["cost"] == {"model": "book", "fees": "mffu", "event_costs": "off",
                           "event_stance": _form().event_stance}  # fmt: skip
    assert raw["power"]["sd_source"] == "run:synthetic-x"
    assert raw["power"]["trade_sd_cents"] == SOURCE.trade_sd_cents
    assert raw["suite"]["sizing_plan"] == PLANS[0]
    assert raw["suite"]["neighbourhood"] is False
    assert raw["secondary"]["prop_plans"] == PLANS
    assert raw["statement"] == _form().statement.strip()
    assert raw["no_signal_reason"] == ""


def test_full_grid_and_two_windows_set_the_budget_and_window_order() -> None:
    # Arrange
    spec = read_spec(TEMPLATE.read_text())
    assert spec is not None
    points = 1
    for item in spec["params"].values():
        points *= len(item["grid"])
    # Act
    text = build(_state(), [WINDOW_B, WINDOW_A], PLANS, _form(full_grid=True))
    hyp = _parsed(text)
    # Assert
    assert hyp.budget == points * 2
    assert {k: list(v) for k, v in hyp.grid.items()} == {
        k: list(p["grid"]) for k, p in spec["params"].items()
    }
    assert [w["first"] for w in tomllib.loads(text)["window"]] == [date(2025, 4, 1), date(2025, 10, 1)]


def test_no_plan_gives_the_default_fees_and_no_sizing_plan() -> None:
    # Act
    raw = tomllib.loads(build(_state(), [WINDOW_A], [], _form()))
    # Assert
    assert raw["cost"]["fees"] == hypgen.DEFAULT_FEES
    assert raw["suite"]["sizing_plan"] == ""
    assert raw["secondary"]["prop_plans"] == []


def test_event_costs_off_with_a_blank_stance_is_refused_by_the_shape_check() -> None:
    # Act
    error = shape_error(build(_state(), [WINDOW_A], PLANS, _form(event_stance="  ")), "test")
    # Assert
    assert error is not None and error.startswith("HypothesisInvalid")


def test_event_costs_on_may_leave_the_stance_empty() -> None:
    # Act
    raw = tomllib.loads(build(_state(), [WINDOW_A], PLANS, _form(event_costs="on", event_stance="")))
    # Assert
    assert raw["cost"]["event_costs"] == "on"
    assert raw["cost"]["event_stance"] == ""


@pytest.mark.parametrize(
    ("windows", "spec", "form"),
    [
        ([], None, {}),
        ([{"first": "2025-04-01", "last": None}], None, {}),
        ([WINDOW_A], {"root": "MNQ"}, {}),
        ([WINDOW_A], None, {"event_costs": "maybe"}),
    ],
)
def test_bad_inputs_raise_hypgen_error(
    windows: list[dict[str, Any]], spec: Any, form: dict[str, Any]
) -> None:
    # Act / Assert
    with pytest.raises(HypGenError):
        build(_state(spec), windows, PLANS, _form(**form))


def test_trades_per_year_is_never_above_the_source_runs_rate() -> None:
    # Arrange / Act / Assert: registration refuses a stated rate above trades x 252 / days
    for trades in range(2, 60, 7):
        for days in range(2, 400, 13):
            source = replace(SOURCE, trades=trades, days=days)
            exact = Fraction(trades * 252, days)
            stated = source.trades_per_year()
            assert Fraction(stated) <= exact
            assert exact - Fraction(stated) < Fraction(2, 10)  # one step down when the float rounds up


def test_power_source_reads_an_ok_runs_realized_figures() -> None:
    # Arrange: the shape of a synthetic result record (metrics.suite.power)
    power = {"trades": 21, "days": 21, "trade_sd_cents": 2170, "trade_skew": 0.4272, "trade_kurtosis": 1.6249}
    record = {"status": "ok", "metrics": {"suite": {"power": power}}}
    # Act / Assert
    assert power_source("r1", record) == PowerSource("r1", 21, 21, 2170, 0.4272, 1.6249)
    assert power_source("r1", {**record, "status": "failed"}) is None
    assert power_source("r1", None) is None
    assert (
        power_source("r1", {"status": "ok", "metrics": {"suite": {"power": {**power, "trades": 1}}}}) is None
    )
    assert power_source("r1", {"status": "ok", "metrics": {"suite": {"power": None}}}) is None


def test_hypothesis_id_is_a_valid_id_in_utc() -> None:
    # Arrange
    local = NOW.astimezone(timezone(timedelta(hours=-8)))
    # Act
    hyp_id = hypothesis_id(SHA, local)
    # Assert
    assert hyp_id == "h_abababab_20261117T123005Z"
    assert ID_RE.fullmatch(hyp_id)
    assert hypgen.hypothesis_rel(hyp_id) == f"hypotheses/uploads/{hyp_id}.toml"


def test_power_preview_matches_the_registration_rule() -> None:
    # Act
    low = power_preview(_form(target_edge_cents=1), [WINDOW_A, WINDOW_B], 1002)
    high = power_preview(_form(target_edge_cents=HUGE_EDGE), [WINDOW_A], 1002)
    # Assert
    assert [(r.first, r.n_used) for r in low] == [(date(2025, 4, 1), 1002), (date(2025, 10, 1), 1002)]
    assert not any(r.passed for r in low)
    assert high[0].passed is (high[0].min_edge_cents is not None)


def test_missing_fields_names_what_the_user_still_types() -> None:
    # Act / Assert
    assert missing_fields("", 0, "off", "") == ["statement", "target_edge_cents", "event_stance"]
    assert missing_fields("x", 10, "on", "") == []
    assert missing_fields("x", 10, "off", " ") == ["event_stance"]


def test_hypgen_is_pure() -> None:
    # Arrange
    tree = ast.parse(Path(hypgen.__file__).read_text())
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    called = {
        n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
    }
    # Assert
    assert not {m for m in imported if m.split(".")[0] in {"streamlit", "subprocess", "os"}}
    assert "pfb.dash.actions" not in imported
    assert not called & {"open", "write_text", "write_bytes", "read_text", "read_bytes", "mkdir", "now"}
