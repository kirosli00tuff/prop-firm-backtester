"""SuiteSpec: the [suite] table of docs/TRIALS.md section 2, validated on construction."""

from __future__ import annotations

import pytest

from pfb.analysis.spec import SUITE_KEYS, SuiteSpec

TABLE: dict[str, object] = {
    "cost_stress_ticks": [1, 2],
    "baseline_draws": 200,
    "baseline_seed": 7,
    "walk_forward_folds": 5,
    "neighbourhood": True,
    "sizing_plan": "topstep/combine_xfa_standard_50k",
}


def test_defaults_match_the_seam() -> None:
    spec = SuiteSpec()
    assert spec.cost_stress_ticks == (1, 2)
    assert (spec.baseline_draws, spec.baseline_seed, spec.walk_forward_folds) == (200, 7, 5)
    assert spec.neighbourhood is False and spec.sizing_plan == ""


def test_from_mapping_round_trips_through_to_json() -> None:
    spec = SuiteSpec.from_mapping(TABLE)
    assert spec.to_json() == TABLE
    assert list(spec.to_json()) == list(SUITE_KEYS)


@pytest.mark.parametrize("drop", SUITE_KEYS)
def test_a_missing_key_is_refused(drop: str) -> None:
    table = {k: v for k, v in TABLE.items() if k != drop}
    with pytest.raises(ValueError, match="missing"):
        SuiteSpec.from_mapping(table)


def test_an_extra_key_is_refused() -> None:
    with pytest.raises(ValueError, match="extra"):
        SuiteSpec.from_mapping(TABLE | {"cost_model": "book"})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("baseline_draws", 199),  # fewer than 200 draws is refused
        ("baseline_draws", 1),
        ("baseline_draws", True),
        ("baseline_draws", 200.0),
        ("walk_forward_folds", 1),
        ("walk_forward_folds", "5"),
        ("baseline_seed", -1),
        ("cost_stress_ticks", [0]),
        ("cost_stress_ticks", [1, 1]),
        ("cost_stress_ticks", [1.5]),
        ("cost_stress_ticks", "1"),
        ("neighbourhood", 1),
        ("sizing_plan", "topstep"),
        ("sizing_plan", 3),
    ],
)
def test_bad_values_are_refused(key: str, value: object) -> None:
    with pytest.raises(ValueError):
        SuiteSpec.from_mapping(TABLE | {key: value})


def test_off_turns_every_part_off() -> None:
    off = SuiteSpec.off()
    assert off.cost_stress_ticks == () and off.baseline_draws == 0 and off.walk_forward_folds == 0
    assert off.neighbourhood is False and off.sizing_plan == ""


def test_zero_draws_and_zero_folds_mean_off_and_are_accepted() -> None:
    off = {"baseline_draws": 0, "walk_forward_folds": 0, "cost_stress_ticks": []}
    spec = SuiteSpec.from_mapping(TABLE | off)
    assert spec.baseline_draws == 0 and spec.walk_forward_folds == 0 and spec.cost_stress_ticks == ()


def test_spec_is_frozen() -> None:
    spec = SuiteSpec()
    with pytest.raises(AttributeError):
        spec.baseline_draws = 500  # type: ignore[misc]
