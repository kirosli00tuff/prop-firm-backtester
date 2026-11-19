"""A known suite error after begin makes that part null with a warning; the run
itself stays ok. Synthetic bars only; nothing is registered."""

from __future__ import annotations

from typing import Any, cast

import pytest
from suite_planting import REFERENCE

import pfb.analysis.inrun as inrun
from pfb.analysis.baseline import BaselineError
from pfb.analysis.inrun import SUITE_KEYS, SUITE_PART_ERRORS, suite_error_warning
from pfb.analysis.power import PowerInputError
from pfb.analysis.spec import SuiteError, SuiteSpec
from pfb.analysis.walkforward import FoldError, fold_bounds
from pfb.engine.run import RunResult, run_strategy

TOO_MANY_FOLDS = 10**6


def _run(spec: SuiteSpec, n_trials: int | None = 1000) -> RunResult:
    return run_strategy(REFERENCE, source="synthetic", cost_model="fixed:1", seed=3, n_trials=n_trials,
                        suite=spec)  # fmt: skip


def _suite(result: RunResult) -> dict[str, Any]:
    return cast(dict[str, Any], result.summary["suite"])


def _warnings(result: RunResult) -> list[str]:
    return cast(list[str], result.summary["warnings"])


def _suite_warnings(result: RunResult) -> list[str]:
    return [w for w in _warnings(result) if w.startswith("suite: ")]


def _others_present(result: RunResult, null_part: str) -> None:
    suite = _suite(result)
    assert set(suite) == {*SUITE_KEYS, "spec"}
    assert all(suite[k] is not None for k in SUITE_KEYS if k != null_part)


def test_the_fold_error_is_a_suite_error_and_a_value_error() -> None:
    with pytest.raises(FoldError, match="cannot make 5 folds") as caught:
        fold_bounds(3, 5)
    assert isinstance(caught.value, SuiteError) and isinstance(caught.value, ValueError)
    assert issubclass(PowerInputError, ValueError)
    assert set(SUITE_PART_ERRORS) == {SuiteError, BaselineError, FoldError, PowerInputError}


def test_fewer_days_than_folds_makes_walk_forward_null_with_a_warning() -> None:
    # Arrange
    plain = _run(SuiteSpec())
    # Act
    result = _run(SuiteSpec(walk_forward_folds=TOO_MANY_FOLDS))
    # Assert
    assert _suite(result)["walk_forward"] is None
    _others_present(result, "walk_forward")
    days = result.daily.height
    assert _suite_warnings(result) == [
        f"suite: walk_forward not computed: FoldError: {days} day(s) cannot make {TOO_MANY_FOLDS} folds"
    ]
    assert _warnings(result) == [*_warnings(plain), *_suite_warnings(result)]
    assert result.trades.equals(plain.trades)  # the run itself is not affected


def test_a_missing_sizing_plan_makes_sizing_null_with_a_warning() -> None:
    result = _run(SuiteSpec(cost_stress_ticks=(), baseline_draws=0, walk_forward_folds=0,
                            sizing_plan="nofirm/noplan"))  # fmt: skip
    assert _suite(result)["sizing"] is None
    [warning] = _suite_warnings(result)
    assert warning.startswith("suite: sizing not computed: SuiteError: sizing plan 'nofirm/noplan': no file")


def test_a_baseline_error_makes_baseline_null_and_drops_its_draws(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object, **kwargs: object) -> object:
        raise BaselineError("1 trade(s) have no eligible entry bar in mode tod")

    monkeypatch.setattr(inrun.B, "random_baseline", broken)
    result = _run(SuiteSpec())
    assert _suite(result)["baseline"] is None
    _others_present(result, "baseline")
    assert "baseline_draws" not in result.extras and "rolling_sharpe" in result.extras
    assert _suite_warnings(result) == [
        "suite: baseline not computed: BaselineError: 1 trade(s) have no eligible entry bar in mode tod"
    ]


@pytest.mark.parametrize(
    ("part", "target", "error"),
    [
        ("cost_stress", "cost_stress", SuiteError("cost stress broke")),
        ("breakdowns", "breakdowns", SuiteError("breakdowns broke")),
        ("sizing", "sizing_table", SuiteError("sizing broke")),
    ],
)
def test_a_suite_error_in_a_part_makes_it_null(
    monkeypatch: pytest.MonkeyPatch, part: str, target: str, error: Exception
) -> None:
    def broken(*args: object, **kwargs: object) -> object:
        raise error

    monkeypatch.setattr(inrun, target, broken)
    result = _run(SuiteSpec())
    assert _suite(result)[part] is None
    _others_present(result, part)
    assert _suite_warnings(result) == [f"suite: {part} not computed: SuiteError: {error}"]
    if part == "breakdowns":
        assert "rolling_sharpe" not in result.extras


def test_a_power_input_error_makes_power_null(monkeypatch: pytest.MonkeyPatch) -> None:
    import pfb.analysis.power as power

    def broken(**kwargs: object) -> object:
        raise PowerInputError("trade_sd_cents 0 must be an int > 0")

    monkeypatch.setattr(power, "min_edge", broken)
    result = _run(SuiteSpec())
    assert _suite(result)["power"] is None
    _others_present(result, "power")
    assert _suite_warnings(result) == [
        "suite: power not computed: PowerInputError: trade_sd_cents 0 must be an int > 0"
    ]


def test_two_parts_failing_give_two_warnings_in_suite_order(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object, **kwargs: object) -> object:
        raise SuiteError("no")

    monkeypatch.setattr(inrun, "cost_stress", broken)
    result = _run(SuiteSpec(walk_forward_folds=TOO_MANY_FOLDS))
    parts = [w.split(" ")[1] for w in _suite_warnings(result)]
    assert parts == ["cost_stress", "walk_forward"]


def test_an_unknown_error_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object, **kwargs: object) -> object:
        raise ZeroDivisionError("a bug, not a suite error")

    monkeypatch.setattr(inrun, "cost_stress", broken)
    with pytest.raises(ZeroDivisionError):
        _run(SuiteSpec())


def test_a_suite_without_errors_adds_no_warning() -> None:
    result = _run(SuiteSpec())
    assert _suite_warnings(result) == []


def test_the_warning_text_names_the_part_and_the_error() -> None:
    assert (
        suite_error_warning("power", PowerInputError("x")) == "suite: power not computed: PowerInputError: x"
    )
