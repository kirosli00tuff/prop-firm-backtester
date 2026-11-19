"""S5 neighbourhood: one-step neighbours on the grid and the plateau / spike verdict (known answers)."""

from __future__ import annotations

import pytest

from pfb.analysis.neighbourhood import PLATEAU_RATIO, SPIKE_RATIO, neighbours, verdict

GRID = {"buffer_ticks": [2, 4, 6], "lookback": [10.0, 20.0], "side": ["long", "short"], "mode": [1]}


def run(sharpe: float | None, net: int) -> dict[str, object]:
    return {"sharpe_annual": sharpe, "net_cents": net}


def test_neighbours_interior_and_edges() -> None:
    centre = {"buffer_ticks": 4, "lookback": 10.0, "side": "long", "mode": 1}
    assert neighbours(GRID, centre) == [
        {"buffer_ticks": 2, "lookback": 10.0, "side": "long", "mode": 1},
        {"buffer_ticks": 6, "lookback": 10.0, "side": "long", "mode": 1},
        {"buffer_ticks": 4, "lookback": 20.0, "side": "long", "mode": 1},
    ]
    edge = {"buffer_ticks": 6, "lookback": 20.0, "side": "short", "mode": 1}
    assert neighbours(GRID, edge) == [
        {"buffer_ticks": 4, "lookback": 20.0, "side": "short", "mode": 1},
        {"buffer_ticks": 6, "lookback": 10.0, "side": "short", "mode": 1},
    ]


def test_neighbours_skip_non_monotone_and_bool_lists() -> None:
    grid = {"a": [3, 1, 2], "b": [True, False], "c": [1, 1, 2], "d": [0.5, 1.5]}
    centre = {"a": 1, "b": True, "c": 1, "d": 0.5}
    assert neighbours(grid, centre) == [{"a": 1, "b": True, "c": 1, "d": 1.5}]


def test_neighbours_refuse_foreign_centre() -> None:
    with pytest.raises(ValueError, match="not a point"):
        neighbours(GRID, {"buffer_ticks": 5, "lookback": 10.0, "side": "long", "mode": 1})
    with pytest.raises(ValueError, match="exactly the grid"):
        neighbours(GRID, {"buffer_ticks": 4})


def test_neighbours_do_not_mutate_inputs() -> None:
    centre = {"buffer_ticks": 4, "lookback": 10.0, "side": "long", "mode": 1}
    before = dict(centre)
    out = neighbours(GRID, centre)
    out[0]["buffer_ticks"] = 99
    assert centre == before and GRID["buffer_ticks"] == [2, 4, 6]


def test_flat_response_is_plateau() -> None:
    out = verdict(run(1.2, 50_000), [run(1.2, 50_000), run(1.1, 45_000), run(1.25, 52_000)])
    assert out["verdict"] == "plateau"
    assert out["ratio"] == pytest.approx((1.2 + 1.1 + 1.25) / 3 / 1.2)
    assert out["n_neighbours"] == 3 and out["n_same_sign"] == 3 and out["n_other_sign"] == 0
    assert out["thresholds"] == {"plateau_ratio": 0.7, "spike_ratio": 0.5, "spike_other_sign_share": 0.5}


def test_lone_peak_is_spike() -> None:
    out = verdict(run(2.0, 80_000), [run(0.2, 5_000), run(-0.3, -9_000), run(0.1, 2_000), run(0.0, -100)])
    assert out["verdict"] == "spike"
    assert out["ratio"] == pytest.approx(0.0)


def test_half_other_sign_is_spike_even_with_high_ratio() -> None:
    out = verdict(run(1.0, 10_000), [run(2.0, 30_000), run(1.5, -1)])
    assert out["ratio"] == pytest.approx(1.75) and out["verdict"] == "spike"


def test_mixed_between_thresholds_and_boundaries() -> None:
    assert verdict(run(1.0, 10), [run(0.6, 5), run(0.6, 5)])["verdict"] == "mixed"
    assert verdict(run(1.0, 10), [run(0.5, 5), run(0.5, 5)])["verdict"] == "mixed"  # r = 0.5 is not < 0.5
    assert verdict(run(1.0, 10), [run(0.7, 5), run(0.7, 5)])["verdict"] == "plateau"  # r = 0.7 is >= 0.7
    assert verdict(run(1.0, 10), [run(0.49, 5), run(0.49, 5)])["verdict"] == "spike"
    # a zero net blocks plateau
    assert verdict(run(1.0, 10), [run(0.9, 5), run(0.9, 0)])["verdict"] == "mixed"
    assert (PLATEAU_RATIO, SPIKE_RATIO) == (0.7, 0.5)


def test_not_applicable_cases() -> None:
    assert verdict(run(1.0, 10), [])["verdict"] == "n/a"
    assert verdict(run(0.0, 0), [run(1.0, 10)])["verdict"] == "n/a"
    assert verdict(run(-0.5, -10), [run(1.0, 10)])["verdict"] == "n/a"
    out = verdict(run(None, 0), [run(1.0, 10)])
    assert out["verdict"] == "n/a" and out["ratio"] is None


def test_neighbour_without_sharpe_counts_as_zero() -> None:
    out = verdict(run(1.0, 10), [run(None, 0), run(1.0, 10)])
    assert out["ratio"] == pytest.approx(0.5) and out["verdict"] == "mixed"


def test_bad_summaries_refused() -> None:
    with pytest.raises(ValueError):
        verdict(run(1.0, 10), [{"sharpe_annual": "1.0", "net_cents": 5}])
    with pytest.raises(ValueError):
        verdict(run(1.0, 10), [{"sharpe_annual": 1.0, "net_cents": 5.0}])
