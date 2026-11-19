"""S4 walk-forward: fold report and selection on synthetic daily frames (known answers)."""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from pfb.analysis.walkforward import assert_open, fold_bounds, fold_report, select, sharpe_annual
from pfb.data.errors import ClosedWindowRefused


def frame(nets: list[int], start: date = date(2020, 1, 6), trades: list[int] | None = None) -> pl.DataFrame:
    cols: dict[str, object] = {
        "trade_date": [start + timedelta(days=i) for i in range(len(nets))],
        "net_cents": pl.Series(nets, dtype=pl.Int64),
    }
    if trades is not None:
        cols["trades"] = pl.Series(trades, dtype=pl.Int64)
    return pl.DataFrame(cols)


def wiggle(n: int, level: int, amp: int) -> list[int]:
    return [level + (amp if i % 2 else -amp) for i in range(n)]


def test_fold_bounds_remainder_to_last() -> None:
    assert fold_bounds(23, 5) == [(0, 4), (4, 8), (8, 12), (12, 16), (16, 23)]
    assert fold_bounds(10, 2) == [(0, 5), (5, 10)]
    for bad in (1, 0, True, 2.0):
        with pytest.raises(ValueError):
            fold_bounds(10, bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        fold_bounds(3, 4)


def test_constant_daily_net_gives_equal_folds_and_zero_spread() -> None:
    # alternate 300 / 100 so the sd is non-zero but every even-length fold has the same Sharpe
    out = fold_report(frame(wiggle(100, 200, 100), trades=[1] * 100), 5)
    folds = out["folds"]
    assert isinstance(folds, list) and len(folds) == 5
    sharpes = {round(f["sharpe_annual"], 12) for f in folds}
    assert sharpes == {round(2.0 * math.sqrt(252), 12)}  # mean 200 / pop sd 100
    assert {f["net_cents"] for f in folds} == {4000} and {f["days"] for f in folds} == {20}
    assert {f["trades"] for f in folds} == {20}
    assert out["sharpe_spread"] == pytest.approx(0.0, abs=1e-9) and out["positive_folds"] == 5
    assert folds[0]["first"] == "2020-01-06"
    assert folds[4]["last"] == (date(2020, 1, 6) + timedelta(99)).isoformat()


def test_flat_net_gives_none_sharpe() -> None:
    out = fold_report(frame([500] * 40), 4)
    assert all(f["sharpe_annual"] is None for f in out["folds"])  # type: ignore[union-attr]
    assert out["sharpe_spread"] is None and out["positive_folds"] == 4


def test_positive_then_negative_halves() -> None:
    nets = wiggle(60, 200, 100) + wiggle(60, -200, 100)
    out = fold_report(frame(nets), 4)
    signs = [f["sharpe_annual"] > 0 for f in out["folds"]]  # type: ignore[union-attr,operator]
    assert signs == [True, True, False, False]
    assert out["positive_folds"] == 2
    rolling = out["rolling"]
    anchored = out["anchored"]
    assert isinstance(rolling, dict) and isinstance(anchored, dict)
    # rolling pairs (0,1) agree, (1,2) disagree, (2,3) agree
    assert rolling["same_sign_share"] == pytest.approx(2 / 3)
    # anchored: in-sample 0 / 0..1 positive and 0..2 positive (mean 200/3 > 0) against oos 1, 2, 3
    assert anchored["same_sign_share"] == pytest.approx(1 / 3)
    assert anchored["mean_oos_over_is"] is not None and anchored["mean_oos_over_is"] < 0


def test_fold_nets_sum_to_run_net() -> None:
    nets = [((i * 7919) % 2001) - 950 for i in range(257)]
    out = fold_report(frame(nets), 6)
    assert sum(f["net_cents"] for f in out["folds"]) == sum(nets) == out["net_cents"]  # type: ignore[union-attr,misc]
    assert sum(f["days"] for f in out["folds"]) == 257  # type: ignore[union-attr,misc]


def test_report_without_dates_or_trades() -> None:
    out = fold_report(pl.DataFrame({"net_cents": pl.Series([1, 2, 3, 4], dtype=pl.Int64)}), 2)
    f0 = out["folds"][0]  # type: ignore[index]
    assert f0["first"] is None and f0["trades"] is None and f0["net_cents"] == 3


def test_bad_daily_refused() -> None:
    with pytest.raises(ValueError):
        fold_report(pl.DataFrame({"net_cents": [1.0, 2.0, 3.0, 4.0]}), 2)
    with pytest.raises(ValueError, match="increasing"):
        fold_report(frame([1, 2, 3, 4]).reverse(), 2)
    with pytest.raises(ValueError, match="trade_date"):
        fold_report(pl.DataFrame({"net_cents": pl.Series([1, 2, 3, 4], dtype=pl.Int64)}), 2, root="MNQ")


def test_fold_range_crossing_closed_window_refused() -> None:
    # 2026-04-27 .. 2026-05-16: the last fold crosses the 2026-05-09 embargo (and 2026-05-11)
    daily = frame(wiggle(20, 100, 50), start=date(2026, 4, 27))
    with pytest.raises(ClosedWindowRefused):
        fold_report(daily, 2, root="MNQ")
    with pytest.raises(ClosedWindowRefused):
        assert_open("MNQ", date(2026, 5, 6), date(2026, 5, 12))
    assert_open("MNQ", date(2023, 1, 3), date(2023, 12, 29))
    out = fold_report(frame(wiggle(20, 100, 50), start=date(2023, 1, 3)), 2, root="MNQ")
    assert out["n_folds"] == 2


def test_select_picks_the_config_better_in_every_fold() -> None:
    good = frame(wiggle(100, 300, 100))
    bad = frame(wiggle(100, 50, 100))
    for mode in ("anchored", "rolling"):
        out = select({"run_bad": bad, "run_good": good}, 5, mode)
        choices = out["choices"]
        assert isinstance(choices, list) and [c["chosen"] for c in choices] == ["run_good"] * 4
        assert out["days"] == 80 and out["net_cents"] == 300 * 80
        stitched = out["daily"]
        assert isinstance(stitched, pl.DataFrame) and stitched.height == 80
        assert stitched["trade_date"][0] == date(2020, 1, 6) + timedelta(days=20)


def test_select_follows_the_in_sample_leader() -> None:
    a = frame(wiggle(40, 300, 100) + wiggle(40, -300, 100))  # good early, bad late
    b = frame(wiggle(80, 100, 100))
    out = select({"a": a, "b": b}, 4, "rolling")
    # rolling: fold 0 picks a for fold 1; fold 1 picks a for fold 2 (a's fold 2 is bad); fold 2 picks b
    assert [c["chosen"] for c in out["choices"]] == ["a", "a", "b"]  # type: ignore[union-attr,index]
    assert out["net_cents"] == 300 * 20 - 300 * 20 + 100 * 20


def test_select_refuses_mismatched_runs_and_bad_mode() -> None:
    with pytest.raises(ValueError):
        select({"a": frame([1, 2, 3, 4]), "b": frame([1, 2, 3])}, 2)
    with pytest.raises(ValueError):
        select({"a": frame([1, 2, 3, 4]), "b": frame([1, 2, 3, 4], start=date(2021, 1, 4))}, 2)
    with pytest.raises(ValueError):
        select({"a": frame([1, 2, 3, 4])}, 2, "expanding")


def test_sharpe_matches_metrics_convention() -> None:
    assert sharpe_annual([1]) is None and sharpe_annual([5, 5]) is None
    assert sharpe_annual([300, 100]) == pytest.approx(2.0 * math.sqrt(252))
