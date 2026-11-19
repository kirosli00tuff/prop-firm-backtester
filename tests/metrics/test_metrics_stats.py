"""pfb.metrics.stats: pinned known answers and hand-computed cases."""

import math
from datetime import date
from fractions import Fraction

import numpy as np
import pytest

from pfb.metrics import stats as S

# --------------------------------------------------------------------------- normal distribution


def test_norm_cdf_at_zero_is_one_half() -> None:
    assert S.norm_cdf(0.0) == pytest.approx(0.5, abs=1e-12)


def test_norm_ppf_known_points() -> None:
    assert S.norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
    assert S.norm_ppf(0.975) == pytest.approx(1.959963985, abs=1e-6)


@pytest.mark.parametrize("p", [0.01, 0.2, 0.5, 0.8, 0.99])
def test_norm_cdf_inverts_norm_ppf(p: float) -> None:
    assert S.norm_cdf(S.norm_ppf(p)) == pytest.approx(p, abs=1e-8)


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.5])
def test_norm_ppf_outside_open_unit_interval_raises(p: float) -> None:
    with pytest.raises(ValueError):
        S.norm_ppf(p)


# --------------------------------------------------------------------------- expected max Sharpe, bracket


def test_expected_max_sharpe_pinned_answers() -> None:
    assert S.expected_max_sharpe(1, 0.25) == 0.0
    assert S.expected_max_sharpe(50, 0.0) == 0.0
    assert S.expected_max_sharpe(10, 1.0) == pytest.approx(1.5748, abs=1e-3)
    assert S.expected_max_sharpe(20, 0.04) == pytest.approx(0.380, abs=0.005)


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (2, 0.5197553440822881),
        (10, 1.5745983018549996),
        (100, 2.530602894919986),
        (300, 2.8955015064655756),
        (1000, 3.255121510917663),
    ],
)
def test_expected_max_sharpe_matches_known_answers_at_unit_variance(n: int, expected: float) -> None:
    assert S.expected_max_sharpe(n, 1.0) == pytest.approx(expected, abs=1e-12)


def test_expected_max_sharpe_is_monotone_in_trials() -> None:
    assert S.expected_max_sharpe(100, 1.0) > S.expected_max_sharpe(10, 1.0) > S.expected_max_sharpe(2, 1.0)


def test_expected_max_sharpe_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        S.expected_max_sharpe(0, 1.0)
    with pytest.raises(ValueError):
        S.expected_max_sharpe(10, -0.1)


def test_bracket_at_1000_is_the_adr_value() -> None:
    assert round(S.bracket(1000), 7) == 3.2551215


def test_bracket_at_one_trial_is_zero_and_below_one_raises() -> None:
    assert S.bracket(1) == 0.0
    with pytest.raises(ValueError):
        S.bracket(0)


def test_bracket_equals_expected_max_sharpe_at_unit_variance() -> None:
    assert S.bracket(300) == S.expected_max_sharpe(300, 1.0)


# --------------------------------------------------------------------------- DSR, PSR, MinBTL


def test_dsr_at_the_benchmark_is_one_half() -> None:
    sr = S.expected_max_sharpe(20, 0.04)
    out = S.deflated_sharpe_ratio(sr, 142, 20, 0.04)
    assert out["z"] == pytest.approx(0.0, abs=1e-12)
    assert out["deflated_sharpe_ratio"] == pytest.approx(0.5, abs=1e-12)


def test_dsr_one_trial_zero_sharpe_is_one_half() -> None:
    out = S.deflated_sharpe_ratio(0.0, 142, 1, 0.04)
    assert out["expected_max_sharpe_under_null"] == 0.0
    assert out["deflated_sharpe_ratio"] == pytest.approx(0.5, abs=1e-12)


def test_dsr_known_answers() -> None:
    a = S.deflated_sharpe_ratio(0.20, 142, 20, 0.04)
    b = S.deflated_sharpe_ratio(0.60, 142, 20, 0.04, skewness=-1.5, kurtosis=9.0)
    assert a["deflated_sharpe_ratio"] == pytest.approx(0.01708804890241744, abs=1e-12)
    assert a["z"] == pytest.approx(-2.117987836790175, abs=1e-12)
    assert b["deflated_sharpe_ratio"] == pytest.approx(0.946614684125646, abs=1e-12)
    assert b["z"] == pytest.approx(1.6128797669695276, abs=1e-12)


def test_dsr_more_trials_lower_it_and_ugly_shape_lowers_a_high_sharpe() -> None:
    assert (
        S.deflated_sharpe_ratio(0.2, 142, 200, 0.04)["deflated_sharpe_ratio"]
        < S.deflated_sharpe_ratio(0.2, 142, 2, 0.04)["deflated_sharpe_ratio"]
    )
    normal = S.deflated_sharpe_ratio(0.6, 142, 20, 0.04)["deflated_sharpe_ratio"]
    ugly = S.deflated_sharpe_ratio(0.6, 142, 20, 0.04, skewness=-1.5, kurtosis=9.0)["deflated_sharpe_ratio"]
    assert ugly < normal


def test_dsr_below_benchmark_ugly_shape_raises_it_but_stays_below_half() -> None:
    normal = S.deflated_sharpe_ratio(0.2, 142, 20, 0.04)["deflated_sharpe_ratio"]
    ugly = S.deflated_sharpe_ratio(0.2, 142, 20, 0.04, skewness=-1.5, kurtosis=9.0)["deflated_sharpe_ratio"]
    assert normal < ugly < 0.5


def test_dsr_rejects_one_observation_and_degenerate_shape() -> None:
    with pytest.raises(ValueError):
        S.deflated_sharpe_ratio(0.2, 1, 20, 0.04)
    with pytest.raises(ValueError):
        S.deflated_sharpe_ratio(1.0, 50, 5, 0.04, skewness=5.0, kurtosis=1.0)


def test_psr_hand_computed_normal_case() -> None:
    # (0.1 - 0) * sqrt(100) / sqrt(1 - 0 + (3 - 1) / 4 * 0.01) = 1 / sqrt(1.005)
    z = 1.0 / math.sqrt(1.005)
    expected = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    assert S.probabilistic_sharpe_ratio(0.1, 101, 0.0, 3.0) == pytest.approx(expected, abs=1e-15)


def test_psr_hand_computed_with_benchmark_and_shape() -> None:
    # SR 0.3, SR* 0.1, T 50, skew -0.5, kurt 5: z = 0.2 * 7 / sqrt(1 + 0.15 + 1.0 * 0.09)
    z = 0.2 * 7.0 / math.sqrt(1.24)
    expected = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    got = S.probabilistic_sharpe_ratio(0.3, 50, -0.5, 5.0, benchmark=0.1)
    assert got == pytest.approx(expected, abs=1e-15)


def test_psr_equals_dsr_with_one_trial() -> None:
    dsr = S.deflated_sharpe_ratio(0.15, 80, 1, 0.3, skewness=0.4, kurtosis=4.0)["deflated_sharpe_ratio"]
    assert S.probabilistic_sharpe_ratio(0.15, 80, 0.4, 4.0) == pytest.approx(dsr, abs=1e-15)


def test_psr_rejects_short_sample_and_degenerate_shape() -> None:
    with pytest.raises(ValueError):
        S.probabilistic_sharpe_ratio(0.1, 1, 0.0, 3.0)
    with pytest.raises(ValueError):
        S.probabilistic_sharpe_ratio(1.0, 50, 5.0, 1.0)


def test_min_btl_hand_computed() -> None:
    b = S.bracket(1000)
    assert S.min_btl_years(1000, 1.0) == pytest.approx(b * b, abs=1e-12)
    assert S.min_btl_years(1000, 2.0) == pytest.approx(b * b / 4.0, abs=1e-12)
    assert S.min_btl_years(1000, 1.0) == pytest.approx(10.60, abs=0.01)


def test_min_btl_one_trial_is_zero_and_zero_sharpe_raises() -> None:
    assert S.min_btl_years(1, 1.5) == 0.0
    with pytest.raises(ValueError):
        S.min_btl_years(10, 0.0)


# --------------------------------------------------------------------------- HLZ, PBO


def test_hlz_pinned_answers() -> None:
    a = S.harvey_liu_zhu_verdict(0.25, 1.0, 100)
    b = S.harvey_liu_zhu_verdict(0.5, 1.0, 144)
    assert a["t_stat"] == 2.5 and a["passes_conventional"] and not a["passes_hlz"]
    assert b["t_stat"] == 6.0 and b["passes_hlz"]


def test_hlz_rejects_zero_sd_and_one_observation() -> None:
    with pytest.raises(ValueError):
        S.harvey_liu_zhu_verdict(0.1, 0.0, 10)
    with pytest.raises(ValueError):
        S.harvey_liu_zhu_verdict(0.1, 1.0, 1)


def test_pbo_one_candidate_is_degenerate() -> None:
    r = S.probability_of_backtest_overfitting([[1.0] * 8])
    assert r.degenerate and math.isnan(r.pbo) and r.n_splits == 0


def test_pbo_dominant_strategy_is_zero() -> None:
    r = S.probability_of_backtest_overfitting([[10.0] * 4, [1.0] * 4], n_blocks=4)
    assert r.pbo == 0.0 and r.n_splits == 6


def test_pbo_reversing_strategies_is_one() -> None:
    r = S.probability_of_backtest_overfitting([[10.0, 10.0, -10.0, -10.0], [-10.0, -10.0, 10.0, 10.0]], 4)
    assert r.pbo == 1.0 and r.n_splits == 6


def test_pbo_eight_blocks_has_seventy_splits() -> None:
    r = S.probability_of_backtest_overfitting([[float(i) for i in range(8)], [0.0] * 8])
    assert r.n_splits == 70


def test_pbo_rejects_odd_blocks_and_ragged_rows() -> None:
    with pytest.raises(ValueError):
        S.probability_of_backtest_overfitting([[1.0] * 5, [2.0] * 5], n_blocks=5)
    with pytest.raises(ValueError):
        S.probability_of_backtest_overfitting([[1.0] * 8, [2.0] * 7])


# --------------------------------------------------------------------------- further statistics


def test_holm_pinned_answers() -> None:
    got = S.holm({"H1": 0.009, "H2": 0.012, "H3": 0.02, "H4": 0.5, "H5": None})
    assert got == {"H1": True, "H2": True, "H3": False, "H4": False, "H5": False}
    assert S.holm({"a": 0.011, "b": 0.001}) == {"a": True, "b": True}
    assert S.holm({"a": 0.051, "b": 0.001}) == {"a": False, "b": True}


def test_newey_west_with_zero_lags_is_population_se() -> None:
    x = np.random.default_rng(1).normal(size=500)
    assert S.newey_west_se(x, 0) == pytest.approx(float(x.std()) / math.sqrt(500), rel=1e-12)


def test_newey_west_hand_computed_one_lag() -> None:
    # x = [1, 2, 3, 4]: e = [-1.5, -0.5, 0.5, 1.5]; e.e/4 = 1.25; lag-1 sum = 0.75 - 0.25 + 0.75 = 1.25,
    # weight 1/2 -> s = 1.25 + 2 * 0.5 * 1.25 / 4 = 1.5625; se = sqrt(1.5625 / 4) = 0.625
    assert S.newey_west_se(np.array([1.0, 2.0, 3.0, 4.0]), 1) == pytest.approx(0.625, abs=1e-15)


def test_unit_stats_takes_the_larger_p_and_sample_sd_t() -> None:
    x = np.random.default_rng(2).normal(loc=0.1, size=400)
    out = S.unit_stats(list(x), 5)
    p_t, p_nw = out["p_t"], out["p_nw"]
    assert isinstance(p_t, float) and isinstance(p_nw, float)
    assert out["p"] == max(p_t, p_nw)
    assert out["t"] == pytest.approx(float(x.mean()) / (float(x.std(ddof=1)) / math.sqrt(400)), rel=1e-12)


def test_unit_stats_short_or_flat_series_has_null_fields() -> None:
    assert S.unit_stats([1.0], 3)["p"] is None
    assert S.unit_stats([2.0, 2.0, 2.0], 3)["t"] is None


@pytest.mark.parametrize(
    ("t", "df", "expected"),
    [
        (0.0, 7, 0.5),
        (1.0, 1, 0.25),
        (-1.0, 1, 0.75),
        (1.0, 2, 0.5 - 1.0 / (2.0 * math.sqrt(3.0))),  # df 2 closed form: 1/2 - t / (2 sqrt(t^2 + 2))
        (3.0, 2, 0.5 - 3.0 / (2.0 * math.sqrt(11.0))),
        (math.inf, 5, 0.0),
        (-math.inf, 5, 1.0),
    ],
)
def test_one_sided_p_t_closed_forms(t: float, df: int, expected: float) -> None:
    assert S.one_sided_p_t(t, df) == pytest.approx(expected, abs=1e-13)


def test_one_sided_p_t_large_df_approaches_normal() -> None:
    assert S.one_sided_p_t(1.7, 10**7) == pytest.approx(S.one_sided_p_normal(1.7), abs=1e-7)


def test_one_sided_p_normal_known_points() -> None:
    assert S.one_sided_p_normal(0.0) == 0.5
    assert S.one_sided_p_normal(1.959963985) == pytest.approx(0.025, abs=1e-9)


def _years(nets: dict[int, list[float]]) -> list[tuple[date, float]]:
    return [(date(y, 1, 1 + i), v) for y, vs in nets.items() for i, v in enumerate(vs)]


def test_year_table_two_of_three_positive_passes_one_of_three_fails() -> None:
    assert S.year_table(_years({2020: [1.0] * 10, 2021: [1.0] * 10, 2022: [-1.0] * 10}), 10)["passes"]
    assert not S.year_table(_years({2020: [1.0] * 10, 2021: [-1.0] * 10, 2022: [-1.0] * 10}), 10)["passes"]


def test_year_table_short_year_does_not_qualify_and_fails() -> None:
    out = S.year_table(_years({2020: [1.0] * 10, 2021: [1.0] * 10, 2022: [1.0] * 9}), 10)
    assert out["qualifying_years"] == 2 and not out["passes"]


def test_year_table_lower_minimum_passes() -> None:
    assert S.year_table(_years({2020: [1.0] * 6, 2021: [1.0] * 6, 2022: [1.0] * 6}), 6)["passes"]


def test_year_table_defaults_are_pe_constants() -> None:
    assert S.YEAR_MIN_QUALIFYING == 3 and Fraction(2, 3) == S.YEAR_POSITIVE_SHARE


# --------------------------------------------------------------------------- moments, dsr_table


def test_moments_hand_computed() -> None:
    # [1, 2, 3, 6]: mean 3, pop var (4 + 1 + 0 + 9) / 4 = 3.5; m3 = (-8 - 1 + 0 + 27) / 4 = 4.5;
    # m4 = (16 + 1 + 0 + 81) / 4 = 24.5
    m = S.moments([1.0, 2.0, 3.0, 6.0])
    sd = math.sqrt(3.5)
    assert m["n"] == 4 and m["mean"] == 3.0
    assert m["sd"] == pytest.approx(sd, abs=1e-15)
    assert m["skew"] == pytest.approx(4.5 / sd**3, abs=1e-14)
    assert m["kurtosis"] == pytest.approx(24.5 / 3.5**2, abs=1e-14)
    assert m["sharpe"] == pytest.approx(3.0 / sd, abs=1e-15)


def test_moments_flat_series_keeps_pe_convention() -> None:
    assert S.moments([5.0, 5.0]) == {
        "n": 2,
        "mean": 5.0,
        "sd": 0.0,
        "skew": 0.0,
        "kurtosis": 3.0,
        "sharpe": 0.0,
    }


def test_dsr_table_uses_population_variance_of_member_sharpes() -> None:
    moms = {"a": S.moments([1.0, 2.0, 3.0, 6.0]), "b": S.moments([1.0, -1.0, 2.0, 0.5])}
    out = S.dsr_table(moms, ["a", "b"], 10, ["a", "b"])
    var = ((moms["a"]["sharpe"] - moms["b"]["sharpe"]) / 2.0) ** 2
    assert out["sharpe_variance"] == pytest.approx(var, abs=1e-15)
    assert out["expected_max_daily_sharpe_under_null"] == pytest.approx(math.sqrt(var) * S.bracket(10), 1e-12)
