"""SPA, Reality Check and StepM over a synthetic family (pfb.analysis.family_tests).

The sign convention is proved by a planted edge: one configuration with a positive mean daily net is found
by StepM and gives small p-values; the same configuration with the edge negated is not found. A noise family
is not rejected above the nominal rate over many data seeds; K < 2 and degenerate inputs are refused (a
subset of a declared family is refused in test_family_tests_hypothesis.py).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pfb.analysis import family_tests as ft

T = 250
K = 5
PLANTED = 3
EDGE_CENTS = 1500  # about 0.5 daily sd of the noise below: a t-statistic near 8 over 250 days


def _noise(seed: int, t: int = T, k: int = K) -> np.ndarray:
    return np.random.default_rng(seed).integers(-5000, 5001, size=(t, k)).astype(np.int64)


def _centred_noise(seed: int) -> np.ndarray:
    """Noise with every column's sample mean exactly 0 (float), so only the planted column has an edge."""
    net = _noise(seed).astype(np.float64)
    return net - net.mean(axis=0)


# --------------------------------------------------------------------------- the fixed rules


@pytest.mark.parametrize(
    ("t", "block"),
    [(1, 1), (2, 2), (8, 2), (9, 3), (27, 3), (28, 4), (64, 4), (65, 5), (250, 7), (2600, 14)],
)
def test_block_length_is_the_integer_ceiling_of_the_cube_root(t: int, block: int) -> None:
    assert ft.block_length(t) == block
    assert block**3 >= t and (block == 1 or (block - 1) ** 3 < t)


@pytest.mark.parametrize("bad", [0, -3, 2.5, True])
def test_block_length_refuses_a_non_positive_or_non_int_t(bad: object) -> None:
    with pytest.raises(ft.FamilyTestError, match="T must be an int"):
        ft.block_length(bad)  # type: ignore[arg-type]


def test_the_bootstrap_has_no_caller_switch() -> None:
    """Replications, seed, block rule and size are module constants, not keyword arguments."""
    import inspect

    for fn in (ft.family_tests, ft.spa_pvalues, ft.reality_check_pvalue, ft.stepm_superior):
        assert list(inspect.signature(fn).parameters) == ["matrix"]
    params = inspect.signature(ft.for_hypothesis).parameters
    assert set(params) == {"hypothesis_id", "config", "runs_dir", "root", "window"}
    conv = ft.conventions(T)
    assert conv["reps"] == ft.REPS == 1000 and conv["seed"] == ft.BOOTSTRAP_SEED
    assert conv["bootstrap"] == "stationary" and conv["block_size"] == 7 and conv["alpha"] == 0.05


# --------------------------------------------------------------------------- the sign convention


def test_a_planted_positive_edge_is_found_and_only_it() -> None:
    net = _centred_noise(7)
    net[:, PLANTED] += EDGE_CENTS
    result = ft.family_tests(net)
    assert result.stepm_superior == (PLANTED,)
    assert result.spa_pvalues["consistent"] < 0.01 and result.reality_check_pvalue < 0.01
    assert result.mean_net_cents[PLANTED] == pytest.approx(EDGE_CENTS)
    assert result.k == K and result.t == T


def test_the_loss_differential_handed_to_arch_is_the_net() -> None:
    """arch forms benchmark - model (SPA.__init__); with benchmark 0 and model loss -net that is +net."""
    net = _noise(5).astype(np.float64)
    bench, models = ft._losses(net)  # pyright: ignore[reportPrivateUsage]
    assert np.array_equal(bench[:, None] - models, net)
    spa = ft.SPA(bench, models, block_size=3, reps=10, seed=0)
    assert np.array_equal(spa._loss_diff, net)  # pyright: ignore[reportPrivateUsage]


def test_the_same_edge_negated_is_not_found() -> None:
    """With the wrong sign (net passed as a loss) the planted column would be 'superior' here."""
    net = _centred_noise(7)
    net[:, PLANTED] -= EDGE_CENTS
    result = ft.family_tests(net)
    assert result.stepm_superior == ()
    assert result.spa_pvalues["consistent"] > 0.2 and result.reality_check_pvalue > 0.2


def test_the_result_is_a_pure_function_of_the_matrix() -> None:
    net = _noise(11)
    assert ft.family_tests(net) == ft.family_tests(net.copy())


def test_spa_p_values_are_ordered_lower_consistent_upper() -> None:
    p = ft.spa_pvalues(_noise(12))
    assert set(p) == {"lower", "consistent", "upper"}
    assert 0.0 <= p["lower"] <= p["consistent"] <= p["upper"] <= 1.0


# --------------------------------------------------------------------------- noise families


TOLERANCE = 0.01  # a correct test at level ALPHA exceeds the allowed count with probability below 1 %


def _binomial_tail(n: int, p: float, c: int) -> float:
    """P(Bin(n, p) > c), exact."""
    return sum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(c + 1, n + 1))


def _allowed_rejections(n: int) -> int:
    """The smallest count c with P(Bin(n, ALPHA) > c) < TOLERANCE: more rejections than c over n independent
    null families says the size is above ALPHA at the 1 % level."""
    return next(c for c in range(n + 1) if _binomial_tail(n, ft.ALPHA, c) < TOLERANCE)


def test_the_allowed_rejection_counts_are_the_stated_binomial_bounds() -> None:
    assert _allowed_rejections(40) == 6 and _allowed_rejections(200) == 18 and _allowed_rejections(400) == 31
    assert _binomial_tail(40, 0.05, 5) > TOLERANCE > _binomial_tail(40, 0.05, 6)


def _noise_rejections(seeds: range) -> tuple[int, int, int]:
    """Rejections at ALPHA over independent zero-mean families, one data seed each (the bootstrap seed is the
    fixed BOOTSTRAP_SEED, so the rate is over the data, as a size is)."""
    spa = rc = stepm = 0
    for seed in seeds:
        result = ft.family_tests(_noise(seed))
        spa += result.spa_pvalues["consistent"] < ft.ALPHA
        rc += result.reality_check_pvalue < ft.ALPHA
        stepm += bool(result.stepm_superior)
    return spa, rc, stepm


def test_a_noise_family_is_not_rejected_above_the_nominal_rate_40_seeds() -> None:
    """40 zero-mean families (about 4 s): 2 rejections expected at 5 %, at most 6 allowed (1 % tail)."""
    seeds = range(1000, 1040)
    allowed = _allowed_rejections(len(seeds))
    spa, rc, stepm = _noise_rejections(seeds)
    assert spa <= allowed and rc <= allowed and stepm <= allowed, (spa, rc, stepm)


@pytest.mark.slow
def test_a_noise_family_is_not_rejected_above_the_nominal_rate_200_seeds() -> None:
    """200 families (about 25 s): 10 expected at 5 %, at most 18 allowed (1 % tail)."""
    seeds = range(2000, 2200)
    allowed = _allowed_rejections(len(seeds))
    spa, rc, stepm = _noise_rejections(seeds)
    assert spa <= allowed and rc <= allowed and stepm <= allowed, (spa, rc, stepm)


# --------------------------------------------------------------------------- refusals


def test_fewer_than_two_configurations_are_refused() -> None:
    one = _noise(1)[:, :1]
    for fn in (ft.family_tests, ft.spa_pvalues, ft.reality_check_pvalue, ft.stepm_superior):
        with pytest.raises(ft.FamilyTestError, match="at least 2 configurations"):
            fn(one)
    with pytest.raises(ft.FamilyTestError, match="2-D"):
        ft.family_tests(_noise(1)[:, 0])


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.zeros((2, 3)) + np.arange(3), "at least 3 rows"),
        (np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]]), "finite"),
        (np.array([[True, False], [False, True], [True, True]]), "numeric"),
        (np.array([["a", "b"], ["c", "d"], ["e", "f"]]), "numeric"),
    ],
)
def test_degenerate_matrices_are_refused(matrix: np.ndarray, message: str) -> None:
    with pytest.raises(ft.FamilyTestError, match=message):
        ft.family_tests(matrix)


def test_a_constant_configuration_is_refused_not_dropped() -> None:
    net = _noise(3)
    net[:, 2] = 0  # a configuration that never traded
    with pytest.raises(ft.FamilyTestError, match=r"column\(s\) \[2\] are constant"):
        ft.family_tests(net)


def test_arch_s_zero_variance_warning_is_a_refusal() -> None:
    """A column constant up to rounding passes check_family but arch warns in SPA.__init__: refused."""
    net = _noise(4).astype(np.float64)
    net[:, 1] = 1e15 + np.resize([0.0, 0.125], T)
    refusal = "arch refused the matrix: The loss differential has a variance of 0"
    with pytest.raises(ft.FamilyTestError, match=refusal):
        ft.family_tests(net)
