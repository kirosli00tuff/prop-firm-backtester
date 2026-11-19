"""Statistics: expected maximum Sharpe, the deflated and probabilistic Sharpe ratios, and MinBTL.

Design brief section 5 and ADR-001 sections 11 and 16 (A8, A14 G8). Every Sharpe here is per PERIOD (per trade
date), not annualized, unless a name says annual; kurtosis is RAW (normal = 3.0). Inputs are derived
statistics, so floats are used; nothing here carries money or decides a rule boundary. Imports nothing
from pfb, and no scipy: the Student t tail is computed from the regularized incomplete beta function below.

Not included: per-test verdict helpers and a PBO wrapper over per-test series.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from fractions import Fraction
from itertools import combinations

import numpy as np

EULER_MASCHERONI = 0.5772156649015329
HLZ_HURDLE = 3.0
CONVENTIONAL_HURDLE = 2.0
HOLM_ALPHA = 0.05
YEAR_MIN_QUALIFYING = 3  # fewer qualifying years: the criterion FAILS (not vacuous)
YEAR_POSITIVE_SHARE = Fraction(2, 3)
NORMAL_RAW_KURTOSIS = 3.0
PBO_RANK_CLAMP = 1e-9

_BETACF_MAX_ITER = 500
_BETACF_EPS = 1e-16
_BETACF_TINY = 1e-300

# Acklam's rational approximation of the inverse normal CDF (|err| < 1.15e-9), as in upstream.
_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
_P_LOW = 0.02425


# --------------------------------------------------------------------------- normal and Student t


def norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _tail_ppf(q: float) -> float:
    num = ((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]
    return num / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam, upstream `_norm_ppf`); ValueError unless 0 < p < 1."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p {p!r} must be strictly inside (0, 1)")
    if p < _P_LOW:
        return _tail_ppf(math.sqrt(-2 * math.log(p)))
    if p > 1 - _P_LOW:
        return -_tail_ppf(math.sqrt(-2 * math.log(1 - p)))
    q = p - 0.5
    r = q * q
    num = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q
    return num / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1)


def _nonzero(v: float) -> float:
    return _BETACF_TINY if abs(v) < _BETACF_TINY else v


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 / _nonzero(1.0 - qab * x / qap)
    h = d
    for m in range(1, _BETACF_MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 / _nonzero(1.0 + aa * d)
        c = _nonzero(1.0 + aa / c)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 / _nonzero(1.0 + aa * d)
        c = _nonzero(1.0 + aa / c)
        step = d * c
        h *= step
        if abs(step - 1.0) < _BETACF_EPS:
            return h
    raise ArithmeticError(f"incomplete beta continued fraction did not converge (a={a}, b={b}, x={x})")


def _betai(a: float, b: float, x: float, y: float) -> float:
    """Regularized incomplete beta I_x(a, b); y = 1 - x is passed in so it keeps its precision."""
    if x <= 0.0:
        return 0.0
    if y <= 0.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(y))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, y) / b


def one_sided_p_t(t: float, df: int) -> float:
    """P(T > t) for Student t with ``df`` degrees of freedom (upstream uses scipy's t.sf)."""
    if df < 1:
        raise ValueError(f"df {df!r} must be >= 1")
    if math.isnan(t):
        raise ValueError("t is NaN")
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    t2 = t * t
    tail = 0.5 * _betai(df / 2.0, 0.5, df / (df + t2), t2 / (df + t2))
    return tail if t > 0 else 1.0 - tail


def one_sided_p_normal(z: float) -> float:
    """P(Z > z) for a standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


# --------------------------------------------------------------------------- Sharpe deflation


def bracket(n_trials: int) -> float:
    """(1 - g) ppf(1 - 1/N) + g ppf(1 - 1/(N e)); 0.0 at N = 1 (no selection took place)."""
    if n_trials < 1:
        raise ValueError(f"n_trials {n_trials!r} must be >= 1")
    if n_trials == 1:
        return 0.0
    g = EULER_MASCHERONI
    return (1 - g) * norm_ppf(1 - 1.0 / n_trials) + g * norm_ppf(1 - 1.0 / (n_trials * math.e))


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """E[max SR] under the null (Bailey and Lopez de Prado eq. 3): sqrt(V) x bracket(N)."""
    if n_trials < 1:
        raise ValueError(f"n_trials {n_trials!r} must be >= 1")
    if sharpe_variance < 0:
        raise ValueError(f"sharpe_variance {sharpe_variance!r} must be >= 0")
    if n_trials == 1:
        return 0.0
    return math.sqrt(sharpe_variance) * bracket(n_trials)


def _psr_z(sharpe: float, n_observations: int, skewness: float, kurtosis: float, benchmark: float) -> float:
    if n_observations < 2:
        raise ValueError(f"n_observations {n_observations!r} must be >= 2")
    denominator_sq = 1.0 - skewness * sharpe + 0.25 * (kurtosis - 1.0) * sharpe**2
    if denominator_sq <= 0:
        raise ValueError(
            f"degenerate variance term {denominator_sq!r}: skew/kurtosis are inconsistent "
            f"with observed_sharpe {sharpe!r}"
        )
    return (sharpe - benchmark) * math.sqrt(n_observations - 1) / math.sqrt(denominator_sq)


def probabilistic_sharpe_ratio(
    sharpe: float, n_observations: int, skewness: float, kurtosis: float, benchmark: float = 0.0
) -> float:
    """PSR(SR*) = Phi((SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2)), per-period SR, raw g4."""
    return norm_cdf(_psr_z(sharpe, n_observations, skewness, kurtosis, benchmark))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_observations: int,
    n_trials: int,
    sharpe_variance: float,
    skewness: float = 0.0,
    kurtosis: float = NORMAL_RAW_KURTOSIS,
) -> dict[str, float | int]:
    """Upstream's DSR: PSR at the benchmark expected_max_sharpe(N, V); same keys as upstream."""
    if n_observations < 2:
        raise ValueError(f"n_observations {n_observations!r} must be >= 2")
    benchmark = expected_max_sharpe(n_trials, sharpe_variance)
    z = _psr_z(observed_sharpe, n_observations, skewness, kurtosis, benchmark)
    return {
        "deflated_sharpe_ratio": norm_cdf(z),
        "expected_max_sharpe_under_null": benchmark,
        "observed_sharpe": observed_sharpe,
        "z": z,
        "n_trials": n_trials,
        "n_observations": n_observations,
    }


def min_btl_years(n_trials: int, annual_sharpe: float) -> float:
    """MinBTL (years) = bracket(N)^2 / (annual Sharpe)^2; ValueError at a zero Sharpe."""
    if annual_sharpe == 0.0 or not math.isfinite(annual_sharpe):
        raise ValueError(f"annual_sharpe {annual_sharpe!r} must be finite and non-zero")
    return bracket(n_trials) ** 2 / annual_sharpe**2


def harvey_liu_zhu_verdict(
    mean: float, stdev: float, n_observations: int, hurdle: float = HLZ_HURDLE
) -> dict[str, float | bool]:
    """t-stat of a mean against the HLZ (2016) hurdle: t > 3.0, not the usual 2.0."""
    if n_observations < 2:
        raise ValueError(f"n_observations {n_observations!r} must be >= 2")
    if stdev <= 0:
        raise ValueError(f"stdev {stdev!r} must be > 0")
    t = mean / (stdev / math.sqrt(n_observations))
    return {
        "t_stat": t,
        "hurdle": hurdle,
        "passes_hlz": t > hurdle,
        "passes_conventional": t > CONVENTIONAL_HURDLE,
    }


# --------------------------------------------------------------------------- PBO (CSCV)


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    n_splits: int
    n_strategies: int
    logits: tuple[float, ...]
    degenerate: bool
    note: str


def _split_logit(performance: Sequence[Sequence[float]], train: tuple[int, ...], n_blocks: int) -> float:
    test = [b for b in range(n_blocks) if b not in set(train)]
    in_sample = [sum(row[b] for b in train) for row in performance]
    out_sample = [sum(row[b] for b in test) for row in performance]
    n = len(performance)
    best = max(range(n), key=lambda s: in_sample[s])  # first maximum on ties, as in upstream
    worse = sum(1 for s in range(n) if out_sample[s] < out_sample[best])
    rank = min(max((worse + 0.5) / n, PBO_RANK_CLAMP), 1 - PBO_RANK_CLAMP)
    return math.log(rank / (1 - rank))


def probability_of_backtest_overfitting(
    performance: Sequence[Sequence[float]], n_blocks: int = 8
) -> PBOResult:
    """CSCV PBO: share of balanced splits whose in-sample best is at or below the out-of-sample median."""
    n_strategies = len(performance)
    if n_strategies < 2:
        return PBOResult(
            float("nan"),
            0,
            n_strategies,
            (),
            True,
            "PBO undefined with fewer than 2 candidates: no selection occurs.",
        )
    if n_blocks % 2 != 0:
        raise ValueError(f"n_blocks {n_blocks!r} must be even for a symmetric split")
    widths = {len(row) for row in performance}
    if widths != {n_blocks}:
        raise ValueError(f"every strategy needs exactly {n_blocks} blocks; got {sorted(widths)}")
    splits = combinations(range(n_blocks), n_blocks // 2)
    logits = [_split_logit(performance, train, n_blocks) for train in splits]
    pbo = sum(1 for lg in logits if lg <= 0) / len(logits)
    return PBOResult(
        pbo,
        len(logits),
        n_strategies,
        tuple(logits),
        False,
        "Fraction of symmetric splits where the in-sample best ranked at or below the out-of-sample median.",
    )


# --------------------------------------------------------------------------- further statistics


def newey_west_se(x: np.ndarray, lags: int) -> float:
    """HAC standard error of the mean (Bartlett kernel, population moments, no small-sample correction)."""
    n = len(x)
    e = x - x.mean()
    s = float(e @ e) / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        s += 2.0 * w * float(e[lag:] @ e[:-lag]) / n
    return math.sqrt(max(s, 0.0) / n)


def unit_stats(values: Sequence[float], lags: int) -> dict[str, float | int | None]:
    """n, mean, sample sd, t, Newey-West t and the LARGER of the two one-sided p (upstream `unit_stats`)."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 2:
        return {
            "n": n,
            "mean": float(x.mean()) if n else None,
            "sd": None,
            "t": None,
            "p_t": None,
            "p_nw": None,
            "p": None,
            "nw_lags": lags,
        }
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    if sd == 0:
        return {
            "n": n,
            "mean": mean,
            "sd": 0.0,
            "t": None,
            "p_t": None,
            "p_nw": None,
            "p": None,
            "nw_lags": lags,
        }
    t = mean / (sd / math.sqrt(n))
    se_nw = newey_west_se(x, lags)
    t_nw = mean / se_nw if se_nw > 0 else math.copysign(math.inf, mean)
    p_t, p_nw = one_sided_p_t(t, n - 1), one_sided_p_normal(t_nw)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "t": t,
        "t_nw": t_nw,
        "p_t": p_t,
        "p_nw": p_nw,
        "p": max(p_t, p_nw),
        "nw_lags": lags,
    }


def holm(pvalues: Mapping[str, float | None], alpha: float = HOLM_ALPHA) -> dict[str, bool]:
    """Holm step-down; a test with no p is never rejected and still counts in m."""
    m = len(pvalues)

    def sort_key(k: str) -> tuple[float, str]:
        p = pvalues[k]
        return (math.inf if p is None else p, k)

    out = dict.fromkeys(pvalues, False)
    for i, k in enumerate(sorted(pvalues, key=sort_key)):
        p = pvalues[k]
        if p is None or p > alpha / (m - i):
            break
        out[k] = True
    return out


def year_table(
    series: Sequence[tuple[date, float]],
    min_units: int,
    min_qualifying: int = YEAR_MIN_QUALIFYING,
    positive_share: Fraction = YEAR_POSITIVE_SHARE,
) -> dict[str, object]:
    """Per calendar year: units, net, qualifies, positive; passes on enough positive qualifying years."""
    years: dict[int, list[float]] = {}
    for d, v in series:
        years.setdefault(d.year, []).append(float(v))
    rows = {
        str(y): {
            "units": len(v),
            "net": float(sum(v)),
            "qualifies": len(v) >= min_units,
            "positive": sum(v) > 0,
        }
        for y, v in sorted(years.items())
    }
    q = [r for r in rows.values() if r["qualifies"]]
    pos = sum(1 for r in q if r["positive"])
    passed = len(q) >= min_qualifying and Fraction(pos, len(q)) >= positive_share
    return {
        "min_units": min_units,
        "qualifying_years": len(q),
        "positive_years": pos,
        "passes": bool(passed),
        "years": rows,
    }


# --------------------------------------------------------------------------- an upstream module


def moments(xs: Sequence[float]) -> dict[str, float | int]:
    """Upstream `moments`: n, mean, population sd, skew, RAW kurtosis, per-period Sharpe; sd 0 gives Sharpe
    0."""
    vals = [float(x) for x in xs]
    n = len(vals)
    mean = statistics.fmean(vals)
    sd = statistics.pstdev(vals)
    if sd == 0:
        return {"n": n, "mean": mean, "sd": 0.0, "skew": 0.0, "kurtosis": NORMAL_RAW_KURTOSIS, "sharpe": 0.0}
    skew = sum((x - mean) ** 3 for x in vals) / n / sd**3
    kurt = sum((x - mean) ** 4 for x in vals) / n / sd**4
    return {"n": n, "mean": mean, "sd": sd, "skew": skew, "kurtosis": kurt, "sharpe": mean / sd}


def dsr_of(mom: Mapping[str, float | int], n_trials: int, variance: float) -> tuple[float | None, str | None]:
    """DSR, or (None, reason) when sd = 0 or the moments are degenerate (upstream `dsr`)."""
    if mom["sd"] <= 0:
        return None, "sd = 0"
    try:
        out = deflated_sharpe_ratio(
            float(mom["sharpe"]),
            int(mom["n"]),
            n_trials,
            variance,
            float(mom["skew"]),
            float(mom["kurtosis"]),
        )
    except ValueError as exc:
        return None, str(exc)
    return float(out["deflated_sharpe_ratio"]), None


def dsr_table(
    moms: Mapping[str, Mapping[str, float | int]],
    variance_over: Sequence[str],
    n_trials: int,
    members: Sequence[str],
) -> dict[str, object]:
    """Upstream `dsr_table`: the Sharpe variance is the population variance over ``variance_over``."""
    variance = statistics.pvariance([float(moms[k]["sharpe"]) for k in variance_over])
    benchmark = expected_max_sharpe(n_trials, variance)
    values: dict[str, float | None] = {}
    notes: dict[str, str] = {}
    for key in members:
        values[key], note = dsr_of(moms[key], n_trials, variance)
        if note:
            notes[key] = note
    return {
        "n_trials": n_trials,
        "variance_over_n_members": len(variance_over),
        "sharpe_variance": variance,
        "expected_max_daily_sharpe_under_null": benchmark,
        "dsr": values,
        "undefined": notes,
    }
