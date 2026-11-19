"""S3 power check (docs/TRIALS.md section 5): hand-computed known answers, monotonicity, the boundary, and a
Monte Carlo check of the 0.80 power at the returned edge (synthetic data only)."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from pfb.analysis.power import ALPHA, ALPHA_RULE, LEGACY_ALPHA_RULE, POWER, PowerResult, min_edge, years_of
from pfb.metrics.stats import bracket, norm_ppf, probabilistic_sharpe_ratio

Z = 1.6448536269514715 + 0.8416212335729144  # ppf(0.95) + ppf(0.80) = 2.4864748605


def example(target_edge_cents: int | None = None) -> PowerResult:
    """The TRIALS.md example inputs: m = 220, s = 21000, Y = 10, N = 1000."""
    return min_edge(
        trades_per_year=220.0,
        trade_sd_cents=21000,
        years=10.0,
        n_trials=1000,
        target_edge_cents=target_edge_cents,
    )


def test_constants() -> None:
    assert (ALPHA, POWER) == (0.05, 0.80)
    assert norm_ppf(0.95) + norm_ppf(0.80) == pytest.approx(Z, abs=1e-8)


def test_years_of_sums_inclusive_calendar_days() -> None:
    # 2024-01-01..2024-12-31 = 366 days; 2025-01-01..2025-01-10 = 10 days; 376 / 365.25
    assert years_of(
        [(date(2024, 1, 1), date(2024, 12, 31)), (date(2025, 1, 1), date(2025, 1, 10))]
    ) == pytest.approx(376 / 365.25)
    assert years_of([(date(2020, 3, 1), date(2020, 3, 1))]) == pytest.approx(1 / 365.25)
    with pytest.raises(ValueError):
        years_of([])
    with pytest.raises(ValueError):
        years_of([(date(2020, 3, 2), date(2020, 3, 1))])


def test_known_answer_a_single_trial_one_year() -> None:
    """m = 252, s = 10000, Y = 1, N = 1.

    T = max(2, round(252)) = 252; n = 252; b = bracket(1) = 0
    z = 2.4864748605; g4 = 3 (1 + 252/252) = 6
    s0 = (0 + 2.4864748605) / sqrt(251) = 2.4864748605 / 15.8429795178 = 0.1569450
    k = sqrt(1 + (5/4) 0.1569450^2) = sqrt(1 + 1.25 x 0.0246317) = sqrt(1.0307896) = 1.0152781
    t_dsr = t_alpha = 0 + 2.4864748605 x 1.0152781 = 2.5244635
    c = 2.5244635 / sqrt(252 x 251 / 252) = 2.5244635 / 15.8429795 = 0.1593428
    edge = 0.1593428 x 10000 / sqrt(1 - 0.1593428^2) = 1593.428 / sqrt(0.9746099) = 1593.428 / 0.9872233
         = 1614.05  ->  ceil = 1615 (both figures, since b = 0)
    """
    s0 = Z / math.sqrt(251)
    k = math.sqrt(1 + 1.25 * s0**2)
    t = Z * k
    c = t / math.sqrt(251)
    assert (round(s0, 6), round(k, 6), round(t, 6), round(c, 6)) == (0.156945, 1.015278, 2.524463, 0.159343)
    assert c * 10000 / math.sqrt(1 - c * c) == pytest.approx(1614.0494, abs=1e-3)
    r = min_edge(trades_per_year=252.0, trade_sd_cents=10000, years=1.0, n_trials=1)
    assert (r.days, r.expected_trades, r.bracket) == (252, 252.0, 0.0)
    assert r.t_dsr == pytest.approx(2.5244635, abs=1e-6)
    assert r.t_alpha == pytest.approx(r.t_dsr)
    assert (r.min_edge_cents, r.min_edge_cents_alpha) == (1615, 1615)
    assert r.passed is None and r.target_edge_cents is None


def test_known_answer_b_example_hypothesis_at_n_1000() -> None:
    """m = 220, s = 21000, Y = 10, N = 1000 (the TRIALS.md example inputs).

    T = 2520; n = 2200; b = 0.4228 ppf(1 - 1/1000) + 0.5772 ppf(1 - 1/(1000 e)) = 3.2551215
    g4 = 3 (1 + 252/220) = 6.4363636; (g4 - 1)/4 = 1.3590909
    s0 = (3.2551215 + 2.4864749) / sqrt(2519) = 5.7415964 / 50.1896404 = 0.1143980
    k = sqrt(1 + 1.3590909 x 0.0130869) = sqrt(1.0177863) = 1.0088540
    t_dsr = 3.2551215 + 2.4864749 x 1.0088540 = 5.7636115
    sqrt(n (T - 1)/T) = sqrt(2200 x 2519 / 2520) = sqrt(2199.1270) = 46.8948505
    c_dsr = 5.7636115 / 46.8948505 = 0.1229050 -> edge = 0.1229050 x 21000 / sqrt(1 - 0.0151056)
          = 2581.005 / 0.9924184 = 2600.72 -> 2601
    alpha (b = 0): sa = 2.4864749 / 50.1896404 = 0.0495416; sa^2 = 0.0024544
    ka = sqrt(1 + 1.3590909 x 0.0024544) = sqrt(1.0033357) = 1.0016665; t_alpha = 2.4906185
    c_alpha = 2.4906185 / 46.8948505 = 0.0531107 -> edge = 1115.325 / 0.9985886 = 1116.90 -> 1117
    legacy alpha (rule 1, k of the deflated Sharpe): t_alpha = 2.4864749 x 1.0088540 = 2.5084900
    c_alpha = 2.5084900 / 46.8948505 = 0.0534918 -> edge = 1123.328 / 0.9985683 = 1124.94 -> 1125
    """
    b = 0.42278433509846713 * norm_ppf(1 - 1 / 1000) + 0.5772156649015329 * norm_ppf(1 - 1 / (1000 * math.e))
    assert b == pytest.approx(bracket(1000)) and b == pytest.approx(3.2551215, abs=1e-6)
    s0 = (b + Z) / math.sqrt(2519)
    k = math.sqrt(1 + (3 * (1 + 252 / 220) - 1) / 4 * s0**2)
    assert (round(s0, 6), round(k, 6)) == (0.114398, 1.008854)
    root = math.sqrt(2200 * 2519 / 2520)
    c_dsr, c_legacy = (b + Z * k) / root, Z * k / root
    assert c_dsr * 21000 / math.sqrt(1 - c_dsr**2) == pytest.approx(2600.7225, abs=1e-3)
    assert c_legacy * 21000 / math.sqrt(1 - c_legacy**2) == pytest.approx(1124.9384, abs=1e-3)
    sa = Z / math.sqrt(2519)
    ka = math.sqrt(1 + (3 * (1 + 252 / 220) - 1) / 4 * sa**2)
    assert (round(sa, 7), round(ka, 7)) == (0.0495416, 1.0016665)
    c_alpha = Z * ka / root
    assert c_alpha * 21000 / math.sqrt(1 - c_alpha**2) == pytest.approx(1116.9011, abs=1e-3)
    r = example(target_edge_cents=3000)
    assert (r.days, r.expected_trades) == (2520, 2200.0)
    assert r.t_dsr == pytest.approx(5.7636115, abs=1e-6) and r.t_alpha == pytest.approx(2.4906185, abs=1e-6)
    assert (r.min_edge_cents, r.min_edge_cents_alpha, r.passed) == (2601, 1117, True)
    legacy = min_edge(trades_per_year=220.0, trade_sd_cents=21000, years=10.0, n_trials=1000,
                      target_edge_cents=3000, alpha_rule=LEGACY_ALPHA_RULE)  # fmt: skip
    assert legacy.t_alpha == pytest.approx(2.5084900, abs=1e-6)
    assert (legacy.min_edge_cents, legacy.min_edge_cents_alpha, legacy.passed) == (2601, 1125, True)
    assert legacy.t_dsr == r.t_dsr


def test_known_answer_c_skewed_trades() -> None:
    """Set B inputs with trade skew -2 and raw kurtosis 9, then skew +2 (k floored at 1).

    lam = 220/252 = 0.8730159; g3 = -2 / sqrt(lam) = -2 / 0.9343531 = -2.1405183
    g4 = 3 + 9 / lam = 3 + 10.3090909 = 13.3090909; (g4 - 1)/4 = 3.0772727
    s0 = 0.1143980 (set B); s0^2 = 0.0130869
    k^2 = max(1, 1 + 2.1405183 x 0.1143980 + 3.0772727 x 0.0130869) = max(1, 1.2448711 + 0.0402720)
        = 1.2851431 -> k = 1.1336415
    t_dsr = 3.2551215 + 2.4864749 x 1.1336415 = 6.0738926
    alpha: sa = 0.0495416; ka^2 = 1 + 2.1405182 x 0.0495416 + 3.0772727 x 0.0024544
        = 1 + 0.1060447 + 0.0075528 = 1.1135975 -> ka = 1.0552713; t_alpha = 2.6239055 -> edge 1176.86 -> 1177
        (legacy rule 1: t_alpha = 2.4864749 x 1.1336415 = 2.8187711 -> 1265)
    c = 6.0738926 / 46.8948505 = 0.1295215 -> edge = 2719.952 / sqrt(1 - 0.0167758) = 2743.06 -> 2744
    skew +2: k^2 = max(1, 0.7551289 + 0.0402720) = 1 -> t_dsr = b + z = 5.7415964, c = 0.1224355
        -> edge = 2571.146 / 0.9924765 = 2590.64 -> 2591 (below the normal-default 2601, since normal k > 1)
    """
    neg = min_edge(
        trades_per_year=220.0,
        trade_sd_cents=21000,
        years=10.0,
        n_trials=1000,
        trade_skew=-2.0,
        trade_kurtosis=9.0,
    )
    assert neg.t_dsr == pytest.approx(6.0738926, abs=1e-6)
    assert neg.t_alpha == pytest.approx(2.6239055, abs=1e-6)
    assert (neg.min_edge_cents, neg.min_edge_cents_alpha) == (2744, 1177)
    old = min_edge(trades_per_year=220.0, trade_sd_cents=21000, years=10.0, n_trials=1000, trade_skew=-2.0,
                   trade_kurtosis=9.0, alpha_rule=LEGACY_ALPHA_RULE)  # fmt: skip
    assert old.t_alpha == pytest.approx(2.8187711, abs=1e-6)
    assert (old.min_edge_cents, old.min_edge_cents_alpha) == (2744, 1265)
    assert (neg.trade_skew, neg.trade_kurtosis) == (-2.0, 9.0)
    pos = min_edge(
        trades_per_year=220.0,
        trade_sd_cents=21000,
        years=10.0,
        n_trials=1000,
        trade_skew=2.0,
        trade_kurtosis=9.0,
    )
    assert pos.t_dsr == pytest.approx(bracket(1000) + Z, abs=1e-6) and pos.min_edge_cents == 2591


def test_defaults_equal_explicit_normal_shape() -> None:
    explicit = min_edge(
        trades_per_year=220.0,
        trade_sd_cents=21000,
        years=10.0,
        n_trials=1000,
        trade_skew=0.0,
        trade_kurtosis=3.0,
    )
    assert explicit == example()
    assert example().to_json()["trade_skew"] == 0.0 and example().to_json()["trade_kurtosis"] == 3.0


def test_target_one_cent_under_fails_and_at_passes() -> None:
    base = {"trades_per_year": 220.0, "trade_sd_cents": 21000, "years": 10.0, "n_trials": 1000}
    edge = min_edge(**base).min_edge_cents
    assert edge == 2601
    assert min_edge(**base, target_edge_cents=edge - 1).passed is False
    assert min_edge(**base, target_edge_cents=edge).passed is True


def test_undetectable_gives_none_and_fails() -> None:
    # 2 trades in 0.01 years: c = t / sqrt(n (T-1)/T) with n = 0.02 is far above 1
    r = min_edge(
        trades_per_year=2.0, trade_sd_cents=10000, years=0.01, n_trials=1000, target_edge_cents=10**9
    )
    assert r.days == 3 and r.min_edge_cents is None and r.min_edge_cents_alpha is None
    assert r.passed is False
    assert min_edge(trades_per_year=2.0, trade_sd_cents=10000, years=0.01, n_trials=5).passed is False


def test_days_floor_is_two() -> None:
    assert min_edge(trades_per_year=50000.0, trade_sd_cents=100, years=0.001, n_trials=1).days == 2


def _edge(**kw: object) -> int:
    base: dict[str, object] = {
        "trades_per_year": 220.0,
        "trade_sd_cents": 21000,
        "years": 5.0,
        "n_trials": 500,
    }
    base.update(kw)
    value = min_edge(**base).min_edge_cents  # type: ignore[arg-type]
    assert value is not None
    return value


def test_monotone() -> None:
    trades = [_edge(trades_per_year=m) for m in (50.0, 100.0, 220.0, 500.0, 2000.0)]
    years = [_edge(years=y) for y in (1.0, 2.0, 5.0, 10.0, 20.0)]
    n_trials = [_edge(n_trials=n) for n in (1, 10, 100, 1000, 10000)]
    sds = [_edge(trade_sd_cents=s) for s in (5000, 10000, 21000, 40000)]
    assert trades == sorted(trades, reverse=True) and len(set(trades)) == len(trades)
    assert years == sorted(years, reverse=True) and len(set(years)) == len(years)
    assert n_trials == sorted(n_trials) and len(set(n_trials)) == len(n_trials)
    assert sds == sorted(sds) and len(set(sds)) == len(sds)


def test_pure_and_json() -> None:
    a = example(target_edge_cents=3000)
    b = example(target_edge_cents=3000)
    assert a == b and isinstance(a, PowerResult)
    js = a.to_json()
    assert js["min_edge_cents"] == 2601 and js["passed"] is True and js["n_used"] == 1000
    for key in (
        "n_used",
        "years",
        "days",
        "expected_trades",
        "bracket",
        "t_dsr",
        "t_alpha",
        "min_edge_cents",
        "min_edge_cents_alpha",
        "target_edge_cents",
        "passed",
        "trade_skew",
        "trade_kurtosis",
    ):
        assert key in js
    with pytest.raises(AttributeError):
        a.days = 3  # type: ignore[misc]


@pytest.mark.parametrize(
    "bad",
    [
        {"trades_per_year": 0.0},
        {"trades_per_year": float("nan")},
        {"trade_sd_cents": 0},
        {"trade_sd_cents": 21000.0},
        {"trade_sd_cents": True},
        {"years": 0.0},
        {"years": float("inf")},
        {"n_trials": 0},
        {"n_trials": 2.0},
        {"target_edge_cents": 30.5},
        {"trade_kurtosis": 0.99},
        {"trade_kurtosis": float("nan")},
        {"trade_skew": float("inf")},
        {"trade_skew": True},
        {"trade_kurtosis": "3"},
        {"alpha_rule": 0},
        {"alpha_rule": 3},
        {"alpha_rule": True},
    ],
)
def test_bad_inputs_refused(bad: dict[str, object]) -> None:
    kw: dict[str, object] = {"trades_per_year": 220.0, "trade_sd_cents": 21000, "years": 10.0, "n_trials": 10}
    kw.update(bad)
    with pytest.raises(ValueError):
        min_edge(**kw)  # type: ignore[arg-type]


# ------------------------------------------------------- the alpha-only figure (alpha rule 2)

# Synthetic grid (round made-up inputs), figures pinned from the code under alpha rule 1:
# (trades_per_year, trade_sd_cents, years, n_used, min_edge_cents, legacy min_edge_cents_alpha)
SYNTHETIC_GRID = [
    (150.0, 15000, 3.5, 1002, 3944, 1693),
    (150.0, 15000, 3.5, 1020, 3948, 1693),
    (150.0, 40000, 3.5, 1002, 10517, 4513),
    (150.0, 40000, 3.5, 1020, 10527, 4513),
    (200.0, 16000, 3.5, 1002, 3603, 1552),
    (200.0, 16000, 3.5, 1020, 3607, 1552),
    (200.0, 45000, 3.5, 1002, 10133, 4365),
    (200.0, 45000, 3.5, 1020, 10142, 4365),
    (200.0, 17000, 1.0, 1002, 7933, 3329),
    (200.0, 17000, 1.0, 1020, 7941, 3330),
    (200.0, 45000, 1.0, 1002, 20997, 8812),
    (200.0, 45000, 1.0, 1020, 21021, 8813),
]
# Other figures: (m, s, Y, N, skew, raw kurtosis, min_edge_cents, legacy alpha)
OTHER_FIGURES = [
    (200.0, 3000, 1.2, 1001, 0.0, 3.0, 1246, 528),
    (252.0, 10000, 1.0, 1, 0.0, 3.0, 1615, 1615),
    (220.0, 21000, 10.0, 1000, 0.0, 3.0, 2601, 1125),
    (220.0, 21000, 10.0, 1000, -2.0, 9.0, 2744, 1265),
    (220.0, 21000, 10.0, 1000, 2.0, 9.0, 2591, 1116),
]
ALL_WRITTEN = [(m, s, y, n, 0.0, 3.0, gate, alpha) for m, s, y, n, gate, alpha in SYNTHETIC_GRID]
ALL_WRITTEN += OTHER_FIGURES


def test_alpha_rule_constants() -> None:
    assert (ALPHA_RULE, LEGACY_ALPHA_RULE) == (2, 1)


@pytest.mark.parametrize(("m", "s", "years", "n", "skew", "kurt", "gate", "legacy_alpha"), ALL_WRITTEN)
def test_the_gate_figure_did_not_move_and_rule_1_is_the_old_alpha(
    m: float, s: int, years: float, n: int, skew: float, kurt: float, gate: int, legacy_alpha: int
) -> None:
    kw: dict[str, object] = {"trades_per_year": m, "trade_sd_cents": s, "years": years, "n_trials": n,
                             "trade_skew": skew, "trade_kurtosis": kurt}  # fmt: skip
    new = min_edge(**kw)  # type: ignore[arg-type]
    old = min_edge(**kw, alpha_rule=LEGACY_ALPHA_RULE)  # type: ignore[arg-type]
    assert new.min_edge_cents == old.min_edge_cents == gate  # to the cent
    assert new.t_dsr == old.t_dsr and new.bracket == old.bracket  # bit for bit
    assert old.min_edge_cents_alpha == legacy_alpha
    assert new.min_edge_cents_alpha is not None and new.min_edge_cents_alpha <= legacy_alpha
    assert new.min_edge_cents_alpha <= gate


@pytest.mark.parametrize(("m", "s", "years", "n", "skew", "kurt", "gate", "legacy_alpha"), ALL_WRITTEN)
def test_the_alpha_figure_is_the_gate_formula_without_deflation(
    m: float, s: int, years: float, n: int, skew: float, kurt: float, gate: int, legacy_alpha: int
) -> None:
    """The direct check, closed form, tolerance 0 cents: the alpha-only minimum is what the gate formula
    gives when nothing is deflated (N = 1, b = 0), so it is the same at every N."""
    kw: dict[str, object] = {"trades_per_year": m, "trade_sd_cents": s, "years": years,
                             "trade_skew": skew, "trade_kurtosis": kurt}  # fmt: skip
    single = min_edge(**kw, n_trials=1)  # type: ignore[arg-type]
    assert single.bracket == 0.0 and single.min_edge_cents == single.min_edge_cents_alpha
    for n_used in (n, 100_000):
        r = min_edge(**kw, n_trials=n_used)  # type: ignore[arg-type]
        assert r.min_edge_cents_alpha == single.min_edge_cents
        assert r.t_alpha == single.t_dsr
    # by hand: ka at sa = z / sqrt(T - 1), then the module's own edge conversion
    lam, days = m / 252.0, single.days
    sa = Z / math.sqrt(days - 1)
    ka = math.sqrt(max(1.0, 1.0 - skew / math.sqrt(lam) * sa + ((3.0 + kurt / lam - 1.0) / 4.0) * sa * sa))
    c = Z * ka / math.sqrt(m * years * (days - 1) / days)
    assert single.min_edge_cents_alpha == math.ceil(c * s / math.sqrt(1.0 - c * c))


def test_the_old_alpha_figure_rose_with_n_and_the_new_one_does_not() -> None:
    kw: dict[str, object] = {"trades_per_year": 200.0, "trade_sd_cents": 3000, "years": 1.2}
    old = [min_edge(**kw, n_trials=n, alpha_rule=LEGACY_ALPHA_RULE).min_edge_cents_alpha  # type: ignore[arg-type]
           for n in (1, 1001, 100_000)]  # fmt: skip
    new = [min_edge(**kw, n_trials=n).min_edge_cents_alpha for n in (1, 1001, 100_000)]  # type: ignore[arg-type]
    assert old == [497, 528, 543] and new == [497, 497, 497]


# --------------------------------------------------------------------------- Monte Carlo


def _psr_share(
    m: float, s: int, days: int, n_trials: int, edge: int, sims: int, seed: int, trade_skew: int = 0
) -> float:
    """Share of simulated runs with PSR(SR0) >= 0.95, SR0 = sqrt(1/(T-1)) bracket(N) (floor variance).

    Daily net = sum of K ~ Poisson(m/252) trades of mean `edge` and sd s; normal trades give the exact
    K edge + sqrt(K) s Z; trade_skew = -1 / +1 uses the standardized -/+(Exp(1) - 1) trade (skew -2 / +2,
    raw kurtosis 9).
    """
    rng = np.random.default_rng(seed)
    k = rng.poisson(m / 252.0, size=(sims, days))
    if trade_skew == 0:
        x = k * edge + np.sqrt(k) * s * rng.standard_normal((sims, days))
    else:
        x = np.zeros((sims, days))
        for j in range(int(k.max())):
            x += np.where(k > j, edge + trade_skew * s * (rng.exponential(1.0, (sims, days)) - 1.0), 0.0)
    x = np.rint(x)  # integer cents
    mean, sd = x.mean(1), x.std(1)
    dev = x - mean[:, None]
    skew, kurt = (dev**3).mean(1) / sd**3, (dev**4).mean(1) / sd**4
    sr0 = math.sqrt(1.0 / (days - 1)) * bracket(n_trials)
    hits = sum(
        probabilistic_sharpe_ratio(float(mean[i] / sd[i]), days, float(skew[i]), float(kurt[i]), sr0) >= 0.95
        for i in range(sims)
    )
    return hits / sims


def _se(sims: int) -> float:
    return math.sqrt(POWER * (1 - POWER) / sims)


# trade shapes simulated by _psr_share: (code, true skew, true raw kurtosis) of the per-trade P&L
NORMAL, NEG_SKEW, POS_SKEW = (0, 0.0, 3.0), (-1, -2.0, 9.0), (1, 2.0, 9.0)


@pytest.mark.parametrize(
    ("m", "s", "years", "n_trials", "shape", "sims"),
    [
        (220.0, 21000, 10.0, 1000, NORMAL, 4000),
        (220.0, 21000, 10.0, 1000, NEG_SKEW, 4000),
        (220.0, 21000, 10.0, 1000, POS_SKEW, 4000),
        (1000.0, 5000, 3.0, 1000, NORMAL, 2000),
        (50.0, 20000, 10.0, 1000, NORMAL, 2000),
        (252.0, 10000, 1.0, 1, NEG_SKEW, 4000),
    ],
)
def test_mc_power_never_optimistic(
    m: float, s: int, years: float, n_trials: int, shape: tuple[int, float, float], sims: int
) -> None:
    """One-sided rule (TRIALS.md s5): with the true trade skew and kurtosis stated, the realized power at the
    returned edge is at least 0.80 - 3 SE."""
    code, skew, kurt = shape
    r = min_edge(
        trades_per_year=m,
        trade_sd_cents=s,
        years=years,
        n_trials=n_trials,
        trade_skew=skew,
        trade_kurtosis=kurt,
    )
    assert r.min_edge_cents is not None
    share = _psr_share(m, s, r.days, n_trials, r.min_edge_cents, sims, seed=21, trade_skew=code)
    assert share >= POWER - 3 * _se(sims)


def test_mc_skewed_trades_under_normal_defaults_fall_short() -> None:
    """Why the shape must be stated: trades of skew -2 and raw kurtosis 9 checked with the normal defaults
    get realized power well under 0.80 (about 0.74) at the returned edge."""
    r = min_edge(trades_per_year=220.0, trade_sd_cents=21000, years=10.0, n_trials=1000)
    assert r.min_edge_cents == 2601
    sims = 4000
    share = _psr_share(220.0, 21000, r.days, 1000, r.min_edge_cents, sims, seed=5, trade_skew=-1)
    assert share < POWER - 3 * _se(sims)


ALPHA_POWER_SLACK = 0.05  # stated tolerance of the direct check: realized power in [0.80, 0.85] +- 3 SE


@pytest.mark.parametrize(
    ("m", "s", "years", "shape", "sims"),
    [
        (220.0, 21000, 10.0, NORMAL, 4000),
        (220.0, 21000, 10.0, NEG_SKEW, 4000),
        (252.0, 10000, 1.0, NEG_SKEW, 4000),
    ],
)
def test_mc_alpha_power_at_the_alpha_figure(
    m: float, s: int, years: float, shape: tuple[int, float, float], sims: int
) -> None:
    """The direct check by simulation: at min_edge_cents_alpha the alpha-only test, PSR(0) >=
    0.95, is passed by 0.80 to 0.85 of the runs (+- 3 SE), whatever N the figure was asked at."""
    code, skew, kurt = shape
    r = min_edge(trades_per_year=m, trade_sd_cents=s, years=years, n_trials=1000, trade_skew=skew,
                 trade_kurtosis=kurt)  # fmt: skip
    assert r.min_edge_cents_alpha is not None
    share = _psr_share(m, s, r.days, 1, r.min_edge_cents_alpha, sims, seed=21, trade_skew=code)
    assert POWER - 3 * _se(sims) <= share <= POWER + ALPHA_POWER_SLACK + 3 * _se(sims)


def test_mc_the_old_alpha_figure_was_above_the_band() -> None:
    """Why rule 1 was replaced: one year of skew -2 trades at N 1000, the old figure (2328) is passed by
    about 0.95 of the runs, far over the 0.80 it states."""
    old = min_edge(trades_per_year=252.0, trade_sd_cents=10000, years=1.0, n_trials=1000, trade_skew=-2.0,
                   trade_kurtosis=9.0, alpha_rule=LEGACY_ALPHA_RULE)  # fmt: skip
    assert old.min_edge_cents_alpha == 2328
    sims = 4000
    share = _psr_share(252.0, 10000, old.days, 1, old.min_edge_cents_alpha, sims, seed=21, trade_skew=-1)
    assert share > POWER + ALPHA_POWER_SLACK + 3 * _se(sims)
