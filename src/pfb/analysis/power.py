"""Power check (S3): the minimum net edge per trade a hypothesis can detect at the live N (docs/TRIALS.md s5).

One pure function used by registration, begin, replay and the suite: no I/O, no registry read, no clock,
so the same arguments always give the same figure. Inputs: trades per year m, per-trade net sd s (cents),
years Y (of ONE window, never a sum: pfb.registry.powercheck.window_years), N_used, and the
per-trade skew and raw kurtosis (defaults 0 and 3, normal trades). Constants: alpha 0.05 one-sided, power
0.80.

    T = max(2, round(252 Y));  n = m Y;  b = pfb.metrics.stats.bracket(N_used)
    z = norm_ppf(0.95) + norm_ppf(0.80);  lam = m / 252
    g3 = trade_skew / sqrt(lam);  g4 = 3 + trade_kurtosis / lam          # daily moments, compound Poisson
    K(x) = sqrt(max(1, 1 - g3 x + ((g4 - 1) / 4) x^2))                   # the Sharpe estimator's sd factor
    s0 = (b + z) / sqrt(T - 1);  k = K(s0);  t_dsr = b + z k             # the gate: at the deflated Sharpe
    sa = z / sqrt(T - 1);  ka = K(sa);  t_alpha = z ka                   # alpha only: the same with b = 0
    c = t / sqrt(n (T - 1) / T);  edge = c s / sqrt(1 - c^2) if c < 1 else undetectable
    min_edge_cents = ceil(edge at t_dsr);  min_edge_cents_alpha = ceil(edge at t_alpha)

The daily variance is the compound-Poisson one (trades arrive at random), and the DSR bar uses the floor
variance 1 / (T - 1), so the figure is a lower bound: a wider spread of Sharpe across the grid raises the real
bar. The stated trade skew and kurtosis enter the Sharpe estimator's variance; k never falls below 1, so a
positive skew cannot lower the bar under the normal case. With the defaults the gate formula is the
original. `round` is Python's round (half to even); it only matters when 252 Y lands exactly on a half.

Alpha rule. Rule 2, the default: the alpha-only figure takes its variance factor at
the alpha test's own Sharpe sa, so it is the gate formula with nothing deflated, the same at every N_used and
equal to min_edge_cents at N_used = 1. Rule 1 took t_alpha = z k with the gate's k, which
rose with N_used and overstated the figure; it is kept only so replay can recompute a hypothesis record
written under it (pfb.registry.records). min_edge_cents, t_dsr and passed are the same under both rules.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date

from pfb.metrics.stats import bracket, norm_ppf

ALPHA = 0.05
POWER = 0.80
TRADING_DAYS_PER_YEAR = 252
DAYS_PER_CALENDAR_YEAR = 365.25
MIN_DAYS = 2
NORMAL_RAW_KURTOSIS = 3.0
MIN_RAW_KURTOSIS = 1.0
ALPHA_RULE = 2  # the alpha-only variance factor at z / sqrt(T - 1)
LEGACY_ALPHA_RULE = 1  # and D: the gate's factor (at the deflated Sharpe) for the alpha figure too
ALPHA_RULES = (LEGACY_ALPHA_RULE, ALPHA_RULE)


class PowerInputError(ValueError):
    """An input of min_edge is out of range (a ValueError: registration and begin catch it as before)."""


@dataclass(frozen=True)
class PowerResult:
    """The power check at one N. A None edge means undetectable (c >= 1): no edge reaches the bar."""

    n_used: int
    years: float
    days: int
    expected_trades: float
    bracket: float
    t_dsr: float
    t_alpha: float
    min_edge_cents: int | None
    min_edge_cents_alpha: int | None
    target_edge_cents: int | None
    passed: bool | None
    trades_per_year: float
    trade_sd_cents: int
    trade_skew: float
    trade_kurtosis: float

    def to_json(self) -> dict[str, object]:
        """JSON-ready dict (int, float, bool or None values)."""
        return asdict(self)


def years_of(windows: Sequence[tuple[date, date]]) -> float:
    """Sum of inclusive calendar days / 365.25 over the windows; refuses an empty or a reversed window."""
    if not windows:
        raise ValueError("windows must name at least one (first, last) pair")
    total_days = 0
    for first, last in windows:
        if last < first:
            raise ValueError(f"window {first}..{last}: last is before first")
        total_days += (last - first).days + 1
    return total_days / DAYS_PER_CALENDAR_YEAR


def _check_inputs(
    trades_per_year: float, trade_sd_cents: int, years: float, n_trials: int, target_edge_cents: int | None
) -> None:
    if isinstance(trades_per_year, bool) or not math.isfinite(trades_per_year) or trades_per_year <= 0:
        raise PowerInputError(f"trades_per_year {trades_per_year!r} must be a finite number > 0")
    if isinstance(trade_sd_cents, bool) or not isinstance(trade_sd_cents, int) or trade_sd_cents <= 0:
        raise PowerInputError(f"trade_sd_cents {trade_sd_cents!r} must be an int > 0")
    if isinstance(years, bool) or not math.isfinite(years) or years <= 0:
        raise PowerInputError(f"years {years!r} must be a finite number > 0")
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        raise PowerInputError(f"n_trials {n_trials!r} must be an int >= 1")
    if target_edge_cents is not None and (
        isinstance(target_edge_cents, bool) or not isinstance(target_edge_cents, int)
    ):
        raise PowerInputError(f"target_edge_cents {target_edge_cents!r} must be an int or None")


def _check_shape(trade_skew: float, trade_kurtosis: float) -> None:
    if (
        isinstance(trade_skew, bool)
        or not isinstance(trade_skew, int | float)
        or not math.isfinite(trade_skew)
    ):
        raise PowerInputError(f"trade_skew {trade_skew!r} must be a finite number")
    if (
        isinstance(trade_kurtosis, bool)
        or not isinstance(trade_kurtosis, int | float)
        or not math.isfinite(trade_kurtosis)
        or trade_kurtosis < MIN_RAW_KURTOSIS
    ):
        raise PowerInputError(
            f"trade_kurtosis {trade_kurtosis!r} (raw) must be finite and >= {MIN_RAW_KURTOSIS}"
        )


def _sd_factor(sharpe: float, g3: float, g4: float) -> float:
    """K(x): sd of the Sharpe estimator x sqrt(T - 1) at daily Sharpe x, daily skew g3 and raw kurtosis g4;
    never below 1."""
    return math.sqrt(max(1.0, 1.0 - g3 * sharpe + ((g4 - 1.0) / 4.0) * sharpe * sharpe))


def _edge_cents(t: float, expected_trades: float, days: int, trade_sd_cents: int) -> int | None:
    """ceil(c s / sqrt(1 - c^2)) with c = t / sqrt(n (T - 1) / T); None when c >= 1."""
    c = t / math.sqrt(expected_trades * (days - 1) / days)
    if c >= 1.0:
        return None
    return math.ceil(c * trade_sd_cents / math.sqrt(1.0 - c * c))


def min_edge(
    *,
    trades_per_year: float,
    trade_sd_cents: int,
    years: float,
    n_trials: int,
    target_edge_cents: int | None = None,
    trade_skew: float = 0.0,
    trade_kurtosis: float = NORMAL_RAW_KURTOSIS,
    alpha_rule: int = ALPHA_RULE,
) -> PowerResult:
    """The minimum detectable net edge per trade (cents) at N = n_trials; see the module docstring.

    passed is False when the edge is undetectable, None when no target is given and the edge is detectable,
    else target_edge_cents >= min_edge_cents (integer comparison). alpha_rule changes min_edge_cents_alpha
    and t_alpha only (LEGACY_ALPHA_RULE: the figure as written, for replay).
    """
    _check_inputs(trades_per_year, trade_sd_cents, years, n_trials, target_edge_cents)
    _check_shape(trade_skew, trade_kurtosis)
    if isinstance(alpha_rule, bool) or alpha_rule not in ALPHA_RULES:
        raise PowerInputError(f"alpha_rule {alpha_rule!r} must be one of {ALPHA_RULES}")
    days = max(MIN_DAYS, round(TRADING_DAYS_PER_YEAR * years))
    expected_trades = trades_per_year * years
    b = bracket(n_trials)
    z = norm_ppf(1.0 - ALPHA) + norm_ppf(POWER)
    lam = trades_per_year / TRADING_DAYS_PER_YEAR
    g3 = trade_skew / math.sqrt(lam)
    g4 = NORMAL_RAW_KURTOSIS + trade_kurtosis / lam
    k = _sd_factor((b + z) / math.sqrt(days - 1), g3, g4)
    k_alpha = k if alpha_rule == LEGACY_ALPHA_RULE else _sd_factor(z / math.sqrt(days - 1), g3, g4)
    t_dsr = b + z * k
    t_alpha = z * k_alpha
    edge = _edge_cents(t_dsr, expected_trades, days, trade_sd_cents)
    edge_alpha = _edge_cents(t_alpha, expected_trades, days, trade_sd_cents)
    if edge is None:
        passed: bool | None = False
    elif target_edge_cents is None:
        passed = None
    else:
        passed = target_edge_cents >= edge
    return PowerResult(
        n_used=n_trials,
        years=years,
        days=days,
        expected_trades=expected_trades,
        bracket=b,
        t_dsr=t_dsr,
        t_alpha=t_alpha,
        min_edge_cents=edge,
        min_edge_cents_alpha=edge_alpha,
        target_edge_cents=target_edge_cents,
        passed=passed,
        trades_per_year=float(trades_per_year),
        trade_sd_cents=trade_sd_cents,
        trade_skew=float(trade_skew),
        trade_kurtosis=float(trade_kurtosis),
    )
