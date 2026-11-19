"""The power check as the registry applies it (docs/TRIALS.md section 5): registration at N + budget, a
trial's begin at n_before + budget - configurations already counted, and the replay of both.

Years are those of ONE window (TRIALS.md section 5): at registration every declared window must
pass on its own and the record stores the figures of the shortest window (the largest minimum edge); at begin
and in replay the trial's own [first, last]. The figure is pfb.analysis.power.min_edge, a pure function, so
replay recomputes exactly what begin and registration wrote.

A book (schema 3) is checked as the BOOK, never as a member: Y is the union of the
member windows' days (book_years), at registration at N + K, at each member's begin at n_before + K - members
already counted, and in replay of both (book_hypothesis_power, book_begin_power).

`alpha_rule` reaches min_edge from the hypothesis record's power object only (the one place the
alpha-only figure is written). Registration writes ALPHA_RULE; replay also recomputes LEGACY_ALPHA_RULE for a
record written (pfb.registry.records). The gate figure is the same under both.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from pfb.analysis.power import ALPHA_RULE, DAYS_PER_CALENDAR_YEAR, PowerResult, min_edge, years_of
from pfb.registry.bookfile import union_days
from pfb.registry.errors import PowerInsufficient

POWER_INPUTS = (
    "trades_per_year",
    "trade_sd_cents",
    "trade_skew",
    "trade_kurtosis",
    "sd_source",
    "target_edge_cents",
)
HYPOTHESIS_POWER_KEYS = frozenset(POWER_INPUTS) | {
    "years",
    "n_used",
    "min_edge_cents",
    "min_edge_cents_alpha",
}
BEGIN_POWER_KEYS = frozenset({"n_used", "min_edge_cents", "target_edge_cents"})


def window_years(windows: Sequence[Sequence[str]]) -> float:
    """Years (inclusive calendar days / 365.25) of exactly one [first, last] window; never a sum."""
    if len(windows) != 1:
        raise ValueError(f"power years are those of one window, never a sum: got {len(windows)} windows")
    return years_of([(date.fromisoformat(first), date.fromisoformat(last)) for first, last in windows])


def window_days(window: Sequence[str]) -> int:
    first, last = window
    return (date.fromisoformat(last) - date.fromisoformat(first)).days + 1


def shortest_window(windows: Sequence[Sequence[str]]) -> Sequence[str]:
    """The declared window with the fewest days (the first on a tie): its minimum edge is the largest."""
    if not windows:
        raise ValueError("windows must name at least one [first, last] pair")
    return min(windows, key=window_days)


def power_at(
    inputs: Mapping[str, Any], windows: Sequence[Sequence[str]], n_used: int, alpha_rule: int = ALPHA_RULE
) -> PowerResult:
    """min_edge for the hypothesis's stated inputs over ONE window (a one-element list) at N = n_used."""
    return min_edge(
        trades_per_year=float(inputs["trades_per_year"]),
        trade_sd_cents=int(inputs["trade_sd_cents"]),
        trade_skew=float(inputs["trade_skew"]),
        trade_kurtosis=float(inputs["trade_kurtosis"]),
        years=window_years(windows),
        n_trials=n_used,
        target_edge_cents=int(inputs["target_edge_cents"]),
        alpha_rule=alpha_rule,
    )


def require_power(result: PowerResult, where: str) -> None:
    """PowerInsufficient unless the target edge reaches the minimum detectable edge."""
    if result.min_edge_cents is None:
        raise PowerInsufficient(
            f"{where}: no edge is detectable at N {result.n_used} with these trades per year, sd and years"
        )
    if result.passed is not True:
        raise PowerInsufficient(
            f"{where}: target_edge_cents {result.target_edge_cents} is below the minimum detectable edge "
            f"{result.min_edge_cents} at N {result.n_used}"
        )


def require_each_window(
    inputs: Mapping[str, Any], windows: Sequence[Sequence[str]], n_used: int, where: str
) -> None:
    """Registration and its replay: every declared window passes the power check on its own."""
    for first, last in windows:
        require_power(power_at(inputs, [[first, last]], n_used), f"{where}, window {first}..{last}")


def hypothesis_power(
    inputs: Mapping[str, Any], windows: Sequence[Sequence[str]], n_used: int, alpha_rule: int = ALPHA_RULE
) -> dict[str, Any]:
    """The hypothesis record's power object: the four inputs, years, n_used and both minimum edges, all of
    the shortest declared window."""
    result = power_at(inputs, [shortest_window(windows)], n_used, alpha_rule)
    return {
        **{key: inputs[key] for key in POWER_INPUTS},
        "years": result.years,
        "n_used": n_used,
        "min_edge_cents": result.min_edge_cents,
        "min_edge_cents_alpha": result.min_edge_cents_alpha,
    }


def book_years(member_windows: Sequence[Sequence[str]]) -> float:
    """A book's Y: the inclusive calendar days in at least one member window / 365.25
    (days counted once however many members cover them; never a sum over members)."""
    return union_days(member_windows) / DAYS_PER_CALENDAR_YEAR


def book_power_at(
    inputs: Mapping[str, Any],
    member_windows: Sequence[Sequence[str]],
    n_used: int,
    alpha_rule: int = ALPHA_RULE,
) -> PowerResult:
    """min_edge of a BOOK at N = n_used over Y = book_years, on the stated [power] inputs (which
    pfb.registry.bookfile.require_not_flattered binds to be no more flattering than the members derive)."""
    return min_edge(
        trades_per_year=float(inputs["trades_per_year"]),
        trade_sd_cents=int(inputs["trade_sd_cents"]),
        trade_skew=float(inputs["trade_skew"]),
        trade_kurtosis=float(inputs["trade_kurtosis"]),
        years=book_years(member_windows),
        n_trials=n_used,
        target_edge_cents=int(inputs["target_edge_cents"]),
        alpha_rule=alpha_rule,
    )


def book_hypothesis_power(
    inputs: Mapping[str, Any],
    member_windows: Sequence[Sequence[str]],
    n_used: int,
    where: str,
    alpha_rule: int = ALPHA_RULE,
) -> dict[str, Any]:
    """A book's hypothesis record power object (the keys of hypothesis_power) over the union of the members'
    windows; PowerInsufficient when the book's check fails."""
    result = book_power_at(inputs, member_windows, n_used, alpha_rule)
    require_power(result, where)
    return {
        **{key: inputs[key] for key in POWER_INPUTS},
        "years": result.years,
        "n_used": n_used,
        "min_edge_cents": result.min_edge_cents,
        "min_edge_cents_alpha": result.min_edge_cents_alpha,
    }


def book_begin_power(
    inputs: Mapping[str, Any], member_windows: Sequence[Sequence[str]], n_used: int, where: str
) -> dict[str, Any]:
    """A book member's begin power object: the BOOK's check (never the member's alone) at n_used."""
    result = book_power_at(inputs, member_windows, n_used)
    require_power(result, where)
    return {
        "n_used": n_used,
        "min_edge_cents": result.min_edge_cents,
        "target_edge_cents": int(inputs["target_edge_cents"]),
    }


def begin_power(inputs: Mapping[str, Any], windows: Sequence[Sequence[str]], n_used: int) -> dict[str, Any]:
    """A trial begin record's power object over the trial's own window (a one-element list): n_used,
    min_edge_cents, target_edge_cents."""
    result = power_at(inputs, windows, n_used)
    return {
        "n_used": n_used,
        "min_edge_cents": result.min_edge_cents,
        "target_edge_cents": int(inputs["target_edge_cents"]),
    }
