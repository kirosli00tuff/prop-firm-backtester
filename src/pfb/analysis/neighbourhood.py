"""Neighbourhood (S5): one-step neighbours of a grid point and the plateau / spike verdict.

Pure functions: no bar read, no registry access. The runner begins each neighbour as its own counted trial
(docs/TRIALS.md C3); this module only plans the parameter sets and judges the finished run summaries.

neighbours: for each parameter whose grid is a strictly increasing numeric list (int or float, never bool),
the centre with that parameter one step down and one step up on the grid; a centre at an edge of the list
gives one neighbour for that parameter, any other parameter gives none. Order: grid key order, down before up.

verdict: s0 = the centre's sharpe_annual, r = mean neighbour sharpe_annual / s0.
    "n/a"      no neighbour, or s0 is None or s0 <= 0
    "plateau"  r >= 0.7 and every neighbour's net_cents has the centre's sign
    "spike"    r < 0.5, or at least half the neighbours' net_cents have the other sign
    "mixed"    otherwise
A neighbour whose sharpe_annual is None (no variance, e.g. no trade) counts as Sharpe 0.0, the conservative
reading. A net of zero has neither sign: it blocks "plateau" and does not count toward "spike".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction

PLATEAU_RATIO = 0.7
SPIKE_RATIO = 0.5
SPIKE_OTHER_SIGN_SHARE = Fraction(1, 2)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _stepped(values: Sequence[object]) -> bool:
    """True when values is a numeric list of length >= 2, strictly increasing."""
    if len(values) < 2 or not all(_is_number(v) for v in values):
        return False
    nums = [float(v) for v in values]  # type: ignore[arg-type]  # numbers checked above
    return all(a < b for a, b in zip(nums, nums[1:], strict=False))


def _check_centre(grid: Mapping[str, Sequence[object]], centre: Mapping[str, object]) -> None:
    if set(centre) != set(grid):
        missing, extra = sorted(set(grid) - set(centre)), sorted(set(centre) - set(grid))
        raise ValueError(f"centre must name exactly the grid parameters (missing {missing}, extra {extra})")
    for name, values in grid.items():
        if centre[name] not in list(values):
            raise ValueError(f"centre {name}={centre[name]!r} is not a point of the grid {list(values)!r}")


def neighbours(grid: Mapping[str, Sequence[object]], centre: Mapping[str, object]) -> list[dict[str, object]]:
    """The one-step neighbours of centre on grid (module docstring); ValueError on a foreign centre."""
    _check_centre(grid, centre)
    out: list[dict[str, object]] = []
    for name, values in grid.items():
        points = list(values)
        if not _stepped(points):
            continue
        i = points.index(centre[name])
        for j in (i - 1, i + 1):
            if 0 <= j < len(points):
                out.append({**centre, name: points[j]})
    return out


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _sharpe(summary: Mapping[str, object], label: str) -> float | None:
    value = summary.get("sharpe_annual")
    if value is None:
        return None
    if not _is_number(value):
        raise ValueError(f"{label} sharpe_annual {value!r} must be a number or None")
    return float(value)  # type: ignore[arg-type]  # number checked above


def _net(summary: Mapping[str, object], label: str) -> int:
    value = summary.get("net_cents")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} net_cents {value!r} must be an int")
    return value


def _thresholds() -> dict[str, object]:
    return {
        "plateau_ratio": PLATEAU_RATIO,
        "spike_ratio": SPIKE_RATIO,
        "spike_other_sign_share": float(SPIKE_OTHER_SIGN_SHARE),
    }


def verdict(centre: Mapping[str, object], around: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Plateau / spike / mixed / n/a from run summaries (keys sharpe_annual and net_cents)."""
    s0 = _sharpe(centre, "centre")
    centre_sign = _sign(_net(centre, "centre"))
    sharpes = [_sharpe(n, f"neighbour {i}") for i, n in enumerate(around)]
    signs = [_sign(_net(n, f"neighbour {i}")) for i, n in enumerate(around)]
    filled = [0.0 if s is None else s for s in sharpes]
    n_same = sum(1 for s in signs if s == centre_sign and s != 0)
    n_other = sum(1 for s in signs if s == -centre_sign and s != 0)
    out: dict[str, object] = {
        "verdict": "n/a",
        "ratio": None,
        "n_neighbours": len(around),
        "centre_sharpe": s0,
        "mean_neighbour_sharpe": sum(filled) / len(filled) if filled else None,
        "n_same_sign": n_same,
        "n_other_sign": n_other,
        "thresholds": _thresholds(),
    }
    if not around or s0 is None or s0 <= 0:
        return out
    ratio = (sum(filled) / len(filled)) / s0
    if ratio >= PLATEAU_RATIO and n_same == len(around):
        label = "plateau"
    elif ratio < SPIKE_RATIO or Fraction(n_other, len(around)) >= SPIKE_OTHER_SIGN_SHARE:
        label = "spike"
    else:
        label = "mixed"
    out.update(verdict=label, ratio=ratio)
    return out
