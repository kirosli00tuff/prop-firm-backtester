"""Combinatorial purged cross-validation (CPCV) of a declared grid's selection.

Definition (Lopez de Prado; worked from the definition, no text fetched): cut the T rows (trading
days, in time order) into N contiguous groups; every choice of k groups is one split's test set (C(N, k)
splits) and the other groups, less the purged and embargoed rows, its train set. Each group is tested in
C(N-1, k-1) splits, so the test results assemble into phi = k / N * C(N, k) = C(N-1, k-1) backtest paths,
each covering all T rows once: path p takes group g from the p-th split (in lexicographic order of the
test-group combinations) that tests g.

Here the "model" fitted on a train set is the selection of one configuration of the grid: the configuration
with the best train performance (pfb.analysis.pbo's metric and is_tie conventions). A path is the daily net
series of the configurations chosen, stitched from the test groups; its performance distribution over the
phi paths is the out-of-sample distribution of the selection procedure, not of any one configuration.

THE CALLER OWES THE WHOLE DECLARED GRID (as for PBO: a subset understates what the selection costs); the
array function refuses K < 2, and `for_hypothesis` reads every declared configuration or refuses.

Conventions (keyword arguments with the defaults named here, echoed in the result's "conventions"):
- n_groups=6, k_test=2 (15 splits, 5 paths): the textbook illustration; k_test < n_groups.
- groups: T // N rows each, the remainder to the LAST group (pfb.analysis.walkforward.fold_bounds).
- purge_days=1: the trading days just BEFORE each test group are dropped from train (a position held over
  night carries the train day's decision into the test group); embargo_days=5 (one trading week): the
  trading days just AFTER each test group are dropped from train (serial dependence leaking test into train).
  Both are counted in rows of the daily matrix, i.e. trading days. Rows of another test group are never
  dropped.
- min_train_rows=2: a split whose purged train set has fewer rows is refused.
- metric="sharpe" (per-day, population sd, sd 0 gives 0), is_tie="worst" (an in-sample tie picks the one
  worst on the split's test rows): pfb.analysis.pbo conventions.

FIXED FOR A HYPOTHESIS: cpcv_select keeps its keyword conventions; the
registry entry for_hypothesis takes NONE and applies HYPOTHESIS_CONVENTIONS (the defaults above, module
constants), echoed in "conventions". Its matrix is pbo.grid_matrix's, each daily file checked against its
run's result record (pbo.GridFileMismatch): by the sha256 in the record's artifact_sha256 where it has one,
by the sum of net cents and the row count for a record written before that key.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from pfb.analysis.pbo import (
    IS_TIES,
    METRICS,
    PboError,
    check_matrix,
    choose,
    column_performance,
    grid_matrix,
)
from pfb.analysis.walkforward import fold_bounds
from pfb.registry.api import RegistryConfig

DEFAULT_GROUPS = 6
DEFAULT_TEST_GROUPS = 2
DEFAULT_PURGE_DAYS = 1
DEFAULT_EMBARGO_DAYS = 5
DEFAULT_MIN_TRAIN_ROWS = 2
# the only conventions for_hypothesis applies: cpcv_select's defaults
HYPOTHESIS_CONVENTIONS: Mapping[str, Any] = MappingProxyType(
    {
        "n_groups": DEFAULT_GROUPS,
        "k_test": DEFAULT_TEST_GROUPS,
        "purge_days": DEFAULT_PURGE_DAYS,
        "embargo_days": DEFAULT_EMBARGO_DAYS,
        "metric": "sharpe",
        "is_tie": "worst",
        "min_train_rows": DEFAULT_MIN_TRAIN_ROWS,
    }
)


@dataclass(frozen=True)
class CpcvSplit:
    index: int
    test_groups: tuple[int, ...]
    test_bounds: tuple[tuple[int, int], ...]  # [start, stop) rows of each test group
    test_rows: np.ndarray
    train_rows: np.ndarray
    removed_rows: int  # rows dropped from train by purge and embargo


def _count(name: str, value: object, low: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < low:
        raise PboError(f"{name} {value!r} must be an int >= {low}")
    return value


def _check_plan(n_groups: int, k_test: int, purge_days: int, embargo_days: int) -> None:
    _count("n_groups", n_groups, 2)
    _count("k_test", k_test, 1)
    _count("purge_days", purge_days, 0)
    _count("embargo_days", embargo_days, 0)
    if k_test >= n_groups:
        raise PboError(f"k_test {k_test} must be below n_groups {n_groups} (a split needs a train group)")


def n_paths(n_groups: int, k_test: int) -> int:
    """phi = k / N * C(N, k) = C(N - 1, k - 1)."""
    return math.comb(n_groups - 1, k_test - 1)


def splits(
    t: int,
    n_groups: int,
    k_test: int,
    *,
    purge_days: int = DEFAULT_PURGE_DAYS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
) -> tuple[CpcvSplit, ...]:
    """Every split, in lexicographic order of its test groups, with purged and embargoed train rows."""
    _check_plan(n_groups, k_test, purge_days, embargo_days)
    try:
        bounds = fold_bounds(t, n_groups)
    except ValueError as exc:
        raise PboError(str(exc)) from exc
    out: list[CpcvSplit] = []
    for index, groups in enumerate(itertools.combinations(range(n_groups), k_test)):
        test = np.zeros(t, dtype=bool)
        dropped = np.zeros(t, dtype=bool)
        for g in groups:
            start, stop = bounds[g]
            test[start:stop] = True
            dropped[max(0, start - purge_days) : start] = True
            dropped[stop : min(t, stop + embargo_days)] = True
        dropped &= ~test
        out.append(
            CpcvSplit(
                index=index,
                test_groups=tuple(groups),
                test_bounds=tuple(bounds[g] for g in groups),
                test_rows=np.flatnonzero(test),
                train_rows=np.flatnonzero(~test & ~dropped),
                removed_rows=int(dropped.sum()),
            )
        )
    return tuple(out)


def path_table(n_groups: int, k_test: int) -> np.ndarray:
    """(phi, N) split indices: path p takes group g from the p-th split (lexicographic) that tests g."""
    combos = list(itertools.combinations(range(n_groups), k_test))
    table = np.zeros((n_paths(n_groups, k_test), n_groups), dtype=np.int64)
    for g in range(n_groups):
        table[:, g] = [i for i, combo in enumerate(combos) if g in combo]
    return table


@dataclass(frozen=True)
class CpcvResult:
    n_splits: int
    n_paths: int
    k: int
    t: int
    chosen: tuple[int, ...]  # per split
    removed_rows: tuple[int, ...]  # per split
    path_series: tuple[np.ndarray, ...]  # per path, T values
    path_perf: tuple[float, ...]
    path_net: tuple[float, ...]
    conventions: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        perf = np.asarray(self.path_perf)
        return {
            "n_splits": self.n_splits,
            "n_paths": self.n_paths,
            "k": self.k,
            "t": self.t,
            "chosen": list(self.chosen),
            "removed_rows": list(self.removed_rows),
            "path_perf": list(self.path_perf),
            "path_net": list(self.path_net),
            "perf_mean": float(perf.mean()),
            "perf_sd": float(perf.std()),
            "perf_min": float(perf.min()),
            "perf_max": float(perf.max()),
            "paths_not_positive": float((perf <= 0).mean()),
            "conventions": dict(self.conventions),
        }


def cpcv_select(
    matrix: np.ndarray | Sequence[Sequence[float]],
    *,
    n_groups: int = DEFAULT_GROUPS,
    k_test: int = DEFAULT_TEST_GROUPS,
    purge_days: int = DEFAULT_PURGE_DAYS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    metric: str = "sharpe",
    is_tie: str = "worst",
    min_train_rows: int = DEFAULT_MIN_TRAIN_ROWS,
) -> CpcvResult:
    """CPCV of best-in-train selection over a T x K matrix (columns = EVERY configuration of the declared
    grid, K >= 2; the caller owes the whole grid). Conventions: module docstring."""
    if metric not in METRICS or is_tie not in IS_TIES:
        raise PboError(f"metric {metric!r} / is_tie {is_tie!r} must be in {METRICS} / {IS_TIES}")
    _check_plan(n_groups, k_test, purge_days, embargo_days)
    _count("min_train_rows", min_train_rows, 1)
    arr = check_matrix(matrix)
    t, k = arr.shape
    plan = splits(t, n_groups, k_test, purge_days=purge_days, embargo_days=embargo_days)
    chosen: list[int] = []
    for split in plan:
        if len(split.train_rows) < min_train_rows:
            raise PboError(
                f"split {split.test_groups}: {len(split.train_rows)} train rows after purge and "
                f"embargo, {min_train_rows} needed"
            )
        ins = column_performance(arr[split.train_rows], metric)[None, :]
        outs = column_performance(arr[split.test_rows], metric)[None, :]
        chosen.append(int(choose(ins, outs, is_tie)[0]))
    table = path_table(n_groups, k_test)
    series: list[np.ndarray] = []
    for row in table:
        path = np.empty(t, dtype=np.float64)
        for g, s in enumerate(row):
            start, stop = plan[s].test_bounds[plan[s].test_groups.index(g)]
            path[start:stop] = arr[start:stop, chosen[s]]
        series.append(path)
    return CpcvResult(
        n_splits=len(plan),
        n_paths=len(table),
        k=k,
        t=t,
        chosen=tuple(chosen),
        removed_rows=tuple(s.removed_rows for s in plan),
        path_series=tuple(series),
        path_perf=tuple(float(column_performance(p[:, None], metric)[0]) for p in series),
        path_net=tuple(float(p.sum()) for p in series),
        conventions={
            "n_groups": n_groups,
            "k_test": k_test,
            "purge_days": purge_days,
            "embargo_days": embargo_days,
            "metric": metric,
            "is_tie": is_tie,
            "min_train_rows": min_train_rows,
            "groups": "T // N rows each, remainder to the last group",
        },
    )


def for_hypothesis(
    hypothesis_id: str,
    *,
    config: RegistryConfig,
    runs_dir: Path | None = None,
    root: str | None = None,
    window: tuple[str | date, str | date] | None = None,
) -> dict[str, object]:
    """CPCV of one registered hypothesis's whole declared grid (pbo.grid_matrix); read-only, N unchanged.
    NO convention argument: HYPOTHESIS_CONVENTIONS always, echoed under cpcv.conventions."""
    grid = grid_matrix(config, hypothesis_id, runs_dir=runs_dir, root=root, window=window)
    return {**grid.describe(), "cpcv": cpcv_select(grid.matrix, **HYPOTHESIS_CONVENTIONS).to_dict()}
