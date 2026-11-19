"""CPCV: split and path counts by formula, purge and embargo, selection on synthetic data."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from pfb.analysis.cpcv import cpcv_select, n_paths, path_table, splits
from pfb.analysis.pbo import PboError


@pytest.mark.parametrize("n_groups, k_test", [(2, 1), (6, 2), (6, 3), (8, 2), (10, 4), (5, 4)])
def test_split_and_path_counts_follow_the_formula(n_groups: int, k_test: int) -> None:
    plan = splits(10 * n_groups, n_groups, k_test, purge_days=0, embargo_days=0)
    assert len(plan) == math.comb(n_groups, k_test)
    phi = n_paths(n_groups, k_test)
    assert phi == k_test * math.comb(n_groups, k_test) // n_groups == math.comb(n_groups - 1, k_test - 1)
    table = path_table(n_groups, k_test)
    assert table.shape == (phi, n_groups)
    for g in range(n_groups):
        # each path takes group g from a split that tests g, and each such split serves exactly one path
        assert all(g in plan[s].test_groups for s in table[:, g])
        assert sorted(table[:, g].tolist()) == [s.index for s in plan if g in s.test_groups]


@pytest.mark.parametrize("purge, embargo", [(0, 0), (1, 5), (3, 0), (0, 4), (7, 7)])
def test_purged_train_never_overlaps_test_and_the_gaps_are_where_stated(purge: int, embargo: int) -> None:
    t, n_groups = 63, 6  # groups of 10 rows, the last of 13
    for split in splits(t, n_groups, 2, purge_days=purge, embargo_days=embargo):
        train, test = set(split.train_rows.tolist()), set(split.test_rows.tolist())
        assert not train & test
        gap: set[int] = set()
        for start, stop in split.test_bounds:
            gap |= set(range(max(0, start - purge), start)) | set(range(stop, min(t, stop + embargo)))
        gap -= test
        assert not train & gap  # purged before each test group, embargoed after it
        assert train | test | gap == set(range(t))  # nothing else is dropped
        assert split.removed_rows == len(gap)
        assert list(split.train_rows) == sorted(train)


def test_adjacent_test_groups_lose_no_test_rows_to_the_purge() -> None:
    plan = splits(60, 6, 2, purge_days=2, embargo_days=2)
    adjacent = next(s for s in plan if s.test_groups == (1, 2))
    assert adjacent.test_rows.tolist() == list(range(10, 30))
    assert set(adjacent.train_rows.tolist()) == set(range(0, 8)) | set(range(32, 60))


def test_paths_cover_every_row_once_and_one_better_configuration_wins_every_split() -> None:
    rng = np.random.default_rng(5)
    m = rng.normal(0, 100, size=(600, 6))
    m[:, 2] += 40
    res = cpcv_select(m, n_groups=6, k_test=2)
    assert res.n_splits == 15 and res.n_paths == 5
    assert set(res.chosen) == {2}
    for path in res.path_series:
        assert path.shape == (600,)
        np.testing.assert_array_equal(path, m[:, 2])  # every split chose 2, so each path is column 2
    assert min(res.path_perf) > 0


def test_pure_noise_paths_centre_near_zero() -> None:
    rng = np.random.default_rng(9)
    m = rng.normal(0, 100, size=(1200, 8))
    res = cpcv_select(m, n_groups=8, k_test=2)
    assert abs(float(np.mean(res.path_perf))) < 0.06
    json.dumps(res.to_dict())


def test_a_path_takes_each_groups_rows_from_its_own_splits_choice() -> None:
    # 4 groups of 3 rows, k = 2: column 0 is best in groups 0-1, column 1 in groups 2-3 (metric mean).
    m = np.array([[5, 0]] * 6 + [[0, 5]] * 6, dtype=np.int64)
    res = cpcv_select(m, n_groups=4, k_test=2, purge_days=0, embargo_days=0, metric="mean", min_train_rows=1)
    table = path_table(4, 2)
    plan = splits(12, 4, 2, purge_days=0, embargo_days=0)
    for p, path in enumerate(res.path_series):
        for g in range(4):
            s = int(table[p, g])
            col = res.chosen[s]
            start, stop = plan[s].test_bounds[plan[s].test_groups.index(g)]
            np.testing.assert_array_equal(path[start:stop], m[start:stop, col])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n_groups": 1}, "n_groups"),
        ({"k_test": 0}, "k_test"),
        ({"n_groups": 4, "k_test": 4}, "train"),
        ({"purge_days": -1}, "purge_days"),
        ({"embargo_days": True}, "embargo_days"),
    ],
)
def test_bad_plans_are_refused(kwargs: dict, match: str) -> None:
    with pytest.raises(PboError, match=match):
        cpcv_select(np.zeros((60, 3)), **kwargs)


def test_k_below_two_is_refused() -> None:
    with pytest.raises(PboError, match="at least 2 configurations"):
        cpcv_select(np.zeros((60, 1)))


def test_a_train_set_too_small_after_purge_is_refused() -> None:
    with pytest.raises(PboError, match="train rows"):
        cpcv_select(np.zeros((12, 3)), n_groups=4, k_test=3, purge_days=2, embargo_days=2)
