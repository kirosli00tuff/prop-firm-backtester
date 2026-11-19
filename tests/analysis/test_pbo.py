"""PBO by CSCV: known answers on synthetic matrices only (no ledger, no data)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pfb.analysis.pbo import (
    DEFAULT_BLOCKS,
    PboError,
    block_layout,
    column_performance,
    pbo_cscv,
    split_masks,
)

# A hand case, metric "mean", S = 2 blocks of 2 rows, K = 4 (worked in the module docstring's terms):
# block means B0 = (3, 1, -5, 0), B1 = (-1, 2, -5, 0).
# Split 1, in-sample B0: best is column 0 (3); out of sample B1 it is 2nd of 4 -> w = 2/5, logit ln(2/3) < 0.
# Split 2, in-sample B1: best is column 1 (2); out of sample B0 it is 3rd of 4 -> w = 3/5, logit ln(3/2) > 0.
# PBO = 1/2.
HAND = np.array(
    [
        [3, 1, -5, 0],
        [3, 1, -5, 0],
        [-1, 2, -5, 0],
        [-1, 2, -5, 0],
    ],
    dtype=np.int64,
)


def test_the_hand_case_gives_the_hand_logits_and_pbo_one_half() -> None:
    res = pbo_cscv(HAND, metric="mean", n_blocks=2, min_block_rows=1)
    assert res.n_splits == 2
    assert res.chosen == (0, 1)
    assert res.oos_rank == (2.0, 3.0)
    assert res.logits == pytest.approx((math.log(2 / 3), math.log(3 / 2)))
    assert res.pbo == 0.5
    assert res.prob_oos_loss == 0.5  # split 1 chose column 0, whose out-of-sample mean is -1


def test_a_logit_of_exactly_zero_counts_as_overfit() -> None:
    # K = 3: split 2 chooses column 1, out of sample (B0) it is 2nd of 3 -> w = 1/2, logit 0 -> counted.
    m = np.array([[3, 1, 0], [3, 1, 0], [-1, 2, 0], [-1, 2, 0]], dtype=np.int64)
    res = pbo_cscv(m, metric="mean", n_blocks=2, min_block_rows=1)
    assert res.logits[1] == 0.0
    assert res.pbo == 1.0


def test_an_in_sample_tie_takes_the_worst_out_of_sample_by_default() -> None:
    # Columns 0 and 1 tie in B0 (mean 2); out of sample (B1) column 0 is 5, column 1 is -5.
    m = np.array([[2, 2, 0], [2, 2, 0], [5, -5, 0], [5, -5, 0]], dtype=np.int64)
    worst = pbo_cscv(m, metric="mean", n_blocks=2, min_block_rows=1)
    first = pbo_cscv(m, metric="mean", n_blocks=2, min_block_rows=1, is_tie="first")
    assert worst.chosen[0] == 1 and worst.oos_rank[0] == 1.0
    assert first.chosen[0] == 0 and first.oos_rank[0] == 3.0
    assert worst.pbo >= first.pbo


def test_an_out_of_sample_tie_takes_the_lowest_rank_by_default() -> None:
    # In B0 column 0 is best; out of sample (B1) all three tie: rank 1 (min) or 2 (average).
    m = np.array([[9, 1, 0], [9, 1, 0], [4, 4, 4], [4, 4, 4]], dtype=np.int64)
    low = pbo_cscv(m, metric="mean", n_blocks=2, min_block_rows=1)
    avg = pbo_cscv(m, metric="mean", n_blocks=2, min_block_rows=1, oos_tie="average")
    assert low.oos_rank[0] == 1.0
    assert avg.oos_rank[0] == 2.0
    assert low.logits[0] < avg.logits[0] == 0.0


@pytest.mark.parametrize("s", [2, 4, 6, 8, 10, 16])
def test_the_split_count_is_s_choose_s_over_two_and_every_split_is_half(s: int) -> None:
    masks = split_masks(s)
    assert masks.shape == (math.comb(s, s // 2), s)
    assert (masks.sum(axis=1) == s // 2).all()
    assert len({row.tobytes() for row in masks}) == masks.shape[0]  # no split twice
    # symmetric: the complement of every split is a split too
    rows = {row.tobytes() for row in masks}
    assert all((~row).tobytes() in rows for row in masks)


def test_block_layout_drops_the_oldest_remainder_by_default() -> None:
    assert block_layout(35, 4, "drop_first", 2) == (3, 8)
    assert block_layout(35, 4, "drop_last", 2) == (0, 8)
    assert block_layout(32, 4, "refuse", 2) == (0, 8)
    with pytest.raises(PboError, match="not a multiple"):
        block_layout(35, 4, "refuse", 2)
    with pytest.raises(PboError, match="rows per block"):
        block_layout(7, 4, "drop_first", 2)


def test_the_result_says_which_rows_were_used() -> None:
    rng = np.random.default_rng(3)
    m = rng.normal(0, 1, size=(35, 3))
    res = pbo_cscv(m, n_blocks=4)
    assert (res.t, res.rows_used, res.rows_dropped, res.block_rows) == (35, 32, 3, 8)
    assert res.conventions["remainder"] == "drop_first"
    assert res.conventions["n_blocks"] == 4


def test_block_sharpe_equals_the_direct_sharpe_of_the_in_sample_rows() -> None:
    rng = np.random.default_rng(11)
    m = rng.integers(-5000, 5000, size=(48, 5)).astype(np.int64)
    m[:, 4] = 7  # a constant column: sd 0 gives Sharpe 0 (pfb.metrics.stats.moments)
    res = pbo_cscv(m, n_blocks=6)
    masks = split_masks(6)
    for c in (0, 7, 19):
        rows = np.concatenate([m[b * 8 : (b + 1) * 8] for b in range(6) if masks[c, b]])
        direct = column_performance(rows, "sharpe")
        assert res.is_perf[c] == pytest.approx(float(direct.max()), rel=1e-12, abs=1e-15)
    assert column_performance(m, "sharpe")[4] == 0.0


def test_pure_noise_gives_pbo_near_one_half() -> None:
    # K even, so under exchangeability P(rank <= K / 2) = 1/2 exactly; average over 40 seeds.
    values = []
    for seed in range(40):
        rng = np.random.default_rng(1000 + seed)
        values.append(pbo_cscv(rng.normal(0, 100, size=(400, 10)), n_blocks=8).pbo)
    mean = float(np.mean(values))
    assert abs(mean - 0.5) < 0.08, mean


def test_one_truly_better_configuration_gives_pbo_near_zero() -> None:
    values = []
    for seed in range(10):
        rng = np.random.default_rng(2000 + seed)
        m = rng.normal(0, 100, size=(1000, 10))
        m[:, 3] += 40  # daily Sharpe 0.4 against 0 for the rest
        res = pbo_cscv(m, n_blocks=DEFAULT_BLOCKS)
        values.append(res.pbo)
        assert all(c == 3 for c in res.chosen)
    assert max(values) < 0.02, values


@pytest.mark.parametrize(
    "bad, match",
    [
        (np.zeros((40, 1)), "at least 2 configurations"),
        (np.zeros(40), "2-D"),
        (np.full((40, 3), np.nan), "finite"),
        (np.zeros((0, 3)), "no rows"),
    ],
)
def test_bad_matrices_are_refused(bad: np.ndarray, match: str) -> None:
    with pytest.raises(PboError, match=match):
        pbo_cscv(bad, n_blocks=2)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n_blocks": 3}, "even"),
        ({"n_blocks": 0}, "even"),
        ({"metric": "sortino"}, "metric"),
        ({"is_tie": "best"}, "is_tie"),
        ({"oos_tie": "max"}, "oos_tie"),
        ({"remainder": "pad"}, "remainder"),
        ({"min_block_rows": 0}, "min_block_rows"),
    ],
)
def test_bad_conventions_are_refused(kwargs: dict, match: str) -> None:
    with pytest.raises(PboError, match=match):
        pbo_cscv(np.zeros((40, 3)), **kwargs)


def test_the_result_is_json_ready() -> None:
    import json

    res = pbo_cscv(HAND, metric="mean", n_blocks=2, min_block_rows=1)
    out = res.to_dict()
    json.dumps(out)
    assert out["pbo"] == 0.5 and out["k"] == 4 and "logits" not in out
    assert res.to_dict(detail=True)["logits"] == list(res.logits)


def _naive_pbo(m: np.ndarray, s: int, metric: str) -> tuple[list[int], list[float]]:
    """Brute force from the definition: concatenate the blocks of each split and rank directly."""
    rows = len(m) // s
    used = m[len(m) - rows * s :]
    blocks = [used[b * rows : (b + 1) * rows] for b in range(s)]
    chosen, logits = [], []
    for combo in __import__("itertools").combinations(range(s), s // 2):
        ins = column_performance(np.concatenate([blocks[b] for b in combo]), metric)
        outs = column_performance(np.concatenate([blocks[b] for b in range(s) if b not in combo]), metric)
        best = int(np.flatnonzero(ins == ins.max())[np.argmin(outs[ins == ins.max()])])
        rank = 1 + int((outs < outs[best]).sum())
        chosen.append(best)
        logits.append(math.log(rank / (m.shape[1] + 1 - rank)))
    return chosen, logits


@pytest.mark.parametrize("metric", ["sharpe", "mean"])
def test_the_batched_block_sums_match_a_brute_force_from_the_definition(metric: str) -> None:
    rng = np.random.default_rng(21)
    m = rng.integers(-3000, 3000, size=(77, 7)).astype(np.int64)  # 77 rows, S = 6: 5 oldest rows dropped
    m[:, 5] = m[:, 2]  # an exact duplicate column: an in-sample tie every split
    res = pbo_cscv(m, metric=metric, n_blocks=6)
    chosen, logits = _naive_pbo(m, 6, metric)
    assert list(res.chosen) == chosen
    assert list(res.logits) == pytest.approx(logits, abs=1e-12)
    assert res.pbo == float(np.mean(np.asarray(logits) <= 0))
