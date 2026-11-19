"""Path generators and the cost stress (src/pfb/prop/paths.py)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pfb.prop.days import from_day_nets, from_minutes
from pfb.prop.paths import PAD, block_bootstrap, padded_paths, random_start, stress


def _rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def _block_lengths(seqs: np.ndarray, n_days: int) -> np.ndarray:
    """Lengths of runs where each day follows the previous one (with wrap)."""
    cont = seqs[:, 1:] == (seqs[:, :-1] + 1) % n_days
    lengths = []
    for row in cont:
        run = 1
        for c in row:
            if c:
                run += 1
            else:
                lengths.append(run)
                run = 1
        # the last run is censored by the horizon: left out
    return np.array(lengths)


def test_block_bootstrap_mean_block_length() -> None:
    n_days, mean_block = 1000, 5
    seqs = block_bootstrap(n_days, 200, 2000, mean_block, _rng(11))
    lengths = _block_lengths(seqs, n_days)
    # a block ends on a jump (prob 1/m) that does not land on the next day (prob 1 - 1/n_days)
    expected = mean_block / (1 - 1 / n_days)
    assert abs(lengths.mean() - expected) < 0.1, (lengths.mean(), expected)


def test_block_bootstrap_mean_block_one_is_iid() -> None:
    seqs = block_bootstrap(4, 50, 4000, 1, _rng(3))
    counts = np.bincount(seqs.ravel(), minlength=4) / seqs.size
    assert np.allclose(counts, 0.25, atol=0.01)
    follow = (seqs[:, 1:] == (seqs[:, :-1] + 1) % 4).mean()
    assert abs(follow - 0.25) < 0.01  # no persistence beyond chance


def test_block_bootstrap_wraps_at_end() -> None:
    seqs = block_bootstrap(3, 100, 500, 1000, _rng(5))
    cont = seqs[:, 1:] == (seqs[:, :-1] + 1) % 3
    wrapped = (seqs[:, :-1] == 2) & cont
    assert wrapped.any()
    assert np.all(seqs[:, 1:][wrapped] == 0)
    assert seqs.min() >= 0 and seqs.max() <= 2


def test_block_bootstrap_deterministic_by_seed() -> None:
    a = block_bootstrap(50, 30, 100, 5, _rng(42))
    b = block_bootstrap(50, 30, 100, 5, _rng(42))
    c = block_bootstrap(50, 30, 100, 5, _rng(43))
    assert a.dtype == np.int64 and a.shape == (100, 30)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_block_bootstrap_refuses_bad_arguments() -> None:
    with pytest.raises(ValueError, match="mean_block"):
        block_bootstrap(5, 5, 5, 0, _rng(1))
    with pytest.raises(ValueError, match="mean_block"):
        block_bootstrap(5, 5, 5, 2.5, _rng(1))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 1"):
        block_bootstrap(0, 5, 5, 5, _rng(1))


def test_random_start_draws_only_full_windows() -> None:
    seqs = random_start(10, 6, 3000, _rng(9))
    starts = seqs[:, 0]
    assert set(np.unique(starts).tolist()) == set(range(5))  # 10 - 6 + 1 starts leave 6 days of history
    assert np.array_equal(seqs, starts[:, None] + np.arange(6)[None, :])
    assert not padded_paths(seqs).any()
    exact = random_start(6, 6, 20, _rng(9))
    assert np.array_equal(exact, np.tile(np.arange(6), (20, 1)))


def test_random_start_pads_when_history_is_shorter_than_the_horizon() -> None:
    seqs = random_start(10, 12, 3000, _rng(9))
    starts = seqs[:, 0]
    assert set(np.unique(starts).tolist()) == set(range(10))
    for row in seqs[:50]:
        valid = row[row != PAD]
        assert np.array_equal(valid, np.arange(row[0], row[0] + valid.size))
        assert np.all(row[valid.size :] == PAD)  # padding only at the end
        assert valid.size == 10 - row[0]
    assert padded_paths(seqs).all()


def test_random_start_deterministic_by_seed() -> None:
    assert np.array_equal(random_start(20, 5, 50, _rng(1)), random_start(20, 5, 50, _rng(1)))


def test_stress_at_fill_minutes() -> None:
    days = from_minutes([([0, -10, -20], [5, 5, 0], [0, -10, -20])], sides=[0])
    days = replace(days, fill_sides=np.array([2, 0, 2]), sides=np.array([4]))
    out = stress(days, 1, 25)
    assert out.close.tolist() == [-50, -60, -120]
    assert out.low.tolist() == [-50, -60, -120]
    assert out.high.tolist() == [5, -45, -50]  # high carries only the cost filled before the minute
    assert np.array_equal(days.close, [0, -10, -20])  # input untouched


def test_stress_unknown_fill_minutes_whole_day_first_minute() -> None:
    days = from_minutes([([0, 0], [0, 10], [0, 10]), ([0], [0], [0])], sides=[2, 0])
    days = replace(days, fill_sides=np.zeros(3, dtype=np.int64), sides=np.array([6, 0]))
    out = stress(days, 2, 125)
    assert out.close.tolist() == [-1500, -1490, 0]
    assert out.day_net().tolist() == [-1490, 0]


def test_stress_zero_and_refusals() -> None:
    days = from_day_nets([10, -10])
    assert stress(days, 0, 125) is days
    assert stress(days, 1, 125).day_net().tolist() == [-240, -260]
    with pytest.raises(ValueError, match="k_ticks"):
        stress(days, -1, 125)
    with pytest.raises(ValueError, match="tick_cents"):
        stress(days, 1, 0)
