"""family_tests.for_hypothesis on a temporary ledger and git repository (never the real one).

The whole declared family or nothing: the matrix comes from pbo.grid_matrix, so a grid point lacking an ok
trial result is refused, a failed run does not fill a point, and reading appends nothing (the temporary ledger
is byte-identical and N unchanged).
"""

from __future__ import annotations

import io
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "registry"))

from registry_env import HYPOTHESIS_REL, Env  # noqa: E402  # pyright: ignore[reportMissingImports]

from pfb.analysis import family_tests as ft  # noqa: E402
from pfb.analysis import pbo  # noqa: E402
from pfb.registry.api import create_genesis, finish, read_state, register_hypothesis  # noqa: E402
from pfb.registry.artifacts import bytes_sha256  # noqa: E402

DAYS = 30  # the registered window 2025-04-01..2025-04-30
GRID = (1, 2, 3)
EDGE_CENTS = 3000


@pytest.fixture
def env(tmp_path: Path) -> Env:
    """Genesis plus a registered H1 whose grid a in (1, 2, 3) the budget of 3 can cover."""
    e = Env(tmp_path)
    create_genesis(e.config)
    e.write(HYPOTHESIS_REL, e.hypothesis_text(budget=3))
    e.commit()
    register_hypothesis(e.repo / HYPOTHESIS_REL, e.config, e.facts())
    e.checkpoint()
    return e


def _daily_bytes(nets: np.ndarray) -> bytes:
    start = date(2025, 4, 1)
    buffer = io.BytesIO()
    pl.DataFrame(
        {
            "trade_date": [start + timedelta(days=i) for i in range(len(nets))],
            "eq_close": np.cumsum(nets).astype(np.int64),
            "net_cents": nets.astype(np.int64),
        }
    ).write_parquet(buffer)
    return buffer.getvalue()


def _run(env: Env, a: int, nets: np.ndarray | None, status: str = "ok") -> str:
    """A finished trial whose result record pins its daily file by sha256, as the runner
    writes it; grid_matrix checks the file against that hash."""
    receipt = env.begin_trial(params={"a": a})
    data = None if nets is None else _daily_bytes(nets)
    if status == "ok":
        metrics = {"sharpe_daily": 0.1}
        if nets is not None:
            metrics |= {"net_cents": int(nets.sum()), "n_days": len(nets)}
        shas = {} if data is None else {f"runs/{receipt.run_id}/daily.parquet": bytes_sha256(data)}
        finish(receipt, "ok", metrics, list(shas), config=env.config, artifact_sha256=shas)
    else:
        finish(receipt, "failed", None, [], config=env.config, error="ZeroDivisionError")
    env.checkpoint()
    if data is not None:
        folder = env.repo / "runs" / receipt.run_id
        folder.mkdir(parents=True)
        (folder / "daily.parquet").write_bytes(data)
    return receipt.run_id


def _nets(seed: int, edge: int = 0) -> np.ndarray:
    raw = np.random.default_rng(seed).integers(-5000, 5001, size=DAYS)
    return (raw - int(round(raw.mean())) + edge).astype(np.int64)


def test_the_whole_family_is_tested_and_the_planted_point_named(env: Env) -> None:
    run_ids = [_run(env, a, _nets(a, EDGE_CENTS if a == 2 else 0)) for a in GRID]
    before, n_before = env.config.registry_path.read_bytes(), read_state(env.config).n
    out = ft.for_hypothesis("H1", config=env.config)
    tests = out["family_tests"]
    assert out["k"] == 3 and out["t"] == DAYS and out["run_ids"] == run_ids and out["n"] == n_before
    assert tests["stepm_superior"] == [1] and out["stepm_superior_params"] == [{"a": 2}]  # type: ignore[index]
    assert tests["spa_pvalues"]["consistent"] < 0.05  # type: ignore[index]
    assert tests["conventions"]["block_size"] == ft.block_length(DAYS) == 4  # type: ignore[index]
    assert out["warnings"] == [] and out["sum_checked_run_ids"] == []  # every file checked by its sha256
    assert env.config.registry_path.read_bytes() == before and read_state(env.config).n == n_before


def test_a_subset_of_the_declared_family_is_refused(env: Env) -> None:
    _run(env, 1, _nets(1))
    _run(env, 3, _nets(3, EDGE_CENTS))
    with pytest.raises(pbo.GridIncomplete, match=r"1 of 3 declared configurations.*\{'a': 2\}"):
        ft.for_hypothesis("H1", config=env.config)


def test_a_failed_run_does_not_fill_its_point(env: Env) -> None:
    _run(env, 1, _nets(1))
    _run(env, 2, None, status="failed")
    _run(env, 3, _nets(3))
    with pytest.raises(pbo.GridIncomplete, match=r"\{'a': 2\}"):
        ft.for_hypothesis("H1", config=env.config)


def test_a_rerun_supersedes_and_is_reported(env: Env) -> None:
    _run(env, 1, _nets(1))
    _run(env, 2, _nets(2))
    _run(env, 3, _nets(3))
    latest = _run(env, 3, _nets(33))
    out = ft.for_hypothesis("H1", config=env.config)
    assert out["run_ids"][-1] == latest  # type: ignore[index]
    assert len(out["warnings"]) == 1 and "superseded" in out["warnings"][0]  # type: ignore[index,arg-type]
