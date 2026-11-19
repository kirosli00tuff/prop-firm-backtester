"""pbo.for_hypothesis and cpcv.for_hypothesis on a temporary ledger and git repository (never the real one).

The whole declared grid or nothing: a hypothesis with a grid point lacking an ok trial result is refused, a
failed run does not fill a point, and reading appends nothing (the temporary ledger is byte-identical).

Each daily file is checked against its run's result record: by artifact_sha256 for a record that carries it
(as the runner writes every record since), by net sum and row count for a record written before the key
(made here by stripping the key as the record is appended, so the ledger replays it as result tier 2).
"""

from __future__ import annotations

import io
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "registry"))

from registry_env import HYPOTHESIS_REL, Env  # noqa: E402  # pyright: ignore[reportMissingImports]

from pfb.analysis import cpcv, pbo  # noqa: E402
from pfb.registry import api as registry_api  # noqa: E402
from pfb.registry.api import create_genesis, finish, read_state, register_hypothesis  # noqa: E402
from pfb.registry.artifacts import bytes_sha256  # noqa: E402
from pfb.registry.errors import HypothesisNotRegistered  # noqa: E402
from pfb.registry.records import ARTIFACT_SHA_KEY  # noqa: E402

DAYS = 64
GRID = (1, 2, 3)


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


def _daily_bytes(nets: np.ndarray, start: date = date(2025, 4, 1)) -> bytes:
    buffer = io.BytesIO()
    pl.DataFrame(
        {
            "trade_date": [start + timedelta(days=i) for i in range(len(nets))],
            "eq_close": np.cumsum(nets).astype(np.int64),
            "net_cents": nets.astype(np.int64),
        }
    ).write_parquet(buffer)
    return buffer.getvalue()


def _daily(env: Env, run_id: str, nets: np.ndarray, start: date = date(2025, 4, 1)) -> None:
    folder = env.repo / "runs" / run_id
    folder.mkdir(parents=True)
    (folder / "daily.parquet").write_bytes(_daily_bytes(nets, start))


def _name(run_id: str) -> str:
    """The artifact name the runner records for a run's daily file inside the repository."""
    return f"runs/{run_id}/daily.parquet"


@contextmanager
def _records_before_the_key() -> Iterator[None]:
    """Result records appended inside the block lack artifact_sha256, as every record written before
    did (result tier 2). The ledger accepts them only before the first record that carries the key."""
    real = registry_api.locked_append

    def append(path: Path, build: Any, validate: Any, **kwargs: Any) -> Any:
        def without_key(lines: Any) -> dict[str, Any]:
            return {k: v for k, v in build(lines).items() if k != ARTIFACT_SHA_KEY}

        return real(path, without_key, validate, **kwargs)

    registry_api.locked_append = append  # type: ignore[assignment]
    try:
        yield
    finally:
        registry_api.locked_append = real


def _metrics(nets: np.ndarray | None) -> dict[str, object]:
    """The result record's figures the daily file is checked against."""
    if nets is None:
        return {"sharpe_daily": 0.1}
    return {"sharpe_daily": 0.1, "net_cents": int(nets.sum()), "n_days": len(nets)}


def _run(
    env: Env, a: int, nets: np.ndarray | None, status: str = "ok", *, before_the_key: bool = False
) -> str:
    """A finished trial; its record pins the daily file by sha256 unless before_the_key."""
    receipt = env.begin_trial(params={"a": a})
    if status == "ok" and before_the_key:
        with _records_before_the_key():
            finish(receipt, "ok", _metrics(nets), [], config=env.config)
    elif status == "ok":
        shas = {} if nets is None else {_name(receipt.run_id): bytes_sha256(_daily_bytes(nets))}
        finish(receipt, "ok", _metrics(nets), list(shas), config=env.config, artifact_sha256=shas)
    else:
        finish(receipt, "failed", None, [], config=env.config, error="ZeroDivisionError")
    env.checkpoint()
    if nets is not None:
        _daily(env, receipt.run_id, nets)
    return receipt.run_id


def _nets(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(-5000, 5000, size=DAYS)


def _ledger(env: Env) -> bytes:
    return env.config.registry_path.read_bytes()


def test_the_whole_grid_gives_a_k_by_t_matrix_and_appends_nothing(env: Env) -> None:
    run_ids = [_run(env, a, _nets(a)) for a in GRID]
    before, n_before = _ledger(env), read_state(env.config).n
    out = pbo.for_hypothesis("H1", config=env.config)
    assert out["k"] == 3 and out["t"] == DAYS and out["run_ids"] == run_ids
    assert out["params"] == [{"a": 1}, {"a": 2}, {"a": 3}]
    assert out["n"] == n_before and out["root"] == "MNQ" and out["window"] == ["2025-04-01", "2025-04-30"]
    assert 0.0 <= out["pbo"]["pbo"] <= 1.0 and out["pbo"]["n_splits"] == 12870  # type: ignore[index]
    assert out["sum_checked_run_ids"] == [] and out["warnings"] == []  # every file checked by its sha256
    grid = pbo.grid_matrix(env.config, "H1")
    np.testing.assert_array_equal(grid.matrix[:, 1], _nets(2))
    c = cpcv.for_hypothesis("H1", config=env.config)
    assert c["cpcv"]["n_paths"] == 5  # type: ignore[index]
    assert _ledger(env) == before and read_state(env.config).n == n_before


def test_for_hypothesis_takes_no_convention_and_echoes_the_module_constants(env: Env) -> None:
    for a in GRID:
        _run(env, a, _nets(a))
    for kwargs in ({"n_blocks": 4}, {"metric": "mean"}, {"remainder": "drop_last"}, {"is_tie": "first"}):
        with pytest.raises(TypeError):
            pbo.for_hypothesis("H1", config=env.config, **kwargs)  # type: ignore[call-arg]
    for kwargs in ({"n_groups": 4}, {"purge_days": 0}, {"embargo_days": 0}, {"k_test": 1}):
        with pytest.raises(TypeError):
            cpcv.for_hypothesis("H1", config=env.config, **kwargs)  # type: ignore[call-arg]
    out = pbo.for_hypothesis("H1", config=env.config)
    assert out["pbo"]["conventions"] == pbo.Conventions().to_dict()  # type: ignore[index]
    assert pbo.Conventions() == pbo.HYPOTHESIS_CONVENTIONS
    c = cpcv.for_hypothesis("H1", config=env.config)
    echoed = c["cpcv"]["conventions"]  # type: ignore[index]
    assert {k: echoed[k] for k in cpcv.HYPOTHESIS_CONVENTIONS} == dict(cpcv.HYPOTHESIS_CONVENTIONS)
    assert dict(cpcv.HYPOTHESIS_CONVENTIONS) == {
        "n_groups": 6, "k_test": 2, "purge_days": 1, "embargo_days": 5, "metric": "sharpe", "is_tie": "worst",
        "min_train_rows": 2,
    }  # fmt: skip


@pytest.mark.parametrize(
    ("unrecorded", "match"), [(False, "netting"), (True, "no integer metrics.net_cents")]
)
def test_a_record_before_the_key_refuses_a_daily_file_whose_sums_differ(
    env: Env, unrecorded: bool, match: str
) -> None:
    """Records written before artifact_sha256: grid point a = 2's file has one cent moved (or its record has
    no figures to check it against)."""
    _run(env, 1, _nets(1), before_the_key=True)
    _run(env, 3, _nets(3), before_the_key=True)
    receipt = env.begin_trial(params={"a": 2})
    nets = _nets(2)
    with _records_before_the_key():
        finish(receipt, "ok", {"sharpe_daily": 0.1} if unrecorded else _metrics(nets), [], config=env.config)
    env.checkpoint()
    assert ARTIFACT_SHA_KEY not in read_state(env.config).results[receipt.run_id]
    written = nets.copy()
    written[10] += 0 if unrecorded else 1
    _daily(env, receipt.run_id, written)
    with pytest.raises(pbo.GridFileMismatch, match=f"run {receipt.run_id}: .*{match}"):
        pbo.grid_matrix(env.config, "H1")
    with pytest.raises(pbo.GridFileMismatch):
        pbo.for_hypothesis("H1", config=env.config)
    with pytest.raises(pbo.GridFileMismatch):
        cpcv.for_hypothesis("H1", config=env.config)


@pytest.mark.parametrize(
    ("tamper", "match"),
    [
        ("cent", "daily.parquet: sha256 differs from the result record"),
        ("swap", "daily.parquet: sha256 differs from the result record"),
        ("unnamed", "artifact_sha256 does not name daily.parquet"),
        ("other_run", "artifact_sha256 does not name daily.parquet"),
    ],
)
def test_a_record_with_the_key_refuses_a_daily_file_that_does_not_hash_to_it(
    env: Env, tamper: str, match: str
) -> None:
    """Grid point a = 2's file has one cent moved, or two days swapped (net sum and row count unchanged: only
    the hash sees it), or its record's artifact_sha256 does not name its daily file."""
    _run(env, 1, _nets(1))
    _run(env, 3, _nets(3))
    receipt = env.begin_trial(params={"a": 2})
    nets = _nets(2)
    recorded = {
        "unnamed": {},
        "other_run": {_name("someone-else"): bytes_sha256(_daily_bytes(nets))},
    }.get(tamper, {_name(receipt.run_id): bytes_sha256(_daily_bytes(nets))})
    finish(receipt, "ok", _metrics(nets), list(recorded), config=env.config, artifact_sha256=recorded)
    env.checkpoint()
    written = nets.copy()
    if tamper == "cent":
        written[10] += 1
    elif tamper == "swap":
        written[[3, 4]] = written[[4, 3]]
        assert int(written.sum()) == int(nets.sum()) and not np.array_equal(written, nets)
    _daily(env, receipt.run_id, written)
    with pytest.raises(pbo.GridFileMismatch, match=f"run {receipt.run_id}: .*{match}"):
        pbo.grid_matrix(env.config, "H1")
    with pytest.raises(pbo.GridFileMismatch, match=match):
        pbo.for_hypothesis("H1", config=env.config)
    with pytest.raises(pbo.GridFileMismatch, match=match):
        cpcv.for_hypothesis("H1", config=env.config)


def test_records_before_and_after_the_key_are_each_checked_their_own_way(env: Env) -> None:
    """a = 1 was finished before the key (sum and row count), a = 2 and 3 after it (sha256). A day swap in
    the older run's file keeps its sums and is not seen: the limit of the fallback, named in the output."""
    old = _run(env, 1, _nets(1), before_the_key=True)
    new = [_run(env, a, _nets(a)) for a in (2, 3)]
    state = read_state(env.config)
    assert ARTIFACT_SHA_KEY not in state.results[old]
    assert all(ARTIFACT_SHA_KEY in state.results[r] for r in new)
    grid = pbo.grid_matrix(env.config, "H1")
    assert grid.run_ids == (old, *new) and grid.sum_checked_run_ids == (old,)
    out = pbo.for_hypothesis("H1", config=env.config)
    assert out["sum_checked_run_ids"] == [old]
    assert any("before artifact_sha256" in w for w in out["warnings"])  # type: ignore[union-attr]
    assert cpcv.for_hypothesis("H1", config=env.config)["sum_checked_run_ids"] == [old]
    # the older run's file with two days swapped: same sums, same rows, so the fallback passes it
    swapped = _nets(1)
    swapped[[3, 4]] = swapped[[4, 3]]
    (env.repo / "runs" / old / "daily.parquet").write_bytes(_daily_bytes(swapped))
    assert pbo.grid_matrix(env.config, "H1").sum_checked_run_ids == (old,)
    # a moved cent in the older run's file is refused by the sums
    moved = _nets(1)
    moved[0] += 1
    (env.repo / "runs" / old / "daily.parquet").write_bytes(_daily_bytes(moved))
    with pytest.raises(pbo.GridFileMismatch, match=f"run {old}: .*netting"):
        pbo.grid_matrix(env.config, "H1")


def test_check_daily_file_uses_the_hash_when_the_record_has_one(tmp_path: Path) -> None:
    nets = _nets(2)
    data = _daily_bytes(nets)
    (tmp_path / "daily.parquet").write_bytes(data)
    frame = pl.read_parquet(tmp_path / "daily.parquet").select(pbo.DAILY_COLUMNS)
    keyed = {"metrics": {}, ARTIFACT_SHA_KEY: {_name("r"): bytes_sha256(data)}}
    assert pbo.check_daily_file(tmp_path, "r", frame, keyed) is True  # no figures needed with the hash
    assert pbo.check_daily_file(tmp_path, "r", frame, {"metrics": _metrics(nets)}) is False
    (tmp_path / "daily.parquet").unlink()
    with pytest.raises(pbo.GridFileMismatch, match="daily.parquet: missing"):
        pbo.check_daily_file(tmp_path, "r", frame, keyed)


def test_check_against_record_compares_the_net_sum_and_the_row_count() -> None:
    nets = _nets(2)
    frame = pl.DataFrame({"trade_date": [date(2025, 4, 1) + timedelta(days=i) for i in range(DAYS)],
                          "net_cents": nets.astype(np.int64)})  # fmt: skip
    pbo.check_against_record("r", frame, _metrics(nets))  # the file the trial wrote: no refusal
    with pytest.raises(pbo.GridFileMismatch, match=f"has {DAYS - 1} days"):
        pbo.check_against_record(
            "r", frame.head(DAYS - 1), {**_metrics(nets), "net_cents": int(nets[:-1].sum())}
        )
    with pytest.raises(pbo.GridFileMismatch, match="metrics.n_days"):
        pbo.check_against_record("r", frame, {"net_cents": int(nets.sum()), "n_days": float(DAYS)})
    with pytest.raises(pbo.GridFileMismatch, match="metrics.net_cents"):
        pbo.check_against_record("r", frame, None)


def test_a_grid_point_without_a_run_is_refused(env: Env) -> None:
    _run(env, 1, _nets(1))
    _run(env, 3, _nets(3))
    with pytest.raises(pbo.GridIncomplete, match=r"1 of 3 declared configurations.*\{'a': 2\}"):
        pbo.for_hypothesis("H1", config=env.config)
    with pytest.raises(pbo.GridIncomplete):
        cpcv.for_hypothesis("H1", config=env.config)


def test_a_failed_run_does_not_fill_its_grid_point(env: Env) -> None:
    _run(env, 1, _nets(1))
    _run(env, 2, None, status="failed")
    _run(env, 3, _nets(3))
    with pytest.raises(pbo.GridIncomplete, match=r"\{'a': 2\}"):
        pbo.for_hypothesis("H1", config=env.config)


def test_a_missing_daily_file_is_refused(env: Env) -> None:
    _run(env, 1, _nets(1))
    _run(env, 2, None)
    _run(env, 3, _nets(3))
    with pytest.raises(pbo.GridIncomplete, match="daily.parquet is missing"):
        pbo.for_hypothesis("H1", config=env.config)


def test_runs_on_different_days_are_refused(env: Env) -> None:
    _run(env, 1, _nets(1))
    _run(env, 2, _nets(2))
    receipt = env.begin_trial(params={"a": 3})
    finish(receipt, "ok", {"sharpe_daily": 0.1}, [], config=env.config)
    env.checkpoint()
    _daily(env, receipt.run_id, _nets(3), start=date(2025, 4, 2))
    with pytest.raises(ValueError, match="trade dates differ"):
        pbo.for_hypothesis("H1", config=env.config)


def test_an_unregistered_hypothesis_and_an_undeclared_window_are_refused(env: Env) -> None:
    with pytest.raises(HypothesisNotRegistered):
        pbo.for_hypothesis("H9", config=env.config)
    with pytest.raises(pbo.PboError, match="not one of"):
        pbo.grid_matrix(env.config, "H1", window=(date(2025, 4, 1), date(2025, 4, 29)))
    with pytest.raises(pbo.PboError, match="not one of"):
        pbo.grid_matrix(env.config, "H1", root="MES")


def test_declared_points_is_the_cartesian_product_in_key_order() -> None:
    assert pbo.declared_points({"b": [1, 2], "a": ["x"]}) == [{"b": 1, "a": "x"}, {"b": 2, "a": "x"}]
    assert len(pbo.declared_points({"a": [1, 2, 3], "b": [1, 2], "c": [0.5, 1.0]})) == 12
