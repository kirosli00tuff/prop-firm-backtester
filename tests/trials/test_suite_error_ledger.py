"""End to end: a synthetic run whose walk-forward asks for more folds than the
run has days is ok; walk_forward is null and the warning is in the result, the ledger result record and the
contract. Temporary ledger in a temporary git repository; the real engine on synthetic bars."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from registry_env import Env  # pyright: ignore[reportMissingImports]
from trial_support import kinds, n_of, temp_paths_toml, without_repo_paths  # pyright: ignore[reportMissingImports]

from pfb.analysis.spec import SuiteIntegrityError, SuiteSpec
from pfb.contract import build_result, validate
from pfb.registry.chain import read_ledger

REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / "strategies" / "reference_breakout.py"
TOO_MANY_FOLDS = 10**6
WARNING_START = "suite: walk_forward not computed: FoldError: "


@pytest.fixture
def real_deps(genesis_env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The real runner deps on a temporary paths file: config/paths.toml is not needed."""
    from pfb.runner.deps import default_deps

    without_repo_paths(monkeypatch, tmp_path)
    toml = temp_paths_toml(tmp_path / "paths")
    return replace(default_deps(toml), registry_config=lambda: genesis_env.config)


def test_a_fold_error_leaves_the_run_ok_with_the_part_null_and_the_warning_recorded(
    genesis_env: Env, real_deps: Any
) -> None:
    # Arrange
    from pfb.runner.api import run

    strategy = genesis_env.write("strategies/ref.py", REFERENCE.read_text())
    n_before = n_of(genesis_env)
    # Act
    suite = SuiteSpec(walk_forward_folds=TOO_MANY_FOLDS)
    outcome = run(strategy, kind="synthetic", cost_model="fixed:1", deps=real_deps,
                  runs_dir=genesis_env.repo / "runs", suite=suite)  # fmt: skip
    # Assert: the run
    assert outcome.result_record["status"] == "ok"
    summary = outcome.result.summary
    assert summary["suite"]["walk_forward"] is None and summary["suite"]["baseline"] is not None
    [warning] = [w for w in summary["warnings"] if w.startswith("suite: ")]
    assert warning.startswith(WARNING_START) and f"cannot make {TOO_MANY_FOLDS} folds" in warning
    # the ledger result record (read back from the file, not the outcome)
    lines = read_ledger(genesis_env.config.registry_path)
    assert kinds(genesis_env) == ["genesis", "synthetic", "result"] and n_of(genesis_env) == n_before
    record = lines[-1].record
    assert record["status"] == "ok" and record["metrics"]["suite"]["walk_forward"] is None
    assert warning in record["metrics"]["warnings"]
    # the result contract built from those records
    doc = build_result(begin=lines[-2].record, result=record, begin_sha256=lines[-2].sha256,
                       result_sha256=lines[-1].sha256)  # fmt: skip
    validate(doc)
    assert doc["status"] == "ok" and doc["walk_forward"] is None and warning in doc["warnings"]


def test_an_integrity_error_fails_the_run_and_the_ledger_records_it_failed(
    genesis_env: Env, real_deps: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C.1 R2: a stress replay with another trade count than the base run is a SuiteIntegrityError, not a
    part error: the run fails (failed result record), as before C.1."""
    import pfb.analysis.stress as stress
    from pfb.engine import run as engine_run
    from pfb.runner.api import run

    real = engine_run.rerun_kernel

    def one_trade_less(sim: Any, slippage: Any, calibrated: Any) -> Any:
        out = real(sim, slippage, calibrated)
        return replace(out, trades=out.trades.head(out.trades.height - 1))

    monkeypatch.setattr(stress, "rerun_kernel", one_trade_less)
    strategy = genesis_env.write("strategies/ref.py", REFERENCE.read_text())
    with pytest.raises(SuiteIntegrityError, match="trades, the base run has"):
        run(strategy, kind="synthetic", cost_model="fixed:1", deps=real_deps,
            runs_dir=genesis_env.repo / "runs", suite=SuiteSpec())  # fmt: skip
    record = read_ledger(genesis_env.config.registry_path)[-1].record
    assert kinds(genesis_env) == ["genesis", "synthetic", "result"]
    assert (record["status"], record["error"]) == ("failed", "SuiteIntegrityError")
