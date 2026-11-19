"""`pfb prop` on a trial pins contracts_per_mini, and every run's kind is read
from its begin record in the ledger, never from its run-id prefix; a run with no begin record is refused.
Temporary ledgers and synthetic coin-flip runs only (the helpers of test_cli_prop)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from test_cli_prop import (  # pyright: ignore[reportMissingImports]
    RUN_ID,
    TRIAL_ID,
    _error,
    _firms,
    _invoke,
    _ledger,
    _trial_ledger,
    _write_run,
)

from pfb.prop.matrix import MINIS_PER_TYPE
from pfb.prop.service import PropRefused, Request, require_declared, resolve_source

RENAMED_ID = "synthetic-" + TRIAL_ID.split("-", 1)[1]
DECLARED = ["topstep/ka_static_50k"]
TRIAL_ARGS = [TRIAL_ID, "--firm", "topstep", "--plan", "ka_static", "--size", "50k", "--mc", "300", "--seed",
              "11"]  # fmt: skip


def _request(per_mini: int | None, tmp: Path, ledger: Path, run_id: str = TRIAL_ID) -> Request:
    args = {"mc": 300, "seed": 11, "block": 5, "generator": "block", "stress_ticks": 0}
    return Request(False, run_id, [], "topstep", "ka_static", "50k", args, None, per_mini, tmp / "runs",
                   tmp / "firms", ledger)  # fmt: skip


# ------------------------------------------------------------------------- contracts per mini ----
def test_the_pinned_value_is_the_one_the_runners_secondary_output_uses() -> None:
    from pfb.prop import service

    assert service.MINIS_PER_TYPE is MINIS_PER_TYPE  # one table for pfb prop and pfb.runner.secondary


@pytest.mark.parametrize("per_mini", [None, 1])
def test_a_trial_takes_its_roots_contracts_per_mini_or_none(tmp_path: Path, per_mini: int | None) -> None:
    _write_run(tmp_path / "runs", TRIAL_ID)
    ledger = _trial_ledger(tmp_path, DECLARED)
    source = resolve_source(TRIAL_ID, tmp_path / "runs", ledger, None)
    require_declared(source, _request(per_mini, tmp_path, ledger))  # NQ is a mini: 1 per mini, no refusal


def test_a_trial_refuses_another_contracts_per_mini(tmp_path: Path) -> None:
    # Arrange
    _write_run(tmp_path / "runs", TRIAL_ID)
    ledger = _trial_ledger(tmp_path, DECLARED)
    source = resolve_source(TRIAL_ID, tmp_path / "runs", ledger, None)
    # Act, Assert: the service
    with pytest.raises(PropRefused, match="contracts per mini 1") as caught:
        require_declared(source, _request(10, tmp_path, ledger))
    assert caught.value.error == "TrialNotDeclared"
    # and the command
    res = _invoke([*TRIAL_ARGS, "--contracts-per-mini", "10"], tmp_path, _firms(tmp_path), ledger=ledger)
    assert res.exit_code == 2, res.stdout
    assert _error(res)["code"] == "TrialNotDeclared"
    assert not (tmp_path / "runs" / TRIAL_ID / "prop").exists()


def test_a_synthetic_run_still_takes_any_contracts_per_mini(tmp_path: Path) -> None:
    _write_run(tmp_path / "runs", RUN_ID)
    ledger = _ledger(tmp_path)
    source = resolve_source(RUN_ID, tmp_path / "runs", ledger, "NQ")
    require_declared(source, _request(10, tmp_path, ledger, RUN_ID))  # nothing is pinned outside a trial


# ---------------------------------------------------------------------- kind from the ledger ----
def test_the_kind_comes_from_the_begin_record(tmp_path: Path) -> None:
    _write_run(tmp_path / "runs", TRIAL_ID)
    ledger = _trial_ledger(tmp_path, DECLARED)
    source = resolve_source(TRIAL_ID, tmp_path / "runs", ledger, None)
    assert source.kind == "trial" and source.declared is not None


def test_a_run_with_no_begin_record_is_refused_whatever_its_prefix(tmp_path: Path) -> None:
    _write_run(tmp_path / "runs", RUN_ID)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(PropRefused, match="no begin record") as caught:
        resolve_source(RUN_ID, tmp_path / "runs", empty, "NQ")
    assert caught.value.error == "RunNotFound"
    with pytest.raises(PropRefused, match="no begin record"):
        resolve_source(RUN_ID, tmp_path / "runs", tmp_path / "missing.jsonl", "NQ")


def test_a_trial_folder_renamed_to_synthetic_is_refused(tmp_path: Path) -> None:
    # Arrange: the trial's folder copied to a name starting "synthetic-"; the ledger knows only the trial
    runs = tmp_path / "runs"
    _write_run(runs, TRIAL_ID)
    shutil.copytree(runs / TRIAL_ID, runs / RENAMED_ID)
    ledger = _trial_ledger(tmp_path, DECLARED)
    firms = _firms(tmp_path)
    # Act, Assert: the service refuses it (no begin record names the copy) ...
    with pytest.raises(PropRefused, match="no begin record"):
        resolve_source(RENAMED_ID, runs, ledger, "NQ")
    # ... and the command, with options a trial would refuse
    args = [RENAMED_ID, "--firm", "topstep", "--plan", "ka_eod_trailing", "--size", "50k", "--mc", "300",
            "--seed", "5", "--root", "NQ", "--contracts-per-mini", "10"]  # fmt: skip
    res = _invoke(args, tmp_path, firms, ledger=ledger)
    assert res.exit_code == 2, res.stdout
    assert _error(res)["code"] == "RunNotFound"
    assert not (runs / RENAMED_ID / "prop").exists()


def test_a_prefix_that_differs_from_the_begin_records_kind_is_refused(tmp_path: Path) -> None:
    # a synthetic-named folder whose begin record (by the same run id) says trial
    runs = tmp_path / "runs"
    _write_run(runs, RENAMED_ID)
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"kind": "trial", "run_id": RENAMED_ID, "hypothesis_id": "H9", "root": "NQ"})
                      + "\n")  # fmt: skip
    with pytest.raises(PropRefused, match="differs from the run id's kind"):
        resolve_source(RENAMED_ID, runs, ledger, None)


def test_a_pool_member_with_no_begin_record_is_refused(tmp_path: Path) -> None:
    other = "synthetic-20261116T000000000001Z-0000abce"
    _write_run(tmp_path / "runs", RUN_ID)
    _write_run(tmp_path / "runs", other, seed=8)
    res = _invoke([RUN_ID, "--firm", "topstep", "--plan", "ka_static", "--size", "50k", "--mc", "300",
                   "--root", "NQ", "--pool", other], tmp_path, _firms(tmp_path))  # fmt: skip
    assert res.exit_code == 2 and "no begin record" in _error(res)["reason"]
