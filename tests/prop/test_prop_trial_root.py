"""`pfb prop` on a trial takes the root from the trial's begin
record in the ledger, and refuses when summary.json names another root; the other kinds keep pool.run_root
(summary.json "root", else the ledger). Temporary ledgers and synthetic coin-flip runs only."""

from __future__ import annotations

import json
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

from pfb.prop.service import PropRefused, resolve_source

DECLARED = ["topstep/ka_static_50k"]
TRIAL_ARGS = [TRIAL_ID, "--firm", "topstep", "--plan", "ka_static", "--size", "50k", "--mc", "300", "--seed",
              "11"]  # fmt: skip


def _summary(run_dir: Path, root: str) -> None:
    (run_dir / "summary.json").write_text(json.dumps({"root": root}))


def test_a_trial_takes_its_root_from_the_begin_record(tmp_path: Path) -> None:
    _write_run(tmp_path / "runs", TRIAL_ID)  # no summary.json
    ledger = _trial_ledger(tmp_path, DECLARED)  # begin record root NQ
    assert resolve_source(TRIAL_ID, tmp_path / "runs", ledger, None).root == "NQ"


def test_a_trial_whose_summary_names_its_begin_root_is_accepted(tmp_path: Path) -> None:
    _summary(_write_run(tmp_path / "runs", TRIAL_ID), "NQ")
    ledger = _trial_ledger(tmp_path, DECLARED)
    assert resolve_source(TRIAL_ID, tmp_path / "runs", ledger, None).root == "NQ"


def test_a_trial_whose_summary_names_another_root_is_refused(tmp_path: Path) -> None:
    _summary(_write_run(tmp_path / "runs", TRIAL_ID), "MNQ")
    ledger = _trial_ledger(tmp_path, DECLARED)
    with pytest.raises(PropRefused, match="begin record"):
        resolve_source(TRIAL_ID, tmp_path / "runs", ledger, None)


@pytest.mark.parametrize("extra", [[], ["--contracts-per-mini", "10"]], ids=["default", "per-mini-10"])
def test_the_review_scenario_is_refused_through_the_cli(tmp_path: Path, extra: list[str]) -> None:
    """The summary.json edited to MNQ under a trial begun on NQ; neither invocation runs."""
    _summary(_write_run(tmp_path / "runs", TRIAL_ID), "MNQ")
    ledger = _trial_ledger(tmp_path, DECLARED)
    res = _invoke([*TRIAL_ARGS, *extra], tmp_path, _firms(tmp_path), ledger=ledger)
    assert res.exit_code != 0
    assert "begin record" in _error(res)["reason"]
    assert not (tmp_path / "runs" / TRIAL_ID / "prop").exists()


def test_a_trial_given_another_root_is_refused(tmp_path: Path) -> None:
    _write_run(tmp_path / "runs", TRIAL_ID)
    ledger = _trial_ledger(tmp_path, DECLARED)
    with pytest.raises(PropRefused, match="--root MNQ differs"):
        resolve_source(TRIAL_ID, tmp_path / "runs", ledger, "MNQ")


def test_a_trial_begin_record_without_a_root_is_refused(tmp_path: Path) -> None:
    _write_run(tmp_path / "runs", TRIAL_ID)
    ledger = _trial_ledger(tmp_path, DECLARED)
    lines = [json.loads(line) for line in ledger.read_text().splitlines()]
    for record in lines:
        record.pop("root", None)
    ledger.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in lines))
    with pytest.raises(PropRefused, match="begin record"):
        resolve_source(TRIAL_ID, tmp_path / "runs", ledger, "NQ")


def test_a_synthetic_run_still_takes_its_root_from_summary_json(tmp_path: Path) -> None:
    _summary(_write_run(tmp_path / "runs", RUN_ID), "MNQ")
    ledger = _ledger(tmp_path, root="NQ")
    assert resolve_source(RUN_ID, tmp_path / "runs", ledger, None).root == "MNQ"
