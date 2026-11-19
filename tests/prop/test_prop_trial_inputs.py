"""At `pfb prop`: a source folder that does not match its result record's
artifact_sha256 is refused (ArtifactMismatch), and on a trial a ledger other than the registry's own or a
firms directory other than the repository's config/firms is refused (TrialInputsNotRepository). Ledger
fragments in pytest's temporary directory only; the real ledger is never written."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from pfb.cli.main import app
from pfb.prop.service import (
    REPO_FIRMS_DIR,
    PropRefused,
    Request,
    require_repository_inputs,
    resolve_source,
)

TRIAL_ID = "trial-20261117T000000000000Z-0000aaaa"
SYNTH_ID = "synthetic-20261117T000000000000Z-0000bbbb"
FIRM, PLAN, SIZE, MC, SEED = "topstep", "combine_xfa_standard", "50k", 300, 11
DECLARED = [f"{FIRM}/{PLAN}_{SIZE}"]
runner = CliRunner()


def _ledger(tmp: Path, records: list[dict[str, Any]]) -> Path:
    path = tmp / "ledger.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    return path


def _trial_ledger(tmp: Path) -> Path:
    hyp = {"kind": "hypothesis", "hypothesis_id": "HQ",
           "secondary": {"prop_plans": DECLARED, "prop_mc": MC, "prop_seed": SEED}}  # fmt: skip
    begin = {"kind": "trial", "run_id": TRIAL_ID, "hypothesis_id": "HQ", "root": "NQ"}
    return _ledger(tmp, [hyp, begin])


def _request(tmp: Path, *, ledger: Path, firms_dir: Path, registry_ledger: Path) -> Request:
    args = {"mc": MC, "seed": SEED, "block": 5, "generator": "block", "stress_ticks": 0}
    return Request(False, TRIAL_ID, [], FIRM, PLAN, SIZE, args, None, None, tmp / "runs", firms_dir, ledger,
                   registry_ledger)  # fmt: skip


# ------------------------------------------------------------------- the repository's inputs ----
def test_a_trial_refuses_another_ledger_or_firms_directory(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "runs" / TRIAL_ID).mkdir(parents=True)
    ledger = _trial_ledger(tmp_path)
    source = resolve_source(TRIAL_ID, tmp_path / "runs", ledger, None)
    firms_copy = tmp_path / "firms"
    shutil.copytree(REPO_FIRMS_DIR / FIRM, firms_copy / FIRM)
    # Act / Assert: the ledger is not the registry's
    other_ledger = _request(tmp_path, ledger=ledger, firms_dir=REPO_FIRMS_DIR, registry_ledger=tmp_path / "x")
    with pytest.raises(PropRefused) as caught:
        require_repository_inputs(source, other_ledger)
    assert caught.value.error == "TrialInputsNotRepository"
    # the firms directory is a copy, not the repository's config/firms
    other_firms = _request(tmp_path, ledger=ledger, firms_dir=firms_copy, registry_ledger=ledger)
    with pytest.raises(PropRefused) as caught:
        require_repository_inputs(source, other_firms)
    assert caught.value.error == "TrialInputsNotRepository"
    # both the registry's own: no refusal
    require_repository_inputs(
        source, _request(tmp_path, ledger=ledger, firms_dir=REPO_FIRMS_DIR, registry_ledger=ledger)
    )


def test_pfb_prop_on_a_trial_with_a_hidden_ledger_is_refused(tmp_path: Path) -> None:
    # Arrange: a declared request (plan, mc, seed), so only the ledger path is at fault
    (tmp_path / "runs" / TRIAL_ID).mkdir(parents=True)
    ledger = _trial_ledger(tmp_path)
    args = [TRIAL_ID, "--firm", FIRM, "--plan", PLAN, "--size", SIZE, "--mc", str(MC), "--seed", str(SEED),
            "--runs-dir", str(tmp_path / "runs"), "--ledger", str(ledger), "--json"]  # fmt: skip
    # Act
    res = runner.invoke(app, ["prop", *args])
    # Assert
    assert res.exit_code == 2, res.output
    assert json.loads(res.stdout)["error"]["code"] == "TrialInputsNotRepository"
    assert not (tmp_path / "runs" / TRIAL_ID / "prop").exists()


def test_a_synthetic_run_keeps_its_hidden_ledger(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / SYNTH_ID
    run_dir.mkdir(parents=True)
    ledger = _ledger(tmp_path, [{"kind": "synthetic", "run_id": SYNTH_ID, "root": "MNQ"}])
    source = resolve_source(SYNTH_ID, tmp_path / "runs", ledger, "MNQ")
    request = _request(tmp_path, ledger=ledger, firms_dir=tmp_path / "f", registry_ledger=tmp_path / "x")
    require_repository_inputs(source, request)  # not a trial: no refusal


# -------------------------------------------------------------------- the folder's hashes ----
def _synthetic_with_hash(tmp: Path, data: bytes, recorded: bytes) -> Path:
    run_dir = tmp / "runs" / SYNTH_ID
    run_dir.mkdir(parents=True)
    (run_dir / "daily.parquet").write_bytes(data)
    name = f"runs/{SYNTH_ID}/daily.parquet"
    result = {"kind": "result", "run_id": SYNTH_ID, "status": "ok", "artifacts": [name],
              "artifact_sha256": {name: hashlib.sha256(recorded).hexdigest()}}  # fmt: skip
    return _ledger(tmp, [{"kind": "synthetic", "run_id": SYNTH_ID, "root": "MNQ"}, result])


def test_a_folder_whose_frames_differ_from_its_record_is_refused(tmp_path: Path) -> None:
    ledger = _synthetic_with_hash(tmp_path, b"copied frames", b"the frames the run wrote")
    with pytest.raises(PropRefused) as caught:
        resolve_source(SYNTH_ID, tmp_path / "runs", ledger, "MNQ")
    assert caught.value.error == "ArtifactMismatch" and "daily.parquet" in caught.value.reason


def test_a_folder_that_matches_its_record_is_accepted(tmp_path: Path) -> None:
    ledger = _synthetic_with_hash(tmp_path, b"the frames", b"the frames")
    assert resolve_source(SYNTH_ID, tmp_path / "runs", ledger, "MNQ").kind == "synthetic"
