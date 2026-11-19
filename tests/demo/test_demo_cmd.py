"""`pfb demo` end to end in a temporary folder: store, gateway reads, check, synthetic run with a prop MC,
nothing added to N; the repository's own ledger, runs and catalog are never touched."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from pfb.cli.main import app
from pfb.registry.api import REPO_ROOT

REPO_LEDGER = REPO_ROOT / "ledger" / "registry.jsonl"


def test_pfb_demo_runs_the_pipeline_in_a_temporary_folder(tmp_path: Path) -> None:
    ledger_before = REPO_LEDGER.read_bytes() if REPO_LEDGER.is_file() else None
    args = [
        "demo", "--json", "--mc", "20",
        "--store-dir", str(tmp_path / "demo"), "--runs-dir", str(tmp_path / "runs"),
        "--ledger", str(tmp_path / "ledger" / "registry.jsonl"), "--catalog", str(tmp_path / "catalog.json"),
    ]  # fmt: skip

    result = CliRunner().invoke(app, args)

    assert result.exit_code == 0, result.output
    out = json.loads(result.stdout)
    store, run = out["store"], out["run"]
    assert store["files"] == 2 and store["open_read"]["rows"] > 0
    assert "demo_embargo_tail" in store["closed_read"]["refused"]
    assert run["kind"] == "synthetic" and run["status"] == "ok" and run["n_added"] == 0
    assert (tmp_path / "runs" / run["run_id"] / "result.json").is_file()
    assert out["prop"][0]["status"] == "ok" and out["prop"][0]["n_paths"] == 20
    assert out["ledger"]["n"] == 0 and out["ledger"]["records"] >= 3
    assert (tmp_path / "catalog.json").is_file()
    assert (REPO_LEDGER.read_bytes() if REPO_LEDGER.is_file() else None) == ledger_before


def test_pfb_demo_help_names_every_step() -> None:
    result = CliRunner().invoke(app, ["demo", "--help"])

    assert result.exit_code == 0
    text = " ".join(result.output.split())
    for word in ("demo store", "DataGateway", "pfb check", "synthetic run", "prop Monte Carlo", "--dash"):
        assert word in text, word
