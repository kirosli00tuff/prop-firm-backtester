"""`pfb check <file> --json` prints validate_strategy's object as one JSON line; a refusal
prints the refused envelope and exits 2; plain `pfb check` output is unchanged. Nothing is logged."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from pfb.cli.main import app

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "strategies" / "_template.py"
PATHS_TOML = REPO / "config" / "paths.toml"
LINT_FAILING = "import os\n\n" + TEMPLATE.read_text()
EXIT_REFUSED = 2



def _invoke(*args: str) -> Any:
    return CliRunner().invoke(app, list(args))


def _copy(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "s.py"
    path.write_text(text)
    return path


def test_check_json_prints_the_validation_object_on_one_line(tmp_path: Path) -> None:
    # Arrange
    path = _copy(tmp_path, TEMPLATE.read_text())
    # Act
    done = _invoke("check", str(path), "--json")
    # Assert
    assert done.exit_code == 0, done.output
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    out = json.loads(lines[0])
    assert out["schema"] == "pfb.validation/1"
    assert out["ok"] is True
    assert out["spec"]["root"] == "MNQ"
    assert isinstance(out["warnings"], list)


def test_check_json_refusal_prints_the_refused_envelope_and_exits_2(tmp_path: Path) -> None:
    # Arrange
    path = _copy(tmp_path, LINT_FAILING)
    # Act
    done = _invoke("check", str(path), "--json")
    # Assert
    assert done.exit_code == EXIT_REFUSED
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["status"] == "refused"
    assert out["error"]["code"] == "RunRefused"
    assert "lint" in out["error"]["reason"]


def test_plain_check_output_is_unchanged(tmp_path: Path) -> None:
    # Arrange
    path = _copy(tmp_path, TEMPLATE.read_text())
    # Act
    done = _invoke("check", str(path))
    # Assert
    assert done.exit_code == 0, done.output
    assert done.stdout.startswith(f"ok: {path.name} passes the lint and the truncation check at seed ")
    assert "{" not in done.stdout.splitlines()[0]


def test_plain_check_refusal_prints_no_json(tmp_path: Path) -> None:
    # Arrange
    path = _copy(tmp_path, LINT_FAILING)
    # Act
    done = _invoke("check", str(path))
    # Assert
    assert done.exit_code == EXIT_REFUSED
    assert done.stdout.strip() == ""
