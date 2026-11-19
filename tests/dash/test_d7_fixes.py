"""The dashboard bugs found. Nothing here reads a bar,
counts a trial or writes outside tmp_path."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

import pfb.data.paths as paths_module
from pfb.cli import dash_cmd
from pfb.cli.main import app
from pfb.dash import actions, view_data, view_runs
from pfb.dash.actions import CliResult
from pfb.dash.windows import Sources
from pfb.data.errors import PathsConfigError
from pfb.runner.deps import default_deps

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "strategies" / "_template.py"


# ------------------------------------------------- bug 1: Check without config/paths.toml ----
def test_default_deps_reads_no_paths_file_until_a_real_data_step_needs_it(tmp_path: Path) -> None:
    # Arrange
    missing = tmp_path / "paths.toml"
    # Act
    deps = default_deps(missing)
    # Assert
    assert callable(deps.lint_strategy)
    with pytest.raises(PathsConfigError, match="not found"):
        deps.registry_config()
    with pytest.raises(PathsConfigError, match="not found"):
        deps.store_roots()


def test_check_json_passes_without_a_paths_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(paths_module, "DEFAULT_PATHS_TOML", tmp_path / "absent" / "paths.toml")
    path = tmp_path / "s.py"
    path.write_text(TEMPLATE.read_text())
    # Act
    done = CliRunner().invoke(app, ["check", str(path), "--json"])
    # Assert
    assert done.exit_code == 0, done.output
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["ok"] is True
    assert out["spec"]["root"] == "MNQ"


def _refused(code: str, reason: str) -> CliResult:
    envelope = {"status": "refused", "error": {"code": code, "reason": reason}}
    return CliResult(returncode=2, stdout=json.dumps(envelope), stderr="", json=envelope)


def test_synthetic_item_without_a_paths_file_is_one_plain_line() -> None:
    # Arrange
    reason = "PathsConfigError: /x/config/paths.toml: not found (copy config/paths.example.toml, fill it in)"
    # Act
    item = actions.synthetic_item(_refused("RunRefused", reason))
    # Assert
    assert item.ok is False
    assert item.message == actions.NO_PATHS_TEXT
    assert "copy config/paths.example.toml to config/paths.toml for real-data steps" in item.message
    assert "PathsConfigError" not in item.message


def test_lint_item_without_a_paths_file_is_one_plain_line() -> None:
    # Arrange
    reason = "/x/config/paths.toml: not found (copy config/paths.example.toml and fill it in)"
    # Act
    item = actions.lint_item(_refused("PathsConfigError", reason))
    # Assert
    assert item.message == actions.NO_PATHS_TEXT


def test_a_broken_paths_file_keeps_its_own_reason() -> None:
    # Arrange
    reason = "PathsConfigError: /x/config/paths.toml: not valid TOML: line 1"
    # Act
    item = actions.synthetic_item(_refused("RunRefused", reason))
    # Assert
    assert "not valid TOML" in item.message


def test_a_plain_stderr_refusal_without_a_paths_file_is_one_plain_line() -> None:
    # Arrange
    stderr = "refused: RunRefused: PathsConfigError: /x/config/paths.toml: not found (copy ...)\n"
    # Act
    item = actions.synthetic_item(CliResult(returncode=2, stdout="", stderr=stderr, json=None))
    # Assert
    assert item.message == actions.NO_PATHS_TEXT


# ------------------------------------------------- bug 2: step 3 without runs/catalog.json ----
def test_step_3_without_a_catalog_names_the_command_that_builds_it(tmp_path: Path) -> None:
    # Arrange
    sources = Sources(catalog_json=tmp_path / "runs" / "catalog.json")
    # Act
    text = view_data.missing_catalog_text("MNQ", sources)
    # Assert
    assert text is not None
    assert "pfb data catalog" in text
    assert "CatalogError" not in text and "\n" not in text


def test_step_3_with_a_catalog_has_no_missing_line(tmp_path: Path) -> None:
    # Arrange
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}")
    # Act
    text = view_data.missing_catalog_text("MNQ", Sources(catalog_json=catalog))
    # Assert
    assert text is None


# ------------------------------------------------- bug 4: the Runs table's strategy column ----
SHA = "60b3faaa" + "0" * 56


@pytest.mark.parametrize(
    ("name", "shown"),
    [
        (f"{SHA}.py", "60b3faaa"),
        ("dash_random.py", "dash_random.py"),
        ("", ""),
        ("60b3faaa.py", "60b3faaa.py"),
    ],
)
def test_the_runs_table_shows_sha8_for_an_upload_named_by_its_sha256(name: str, shown: str) -> None:
    # Act
    text = view_runs.strategy_label(name)
    # Assert
    assert text == shown


# ------------------------------------------------- bug 5: the uploader's size limit ----
def test_the_upload_limit_is_one_megabyte_in_the_config_file_and_the_flags() -> None:
    # Arrange
    server = tomllib.loads((REPO / ".streamlit" / "config.toml").read_text())["server"]
    # Act
    cmd = dash_cmd.streamlit_command(8605)
    # Assert
    assert server["maxUploadSize"] == 1
    assert cmd[cmd.index("--server.maxUploadSize") + 1] == "1"
