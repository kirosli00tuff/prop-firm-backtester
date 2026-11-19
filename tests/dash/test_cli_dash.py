"""`pfb dash`: the Streamlit server command (127.0.0.1, usage statistics off, headless, no file watcher) and
the environment that tells the app where the ledger and the runs are. No server is started here."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

import pfb.cli.dash_cmd as dash_cmd
from pfb.cli.main import app

CONFIG_TOML = Path(__file__).resolve().parents[2] / ".streamlit" / "config.toml"


def _flag(cmd: list[str], name: str) -> str:
    return cmd[cmd.index(name) + 1]


def test_the_server_command_binds_localhost_and_turns_usage_statistics_off() -> None:
    # Act
    cmd = dash_cmd.streamlit_command(8601)
    # Assert
    assert cmd[1:4] == ["-m", "streamlit", "run"]
    assert Path(cmd[4]).name == "app.py" and Path(cmd[4]).is_file()
    assert _flag(cmd, "--server.address") == "127.0.0.1"
    assert _flag(cmd, "--server.port") == "8601"
    assert _flag(cmd, "--browser.gatherUsageStats") == "false"
    assert _flag(cmd, "--server.headless") == "true"
    assert _flag(cmd, "--server.fileWatcherType") == "none"
    assert _flag(cmd, "--server.enableAgentApi") == "false"
    assert _flag(cmd, "--client.showErrorLinks") == "false"


def test_the_theme_flags_agree_with_the_streamlit_config_file() -> None:
    # Arrange
    theme = tomllib.loads(CONFIG_TOML.read_text())["theme"]
    # Act
    cmd = dash_cmd.streamlit_command(8603)
    flags = {
        k.removeprefix("--theme."): v for k, v in zip(cmd, cmd[1:], strict=False) if k.startswith("--theme.")
    }
    # Assert
    assert flags == theme
    assert theme["base"] == "dark" and theme["backgroundColor"] == "#0E1014"
    assert theme["primaryColor"] == "#4C9AFF"


def test_the_toolbar_flag_agrees_with_the_streamlit_config_file_and_hides_deploy() -> None:
    # Arrange
    client = tomllib.loads(CONFIG_TOML.read_text())["client"]
    # Act
    cmd = dash_cmd.streamlit_command(8604)
    # Assert
    assert client["toolbarMode"] == "minimal"
    assert _flag(cmd, "--client.toolbarMode") == client["toolbarMode"]


def test_pfb_dash_runs_streamlit_with_the_dash_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    seen: dict = {}

    def fake_call(cmd: list[str], env: dict[str, str]) -> int:
        seen.update(cmd=cmd, env=env)
        return 0

    monkeypatch.setattr(dash_cmd.subprocess, "call", fake_call)
    runs = tmp_path / "runs"
    # Act
    done = CliRunner().invoke(app, ["dash", "--port", "8602", "--runs-dir", str(runs)])
    # Assert
    assert done.exit_code == 0, done.output
    assert _flag(seen["cmd"], "--server.port") == "8602"
    assert seen["env"]["PFB_DASH_RUNS"] == str(runs)
    assert seen["env"]["PFB_DASH_LEDGER"].endswith("ledger/registry.jsonl")
    assert "PFB_DASH_REPO" in seen["env"]


def test_pfb_dash_passes_the_servers_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dash_cmd.subprocess, "call", lambda cmd, env: 3)
    done = CliRunner().invoke(app, ["dash"])
    assert done.exit_code == 3
