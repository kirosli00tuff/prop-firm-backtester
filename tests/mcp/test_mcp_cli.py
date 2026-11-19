"""`pfb mcp` is registered and serves the same server as `python -m pfb.mcp` (no stdio started here)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

import pfb.mcp.server as server
from pfb.cli.main import app

runner = CliRunner()


def test_pfb_mcp_is_a_registered_command() -> None:
    # Act
    result = runner.invoke(app, ["mcp", "--help"])
    # Assert
    assert result.exit_code == 0, result.output
    assert "MCP client" in result.output


def test_pfb_mcp_calls_serve_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    calls: list[str] = []
    monkeypatch.setattr(server, "serve_stdio", lambda: calls.append("served"))
    # Act
    result = runner.invoke(app, ["mcp"])
    # Assert
    assert result.exit_code == 0, result.output
    assert calls == ["served"]


class FakeServer:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def run(self, transport: str) -> None:
        self.calls.append(f"run:{transport}")


def test_serve_stdio_moves_to_the_repo_root_then_runs_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    calls: list[str] = []
    monkeypatch.setattr(server.os, "chdir", lambda path: calls.append(f"chdir:{path}"))
    monkeypatch.setattr(server, "build_server", lambda: FakeServer(calls))
    # Act
    server.serve_stdio()
    # Assert
    assert calls == [f"chdir:{server.REPO_ROOT}", "run:stdio"]
    assert (server.REPO_ROOT / "pyproject.toml").is_file()
