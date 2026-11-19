"""`pfb registry status|sync|genesis` and `pfb run` on temporary registries (never the real ledger)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from registry_env import Env  # pyright: ignore[reportMissingImports]
from typer.testing import CliRunner

import pfb.app as pfb_app
import pfb.cli.registry_cmd as registry_cmd
import pfb.cli.run_cmd as run_cmd
import pfb.runner.api as runner_api
from pfb.cli.main import app
from pfb.data.errors import NoBarsFound, PathsConfigError
from pfb.registry.api import begin

runner = CliRunner()


@pytest.fixture
def cli_env(env: Env, monkeypatch: pytest.MonkeyPatch) -> Env:
    monkeypatch.setattr(registry_cmd, "registry_config", lambda _: env.config)
    return env


def test_genesis_then_status_reports_n_and_open_runs(cli_env: Env) -> None:
    # Arrange
    assert runner.invoke(app, ["registry", "genesis"]).exit_code == 0
    receipt = begin(cli_env.request("synthetic"), config=cli_env.config)
    # Act
    result = runner.invoke(app, ["registry", "status"])
    # Assert
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert (report["n"], report["chain"], report["pe_in_sync"]) == (302, "ok", True)
    assert report["open_or_crashed"] == [receipt.run_id]


def test_a_second_genesis_is_refused_with_exit_2(cli_env: Env) -> None:
    runner.invoke(app, ["registry", "genesis"])
    result = runner.invoke(app, ["registry", "genesis"])
    assert result.exit_code == 2
    assert "GenesisExists" in result.output


def test_sync_with_nothing_new_appends_nothing(cli_env: Env) -> None:
    runner.invoke(app, ["registry", "genesis"])
    result = runner.invoke(app, ["registry", "sync"])
    assert result.exit_code == 0 and "nothing appended" in result.output


def test_status_on_a_tampered_ledger_is_refused(cli_env: Env) -> None:
    runner.invoke(app, ["registry", "genesis"])
    path = cli_env.config.registry_path
    path.write_bytes(path.read_bytes().replace(b'"n_after":302', b'"n_after":303'))
    result = runner.invoke(app, ["registry", "status"])
    assert result.exit_code == 2 and "ChainBroken" in result.output


def test_run_json_prints_the_result_contract_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # `pfb run --json` prints what pfb.app.run_backtest returns (the result contract) and nothing
    # else on stdout; the contract's keys and their sources are tested in tests/contract.
    seen: dict[str, Any] = {}
    contract = {"schema": "pfb.result/1", "run_id": "r", "kind": "engine_check", "n": 1,
                "ledger": {"begin_sha256": "b" * 64, "result_sha256": "c" * 64}}  # fmt: skip

    def fake_run(strategy: str, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return contract

    monkeypatch.setattr(pfb_app, "run_backtest", fake_run)
    result = runner.invoke(
        app, ["run", str(tmp_path / "s.py"), "--kind", "engine_check", "--check-id", "E3", "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == contract
    assert (seen["kind"], seen["check_id"]) == ("engine_check", "E3")


def test_run_refusal_exits_2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def refuse(strategy: str, **kwargs: Any) -> Any:
        raise pfb_app.Refused("RunRefused", "no")

    monkeypatch.setattr(pfb_app, "run_backtest", refuse)
    result = runner.invoke(app, ["run", str(tmp_path / "s.py"), "--kind", "trial"])
    assert result.exit_code == 2 and "RunRefused" in result.output


def _no_paths(*args: Any, **kwargs: Any) -> Any:
    raise PathsConfigError("config/paths.toml: not found (copy config/paths.example.toml and fill it in)")


@pytest.mark.parametrize("command", [["run", "--kind", "synthetic"], ["check"]], ids=["run", "check"])
def test_a_paths_config_error_before_begin_prints_refused_and_exits_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, command: list[str]
) -> None:
    # before the fix this was an uncaught PathsConfigError (a traceback, exit 1)
    strategy = tmp_path / "s.py"
    strategy.write_text("x = 1\n")
    monkeypatch.setattr(runner_api, "default_deps", _no_paths)
    monkeypatch.setattr(run_cmd, "default_deps", _no_paths)
    result = runner.invoke(app, [command[0], str(strategy), *command[1:]])
    assert result.exit_code == 2, result.output
    assert "refused:" in result.output and "PathsConfigError" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_a_gateway_refusal_before_begin_prints_refused_and_exits_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def closed(*args: Any, **kwargs: Any) -> Any:
        raise NoBarsFound("no catalogued file")

    strategy = tmp_path / "s.py"
    strategy.write_text("x = 1\n")
    monkeypatch.setattr(runner_api, "default_deps", closed)
    result = runner.invoke(app, ["run", str(strategy), "--kind", "engine_check", "--check-id", "E3"])
    assert result.exit_code == 2, result.output
    assert "refused:" in result.output and "NoBarsFound" in result.output
