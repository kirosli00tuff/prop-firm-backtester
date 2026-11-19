"""TRIALS.md section 4 checkpoint, and the CLI: `pfb registry checkpoint`, `pfb hypothesis register|show`,
`pfb run --kind trial`, `pfb check` warnings, `--param` parsing. Temporary repositories only."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from registry_env import HYPOTHESIS_REL, STRATEGY_REL, Env, git  # pyright: ignore[reportMissingImports]
from trial_support import FakeTrialEngine, clean, kinds, n_of  # pyright: ignore[reportMissingImports]
from typer.testing import CliRunner

import pfb.cli.registry_cmd as registry_cmd
import pfb.cli.run_cmd as run_cmd
import pfb.runner.api as runner_api
from pfb import app as service
from pfb.cli.main import app
from pfb.registry.api import begin, checkpoint
from pfb.registry.errors import LedgerNotTracked
from pfb.runner.api import RunRefused

runner = CliRunner()


# ------------------------------------------------------------------------------------------ checkpoint ----


def test_checkpoint_commits_the_ledger_alone(genesis_env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: a staged file, an unstaged edit and new ledger lines
    monkeypatch.setenv("PFB_COMMIT_TRAILER", "Trailer-One: a\nTrailer-Two: b")
    genesis_env.write("staged.txt", "staged\n")
    git(genesis_env.repo, "add", "staged.txt")
    genesis_env.write(STRATEGY_REL, "x = 5  # unstaged\n")
    begin(genesis_env.request("synthetic"), config=genesis_env.config)
    # Act
    done = checkpoint(genesis_env.config)
    # Assert
    assert done is not None and (done.n, done.records) == (302, 2)
    files = git(genesis_env.repo, "show", "--name-only", "--format=", "HEAD").split()
    assert files == ["ledger/registry.jsonl"]
    message = git(genesis_env.repo, "log", "-1", "--format=%B")
    assert message.startswith("ledger checkpoint: N 302, records 2\n\nTrailer-One: a\nTrailer-Two: b")
    porcelain = git(genesis_env.repo, "status", "--porcelain")
    assert "A  staged.txt" in porcelain and " M strategies/s.py" in porcelain


def test_checkpoint_with_the_ledger_at_head_commits_nothing(genesis_env: Env) -> None:
    head = git(genesis_env.repo, "rev-parse", "HEAD")
    assert checkpoint(genesis_env.config) is None
    assert git(genesis_env.repo, "rev-parse", "HEAD") == head


def test_checkpoint_outside_a_work_tree_is_refused(env: Env, tmp_path: Path) -> None:
    from pfb.registry.api import create_genesis

    plain = tmp_path / "plain"
    plain.mkdir()
    config = replace(env.config, registry_path=plain / "registry.jsonl", repo_root=plain)
    create_genesis(config)
    with pytest.raises(LedgerNotTracked):
        checkpoint(config)


def test_checkpoint_of_a_ledger_outside_the_repository_is_refused(genesis_env: Env, tmp_path: Path) -> None:
    from pfb.registry.api import create_genesis

    config = replace(genesis_env.config, registry_path=tmp_path / "elsewhere.jsonl")
    create_genesis(config)
    with pytest.raises(LedgerNotTracked):
        checkpoint(config)


# ---------------------------------------------------------------------------------------------- CLI ----


@pytest.fixture
def cli_env(genesis_env: Env, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Env, FakeTrialEngine]]:
    """The CLI on the temporary repository; run folders go to its own runs/, never the real one."""
    engine = FakeTrialEngine()
    deps = engine.deps(genesis_env)
    monkeypatch.setattr(registry_cmd, "registry_config", lambda _: genesis_env.config)
    monkeypatch.setattr(registry_cmd, "default_deps", lambda *_: deps)
    monkeypatch.setattr(runner_api, "default_deps", lambda *_: deps)
    monkeypatch.setattr(run_cmd, "default_deps", lambda *_: deps)
    monkeypatch.chdir(genesis_env.repo)
    with service.using(runs_dir=genesis_env.repo / "runs"):
        yield genesis_env, engine


def test_cli_registers_shows_and_runs_a_trial(cli_env: tuple[Env, FakeTrialEngine]) -> None:
    env, engine = cli_env
    env.write(HYPOTHESIS_REL, env.hypothesis_text())
    env.commit()
    # register
    res = runner.invoke(app, ["hypothesis", "register", HYPOTHESIS_REL])
    assert res.exit_code == 0, res.output
    assert res.output.startswith("registered H1: N 302, budget 2")
    # show
    shown = runner.invoke(app, ["hypothesis", "show", "H1"])
    assert shown.exit_code == 0 and json.loads(shown.output)["hypothesis_id"] == "H1"
    # run
    out = runner.invoke(app, ["run", STRATEGY_REL, "--kind", "trial", "--hypothesis", HYPOTHESIS_REL,
                              "--param", "a=2", "--json"])  # fmt: skip
    assert out.exit_code == 0, out.output
    payload = json.loads(out.output)
    assert (payload["kind"], payload["n"] - payload["n_added"], payload["n"]) == ("trial", 302, 303)
    assert engine.called("run_strategy")[0]["params"] == {"a": 2}
    assert clean(env)


def test_cli_register_json_prints_the_record_and_the_text_line_is_unchanged(
    cli_env: tuple[Env, FakeTrialEngine], tmp_path: Path
) -> None:
    """--json prints the bytes `hypothesis show` prints; the text line is unchanged."""
    # Arrange: a committed hypothesis; --dry-run with and without --json
    env, _ = cli_env
    env.write(HYPOTHESIS_REL, env.hypothesis_text())
    env.commit()
    dry_plain = runner.invoke(app, ["hypothesis", "register", HYPOTHESIS_REL, "--dry-run"])
    dry_json = runner.invoke(app, ["hypothesis", "register", HYPOTHESIS_REL, "--dry-run", "--json"])
    # Act
    res = runner.invoke(app, ["hypothesis", "register", HYPOTHESIS_REL, "--json"])
    # Assert
    assert res.exit_code == 0, res.output
    shown = runner.invoke(app, ["hypothesis", "show", "H1"])
    assert res.stdout == shown.stdout
    record = json.loads(res.stdout)
    assert res.stdout == json.dumps(record, sort_keys=True) + "\n"
    assert (record["hypothesis_id"], record["n_after"]) == ("H1", 302)
    assert (dry_plain.exit_code, dry_plain.stdout) == (dry_json.exit_code, dry_json.stdout)
    assert kinds(env) == ["genesis", "hypothesis"]


def test_cli_register_text_line_is_exact(cli_env: tuple[Env, FakeTrialEngine]) -> None:
    """The text line, and the dashboard's copy of it built from the --json record, are the same bytes."""
    from pfb.dash.actions import register_text

    env, _ = cli_env
    env.write(HYPOTHESIS_REL, env.hypothesis_text())
    env.commit()
    res = runner.invoke(app, ["hypothesis", "register", HYPOTHESIS_REL])
    assert res.exit_code == 0, res.output
    record = json.loads(runner.invoke(app, ["hypothesis", "show", "H1"]).stdout)
    power = record["power"]
    assert res.stdout == (
        f"registered H1: N {record['n_after']}, budget {record['budget']}, "
        f"min_edge_cents {power['min_edge_cents']} at N {power['n_used']}\n"
    )
    assert res.stdout == register_text(record) + "\n"


def test_cli_register_json_refusal_is_the_text_refusal(cli_env: tuple[Env, FakeTrialEngine]) -> None:
    env, _ = cli_env
    env.write(HYPOTHESIS_REL, env.hypothesis_text(power__target_edge_cents="300"))
    env.commit()
    res = runner.invoke(app, ["hypothesis", "register", HYPOTHESIS_REL, "--json"])
    assert res.exit_code == 2 and "refused: PowerInsufficient:" in res.output
    assert n_of(env) == 302


def test_cli_trial_refusals_print_the_class_and_exit_2(cli_env: tuple[Env, FakeTrialEngine]) -> None:
    env, _ = cli_env
    env.write(HYPOTHESIS_REL, env.hypothesis_text())
    env.commit()
    assert runner.invoke(app, ["hypothesis", "register", HYPOTHESIS_REL]).exit_code == 0
    res = runner.invoke(app, ["run", STRATEGY_REL, "--kind", "trial", "--hypothesis", HYPOTHESIS_REL,
                              "--fees", "lucid"])  # fmt: skip
    assert res.exit_code == 2 and "refused: SuiteMismatch:" in res.output
    res = runner.invoke(app, ["run", STRATEGY_REL, "--kind", "trial", "--hypothesis", HYPOTHESIS_REL,
                              "--param", "a=9"])  # fmt: skip
    assert res.exit_code == 2 and "refused: RunRefused:" in res.output  # a = 9 is outside the SPEC grid
    assert kinds(env) == ["genesis", "hypothesis"]


def test_cli_registration_refusal_exits_2(cli_env: tuple[Env, FakeTrialEngine]) -> None:
    env, _ = cli_env
    env.write(HYPOTHESIS_REL, env.hypothesis_text(power__target_edge_cents="300"))
    env.commit()
    res = runner.invoke(app, ["hypothesis", "register", HYPOTHESIS_REL])
    assert res.exit_code == 2 and "refused: PowerInsufficient:" in res.output
    assert n_of(env) == 302


def test_cli_registry_checkpoint(cli_env: tuple[Env, FakeTrialEngine]) -> None:
    env, _ = cli_env
    begin(env.request("synthetic"), config=env.config)
    res = runner.invoke(app, ["registry", "checkpoint"])
    assert res.exit_code == 0 and res.output.startswith("ledger checkpoint: N 302, records 2, commit ")
    assert runner.invoke(app, ["registry", "checkpoint"]).output.startswith("ledger already at HEAD")


def test_cli_synthetic_neighbourhood(cli_env: tuple[Env, FakeTrialEngine]) -> None:
    env, _ = cli_env
    res = runner.invoke(app, ["run", STRATEGY_REL, "--kind", "synthetic", "--param", "a=2", "--neighbourhood",
                              "--json"])  # fmt: skip
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert len(payload["neighbourhood"]["run_ids"]) == 2 and payload["n"] == 302
    assert (env.repo / "runs" / payload["run_id"] / "neighbourhood.json").is_file()
    for run_id in payload["neighbourhood"]["run_ids"]:  # written under the temporary repository's runs/
        assert (env.repo / "runs" / run_id / "neighbourhood.json").exists() is False
        assert (env.repo / "runs" / run_id / "summary.json").is_file()


def test_cli_check_prints_the_step_one_warnings(cli_env: tuple[Env, FakeTrialEngine]) -> None:
    _, engine = cli_env
    engine.lookahead_warnings = ("look-ahead check saw no signal",)
    res = runner.invoke(app, ["check", STRATEGY_REL])
    assert res.exit_code == 0
    assert "warning: step 1: look-ahead check saw no signal" in res.output


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        (["a=2"], {"a": 2}),
        (["a=2.5", "b=true"], {"a": 2.5, "b": True}),
        (['s="x"', "t=word"], {"s": "x", "t": "word"}),
    ],
)
def test_param_values_are_toml_scalars(items: list[str], expected: dict) -> None:
    assert run_cmd.parse_params(items) == expected


@pytest.mark.parametrize("items", [["a"], ["=1"], ["a=[1, 2]"]])
def test_a_malformed_param_is_refused(items: list[str]) -> None:
    with pytest.raises(RunRefused):
        run_cmd.parse_params(items)
