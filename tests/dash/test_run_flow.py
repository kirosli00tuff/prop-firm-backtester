"""Step 4's run path in pfb.dash.actions and pfb.dash.view_run.

Nothing here registers or runs a trial. The register and trial children are replaced by a recorder (the
command lines it receives are checked against the pfb CLI's own parser), git state is read from temporary
repositories, and one real `pfb hypothesis register --dry-run` runs against a file outside this repository:
it appends nothing (the ledger's sha256 is compared before and after) and is refused as not committed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import typer

from pfb.dash import actions, hypgen, view_run
from pfb.dash.actions import (
    BUSY_TEXT,
    CliResult,
    StrategyState,
    commit_command,
    commit_state,
    dry_run_of,
    machine_lock,
    read_spec,
    register_args,
    run_trials,
    save_hypothesis,
    trial_args,
)
from pfb.dash.data import DashConfig, ledger_view
from pfb.dash.hypgen import HypForm, PowerSource, build, hypothesis_id

REPO = Path(__file__).resolve().parents[2]
TEMPLATE_REL = "strategies/_template.py"
LEDGER = REPO / "ledger" / "registry.jsonl"
PATHS_TOML = REPO / "config" / "paths.toml"
STRATEGY_TEXT = (REPO / TEMPLATE_REL).read_text()
OPEN_WINDOW = {"root": "MNQ", "first": "2025-04-01", "last": "2026-05-08", "status": "open"}
NOW = datetime(2026, 11, 17, 12, 0, 0, tzinfo=UTC)
SOURCE = PowerSource("synthetic-x", 21, 21, 2170, 0.4, 1.6)
REGISTER_REFUSAL = (
    "refused: PowerInsufficient: target_edge_cents 10 is below the minimum detectable edge 2601"
)
DRY_RUN_TIMEOUT_S = 600.0


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True, text=True)
    return done.stdout


def _form(**changes: Any) -> HypForm:
    hyp_id = hypothesis_id(hashlib.sha256(STRATEGY_TEXT.encode()).hexdigest(), NOW)
    base = HypForm(hyp_id, "a statement", 3000, "off", "a stance", False, SOURCE)
    return replace(base, **changes)


def _state(rel: str, text: str = STRATEGY_TEXT) -> StrategyState:
    sha = hashlib.sha256(text.encode()).hexdigest()
    return StrategyState(rel_path=rel, sha256=sha, spec=read_spec(text), checks=(), passed=True)


@pytest.fixture
def repo(tmp_path: Path) -> DashConfig:
    """A temporary git repository with one commit, an upload and a ledger path (never this repo's)."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "test")
    (root / "README").write_text("x\n")
    _git(root, "add", "README")
    _git(root, "commit", "-q", "-m", "init")
    return DashConfig(repo_root=root, ledger_path=root / "ledger" / "registry.jsonl", runs_dir=root / "runs")


def _upload_and_hypothesis(config: DashConfig) -> tuple[StrategyState, str]:
    upload = actions.save_upload(STRATEGY_TEXT, config.repo_root)
    state = _state(upload.rel_path)
    form = _form(hypothesis_id=hypothesis_id(state.sha256, NOW))
    text = build(state, [OPEN_WINDOW], ["topstep/combine_xfa_standard_50k"], form)
    saved = save_hypothesis(text, form.hypothesis_id, config.repo_root)
    return state, saved.rel_path


def _commit_all(config: DashConfig) -> None:
    _git(config.repo_root, "add", "-A")
    _git(config.repo_root, "commit", "-q", "-m", "files")


class Recorder:
    """Stands in for actions._run_unlocked: records each child's pfb arguments, answers from a queue."""

    def __init__(self, answers: Sequence[CliResult] = ()) -> None:
        self.calls: list[list[str]] = []
        self.answers = list(answers)

    def __call__(self, args: Sequence[str], **_: Any) -> CliResult:
        self.calls.append(list(args))
        if self.answers:
            return self.answers.pop(0)
        return CliResult(0, "", "", None)


def _trial_ok(run_id: str, n: int) -> CliResult:
    out = {"schema": "pfb.result/1", "status": "ok", "run_id": run_id, "n": n, "n_added": 1}
    return CliResult(0, json.dumps(out), "", out)


def _refused(code: str, reason: str) -> CliResult:
    out = {"schema": "pfb.result/1", "status": "refused", "error": {"code": code, "reason": reason}}
    return CliResult(2, json.dumps(out), f"refused: {code}: {reason}", out)


def _parses_as_cli(args: Sequence[str]) -> None:
    """The pfb CLI's own parser accepts args (nothing is invoked)."""
    from pfb.cli.main import app

    command: Any = typer.main.get_command(app)
    rest = list(args)
    while hasattr(command, "commands"):  # a group: descend by name
        command, rest = command.commands[rest[0]], rest[1:]
    command.make_context(command.name or "", rest)  # parses and type-checks every option; invokes nothing


# ----------------------------------------------------------------------------- save and git ----
def test_save_hypothesis_is_idempotent_and_never_overwrites(repo: DashConfig) -> None:
    # Act
    first = save_hypothesis("a = 1\n", "h_x", repo.repo_root)
    again = save_hypothesis("a = 1\n", "h_x", repo.repo_root)
    # Assert
    assert first == again
    assert first.rel_path == "hypotheses/uploads/h_x.toml"
    assert (repo.repo_root / first.rel_path).read_text() == "a = 1\n"
    with pytest.raises(actions.HypothesisRefused):
        save_hypothesis("a = 2\n", "h_x", repo.repo_root)
    with pytest.raises(actions.HypothesisRefused):
        save_hypothesis("a = 1\n", "../escape", repo.repo_root)


def test_commit_state_reads_head_and_the_tree(repo: DashConfig) -> None:
    # Arrange
    state, hyp_rel = _upload_and_hypothesis(repo)
    paths = (state.rel_path, hyp_rel)
    # Act
    before = commit_state(repo.repo_root, paths)
    _commit_all(repo)
    after = commit_state(repo.repo_root, paths)
    (repo.repo_root / "README").write_text("changed\n")
    dirty = commit_state(repo.repo_root, paths)
    (repo.repo_root / "README").write_text("x\n")
    (repo.repo_root / "ledger").mkdir()
    repo.ledger_path.write_text("{}\n")
    ledger_only = commit_state(repo.repo_root, paths)
    # Assert
    assert before.committed == ((state.rel_path, False), (hyp_rel, False)) and not before.ready
    assert after.committed == ((state.rel_path, True), (hyp_rel, True)) and after.ready
    assert dirty.dirty == ("README",) and not dirty.ready
    assert "tree not clean" in dirty.text()
    assert ledger_only.ready


def test_commit_command_is_the_exact_git_line() -> None:
    # Act
    line = commit_command(["strategies/uploads/a.py", "hypotheses/uploads/h_a.toml"], "hypothesis h_a: it's")
    # Assert
    paths = "strategies/uploads/a.py hypotheses/uploads/h_a.toml"
    assert line == (f"git add -- {paths} && "
                    f"git commit --only -m 'hypothesis h_a: it'\"'\"'s' -- {paths}")  # fmt: skip


# ----------------------------------------------------------------------------- run_trials ----
def test_uncommitted_files_mean_no_register_call(repo: DashConfig, monkeypatch: pytest.MonkeyPatch,
                                                 tmp_path: Path) -> None:  # fmt: skip
    # Arrange
    state, hyp_rel = _upload_and_hypothesis(repo)
    recorder = Recorder()
    monkeypatch.setattr(actions, "_run_unlocked", recorder)
    # Act
    steps = run_trials(state.rel_path, hyp_rel, [1], config=repo, registered=False, lock_path=tmp_path / "l")
    # Assert
    assert recorder.calls == []
    assert [(s.name, s.ok) for s in steps] == [(actions.STEP_COMMITTED, False)]
    assert "not committed" in steps[0].message


def test_a_busy_lock_means_no_call(repo: DashConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Arrange
    state, hyp_rel = _upload_and_hypothesis(repo)
    _commit_all(repo)
    recorder = Recorder()
    monkeypatch.setattr(actions, "_run_unlocked", recorder)
    lock = tmp_path / "run.lock"
    # Act
    with machine_lock(lock) as held:
        assert held
        steps = run_trials(state.rel_path, hyp_rel, [1], config=repo, registered=False, lock_path=lock)
        dry = actions.dry_run(repo.repo_root / hyp_rel, config=repo, lock_path=lock)
    # Assert
    assert recorder.calls == []
    assert [(s.name, s.ok, s.message) for s in steps] == [(actions.STEP_LOCK, False, BUSY_TEXT)]
    assert not dry.ok and dry.message == BUSY_TEXT


def test_committed_files_register_then_run_with_the_clis_command_lines(
    repo: DashConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    state, hyp_rel = _upload_and_hypothesis(repo)
    _commit_all(repo)
    register_line = "registered h_x: N 1001, budget 1, min_edge_cents 2601 at N 1002"
    record = {"hypothesis_id": "h_x", "n_after": 1001, "budget": 1,
              "power": {"min_edge_cents": 2601, "n_used": 1002}}  # fmt: skip
    register_out = json.dumps(record, sort_keys=True)
    recorder = Recorder([CliResult(0, register_out + "\n", "", record), _trial_ok("trial-1", 1002)])
    monkeypatch.setattr(actions, "_run_unlocked", recorder)
    seen: list[str] = []
    strategy, hyp = repo.repo_root / state.rel_path, repo.repo_root / hyp_rel
    # Act
    steps = run_trials(state.rel_path, hyp_rel, [1], config=repo, registered=False, on_step=seen.append,
                       lock_path=tmp_path / "l")  # fmt: skip
    # Assert
    assert recorder.calls == [
        ["hypothesis", "register", str(hyp), "--json"],
        ["run", str(strategy), "--kind", "trial", "--hypothesis", str(hyp), "--window", "1", "--json"],
    ]
    for args in [*recorder.calls, register_args(hyp, dry_run=True)]:
        _parses_as_cli(args)
    assert [s.ok for s in steps] == [True, True, True]
    assert steps[1].message == register_line
    assert steps[2].run_id == "trial-1" and "N 1002 (+1)" in steps[2].message
    assert seen == [actions.STEP_COMMITTED, actions.STEP_REGISTER, "trial, window 1"]


def test_a_registered_hypothesis_is_not_registered_again(
    repo: DashConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    state, hyp_rel = _upload_and_hypothesis(repo)
    _commit_all(repo)
    recorder = Recorder([_trial_ok("t1", 5), _refused("BudgetExceeded", "budget 1 is spent")])
    monkeypatch.setattr(actions, "_run_unlocked", recorder)
    # Act
    steps = run_trials(state.rel_path, hyp_rel, [1, 2, 3], config=repo, registered=True,
                       lock_path=tmp_path / "l")  # fmt: skip
    # Assert: no register call; the second trial's refusal stops the third
    assert [c[0] for c in recorder.calls] == ["run", "run"]
    assert [c[-2] for c in recorder.calls] == ["1", "2"]
    assert [s.ok for s in steps] == [True, True, False]
    assert steps[-1].message == "refused: BudgetExceeded: budget 1 is spent"


def test_a_register_refusal_is_shown_and_no_trial_runs(
    repo: DashConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    state, hyp_rel = _upload_and_hypothesis(repo)
    _commit_all(repo)
    recorder = Recorder([CliResult(2, "", REGISTER_REFUSAL + "\n", None)])
    monkeypatch.setattr(actions, "_run_unlocked", recorder)
    # Act
    steps = run_trials(state.rel_path, hyp_rel, [1], config=repo, registered=False, lock_path=tmp_path / "l")
    # Assert
    assert len(recorder.calls) == 1
    assert (steps[-1].name, steps[-1].ok) == (actions.STEP_REGISTER, False)
    assert REGISTER_REFUSAL in steps[-1].message and "2601" in steps[-1].message


def test_a_closed_window_refusal_of_the_trial_is_shown(
    repo: DashConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    state, hyp_rel = _upload_and_hypothesis(repo)
    _commit_all(repo)
    reason = "MNQ 2026-05-09..2026-05-30 touches a closed window: demo_embargo_tail"
    monkeypatch.setattr(actions, "_run_unlocked", Recorder([_refused("ClosedWindowTouched", reason)]))
    # Act
    steps = run_trials(state.rel_path, hyp_rel, [1], config=repo, registered=True, lock_path=tmp_path / "l")
    # Assert
    assert steps[-1].message == f"refused: ClosedWindowTouched: {reason}"
    assert steps[-1].run_id is None


def test_dry_run_output_is_read_into_refusals() -> None:
    # Arrange
    refused = {"schema": actions.DRY_RUN_SCHEMA, "ok": False, "appended": False, "path": "x",
               "refusals": [{"code": "PowerInsufficient", "reason": "target 10 below 2601"}]}  # fmt: skip
    passed = {**refused, "ok": True, "refusals": []}
    # Act
    bad = dry_run_of(CliResult(2, json.dumps(refused), "", refused))
    good = dry_run_of(CliResult(0, json.dumps(passed), "", passed))
    broken = dry_run_of(CliResult(1, "", "Traceback\nKeyError: x", None))
    # Assert
    assert not bad.ok and bad.refusals == (actions.Refusal("PowerInsufficient", "target 10 below 2601"),)
    assert good.ok and good.refusals == ()
    assert not broken.ok and "KeyError: x" in broken.message


def test_trial_args_carry_the_window_index() -> None:
    # Act / Assert
    assert trial_args(Path("/r/s.py"), Path("/r/h.toml"), 2)[-3:] == ["--window", "2", "--json"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_one_real_dry_run_appends_nothing(tmp_path: Path) -> None:
    # Arrange: the template (committed in this repo) and a hypothesis file outside the repository
    state = _state(TEMPLATE_REL)
    form = _form(hypothesis_id=hypothesis_id(state.sha256, NOW))
    text = build(state, [OPEN_WINDOW], ["topstep/combine_xfa_standard_50k"], form)
    hyp = tmp_path / f"{form.hypothesis_id}.toml"
    hyp.write_text(text)
    before = _sha(LEDGER)
    # Act
    dry = actions.dry_run(hyp, config=DashConfig.from_env({}), env={"PYTHONDONTWRITEBYTECODE": "1"},
                          timeout_s=DRY_RUN_TIMEOUT_S, lock_path=tmp_path / "run.lock")  # fmt: skip
    # Assert
    assert _sha(LEDGER) == before
    assert not dry.ok, dry
    assert "InputNotCommitted" in [r.code for r in dry.refusals], dry


# ---------------------------------------------------------------------------- view (AppTest) ----
def _run_page(config: object, state: object, windows: object) -> None:
    # AppTest runs this function's source alone: builtins only in the annotations
    import streamlit as st

    from pfb.dash import view_run

    returned = view_run.render(config, state, windows)  # pyright: ignore[reportArgumentType]
    st.session_state["returned"] = returned


@pytest.fixture
def page_env(tmp_path: Path, dash_env: Any) -> tuple[DashConfig, StrategyState]:
    """A copy of the dashboard fixture's repository and upstream (git, ledger with ok synthetic runs)."""
    base = dash_env.config.repo_root.parent
    copy = tmp_path / "copy"
    shutil.copytree(base, copy, symlinks=True)
    ledger_rel = dash_env.config.ledger_path.relative_to(dash_env.config.repo_root)
    repo = copy / "repo"
    config = DashConfig(repo, repo / ledger_rel, repo / "runs", pe_root=copy / "pe")
    text = (config.repo_root / "strategies" / "dash_random.py").read_text()
    return config, _state("strategies/dash_random.py", text)


def test_view_run_builds_shows_refusals_and_runs_through_actions(
    page_env: tuple[DashConfig, StrategyState], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: the dry run and the run are replaced; everything else is real on the copied repository
    from streamlit.testing.v1 import AppTest

    config, state = page_env
    view = ledger_view(config)
    if view_run.synthetic_source(view, state.sha256) is None:
        pytest.skip("the fixture's synthetic runs carry no realized power figures")
    window = {"root": str((state.spec or {})["root"]), "first": "2025-01-02", "last": "2025-03-31"}
    refusal = actions.Refusal(
        "PowerInsufficient", "target_edge_cents 5 is below the minimum detectable edge 4321"
    )
    monkeypatch.setattr(
        actions, "dry_run", lambda *_a, **_k: actions.DryRun(False, (refusal,), "1 refusal(s)")
    )
    calls: list[tuple[Any, ...]] = []

    def fake_trials(strategy_rel: str, hyp_rel: str, indexes: list[int], **kwargs: Any) -> tuple[Any, ...]:
        calls.append((strategy_rel, hyp_rel, tuple(indexes), kwargs["registered"]))
        return (actions.StepResult(actions.STEP_TRIAL, True, "status ok, run trial-9", "trial-9"),)

    monkeypatch.setattr(actions, "run_trials", fake_trials)
    at = AppTest.from_function(_run_page, args=(config, state, [window]), default_timeout=120)
    # Act: fill the form, build, validate
    at.run()
    at.text_area(key="run_statement").input("a statement")
    at.number_input(key="run_target").set_value(5)
    at.text_input(key="run_stance").input("a stance")
    at.run()
    at.button(key="run_build").click()
    at.run()
    at.button(key="run_dry").click()
    at.run()
    shown = " ".join(m.value for m in at.markdown)
    codes = " ".join(c.value for c in at.code)
    # Assert: the file is shown in full, the refusal with its figure, the commit state and command
    assert not at.exception, [e.value for e in at.exception]
    saved = sorted((config.repo_root / hypgen.HYPOTHESES_UPLOADS_REL).glob("*.toml"))
    assert len(saved) == 1 and saved[0].read_text().strip() in codes
    assert "PowerInsufficient" in shown and "4321" in shown
    assert "committed: **no**" in shown and "git add" in codes
    assert not calls
    # Act: commit, refresh, confirm and run
    _git(config.repo_root, "add", "-A")
    _git(config.repo_root, "commit", "-q", "-m", "files")
    at.button(key="run_refresh").click()
    at.run()
    n = view.n
    box = next(c for c in at.checkbox if c.label == view_run.confirm_text(n, 1))
    box.check()
    at.run()
    at.button(key="run_go").click()
    at.run()
    # Assert
    assert not at.exception, [e.value for e in at.exception]
    assert calls == [(state.rel_path, saved[0].relative_to(config.repo_root).as_posix(), (1,), False)]
    assert at.session_state["returned"] == "trial-9"


def test_validate_reruns_after_the_commit_and_leaves_no_stale_fail(
    page_env: tuple[DashConfig, StrategyState], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: before the commit the dry run refuses InputNotCommitted, after it passes
    from streamlit.testing.v1 import AppTest

    config, state = page_env
    if view_run.synthetic_source(ledger_view(config), state.sha256) is None:
        pytest.skip("the fixture's synthetic runs carry no realized power figures")
    window = {"root": str((state.spec or {})["root"]), "first": "2025-01-02", "last": "2025-03-31"}
    not_committed = actions.Refusal("InputNotCommitted", "hypotheses/uploads/h.toml is not tracked by git")
    calls: list[int] = []

    def fake_dry_run(*_a: Any, **_k: Any) -> actions.DryRun:
        calls.append(1)
        if len(calls) == 1:
            return actions.DryRun(False, (not_committed,), "1 refusal(s)")
        return actions.DryRun(True, (), "no refusal; N unchanged")

    monkeypatch.setattr(actions, "dry_run", fake_dry_run)
    at = AppTest.from_function(_run_page, args=(config, state, [window]), default_timeout=120)
    at.run()
    at.text_area(key="run_statement").input("a statement")
    at.number_input(key="run_target").set_value(5)
    at.text_input(key="run_stance").input("a stance")
    at.run()
    at.button(key="run_build").click()
    at.run()
    # Act: validate before the commit
    at.button(key="run_dry").click()
    at.run()
    before = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    # Assert: InputNotCommitted is shown as expected before the commit, not as a fail
    lines = [m.value for m in at.markdown if "InputNotCommitted" in m.value]
    assert lines and all("(fail)" not in line for line in lines)
    assert view_run.NOT_COMMITTED_YET in before
    # Act: commit by hand and refresh; nothing else is clicked
    _git(config.repo_root, "add", "-A")
    _git(config.repo_root, "commit", "-q", "-m", "files")
    at.button(key="run_refresh").click()
    at.run()
    after = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    # Assert: the dry run reran once on its own and only its pass is shown
    assert not at.exception, [e.value for e in at.exception]
    assert len(calls) == 2
    assert "InputNotCommitted" not in after
    assert "no refusal; N unchanged" in after
    at.run()
    assert len(calls) == 2
