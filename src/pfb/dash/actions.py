"""The dashboard's only writer and only subprocess spawner (docs/design/DASH_V2.md).

Every other module under pfb.dash stays read-only (tests/dash/test_read_only.py). This one does three things:

- save_upload: a strategy pasted or dropped in the dashboard becomes strategies/uploads/<sha256>.py under the
  dashboard's repository (sha256 of the exact bytes; idempotent; non UTF-8, blank or oversized refused).
- run_cli: the pfb command line in a child process, one at a time on this machine, with a memory cap and a
  timeout. It is the dashboard's one way to run anything; the dashboard adds no second execution path.
- check: step 2 of the Backtest page, `pfb check --json` then `pfb run --kind synthetic --json` (N unchanged).

Where the child reads and writes. The CLI itself has no option or variable that names a ledger or a runs
directory for `pfb run`, `pfb check` or `pfb hypothesis register`: they use pfb.app's AppContext, whose
default is this repository (runner default_deps: ledger/registry.jsonl, runs/, the upstream checkout of
config/paths.toml). The child is therefore a small bootstrap (CHILD_BOOTSTRAP) that reads the dashboard's
DashConfig from the PFB_DASH_* variables (pfb.dash.data) and, only when they name somewhere other than the
defaults, enters `pfb.app.using(deps=default_deps() with that registry config, runs_dir, ledger)` before it
calls the pfb.cli.main app. With the defaults the child runs the CLI exactly as a shell would. `pfb prop`
takes --runs-dir and --ledger options of its own (defaults: this repository's); a caller passes them.

A synthetic run appends its begin and result records to the ledger it is pointed at (kind "synthetic", N +0)
and writes runs/<run_id>/; a real-data run refuses any ledger but this repository's (runner
refuse_throwaway_ledger).
"""

from __future__ import annotations

import ast
import contextlib
import fcntl
import functools
import hashlib
import json
import os
import resource
import shlex
import signal
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pfb.dash.data import ENV_LEDGER, ENV_PE, ENV_REPO, ENV_RUNS, DashConfig
from pfb.dash.hypgen import hypothesis_rel
from pfb.registry.api import LEDGER_REL
from pfb.registry.errors import GitError
from pfb.registry.gitstate import (
    changed_paths,
    commit_paths,
    head_blob,
    staged_paths,
    with_trailer,
)
from pfb.registry.inputs import ID_RE

UPLOADS_REL = "strategies/uploads"
MAX_UPLOAD_BYTES = 256 * 1024
GIB = 1024**3
DEFAULT_MEM_BYTES = 4 * GIB
DEFAULT_TIMEOUT_S = 900.0
KILL_GRACE_S = 5.0
# glibc reserves up to 8 x cores malloc arenas of 64 MiB of address space each; on a many-core laptop that
# alone breaks a 6 GiB RLIMIT_AS (a synthetic run aborted in polars' allocator) while two arenas let the same
# run live under 4 GiB. Address-space reservation only: results do not change. A caller's env overrides it.
CHILD_DEFAULT_ENV = {"MALLOC_ARENA_MAX": "2"}
FAILURE_TAIL_LINES = 3
LOCK_ENV = "PFB_DASH_LOCK"
DEFAULT_LOCK_PATH = Path.home() / ".cache" / "pfb" / "dash" / "run.lock"
STATE_DONE, STATE_BUSY, STATE_TIMEOUT = "done", "busy", "timeout"
CHECK_LINT, CHECK_SYNTHETIC = "lint and look-ahead", "synthetic run"
SPEC_NAME = "SPEC"
BUSY_TEXT = "another dashboard action is running on this machine; try again when it ends"
# a checkout without config/paths.toml gets this line, not the PathsConfigError text.
NO_PATHS_TEXT = (
    "no config/paths.toml: copy config/paths.example.toml to config/paths.toml for real-data steps"
)
PATHS_MISSING_MARKS = ("PathsConfigError", "paths.toml: not found")

# The child: read the dashboard's DashConfig from the PFB_DASH_* variables; with the defaults run the CLI as
# is, else point pfb.app at that ledger, runs directory and repository first. argv[1:] are the pfb arguments.
CHILD_BOOTSTRAP = """\
import os, sys
from dataclasses import replace
from pfb.dash.data import DashConfig
from pfb.cli.main import app
config = DashConfig.from_env(os.environ)
if config == DashConfig.from_env({}):
    app(args=sys.argv[1:], prog_name="pfb")
else:
    from pfb import app as service
    from pfb.registry.api import RegistryConfig
    from pfb.runner.deps import default_deps
    registry = replace(
        RegistryConfig.for_repo(config.pe_root, config.repo_root), registry_path=config.ledger_path
    )
    deps = replace(default_deps(), registry_config=lambda: registry)
    with service.using(deps=deps, runs_dir=config.runs_dir, ledger=config.ledger_path):
        app(args=sys.argv[1:], prog_name="pfb")
"""


class UploadRefused(ValueError):
    """The uploaded text is empty, not UTF-8 or larger than MAX_UPLOAD_BYTES."""


@dataclass(frozen=True)
class Upload:
    rel_path: str  # strategies/uploads/<sha256>.py, relative to the dashboard's repository
    sha256: str


@dataclass(frozen=True)
class CliResult:
    """One child run. state "done" (returncode is the child's), "busy" (nothing ran, returncode None) or
    "timeout" (the process group was killed; returncode is the kill's). json: the last stdout line when it
    is a JSON object, else None."""

    returncode: int | None
    stdout: str
    stderr: str
    json: dict[str, Any] | None
    state: str = STATE_DONE


@dataclass(frozen=True)
class CheckItem:
    name: str
    ok: bool
    message: str


@dataclass(frozen=True)
class StrategyState:
    """Step 1-2's outcome for one upload: the parsed SPEC (None when it cannot be read) and the checks."""

    rel_path: str
    sha256: str
    spec: dict[str, Any] | None
    checks: tuple[CheckItem, ...]
    passed: bool


# ------------------------------------------------------------------------------------ upload ----
def _upload_bytes(source: bytes | str) -> bytes:
    data = source.encode("utf-8") if isinstance(source, str) else bytes(source)
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadRefused(f"the file is {len(data)} bytes; the limit is {MAX_UPLOAD_BYTES}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UploadRefused(f"the file is not UTF-8 text: {exc.reason} at byte {exc.start}") from exc
    if not text.strip():
        raise UploadRefused("the file is empty")
    return data


def save_upload(source: bytes | str, repo_root: Path) -> Upload:
    """Write source to <repo_root>/strategies/uploads/<sha256>.py (sha256 of the exact bytes; a str is
    encoded as UTF-8, so pasted text and a dropped file with the same bytes give one file). Idempotent: an
    existing file of that name is left as it is. Raises UploadRefused for blank, non UTF-8 or oversized
    input (MAX_UPLOAD_BYTES)."""
    data = _upload_bytes(source)
    sha = hashlib.sha256(data).hexdigest()
    rel = f"{UPLOADS_REL}/{sha}.py"
    target = repo_root / rel
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f".{sha}.{os.getpid()}.part")
        partial.write_bytes(data)
        partial.replace(target)
    return Upload(rel_path=rel, sha256=sha)


def read_spec(source: bytes | str) -> dict[str, Any] | None:
    """The module-level SPEC dict literal, read without running the file (ast.literal_eval); None when the
    text does not parse or holds no literal SPEC."""
    text = source.decode("utf-8", errors="replace") if isinstance(source, bytes) else source
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == SPEC_NAME for t in node.targets
        ):
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                return None
            return value if isinstance(value, dict) else None
    return None


# ----------------------------------------------------------------------------------- the CLI ----
def lock_path_from(environ: Mapping[str, str]) -> Path:
    """The machine lock: $PFB_DASH_LOCK, else a local scratch folder"""
    given = environ.get(LOCK_ENV)
    return Path(given) if given else DEFAULT_LOCK_PATH


@contextlib.contextmanager
def machine_lock(lock_path: Path) -> Iterator[bool]:
    """A non-blocking exclusive flock on lock_path: yields True when held, False when another holder has it
    (never waits, never queues)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def config_env(config: DashConfig) -> dict[str, str]:
    """The PFB_DASH_* variables that make DashConfig.from_env give back config."""
    env = {
        ENV_REPO: str(config.repo_root),
        ENV_LEDGER: str(config.ledger_path),
        ENV_RUNS: str(config.runs_dir),
    }
    if config.pe_root is not None:
        env[ENV_PE] = str(config.pe_root)
    return env


def _json_line(stdout: str) -> dict[str, Any] | None:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        value = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _kill_group(proc: subprocess.Popen[str]) -> tuple[str, str]:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    try:
        return proc.communicate(timeout=KILL_GRACE_S)
    except subprocess.TimeoutExpired:
        return "", ""


def run_capped(
    argv: Sequence[str], *, env: Mapping[str, str], cwd: Path, timeout_s: float, mem_bytes: int
) -> CliResult:
    """argv in a new session (its own process group) with RLIMIT_AS = mem_bytes set in the child before exec
    and a timeout after which the whole group is killed (SIGKILL). No lock: run_cli holds it."""
    limits = (mem_bytes, mem_bytes)
    preexec = functools.partial(resource.setrlimit, resource.RLIMIT_AS, limits)
    proc = subprocess.Popen(  # argv is ours: sys.executable and the pfb arguments
        list(argv), cwd=cwd, env=dict(env), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, start_new_session=True, preexec_fn=preexec,
    )  # fmt: skip
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        stdout, stderr = _kill_group(proc)
        note = f"\ntimeout: the run passed {timeout_s:g} s and its process group was killed"
        return CliResult(proc.returncode, stdout, stderr + note, None, STATE_TIMEOUT)
    return CliResult(proc.returncode, stdout, stderr, _json_line(stdout))


def cli_argv(args: Sequence[str]) -> list[str]:
    """The child's command: this interpreter, safe-path mode (-P), the bootstrap, then the pfb arguments."""
    return [sys.executable, "-P", "-c", CHILD_BOOTSTRAP, *args]


def _child_env(config: DashConfig, env: Mapping[str, str] | None) -> dict[str, str]:
    return {**os.environ, **CHILD_DEFAULT_ENV, **config_env(config), **(env or {})}


def _busy() -> CliResult:
    return CliResult(None, "", BUSY_TEXT, None, STATE_BUSY)


def run_cli(
    args: Sequence[str],
    *,
    config: DashConfig,
    env: Mapping[str, str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    mem_bytes: int = DEFAULT_MEM_BYTES,
    lock_path: Path | None = None,
) -> CliResult:
    """Run `pfb <args>` in a child process against the dashboard's ledger, runs and repository; the one way
    the dashboard runs anything.

    Contract:
    - args: any pfb arguments as given on a command line, e.g. ["check", path, "--json"], ["run", path,
      "--kind", "synthetic", "--json"], ["hypothesis", "register", path], ["run", path, "--kind", "trial",
      "--hypothesis", file, "--json"], ["prop", run_id, ..., "--runs-dir", d, "--ledger", f, "--json"].
      Paths should be absolute (the child's working directory is config.repo_root).
    - config: where the child reads and writes (see the module docstring): the PFB_DASH_* variables are set
      from it; with this repository's defaults the child is the plain CLI.
    - env: extra variables, applied last (over os.environ, CHILD_DEFAULT_ENV and config's variables).
    - timeout_s: wall-clock limit; then the child's process group is killed and state is "timeout".
    - mem_bytes: RLIMIT_AS of the child (address space, default 4 GiB); an allocation past it fails in the
      child (MemoryError or a refusal of the library), it does not touch the dashboard.
    - lock_path: the machine lock (default lock_path_from(os.environ)); held for the child's whole life. When
      another holder has it the result is state "busy" at once (returncode None): never a queue.
    Returns CliResult; json is the last stdout line parsed as an object (the --json contract), else None.
    """
    with machine_lock(lock_path or lock_path_from(os.environ)) as held:
        if not held:
            return _busy()
        return _run_unlocked(args, config=config, env=env, timeout_s=timeout_s, mem_bytes=mem_bytes)


def _run_unlocked(
    args: Sequence[str],
    *,
    config: DashConfig,
    env: Mapping[str, str] | None,
    timeout_s: float,
    mem_bytes: int,
) -> CliResult:
    return run_capped(cli_argv(args), env=_child_env(config, env), cwd=config.repo_root, timeout_s=timeout_s,
                      mem_bytes=mem_bytes)  # fmt: skip


# ------------------------------------------------------------------------------------- check ----
def _paths_missing(text: str) -> bool:
    return all(mark in text for mark in PATHS_MISSING_MARKS)


def _failure_text(result: CliResult) -> str:
    if result.state == STATE_BUSY:
        return BUSY_TEXT
    out = result.json or {}
    error = out.get("error")
    if isinstance(error, Mapping):
        if _paths_missing(f"{error.get('code')}: {error.get('reason')}"):
            return NO_PATHS_TEXT
        return f"{out.get('status', 'refused')}: {error.get('code')}: {error.get('reason')}"
    if _paths_missing(result.stderr or result.stdout):
        return NO_PATHS_TEXT
    lines = [line.strip() for line in (result.stderr or result.stdout).splitlines() if line.strip()]
    tail = " | ".join(lines[-FAILURE_TAIL_LINES:]) if lines else "no output"
    return f"exit code {result.returncode}: {tail}"


def _warnings_text(warnings: Any) -> str:
    items = [str(w) for w in warnings or ()]
    return "; ".join(items) if items else "no warnings"


def lint_item(result: CliResult) -> CheckItem:
    """From `pfb check --json`: ok with the warnings, or the refusal reason."""
    out = result.json
    if result.returncode == 0 and out is not None and out.get("ok") is True:
        return CheckItem(CHECK_LINT, True, _warnings_text(out.get("warnings")))
    return CheckItem(CHECK_LINT, False, _failure_text(result))


def synthetic_item(result: CliResult) -> CheckItem:
    """From `pfb run --kind synthetic --json`: status, run id, N and its change, and the warnings."""
    out = result.json
    if result.returncode != 0 or out is None or out.get("status") in {"refused", "error"}:
        return CheckItem(CHECK_SYNTHETIC, False, _failure_text(result))
    text = (
        f"status {out.get('status')}, run {out.get('run_id')}, N {out.get('n')} (+{out.get('n_added')}); "
        f"{_warnings_text(out.get('warnings'))}"
    )
    return CheckItem(CHECK_SYNTHETIC, True, text)


def check(
    upload: Upload,
    *,
    config: DashConfig,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    mem_bytes: int = DEFAULT_MEM_BYTES,
    lock_path: Path | None = None,
) -> tuple[CheckItem, ...]:
    """Step 2 (N unchanged): "lint and look-ahead" from `pfb check --json`, then "synthetic run" from `pfb run
    --kind synthetic --json` (skipped, and failed, when the first fails). One lock holds both children; when
    it is busy both items say so."""
    path = str(config.repo_root / upload.rel_path)
    with machine_lock(lock_path or lock_path_from(os.environ)) as held:
        if not held:
            return (CheckItem(CHECK_LINT, False, BUSY_TEXT), CheckItem(CHECK_SYNTHETIC, False, BUSY_TEXT))
        common: dict[str, Any] = {
            "config": config,
            "env": None,
            "timeout_s": timeout_s,
            "mem_bytes": mem_bytes,
        }
        lint = lint_item(_run_unlocked(["check", path, "--json"], **common))
        if not lint.ok:
            return (lint, CheckItem(CHECK_SYNTHETIC, False, "not run: the lint and look-ahead check failed"))
        run = _run_unlocked(["run", path, "--kind", "synthetic", "--json"], **common)
        return (lint, synthetic_item(run))


def check_strategy(
    upload: Upload, source: bytes | str, *, config: DashConfig, **options: Any
) -> StrategyState:
    """save_upload's result checked: the SPEC read from source and the two check items."""
    checks = check(upload, config=config, **options)
    return StrategyState(
        rel_path=upload.rel_path,
        sha256=upload.sha256,
        spec=read_spec(source),
        checks=checks,
        passed=all(item.ok for item in checks),
    )


# ------------------------------------------------------------------------------- step 4: run ----
# item 4 (pfb.dash.view_run). The hypothesis built by pfb.dash.hypgen is saved here, checked by
# `pfb hypothesis register <file> --dry-run` (N unchanged), and, after the user commits both files and ticks
# the confirm box, registered and run as one trial per picked window. `pfb hypothesis register` and `pfb run
# --kind trial` always use this repository's ledger (registry_cmd calls runner default_deps directly and a
# real-data run refuses any other ledger), whatever DashConfig says. --dry-run prints its
# pfb.registration_check/1 object always; register runs with --json and prints the appended
# record; a refusal of either is "refused: <code>: <reason>" on stderr with exit 2. The trial computes its
# [secondary] prop plans itself (docs/TRIALS.md section 6), so no `pfb prop` follows it.
TRIAL_TIMEOUT_S = 4 * 3600.0
# A trial counts at begin (N + 1) even when it dies afterwards, so its cap is set wider than a check's: 8 GiB
# (a synthetic run lived under 8 GiB even without MALLOC_ARENA_MAX). Not measured on a real-data trial.
TRIAL_MEM_BYTES = 8 * GIB
STEP_LOCK, STEP_COMMITTED, STEP_REGISTER, STEP_TRIAL = "lock", "committed at HEAD", "register", "trial"
DRY_RUN_SCHEMA = "pfb.registration_check/1"
DIRTY_SHOWN = 3


class HypothesisRefused(ValueError):
    """The hypothesis text cannot be saved (a bad id, or another file already holds that name)."""


@dataclass(frozen=True)
class SavedHypothesis:
    rel_path: str  # hypotheses/uploads/<id>.toml, relative to the dashboard's repository
    sha256: str


def save_hypothesis(text: str, hyp_id: str, repo_root: Path) -> SavedHypothesis:
    """Write text to <repo_root>/hypotheses/uploads/<hyp_id>.toml. The same bytes again are a no-op; other
    bytes under an existing name are refused (a hypothesis file is never overwritten)."""
    if not ID_RE.fullmatch(hyp_id):
        raise HypothesisRefused(f"hypothesis id {hyp_id!r} must match {ID_RE.pattern}")
    data = text.encode("utf-8")
    rel = hypothesis_rel(hyp_id)
    target = repo_root / rel
    if target.is_file():
        if target.read_bytes() != data:
            raise HypothesisRefused(f"{rel} exists with other content; it is never overwritten")
        return SavedHypothesis(rel, hashlib.sha256(data).hexdigest())
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{hyp_id}.{os.getpid()}.part")
    partial.write_bytes(data)
    partial.replace(target)
    return SavedHypothesis(rel, hashlib.sha256(data).hexdigest())


@dataclass(frozen=True)
class CommitState:
    """Whether each file is at HEAD with its current bytes, and the paths other than the ledger that differ
    from HEAD (a trial needs a clean tree; the runner checkpoints the ledger itself). error: git failed."""

    committed: tuple[tuple[str, bool], ...]
    dirty: tuple[str, ...]
    error: str = ""

    @property
    def ready(self) -> bool:
        return not self.error and not self.dirty and all(ok for _, ok in self.committed)

    def text(self) -> str:
        if self.error:
            return f"git: {self.error}"
        parts = [f"{rel}: {'committed' if ok else 'not committed'}" for rel, ok in self.committed]
        if self.dirty:
            shown = ", ".join(self.dirty[:DIRTY_SHOWN])
            parts.append(f"tree not clean ({len(self.dirty)} path(s) differ from HEAD: {shown})")
        return "; ".join(parts)


def commit_state(repo_root: Path, rel_paths: Sequence[str]) -> CommitState:
    """Read-only git (pfb.registry.gitstate, GIT_OPTIONAL_LOCKS=0): each file's bytes at HEAD against the
    working copy, and `git status` for the rest of the tree."""
    try:
        committed = []
        for rel in rel_paths:
            path = repo_root / rel
            current = path.read_bytes() if path.is_file() else None
            committed.append((rel, current is not None and head_blob(repo_root, rel) == current))
        dirty = tuple(p for p in changed_paths(repo_root) if p != LEDGER_REL)
    except (GitError, OSError) as exc:
        return CommitState(tuple((rel, False) for rel in rel_paths), (), f"{type(exc).__name__}: {exc}")
    return CommitState(tuple(committed), dirty)


def commit_command(rel_paths: Sequence[str], message: str) -> str:
    """The git command commit_files runs (the user may also run it by hand)."""
    paths = " ".join(shlex.quote(rel) for rel in rel_paths)
    return f"git add -- {paths} && git commit --only -m {shlex.quote(message)} -- {paths}"


# the "Commit these two files" button. The dashboard's one git write: the upload
# and its hypothesis, nothing else, after the user's confirm box. The message shown is the message committed:
# commit_message adds the lines of $PFB_COMMIT_TRAILER, as `pfb registry checkpoint` does.
COMMIT_DIRS = (f"{UPLOADS_REL}/", "hypotheses/")


@dataclass(frozen=True)
class CommitDone:
    """ok: the new HEAD in commit; else message holds every refusal reason and nothing was committed."""

    ok: bool
    message: str
    commit: str | None = None


def commit_message(message: str) -> str:
    return with_trailer(message)


def _outside(repo_root: Path, rel: str, prefix: str) -> str | None:
    """Why rel cannot be committed by the button (not a file under repo_root/prefix), else None."""
    pure = PurePosixPath(rel)
    if pure.is_absolute() or ".." in pure.parts or not rel.startswith(prefix):
        return f"{rel} is outside {prefix}"
    target, allowed = (repo_root / rel).resolve(), (repo_root / prefix).resolve()
    if not target.is_relative_to(allowed):
        return f"{rel} resolves outside {prefix}"
    if not target.is_file():
        return f"{rel} is not a file"
    return None


def commit_files(repo_root: Path, upload_rel: str, hyp_rel: str, message: str) -> CommitDone:
    """`git commit --only <upload> <hypothesis> -m <message>`. Refused, with nothing staged or committed, when
    the upload is outside strategies/uploads/, the hypothesis outside hypotheses/, or any other path is
    staged."""
    rels = (upload_rel, hyp_rel)
    checked = (_outside(repo_root, rel, prefix) for rel, prefix in zip(rels, COMMIT_DIRS, strict=True))
    reasons = [reason for reason in checked if reason]
    try:
        if not reasons:
            others = [p for p in staged_paths(repo_root) if p not in rels]
            if others:
                shown = ", ".join(others[:DIRTY_SHOWN])
                reasons.append(f"{len(others)} other path(s) are staged ({shown}); unstage them first")
        if reasons:
            return CommitDone(False, "; ".join(reasons))
        commit = commit_paths(repo_root, rels, message)
    except (GitError, OSError) as exc:
        return CommitDone(False, f"git: {type(exc).__name__}: {exc}")
    return CommitDone(True, f"committed {commit[:12]}", commit)


@dataclass(frozen=True)
class Refusal:
    code: str
    reason: str


@dataclass(frozen=True)
class DryRun:
    """`pfb hypothesis register <file> --dry-run`: ok, every refusal, or (no JSON object) message."""

    ok: bool
    refusals: tuple[Refusal, ...]
    message: str


def register_args(hyp_path: Path, *, dry_run: bool = False) -> list[str]:
    return ["hypothesis", "register", str(hyp_path), "--dry-run" if dry_run else "--json"]


def trial_args(strategy_path: Path, hyp_path: Path, window_index: int) -> list[str]:
    """One trial: the hypothesis's [[window]] number window_index (1-based), SPEC defaults."""
    return ["run", str(strategy_path), "--kind", "trial", "--hypothesis", str(hyp_path), "--window",
            str(window_index), "--json"]  # fmt: skip


def dry_run_of(result: CliResult) -> DryRun:
    out = result.json
    if out is None or out.get("schema") != DRY_RUN_SCHEMA:
        return DryRun(False, (), _failure_text(result))
    items = out.get("refusals") or []
    refusals = tuple(
        Refusal(str(r.get("code")), str(r.get("reason"))) for r in items if isinstance(r, Mapping)
    )
    ok = out.get("ok") is True and not refusals and result.returncode == 0
    return DryRun(ok, refusals, "would register" if ok else f"{len(refusals)} refusal(s)")


def dry_run(
    hyp_path: Path,
    *,
    config: DashConfig,
    env: Mapping[str, str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    mem_bytes: int = DEFAULT_MEM_BYTES,
    lock_path: Path | None = None,
) -> DryRun:
    """Every registration check, nothing appended (N unchanged), through run_cli (busy: one message)."""
    result = run_cli(register_args(hyp_path, dry_run=True), config=config, env=env, timeout_s=timeout_s,
                     mem_bytes=mem_bytes, lock_path=lock_path)  # fmt: skip
    return dry_run_of(result)


@dataclass(frozen=True)
class StepResult:
    """One step of the run: its name, ok, the message shown (a refusal's exact text) and a trial's run_id."""

    name: str
    ok: bool
    message: str
    run_id: str | None = None


def register_text(record: Mapping[str, Any]) -> str:
    """The line `pfb hypothesis register` prints without --json, from its --json record."""
    raw = record.get("power")
    power: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    return (
        f"registered {record.get('hypothesis_id')}: N {record.get('n_after')}, "
        f"budget {record.get('budget')}, min_edge_cents {power.get('min_edge_cents')} "
        f"at N {power.get('n_used')}"
    )


def register_step(result: CliResult) -> StepResult:
    """`pfb hypothesis register <file> --json`: ok only with exit 0 and a record holding hypothesis_id."""
    out = result.json
    if result.returncode == 0 and result.state == STATE_DONE and out and "hypothesis_id" in out:
        return StepResult(STEP_REGISTER, True, register_text(out))
    return StepResult(STEP_REGISTER, False, _failure_text(result))


def trial_step(result: CliResult, window_index: int) -> StepResult:
    out = result.json or {}
    run_id = out.get("run_id") if isinstance(out.get("run_id"), str) else None
    name = f"{STEP_TRIAL}, window {window_index}"
    if result.returncode != 0 or not out or out.get("status") in {"refused", "error"}:
        return StepResult(name, False, _failure_text(result), run_id)
    text = f"status {out.get('status')}, run {run_id}, N {out.get('n')} (+{out.get('n_added')})"
    return StepResult(name, True, text, run_id)


def run_trials(
    strategy_rel: str,
    hyp_rel: str,
    window_indexes: Sequence[int],
    *,
    config: DashConfig,
    registered: bool,
    on_step: Callable[[str], None] | None = None,
    env: Mapping[str, str] | None = None,
    timeout_s: float = TRIAL_TIMEOUT_S,
    mem_bytes: int = TRIAL_MEM_BYTES,
    lock_path: Path | None = None,
) -> tuple[StepResult, ...]:
    """Step 4's confirm: under one machine lock, (1) both files committed at HEAD and the tree clean, else
    nothing is called; (2) `pfb hypothesis register <file>` unless registered; (3) `pfb run <strategy> --kind
    trial --hypothesis <file> --window k --json` for each k, stopping at the first failure. Each trial is
    N + 1. on_step(name) is called before each child starts (progress)."""
    notify = on_step or (lambda _name: None)
    with machine_lock(lock_path or lock_path_from(os.environ)) as held:
        if not held:
            return (StepResult(STEP_LOCK, False, BUSY_TEXT),)
        notify(STEP_COMMITTED)
        state = commit_state(config.repo_root, (strategy_rel, hyp_rel))
        if not state.ready:
            return (StepResult(STEP_COMMITTED, False, state.text()),)
        steps = [StepResult(STEP_COMMITTED, True, state.text())]
        strategy_path, hyp_path = config.repo_root / strategy_rel, config.repo_root / hyp_rel
        if not registered:
            notify(STEP_REGISTER)
            done = _run_unlocked(register_args(hyp_path), config=config, env=env, timeout_s=DEFAULT_TIMEOUT_S,
                                 mem_bytes=DEFAULT_MEM_BYTES)  # fmt: skip
            step = register_step(done)
            steps.append(step)
            if not step.ok:
                return tuple(steps)
        for index in window_indexes:
            notify(f"{STEP_TRIAL}, window {index}")
            result = _run_unlocked(trial_args(strategy_path, hyp_path, index), config=config, env=env,
                                   timeout_s=timeout_s, mem_bytes=mem_bytes)  # fmt: skip
            steps.append(trial_step(result, index))
            if not steps[-1].ok:
                break
        return tuple(steps)
