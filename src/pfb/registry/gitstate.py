"""The git facts the registry records or refuses on (ADR A10, A14 a). Read-only git commands, in the
registry's own repository, with GIT_OPTIONAL_LOCKS=0 (no index refresh, so not even git status writes).

The writing commands are commit_only (docs/TRIALS.md section 4, `pfb registry checkpoint`) and commit_paths
(the dashboard's commit button): they stage and commit the named paths and nothing else
(`git commit --only`), so other staged or changed files stay as they were. with_trailer appends the lines of
$PFB_COMMIT_TRAILER to a commit message."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pfb.registry.errors import GitError, InputNotCommitted

_GIT_TIMEOUT_S = 60
TRAILER_ENV = "PFB_COMMIT_TRAILER"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            check=False,
            timeout=_GIT_TIMEOUT_S,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitError(f"git {args[0]} could not run: {exc}") from exc


def head_commit(repo: Path) -> str | None:
    """HEAD's commit, or None when repo is not a git work tree or has no commit."""
    done = _git(repo, "rev-parse", "--verify", "--quiet", "HEAD")
    return done.stdout.decode().strip() if done.returncode == 0 else None


def changed_paths(repo: Path) -> tuple[str, ...]:
    """Every path `git status --porcelain --untracked-files=all` reports (both sides of a rename)."""
    done = _git(repo, "status", "--porcelain", "-z", "--untracked-files=all")
    if done.returncode != 0:
        raise GitError(f"git status failed in {repo}: {done.stderr.decode().strip()[:200]}")
    tokens = [t for t in done.stdout.decode("utf-8", "surrogateescape").split("\0") if t]
    paths: list[str] = []
    i = 0
    while i < len(tokens):
        status, path = tokens[i][:2], tokens[i][3:]
        paths.append(path)
        if "R" in status or "C" in status:
            i += 1
            paths.append(tokens[i])
        i += 1
    return tuple(paths)


def is_dirty(repo: Path, exempt: str) -> bool | None:
    """True when any path other than `exempt` (ledger/registry.jsonl) is changed; None outside a git repo."""
    if head_commit(repo) is None and _git(repo, "rev-parse", "--is-inside-work-tree").returncode != 0:
        return None
    return any(path != exempt for path in changed_paths(repo))


def repo_relative(repo: Path, path: Path) -> str:
    """POSIX path of `path` relative to the repository; refused when it lies outside."""
    resolved = (repo / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        return resolved.relative_to(repo.resolve()).as_posix()
    except ValueError as exc:
        raise InputNotCommitted(f"{path} lies outside the repository, so it cannot be committed") from exc


def require_committed(repo: Path, rel_path: str) -> None:
    """Tracked and unchanged against HEAD (ADR A14 a), else InputNotCommitted."""
    if _git(repo, "ls-files", "--error-unmatch", "--", rel_path).returncode != 0:
        raise InputNotCommitted(f"{rel_path} is not tracked by git")
    if _git(repo, "diff", "--quiet", "HEAD", "--", rel_path).returncode != 0:
        raise InputNotCommitted(f"{rel_path} differs from HEAD (commit it first)")


def head_blob(repo: Path, rel_path: str) -> bytes | None:
    """The bytes of `rel_path` (relative to `repo`) at HEAD; None when there is no HEAD or HEAD lacks it."""
    if head_commit(repo) is None:
        return None
    spec = f"HEAD:./{rel_path}"
    if _git(repo, "cat-file", "-e", spec).returncode != 0:
        return None
    done = _git(repo, "cat-file", "blob", spec)
    if done.returncode != 0:
        raise GitError(f"git cat-file blob {spec} failed: {done.stderr.decode().strip()[:200]}")
    return done.stdout


def require_at_head(repo: Path, rel_path: str) -> None:
    """`git cat-file -e HEAD:<path>` succeeds, else InputNotCommitted."""
    if _git(repo, "cat-file", "-e", f"HEAD:{rel_path}").returncode != 0:
        raise InputNotCommitted(f"{rel_path} does not exist at HEAD")


def with_trailer(message: str) -> str:
    """message, then a blank line and the lines of $PFB_COMMIT_TRAILER when it is set."""
    trailer = os.environ.get(TRAILER_ENV, "").strip("\n")
    return f"{message}\n\n{trailer}" if trailer else message


def staged_paths(repo: Path) -> tuple[str, ...]:
    """The paths staged against HEAD (`git diff --cached --name-only`; every path in a repo without HEAD)."""
    args = ("diff", "--cached", "--name-only", "-z") if head_commit(repo) else ("ls-files", "-z")
    done = _git(repo, *args)
    if done.returncode != 0:
        raise GitError(f"git {args[0]} failed in {repo}: {done.stderr.decode().strip()[:200]}")
    return tuple(t for t in done.stdout.decode("utf-8", "surrogateescape").split("\0") if t)


def commit_paths(repo: Path, rel_paths: tuple[str, ...], message: str) -> str:
    """Stage and commit `rel_paths` alone (other changes, staged or not, stay out); return the new HEAD.
    GitError when git refuses or HEAD's blob of a path then differs from its working file."""
    added = _git(repo, "add", "--", *rel_paths)
    if added.returncode != 0:
        raise GitError(f"git add {' '.join(rel_paths)} failed: {added.stderr.decode().strip()[:200]}")
    done = _git(repo, "commit", "--only", "--quiet", "-m", message, "--", *rel_paths)
    if done.returncode != 0:
        raise GitError(f"git commit of {' '.join(rel_paths)} failed: {done.stderr.decode().strip()[:300]}")
    for rel_path in rel_paths:
        if head_blob(repo, rel_path) != (repo / rel_path).read_bytes():
            raise GitError(f"after the commit, {rel_path} at HEAD differs from the working file")
    commit = head_commit(repo)
    if commit is None:
        raise GitError("no HEAD after the commit")
    return commit


def commit_only(repo: Path, rel_path: str, message: str) -> str:
    """Stage and commit `rel_path` alone (other changes, staged or not, stay out); return the new HEAD.
    GitError when git refuses or HEAD's blob of the path then differs from the working file."""
    return commit_paths(repo, (rel_path,), message)
