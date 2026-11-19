"""Backtest step 4's "Commit these two files" button, pfb.dash.actions
.commit_files. Every git command runs in a temporary repository, never in this one."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from pfb.dash.actions import commit_command, commit_files, commit_message

UPLOAD_REL = "strategies/uploads/abc.py"
HYP_REL = "hypotheses/uploads/h_abc.toml"
MESSAGE = "hypothesis h_abc: strategy upload abc from the dashboard"
TRAILER = "Trailer-One: a\nTrailer-Two: b"


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True, text=True)
    return done.stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A temporary repository with one commit, an uncommitted upload and hypothesis, and a tracked file."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "test")
    (root / "README").write_text("x\n")
    _git(root, "add", "README")
    _git(root, "commit", "-q", "-m", "init")
    for rel, text in ((UPLOAD_REL, "x = 1\n"), (HYP_REL, "id = 'h_abc'\n")):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def _head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD").strip()


def test_commit_files_commits_the_two_files_alone_with_the_trailer(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: the program's trailer lines, and an unstaged edit that must stay out
    monkeypatch.setenv("PFB_COMMIT_TRAILER", TRAILER)
    (repo / "README").write_text("edited\n")
    message = commit_message(MESSAGE)
    # Act
    done = commit_files(repo, UPLOAD_REL, HYP_REL, message)
    # Assert
    assert done.ok and done.commit == _head(repo), done.message
    files = sorted(_git(repo, "show", "--name-only", "--format=", "HEAD").split())
    assert files == sorted([UPLOAD_REL, HYP_REL])
    assert _git(repo, "log", "-1", "--format=%B").rstrip("\n") == f"{MESSAGE}\n\n{TRAILER}"
    assert _git(repo, "status", "--porcelain").splitlines() == [" M README"]


def test_without_the_trailer_variable_the_message_is_the_shown_line(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PFB_COMMIT_TRAILER", raising=False)
    assert commit_message(MESSAGE) == MESSAGE
    assert commit_files(repo, UPLOAD_REL, HYP_REL, MESSAGE).ok
    assert _git(repo, "log", "-1", "--format=%B").rstrip("\n") == MESSAGE


def test_the_two_files_already_staged_are_committed(repo: Path) -> None:
    _git(repo, "add", UPLOAD_REL, HYP_REL)
    done = commit_files(repo, UPLOAD_REL, HYP_REL, MESSAGE)
    assert done.ok, done.message
    assert _git(repo, "status", "--porcelain") == ""


def test_anything_else_staged_refuses_and_commits_nothing(repo: Path) -> None:
    # Arrange
    (repo / "other.txt").write_text("o\n")
    _git(repo, "add", "other.txt")
    head = _head(repo)
    # Act
    done = commit_files(repo, UPLOAD_REL, HYP_REL, MESSAGE)
    # Assert: refused with the reason; HEAD, the index and the two files untouched
    assert not done.ok and done.commit is None
    assert "1 other path(s) are staged (other.txt)" in done.message
    assert _head(repo) == head
    status = _git(repo, "status", "--porcelain", "--untracked-files=all").splitlines()
    assert sorted(status) == sorted(["A  other.txt", f"?? {UPLOAD_REL}", f"?? {HYP_REL}"])


@pytest.mark.parametrize(
    ("upload", "hyp", "reason"),
    [
        ("strategies/abc.py", HYP_REL, "strategies/abc.py is outside strategies/uploads/"),
        (UPLOAD_REL, "config/h.toml", "config/h.toml is outside hypotheses/"),
        ("strategies/uploads/../x.py", HYP_REL, "is outside strategies/uploads/"),
        ("/etc/passwd", HYP_REL, "is outside strategies/uploads/"),
        (UPLOAD_REL, "hypotheses/missing.toml", "hypotheses/missing.toml is not a file"),
    ],
)
def test_a_path_outside_the_two_folders_refuses(repo: Path, upload: str, hyp: str, reason: str) -> None:
    (repo / "strategies" / "abc.py").write_text("y = 2\n")
    (repo / "config").mkdir()
    (repo / "config" / "h.toml").write_text("z\n")
    head = _head(repo)
    done = commit_files(repo, upload, hyp, MESSAGE)
    assert not done.ok and reason in done.message
    assert _head(repo) == head
    assert _git(repo, "diff", "--cached", "--name-only") == ""


def test_a_symlink_out_of_the_folder_refuses(repo: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("v = 1\n")
    (repo / "strategies" / "uploads" / "link.py").symlink_to(outside)
    head = _head(repo)
    done = commit_files(repo, "strategies/uploads/link.py", HYP_REL, MESSAGE)
    assert not done.ok and "resolves outside strategies/uploads/" in done.message
    assert _head(repo) == head


def test_the_shown_command_makes_the_same_commit_as_the_button(repo: Path) -> None:
    # Arrange: the command line the page shows, run by hand
    line = commit_command([UPLOAD_REL, HYP_REL], MESSAGE)
    # Act
    subprocess.run(["bash", "-c", line], cwd=repo, check=True, capture_output=True)
    # Assert
    files = sorted(_git(repo, "show", "--name-only", "--format=", "HEAD").split())
    assert files == sorted([UPLOAD_REL, HYP_REL])
    assert _git(repo, "log", "-1", "--format=%s") == MESSAGE + "\n"
