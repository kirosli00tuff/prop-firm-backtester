"""The code identity of a run: one sha256 over the working bytes of every *.py under
src/pfb.

Canonical form, defined here only: the *.py files under the code root (recursively), each named by its
POSIX path relative to the code root's PARENT (so "pfb/registry/api.py"), sorted by that name as a str;
for each, in that order, the name in UTF-8, a NUL byte, the byte length in ASCII decimal, a NUL byte, then
the bytes as read from disk now. code_sha256 is the sha256 of the concatenation. Reading from disk (not
from git) means an edit hidden from `git status` (assume-unchanged, skip-worktree) still changes the
identity.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pfb.registry.chain import canonical_bytes, sha256_hex
from pfb.registry.errors import BadRequest

PFB_SRC = Path(__file__).resolve().parents[1]
CODE_SUFFIX = ".py"
SEPARATOR = b"\0"


def code_files(code_root: Path) -> tuple[tuple[str, Path], ...]:
    """(name, path) of every *.py file under code_root, sorted by name; refused when there is none."""
    root = code_root.resolve()
    named = sorted(
        (path.relative_to(root.parent).as_posix(), path)
        for path in root.rglob(f"*{CODE_SUFFIX}")
        if path.is_file()
    )
    if not named:
        raise BadRequest(f"code root {code_root.name!r} holds no {CODE_SUFFIX} file")
    return tuple(named)


def code_canonical_bytes(code_root: Path) -> bytes:
    """The canonical byte string of the module docstring (name, NUL, length, NUL, bytes per file)."""
    parts: list[bytes] = []
    for name, path in code_files(code_root):
        data = path.read_bytes()
        parts.append(name.encode("utf-8") + SEPARATOR + str(len(data)).encode("ascii") + SEPARATOR + data)
    return b"".join(parts)


def code_sha256(code_root: Path = PFB_SRC) -> str:
    """sha256 of code_canonical_bytes(code_root): the identity of the code a run executes."""
    return hashlib.sha256(code_canonical_bytes(code_root)).hexdigest()


CALENDARS_DIR = PFB_SRC.parents[1] / "config" / "calendars"
CALENDAR_SUFFIX = ".json"


def calendars_sha256(directory: Path = CALENDARS_DIR) -> str:
    """sha256 of the canonical list of [file name, sha256] over config/calendars/*.json, sorted by name
    (docs/TRIALS.md section 3; the calendars decide flat-by and session ends, so they enter config_id)."""
    files = sorted((p for p in directory.glob(f"*{CALENDAR_SUFFIX}") if p.is_file()), key=lambda p: p.name)
    if not files:
        raise BadRequest(f"no {CALENDAR_SUFFIX} calendar file in {directory.name}")
    return sha256_hex(canonical_bytes([[p.name, sha256_hex(p.read_bytes())] for p in files]))
