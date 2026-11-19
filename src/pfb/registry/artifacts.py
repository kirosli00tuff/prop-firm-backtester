"""A run folder checked against its result record's artifact_sha256.

The runner serializes every artifact before the result record, records each name with the sha256 of its bytes
(records.ARTIFACT_SHA_KEY, result record tier 3) and then writes exactly those bytes. An artifact name is
"<base>/<run_id>/<file>" (base "runs" inside the repository, "external:<runs dir name>" outside it), so the
file is <run folder>/<file>. A record written before the key existed (tier 1 or 2) carries no hashes: its
folder cannot be checked and is reported as unchecked, never as matching.

- folder_problems: the mismatches of one folder (missing file, other sha256, a name outside the folder).
- require_folder_matches: ArtifactMismatch on the first problem; used where a folder is an input to
  anything that counts (pfb prop's source runs).
- verify_folders: `pfb registry verify`, every finished run whose folder is present under runs_dir.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pfb.registry.errors import ArtifactMismatch
from pfb.registry.records import ARTIFACT_SHA_KEY, LedgerState

CHUNK = 1 << 20


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def recorded_shas(result: Mapping[str, Any] | None) -> Mapping[str, str] | None:
    """The result record's artifact_sha256, or None for a record without it (an older record)."""
    if result is None or ARTIFACT_SHA_KEY not in result:
        return None
    shas = result[ARTIFACT_SHA_KEY]
    return shas if isinstance(shas, dict) else None


def file_in_folder(name: str, run_id: str) -> str | None:
    """The file name inside the run folder that an artifact name denotes; None when it is not
    "<base>/<run_id>/<file>" with a plain file name."""
    marker = f"/{run_id}/"
    if marker not in name:
        return None
    rest = name.split(marker, 1)[1]
    if not rest or "/" in rest or rest in (".", ".."):
        return None
    return rest


def folder_problems(run_dir: Path, run_id: str, shas: Mapping[str, str]) -> list[str]:
    """Every artifact of the record that is missing from the folder or hashes otherwise."""
    problems = []
    for name, sha in sorted(shas.items()):
        file = file_in_folder(name, run_id)
        if file is None:
            problems.append(f"{name}: not a file of run folder {run_id}")
        elif not (run_dir / file).is_file():
            problems.append(f"{file}: missing")
        elif file_sha256(run_dir / file) != sha:
            problems.append(f"{file}: sha256 differs from the result record")
    return problems


def require_folder_matches(run_dir: Path, run_id: str, result: Mapping[str, Any] | None) -> bool:
    """ArtifactMismatch when the record carries hashes and the folder does not match them. Returns True when
    the folder was checked, False when the record carries no hashes (an older record: unchecked)."""
    shas = recorded_shas(result)
    if shas is None:
        return False
    problems = folder_problems(run_dir, run_id, shas)
    if problems:
        raise ArtifactMismatch(
            f"run folder {run_id} does not match its result record ({len(problems)} problem(s); first: "
            f"{problems[0]}): it is refused as an input"
        )
    return True


@dataclass(frozen=True)
class FolderReport:
    checked: tuple[str, ...]  # folders present whose files all match
    unchecked: tuple[str, ...]  # folders present whose result record carries no hashes (older records)
    absent: tuple[str, ...]  # finished runs with hashes whose folder is not under runs_dir
    mismatched: Mapping[str, tuple[str, ...]]  # run id -> problems

    @property
    def ok(self) -> bool:
        return not self.mismatched

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checked": len(self.checked),
            "unchecked": len(self.unchecked),
            "absent": len(self.absent),
            "mismatched": {k: list(v) for k, v in self.mismatched.items()},
        }


def verify_folders(state: LedgerState, runs_dir: Path) -> FolderReport:
    """Compare every finished run's folder under runs_dir with its result record, where both exist."""
    checked: list[str] = []
    unchecked: list[str] = []
    absent: list[str] = []
    mismatched: dict[str, tuple[str, ...]] = {}
    for run_id, result in state.results.items():
        shas = recorded_shas(result)
        run_dir = runs_dir / run_id
        if not run_dir.is_dir():
            if shas:
                absent.append(run_id)
            continue
        if shas is None:
            unchecked.append(run_id)
            continue
        problems = folder_problems(run_dir, run_id, shas)
        if problems:
            mismatched[run_id] = tuple(problems)
        else:
            checked.append(run_id)
    return FolderReport(tuple(checked), tuple(unchecked), tuple(absent), mismatched)
