"""An optional upstream ledger, imported byte for byte, and the genesis record (ADR 13).

The program's one N may continue the trial count of an earlier research checkout (config/paths.toml
[upstream] root) whose append-only ledger is two JSONL files: trial_registrations.jsonl (a baseline line,
then lines that each add their test ids to N) and trial_amendments.jsonl (lines that never change N). Its
read checks are ported here so that its N is the last line's n_after only after the same checks. The
upstream files are opened read-only and never written.

Without an upstream the genesis record imports nothing: pe_files is {} and n_pe is 0.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pfb.registry.chain import sha256_hex
from pfb.registry.errors import PeLedgerInvalid

PE_REGISTRATIONS = "ledger/trial_registrations.jsonl"
PE_AMENDMENTS = "ledger/trial_amendments.jsonl"
PE_FILES = (PE_REGISTRATIONS, PE_AMENDMENTS)

PE_SCHEMA = "trial_registration/1"
PE_FIELDS = frozenset(
    {
        "schema",
        "entry_id",
        "test",
        "test_ids",
        "freeze_path",
        "freeze_sha256",
        "harness_sha256",
        "n_before",
        "n_after",
        "time_local",
        "source",
        "note",
    }
)
PE_BASELINE_ENTRY_ID = "baseline"
PE_BASELINE_N = 300


@dataclass(frozen=True)
class PeFile:
    """One upstream ledger file as read now: its repo-relative name, sha256 and lines (no newline)."""

    name: str
    sha256: str
    lines: tuple[str, ...]


def split_lines(data: bytes, name: str) -> tuple[str, ...]:
    """Lines of an upstream ledger file; refused unless joining them back with newlines gives the same
    bytes."""
    if not data:
        return ()
    if not data.endswith(b"\n") or b"\r" in data:
        raise PeLedgerInvalid(
            f"upstream {name}: not newline-terminated LF lines; cannot import it byte for byte"
        )
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PeLedgerInvalid(f"upstream {name} is not UTF-8: {exc}") from exc
    return tuple(text[:-1].split("\n"))


def join_lines(lines: Sequence[str]) -> bytes:
    return "".join(line + "\n" for line in lines).encode("utf-8")


def read_pe_file(pe_root: Path, name: str) -> PeFile:
    path = pe_root / name
    if not path.is_file():
        raise PeLedgerInvalid(f"upstream ledger file {name} not found under the configured upstream root")
    data = path.read_bytes()
    return PeFile(name=name, sha256=sha256_hex(data), lines=split_lines(data, name))


def read_pe_files(pe_root: Path) -> dict[str, PeFile]:
    return {name: read_pe_file(pe_root, name) for name in PE_FILES}


def pe_hashes(pe_root: Path) -> dict[str, str]:
    """sha256 of each upstream ledger file as it is now."""
    return {name: read_pe_file(pe_root, name).sha256 for name in PE_FILES}


def _json_line(line: str, where: str) -> Any:
    if not line.strip():
        raise PeLedgerInvalid(f"{where} is blank")
    try:
        return json.loads(line)
    except ValueError as exc:
        raise PeLedgerInvalid(f"{where} is not JSON: {exc}") from exc


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_pe_line(entry: Any, where: str, previous: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """The upstream ledger's own line check, with test ids also required to be strings."""
    if not isinstance(entry, dict) or set(entry) != PE_FIELDS or entry["schema"] != PE_SCHEMA:
        raise PeLedgerInvalid(f"{where}: not a {PE_SCHEMA} record with upstream's 12 fields")
    if not _is_int(entry["n_after"]):
        raise PeLedgerInvalid(f"{where}: n_after is not an integer")
    ids = entry["test_ids"]
    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
        raise PeLedgerInvalid(f"{where}: test_ids is not a list of strings")
    if previous is None:
        if entry["entry_id"] != PE_BASELINE_ENTRY_ID or entry["n_after"] != PE_BASELINE_N or ids:
            raise PeLedgerInvalid(f"{where}: the first line must be upstream's baseline N = {PE_BASELINE_N}")
        return entry
    if (
        not ids
        or not _is_int(entry["n_before"])
        or entry["n_before"] != previous["n_after"]
        or entry["n_after"] != entry["n_before"] + len(ids)
    ):
        raise PeLedgerInvalid(
            f"{where}: does not chain (n_before = previous n_after, n_after = n_before + ids)"
        )
    return entry


def pe_current_n(lines: Sequence[str]) -> int:
    """Upstream's current_n: the last registration line's n_after after every read_registry check."""
    entries: list[Mapping[str, Any]] = []
    for i, line in enumerate(lines):
        where = f"upstream {PE_REGISTRATIONS} line {i + 1}"
        entries.append(_check_pe_line(_json_line(line, where), where, entries[-1] if entries else None))
    if not entries:
        raise PeLedgerInvalid(f"upstream {PE_REGISTRATIONS} is empty: the baseline line is missing")
    ids = [x for e in entries for x in e["test_ids"]]
    tests = [e["test"] for e in entries[1:]]
    entry_ids = [e["entry_id"] for e in entries]
    for values, what in ((ids, "test id"), (tests, "test"), (entry_ids, "entry id")):
        if len(set(map(json.dumps, values))) != len(values):
            raise PeLedgerInvalid(f"upstream {PE_REGISTRATIONS}: a {what} repeats")
    return int(entries[-1]["n_after"])


def check_pe_amendments(lines: Sequence[str]) -> None:
    """Every amendment line is a JSON object that leaves N unchanged (upstream has no reader for this
    file)."""
    for i, line in enumerate(lines):
        where = f"upstream {PE_AMENDMENTS} line {i + 1}"
        entry = _json_line(line, where)
        if not isinstance(entry, dict):
            raise PeLedgerInvalid(f"{where} is not a JSON object")
        n_before, n_after = entry.get("n_before"), entry.get("n_after")
        if not (_is_int(n_before) and _is_int(n_after)) or n_before != n_after:
            raise PeLedgerInvalid(
                f"{where}: an amendment that changes N cannot be imported (n_before != n_after)"
            )
        if entry.get("configurations_added", 0) != 0:
            raise PeLedgerInvalid(f"{where}: configurations_added is not 0")


def pe_n_of(lines_by_file: Mapping[str, Sequence[str]]) -> int:
    """Check both files and return upstream's N."""
    check_pe_amendments(lines_by_file[PE_AMENDMENTS])
    return pe_current_n(lines_by_file[PE_REGISTRATIONS])


def genesis_body(pe_root: Path | None) -> dict[str, Any]:
    """The genesis-specific fields: the upstream lines unchanged with each file's sha256, and n_pe; without
    an upstream, nothing imported and n_pe 0."""
    if pe_root is None:
        return {"n_pe": 0, "pe_files": {}}
    files = read_pe_files(pe_root)
    n_pe = pe_n_of({name: f.lines for name, f in files.items()})
    pe_files = {name: {"sha256": f.sha256, "lines": list(f.lines)} for name, f in files.items()}
    return {"n_pe": n_pe, "pe_files": pe_files}
