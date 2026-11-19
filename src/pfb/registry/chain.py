"""The hash chain of ledger/registry.jsonl: canonical lines, verification and the one locked append
(ADR 13, A14 b).

One record per line in canonical JSON (sorted keys, separators "," and ":", ASCII, no NaN). Each record's
prev_sha256 is the sha256 of the previous line's bytes without its newline; the first record uses 64 zeros.
The file is only ever appended to: every append holds fcntl.flock(LOCK_EX) on the ledger itself, re-reads and
verifies the whole file inside the lock, builds the record from that state, validates the result, appends,
fsyncs and unlocks.
Anchor: when the caller passes the ledger's bytes at git HEAD, the file read inside the lock must
start with them byte for byte, else LedgerAnchorBroken and nothing is written.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pfb.registry.errors import BadRequest, ChainBroken, GenesisExists, LedgerAnchorBroken, RegistryMissing

ZERO_HASH = "0" * 64
NEWLINE = b"\n"
_FILE_MODE = 0o644
_READ_BLOCK = 1 << 20


def canonical_bytes(value: object) -> bytes:
    """Canonical JSON: sorted keys, no spaces, ASCII only, NaN and infinity refused."""
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise BadRequest(f"value is not plain JSON (a NaN, infinity or non-JSON type): {exc}") from exc
    return text.encode("ascii")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class LedgerLine:
    """One verified line: its index, raw bytes (no newline), their sha256 and the parsed record."""

    index: int
    raw: bytes
    sha256: str
    record: Mapping[str, Any]


def _parse_one(raw: bytes, index: int, name: str) -> dict[str, Any]:
    where = f"{name} line {index + 1}"
    if not raw:
        raise ChainBroken(f"{where} is blank")
    try:
        record = json.loads(raw)
    except ValueError as exc:
        raise ChainBroken(f"{where} is not JSON: {exc}") from exc
    if not isinstance(record, dict):
        raise ChainBroken(f"{where} is not a JSON object")
    try:
        canonical = canonical_bytes(record)
    except BadRequest as exc:
        raise ChainBroken(f"{where}: {exc}") from exc
    if canonical != raw:
        raise ChainBroken(f"{where} is not in canonical form (edited by hand?)")
    return record


def parse_ledger(data: bytes, name: str = "registry.jsonl") -> tuple[LedgerLine, ...]:
    """Every line, with the prev_sha256 chain verified. An empty file has no lines."""
    if not data:
        return ()
    if not data.endswith(NEWLINE):
        raise ChainBroken(f"{name}: the last line has no newline (a torn append?)")
    lines: list[LedgerLine] = []
    previous = ZERO_HASH
    for index, raw in enumerate(data[:-1].split(NEWLINE)):
        record = _parse_one(raw, index, name)
        if record.get("prev_sha256") != previous:
            raise ChainBroken(
                f"{name} line {index + 1}: prev_sha256 does not match the previous line's sha256"
            )
        digest = sha256_hex(raw)
        lines.append(LedgerLine(index=index, raw=raw, sha256=digest, record=record))
        previous = digest
    return tuple(lines)


def require_extends(data: bytes, anchor: bytes | None, name: str) -> None:
    """The ledger's bytes must start with its committed bytes (None: no anchor, nothing to check)."""
    if anchor is not None and not data.startswith(anchor):
        raise LedgerAnchorBroken(
            f"{name} ({len(data)} bytes) does not start with its {len(anchor)} bytes at git HEAD: committed "
            "lines were cut, edited or the file was re-created"
        )


def read_ledger(path: Path, anchor: bytes | None = None) -> tuple[LedgerLine, ...]:
    """Read and chain-verify the ledger (no lock: a reader sees whole appended lines or a torn tail)."""
    if not path.is_file():
        raise RegistryMissing(f"{path} does not exist: write genesis first (`pfb registry genesis`)")
    data = path.read_bytes()
    require_extends(data, anchor, path.name)
    return parse_ledger(data, path.name)


def _read_fd(fd: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, _READ_BLOCK)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def _open_locked(path: Path, *, create: bool) -> int:
    flags = os.O_RDWR | os.O_APPEND | (os.O_CREAT if create else 0)
    try:
        fd = os.open(path, flags, _FILE_MODE)
    except FileNotFoundError as exc:
        raise RegistryMissing(f"{path} does not exist: write genesis first (`pfb registry genesis`)") from exc
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


Builder = Callable[[tuple[LedgerLine, ...]], Mapping[str, Any]]
Validator = Callable[[tuple[Mapping[str, Any], ...]], object]


def locked_append(
    path: Path,
    build: Builder,
    validate: Validator,
    *,
    genesis: bool = False,
    anchor: bytes | None = None,
) -> LedgerLine:
    """Append one record under an exclusive lock on the ledger file itself.

    build(lines) returns the new record without prev_sha256 from the state read INSIDE the lock; the chain
    sets prev_sha256, and validate(all records, new one included) must pass before a byte is written.
    genesis=True creates the file and requires it to be empty; otherwise the file must already exist.
    `anchor` (the ledger's bytes at HEAD) must be a byte prefix of the file read inside the lock.
    """
    fd = _open_locked(path, create=genesis)
    try:
        data = _read_fd(fd)
        require_extends(data, anchor, path.name)
        lines = parse_ledger(data, path.name)
        if genesis and lines:
            raise GenesisExists(f"{path} already holds {len(lines)} records; genesis is written once")
        built = dict(build(lines))
        built["prev_sha256"] = lines[-1].sha256 if lines else ZERO_HASH
        raw = canonical_bytes(built)
        validate(tuple(line.record for line in lines) + (json.loads(raw),))
        _write_all(fd, raw + NEWLINE)
        os.fsync(fd)
        return LedgerLine(index=len(lines), raw=raw, sha256=sha256_hex(raw), record=json.loads(raw))
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
