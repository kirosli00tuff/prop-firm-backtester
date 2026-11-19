"""Upstream ledger sync (ADR 13): refuse any append while upstream's ledger differs from the last genesis or
sync.

`pfb registry sync` imports upstream's new lines; the lines already imported must be an unchanged prefix of
the
file,
else SyncPrefixChanged. A sync adds upstream's N increase to the registry's N.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pfb.registry.errors import SyncPrefixChanged, SyncRequired
from pfb.registry.genesis import PE_FILES, pe_hashes, pe_n_of, read_pe_files
from pfb.registry.records import LedgerState


def changed_pe_files(state: LedgerState, pe_root: Path | None) -> tuple[str, ...]:
    """Upstream ledger files whose sha256 now differs from the last genesis or sync record. Without an
    upstream root: none when the ledger imported none, else SyncRequired (the upstream must be named)."""
    if pe_root is None:
        if state.pe_sha256:
            raise SyncRequired(
                "the ledger imports an upstream ledger, but no upstream root is configured "
                "(config/paths.toml [upstream] root)"
            )
        return ()
    now = pe_hashes(pe_root)
    return tuple(name for name in PE_FILES if now[name] != state.pe_sha256.get(name))


def require_synced(state: LedgerState, pe_root: Path | None) -> None:
    changed = changed_pe_files(state, pe_root)
    if changed:
        raise SyncRequired(
            f"the upstream ledger changed since the last sync ({', '.join(changed)}): "
            "run `pfb registry sync` first"
        )


def sync_body(state: LedgerState, pe_root: Path | None) -> dict[str, Any] | None:
    """The sync-specific fields, or None when upstream's ledger is unchanged (or there is no upstream)."""
    if pe_root is None:
        changed_pe_files(state, pe_root)  # refuses when the ledger imported an upstream
        return None
    files = read_pe_files(pe_root)
    if all(files[name].sha256 == state.pe_sha256.get(name) for name in PE_FILES):
        return None
    pe_files: dict[str, dict[str, Any]] = {}
    for name in PE_FILES:
        old, new = state.pe_lines.get(name, ()), files[name].lines
        if new[: len(old)] != old:
            raise SyncPrefixChanged(
                f"upstream {name}: the {len(old)} lines already imported are no longer an unchanged "
                "prefix of the file; refused (upstream's ledger is append-only)"
            )
        pe_files[name] = {"sha256": files[name].sha256, "lines_added": list(new[len(old) :])}
    n_pe = pe_n_of({name: f.lines for name, f in files.items()})
    return {"n_pe_before": state.n_pe, "n_pe": n_pe, "pe_files": pe_files}
