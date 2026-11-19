"""`pfb registry status|verify|sync|genesis|checkpoint` and `pfb hypothesis register|show` (ADR 13, 14;
docs/TRIALS.md sections 2-4).

verify replays the chain and hashes each run folder present under runs/ against its
result record's artifact_sha256; a mismatch exits 2, a record written before the key is counted unchecked.

checkpoint commits ledger/registry.jsonl alone ("ledger checkpoint: N <n>, records <r>" plus the lines of
$PFB_COMMIT_TRAILER). hypothesis register runs every refusal of TRIALS.md section 2 and appends the hypothesis
record (N unchanged); hypothesis show prints the registered record as JSON. `hypothesis register --dry-run`
makes every check and appends nothing, printing the refusals as JSON (a book's still-closed
member window is one of them). `hypothesis register --json` prints the appended record as one JSON
object, the bytes `hypothesis show` prints for it; the text line is unchanged without it, and --dry-run
prints its own JSON either way.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from pfb.data.errors import GatewayRefusal
from pfb.data.paths import DEFAULT_PATHS_TOML, load_paths
from pfb.registry.api import RegistryConfig, checkpoint, create_genesis, hypothesis_record, status, sync
from pfb.registry.errors import RegistryError
from pfb.runner.api import RunRefused
from pfb.runner.deps import default_deps
from pfb.runner.register import dry_run, register

registry_app = typer.Typer(help="Global trial registry (ledger/registry.jsonl).", no_args_is_help=True)
hypothesis_app = typer.Typer(help="Hypotheses: register (adds nothing to N) and show.", no_args_is_help=True)

PATHS_OPTION = typer.Option(
    DEFAULT_PATHS_TOML, "--paths", help="config/paths.toml (names the optional upstream checkout)."
)
EXIT_REFUSED = 2


def registry_config(paths_toml: Path) -> RegistryConfig:
    return RegistryConfig.for_repo(load_paths(paths_toml).upstream_root)


def _refused(exc: Exception) -> typer.Exit:
    typer.echo(f"refused: {type(exc).__name__}: {exc}", err=True)
    return typer.Exit(code=EXIT_REFUSED)


@registry_app.command("status")
def status_command(paths_toml: Path = PATHS_OPTION) -> None:
    """N, chain state, upstream sync state and runs begun without a result (open or crashed), as one JSON
    object."""
    try:
        report = status(registry_config(paths_toml))
    except (RegistryError, GatewayRefusal) as exc:
        raise _refused(exc) from exc
    out = {
        "n": report.n,
        "n_pe": report.n_pe,
        "n_local": report.n_local,
        "records": report.records,
        "chain": "ok",
        "last_sha256": report.last_sha256,
        "pe_in_sync": report.pe_in_sync,
        "pe_changed": list(report.pe_changed),
        "open_or_crashed": list(report.open_runs),
        "hypotheses": list(report.hypotheses),
    }
    typer.echo(json.dumps(out, sort_keys=True))


RUNS_DIR_OPTION = typer.Option(
    Path(__file__).resolve().parents[3] / "runs", "--runs-dir", help="The runs directory (default runs/)."
)


@registry_app.command("verify")
def verify_command(paths_toml: Path = PATHS_OPTION, runs_dir: Path = RUNS_DIR_OPTION) -> None:
    """Replay the chain, then hash every run folder present under runs/ against its result record's
    artifact_sha256; exit 2 when a folder does not match. Older records carry no
    hashes: their folders are counted as unchecked."""
    from pfb.registry.api import read_state
    from pfb.registry.artifacts import verify_folders

    try:
        state = read_state(registry_config(paths_toml))
    except (RegistryError, GatewayRefusal) as exc:
        raise _refused(exc) from exc
    report = verify_folders(state, runs_dir)
    typer.echo(json.dumps({"chain": "ok", "n": state.n, "folders": report.to_json()}, sort_keys=True))
    if not report.ok:
        raise typer.Exit(code=EXIT_REFUSED)


@registry_app.command("sync")
def sync_command(paths_toml: Path = PATHS_OPTION) -> None:
    """Import upstream's new ledger lines (its old lines must be an unchanged prefix)."""
    try:
        line = sync(registry_config(paths_toml))
    except (RegistryError, GatewayRefusal) as exc:
        raise _refused(exc) from exc
    if line is None:
        typer.echo("in sync: nothing appended")
        return
    typer.echo(f"synced: N {line.record['n_before']} -> {line.record['n_after']}")


@registry_app.command("genesis")
def genesis_command(paths_toml: Path = PATHS_OPTION) -> None:
    """Write the genesis record once (an upstream's ledger lines unchanged and N = its N; else N = 0)."""
    try:
        line = create_genesis(registry_config(paths_toml))
    except (RegistryError, GatewayRefusal) as exc:
        raise _refused(exc) from exc
    typer.echo(f"genesis: N = {line.record['n_after']}, sha256 {line.sha256}")


@registry_app.command("checkpoint")
def checkpoint_command(paths_toml: Path = PATHS_OPTION) -> None:
    """Commit ledger/registry.jsonl alone (nothing else); prints the commit or that HEAD already holds it."""
    try:
        done = checkpoint(registry_config(paths_toml))
    except (RegistryError, GatewayRefusal) as exc:
        raise _refused(exc) from exc
    if done is None:
        typer.echo("ledger already at HEAD: nothing committed")
        return
    typer.echo(f"ledger checkpoint: N {done.n}, records {done.records}, commit {done.commit}")


DRY_RUN_OPTION = typer.Option(
    False, "--dry-run", help="Every check, nothing appended: the refusals as one JSON object (exit 2 if any)."
)

JSON_OPTION = typer.Option(False, "--json", help="Print the appended record as one JSON object.")


def _dry_run(path: Path, paths_toml: Path) -> None:
    try:
        refusals = dry_run(path, default_deps(paths_toml))
    except (RegistryError, GatewayRefusal, RunRefused) as exc:
        raise _refused(exc) from exc
    out = {
        "schema": "pfb.registration_check/1",
        "path": path.as_posix(),
        "appended": False,
        "ok": not refusals,
        "refusals": [{"code": type(exc).__name__, "reason": str(exc)} for exc in refusals],
    }
    typer.echo(json.dumps(out, sort_keys=True))
    if refusals:
        raise typer.Exit(code=EXIT_REFUSED)


@hypothesis_app.command("register")
def register_command(
    path: Path,
    paths_toml: Path = PATHS_OPTION,
    dry_run_flag: bool = DRY_RUN_OPTION,
    json_flag: bool = JSON_OPTION,
) -> None:
    """Register a hypothesis file, schema 2 or a schema-3 book (committed at HEAD); N is unchanged."""
    if dry_run_flag:
        _dry_run(path, paths_toml)
        return
    try:
        line = register(path, default_deps(paths_toml))
    except (RegistryError, GatewayRefusal, RunRefused) as exc:
        raise _refused(exc) from exc
    record = line.record
    if json_flag:
        typer.echo(json.dumps(dict(record), sort_keys=True))
        return
    typer.echo(
        f"registered {record['hypothesis_id']}: N {record['n_after']}, budget {record['budget']}, "
        f"min_edge_cents {record['power']['min_edge_cents']} at N {record['power']['n_used']}"
    )


@hypothesis_app.command("show")
def show_command(hypothesis_id: str, paths_toml: Path = PATHS_OPTION) -> None:
    """The registered hypothesis record, as one JSON object."""
    try:
        record = hypothesis_record(registry_config(paths_toml), hypothesis_id)
    except (RegistryError, GatewayRefusal) as exc:
        raise _refused(exc) from exc
    typer.echo(json.dumps(dict(record), sort_keys=True))
