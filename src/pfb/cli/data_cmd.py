"""`pfb data ...` commands: the store catalog."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import typer

from pfb.data.catalog import DEFAULT_CATALOG_JSON, Catalog, build_catalog, write_catalog
from pfb.data.errors import GatewayRefusal
from pfb.data.paths import DEFAULT_PATHS_TOML, load_paths
from pfb.data.windows import DEFAULT_WINDOWS_TOML, load_windows

data_app = typer.Typer(help="Bar store commands.", no_args_is_help=True)

PATHS_OPTION = typer.Option(DEFAULT_PATHS_TOML, "--paths", help="config/paths.toml to read.")
WINDOWS_OPTION = typer.Option(DEFAULT_WINDOWS_TOML, "--windows", help="Closed-window table to hash.")
OUT_OPTION = typer.Option(DEFAULT_CATALOG_JSON, "--out", help="Where to write the catalog JSON.")


def summary_lines(catalog: Catalog, out: Path) -> list[str]:
    """Counts only: files per era, ignored files, pin statuses. Never a row or a price."""
    eras = Counter(e.era for e in catalog.files)
    pins = Counter(e.pe_pin_status for e in catalog.files)
    lines = [f"catalog: {len(catalog.files)} files -> {out}"]
    lines += [f"  era {era}: {count}" for era, count in sorted(eras.items())]
    lines.append(f"  ignored: {len(catalog.ignored)}")
    lines.append("  pe_pin: " + ", ".join(f"{k} {v}" for k, v in sorted(pins.items())))
    lines += [f"  PIN MISMATCH: {e.root} {e.era} {e.path}" for e in catalog.pin_mismatches()]
    return lines


@data_app.command("catalog")
def catalog_command(
    paths_toml: Path = PATHS_OPTION,
    windows_toml: Path = WINDOWS_OPTION,
    out: Path = OUT_OPTION,
) -> None:
    """Hash and describe every store file the gateway may open; exit 1 on an upstream pin mismatch."""
    try:
        paths = load_paths(paths_toml)
        windows = load_windows(windows_toml)
        created = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        catalog = build_catalog(paths, windows.sha256, created)
    except GatewayRefusal as exc:
        typer.echo(f"refused: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    write_catalog(catalog, out)
    for line in summary_lines(catalog, out):
        typer.echo(line)
    if catalog.pin_mismatches():
        raise typer.Exit(code=1)
