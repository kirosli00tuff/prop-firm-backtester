"""`pfb new <name>`: copy strategies/_template.py to strategies/<name>.py with hypothesis_id
"h_<name>", refusing to overwrite. The work is pfb.app.new_strategy; this only parses and prints."""

from __future__ import annotations

import json

import typer

from pfb import app as service
from pfb.cli.run_cmd import refused

new_app = typer.Typer(help="Start a strategy from the template.")


@new_app.command("new")
def new_command(
    name: str = typer.Argument(..., help="The strategy name: letters, digits and underscores."),
    as_json: bool = typer.Option(False, "--json", help="Print one JSON object."),
) -> None:
    """Copy the strategy template to strategies/<name>.py (never overwrites)."""
    try:
        out = service.new_strategy(name)
    except service.Refused as exc:
        raise refused(exc, as_json) from exc
    if as_json:
        typer.echo(json.dumps(out, sort_keys=True))
        return
    typer.echo(f"wrote {out['path']} (hypothesis_id {out['hypothesis_id']}); next: pfb check {out['path']}")
