"""Command-line entry point for pfb."""

import typer

import pfb
from pfb.cli.dash_cmd import dash_app
from pfb.cli.data_cmd import data_app
from pfb.cli.demo_cmd import demo_app
from pfb.cli.mcp_cmd import mcp_app
from pfb.cli.new_cmd import new_app
from pfb.cli.prop_cmd import prop_app
from pfb.cli.registry_cmd import hypothesis_app, registry_app
from pfb.cli.run_cmd import run_app

app = typer.Typer(help="PropFirmBacktester command line.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(registry_app, name="registry")
app.add_typer(hypothesis_app, name="hypothesis")
app.add_typer(run_app)
app.add_typer(prop_app)
app.add_typer(new_app)
app.add_typer(mcp_app)
app.add_typer(dash_app)
app.add_typer(demo_app)


@app.callback()
def main() -> None:
    """Keep the app a command group so later commands sit beside `version`."""


@app.command()
def version() -> None:
    """Print the installed pfb version."""
    typer.echo(pfb.__version__)


if __name__ == "__main__":
    app()
