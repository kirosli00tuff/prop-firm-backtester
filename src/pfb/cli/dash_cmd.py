"""`pfb dash`: start the read-only Streamlit dashboard.

The server binds 127.0.0.1 only, sends no usage statistics, runs headless (no browser is opened, no e-mail
prompt), watches no file and serves no agent API, and carries the dark theme as --theme.* flags (the values
of .streamlit/config.toml). The app itself makes no network call. The ledger, runs and upstream locations
reach the
app through the PFB_DASH_* environment variables (pfb.dash.data). Stop with Ctrl-C.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import typer

import pfb.dash
from pfb.dash.data import ENV_LEDGER, ENV_PE, ENV_REPO, ENV_RUNS
from pfb.registry.api import LEDGER_REL, REPO_ROOT

dash_app = typer.Typer()

APP_SCRIPT = Path(pfb.dash.__file__).resolve().parent / "app.py"
ADDRESS = "127.0.0.1"
DEFAULT_PORT = 8501
SERVER_FLAGS = (
    ("--server.address", ADDRESS),
    ("--browser.gatherUsageStats", "false"),
    ("--server.headless", "true"),
    ("--server.fileWatcherType", "none"),
    ("--server.runOnSave", "false"),
    ("--server.enableAgentApi", "false"),
    ("--server.enableStaticServing", "false"),
    ("--client.showErrorLinks", "false"),
    ("--client.toolbarMode", "minimal"),
    ("--server.maxUploadSize", "1"),  # MB; the upload itself is capped at 256 KiB
)

THEME_FLAGS = (
    ("--theme.base", "dark"),
    ("--theme.primaryColor", "#4C9AFF"),
    ("--theme.backgroundColor", "#0E1014"),
    ("--theme.secondaryBackgroundColor", "#161A20"),
    ("--theme.textColor", "#E6E8EB"),
    ("--theme.font", "sans-serif"),
    ("--theme.codeFont", "monospace"),
)

PORT_OPTION = typer.Option(DEFAULT_PORT, "--port", help="Port on 127.0.0.1.")
RUNS_OPTION = typer.Option(REPO_ROOT / "runs", "--runs-dir", help="The runs directory to show.")
LEDGER_OPTION = typer.Option(REPO_ROOT / LEDGER_REL, "--ledger", help="The ledger to show.")
REPO_OPTION = typer.Option(REPO_ROOT, "--repo", hidden=True, help="Repository whose HEAD anchors the ledger.")
PE_OPTION = typer.Option(None, "--pe-root", hidden=True, help="Upstream checkout (ledger sync state).")


def streamlit_command(port: int) -> list[str]:
    """`python -m streamlit run <app.py>` with the fixed server and theme flags and the port."""
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP_SCRIPT)]
    for flag, value in (*SERVER_FLAGS, *THEME_FLAGS):
        cmd += [flag, value]
    return [*cmd, "--server.port", str(port)]


@dash_app.command("dash")
def dash_command(
    port: int = PORT_OPTION,
    runs_dir: Path = RUNS_OPTION,
    ledger: Path = LEDGER_OPTION,
    repo: Path = REPO_OPTION,
    pe_root: Path | None = PE_OPTION,
) -> None:
    """Start the read-only dashboard on http://127.0.0.1:<port> (backtest, runs)."""
    env = {**os.environ, ENV_REPO: str(repo), ENV_LEDGER: str(ledger), ENV_RUNS: str(runs_dir)}
    if pe_root is not None:
        env[ENV_PE] = str(pe_root)
    typer.echo(f"pfb dash: http://{ADDRESS}:{port} (Ctrl-C stops the server)", err=True)
    raise typer.Exit(code=subprocess.call(streamlit_command(port), env=env))
