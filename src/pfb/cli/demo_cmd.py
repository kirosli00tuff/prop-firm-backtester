"""`pfb demo`: the pipeline end to end on synthetic data (pfb.demo), then optionally the dashboard on it."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import typer

from pfb.cli.dash_cmd import ADDRESS, DEFAULT_PORT, streamlit_command
from pfb.dash.data import ENV_LEDGER, ENV_REPO, ENV_RUNS
from pfb.demo import DEFAULT_FIRM, DEFAULT_MC, DEFAULT_PLAN, DemoLocations, run_demo
from pfb.registry.api import REPO_ROOT
from pfb.registry.hypfile import PROGRAM_LOOKAHEAD_SEED

demo_app = typer.Typer()

EXIT_REFUSED = 2
MC_OPTION = typer.Option(DEFAULT_MC, "--mc", min=1, help="Prop Monte Carlo paths.")
SEED_OPTION = typer.Option(PROGRAM_LOOKAHEAD_SEED, "--seed", help="Seed of the bars, the checks and the MC.")
FIRM_OPTION = typer.Option(DEFAULT_FIRM, "--firm", help="Firm of the prop plan (config/firms/<firm>).")
PLAN_OPTION = typer.Option(DEFAULT_PLAN, "--plan", help="Plan file stem, e.g. combine_xfa_standard_50k.")
DASH_OPTION = typer.Option(False, "--dash", help="Then start the dashboard on the demo (Ctrl-C stops it).")
PORT_OPTION = typer.Option(DEFAULT_PORT, "--port", help="Dashboard port on 127.0.0.1 (with --dash).")
JSON_OPTION = typer.Option(False, "--json", help="Print the summary as one JSON object only.")
STORE_OPTION = typer.Option(None, "--store-dir", hidden=True, help="Demo store folder (default data/demo).")
RUNS_OPTION = typer.Option(None, "--runs-dir", hidden=True, help="Runs folder (default runs/).")
LEDGER_OPTION = typer.Option(None, "--ledger", hidden=True, help="Ledger (default ledger/registry.jsonl).")
CATALOG_OPTION = typer.Option(None, "--catalog", hidden=True, help="Catalog JSON to write.")


def _locations(
    store_dir: Path | None, runs_dir: Path | None, ledger: Path | None, catalog: Path | None
) -> DemoLocations:
    base = DemoLocations.default()
    return DemoLocations(
        store_dir=store_dir or base.store_dir,
        catalog_json=catalog if catalog is not None else base.catalog_json,
        runs_dir=runs_dir or base.runs_dir,
        ledger=ledger or base.ledger,
    )


def summary_lines(out: dict[str, Any], seconds: float) -> list[str]:
    """The short human summary: store, gateway reads, check, run, prop, ledger."""
    store, run = out["store"], out["run"]
    opened, closed = store["open_read"], store["closed_read"]
    lines = [
        f"demo store: {store['files']} files, {store['rows']} synthetic bars ({', '.join(store['roots'])}), "
        f"{'built' if store['built'] else 'reused'} in {store['store_dir']}",
        f"gateway read {opened['root']} {opened['first']}..{opened['last']}: {opened['rows']} bars",
        f"gateway read {opened['root']} {closed['first']}..{closed['last']}: refused ({closed['refused']})",
        f"check {out['check']['strategy']}: passed at seed {out['check']['seed']}",
        f"run {run['run_id']} ({run['kind']}, {run['status']}): net_cents {run['net_cents']}, "
        f"trades {run['trades']}, sharpe {run['sharpe']}, N +{run['n_added']}",
    ]
    for row in out["prop"] or []:
        lines.append(
            f"prop {row['firm']} {row['plan']}: status {row['status']}, p_pass {row['p_pass']}, "
            f"ev_per_purchase_cents {row['ev_per_purchase_cents']} "
            f"({row['n_paths']} paths, seed {row['seed']})"
        )
    lines += [f"  check warning: {w}" for w in out["check"]["warnings"]]
    ledger = out["ledger"]
    lines.append(f"ledger {ledger['path']}: N {ledger['n']}, {ledger['records']} records")
    lines.append(f"run id: {run['run_id']} (artifacts in {out['runs_dir']}); {seconds:.1f} s")
    return lines


def _dashboard(locations: DemoLocations, port: int) -> int:
    env = {**os.environ, ENV_REPO: str(REPO_ROOT), ENV_LEDGER: str(locations.ledger),
           ENV_RUNS: str(locations.runs_dir)}  # fmt: skip
    typer.echo(f"pfb dash: http://{ADDRESS}:{port} (Ctrl-C stops the server)", err=True)
    return subprocess.call(streamlit_command(port), env=env)


@demo_app.command("demo")
def demo_command(
    mc: int = MC_OPTION,
    seed: int = SEED_OPTION,
    firm: str = FIRM_OPTION,
    plan: str = PLAN_OPTION,
    dash: bool = DASH_OPTION,
    port: int = PORT_OPTION,
    as_json: bool = JSON_OPTION,
    store_dir: Path | None = STORE_OPTION,
    runs_dir: Path | None = RUNS_OPTION,
    ledger: Path | None = LEDGER_OPTION,
    catalog: Path | None = CATALOG_OPTION,
) -> None:
    """Run the pipeline on synthetic data: build the demo store (data/demo/, git-ignored) and read it through
    the DataGateway (an open range, then a range the closed-window table refuses), `pfb check` the reference
    strategy, one synthetic run (adds nothing to N) with the analysis suite and a prop Monte Carlo on one firm
    plan at a fixed seed; print a short summary and the run id. With --dash, start the dashboard on it."""
    from pfb import app as service

    locations = _locations(store_dir, runs_dir, ledger, catalog)
    started = time.monotonic()
    try:
        out = run_demo(locations, firm=firm, plan=plan, mc=mc, seed=seed,
                       progress=None if as_json else lambda text: typer.echo(text, err=True))  # fmt: skip
    except service.Refused as exc:
        typer.echo(f"refused: {exc.code}: {exc.reason}", err=True)
        raise typer.Exit(code=EXIT_REFUSED) from exc
    seconds = time.monotonic() - started
    if as_json:
        typer.echo(json.dumps(out, sort_keys=True))
    else:
        for line in summary_lines(out, seconds):
            typer.echo(line)
    if dash:
        raise typer.Exit(code=_dashboard(locations, port))
