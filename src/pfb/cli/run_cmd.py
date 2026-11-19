"""`pfb run` and `pfb check` (ADR 14; A11; docs/TRIALS.md). Both call pfb.app; nothing here reads
a bar or holds logic beyond parsing and printing.

One command: `pfb run <strategy.py> --kind <k> [--hypothesis f] [--firm f --plan p --mc n] --json`. With
--json it prints the result contract (docs/result.schema.json) and nothing else on stdout. A refusal prints
{"schema": "pfb.result/1", "status": "refused", "error": {"code", "reason"}} on stdout under --json, and
always "refused: <code>: <reason>" on stderr, and exits 2. An unexpected error after begin (the run counts
and stays open or failed) prints the same object with status "error" under --json and exits 1.

A trial: `pfb run <strategy> --kind trial --hypothesis hypotheses/<id>.toml [--root R] [--window K] [--param
name=value ...]`; fees, cost model, suite, neighbourhood, prop outputs and the look-ahead seed
(lookahead_seed; a --seed that differs is refused) come from the hypothesis. A
synthetic run chooses its suite parts (--no-stress, --no-baseline, --no-walk-forward), --neighbourhood and a
prop plan (--firm, --plan as the plan file stem, --mc). `pfb check` prints the warnings of the look-ahead
check, prefixed "step 1: "; `pfb check <file> --json` prints validate_strategy's object
(schema pfb.validation/1) as one line, a refusal the same refused envelope as `pfb run --json` (exit 2).
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import typer

from pfb import app as service
from pfb.registry.hypfile import PROGRAM_LOOKAHEAD_SEED
from pfb.runner.api import RunRefused
from pfb.runner.deps import default_deps

__all__ = ["check_command", "default_deps", "parse_params", "run_app", "run_command"]

run_app = typer.Typer(help="Run or check a strategy.")

EXIT_REFUSED = 2
EXIT_ERROR = 1
KIND_OPTION = typer.Option(..., "--kind", help="trial (counted), engine_check (allowlisted) or synthetic.")
CHECK_ID_OPTION = typer.Option(None, "--check-id", help="The engine check's id in config/engine_checks.toml.")
JSON_OPTION = typer.Option(False, "--json", help="Print the result contract (docs/result.schema.json) only.")
ROOT_OPTION = typer.Option(None, "--root", help="Synthetic or trial runs: the root (default: the SPEC root).")
SEED_OPTION = typer.Option(
    PROGRAM_LOOKAHEAD_SEED,
    "--seed",
    help=f"Seed of the synthetic bars and the look-ahead cuts (default: the program seed "
    f"{PROGRAM_LOOKAHEAD_SEED}; another seed is marked and stands for no trial's check).",
)
RUN_SEED_OPTION = typer.Option(
    None,
    "--seed",
    help="Synthetic and engine_check runs: seed of the synthetic bars, the look-ahead cuts and a prop MC "
    f"(default {PROGRAM_LOOKAHEAD_SEED}, the program seed). A trial runs at its hypothesis's lookahead_seed; "
    "another --seed is refused.",
)
FEES_OPTION = typer.Option(service.DEFAULT_FEES, "--fees", help="Synthetic runs: the firm's fee file.")
COST_OPTION = typer.Option(
    service.DEFAULT_COST_MODEL, "--cost-model", help="Synthetic runs: book, fixed:<k> or stress:<spec>:<k>."
)
HYPOTHESIS_OPTION = typer.Option(
    None, "--hypothesis", help="A trial's hypothesis file (hypotheses/<id>.toml)."
)
WINDOW_OPTION = typer.Option(
    None, "--window", help="A trial: the 1-based index of the hypothesis's [[window]]."
)
PARAM_OPTION = typer.Option(
    None, "--param", help="name=value (repeatable): a parameter other than the SPEC's."
)
NEIGHBOURHOOD_OPTION = typer.Option(
    False, "--neighbourhood", help="Synthetic runs: run the one-step neighbours."
)
FIRM_OPTION = typer.Option(None, "--firm", help="Synthetic runs: prop Monte Carlo on this firm's plan.")
PLAN_OPTION = typer.Option(None, "--plan", help="The plan file stem, e.g. combine_xfa_standard_50k.")
MC_OPTION = typer.Option(0, "--mc", help="Prop Monte Carlo paths (with --firm and --plan).")
STRESS_OPTION = typer.Option(True, "--stress/--no-stress", help="Synthetic runs: cost stress (S1).")
BASELINE_OPTION = typer.Option(True, "--baseline/--no-baseline", help="Synthetic runs: random baseline (S2).")
WF_OPTION = typer.Option(True, "--walk-forward/--no-walk-forward", help="Synthetic runs: walk-forward (S4).")


def parse_params(items: list[str] | None) -> dict[str, Any]:
    """`name=value` pairs; the value is read as a TOML scalar (1, 2.5, true, "x"), else kept as text."""
    out: dict[str, Any] = {}
    for item in items or []:
        name, sep, text = item.partition("=")
        if not sep or not name.strip():
            raise RunRefused(f"--param {item!r} is not name=value")
        try:
            value = tomllib.loads(f"v = {text.strip()}")["v"]
        except tomllib.TOMLDecodeError:
            value = text.strip()
        if isinstance(value, dict | list):
            raise RunRefused(f"--param {name}: a parameter is a scalar")
        out[name.strip()] = value
    return out


def refused(exc: service.Refused, as_json: bool) -> typer.Exit:
    """Print a refusal (JSON on stdout under --json; one line on stderr always); exit 2."""
    if as_json:
        typer.echo(json.dumps(exc.to_json(), sort_keys=True))
    typer.echo(f"refused: {exc.code}: {exc.reason}", err=True)
    return typer.Exit(code=EXIT_REFUSED)


def errored(exc: Exception) -> typer.Exit:
    """Under --json, an unexpected error is printed as the contract's error object; exit 1."""
    typer.echo(json.dumps(service.error_json(exc), sort_keys=True))
    typer.echo(f"error: {type(exc).__name__}: {exc}", err=True)
    return typer.Exit(code=EXIT_ERROR)


def _print_human(out: dict[str, Any]) -> None:
    typer.echo(f"run {out['run_id']} ({out['kind']}, {out['status']}): N {out['n']} (+{out['n_added']})")
    for key in ("net_cents", "trades", "win_rate", "sharpe", "sortino", "max_drawdown_cents", "psr", "dsr"):
        typer.echo(f"  {key}: {out[key]}")
    for key in ("cost_stress", "baseline", "power", "walk_forward", "neighbourhood", "prop"):
        if out[key] is not None:
            typer.echo(f"  {key}: {json.dumps(out[key], sort_keys=True)}")
    typer.echo(f"  begin_sha256: {out['ledger']['begin_sha256']}")
    typer.echo(f"  result_sha256: {out['ledger']['result_sha256']}")
    for warning in out["warnings"]:
        typer.echo(f"  warning: {warning}")


@run_app.command("run")
def run_command(
    strategy: Path,
    kind: str = KIND_OPTION,
    check_id: str | None = CHECK_ID_OPTION,
    as_json: bool = JSON_OPTION,
    root: str | None = ROOT_OPTION,
    seed: int | None = RUN_SEED_OPTION,
    fees: str = FEES_OPTION,
    cost_model: str = COST_OPTION,
    hypothesis: Path | None = HYPOTHESIS_OPTION,
    window: int | None = WINDOW_OPTION,
    param: list[str] | None = PARAM_OPTION,
    neighbourhood: bool = NEIGHBOURHOOD_OPTION,
    firm: str | None = FIRM_OPTION,
    plan: str | None = PLAN_OPTION,
    mc: int = MC_OPTION,
    stress: bool = STRESS_OPTION,
    baseline: bool = BASELINE_OPTION,
    walk_forward: bool = WF_OPTION,
) -> None:
    """Run a strategy and log it in the registry; prints its result contract."""
    try:
        try:
            params = parse_params(param) or None
        except RunRefused as exc:
            raise service.Refused(type(exc).__name__, str(exc)) from exc
        out = service.run_backtest(
            str(strategy.resolve()), kind=kind, hypothesis=None if hypothesis is None else str(hypothesis),
            check_id=check_id, root=root, window=window, params=params, seed=seed, fees=fees,
            cost_model=cost_model, firm=firm, plan=plan, mc=mc, stress=stress, baseline=baseline,
            walk_forward=walk_forward, neighbourhood=neighbourhood,
        )  # fmt: skip
    except service.Refused as exc:
        raise refused(exc, as_json) from exc
    except Exception as exc:
        if not as_json:
            raise
        raise errored(exc) from exc
    if as_json:
        typer.echo(json.dumps(out, sort_keys=True))
        return
    _print_human(out)


CHECK_JSON_OPTION = typer.Option(
    False, "--json", help="Print the validation object (schema pfb.validation/1) as one JSON line only."
)


@run_app.command("check")
def check_command(strategy: Path, seed: int = SEED_OPTION, as_json: bool = CHECK_JSON_OPTION) -> None:
    """Lint the strategy and run the truncation check on synthetic bars; logs nothing; prints its warnings
    (with --json: the validation object, or the refused envelope and exit 2)."""
    try:
        out = service.validate_strategy(str(strategy.resolve()), seed=seed)
    except service.Refused as exc:
        raise refused(exc, as_json) from exc
    except Exception as exc:
        if not as_json:
            raise
        raise errored(exc) from exc
    if as_json:
        typer.echo(json.dumps(out, sort_keys=True))
        return
    typer.echo(f"ok: {strategy.name} passes the lint and the truncation check at seed {seed}")
    for warning in out["warnings"]:
        typer.echo(f"warning: {warning}")
