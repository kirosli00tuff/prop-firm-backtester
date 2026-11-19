"""`pfb prop` (docs/TRIALS.md section 6): the prop Monte Carlo of a saved run on one
plan, or on every plan (`pfb prop matrix <run_id>`). The work is pfb.app.run_prop_mc (pfb.prop.service); this
module only parses and prints.

`pfb prop <run_id> --firm f --plan p --size s [--mc n --seed k ...]` runs config/firms/<f>/<p>_<s>.toml;
`pfb prop matrix <run_id> ...` runs every plan file. With --json it prints the run's result contract with
prop = this request's rows (docs/result.schema.json) and nothing else on stdout; a refusal prints the refusal
object under --json and "refused: <code>: <reason>" on stderr, and exits 2. A trial run takes only what its
hypothesis's [secondary] declares, at its root's own contracts per mini (TrialNotDeclared otherwise); a run's
kind is read from its begin record in the ledger, and a run with none is refused. --soft-daily-loss chooses
the booking of a soft daily loss: "limit" (as upstream, the default) or "minute_low" (a sensitivity). The
compact
per-plan records stay in runs/<run_id>/prop/ and prop_matrix.json; the contract's artifacts name them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from pfb import app as service
from pfb.cli.run_cmd import refused
from pfb.prop.matrix import ALL_PLANS
from pfb.prop.mc import GENERATORS
from pfb.prop.rules import SOFT_DLL_MODES
from pfb.prop.service import PropRefused, Request, require_declared, resolve_source

__all__ = ["Request", "prop_app", "prop_command", "require_declared", "resolve_source"]

prop_app = typer.Typer(help="Prop-firm Monte Carlo on a run's day paths.")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"
DEFAULT_FIRMS_DIR = REPO_ROOT / "config" / "firms"
DEFAULT_LEDGER = REPO_ROOT / "ledger" / "registry.jsonl"
MATRIX_WORD = "matrix"

POOL_OPTION = typer.Option(None, "--pool", help="Further run ids traded in the same account (repeatable).")
FIRM_OPTION = typer.Option(None, "--firm", help="Firm directory under config/firms/ (single-plan mode).")
PLAN_OPTION = typer.Option(None, "--plan", help="Plan name: the file is <plan>_<size>.toml.")
SIZE_OPTION = typer.Option(None, "--size", help="Account size, e.g. 50k.")
MC_OPTION = typer.Option(10_000, "--mc", help="Monte Carlo paths.")
SEED_OPTION = typer.Option(0, "--seed", help="Seed (numpy PCG64 via SeedSequence).")
BLOCK_OPTION = typer.Option(5, "--block", help="Mean block length of the stationary bootstrap (days).")
GENERATOR_OPTION = typer.Option("block", "--generator", help=f"Path generator: {' or '.join(GENERATORS)}.")
STRESS_OPTION = typer.Option(0, "--stress-ticks", help="Extra cost in ticks per side traded (cost stress).")
SOFT_OPTION = typer.Option(
    "limit", "--soft-daily-loss", help=f"Soft daily-loss booking: {' or '.join(SOFT_DLL_MODES)}."
)
COPIES_OPTION = typer.Option(
    None, "--copies", help="N accounts copying the signal, beside N independent ones (one plan; not a trial)."
)
JSON_OPTION = typer.Option(False, "--json", help="Print the result contract only.")
ROOT_OPTION = typer.Option(None, "--root", help="The run's root when the ledger record and summary lack one.")
PER_MINI_OPTION = typer.Option(
    None,
    "--contracts-per-mini",
    help="Override: contracts per mini equivalent (a trial: its root's own only).",
)
RUNS_DIR_OPTION = typer.Option(DEFAULT_RUNS_DIR, "--runs-dir", hidden=True)
FIRMS_DIR_OPTION = typer.Option(DEFAULT_FIRMS_DIR, "--firms-dir", hidden=True)
LEDGER_OPTION = typer.Option(DEFAULT_LEDGER, "--ledger", hidden=True)


def parse_positionals(
    first: str, second: str | None, firm: str | None, plan: str | None, size: str | None
) -> tuple[str, str, str]:
    """(run_id, firm, plan file stem); firm and plan are "*" for the matrix."""
    if first == MATRIX_WORD:
        if not second:
            raise PropRefused("usage: pfb prop matrix <run_id> [--mc n --seed k]")
        if firm or plan or size:
            raise PropRefused("pfb prop matrix runs every plan file: drop --firm, --plan and --size")
        return second, ALL_PLANS, ALL_PLANS
    if second is not None:
        raise PropRefused(
            f"unexpected second argument {second!r} (only `pfb prop matrix <run_id>` takes two)"
        )
    if not (firm and plan and size):
        raise PropRefused("pfb prop <run_id> needs --firm, --plan and --size")
    return first, firm, f"{plan}_{size}"


def _print_human(out: dict[str, Any]) -> None:
    trial = "a declared secondary output, not a trial" if out["kind"] == "trial" else "not a trial"
    typer.echo(f"prop on {out['run_id']} ({out['kind']}, {trial})")
    for row in out["prop"] or []:
        if row["status"] != "ok":
            typer.echo(f"  {row['firm']}/{row['plan']}: skipped ({row['reason']})")
            continue
        typer.echo(f"  {row['firm']}/{row['plan']} (n {row['n_paths']}, seed {row['seed']}, soft daily loss "
                   f"{row['soft_daily_loss']})")  # fmt: skip
        typer.echo(f"    p_pass: {row['p_pass']} (SE {row['p_pass_se']})")
        typer.echo(f"    ev_per_purchase_cents: {row['ev_per_purchase_cents']} (SE {row['ev_se_cents']})")
        copied = row.get("copies")
        if copied:
            typer.echo(f"    {copied['n_accounts']} accounts, copies vs independent (no diversification):")
            for key in ("p_all_pass", "p_all_fail", "corr_pass_two_accounts", "net_sum_sd_cents"):
                typer.echo(f"      {key}: {copied['copies'][key]} vs {copied['independent'][key]}")
    for path in out["artifacts"]:
        if "/prop" in path:
            typer.echo(f"  wrote {path}")


@prop_app.command("prop")
def prop_command(
    first: str = typer.Argument(..., help="A run id, or the word `matrix` followed by a run id."),
    second: str | None = typer.Argument(None, help="The run id after `matrix`."),
    pool: list[str] | None = POOL_OPTION,
    firm: str | None = FIRM_OPTION,
    plan: str | None = PLAN_OPTION,
    size: str | None = SIZE_OPTION,
    mc: int = MC_OPTION,
    seed: int = SEED_OPTION,
    block: int = BLOCK_OPTION,
    generator: str = GENERATOR_OPTION,
    stress_ticks: int = STRESS_OPTION,
    soft_daily_loss: str = SOFT_OPTION,
    copies: int | None = COPIES_OPTION,
    as_json: bool = JSON_OPTION,
    root: str | None = ROOT_OPTION,
    contracts_per_mini: int | None = PER_MINI_OPTION,
    runs_dir: Path = RUNS_DIR_OPTION,
    firms_dir: Path = FIRMS_DIR_OPTION,
    ledger: Path = LEDGER_OPTION,
) -> None:
    """Prop Monte Carlo of a run on one plan, or `pfb prop matrix <run_id>` on every plan file."""
    try:
        try:
            run_id, firm_arg, plan_arg = parse_positionals(first, second, firm, plan, size)
        except PropRefused as exc:
            raise service.Refused(exc.error, exc.reason) from exc
        with service.using(runs_dir=runs_dir, firms_dir=firms_dir, ledger=ledger):
            out = service.run_prop_mc(
                run_id, firm=firm_arg, plan=plan_arg, mc=mc, seed=seed, pool=tuple(pool or ()), block=block,
                stress_ticks=stress_ticks, generator=generator, soft_daily_loss=soft_daily_loss, root=root,
                contracts_per_mini=contracts_per_mini, copies=copies,
            )  # fmt: skip
    except service.Refused as exc:
        raise refused(exc, as_json) from exc
    if as_json:
        typer.echo(json.dumps(out, sort_keys=True, separators=(",", ":")))
        return
    _print_human(out)
