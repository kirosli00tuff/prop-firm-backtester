"""The pfb MCP server: ten tools, each a thin call of the pfb.app function of the same name.

No logic lives here beyond argument passing: every tool calls pfb.app (looked up at call time, so there is no
second path) and returns its dict unchanged. A pfb.app.Refused becomes the refusal JSON of the result contract
({"schema": "pfb.result/1", "status": "refused", "error": {"code", "reason"}}), not a tool error; any other
exception is reported by the SDK as a tool error and the server keeps serving. The SDK's stdio transport
points file descriptor 1 at stderr while it serves, so nothing but the protocol reaches stdout.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

import pfb
import pfb.app as app
from pfb.contract import SCHEMA
from pfb.registry.hypfile import PROGRAM_LOOKAHEAD_SEED

JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]

SERVER_NAME = "pfb"
INSTRUCTIONS = (
    "PropFirmBacktester. Every configuration run on real data is a counted trial (N); the suite and the prop "
    "outputs of a trial are fixed by its registered hypothesis. A refusal comes back as "
    '{"status": "refused", "error": {"code", "reason"}}. Read AGENTS.md in the repository first.'
)


def refusal(exc: app.Refused) -> JsonDict:
    """The refusal JSON of the result contract for one pfb.app.Refused."""
    return {"schema": SCHEMA, "status": "refused", "error": {"code": exc.code, "reason": exc.reason}}


def _answer(call: Callable[[], JsonDict]) -> JsonDict:
    try:
        return call()
    except app.Refused as exc:
        return refusal(exc)


def list_instruments() -> JsonDict:
    """The instruments pfb knows (root, tick size, tick value, sessions)."""
    return _answer(lambda: app.list_instruments())


def list_firms() -> JsonDict:
    """The prop firms and plans configured under config/firms."""
    return _answer(lambda: app.list_firms())


def describe_strategy_template() -> JsonDict:
    """The one-file strategy form (SPEC plus generate) with its rules; `pfb new <name>` copies it."""
    return _answer(lambda: app.describe_strategy_template())


def validate_strategy(path: str, seed: int = PROGRAM_LOOKAHEAD_SEED) -> JsonDict:
    """Lint a strategy file and run its look-ahead check on synthetic bars (seeded, default the program
    look-ahead seed; another seed is marked as standing for no trial's check). Adds nothing to N."""
    return _answer(lambda: app.validate_strategy(path, seed=seed))


def register_hypothesis(path: str) -> JsonDict:
    """Register a committed hypothesis file (schema pfb.hypothesis/2). Adds nothing to N."""
    return _answer(lambda: app.register_hypothesis(path))


def run_backtest(
    strategy: str,
    kind: str,
    hypothesis: str | None = None,
    check_id: str | None = None,
    root: str | None = None,
    window: list[str] | None = None,
    params: dict[str, Any] | None = None,
    seed: int | None = None,
    fees: str = "topstep",
    cost_model: str = "book",
    firm: str | None = None,
    plan: str | None = None,
    mc: int = 0,
    stress: bool = True,
    baseline: bool = True,
    walk_forward: bool = True,
    neighbourhood: bool = False,
) -> JsonDict:
    """Run a strategy (kind synthetic, engine_check or trial) and return the result contract.

    A trial counts in N and needs a registered hypothesis; its suite switches and prop arguments must stay
    at their defaults and seed is left out, because the hypothesis fixes them (its lookahead_seed is the
    seed; another seed is refused). Other kinds: seed None means 0.
    """
    return _answer(
        lambda: app.run_backtest(
            strategy,
            kind=kind,
            hypothesis=hypothesis,
            check_id=check_id,
            root=root,
            window=window,
            params=params,
            seed=seed,
            fees=fees,
            cost_model=cost_model,
            firm=firm,
            plan=plan,
            mc=mc,
            stress=stress,
            baseline=baseline,
            walk_forward=walk_forward,
            neighbourhood=neighbourhood,
        )
    )


def run_prop_mc(
    run_id: str,
    firm: str,
    plan: str,
    mc: int,
    seed: int,
    pool: list[str] | tuple[str, ...] = (),
    block: int | None = None,
    stress_ticks: int = 0,
    generator: str = "block",
    soft_daily_loss: str = "limit",
    root: str | None = None,
    contracts_per_mini: int | None = None,
    copies: int | None = None,
) -> JsonDict:
    """Prop-firm Monte Carlo on a finished run (plan "*" with firm "*" runs every plan); on a trial run only
    the declared plans at the declared mc and seed, every other option at its default.

    copies N (one plan only, never on a trial: refused there as TrialNotDeclared, and with plan "*"): N
    accounts copying the same signal beside N independent accounts, from the same Monte Carlo paths. The prop
    row gains "copies" (p_all_pass, p_all_fail, p_any_pass, the pass correlation of two accounts and the
    payout and net sums over the N accounts, for both columns) and prop/<firm>_<plan>_<seed>_copies<N>.json is
    written with the rules it does not model. It is a descriptive output of this request, never a trial
    output: nothing counts toward N."""
    return _answer(
        lambda: app.run_prop_mc(
            run_id,
            firm=firm,
            plan=plan,
            mc=mc,
            seed=seed,
            pool=tuple(pool),
            block=block,
            stress_ticks=stress_ticks,
            generator=generator,
            soft_daily_loss=soft_daily_loss,
            root=root,
            contracts_per_mini=contracts_per_mini,
            copies=copies,
        )
    )


def get_run(run_id: str) -> JsonDict:
    """The saved result contract of one run (runs/<run_id>/result.json)."""
    return _answer(lambda: app.get_run(run_id))


def compare_runs(run_ids: list[str]) -> JsonDict:
    """Several runs' results side by side. Never use a comparison to pick a configuration."""
    return _answer(lambda: app.compare_runs(run_ids))


def trial_count() -> JsonDict:
    """The program's global trial count N and the registry state. Reads the ledger only."""
    return _answer(lambda: app.trial_count())


TOOLS: tuple[Callable[..., JsonDict], ...] = (
    list_instruments,
    list_firms,
    describe_strategy_template,
    validate_strategy,
    register_hypothesis,
    run_backtest,
    run_prop_mc,
    get_run,
    compare_runs,
    trial_count,
)
TOOL_NAMES: tuple[str, ...] = tuple(tool.__name__ for tool in TOOLS)


def build_server() -> MCPServer:
    """A fresh server with the ten tools registered (no transport started)."""
    server: MCPServer = MCPServer(
        SERVER_NAME, version=pfb.__version__, instructions=INSTRUCTIONS, log_level="WARNING"
    )
    for tool in TOOLS:
        server.add_tool(tool)
    return server


def serve_stdio() -> None:
    """Serve over stdio from the repository root, whatever directory the client started the process in."""
    os.chdir(REPO_ROOT)
    build_server().run("stdio")
