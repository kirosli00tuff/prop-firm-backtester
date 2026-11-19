"""The one service layer the CLI and the MCP server call (docs/TRIALS.md).

Every function returns a JSON-ready dict and raises :class:`Refused` (code, reason) for a refusal; a CLI or
MCP function only parses arguments and prints. run_backtest, run_prop_mc and get_run return the result
contract (pfb.contract, docs/result.schema.json); run_backtest saves it as runs/<run_id>/result.json and
get_run reads it back (or builds it from the ledger and the artifacts when the file is absent).

For a trial the hypothesis fixes the suite, the neighbourhood and the prop outputs: the switches stress,
baseline, walk_forward and neighbourhood and the prop arguments firm, plan and mc must stay at their defaults,
else Refused (SuiteMismatch); seed is left out (None) and the trial runs at its hypothesis's lookahead_seed, a
seed that differs from it is Refused (SeedMismatch) before anything runs. For a synthetic or
engine_check run seed None means 0, and the switches build the suite (the defaults
of pfb.analysis.spec.SuiteSpec with the parts turned off set to off), and firm, plan and mc (all three or
none) run the prop Monte Carlo on the saved run after its result record.

Where things are read and written is the AppContext (tests and scratch examples use `using(...)`); by default
the runner's default_deps (this repository's ledger), runs/ and config/firms/.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

from pfb.contract import build_result, refusal, validate
from pfb.registry.hypfile import PROGRAM_LOOKAHEAD_SEED

JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"
DEFAULT_FIRMS_DIR = REPO_ROOT / "config" / "firms"
STRATEGIES_DIR = REPO_ROOT / "strategies"
TEMPLATE = STRATEGIES_DIR / "_template.py"
RESULT_FILE = "result.json"
NEIGHBOURHOOD_FILE = "neighbourhood.json"
PROP_DIR = "prop"
PROP_INDEX = "index.json"
DEFAULT_FEES, DEFAULT_COST_MODEL = "topstep", "book"
MAX_COMPARE = 50
COMPARE_KEYS = (
    "run_id", "kind", "status", "hypothesis_id", "root", "window", "params", "n", "net_cents", "trades",
    "sharpe", "sortino", "max_drawdown_cents", "psr", "dsr",
)  # fmt: skip
TEMPLATE_ID_LINE = '"hypothesis_id": "h_template",'


class Refused(Exception):
    """A refusal: a stable machine code (the refusing class's name) and a human reason."""

    def __init__(self, code: str, reason: str) -> None:
        super().__init__(f"{code}: {reason}")
        self.code = code
        self.reason = reason

    def to_json(self) -> JsonDict:
        return refusal(self.code, self.reason)


# ----------------------------------------------------------------------------------- context ----
@dataclass(frozen=True)
class AppContext:
    """Where the app reads and writes. deps None: pfb.runner.api.default_deps(); ledger None: the
    registry config's ledger."""

    deps: Any = None  # pfb.runner.deps.RunnerDeps
    runs_dir: Path = DEFAULT_RUNS_DIR
    firms_dir: Path = DEFAULT_FIRMS_DIR
    ledger: Path | None = None
    strategies_dir: Path = STRATEGIES_DIR


_CONTEXT: ContextVar[AppContext] = ContextVar("pfb_app_context", default=AppContext())  # noqa: B039


def context() -> AppContext:
    return _CONTEXT.get()


@contextmanager
def using(**changes: Any) -> Iterator[AppContext]:
    """Run the app against other deps, runs directory, firms directory or ledger inside the block."""
    token = _CONTEXT.set(replace(_CONTEXT.get(), **changes))
    try:
        yield _CONTEXT.get()
    finally:
        _CONTEXT.reset(token)


def _deps() -> Any:
    ctx = context()
    if ctx.deps is not None:
        return ctx.deps
    from pfb.runner import api as runner_api

    return runner_api.default_deps()


def _refusal_types() -> tuple[type[BaseException], ...]:
    from pfb.analysis.spec import SuiteError
    from pfb.data.errors import GatewayRefusal
    from pfb.engine.strategy import StrategyError
    from pfb.prop.matrix import PlanRefused
    from pfb.registry.errors import RegistryError
    from pfb.runner.api import RunRefused

    return (RunRefused, RegistryError, GatewayRefusal, StrategyError, SuiteError, PlanRefused)


@contextmanager
def _refusals() -> Iterator[None]:
    """A refusal raised below (runner, registry, gateway, strategy, suite, prop) becomes Refused."""
    from pfb.prop.service import PropRefused

    try:
        yield
    except Refused:
        raise
    except PropRefused as exc:
        field = f" (field {exc.field})" if exc.field else ""
        raise Refused(exc.error, f"{exc.reason}{field}") from exc
    except _refusal_types() as exc:
        raise Refused(type(exc).__name__, str(exc)) from exc


def _ledger_path() -> Path:
    ctx = context()
    return ctx.ledger if ctx.ledger is not None else _deps().registry_config().registry_path


def _registry_ledger() -> Path:
    """The ledger trials are begun in: the deps' registry ledger when the app runs on other deps, else the
    repository's (no paths file is read for it)."""
    from pfb.prop.service import REPO_LEDGER

    ctx = context()
    return ctx.deps.registry_config().registry_path if ctx.deps is not None else REPO_LEDGER


def _repo_root() -> Path:
    """The registry config's repository (artifact names are relative to it); this repository when no
    paths config can be read (a prop request on a saved run needs none)."""
    from pfb.data.errors import GatewayRefusal

    try:
        return _deps().registry_config().repo_root
    except GatewayRefusal:
        return REPO_ROOT


# --------------------------------------------------------------------------------- catalogue ----
def list_instruments() -> JsonDict:
    """Every root of config/instruments.toml with its contract facts (decimals as strings)."""
    from pfb.instruments.table import load_instruments

    table = load_instruments()
    rows = []
    for root in table.roots():
        inst = table.get(root)
        rows.append({
            "root": inst.root, "group": inst.group, "subgroup": inst.subgroup, "exposure": inst.exposure,
            "contract_type": inst.contract_type, "tick_size": str(inst.tick_size),
            "tick_value_usd": str(inst.tick_value_usd), "multiplier_usd": str(inst.multiplier_usd),
            "admissible": inst.admissible,
        })  # fmt: skip
    return {"schema": "pfb.instruments/1", "sha256": table.sha256, "instruments": rows}


def list_firms() -> JsonDict:
    """Every plan file under config/firms/ (firm, plan file stem, label, status, sha256)."""
    from pfb.prop.config import PlanConfigError, load_plan
    from pfb.prop.matrix import discover_plans

    firms_dir = context().firms_dir
    with _refusals():
        paths = discover_plans(firms_dir)
    plans = []
    for path in paths:
        row: JsonDict = {
            "firm": path.parent.name,
            "plan": path.stem,
            "file": f"{path.parent.name}/{path.name}",
        }
        try:
            plan = load_plan(path)
            row |= {"label": plan.label, "status": plan.status, "size": plan.size, "sha256": plan.sha256,
                    "error": None}  # fmt: skip
        except PlanConfigError as exc:
            row |= {"label": None, "status": None, "size": None, "sha256": None, "error": str(exc)}
        plans.append(row)
    return {"schema": "pfb.firms/1", "plans": plans}


def describe_strategy_template() -> JsonDict:
    """The one-file strategy template (strategies/_template.py) and the rules a strategy must keep."""
    text = TEMPLATE.read_text(encoding="utf-8")
    return {
        "schema": "pfb.template/1",
        "path": "strategies/_template.py",
        "text": text,
        "spec_keys": ["hypothesis_id", "root", "bar_minutes", "session", "contracts", "params"],
        "signals": ["entry_long", "entry_short", "exit", "stop_ticks", "target_ticks"],
        "rules": [
            "one file: SPEC (a dict literal) and generate(bars, params) returning the five signal arrays",
            "numpy only; no I/O, no clock, no randomness, no imports beyond the lint's allowlist",
            "a signal on bar i acts at bar i + 1 (the engine fills at the next open); never read bar i + 1",
            "every parameter has a default and a grid; a trial's grid is the hypothesis's [grid]",
            "`pfb new <name>` copies the template; `pfb check` runs the lint and the look-ahead check",
            f"the look-ahead check runs at the program seed {PROGRAM_LOOKAHEAD_SEED}; a hypothesis may set "
            "another lookahead_seed only with a non-empty lookahead_seed_reason",
        ],
    }


# ---------------------------------------------------------------------------------- strategy ----
def _spec_json(spec: Any) -> JsonDict:
    return {
        "hypothesis_id": spec.hypothesis_id, "root": spec.root, "bar_minutes": spec.bar_minutes,
        "session": spec.session, "contracts": spec.contracts,
        "params": {k: {"default": p.default, "grid": list(p.grid)} for k, p in spec.params.items()},
    }  # fmt: skip


def validate_strategy(path: str, *, seed: int = PROGRAM_LOOKAHEAD_SEED) -> JsonDict:
    """The lint, the load and the look-ahead check on synthetic bars (seeded); logs nothing. A seed other
    than the program seed is marked: such a check stands for no trial's, which runs
    its own step 1 at its hypothesis's lookahead_seed."""
    from pfb.runner.api import checked_strategy, lookahead_signal, step1_warnings

    with _refusals():
        deps = _deps()
        strategy, resolved, warnings = checked_strategy(_repo_path(path).resolve(), deps, None, seed)
    marked = [] if seed == PROGRAM_LOOKAHEAD_SEED else [not_program_seed_text(seed)]
    return {
        "schema": "pfb.validation/1", "strategy": str(path), "ok": True, "sha256": strategy.sha256,
        "spec": _spec_json(strategy.spec), "params": resolved,
        "warnings": [*step1_warnings(warnings), *marked], "lookahead_signal": lookahead_signal(warnings),
        "seed": seed, "program_seed": seed == PROGRAM_LOOKAHEAD_SEED,
    }  # fmt: skip


def not_program_seed_text(seed: int) -> str:
    return (
        f"seed {seed} is not the program look-ahead seed {PROGRAM_LOOKAHEAD_SEED}: this check cannot "
        "satisfy a trial's look-ahead requirement (a trial runs its own check at its hypothesis's "
        "lookahead_seed)"
    )


def register_hypothesis(path: str) -> JsonDict:
    """`pfb hypothesis register`: every refusal of TRIALS.md section 2, then the record (N unchanged)."""
    from pfb.runner.register import register

    with _refusals():
        line = register(_repo_path(path), _deps(), context().firms_dir)
    record = dict(line.record)
    return {"schema": "pfb.registration/1", "hypothesis_id": record["hypothesis_id"], "n": record["n_after"],
            "record_sha256": line.sha256, "record": record}  # fmt: skip


def new_strategy(name: str) -> JsonDict:
    """`pfb new <name>`: copy the template to strategies/<name>.py with hypothesis_id "h_<name>"."""
    if not name or not name.replace("_", "").isalnum() or not name[0].isalpha() or not name.isascii():
        raise Refused(
            "BadName", f"{name!r}: a strategy name is letters, digits and underscores, from a letter"
        )
    target = context().strategies_dir / f"{name}.py"
    if target.exists():
        raise Refused("Exists", f"{target.name} exists; pfb new never overwrites")
    text = TEMPLATE.read_text(encoding="utf-8")
    if text.count(TEMPLATE_ID_LINE) != 1:
        raise Refused("TemplateChanged", f"the template has no single line {TEMPLATE_ID_LINE}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as fh:
        fh.write(text.replace(TEMPLATE_ID_LINE, f'"hypothesis_id": "h_{name}",'))
    return {"schema": "pfb.new/1", "path": f"strategies/{target.name}", "hypothesis_id": f"h_{name}"}


# ------------------------------------------------------------------------------- run helpers ----
def _suite(stress: bool, baseline: bool, walk_forward: bool, neighbourhood: bool) -> Any:
    from pfb.analysis.spec import SuiteSpec

    default = SuiteSpec()
    return replace(
        default,
        cost_stress_ticks=default.cost_stress_ticks if stress else (),
        baseline_draws=default.baseline_draws if baseline else 0,
        baseline_seed=default.baseline_seed if baseline else 0,
        walk_forward_folds=default.walk_forward_folds if walk_forward else 0,
        neighbourhood=neighbourhood,
    )


def _trial_fixed(args: Mapping[str, Any]) -> None:
    """C4: a trial's suite, neighbourhood and prop outputs are its hypothesis's; no switch changes them."""
    moved = [k for k, v in (("stress", True), ("baseline", True), ("walk_forward", True),
                            ("neighbourhood", False), ("firm", None), ("plan", None), ("mc", 0))
             if args[k] != v]  # fmt: skip
    if moved:
        raise Refused("SuiteMismatch", f"a trial's suite and prop outputs are fixed by its hypothesis; leave "
                                       f"{moved} at their defaults")  # fmt: skip


def _prop_request(firm: str | None, plan: str | None, mc: int) -> tuple[str, str] | None:
    if firm is None and plan is None and mc == 0:
        return None
    if not (firm and plan and mc > 0):
        raise Refused("BadRequest", "the prop Monte Carlo needs firm, plan and mc > 0 together")
    return firm, plan


def _window_args(kind: str, hypothesis: Path | None, window: Any) -> JsonDict:
    """{first, last, window}: an int is a trial's 1-based [[window]] index; a pair [first, last] is the
    window itself (for a trial, matched to its declared index)."""
    if window is None:
        return {"first": None, "last": None, "window": None}
    if isinstance(window, int) and not isinstance(window, bool):
        return {"first": None, "last": None, "window": window}
    if isinstance(window, str) or not isinstance(window, Sequence) or len(window) != 2:
        raise Refused("BadRequest", f"window {window!r} is a 1-based index or [first, last]")
    try:
        first, last = (date.fromisoformat(str(d)) for d in window)
    except ValueError as exc:
        raise Refused("BadRequest", f"window {list(window)!r}: {exc}") from exc
    index = None
    if kind == "trial" and hypothesis is not None:
        from pfb.registry.hypfile import load_hypothesis

        with _refusals():
            declared = [tuple(w) for w in load_hypothesis(hypothesis).windows]
        pair = (first.isoformat(), last.isoformat())
        if pair not in declared:
            raise Refused("UndeclaredConfiguration", f"window {list(pair)} is not one of {declared}")
        index = declared.index(pair) + 1
    return {"first": first, "last": last, "window": index}


def _repo_path(path: str) -> Path:
    """A path as given when absolute, else relative to the repository root (not to the working directory:
    the MCP server's working directory is not fixed); the CLI passes paths resolved from its own."""
    given = Path(path)
    return given if given.is_absolute() else _repo_root() / given


def _hypothesis_path(hypothesis: str | None) -> Path | None:
    return None if hypothesis is None else _repo_path(hypothesis)


def _wants_neighbours(kind: str, hypothesis: Path | None, flag: bool) -> bool:
    if kind != "trial":
        return flag
    if hypothesis is None:
        return False
    from pfb.registry.hypfile import load_hypothesis

    with _refusals():
        return load_hypothesis(hypothesis).suite.neighbourhood


# ------------------------------------------------------------------------------ run sources ----
def _registry_config() -> Any:
    config = _deps().registry_config()
    ledger = context().ledger
    return config if ledger is None else replace(config, registry_path=ledger)


def _run_dir(run_id: str) -> Path:
    kinds = ("trial", "engine_check", "synthetic")
    plain = bool(run_id) and "/" not in run_id and "\\" not in run_id and not run_id.startswith(".")
    if not plain or run_id.split("-", 1)[0] not in kinds:
        raise Refused("BadRunId", f"run id {run_id!r} is not a plain name starting with a run kind {kinds}")
    return context().runs_dir / run_id


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


@dataclass(frozen=True)
class _Lines:
    """A run's begin and result records with the sha256 of each ledger line (None when absent)."""

    begin: Mapping[str, Any] | None
    begin_sha256: str | None
    result: Mapping[str, Any] | None
    result_sha256: str | None


def _ledger_lines(ledger: Path, run_id: str) -> _Lines:
    from pfb.registry.chain import sha256_hex

    found: dict[str, tuple[Mapping[str, Any], str]] = {}
    if ledger.is_file():
        for raw in ledger.read_bytes().splitlines():
            if run_id.encode() not in raw:
                continue
            record = json.loads(raw)
            if record.get("run_id") == run_id:
                found.setdefault(
                    "result" if record.get("kind") == "result" else "begin", (record, sha256_hex(raw))
                )
    begin, result = found.get("begin"), found.get("result")
    return _Lines(begin[0] if begin else None, begin[1] if begin else None,
                  result[0] if result else None, result[1] if result else None)  # fmt: skip


def _hypothesis_record(ledger: Path, hypothesis_id: Any) -> Mapping[str, Any] | None:
    from pfb.prop.service import ledger_records

    found = [
        r
        for r in ledger_records(ledger, "hypothesis_id", str(hypothesis_id))
        if r.get("kind") == "hypothesis"
    ]
    return found[0] if found else None


def _artifacts(run_id: str, names: Sequence[str]) -> list[str]:
    from pfb.runner.api import planned_artifacts

    return planned_artifacts(context().runs_dir, run_id, _repo_root(), list(names)) if names else []


def _neighbourhood(run_dir: Path) -> Mapping[str, Any] | None:
    doc = _read_json(run_dir / NEIGHBOURHOOD_FILE)
    return doc if isinstance(doc, Mapping) and doc["centre"]["run_id"] == run_dir.name else None


def _trial_prop(run_dir: Path) -> tuple[list[Mapping[str, Any]] | None, list[str]]:
    """A trial's declared prop rows from prop/index.json (None when none was declared or it failed)."""
    index = _read_json(run_dir / PROP_DIR / PROP_INDEX)
    if not isinstance(index, Mapping) or not index["declared"]:
        return None, []
    files = sorted(p.name for p in (run_dir / PROP_DIR).glob("*.json") if p.name != PROP_INDEX)
    return list(index["rows"]), [f"{PROP_DIR}/{PROP_INDEX}", *(f"{PROP_DIR}/{f}" for f in files)]


def _contract(lines: _Lines, ledger: Path, *, warnings: Sequence[str] = (),
              prop: Sequence[Mapping[str, Any]] | None = None, prop_names: Sequence[str] = (),
              saved: bool) -> JsonDict:  # fmt: skip
    """The contract of one run from its ledger lines and its artifacts."""
    assert lines.begin is not None
    run_id = str(lines.begin["run_id"])
    run_dir = _run_dir(run_id)
    is_trial = lines.begin.get("kind") == "trial"
    neighbourhood = _neighbourhood(run_dir)
    names = [NEIGHBOURHOOD_FILE] if neighbourhood is not None else []
    if is_trial:
        prop, prop_names = _trial_prop(run_dir)
    names += [*prop_names, *([RESULT_FILE] if saved else [])]
    return build_result(
        begin=lines.begin, result=lines.result, begin_sha256=lines.begin_sha256,
        result_sha256=lines.result_sha256,
        hypothesis=_hypothesis_record(ledger, lines.begin.get("hypothesis_id")) if is_trial else None,
        neighbourhood=neighbourhood, prop=prop, warnings=warnings, artifacts=_artifacts(run_id, names),
    )  # fmt: skip


def _save(contract: JsonDict) -> JsonDict:
    validate(contract)
    path = _run_dir(contract["run_id"]) / RESULT_FILE
    path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    return contract


def _outcome_lines(outcome: Any) -> _Lines:
    return _Lines(outcome.begin_record, outcome.begin_sha256, outcome.result_record, outcome.result_sha256)


# ------------------------------------------------------------------------------------- runs ----
def run_backtest(
    strategy: str,
    *,
    kind: str,
    hypothesis: str | None = None,
    check_id: str | None = None,
    root: str | None = None,
    window: Any = None,
    params: dict[str, Any] | None = None,
    seed: int | None = None,
    fees: str = DEFAULT_FEES,
    cost_model: str = DEFAULT_COST_MODEL,
    firm: str | None = None,
    plan: str | None = None,
    mc: int = 0,
    stress: bool = True,
    baseline: bool = True,
    walk_forward: bool = True,
    neighbourhood: bool = False,
) -> JsonDict:
    """Run one strategy through pfb.runner (a trial counts in N), save and return its result contract.

    window: a trial's 1-based [[window]] index or [first, last]; a synthetic run takes none, an engine
    check takes its allowlist entry's. A trial's neighbourhood runs when its hypothesis declares it, a
    synthetic run's when neighbourhood is True; each neighbour's contract is saved in its own run directory.
    seed: a synthetic or engine_check run's seed (None: 0); a trial's is its hypothesis's lookahead_seed, and
    a seed given that differs from it is refused. firm, plan (the plan file stem, e.g.
    "combine_xfa_standard_50k") and mc run the prop Monte Carlo of a synthetic or engine_check run at the
    run's seed."""
    from pfb.runner.api import RUNNER_KINDS, run
    from pfb.runner.neighbours import run_with_neighbours

    if kind not in RUNNER_KINDS:
        raise Refused("BadRequest", f"kind {kind!r} is not one of {list(RUNNER_KINDS)}")
    switches = {"stress": stress, "baseline": baseline, "walk_forward": walk_forward,
                "neighbourhood": neighbourhood, "firm": firm, "plan": plan, "mc": mc}  # fmt: skip
    is_trial = kind == "trial"
    if is_trial:
        _trial_fixed(switches)
    prop = None if is_trial else _prop_request(firm, plan, mc)
    with _refusals():
        deps = _deps()
        hyp_path = _hypothesis_path(hypothesis)
        bounds = _window_args(kind, hyp_path, window)
        common: dict[str, Any] = {
            "kind": kind, "root": root, "params": params, "seed": seed, "deps": deps,
            "runs_dir": context().runs_dir, "hypothesis_path": hyp_path, **bounds,
            "fees": None if is_trial and fees == DEFAULT_FEES else fees,
            "cost_model": None if is_trial and cost_model == DEFAULT_COST_MODEL else cost_model,
            "suite": None if is_trial else _suite(stress, baseline, walk_forward, neighbourhood),
        }  # fmt: skip
        if _wants_neighbours(kind, hyp_path, neighbourhood):
            found = run_with_neighbours(_repo_path(strategy), **common)
            outcomes = [found.centre, *found.neighbours]
        else:
            outcomes = [run(_repo_path(strategy), check_id=check_id, **common)]
    ledger = deps.registry_config().registry_path
    for outcome in outcomes[1:]:
        _save(_contract(_outcome_lines(outcome), ledger, warnings=outcome.warnings, saved=True))
    centre = outcomes[0]
    rows, names = (None, []) if prop is None else _run_prop(centre.begin_record["run_id"], prop[0], prop[1],
                                                            mc=mc, seed=int(centre.begin_record["seed"]),
                                                            ledger=ledger)  # fmt: skip
    return _save(_contract(_outcome_lines(centre), ledger, warnings=centre.warnings, prop=rows,
                           prop_names=names, saved=True))  # fmt: skip


def _run_prop(run_id: str, firm: str, plan: str, *, mc: int, seed: int, ledger: Path,
              **options: Any) -> tuple[list[dict[str, Any]], list[str]]:  # fmt: skip
    """The prop service on a saved run: (contract rows, file names under the run directory)."""
    from pfb.prop.matrix import ALL_PLANS
    from pfb.prop.service import TRIAL_DEFAULTS, Request, run_request

    is_matrix = firm == ALL_PLANS and plan == ALL_PLANS
    stem, _, size = plan.rpartition("_")
    if not is_matrix and not (stem and size):
        raise Refused("BadRequest", f"plan {plan!r} is a plan file stem <plan>_<size>, or firm and plan '*'")
    block = options.get("block")
    args = {"mc": mc, "seed": seed, "block": TRIAL_DEFAULTS["block"] if block is None else block,
            "generator": options.get("generator", TRIAL_DEFAULTS["generator"]),
            "stress_ticks": options.get("stress_ticks", 0),
            "soft_daily_loss": options.get("soft_daily_loss", TRIAL_DEFAULTS["soft_daily_loss"]),
            "copies": options.get("copies")}  # fmt: skip
    ctx = context()
    req = Request(is_matrix, run_id, list(options.get("pool", ())), None if is_matrix else firm,
                  None if is_matrix else stem, None if is_matrix else size, args, options.get("root"),
                  options.get("contracts_per_mini"), ctx.runs_dir, ctx.firms_dir, ledger,
                  _registry_ledger())  # fmt: skip
    with _refusals():
        out = run_request(req, repo_root=_repo_root())
    return out.rows, out.artifacts


def run_prop_mc(
    run_id: str,
    *,
    firm: str,
    plan: str,
    mc: int,
    seed: int,
    pool: Sequence[str] = (),
    block: int | None = None,
    stress_ticks: int = 0,
    generator: str = "block",
    soft_daily_loss: str = "limit",
    root: str | None = None,
    contracts_per_mini: int | None = None,
    copies: int | None = None,
) -> JsonDict:
    """The prop Monte Carlo of a saved run on one plan (firm, plan file stem) or on every plan (firm and plan
    "*"): the run's contract with prop = this request's rows and the files it wrote added to artifacts.
    On a trial only the declared plans at the declared mc and seed, with every other option at its default.
    copies N (one plan, not on a trial): N accounts copying the signal beside N independent ones
    (pfb.prop.copies); the row gains "copies". Nothing counts toward N; result.json is not rewritten."""
    if not _run_dir(run_id).is_dir():
        raise Refused("NoRunDirectory", f"no run directory {run_id} under {context().runs_dir.name}/")
    base = get_run(run_id)
    options = {"pool": pool, "block": block, "stress_ticks": stress_ticks, "generator": generator,
               "soft_daily_loss": soft_daily_loss, "root": root,
               "contracts_per_mini": contracts_per_mini, "copies": copies}  # fmt: skip
    rows, names = _run_prop(run_id, firm, plan, mc=mc, seed=seed, ledger=_ledger_path(), **options)
    doc = {
        **base,
        "prop": rows,
        "artifacts": list(dict.fromkeys([*base["artifacts"], *_artifacts(run_id, names)])),
    }
    validate(doc)
    return doc


def get_run(run_id: str) -> JsonDict:
    """The run's saved result.json, or (when absent: an open run, an older run) its contract built from the
    ledger and the artifacts; Refused (RunNotFound) when the ledger has no begin record for it."""
    run_dir = _run_dir(run_id)
    saved = _read_json(run_dir / RESULT_FILE)
    if saved is not None:
        validate(saved)
        return saved
    with _refusals():
        ledger = _ledger_path()
    lines = _ledger_lines(ledger, run_id)
    if lines.begin is None:
        raise Refused("RunNotFound", f"the ledger has no begin record for {run_id!r}")
    doc = _contract(lines, ledger, saved=False)
    validate(doc)
    return doc


def compare_runs(run_ids: Sequence[str]) -> JsonDict:
    """The headline keys of several runs side by side. Descriptive only: nothing reads it to choose a
    configuration (docs/TRIALS.md C4)."""
    ids = list(dict.fromkeys(run_ids))
    if not ids or len(ids) > MAX_COMPARE:
        raise Refused("BadRequest", f"compare 1 to {MAX_COMPARE} run ids, got {len(ids)}")
    runs = [get_run(run_id) for run_id in ids]
    kinds = sorted({r["kind"] for r in runs})
    warnings = [f"the runs mix kinds {kinds}"] if len(kinds) > 1 else []
    return {"schema": "pfb.compare/1", "columns": list(COMPARE_KEYS),
            "rows": [{k: r[k] for k in COMPARE_KEYS} for r in runs], "warnings": warnings}  # fmt: skip


def trial_count() -> JsonDict:
    """N and the registry's state (pfb.registry.api.status); reads, never writes."""
    from pfb.registry.api import status

    with _refusals():
        report = status(_registry_config())
    return {
        "schema": "pfb.trial_count/1", "n": report.n, "n_pe": report.n_pe, "n_local": report.n_local,
        "records": report.records, "pe_in_sync": report.pe_in_sync, "open_runs": list(report.open_runs),
        "hypotheses": list(report.hypotheses),
    }  # fmt: skip


def refusal_json(exc: Refused) -> JsonDict:
    """The contract of a refusal (CLI exit 2; MCP tool result)."""
    return exc.to_json()


def error_json(exc: BaseException) -> JsonDict:
    """The contract of an unexpected error (status "error"; CLI exit 1)."""
    return refusal(type(exc).__name__, str(exc), status="error")
