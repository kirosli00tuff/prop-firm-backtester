"""What the dashboard shows, read from the ledger and from runs/<run_id>/ (no Streamlit here, no writes).

The ledger is read through pfb.registry.api.read_state (chain and anchor checked) and status (upstream sync
state);
the artifacts are the files the runner and `pfb prop` write (summary.json, suite.json, result.json,
neighbourhood.json, the parquet frames, prop/*.json, prop_matrix.json). Every artifact is optional: a page
shows what is missing instead of failing. The configuration comes from four environment variables that
`pfb dash` sets (PFB_DASH_REPO, PFB_DASH_LEDGER, PFB_DASH_RUNS, PFB_DASH_PE); unset, they name this
repository, its ledger, its runs/ and the optional upstream checkout of config/paths.toml.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from pfb.data.errors import PathsConfigError
from pfb.data.paths import DEFAULT_PATHS_TOML, load_paths
from pfb.metrics.stats import bracket
from pfb.registry.api import LEDGER_REL, REPO_ROOT, RegistryConfig, read_state, status

ENV_REPO, ENV_LEDGER, ENV_RUNS, ENV_PE = "PFB_DASH_REPO", "PFB_DASH_LEDGER", "PFB_DASH_RUNS", "PFB_DASH_PE"
TRADING_DAYS = 252
DSR_YEARS = 10
STATUS_OPEN = "open"
NO_RESULT_NOTE = "no result record: the run was begun and never finished (open or crashed)"
SUMMARY, SUITE, RESULT = "summary.json", "suite.json", "result.json"
NEIGHBOURHOOD, MATRIX, PROP_DIR = "neighbourhood.json", "prop_matrix.json", "prop"
DAILY, MINUTE = "daily.parquet", "equity_minute.parquet"
EXTERNAL_PREFIX = "external:"


class LedgerUnreadable(RuntimeError):
    """The ledger could not be read, chain-verified or replayed (the reason is the registry's)."""


def _default_pe_root() -> Path | None:
    try:
        return load_paths(DEFAULT_PATHS_TOML).upstream_root
    except PathsConfigError:
        return None


@dataclass(frozen=True)
class DashConfig:
    """Where the repository (the ledger's git anchor), the ledger, the runs and upstream's checkout are."""

    repo_root: Path
    ledger_path: Path
    runs_dir: Path
    pe_root: Path | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> DashConfig:
        repo = Path(environ.get(ENV_REPO) or REPO_ROOT)
        pe = environ.get(ENV_PE)
        return cls(
            repo_root=repo,
            ledger_path=Path(environ.get(ENV_LEDGER) or repo / LEDGER_REL),
            runs_dir=Path(environ.get(ENV_RUNS) or repo / "runs"),
            pe_root=Path(pe) if pe else _default_pe_root(),
        )

    def registry_config(self) -> RegistryConfig:
        # read_state never looks at pe_root; status does (upstream sync), and is skipped when it is None
        return RegistryConfig(registry_path=self.ledger_path, pe_root=self.pe_root, repo_root=self.repo_root)


@dataclass(frozen=True)
class DsrBar:
    """bracket(N) and the daily SR0 at the floor variance 1 / (T - 1) of a 10-year run (T = 2520 days)."""

    n: int
    days: int
    bracket: float
    sr0_daily: float
    sr0_annual: float


def dsr_bar(n: int, years: int = DSR_YEARS) -> DsrBar:
    days = TRADING_DAYS * years
    b = bracket(n) if n >= 1 else 0.0
    sr0 = b / math.sqrt(days - 1)
    return DsrBar(n=n, days=days, bracket=b, sr0_daily=sr0, sr0_annual=sr0 * math.sqrt(TRADING_DAYS))


@dataclass(frozen=True)
class RunRow:
    run_id: str
    kind: str
    hypothesis_id: str
    purpose: str | None
    status: str
    time_utc: str
    root: str
    params: str
    n_at_run: int
    trades: int | None
    net_cents: int | None
    sharpe_annual: float | None


@dataclass(frozen=True)
class LedgerView:
    n: int
    n_pe: int
    records: int
    open_runs: tuple[str, ...]
    pe_in_sync: bool | None
    pe_note: str
    runs: tuple[RunRow, ...]
    bar: DsrBar
    begun: Mapping[str, Mapping[str, Any]]
    results: Mapping[str, Mapping[str, Any]]


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return None if isinstance(value, float) and not math.isfinite(value) else value


def _int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _row(begin: Mapping[str, Any], result: Mapping[str, Any] | None, hyps: Mapping[str, Any]) -> RunRow:
    metrics = result.get("metrics") if result else None
    metrics = metrics if isinstance(metrics, Mapping) else {}
    hyp = hyps.get(begin.get("hypothesis_id", ""))
    sharpe = _number(metrics.get("sharpe_annual"))
    return RunRow(
        run_id=begin["run_id"],
        kind=begin["kind"],
        hypothesis_id=str(begin.get("hypothesis_id", "")),
        purpose=hyp.get("purpose") if isinstance(hyp, Mapping) else None,
        status=str(result["status"]) if result else STATUS_OPEN,
        time_utc=str(begin.get("time_utc", "")),
        root=str(begin.get("root", "")),
        params=json.dumps(begin.get("params", {}), sort_keys=True),
        n_at_run=int(begin["n_after"]),
        trades=_int(metrics.get("n_trades")),
        net_cents=_int(metrics.get("net_cents")),
        sharpe_annual=None if sharpe is None else float(sharpe),
    )


def _pe_sync(reg: RegistryConfig, pe_root: Path | None) -> tuple[bool | None, str]:
    try:
        report = status(reg)
    except Exception as exc:  # shown on the page; the ledger itself was read
        return None, f"upstream sync state unknown: {type(exc).__name__}: {exc}"
    if pe_root is None:
        return True, "no upstream ledger"
    changed = ", ".join(report.pe_changed)
    if report.pe_in_sync:
        return True, "upstream ledger in sync"
    return False, f"upstream ledger changed: {changed}"


def ledger_view(config: DashConfig) -> LedgerView:
    """N, open runs, the DSR bar and one row per run (newest first); LedgerUnreadable when the ledger is."""
    reg = config.registry_config()
    try:
        state = read_state(reg)
    except Exception as exc:  # any failure to read or verify the ledger is shown, never hidden
        raise LedgerUnreadable(f"{config.ledger_path}: {type(exc).__name__}: {exc}") from exc
    in_sync, note = _pe_sync(reg, config.pe_root)
    rows = [_row(b, state.results.get(r), state.hypotheses) for r, b in state.begun.items()]
    rows.sort(key=lambda row: (row.time_utc, row.run_id), reverse=True)
    return LedgerView(
        n=state.n,
        n_pe=state.n_pe,
        records=state.records,
        open_runs=state.open_runs,
        pe_in_sync=in_sync,
        pe_note=note,
        runs=tuple(rows),
        bar=dsr_bar(state.n),
        begun=state.begun,
        results=state.results,
    )


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    directory: Path
    begin: Mapping[str, Any]
    result_record: Mapping[str, Any] | None
    expected: tuple[str, ...]
    present: tuple[str, ...]
    missing: tuple[str, ...]
    summary: Mapping[str, Any] | None
    suite: Mapping[str, Any] | None
    result: Mapping[str, Any] | None
    neighbourhood: Mapping[str, Any] | None
    prop: tuple[Mapping[str, Any], ...]
    prop_matrix: Mapping[str, Any] | None
    errors: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def metrics(self) -> Mapping[str, Any] | None:
        """summary.json, else the result record's metrics (the same summary, as logged)."""
        if self.summary is not None:
            return self.summary
        logged = self.result_record.get("metrics") if self.result_record else None
        return logged if isinstance(logged, Mapping) else None


def _expected(result: Mapping[str, Any] | None, run_id: str) -> tuple[str, ...]:
    names: list[str] = []
    for entry in (result or {}).get("artifacts") or ():
        text = str(entry).removeprefix(EXTERNAL_PREFIX)
        marker = f"{run_id}/"
        names.append(text.split(marker, 1)[1] if marker in text else Path(text).name)
    return tuple(names)


def _read_json(path: Path, errors: list[str]) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
        return None
    if not isinstance(value, Mapping):
        errors.append(f"{path.name}: not a JSON object")
        return None
    return value


def _notes(result: Mapping[str, Any] | None) -> tuple[str, ...]:
    if result is None:
        return (NO_RESULT_NOTE,)
    if result.get("status") != "ok":
        return (f"the run failed: {result.get('error')}",)
    return ()


def _suite(
    suite_file: Mapping[str, Any] | None, metrics: Mapping[str, Any] | None
) -> Mapping[str, Any] | None:
    if suite_file is not None:
        return suite_file
    inner = metrics.get("suite") if metrics else None
    return inner if isinstance(inner, Mapping) else None


def run_artifacts(config: DashConfig, run_id: str, view: LedgerView | None = None) -> RunArtifacts:
    """Everything under runs/<run_id>/ that the pages show; KeyError for a run the ledger does not name."""
    view = view or ledger_view(config)
    begin = view.begun[run_id]
    result = view.results.get(run_id)
    directory = config.runs_dir / run_id
    errors: list[str] = []
    expected = _expected(result, run_id)
    present = tuple(sorted(p.name for p in directory.iterdir())) if directory.is_dir() else ()
    summary = _read_json(directory / SUMMARY, errors)
    prop_dir = directory / PROP_DIR
    prop_files = sorted(prop_dir.glob("*.json")) if prop_dir.is_dir() else []
    prop = tuple(doc for doc in (_read_json(p, errors) for p in prop_files) if doc is not None)
    logged = result.get("metrics") if result else None
    metrics = summary or (logged if isinstance(logged, Mapping) else None)
    return RunArtifacts(
        run_id=run_id,
        directory=directory,
        begin=begin,
        result_record=result,
        expected=expected,
        present=present,
        missing=tuple(name for name in expected if not (directory / name).is_file()),
        summary=summary,
        suite=_suite(_read_json(directory / SUITE, errors), metrics),
        result=_read_json(directory / RESULT, errors),
        neighbourhood=_read_json(directory / NEIGHBOURHOOD, errors),
        prop=prop,
        prop_matrix=_read_json(directory / MATRIX, errors),
        errors=tuple(errors),
        notes=_notes(result),
    )


def load_frame(run_dir: Path, name: str) -> pl.DataFrame | None:
    path = run_dir / f"{name}.parquet"
    return pl.read_parquet(path) if path.is_file() else None


@dataclass(frozen=True)
class EquityCurve:
    source: str
    frame: pl.DataFrame  # trade_date, equity_cents (end of day), drawdown_cents (<= 0, from the running peak)


def _with_drawdown(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        (pl.col("equity_cents") - pl.col("equity_cents").cum_max()).alias("drawdown_cents")
    )


def equity_curve(run_dir: Path) -> EquityCurve | None:
    """End-of-day equity and drawdown from daily.parquet, else equity_minute.parquet; None without both."""
    daily = run_dir / DAILY
    if daily.is_file():
        frame = pl.read_parquet(daily, columns=["trade_date", "eq_close"])
        return EquityCurve(DAILY, _with_drawdown(frame.rename({"eq_close": "equity_cents"})))
    minute = run_dir / MINUTE
    if minute.is_file():
        frame = (
            pl.scan_parquet(minute)
            .select("trade_date", "eq_close")
            .group_by("trade_date", maintain_order=True)
            .agg(pl.col("eq_close").last().alias("equity_cents"))
            .collect()
        )
        return EquityCurve(MINUTE, _with_drawdown(frame))
    return None


def _out(outputs: Mapping[str, Any], key: str, part: str = "value") -> Any:
    value = outputs.get(key)
    return value.get(part) if isinstance(value, Mapping) else None


def prop_row(doc: Mapping[str, Any]) -> dict[str, Any]:
    """One plan's pass probability, days to pass and EV per purchase (with their Monte Carlo SEs)."""
    outputs = doc.get("outputs")
    outputs = outputs if isinstance(outputs, Mapping) else {}
    return {
        "firm": doc.get("firm"),
        "plan": doc.get("plan"),
        "size": doc.get("size"),
        "status": doc.get("status", "ok"),
        "paths": doc.get("n_paths"),
        "seed": doc.get("seed"),
        "stress_ticks": doc.get("stress_ticks"),
        "p_pass": _out(outputs, "p_pass"),
        "p_pass_se": _out(outputs, "p_pass", "se"),
        "days_to_pass_median": _out(outputs, "days_to_pass_median"),
        "days_to_pass_p90": _out(outputs, "days_to_pass_p90"),
        "ev_per_purchase_cents": _out(outputs, "ev_per_purchase"),
        "ev_se_cents": _out(outputs, "ev_per_purchase", "se"),
        "cost_per_attempt_cents": _out(outputs, "cost_per_attempt"),
        "funded_ev_cents": _out(outputs, "funded_ev"),
        "p_breach_before_first_payout": _out(outputs, "p_breach_before_first_payout"),
    }


def prop_rows(docs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [prop_row(doc) for doc in docs]


def matrix_rows(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per plan of prop_matrix.json; a skipped or refused plan keeps its status and reason."""
    rows: list[dict[str, Any]] = []
    for entry in matrix.get("rows") or ():
        record = entry.get("record")
        base = prop_row(record) if isinstance(record, Mapping) else prop_row({})
        rows.append(
            {
                **base,
                "firm": entry.get("firm"),
                "plan": entry.get("plan"),
                "size": entry.get("size"),
                "status": entry.get("status"),
                "reason": entry.get("reason"),
            }
        )
    return rows


def matrix_pivot(rows: Sequence[Mapping[str, Any]], metric: str) -> pl.DataFrame:
    """firm by plan: one row per firm, one column per plan, the metric in each cell (null where absent)."""
    plans = sorted({str(r["plan"]) for r in rows})
    firms = sorted({str(r["firm"]) for r in rows})
    cells = {(str(r["firm"]), str(r["plan"])): _number(r.get(metric)) for r in rows}
    data: dict[str, list[Any]] = {"firm": firms}
    for plan in plans:
        data[plan] = [cells.get((firm, plan)) for firm in firms]
    return pl.DataFrame(data, schema={"firm": pl.String, **dict.fromkeys(plans, pl.Float64)})


def fmt_cents(value: object) -> str:
    """Integer cents as dollars ("$1,234.56", "-$0.05"); a float is rounded to the cent; None or NaN "n/a"."""
    number = _number(value)
    if number is None:
        return "n/a"
    cents = number if isinstance(number, int) else round(number)
    sign = "-" if cents < 0 else ""
    return f"{sign}${abs(cents) // 100:,}.{abs(cents) % 100:02d}"


def fmt_num(value: object, digits: int = 3) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:,.{digits}f}"


# ------------------------------------------------------------------ item 5: results and Runs ----
LEDGER_RESULT, LEDGER_HYPOTHESIS = "result", "hypothesis"
PROP_INDEX = "index.json"
PASS_END = "pass"
NOT_RECORDED = "not recorded"


@dataclass(frozen=True)
class ResultDoc:
    """The run's result contract and the bytes the JSON export hands out: runs/<id>/result.json as saved, or,
    when the file is absent (an open run, a run saved before pfb.app), the contract built from the ledger and
    the artifacts exactly as pfb.app.get_run builds it (pfb.contract.build_result; nothing is written)."""

    contract: Mapping[str, Any]
    data: bytes
    saved: bool


def _ledger_lines(
    ledger: Path, run_id: str, hypothesis_id: str | None
) -> tuple[dict[str, tuple[Mapping[str, Any], str]], Mapping[str, Any] | None]:
    """The run's begin and result records with each line's sha256, and the trial's hypothesis record."""
    found: dict[str, tuple[Mapping[str, Any], str]] = {}
    hypothesis: Mapping[str, Any] | None = None
    for raw in ledger.read_bytes().splitlines():
        hit_run = run_id.encode() in raw
        hit_hyp = hypothesis_id is not None and hypothesis is None and hypothesis_id.encode() in raw
        if not (hit_run or hit_hyp):
            continue
        record = json.loads(raw)
        if hit_run and record.get("run_id") == run_id:
            part = LEDGER_RESULT if record.get("kind") == LEDGER_RESULT else "begin"
            found.setdefault(part, (record, hashlib.sha256(raw).hexdigest()))
        elif record.get("kind") == LEDGER_HYPOTHESIS and record.get("hypothesis_id") == hypothesis_id:
            hypothesis = record
    return found, hypothesis


def _trial_prop(directory: Path) -> tuple[list[Mapping[str, Any]] | None, list[str]]:
    index = _read_json(directory / PROP_DIR / PROP_INDEX, [])
    if index is None or not index.get("declared"):
        return None, []
    files = sorted(p.name for p in (directory / PROP_DIR).glob("*.json") if p.name != PROP_INDEX)
    return list(index["rows"]), [f"{PROP_DIR}/{PROP_INDEX}", *(f"{PROP_DIR}/{f}" for f in files)]


def _artifact_names(config: DashConfig, run_id: str, names: Sequence[str]) -> list[str]:
    """Repo-relative names inside the repository, "external:<runs dir>/<run_id>/<file>" outside it."""
    out = (config.runs_dir / run_id).resolve()
    try:
        base = out.relative_to(config.repo_root.resolve()).as_posix()
    except ValueError:
        base = f"{EXTERNAL_PREFIX}{config.runs_dir.name}/{run_id}"
    return [f"{base}/{name}" for name in names]


def built_contract(config: DashConfig, art: RunArtifacts) -> Mapping[str, Any]:
    """The contract of a run without result.json, from its ledger lines and artifacts (pfb.app.get_run's)."""
    from pfb.contract import build_result

    is_trial = art.begin.get("kind") == "trial"
    hyp_id = str(art.begin.get("hypothesis_id")) if is_trial else None
    lines, hypothesis = _ledger_lines(config.ledger_path, art.run_id, hyp_id)
    begin, result = lines.get("begin"), lines.get(LEDGER_RESULT)
    doc = art.neighbourhood
    centre = doc.get("centre") if doc is not None else None
    neighbourhood = doc if isinstance(centre, Mapping) and centre.get("run_id") == art.run_id else None
    names = [NEIGHBOURHOOD] if neighbourhood is not None else []
    prop, prop_names = _trial_prop(art.directory) if is_trial else (None, [])
    return build_result(
        begin=begin[0] if begin else art.begin,
        result=result[0] if result else None,
        begin_sha256=begin[1] if begin else None,
        result_sha256=result[1] if result else None,
        hypothesis=hypothesis,
        neighbourhood=neighbourhood,
        prop=prop,
        artifacts=_artifact_names(config, art.run_id, [*names, *prop_names]) if names or prop_names else [],
    )


def result_doc(config: DashConfig, art: RunArtifacts) -> ResultDoc | None:
    """result.json's bytes as saved, else the built contract; None when result.json exists but is unreadable
    (the reason is in art.errors)."""
    path = art.directory / RESULT
    if art.result is not None:
        return ResultDoc(contract=art.result, data=path.read_bytes(), saved=True)
    if path.is_file():
        return None
    contract = built_contract(config, art)
    return ResultDoc(contract=contract, data=(json.dumps(contract, indent=2) + "\n").encode(), saved=False)


def main_end(ends: object) -> str:
    """The commonest end other than a pass with its share of all ends ("breach 100%"); "none" when every
    end passed, NOT_RECORDED without counts. A breach is a drawdown or hard daily-loss breach (pfb.prop)."""
    if not isinstance(ends, Mapping) or not ends:
        return NOT_RECORDED
    counts = {str(k): v for k, v in ends.items() if _int(v) is not None}
    total = sum(counts.values())
    failing = sorted(((-v, k) for k, v in counts.items() if k != PASS_END and v > 0))
    if not failing or total == 0:
        return "none"
    count, name = -failing[0][0], failing[0][1]
    return f"{name} {count / total:.0%}"


def _ends_text(ends: object) -> str:
    if not isinstance(ends, Mapping) or not ends:
        return NOT_RECORDED
    return ", ".join(f"{k} {v}" for k, v in sorted(ends.items()))


def plan_result_row(doc: Mapping[str, Any]) -> dict[str, Any]:
    """One plan of prop/*.json for the results view: pass rate, EV per purchase, the payout count (when the
    output records one), the main eval breach reason and the funded ends."""
    base = prop_row(doc)
    payouts = _int(doc.get("payout_count"))
    return {
        "firm": base["firm"],
        "plan": base["plan"],
        "size": base["size"],
        "status": base["status"],
        "pass_rate": base["p_pass"],
        "pass_rate_se": base["p_pass_se"],
        "ev_per_purchase": fmt_cents(base["ev_per_purchase_cents"]),
        "ev_se": fmt_cents(base["ev_se_cents"]),
        "payout_count": NOT_RECORDED if payouts is None else str(payouts),
        "main_breach_reason": main_end(doc.get("eval_ends")),
        "funded_ends": _ends_text(doc.get("funded_ends")),
        "paths": base["paths"],
    }


@dataclass(frozen=True)
class RunListRow:
    """One row of the Runs page: date, strategy, kind, root, net, trades, verdict (and the run id)."""

    run_id: str
    date: str
    strategy: str
    kind: str
    root: str
    net_cents: int | None
    trades: int | None
    verdict: str


def run_verdict(status: str, contract: Mapping[str, Any] | None) -> str:
    """The run's status ("open", "failed") when it has no ok result; else its neighbourhood verdict (plateau,
    spike, mixed, n/a) when it is a neighbourhood centre, else "ok"."""
    if status != "ok":
        return status
    part = contract.get("neighbourhood") if contract else None
    found = part.get("verdict") if isinstance(part, Mapping) else None
    return str(found) if found is not None else status


def _neighbourhood_verdict(config: DashConfig, run_id: str) -> Mapping[str, Any] | None:
    """result.json's neighbourhood part, else neighbourhood.json's verdict (no error is collected here)."""
    directory = config.runs_dir / run_id
    saved = _read_json(directory / RESULT, [])
    if saved is not None:
        return saved
    doc = _read_json(directory / NEIGHBOURHOOD, [])
    return {"neighbourhood": doc.get("verdict")} if doc is not None else None


def run_list(config: DashConfig, view: LedgerView) -> list[RunListRow]:
    """Every run the ledger names, newest first, with its strategy file and verdict."""
    rows: list[RunListRow] = []
    for row in view.runs:
        strategy = view.begun[row.run_id].get("strategy_path")
        rows.append(
            RunListRow(
                run_id=row.run_id,
                date=row.time_utc[:10],
                strategy=Path(str(strategy)).name if strategy else "",
                kind=row.kind,
                root=row.root,
                net_cents=row.net_cents,
                trades=row.trades,
                verdict=run_verdict(row.status, _neighbourhood_verdict(config, row.run_id)),
            )
        )
    return rows
