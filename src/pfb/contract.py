"""The result contract "pfb.result/1" (docs/TRIALS.md section 7; docs/result.schema.json).

build_result maps one run's records and artifacts onto exactly the keys of section 7 plus `ledger`
({begin_sha256, result_sha256}). Every value is read from a source, never filled from a default or from
another run; a part that was turned off or does not apply is null:

- identity (run_id, kind, hypothesis_id, strategy, root, window, params, n, n_added): the begin record;
  purpose: the trial's hypothesis record; status: the result record ("open" while it has none);
- the metrics: the result record's metrics (the run's summary), under the summary's names (CONTRACT_METRICS);
- cost_stress, baseline, walk_forward and the realized half of power: the summary's "suite" (suite.json);
- baseline: the four figures of the one mode the suite reports, named in `mode` (the largest p_net of
  uniform, tod and cost_matched, ties to cost_matched then tod; two modes; docs/TRIALS.md
  section 7);
- the stated half of power (a trial only): the begin record's power (n_used, min_edge_cents,
  target_edge_cents) and min_edge_cents_alpha recomputed by pfb.registry.powercheck.power_at from the
  hypothesis record's [power] over the run's own window at that n_used (a pure function of the records);
- neighbourhood: runs/<centre>/neighbourhood.json; prop: runs/<run_id>/prop/ (a trial's index.json, or the
  compact records a prop request wrote);
- warnings: the summary's warnings, the result record's, the runner's (a failed secondary output), one when
  every draw of the reported baseline mode has the same net, and one when a minimum edge is stated and the
  realized minimum exceeds it or is None (with the reason); artifacts: the result record's plus the files
  written after it; artifact_sha256: the result record's map of its own artifacts to their sha256 (null for an
  open run or a record written before the key; files written after it carry none).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

SCHEMA = "pfb.result/1"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "docs" / "result.schema.json"
JsonDict = dict[str, Any]

STATUS_OPEN = "open"
STATUS_REFUSED = "refused"
STATUS_ERROR = "error"
TRIAL = "trial"
CONTRACT_METRICS = {
    "net_cents": "net_cents",
    "trades": "n_trades",
    "win_rate": "win_rate",
    "profit_factor": "profit_factor",
    "expectancy_cents": "expectancy_cents",
    "sharpe": "sharpe_annual",
    "sortino": "sortino_annual",
    "max_drawdown_cents": "max_drawdown_cents",
    "psr": "psr",
    "dsr": "dsr",
    "minbtl_years": "minbtl_years",
}
KEYS = (
    "schema", "run_id", "kind", "status", "hypothesis_id", "purpose", "strategy", "root", "window", "params",
    "n", "n_added", *CONTRACT_METRICS, "cost_stress", "baseline", "power", "walk_forward", "neighbourhood",
    "prop", "warnings", "artifacts", "artifact_sha256", "ledger",
)  # fmt: skip
POWER_WARNING = "power: the realized minimum edge {real} cents exceeds the stated minimum {stated} cents"
POWER_NONE_WARNING = ("power: the realized minimum edge is None ({reason}); the stated minimum is {stated} "
                      "cents")  # fmt: skip
POWER_UNDETECTABLE = "undetectable at the run's trades, sd and days"
POWER_ABSENT = "the run has no realized power figures"
BASELINE_SD0_WARNING = ("baseline: every draw of the reported mode {mode} has the same net (draw sd 0); the "
                        "percentile is uninformative")  # fmt: skip


class ContractError(ValueError):
    """A result does not satisfy docs/result.schema.json."""


# ------------------------------------------------------------------------------------- parts ----
def _finite(value: Any) -> Any:
    """A JSON number as it is; a NaN or infinity (never in a ledger record) as None."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _metrics(result: Mapping[str, Any] | None) -> Mapping[str, Any]:
    metrics = result.get("metrics") if result is not None else None
    return metrics if isinstance(metrics, Mapping) else {}


def _suite_part(metrics: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    suite = metrics.get("suite")
    part = suite.get(name) if isinstance(suite, Mapping) else None
    return part if isinstance(part, Mapping) else None


def cost_stress_part(metrics: Mapping[str, Any]) -> JsonDict | None:
    """{"<k>": {net_cents, sharpe}} from the suite's cost_stress levels (sharpe = their sharpe_annual)."""
    part = _suite_part(metrics, "cost_stress")
    if not part:
        return None
    return {
        str(k): {"net_cents": v["net_cents"], "sharpe": _finite(v["sharpe_annual"])} for k, v in part.items()
    }


def _baseline_mode(part: Mapping[str, Any]) -> str | None:
    """The mode the suite's top-level baseline figures come from; a suite.json written before the
    baseline modes names it `headline` (always "uniform")."""
    mode = part.get("mode", part.get("headline"))
    return mode if isinstance(mode, str) else None


def baseline_part(metrics: Mapping[str, Any]) -> JsonDict | None:
    """{draws, mode, net_percentile, sharpe_percentile, p_net, p_sharpe}: the four figures of the one mode the
    suite reports (the largest p_net, ties to cost_matched then tod; pfb.analysis.baseline.reported_mode)."""
    part = _suite_part(metrics, "baseline")
    if part is None:
        return None
    keys = ("net_percentile", "sharpe_percentile", "p_net", "p_sharpe")
    return {"draws": part["draws"], "mode": _baseline_mode(part), **{k: _finite(part[k]) for k in keys}}


def walk_forward_part(metrics: Mapping[str, Any]) -> JsonDict | None:
    part = _suite_part(metrics, "walk_forward")
    if part is None:
        return None
    return {"folds": part["n_folds"], "sharpe_spread": _finite(part["sharpe_spread"]),
            "positive_folds": part["positive_folds"]}  # fmt: skip


def stated_power(begin: Mapping[str, Any], hypothesis: Mapping[str, Any] | None) -> JsonDict | None:
    """A trial's stated figures: the begin record's power, and min_edge_cents_alpha at its n_used over the
    run's OWN window (docs/TRIALS.md section 5: Y is one window, never a sum)."""
    power = begin.get("power")
    if begin.get("kind") != TRIAL or not isinstance(power, Mapping):
        return None
    alpha = None
    window = _window(begin)
    if hypothesis is not None and window is not None:
        from pfb.registry.powercheck import power_at

        alpha = power_at(hypothesis["power"], [window], int(power["n_used"])).min_edge_cents_alpha
    return {"min_edge_cents": power.get("min_edge_cents"), "min_edge_cents_alpha": alpha,
            "target_edge_cents": power.get("target_edge_cents")}  # fmt: skip


def power_part(metrics: Mapping[str, Any], stated: Mapping[str, Any] | None) -> JsonDict | None:
    """The stated half (trial) and the realized half (the suite's power); None when neither exists."""
    realized = _suite_part(metrics, "power")
    if realized is None and stated is None:
        return None
    s, r = stated or {}, realized or {}
    return {
        "min_edge_cents": s.get("min_edge_cents"),
        "min_edge_cents_alpha": s.get("min_edge_cents_alpha"),
        "target_edge_cents": s.get("target_edge_cents"),
        "realized_min_edge_cents": r.get("realized_min_edge_cents"),
        "realized_edge_cents": r.get("realized_edge_cents"),
        "passed": r.get("passed"),
    }


def neighbourhood_part(document: Mapping[str, Any] | None) -> JsonDict | None:
    """{verdict, ratio, run_ids} from a centre's neighbourhood.json."""
    if document is None:
        return None
    found = document["verdict"]
    return {"verdict": found["verdict"], "ratio": _finite(found["ratio"]),
            "run_ids": [n["run_id"] for n in document["neighbours"]]}  # fmt: skip


def _power_warning(power: Mapping[str, Any] | None, realized: Mapping[str, Any] | None) -> list[str]:
    """A warning when the realized minimum exceeds the stated one, or is None while a minimum is stated
    (undetectable, or not computable, with the suite's reason)."""
    if power is None:
        return []
    stated, real = power["min_edge_cents"], power["realized_min_edge_cents"]
    if not isinstance(stated, int):
        return []
    if real is None:
        reason = POWER_ABSENT if realized is None else str(realized.get("reason") or POWER_UNDETECTABLE)
        return [POWER_NONE_WARNING.format(reason=reason, stated=stated)]
    if isinstance(real, int) and real > stated:
        return [POWER_WARNING.format(real=real, stated=stated)]
    return []


def _baseline_warning(metrics: Mapping[str, Any]) -> list[str]:
    """A warning when every draw of the reported mode has the same net (draw sd 0)."""
    part = _suite_part(metrics, "baseline")
    if part is None:
        return []
    mode = _baseline_mode(part)
    found = part.get(mode) if mode is not None else None
    net = found.get("net") if isinstance(found, Mapping) else None
    sd = net.get("sd") if isinstance(net, Mapping) else None
    if isinstance(sd, int | float) and sd == 0:
        return [BASELINE_SD0_WARNING.format(mode=mode)]
    return []


def _window(begin: Mapping[str, Any]) -> list[str] | None:
    first, last = begin.get("first"), begin.get("last")
    return [first, last] if first and last else None


def _n_added(begin: Mapping[str, Any]) -> int | None:
    before, after = begin.get("n_before"), begin.get("n_after")
    return after - before if isinstance(before, int) and isinstance(after, int) else None


# ------------------------------------------------------------------------------------- build ----
def build_result(
    *,
    begin: Mapping[str, Any],
    result: Mapping[str, Any] | None,
    begin_sha256: str | None,
    result_sha256: str | None,
    hypothesis: Mapping[str, Any] | None = None,
    neighbourhood: Mapping[str, Any] | None = None,
    prop: Sequence[Mapping[str, Any]] | None = None,
    warnings: Sequence[str] = (),
    artifacts: Sequence[str] = (),
) -> JsonDict:
    """The contract of one run from its begin and result records (result None: the run is open), the trial's
    hypothesis record, the centre's neighbourhood.json, the prop rows, the runner's warnings and the files
    written after the result record."""
    metrics = _metrics(result)
    power = power_part(metrics, stated_power(begin, hypothesis))
    is_trial = begin.get("kind") == TRIAL
    purpose = hypothesis.get("purpose") if is_trial and hypothesis is not None else None
    record_artifacts = list(result.get("artifacts", ())) if result is not None else []
    out: JsonDict = {
        "schema": SCHEMA,
        "run_id": begin["run_id"],
        "kind": begin["kind"],
        "status": result.get("status", STATUS_OPEN) if result is not None else STATUS_OPEN,
        "hypothesis_id": begin.get("hypothesis_id"),
        "purpose": purpose,
        "strategy": begin.get("strategy_path"),
        "root": begin.get("root"),
        "window": _window(begin),
        "params": begin.get("params"),
        "n": begin.get("n_after"),
        "n_added": _n_added(begin),
        **{key: _finite(metrics.get(source)) for key, source in CONTRACT_METRICS.items()},
        "cost_stress": cost_stress_part(metrics),
        "baseline": baseline_part(metrics),
        "power": power,
        "walk_forward": walk_forward_part(metrics),
        "neighbourhood": neighbourhood_part(neighbourhood),
        "prop": [dict(row) for row in prop] if prop is not None else None,
        "warnings": [
            *metrics.get("warnings", ()),
            *(result.get("warnings", ()) if result is not None else ()),
            *warnings,
            *_baseline_warning(metrics),
            *_power_warning(power, _suite_part(metrics, "power")),
        ],
        "artifacts": list(dict.fromkeys([*record_artifacts, *artifacts])),
        "artifact_sha256": _artifact_shas(result),
        "ledger": {"begin_sha256": begin_sha256, "result_sha256": result_sha256},
    }
    return out


def _artifact_shas(result: Mapping[str, Any] | None) -> dict[str, str] | None:
    """The result record's artifact_sha256: None for an open run or a record written before
    the key existed."""
    shas = result.get("artifact_sha256") if result is not None else None
    return dict(shas) if isinstance(shas, Mapping) else None


def refusal(code: str, reason: str, *, status: str = STATUS_REFUSED) -> JsonDict:
    """The contract of a refusal (exit 2) or of an unexpected error (status "error")."""
    return {"schema": SCHEMA, "status": status, "error": {"code": code, "reason": reason}}


# ---------------------------------------------------------------------------------- validate ----
@cache
def _validator() -> Any:
    from jsonschema import Draft202012Validator

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate(result: Mapping[str, Any]) -> None:
    """Raise ContractError unless `result` satisfies docs/result.schema.json (a result or a refusal)."""
    from jsonschema.exceptions import best_match

    error = best_match(_validator().iter_errors(result))
    if error is not None:
        where = "/".join(str(p) for p in error.absolute_path) or "<top>"
        raise ContractError(f"schema error at {where}: {error.message}")
