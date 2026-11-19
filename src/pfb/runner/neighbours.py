"""Neighbourhood runs (docs/TRIALS.md C3, S5): the centre and its one-step neighbours, each its own run.

For a trial each neighbour is a counted trial inside the declared grid and budget. Before anything is begun
the whole set is planned (lint, SPEC-grid params, look-ahead, data files and config ids); when the budget
cannot cover the configurations not yet counted, or a planned configuration's step-1 check saw no signal and
the hypothesis gives no no_signal_reason, the whole request is refused and nothing is begun. The neighbours
are begun and run first (grid key order, down before up) and the centre LAST, so the centre's N and DSR grid
include its own neighbours. A
trial runs its neighbourhood only when its hypothesis declares [suite] neighbourhood = true (C4). A synthetic
run takes its grid from the SPEC and counts nothing. An engine_check has no neighbours. The verdict
(pfb.analysis.neighbourhood) is written to runs/<centre run_id>/neighbourhood.json; it is an output of the
centre configuration and chooses nothing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from pfb.analysis.neighbourhood import neighbours, verdict
from pfb.registry.api import RegistryConfig, RunRequest, checkpoint, plan_configurations, read_state
from pfb.registry.errors import NoSignalUnexplained, SuiteMismatch
from pfb.registry.hypfile import load_hypothesis
from pfb.registry.records import SIGNAL_NONE, SYNTHETIC, TRIAL
from pfb.runner.api import (
    DEFAULT_RUNS_DIR,
    RunOutcome,
    RunRefused,
    execute_planned,
    prepare,
    run_args,
)
from pfb.runner.deps import RunnerDeps

NEIGHBOURHOOD_SCHEMA = "pfb.neighbourhood/1"
NEIGHBOURHOOD_FILE = "neighbourhood.json"
NEIGHBOUR_KINDS = (TRIAL, SYNTHETIC)


@dataclass(frozen=True)
class NeighbourhoodOutcome:
    centre: RunOutcome
    neighbours: tuple[RunOutcome, ...]
    verdict: Mapping[str, Any]
    document: Mapping[str, Any]


def _grid(kind: str, path: Path, plan: Any, deps: RunnerDeps) -> dict[str, list[object]]:
    if kind == TRIAL:
        hyp = load_hypothesis(plan.hypothesis_path)
        if not hyp.suite.neighbourhood:
            raise SuiteMismatch(
                f"{hyp.id!r} declares neighbourhood = false; its trials run without neighbours"
            )
        return {k: list(v) for k, v in hyp.grid.items()}
    spec = deps.load_strategy(path).spec
    return {name: list(p.grid) for name, p in spec.params.items()}


def _document(
    kind: str, centre: RunOutcome, around: Sequence[RunOutcome], found: Mapping[str, Any]
) -> dict[str, Any]:
    """neighbourhood.json: neighbours in run order; n_before and n_after bound the whole set."""
    first = around[0] if around else centre
    return {
        "schema": NEIGHBOURHOOD_SCHEMA,
        "kind": kind,
        "centre": {"run_id": centre.begin_record["run_id"], "params": centre.begin_record["params"]},
        "neighbours": [
            {"run_id": o.begin_record["run_id"], "params": o.begin_record["params"]} for o in around
        ],
        "n_before": first.begin_record["n_before"],
        "n_after": centre.begin_record["n_after"],
        "verdict": dict(found),
    }


def _refuse_unexplained_no_signal(requests: Sequence[RunRequest], config: RegistryConfig) -> None:
    """C3: a planned trial whose step-1 check saw no signal, under a hypothesis with no
    no_signal_reason, refuses the whole set before any begin (each begin would refuse it on its own)."""
    silent = [r for r in requests if r.lookahead_signal == SIGNAL_NONE]
    if not silent:
        return
    hyp_id = silent[0].hypothesis_id
    hyp = read_state(config).hypotheses.get(hyp_id)
    if hyp is not None and str(hyp["no_signal_reason"]).strip():
        return
    shown = ", ".join(json.dumps(dict(r.params), sort_keys=True) for r in silent)
    raise NoSignalUnexplained(
        f"the step-1 look-ahead check saw no signal for {shown} and {hyp_id!r} gives no no_signal_reason: "
        "the whole set is refused and nothing is begun"
    )


def run_with_neighbours(
    strategy_path: Path | str,
    *,
    kind: str,
    root: str | None = None,
    params: Mapping[str, Any] | None = None,
    fees: str | None = None,
    cost_model: str | None = None,
    seed: int | None = None,
    deps: RunnerDeps | None = None,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    hypothesis_path: Path | None = None,
    window: int | None = None,
    first: date | None = None,
    last: date | None = None,
    suite: Any = None,
) -> NeighbourhoodOutcome:
    """Plan the set, refuse it as a whole or run each neighbour then the centre; write neighbourhood.json."""
    if kind not in NEIGHBOUR_KINDS:
        raise RunRefused(f"a {kind} run has no neighbours (only {list(NEIGHBOUR_KINDS)})")
    path = Path(strategy_path).resolve()
    args = run_args(
        root=root, first=first, last=last, fees=fees, cost_model=cost_model,
        hypothesis_path=hypothesis_path, window=window, suite=suite,
    )  # fmt: skip
    args["allow_neighbourhood"] = True
    centre = prepare(kind, path, args, params, seed, deps)
    deps, config, plan, _ = centre
    sets = neighbours(_grid(kind, path, plan, deps), plan.params)
    planned = [prepare(kind, path, args, p, seed, deps) for p in sets] + [centre]  # the centre last
    if kind == TRIAL:
        checkpoint(config)
        plan_configurations([p[3] for p in planned], config=config)
        _refuse_unexplained_no_signal([p[3] for p in planned], config)
    outcomes = [execute_planned(kind, path, p, seed, runs_dir) for p in planned]
    middle, around = outcomes[-1], outcomes[:-1]
    found = verdict(middle.result.summary, [o.result.summary for o in around])
    document = _document(kind, middle, around, found)
    target = runs_dir / middle.begin_record["run_id"] / NEIGHBOURHOOD_FILE
    target.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    return NeighbourhoodOutcome(middle, tuple(around), found, document)
