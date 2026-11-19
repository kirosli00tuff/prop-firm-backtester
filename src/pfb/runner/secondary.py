"""Outputs of a trial computed after its result record (docs/TRIALS.md C4, C6, section 6), and the DSR grid.

prop_secondary: the plans the hypothesis's [secondary] declares, at its prop_mc and prop_seed, on the run's
saved frames; it writes runs/<run_id>/prop/<firm>_<plan>_<size>_<seed>.json per plan run and
runs/<run_id>/prop/index.json (the rows of the result contract's prop list). Nothing reads a prop figure to
choose a configuration. grid_dsr: the summary's DSR fields recomputed with the per-day Sharpes of every "ok"
configuration of the hypothesis (A14 G8), so the grid variance is live; the engine's own summary used only the
floor variance.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
FIRMS_DIR = REPO_ROOT / "config" / "firms"
PROP_DIR = "prop"
INDEX_NAME = "index.json"
DSR_KEYS = (
    "dsr", "dsr_floor", "dsr_grid", "dsr_variance", "dsr_variance_floor", "dsr_variance_grid", "dsr_bracket",
    "dsr_sr0_daily", "minbtl_years", "n_trials", "dsr_n_grid", "metrics_warnings",
)  # fmt: skip


def grid_dsr(result: Any, n_trials: int, grid: Sequence[float]) -> dict[str, Any]:
    """The DSR fields of metrics.core.summarize over the run's frames with grid_daily_sharpes = grid."""
    from pfb.metrics.core import summarize

    fresh = summarize(
        result.trades, result.equity_minute, result.daily, n_trials=n_trials, grid_daily_sharpes=list(grid)
    )
    return {key: fresh[key] for key in DSR_KEYS if key in fresh}


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n")


def prop_secondary(run_dir: Path, root: str, secondary: Mapping[str, Any]) -> dict[str, Any]:
    """Run the declared plans on the run's day paths; write the per-plan files and the index; return it."""
    from pfb.instruments.table import load_instruments
    from pfb.prop.days import day_paths_from_run
    from pfb.prop.matrix import (
        MINIS_PER_TYPE,
        SECONDARY_SCHEMA,
        McSettings,
        declared_plan_paths,
        run_matrix,
        secondary_row,
    )

    plans = list(secondary["prop_plans"])
    index: dict[str, Any] = {
        "schema": SECONDARY_SCHEMA, "run_id": run_dir.name, "declared": plans,
        "prop_mc": secondary["prop_mc"], "prop_seed": secondary["prop_seed"], "rows": [],
    }  # fmt: skip
    if plans:
        per_mini = MINIS_PER_TYPE[load_instruments().get(root).contract_type]
        days = day_paths_from_run(run_dir, contracts_per_mini=per_mini)
        settings = McSettings(n_paths=secondary["prop_mc"], seed=secondary["prop_seed"], kind="trial")
        rows = run_matrix(days, declared_plan_paths(FIRMS_DIR, plans), settings, repo_root=REPO_ROOT)
        for row in rows:
            if row.record is not None:
                name = f"{row.firm}_{row.plan}_{row.size}_{settings.seed}.json"
                _write_json(run_dir / PROP_DIR / name, {**row.record, "run_id": run_dir.name})
        index["rows"] = [secondary_row(row) for row in rows]
    _write_json(run_dir / PROP_DIR / INDEX_NAME, index)
    return index
