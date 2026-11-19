"""pfb.app's read-only functions: instruments, firms, the template, validate_strategy, trial_count."""

from __future__ import annotations

import json
from typing import Any

from app_env import TEMPLATE, AppEnv  # pyright: ignore[reportMissingImports]

from pfb import app
from pfb.instruments.table import load_instruments


def _json_ready(doc: dict[str, Any]) -> None:
    assert json.loads(json.dumps(doc)) == doc


def test_list_instruments_names_every_root() -> None:
    doc = app.list_instruments()
    _json_ready(doc)
    assert [r["root"] for r in doc["instruments"]] == list(load_instruments().roots())
    mnq = next(r for r in doc["instruments"] if r["root"] == "MNQ")
    assert mnq["contract_type"] == "micro" and doc["sha256"] == load_instruments().sha256


def test_list_firms_lists_every_plan_file() -> None:
    doc = app.list_firms()
    _json_ready(doc)
    files = sorted(
        p.relative_to(app.DEFAULT_FIRMS_DIR).as_posix() for p in app.DEFAULT_FIRMS_DIR.glob("*/*.toml")
    )
    assert sorted(r["file"] for r in doc["plans"]) == files
    row = next(r for r in doc["plans"] if r["plan"] == "combine_xfa_standard_50k")
    assert row["firm"] == "topstep" and row["error"] is None and len(row["sha256"]) == 64


def test_describe_strategy_template_returns_the_file() -> None:
    doc = app.describe_strategy_template()
    _json_ready(doc)
    assert doc["text"] == TEMPLATE.read_text() and doc["path"] == "strategies/_template.py"


def test_validate_strategy_and_trial_count_log_nothing(app_env: AppEnv) -> None:
    before = app_env.ledger.read_bytes()
    doc = app.validate_strategy(str(app_env.strategy))
    _json_ready(doc)
    assert doc["ok"] and doc["spec"]["hypothesis_id"] == "h_template" and doc["lookahead_signal"] == "seen"
    assert doc["params"] == {k: v["default"] for k, v in doc["spec"]["params"].items()}
    count = app.trial_count()
    assert count["n"] == 302 and count["open_runs"] == [] and count["pe_in_sync"]
    assert app_env.ledger.read_bytes() == before
