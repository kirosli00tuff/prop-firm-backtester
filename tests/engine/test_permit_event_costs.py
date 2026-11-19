"""The RunPermit names event_costs (default "off") and
run_strategy refuses a permit whose event_costs differs from its own argument. Synthetic source with a
permit: the check runs before any cost, calendar or bar is loaded."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

import pfb.engine.run as R
from pfb.engine.permit import PermitRequest, PermitRequired, RunPermit, check_permit
from pfb.engine.strategy import load_strategy, resolve_params

REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / "strategies" / "reference_breakout.py"
FIRST, LAST = date(2025, 4, 1), date(2025, 4, 30)


def _permit(**changes: object) -> RunPermit:
    strategy = load_strategy(REFERENCE)
    base = {
        "run_id": "r1",
        "kind": "synthetic",
        "record_sha256": "a" * 64,
        "strategy_sha256": strategy.sha256,
        "root": strategy.spec.root,
        "first": FIRST,
        "last": LAST,
        "params_json": R.params_json(resolve_params(strategy.spec, None)),
        "fees": "topstep",
        "cost_model": "book",
    }
    return RunPermit(**(base | changes))  # type: ignore[arg-type]


def test_a_permit_defaults_to_off_and_refuses_an_unknown_value() -> None:
    assert _permit().event_costs == "off"
    assert _permit(event_costs="on").event_costs == "on"
    with pytest.raises(PermitRequired, match="event_costs"):
        _permit(event_costs="maybe")


def test_check_permit_compares_event_costs_like_every_other_field() -> None:
    permit = _permit(event_costs="on")
    request = PermitRequest(permit.strategy_sha256, permit.root, permit.first, permit.last,
                            permit.params_json, permit.fees, permit.cost_model, "on")  # fmt: skip
    assert check_permit(permit, request) is permit
    with pytest.raises(PermitRequired, match="event_costs"):
        check_permit(permit, replace(request, event_costs="off"))
    assert PermitRequest(*list(vars(request).values())[:7]).event_costs == "off"  # the old 7-field call


@pytest.mark.parametrize(("permit_says", "call_says"), [("off", "on"), ("on", "off")])
def test_run_strategy_refuses_a_permit_whose_event_costs_differs(permit_says: str, call_says: str) -> None:
    with pytest.raises(PermitRequired, match="event_costs"):
        R.run_strategy(REFERENCE, source="synthetic", first=FIRST, last=LAST, cost_model="book",
                       permit=_permit(event_costs=permit_says), event_costs=call_says)  # fmt: skip


def test_an_off_permit_still_runs_an_off_call() -> None:
    result = R.run_strategy(REFERENCE, source="synthetic", first=FIRST, last=LAST, cost_model="fixed:1",
                            permit=_permit(cost_model="fixed:1"))  # fmt: skip
    assert result.provenance["permit_run_id"] == "r1" and result.summary["event_windows"] == "off"
