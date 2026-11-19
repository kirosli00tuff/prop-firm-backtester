"""RunPermit (ADR-001 A1): a real-data run needs the permit pfb.runner builds from its begin receipt.

The permit stops an uncounted run by accident, not by intent: no Python check stops a session that reads
parquet itself, which the project rules forbid.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date, datetime

PERMIT_KINDS = frozenset({"trial", "engine_check", "synthetic"})
SHA256_HEX_LEN = 64
# event_costs of the run "off" unless the trial's hypothesis says
# "on"; run_strategy refuses a permit whose event_costs differs from its own argument. The values are
# pfb.costs.events.EVENT_COSTS_OFF and EVENT_COSTS_ON (not imported: the permit stays dependency-free).
PERMIT_EVENT_COSTS = ("off", "on")


class PermitRequired(PermissionError):
    """A real-data run without a permit, or with one that differs from the call."""


def _is_sha256(text: object) -> bool:
    return (
        isinstance(text, str) and len(text) == SHA256_HEX_LEN and all(c in "0123456789abcdef" for c in text)
    )


def _is_day(value: object) -> bool:
    return isinstance(value, date) and not isinstance(value, datetime)


@dataclass(frozen=True)
class RunPermit:
    run_id: str
    kind: str
    record_sha256: str
    strategy_sha256: str
    root: str
    first: date
    last: date
    params_json: str
    fees: str
    cost_model: str
    event_costs: str = "off"

    def __post_init__(self) -> None:
        if self.event_costs not in PERMIT_EVENT_COSTS:
            raise PermitRequired(
                f"permit event_costs {self.event_costs!r} is not one of {PERMIT_EVENT_COSTS}"
            )
        for name in ("run_id", "root", "params_json", "fees", "cost_model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise PermitRequired(f"permit field {name} must be a non-empty string")
        if self.kind not in PERMIT_KINDS:
            raise PermitRequired(f"permit kind {self.kind!r} is not one of {sorted(PERMIT_KINDS)}")
        for name in ("record_sha256", "strategy_sha256"):
            if not _is_sha256(getattr(self, name)):
                raise PermitRequired(f"permit field {name} must be a lower-case sha256 hex digest")
        if not _is_day(self.first) or not _is_day(self.last):
            raise PermitRequired("permit first and last must be datetime.date values")


@dataclass(frozen=True)
class PermitRequest:
    """What the call asks for; the permit must agree with it in every field."""

    strategy_sha256: str
    root: str
    first: date
    last: date
    params_json: str
    fees: str
    cost_model: str
    event_costs: str = "off"


def check_permit(permit: RunPermit | None, request: PermitRequest) -> RunPermit:
    """Return the permit if it matches the request field for field, else raise PermitRequired."""
    if permit is None:
        raise PermitRequired("a real-data run needs a RunPermit from pfb.runner (ADR A1)")
    if not isinstance(permit, RunPermit):
        raise PermitRequired(f"permit must be a RunPermit, got {type(permit).__name__}")
    differing = [f.name for f in fields(PermitRequest) if getattr(permit, f.name) != getattr(request, f.name)]
    if differing:
        raise PermitRequired(f"the permit differs from the call in {differing}")
    return permit


__all__ = [
    "PERMIT_EVENT_COSTS",
    "PERMIT_KINDS",
    "PermitRequest",
    "PermitRequired",
    "RunPermit",
    "check_permit",
]
