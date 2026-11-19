"""Parsing of the two TOML inputs the registry checks runs against: hypothesis files and the
engine-check allowlist.

Hypothesis files (schema "pfb.hypothesis/2", docs/TRIALS.md section 2) are parsed in pfb.registry.hypfile.
Allowlist (ADR A2), schema "pfb.engine_checks/1": [[check]] with exactly id, declared_in, strategy,
strategy_sha256, root, first, last, fees, cost_model, session, seed (an int) and params (a table of scalars).
session, seed and params were added later: registry.api.begin refuses an engine_check whose
session, seed or params differ from its entry. Begin records written before that hold the nine legacy keys.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any

from pfb.registry.chain import sha256_hex
from pfb.registry.errors import EngineCheckNotDeclared

ENGINE_CHECKS_SCHEMA = "pfb.engine_checks/1"
LEGACY_CHECK_KEYS = frozenset(
    {"id", "declared_in", "strategy", "strategy_sha256", "root", "first", "last", "fees", "cost_model"}
)
CHECK_KEYS = LEGACY_CHECK_KEYS | {"session", "seed", "params"}
EVENT_COSTS_OFF = "off"
EVENT_COSTS_ON = "on"
EVENT_COSTS_VALUES = (EVENT_COSTS_OFF, EVENT_COSTS_ON)
# "on" is priced by pfb.costs.events, charged per bar by the engine and identified,
# begun, recorded and replayed by the runner and registry: accepted at registration. False
# refuses "on" again at registration (the engine and replay keep accepting what was already recorded).
EVENT_COSTS_ENGINE_WIRED = True
EVENT_COSTS_NOT_WIRED = (
    'event_costs "on" is refused until the engine charges event-window costs per bar '
    "(pfb.costs.events.EventCostModel.bar_slippage_cents) and the runner identifies the cost model with them"
)
SHA_RE = re.compile(r"[0-9a-f]{64}")
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
GridValue = str | int | float | bool


def iso_date(value: object, what: str, error: type[Exception]) -> str:
    """A TOML date or an ISO date string, returned as YYYY-MM-DD."""
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError as exc:
            raise error(f"{what} {value!r} is not an ISO date") from exc
    raise error(f"{what} is not a date")


def _text(raw: Mapping[str, Any], key: str, error: type[Exception], where: str) -> str:
    value = raw[key]
    if not isinstance(value, str) or not value:
        raise error(f"{where}: {key} must be a non-empty string")
    return value


def _sha(raw: Mapping[str, Any], key: str, error: type[Exception], where: str) -> str:
    value = _text(raw, key, error, where)
    if not SHA_RE.fullmatch(value):
        raise error(f"{where}: {key} is not a sha256 (64 lowercase hex)")
    return value


@dataclass(frozen=True)
class EngineCheckEntry:
    id: str
    declared_in: str
    strategy: str
    strategy_sha256: str
    root: str
    first: str
    last: str
    fees: str
    cost_model: str
    session: str
    seed: int
    params: Mapping[str, GridValue]

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "declared_in": self.declared_in,
            "strategy": self.strategy,
            "strategy_sha256": self.strategy_sha256,
            "root": self.root,
            "first": self.first,
            "last": self.last,
            "fees": self.fees,
            "cost_model": self.cost_model,
            "session": self.session,
            "seed": self.seed,
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class EngineChecks:
    sha256: str
    entries: Mapping[str, EngineCheckEntry]


def _check_entry(raw: object, where: str) -> EngineCheckEntry:
    if not isinstance(raw, dict) or set(raw) != CHECK_KEYS:
        raise EngineCheckNotDeclared(f"{where}: a [[check]] has exactly the keys {sorted(CHECK_KEYS)}")
    err = EngineCheckNotDeclared
    text_keys = ("id", "declared_in", "strategy", "root", "fees", "cost_model", "session")
    text = {k: _text(raw, k, err, where) for k in text_keys}
    seed = raw["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise err(f"{where}: seed must be an integer")
    return EngineCheckEntry(
        strategy_sha256=_sha(raw, "strategy_sha256", err, where),
        first=iso_date(raw["first"], f"{where}: first", err),
        last=iso_date(raw["last"], f"{where}: last", err),
        seed=seed,
        params=_check_params(raw["params"], where),
        **text,
    )


def _check_params(raw: object, where: str) -> MappingProxyType[str, GridValue]:
    """The entry's params: a table of scalars (the SPEC defaults the check runs at), read-only."""
    if not isinstance(raw, dict) or not all(isinstance(v, str | int | float | bool) for v in raw.values()):
        raise EngineCheckNotDeclared(f"{where}: params must be a table of scalars")
    return MappingProxyType(dict(raw))


def load_engine_checks(path: Path) -> EngineChecks | None:
    """The allowlist, or None when the file is absent (then every engine_check is refused)."""
    if not path.is_file():
        return None
    data = path.read_bytes()
    try:
        raw = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise EngineCheckNotDeclared(f"{path.name} is not UTF-8 TOML: {exc}") from exc
    if (
        set(raw) != {"schema", "check"}
        or raw["schema"] != ENGINE_CHECKS_SCHEMA
        or not isinstance(raw["check"], list)
    ):
        raise EngineCheckNotDeclared(
            f"{path.name}: expected schema {ENGINE_CHECKS_SCHEMA!r} and [[check]] tables"
        )
    entries: dict[str, EngineCheckEntry] = {}
    for i, item in enumerate(raw["check"]):
        entry = _check_entry(item, f"{path.name} check {i + 1}")
        if entry.id in entries:
            raise EngineCheckNotDeclared(f"{path.name}: check id {entry.id!r} repeats")
        entries[entry.id] = entry
    return EngineChecks(sha256=sha256_hex(data), entries=entries)
