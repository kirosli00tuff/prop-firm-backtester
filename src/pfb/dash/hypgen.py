"""Step 4 of the Backtest page, pure: the text of a schema "pfb.hypothesis/2" file (docs/TRIALS.md section 2;
modelled on a plumbing hypothesis file) from step 1-2's StrategyState, step 3's open windows, the picked
firm plans and the form the user fills in. Nothing is read, written or run here; pfb.dash.actions saves the
text to hypotheses/uploads/<id>.toml and pfb.dash.view_run shows it in full before the confirm box.

What the dashboard fixes and what the user types:
- From the upload: strategy (strategies/uploads/<sha256>.py) and strategy_sha256; from its SPEC: roots (the
  SPEC root alone), bar_minutes, session and the grid: the SPEC defaults (one value per parameter) or the
  SPEC's full grid, the user's choice. budget = grid points x picked windows: every picked window is run once
  at the SPEC defaults, and a full grid leaves the rest of the budget for later runs from the command line.
- [power] from the step-2 synthetic run (sd_source "run:<run_id>"): its realized per-trade sd, skew and raw
  kurtosis, and trades per year rounded DOWN to TPY_DECIMALS (registration refuses a stated rate above the
  source run's trades x 252 / days, or a stated sd below its sd).
- [cost] book costs with the fees of the first picked plan's firm (DEFAULT_FEES with no plan); event_costs
  and event_stance from the form. [suite] fixed defaults (no neighbourhood: one trial per window), sizing
  on the first picked plan. [secondary] prop_plans = the picks, PROP_MC paths, PROP_SEED.
- The user types statement, target_edge_cents, event_costs and event_stance.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from fractions import Fraction
from typing import Any

from pfb.registry.errors import RegistryError
from pfb.registry.hypfile import HYPOTHESIS_SCHEMA, parse_hypothesis
from pfb.registry.powercheck import power_at

PURPOSE = "research"
ID_PREFIX = "h_"
SHA_PREFIX_LEN = 8
STAMP_FORMAT = "%Y%m%dT%H%M%SZ"
HYPOTHESES_UPLOADS_REL = "hypotheses/uploads"
COST_MODEL = "book"
DEFAULT_FEES = "topstep"
EVENT_OFF, EVENT_ON = "off", "on"
EVENT_CHOICES = (EVENT_OFF, EVENT_ON)
COST_STRESS_TICKS = (1, 2)
BASELINE_DRAWS = 200
BASELINE_SEED = 7
WALK_FORWARD_FOLDS = 5
PROP_MC = 20000
PROP_SEED = 20261116
TRADING_DAYS_PER_YEAR = 252
TPY_DECIMALS = 1
SD_RUN_PREFIX = "run:"
BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")
SPEC_FIELDS = ("root", "bar_minutes", "session", "params")


class HypGenError(ValueError):
    """The inputs cannot make a hypothesis file (no SPEC, a window without an end, no power source...)."""


@dataclass(frozen=True)
class PowerSource:
    """The step-2 synthetic run's realized figures (its result record's metrics.suite.power)."""

    run_id: str
    trades: int
    days: int
    trade_sd_cents: int
    trade_skew: float
    trade_kurtosis: float

    def trades_per_year(self) -> float:
        """trades x 252 / days rounded down to TPY_DECIMALS, never above the exact rate."""
        exact = Fraction(self.trades * TRADING_DAYS_PER_YEAR, self.days)
        scale = 10**TPY_DECIMALS
        steps = math.floor(exact * scale)
        while steps > 0 and Fraction(float(steps / scale)) > exact:
            steps -= 1
        return steps / scale


@dataclass(frozen=True)
class HypForm:
    """What the user chose and typed in step 4, plus the id and the power source."""

    hypothesis_id: str
    statement: str
    target_edge_cents: int
    event_costs: str
    event_stance: str
    full_grid: bool
    power: PowerSource


def _int_of(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _num_of(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def power_source(run_id: str, result: Mapping[str, Any] | None) -> PowerSource | None:
    """The realized power figures of an ok run's result record, or None when they are missing."""
    if not result or result.get("status") != "ok":
        return None
    metrics = result.get("metrics")
    suite = metrics.get("suite") if isinstance(metrics, Mapping) else None
    power = suite.get("power") if isinstance(suite, Mapping) else None
    if not isinstance(power, Mapping):
        return None
    trades, days, sd = (
        _int_of(power.get("trades")),
        _int_of(power.get("days")),
        _int_of(power.get("trade_sd_cents")),
    )
    skew, kurt = _num_of(power.get("trade_skew")), _num_of(power.get("trade_kurtosis"))
    if trades is None or days is None or sd is None or skew is None or kurt is None:
        return None
    if trades < 2 or days < 2 or sd < 1:
        return None
    return PowerSource(run_id, trades, days, sd, skew, kurt)


def hypothesis_id(strategy_sha256: str, now: datetime) -> str:
    """h_<first 8 hex of the strategy sha256>_<UTC stamp>."""
    stamp = now.astimezone(UTC).strftime(STAMP_FORMAT)
    return f"{ID_PREFIX}{strategy_sha256[:SHA_PREFIX_LEN]}_{stamp}"


def hypothesis_rel(hyp_id: str) -> str:
    return f"{HYPOTHESES_UPLOADS_REL}/{hyp_id}.toml"


def _spec(spec: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(spec, Mapping) or any(name not in spec for name in SPEC_FIELDS):
        raise HypGenError(f"the SPEC must name {', '.join(SPEC_FIELDS)}")
    if not isinstance(spec["params"], Mapping) or not spec["params"]:
        raise HypGenError("the SPEC names no parameter")
    return spec


def grid_of(spec: Mapping[str, Any], full: bool) -> dict[str, list[Any]]:
    """Every SPEC parameter: its full grid, or its default alone."""
    out: dict[str, list[Any]] = {}
    for name, item in _spec(spec)["params"].items():
        if not isinstance(item, Mapping) or "default" not in item or "grid" not in item:
            raise HypGenError(f"SPEC parameter {name!r} needs a default and a grid")
        out[str(name)] = list(item["grid"]) if full else [item["default"]]
    return out


def grid_points(grid: Mapping[str, Sequence[Any]]) -> int:
    return math.prod(len(values) for values in grid.values())


def budget_of(grid: Mapping[str, Sequence[Any]], n_windows: int) -> int:
    return grid_points(grid) * n_windows


def fees_of(plans: Sequence[str]) -> str:
    """The first plan's firm (config/fees/<firm>.toml), else DEFAULT_FEES."""
    return plans[0].partition("/")[0] if plans else DEFAULT_FEES


def windows_of(windows: Sequence[Mapping[str, Any]]) -> list[tuple[date, date]]:
    """The picked windows as (first, last) dates in time order (the order of their 1-based --window index)."""
    out: list[tuple[date, date]] = []
    for item in windows:
        first, last = item.get("first"), item.get("last")
        if not isinstance(first, str) or not isinstance(last, str):
            raise HypGenError(f"window {first}..{last} has no fixed first and last trade date")
        out.append((date.fromisoformat(first), date.fromisoformat(last)))
    if not out:
        raise HypGenError("pick at least one open window")
    return sorted(set(out))


# ------------------------------------------------------------------------------------- TOML ----
def _toml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HypGenError(f"{value!r} cannot be written to the hypothesis file")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)  # a JSON string is a TOML basic string
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml(v) for v in value) + "]"
    raise HypGenError(f"{type(value).__name__} {value!r} cannot be written to the hypothesis file")


def _key(name: str) -> str:
    return name if BARE_KEY.fullmatch(name) else json.dumps(name)


def _lines(pairs: Mapping[str, Any]) -> list[str]:
    return [f"{_key(k)} = {_toml(v)}" for k, v in pairs.items()]


def _section(title: str, pairs: Mapping[str, Any]) -> list[str]:
    return ["", f"[{title}]", *_lines(pairs)]


def _header(state: Any, spec: Mapping[str, Any], form: HypForm, budget: int) -> dict[str, Any]:
    return {
        "schema": HYPOTHESIS_SCHEMA,
        "id": form.hypothesis_id,
        "purpose": PURPOSE,
        "statement": form.statement.strip(),
        "strategy": state.rel_path,
        "strategy_sha256": state.sha256,
        "roots": [spec["root"]],
        "bar_minutes": spec["bar_minutes"],
        "session": spec["session"],
        "budget": budget,
        "no_signal_reason": "",
    }


def _power(form: HypForm) -> dict[str, Any]:
    source = form.power
    return {
        "trades_per_year": source.trades_per_year(),
        "trade_sd_cents": source.trade_sd_cents,
        "trade_skew": source.trade_skew,
        "trade_kurtosis": source.trade_kurtosis,
        "sd_source": f"{SD_RUN_PREFIX}{source.run_id}",
        "target_edge_cents": form.target_edge_cents,
    }


def build(state: Any, windows: Sequence[Mapping[str, Any]], plans: Sequence[str], form: HypForm) -> str:
    """The hypothesis file's text. state: a StrategyState (rel_path, sha256, spec); windows: step 3's open
    window dicts (first, last as ISO dates); plans: "<firm>/<plan stem>" names. Raises HypGenError."""
    spec = _spec(state.spec)
    if form.event_costs not in EVENT_CHOICES:
        raise HypGenError(f"event_costs is one of {EVENT_CHOICES}, not {form.event_costs!r}")
    picked = windows_of(windows)
    grid = grid_of(spec, form.full_grid)
    lines = _lines(_header(state, spec, form, budget_of(grid, len(picked))))
    for first, last in picked:
        lines += ["", "[[window]]", f"first = {first.isoformat()}", f"last = {last.isoformat()}"]
    lines += _section("grid", grid)
    cost = {"model": COST_MODEL, "fees": fees_of(plans), "event_costs": form.event_costs,
            "event_stance": form.event_stance.strip()}  # fmt: skip
    lines += _section("cost", cost)
    lines += _section("power", _power(form))
    suite = {"cost_stress_ticks": list(COST_STRESS_TICKS), "baseline_draws": BASELINE_DRAWS,
             "baseline_seed": BASELINE_SEED, "walk_forward_folds": WALK_FORWARD_FOLDS, "neighbourhood": False,
             "sizing_plan": plans[0] if plans else ""}  # fmt: skip
    lines += _section("suite", suite)
    lines += _section("secondary", {"prop_plans": list(plans), "prop_mc": PROP_MC, "prop_seed": PROP_SEED})
    return "\n".join(lines) + "\n"


def shape_error(text: str, where: str) -> str | None:
    """The hypothesis file's own shape check (pfb.registry.hypfile; no SPEC, ledger or data): None when the
    text passes, else the refusal as "<code>: <reason>"."""
    try:
        parse_hypothesis(text.encode("utf-8"), where)
    except RegistryError as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


@dataclass(frozen=True)
class PowerPreview:
    """The registration power check of one window at N = n_used (pfb.registry.powercheck): the minimum
    detectable edge per trade (None: undetectable) and whether the target reaches it."""

    first: date
    last: date
    n_used: int
    min_edge_cents: int | None
    passed: bool


def power_preview(form: HypForm, windows: Sequence[Mapping[str, Any]], n_used: int) -> list[PowerPreview]:
    """What registration's power check will find for each picked window at N = n_used (N + budget)."""
    inputs = _power(form)
    out = []
    for first, last in windows_of(windows):
        result = power_at(inputs, [[first.isoformat(), last.isoformat()]], n_used)
        out.append(PowerPreview(first, last, n_used, result.min_edge_cents, result.passed is True))
    return out


def missing_fields(statement: str, target_edge_cents: int, event_costs: str, event_stance: str) -> list[str]:
    """The form fields the user still has to fill in (the stance only under event_costs "off")."""
    checks = (
        ("statement", not statement.strip()),
        ("target_edge_cents", target_edge_cents <= 0),
        ("event_stance", event_costs == EVENT_OFF and not event_stance.strip()),
    )
    return [name for name, missing in checks if missing]
