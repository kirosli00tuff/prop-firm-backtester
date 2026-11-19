"""Hypothesis files, schema "pfb.hypothesis/2" (docs/TRIALS.md section 2): parsing and shape checks only.

Exactly the keys of TRIALS.md section 2, missing or extra refused (also inside [cost], [power], [suite],
[secondary] and every [[window]]), plus one optional key, lookahead_seed. Schema 1 files are
refused (none was ever registered). What needs the
strategy's SPEC, the data catalog, the ledger or the allowlist is checked at registration
(pfb.runner.register and pfb.registry.hypothesis), not here.

Shape rules: id matches ID_RE; purpose "research" or "test", and purpose "test" only with budget 1
(PurposeTestBudget); roots a non-empty list of distinct names; bar_minutes an int >= 1; budget an
int >= 1; windows non-empty, each first <= last, no window repeated, and no two windows sharing a day
(WindowsOverlap, C.1 F5: [first, last] is inclusive at both ends, so a window ending on day d and one starting
on day d + 1 are allowed; every window applies to every root, so the rule is per root by construction);
lookahead_seed an int >= 0, default 0, the seed of the trial's step-1 look-ahead check; a seed
other than PROGRAM_LOOKAHEAD_SEED needs a non-empty optional lookahead_seed_reason (LookaheadSeedUnexplained;
schema 2 and schema 3 alike); grid a
non-empty table of non-empty scalar lists, a numeric list strictly increasing, another list without repeats;
cost.model book, fixed:<k> or stress:<spec>:<k> with no zero slippage; event_costs "off" with a non-empty
event_stance, or "on" (stance may be empty) on a book-based model, refused while EVENT_COSTS_ENGINE_WIRED is
False; power inputs positive, trade_kurtosis >= 1 (sd_source "stated" or "run:<run_id>"); suite as
pfb.analysis.spec.SuiteSpec;
secondary.prop_plans [] or ["*"] or distinct "<firm>/<plan file stem>" names, prop_mc >= 1, prop_seed >= 0.
"""

from __future__ import annotations

import math
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pfb.analysis.spec import SuiteSpec
from pfb.registry.chain import sha256_hex
from pfb.registry.errors import HypothesisInvalid, LookaheadSeedUnexplained, ZeroSlippage
from pfb.registry.inputs import (
    EVENT_COSTS_ENGINE_WIRED,
    EVENT_COSTS_NOT_WIRED,
    EVENT_COSTS_OFF,
    EVENT_COSTS_ON,
    EVENT_COSTS_VALUES,
    ID_RE,
    SHA_RE,
    GridValue,
    iso_date,
)

if TYPE_CHECKING:
    from pfb.registry.bookfile import BookHypothesis

HYPOTHESIS_SCHEMA = "pfb.hypothesis/2"
BOOK_SCHEMA_NAME = "pfb.hypothesis/3"  # pfb.registry.bookfile.BOOK_SCHEMA (the book file)
HYPOTHESIS_KEYS = frozenset(
    {
        "schema", "id", "purpose", "statement", "strategy", "strategy_sha256", "roots", "bar_minutes",
        "session", "budget", "no_signal_reason", "window", "grid", "cost", "power", "suite", "secondary",
    }
)  # fmt: skip
WINDOW_KEYS = frozenset({"first", "last"})
COST_KEYS = frozenset({"model", "fees", "event_costs", "event_stance"})
POWER_KEYS = frozenset(
    {"trades_per_year", "trade_sd_cents", "trade_skew", "trade_kurtosis", "sd_source", "target_edge_cents"}
)
MIN_RAW_KURTOSIS = 1.0
SECONDARY_KEYS = frozenset({"prop_plans", "prop_mc", "prop_seed"})
LOOKAHEAD_SEED_KEY = "lookahead_seed"
LOOKAHEAD_SEED_REASON_KEY = "lookahead_seed_reason"  # why another seed is needed
OPTIONAL_KEYS = frozenset({LOOKAHEAD_SEED_KEY, LOOKAHEAD_SEED_REASON_KEY})
PROGRAM_LOOKAHEAD_SEED = 0  # the one program-wide look-ahead seed (template, `pfb check`, every default)
DEFAULT_LOOKAHEAD_SEED = PROGRAM_LOOKAHEAD_SEED
TEST_BUDGET = 1
PURPOSE_RESEARCH, PURPOSE_TEST = "research", "test"
PURPOSES = (PURPOSE_RESEARCH, PURPOSE_TEST)
SD_STATED = "stated"
SD_RUN_PREFIX = "run:"
ALL_PLANS = "*"
PLAN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_.-]*")
COST_BOOK, COST_FIXED, COST_STRESS = "book", "fixed:", "stress:"


class PurposeTestBudget(HypothesisInvalid):
    """purpose "test" with a budget other than 1: a plumbing check is one configuration."""


class WindowsOverlap(HypothesisInvalid):
    """Two declared windows share at least one day: the same days would count twice for a root."""


@dataclass(frozen=True)
class CostDecl:
    model: str
    fees: str
    event_costs: str
    event_stance: str


@dataclass(frozen=True)
class PowerDecl:
    trades_per_year: float
    trade_sd_cents: int
    trade_skew: float
    trade_kurtosis: float
    sd_source: str
    target_edge_cents: int

    def to_json(self) -> dict[str, Any]:
        return {
            "trades_per_year": self.trades_per_year,
            "trade_sd_cents": self.trade_sd_cents,
            "trade_skew": self.trade_skew,
            "trade_kurtosis": self.trade_kurtosis,
            "sd_source": self.sd_source,
            "target_edge_cents": self.target_edge_cents,
        }


@dataclass(frozen=True)
class SecondaryDecl:
    prop_plans: tuple[str, ...]
    prop_mc: int
    prop_seed: int

    def to_json(self) -> dict[str, Any]:
        return {"prop_plans": list(self.prop_plans), "prop_mc": self.prop_mc, "prop_seed": self.prop_seed}


@dataclass(frozen=True)
class Hypothesis:
    """A parsed schema-2 hypothesis file; sha256 is that of the file's bytes."""

    id: str
    sha256: str
    purpose: str
    statement: str
    strategy: str
    strategy_sha256: str
    roots: tuple[str, ...]
    bar_minutes: int
    session: str
    budget: int
    no_signal_reason: str
    windows: tuple[tuple[str, str], ...]
    grid: Mapping[str, tuple[GridValue, ...]]
    cost: CostDecl
    power: PowerDecl
    suite: SuiteSpec
    secondary: SecondaryDecl
    lookahead_seed: int = DEFAULT_LOOKAHEAD_SEED
    lookahead_seed_reason: str = ""

    @property
    def grid_points(self) -> int:
        return math.prod(len(v) for v in self.grid.values())


def _table(raw: Mapping[str, Any], key: str, keys: frozenset[str], where: str) -> Mapping[str, Any]:
    value = raw[key]
    if not isinstance(value, dict) or set(value) != keys:
        raise HypothesisInvalid(f"{where}: [{key}] has exactly the keys {sorted(keys)}")
    return value


def _string(value: object, what: str, *, empty_ok: bool = False) -> str:
    if not isinstance(value, str) or (not value and not empty_ok):
        raise HypothesisInvalid(f"{what} must be a {'' if empty_ok else 'non-empty '}string")
    return value


def _int(value: object, what: str, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise HypothesisInvalid(f"{what} must be an integer >= {minimum}")
    return value


def _number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _grid(raw: object, where: str) -> dict[str, tuple[GridValue, ...]]:
    if not isinstance(raw, dict) or not raw:
        raise HypothesisInvalid(f"{where}: [grid] must be a non-empty table of lists")
    grid: dict[str, tuple[GridValue, ...]] = {}
    for name, values in raw.items():
        what = f"{where}: grid.{name}"
        if not isinstance(values, list) or not values:
            raise HypothesisInvalid(f"{what} must be a non-empty list")
        if not all(isinstance(v, str | int | float | bool) for v in values):
            raise HypothesisInvalid(f"{what} holds a value that is not a scalar")
        numeric = [_number(v) for v in values]
        if any(numeric) and not all(numeric):
            raise HypothesisInvalid(f"{what} mixes numbers and other values")
        if all(numeric):
            if any(not math.isfinite(float(v)) for v in values) or any(
                b <= a for a, b in zip(values, values[1:], strict=False)
            ):
                raise HypothesisInvalid(f"{what}: a numeric list must be finite and strictly increasing")
        elif len({repr(v) for v in values}) != len(values):
            raise HypothesisInvalid(f"{what} repeats a value")
        grid[name] = tuple(values)
    return grid


def _windows(raw: object, where: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw, list) or not raw:
        raise HypothesisInvalid(f"{where}: at least one [[window]] is required")
    out: list[tuple[str, str]] = []
    for i, item in enumerate(raw):
        what = f"{where}: window {i + 1}"
        if not isinstance(item, dict) or set(item) != WINDOW_KEYS:
            raise HypothesisInvalid(f"{what} has exactly the keys first and last")
        first = iso_date(item["first"], f"{what} first", HypothesisInvalid)
        last = iso_date(item["last"], f"{what} last", HypothesisInvalid)
        if first > last:
            raise HypothesisInvalid(f"{what}: first is after last")
        out.append((first, last))
    if len(set(out)) != len(out):
        raise HypothesisInvalid(f"{where}: a window is declared twice")
    _refuse_overlap(out, where)
    return tuple(out)


def first_overlap(
    windows: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], tuple[str, str]] | None:
    """Inclusive [first, last] windows sorted by first: one overlaps an earlier one when it starts on or
    before the latest last seen so far (YYYY-MM-DD strings order as dates). The first such pair, or None.
    Shared with replay (records._check_hypothesis_rules)."""
    ordered = sorted(windows)
    if not ordered:
        return None
    latest = ordered[0]
    for window in ordered[1:]:
        if window[0] <= latest[1]:
            return latest, window
        if window[1] > latest[1]:
            latest = window
    return None


def overlap_text(pair: tuple[tuple[str, str], tuple[str, str]]) -> str:
    (a_first, a_last), (b_first, b_last) = pair
    return (
        f"windows {a_first}..{a_last} and {b_first}..{b_last} overlap (they share at least one day); "
        "declare windows that do not share a day"
    )


def _refuse_overlap(windows: list[tuple[str, str]], where: str) -> None:
    pair = first_overlap(windows)
    if pair is not None:
        raise WindowsOverlap(f"{where}: {overlap_text(pair)}")


def _ticks(text: str, spec: str) -> Decimal:
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise HypothesisInvalid(f"cost model {spec!r}: {text!r} is not a tick count") from exc
    if not value.is_finite() or value < 0:
        raise HypothesisInvalid(f"cost model {spec!r}: ticks must be finite and >= 0")
    return value


def zero_slippage(spec: str) -> bool:
    """True when the cost spec charges no slippage at all (fixed:0, or a stress of one by 0 ticks)."""
    if spec == COST_BOOK:
        return False
    if spec.startswith(COST_FIXED):
        return _ticks(spec[len(COST_FIXED) :], spec) == 0
    if spec.startswith(COST_STRESS):
        base, sep, ticks = spec[len(COST_STRESS) :].rpartition(":")
        if not sep or not base:
            raise HypothesisInvalid(f"cost model {spec!r} is not stress:<spec>:<k>")
        return zero_slippage(base) and _ticks(ticks, spec) == 0
    raise HypothesisInvalid(f"cost model {spec!r} is not book, fixed:<k> or stress:<spec>:<k>")


def _cost(raw: Mapping[str, Any], where: str) -> CostDecl:
    table = _table(raw, "cost", COST_KEYS, where)
    model = _string(table["model"], f"{where}: cost.model")
    if zero_slippage(model):
        raise ZeroSlippage(f"{where}: cost model {model!r} charges no slippage")
    event_costs = table["event_costs"]
    if event_costs not in EVENT_COSTS_VALUES:
        raise HypothesisInvalid(f'{where}: event_costs must be "off" or "on", not {event_costs!r}')
    stance = _string(table["event_stance"], f"{where}: cost.event_stance", empty_ok=True)
    if event_costs == EVENT_COSTS_OFF and not stance.strip():
        raise HypothesisInvalid(f"{where}: event_stance must say why event minutes do not matter here")
    if event_costs == EVENT_COSTS_ON:
        base = model  # Upstream's event rule widens the book's half-spread: book or stress over book only
        while base.startswith(COST_STRESS):
            base = base[len(COST_STRESS) :].rpartition(":")[0]
        if base != COST_BOOK:
            raise HypothesisInvalid(f'{where}: event_costs "on" needs a book-based cost model, not {model!r}')
        if not EVENT_COSTS_ENGINE_WIRED:
            raise HypothesisInvalid(f"{where}: {EVENT_COSTS_NOT_WIRED}")
    return CostDecl(model, _string(table["fees"], f"{where}: cost.fees"), event_costs, stance)


def _power(raw: Mapping[str, Any], where: str) -> PowerDecl:
    table = _table(raw, "power", POWER_KEYS, where)
    rate = table["trades_per_year"]
    if not _number(rate) or not math.isfinite(float(rate)) or float(rate) <= 0:
        raise HypothesisInvalid(f"{where}: power.trades_per_year must be a number > 0")
    skew, kurtosis = table["trade_skew"], table["trade_kurtosis"]
    if not _number(skew) or not math.isfinite(float(skew)):
        raise HypothesisInvalid(f"{where}: power.trade_skew must be a finite number")
    if not _number(kurtosis) or not math.isfinite(float(kurtosis)) or float(kurtosis) < MIN_RAW_KURTOSIS:
        raise HypothesisInvalid(f"{where}: power.trade_kurtosis must be a finite raw kurtosis >= 1")
    source = _string(table["sd_source"], f"{where}: power.sd_source")
    if source != SD_STATED and not (source.startswith(SD_RUN_PREFIX) and len(source) > len(SD_RUN_PREFIX)):
        raise HypothesisInvalid(f'{where}: power.sd_source must be "stated" or "run:<run_id>"')
    return PowerDecl(
        trades_per_year=float(rate),
        trade_sd_cents=_int(table["trade_sd_cents"], f"{where}: power.trade_sd_cents", 1),
        trade_skew=float(skew),
        trade_kurtosis=float(kurtosis),
        sd_source=source,
        target_edge_cents=_int(table["target_edge_cents"], f"{where}: power.target_edge_cents", 1),
    )


def _suite(raw: Mapping[str, Any], where: str) -> SuiteSpec:
    try:
        return SuiteSpec.from_mapping(raw["suite"])
    except ValueError as exc:
        raise HypothesisInvalid(f"{where}: [suite]: {exc}") from exc


def _secondary(raw: Mapping[str, Any], where: str) -> SecondaryDecl:
    table = _table(raw, "secondary", SECONDARY_KEYS, where)
    plans = table["prop_plans"]
    if not isinstance(plans, list) or not all(isinstance(p, str) for p in plans):
        raise HypothesisInvalid(f"{where}: secondary.prop_plans must be a list of strings")
    if ALL_PLANS in plans and plans != [ALL_PLANS]:
        raise HypothesisInvalid(f'{where}: secondary.prop_plans ["*"] stands alone')
    if plans != [ALL_PLANS] and (
        not all(PLAN_RE.fullmatch(p) for p in plans) or len(set(plans)) != len(plans)
    ):
        raise HypothesisInvalid(
            f'{where}: secondary.prop_plans holds distinct "<firm>/<plan file stem>" names'
        )
    return SecondaryDecl(
        prop_plans=tuple(plans),
        prop_mc=_int(table["prop_mc"], f"{where}: secondary.prop_mc", 1),
        prop_seed=_int(table["prop_seed"], f"{where}: secondary.prop_seed", 0),
    )


def _roots(raw: object, where: str) -> tuple[str, ...]:
    if (
        not isinstance(raw, list)
        or not raw
        or not all(isinstance(r, str) and r for r in raw)
        or len(set(raw)) != len(raw)
    ):
        raise HypothesisInvalid(f"{where}: roots must be a non-empty list of distinct root names")
    return tuple(raw)


def _header(raw: Mapping[str, Any], where: str) -> tuple[str, str, str]:
    hyp_id = _string(raw["id"], f"{where}: id")
    if not ID_RE.fullmatch(hyp_id):
        raise HypothesisInvalid(f"{where}: id {hyp_id!r} must match {ID_RE.pattern}")
    purpose = raw["purpose"]
    if purpose not in PURPOSES:
        raise HypothesisInvalid(f"{where}: purpose must be one of {list(PURPOSES)}")
    sha = _string(raw["strategy_sha256"], f"{where}: strategy_sha256")
    if not SHA_RE.fullmatch(sha):
        raise HypothesisInvalid(f"{where}: strategy_sha256 is not a sha256 (64 lowercase hex)")
    return hyp_id, purpose, sha


def _purpose_budget(purpose: str, budget: int, where: str) -> int:
    if purpose == PURPOSE_TEST and budget != TEST_BUDGET:
        raise PurposeTestBudget(
            f'{where}: purpose "test" is a plumbing check of one configuration: budget must be '
            f"{TEST_BUDGET}, not {budget}"
        )
    return budget


def _lookahead_seed(raw: Mapping[str, Any], where: str) -> int:
    if LOOKAHEAD_SEED_KEY not in raw:
        return DEFAULT_LOOKAHEAD_SEED
    return _int(raw[LOOKAHEAD_SEED_KEY], f"{where}: {LOOKAHEAD_SEED_KEY}", 0)


def seed_unexplained(seed: object, reason: object) -> bool:
    """A look-ahead seed other than the program's needs a non-empty reason."""
    return seed != PROGRAM_LOOKAHEAD_SEED and not (isinstance(reason, str) and reason.strip())


def seed_unexplained_text(seed: object) -> str:
    return (
        f"{LOOKAHEAD_SEED_KEY} {seed} is not the program seed {PROGRAM_LOOKAHEAD_SEED}: state why in a "
        f"non-empty {LOOKAHEAD_SEED_REASON_KEY}"
    )


def lookahead_seed_fields(raw: Mapping[str, Any], where: str) -> tuple[int, str]:
    """(lookahead_seed, lookahead_seed_reason) of a schema-2 or schema-3 file; LookaheadSeedUnexplained
    when the seed is not the program seed and the reason is empty."""
    seed = _lookahead_seed(raw, where)
    reason = ""
    if LOOKAHEAD_SEED_REASON_KEY in raw:
        reason = _string(
            raw[LOOKAHEAD_SEED_REASON_KEY], f"{where}: {LOOKAHEAD_SEED_REASON_KEY}", empty_ok=True
        )
    if seed_unexplained(seed, reason):
        raise LookaheadSeedUnexplained(f"{where}: {seed_unexplained_text(seed)}")
    return seed, reason


def parse_hypothesis(data: bytes, where: str) -> Hypothesis | BookHypothesis:
    """Parse and shape-check a schema-2 file (HypothesisInvalid or ZeroSlippage on any fault); a schema-3
    book file is dispatched to pfb.registry.bookfile.parse_book."""
    try:
        raw = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise HypothesisInvalid(f"{where} is not UTF-8 TOML: {exc}") from exc
    if raw.get("schema") == BOOK_SCHEMA_NAME:
        from pfb.registry.bookfile import parse_book

        return parse_book(data, where)
    if raw.get("schema") != HYPOTHESIS_SCHEMA:
        raise HypothesisInvalid(f"{where}: schema must be {HYPOTHESIS_SCHEMA!r} (schema 1 is refused)")
    missing = sorted(HYPOTHESIS_KEYS - set(raw))
    extra = sorted(set(raw) - HYPOTHESIS_KEYS - OPTIONAL_KEYS)
    if missing or extra:
        raise HypothesisInvalid(f"{where}: keys missing {missing}, extra {extra}")
    hyp_id, purpose, strategy_sha = _header(raw, where)
    budget = _purpose_budget(purpose, _int(raw["budget"], f"{where}: budget", 1), where)
    seed, seed_reason = lookahead_seed_fields(raw, where)
    return Hypothesis(
        id=hyp_id,
        sha256=sha256_hex(data),
        purpose=purpose,
        statement=_string(raw["statement"], f"{where}: statement"),
        strategy=_string(raw["strategy"], f"{where}: strategy"),
        strategy_sha256=strategy_sha,
        roots=_roots(raw["roots"], where),
        bar_minutes=_int(raw["bar_minutes"], f"{where}: bar_minutes", 1),
        session=_string(raw["session"], f"{where}: session"),
        budget=budget,
        no_signal_reason=_string(raw["no_signal_reason"], f"{where}: no_signal_reason", empty_ok=True),
        windows=_windows(raw["window"], where),
        grid=_grid(raw["grid"], where),
        cost=_cost(raw, where),
        power=_power(raw, where),
        suite=_suite(raw, where),
        secondary=_secondary(raw, where),
        lookahead_seed=seed,
        lookahead_seed_reason=seed_reason,
    )


def load_hypothesis(path: Path) -> Hypothesis | BookHypothesis:
    if not path.is_file():
        raise HypothesisInvalid(f"hypothesis file {path.name} does not exist")
    return parse_hypothesis(path.read_bytes(), path.name)
