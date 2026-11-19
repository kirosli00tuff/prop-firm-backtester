"""Prop rule configs (pfb-rules-1, docs/PROP_RULES_SCHEMA.md): load, validate, freeze.

``load_plan(path)`` reads one TOML plan file and returns a frozen ``PlanRules``. Every field the schema lists
must be present (no silent default), no extra key is accepted, every ``src`` must resolve in ``[sources]``,
enums and the ``"none"`` marker are checked per field, and an ``unknown`` leaf is kept: ``PlanRules.value``
raises ``RuleUnknown(<dotted path>: <reason>)`` the moment a run asks for it, unless the leaf carries
``assume = true``: then the run reads its alt[0] and every prop result names the assumption
(``PlanRules.unknown_notes``, with the fields the Monte Carlo does not model). The file's sha256 rides on the
result. No rule number lives in this module.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from pfb.prop.types import RuleUnknown

SCHEMA = "pfb-rules-1"
FIRMS = ("topstep", "lucid", "bulenox", "mffu", "tradeify")
STATUSES = ("sourced", "provisional")
KINDS = ("exact", "conservative", "inferred")
NONE = "none"
BPS_ONE = 10_000
TOP_KEYS = (
    "schema",
    "firm",
    "plan",
    "size",
    "label",
    "status",
    "version",
    "effective_from",
    "sources",
    "eval",
    "funded",
    "payout",
    "fees",
)
LEAF_KEYS = frozenset({"v", "unknown", "src", "note", "alt", "kind", "assume"})
# `assume = true` on an `unknown` leaf makes a run read alt[0] (the stricter reading) in its
# place as a labelled assumption; every prop result of the plan names it (PlanRules.unknown_notes). Refused
# where an assumption could flatter the plan: on these fields at all, and as "none" on a daily loss limit.
NO_ASSUME = {
    "funded.funded_reset_cents": "a funded reset is not modelled, so it is never assumed available",
    "fees.rebill_adds_reset_credit": "reset credits are not modelled, so they are never assumed free",
}
ASSUME_NOT_NONE = frozenset({"eval.daily_loss_cents", "funded.daily_loss_cents"})
# Fields the Monte Carlo does not model: a plan whose value is not the neutral one (or is unknown) is named
# on every prop result, with how the result was computed without it.
NOT_MODELLED = {
    "funded.funded_reset_cents": (
        None,
        "computed as if a breached funded account is never bought back (the funded life ends at its breach), "
        "a policy open to every trader, so the figure needs no funded reset",
    ),
    "fees.rebill_adds_reset_credit": (
        False,
        "reset credits are not credited: every retry pays fees.reset_cents, so cost_to_funded_with_resets "
        "and ev_per_purchase can only understate what the credits are worth",
    ),
}
# version = { n = <int >= 1>, rules_read = "<ISO date read>", read_in = "<record>" }.
# n numbers a plan's dated versions in effective_from order; the newest is the file
# config/firms/<firm>/<plan>_<size>.toml and older ones live under
# config/firms/<firm>/versions/<effective_from>/ (not discovered as plans of their own).
VERSION_KEYS = frozenset({"n", "rules_read", "read_in"})
VERSIONS_DIR = "versions"
SOURCE_KEYS = frozenset({"url", "record", "looked_up", "title", "quote"})

# Field tags: "int" (>= 0), "rel" (any sign), "bps" (0..10000), "bool", "str", "date", "enum:a|b",
# "scaling", "caps"; a trailing "?" admits the "none" marker (read as Python None).
_DRAWDOWN: tuple[tuple[str, str], ...] = (
    ("start_balance_cents", "int"),
    ("drawdown_cents", "int"),
    ("drawdown_type", "enum:eod_trailing|intraday_trailing|static"),
    ("breach_check", "enum:intraday|eod"),
    ("breach_at_equal", "bool"),
    ("drawdown_lock_rel_cents", "rel?"),
    ("daily_loss_cents", "int?"),
    ("daily_loss_kind", "enum:soft|hard|none"),
)
SECTION_FIELDS: dict[str, tuple[tuple[str, str], ...]] = {
    "eval": (
        ("start_balance_cents", "int"),
        ("profit_target_cents", "int"),
        ("target_check", "enum:eod|intraday"),
        *_DRAWDOWN[1:],
        ("consistency_type", "enum:none|best_day_frac_of_profit|best_day_frac_of_target"),
        ("consistency_bps", "bps?"),
        ("consistency_strict", "bool"),
        ("min_trading_days", "int"),
        ("min_winning_days", "int"),
        ("winning_day_cents", "int?"),
        ("max_contracts_minis", "int"),
        ("time_limit_calendar_days", "int?"),
        ("flat_by", "str?"),
        ("news_rule", "str?"),
    ),
    "funded": (
        *_DRAWDOWN,
        ("floor_after_first_payout_rel_cents", "rel?"),
        ("scaling", "scaling"),
        ("scaling_boundary", "enum:lower|upper"),
        ("payout_count_limit", "int?"),
        ("inactivity_days", "int?"),
        # the price to reopen a breached funded account (Topstep Back2Funded, Tradeify Funded
        # Reset); none = the firm has no such option. Not modelled (NOT_MODELLED).
        ("funded_reset_cents", "int?"),
    ),
    "payout": (
        ("path", "str"),
        ("win_day_cents", "int?"),
        ("win_day_strict", "bool"),
        ("win_days_required", "int"),
        ("traded_days_required", "int"),
        ("net_positive_since_last", "bool"),
        ("net_min_first_cents", "int?"),
        ("net_min_later_cents", "int?"),
        ("min_balance_rel_cents", "rel?"),
        ("buffer_rel_cents", "rel?"),
        ("frac_of_profit_bps", "bps"),
        ("min_request_cents", "int"),
        ("caps_cents", "caps?"),
        ("window_cap_bps", "int?"),
        ("window_cap_from_payout", "int"),
        ("consistency_bps", "bps?"),
        ("consistency_scope", "enum:window|cumulative"),
        ("consistency_strict", "bool"),
        ("request_weekday", "int?"),
        ("window_calendar_days", "int?"),
        ("dead_days_after_payout", "int"),
        ("payout_day_counts", "bool"),
        ("split_bps", "bps"),
        ("split_first_full_cents", "int?"),
    ),
    "fees": (
        ("billing", "enum:monthly|one_time"),
        ("eval_price_cents", "int"),
        ("billing_period_calendar_days", "int?"),
        ("reset_cents", "int?"),
        ("rebill_adds_reset_credit", "bool"),
        ("reset_pushes_rebill", "bool"),
        ("activation_cents", "int"),
        ("funded_monthly_cents", "int"),
        ("platform_monthly_cents", "int"),
    ),
}
FIELD_TAGS: dict[str, str] = {
    f"{section}.{name}": tag for section, fields in SECTION_FIELDS.items() for name, tag in fields
}
FIELD_TAGS["effective_from"] = "date"


class PlanConfigError(ValueError):
    """The plan file breaks pfb-rules-1; ``problems`` lists every violation found."""

    def __init__(self, path: str, problems: list[str]) -> None:
        self.problems = tuple(problems)
        shown = "; ".join(self.problems[:20])
        more = f" (+{len(self.problems) - 20} more)" if len(self.problems) > 20 else ""
        super().__init__(f"{path}: {len(self.problems)} problem(s): {shown}{more}")


@dataclass(frozen=True)
class Source:
    key: str
    url: str | None
    record: str | None
    looked_up: str
    title: str
    quote: str | None


@dataclass(frozen=True)
class Leaf:
    """One rule leaf. ``unknown`` is the reason when the value is unknown (then ``value`` is None)."""

    path: str
    value: Any  # int, bool, str, tuple, or None for the "none" marker
    unknown: str | None
    src: str | None
    note: str | None = None
    alt: tuple[Any, ...] = ()
    kind: str = "exact"
    assume: bool = False  # an unknown leaf a run reads as alt[0] (a labelled assumption)
    assumed_value: Any = None  # alt[0] parsed like a value ("none" -> None) when assume is true

    @property
    def is_unknown(self) -> bool:
        return self.unknown is not None


@dataclass(frozen=True)
class PlanRules:
    """A validated plan file. Read values with ``value(path)``; an unknown leaf raises RuleUnknown."""

    path: str
    sha256: str
    firm: str
    plan: str
    size: str
    label: str
    status: str
    leaves: Mapping[str, Leaf]
    sources: Mapping[str, Source]
    overrides: tuple[tuple[str, Any], ...] = field(default=())
    file_sha256: str = ""  # the plan file's sha256; differs from sha256 once values are overridden
    version: int = 0  # version.n of the file (1 = the oldest dated version of the plan)
    rules_read: str = ""  # version.rules_read: the ISO date the rules were read
    read_in: str = ""  # version.read_in: the record the rules were read in

    def leaf(self, path: str) -> Leaf:
        try:
            return self.leaves[path]
        except KeyError as exc:
            raise KeyError(f"{path}: not a pfb-rules-1 field") from exc

    def value(self, path: str) -> Any:
        """The value at a dotted path such as ``eval.profit_target_cents`` (None for "none"). An unknown leaf
        raises RuleUnknown, unless it carries ``assume = true``: then its alt[0] (named by unknown_notes)."""
        leaf = self.leaf(path)
        if leaf.unknown is not None:
            if leaf.assume:
                return leaf.assumed_value
            raise RuleUnknown(f"{path}: {leaf.unknown}")
        return leaf.value

    def unknown_notes(self) -> list[str]:
        """: what every prop result of this plan says about values it does not know or does not model:
        each assumed unknown (the alt read in its place) and each NOT_MODELLED field the plan holds."""
        notes: list[str] = []
        for path, leaf in self.leaves.items():
            if leaf.is_unknown and leaf.assume:
                notes.append(f"rule unknown, assumed: {path} = {leaf.alt[0]!r} is an assumption, not a "
                             f"published value ({leaf.unknown})")  # fmt: skip
        for path, (neutral, how) in NOT_MODELLED.items():
            leaf = self.leaves.get(path)
            if leaf is None or (not leaf.is_unknown and leaf.value == neutral):
                continue
            held = f"unknown ({leaf.unknown})" if leaf.is_unknown else f"{leaf.value!r} in the plan"
            notes.append(f"rule not modelled: {path} is {held}; {how}")
        return notes

    def is_unknown(self, path: str) -> bool:
        return self.leaf(path).is_unknown

    @property
    def effective_from(self) -> str:
        return str(self.value("effective_from"))

    @property
    def unknown_paths(self) -> tuple[str, ...]:
        return tuple(p for p, leaf in self.leaves.items() if leaf.is_unknown)

    @property
    def effective_from_int(self) -> int:
        """effective_from as an int yyyymmdd (compared with DayPaths.trade_dates; no float, no clock)."""
        return int(self.effective_from.replace("-", ""))

    def rules_label(self) -> dict[str, Any]:
        """The rule labels every prop result carries."""
        return {
            "firm": self.firm,
            "plan": self.plan,
            "size": self.size,
            "plan_version": self.version,
            "effective_from": self.effective_from,
            "rules_read": self.rules_read,
            "plan_file_sha256": self.file_sha256 or self.sha256,
            "plan_sha256": self.sha256,
        }


# ---------------------------------------------------------------------------------- leaf checks ----
def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _int_in(x: Any, lo: int, hi: int | None) -> bool:
    return _is_int(x) and lo <= int(x) and (hi is None or int(x) <= hi)


def _check_scaling(path: str, v: Any, problems: list[str]) -> tuple[tuple[int | None, int], ...] | None:
    if not isinstance(v, list) or not v:
        problems.append(f'{path}: expected a non-empty list of [below_rel_cents | "none", max_minis]')
        return None
    tiers: list[tuple[int | None, int]] = []
    for i, tier in enumerate(v):
        ok_shape = isinstance(tier, list) and len(tier) == 2
        if not ok_shape or not (_is_int(tier[0]) or tier[0] == NONE) or not _is_int(tier[1]) or tier[1] < 0:
            problems.append(f'{path}[{i}]: expected [int | "none", int >= 0], got {tier!r}')
            return None
        tiers.append((None if tier[0] == NONE else int(tier[0]), int(tier[1])))
    bounds = [b for b, _ in tiers]
    if bounds[-1] is not None or any(b is None for b in bounds[:-1]):
        problems.append(f'{path}: the last tier, and only it, has "none"')
        return None
    finite = [b for b in bounds[:-1] if b is not None]
    if any(b2 <= b1 for b1, b2 in zip(finite, finite[1:], strict=False)):
        problems.append(f"{path}: below_rel_cents not strictly ascending")
        return None
    return tuple(tiers)


def _check_value(path: str, tag: str, v: Any, problems: list[str]) -> tuple[bool, Any]:
    """(ok, parsed value) for one ``v``; "none" becomes None where the tag allows it."""
    allow_none = tag.endswith("?")
    base = tag.rstrip("?")
    if v == NONE and not base.startswith("enum:"):
        if allow_none:
            return True, None
        problems.append(f'{path}: "none" is not allowed here')
        return False, None
    ok = True
    parsed: Any = v
    if base == "int":
        ok = _int_in(v, 0, None)
    elif base == "rel":
        ok = _is_int(v)
    elif base == "bps":
        ok = _int_in(v, 0, BPS_ONE)
    elif base == "bool":
        ok = isinstance(v, bool)
    elif base == "str":
        ok = isinstance(v, str) and bool(v)
    elif base == "date":
        ok = _valid_date(v)
        parsed = v.isoformat() if isinstance(v, dt.date) else v
    elif base.startswith("enum:"):
        ok = v in base[5:].split("|")
    elif base == "caps":
        ok = isinstance(v, list) and bool(v) and all(_is_int(c) and c > 0 for c in v)
        parsed = tuple(v) if ok else None
    elif base == "scaling":
        parsed = _check_scaling(path, v, problems)
        return parsed is not None, parsed
    else:  # pragma: no cover - the tag table is fixed
        raise AssertionError(f"unknown tag {tag}")
    if not ok:
        problems.append(f"{path}: {v!r} does not match {tag}")
    return ok, parsed


def _valid_date(v: Any) -> bool:
    if isinstance(v, dt.date) and not isinstance(v, dt.datetime):
        return True
    if not isinstance(v, str):
        return False
    try:
        dt.date.fromisoformat(v)
    except ValueError:
        return False
    return len(v) == 10


def _assume(
    path: str, raw: Mapping[str, Any], has_unknown: bool, alt: list[Any], problems: list[str]
) -> bool:
    """: ``assume`` is a bool, only on an unknown leaf with an alt, never on a NO_ASSUME field, and never
    "none" on a daily loss limit; anything else is a problem (no silent default)."""
    if "assume" not in raw:
        return False
    assume = raw["assume"]
    if not isinstance(assume, bool):
        problems.append(f"{path}: assume must be true or false")
        return False
    if not assume:
        return False
    if not has_unknown:
        problems.append(f"{path}: assume is only for an unknown leaf")
    elif not alt:
        problems.append(f"{path}: assume needs an alt (the value assumed meanwhile); no silent default")
    elif path in NO_ASSUME:
        problems.append(f"{path}: assume refused: {NO_ASSUME[path]}")
    elif path in ASSUME_NOT_NONE and alt[0] == NONE:
        problems.append(
            f'{path}: assume refused: an unknown daily loss limit is never assumed absent ("none")'
        )
    return True


def _leaf(path: str, tag: str, raw: Any, sources: Mapping[str, Any], problems: list[str]) -> Leaf | None:
    if not isinstance(raw, Mapping):
        problems.append(f"{path}: expected an inline table {{ v = ..., src = ... }}")
        return None
    extra = sorted(set(raw) - LEAF_KEYS)
    if extra:
        problems.append(f"{path}: unknown keys {extra}")
    has_v, has_unknown = "v" in raw, "unknown" in raw
    if has_v == has_unknown:
        problems.append(f"{path}: exactly one of v and unknown is required")
        return None
    src = raw.get("src")
    if src is not None and (not isinstance(src, str) or src not in sources):
        problems.append(f"{path}: src {src!r} is not a key of [sources]")
    if has_v and src is None:
        problems.append(f"{path}: src is required with v")
    kind = raw.get("kind", "exact")
    if kind not in KINDS:
        problems.append(f"{path}: kind {kind!r} not in {KINDS}")
    note = raw.get("note")
    if note is not None and not isinstance(note, str):
        problems.append(f"{path}: note must be a string")
    alt = raw.get("alt", [])
    if not isinstance(alt, list):
        problems.append(f"{path}: alt must be a list")
        alt = []
    parsed_alt = [_check_value(f"{path}.alt[{i}]", tag, a, problems)[1] for i, a in enumerate(alt)]
    assume = _assume(path, raw, has_unknown, alt, problems)
    if has_unknown:
        reason = raw["unknown"]
        if not isinstance(reason, str) or not reason.strip():
            problems.append(f"{path}: unknown needs a non-empty reason")
            return None
        assumed = parsed_alt[0] if assume and parsed_alt else None
        return Leaf(path, None, reason, src, note, tuple(alt), kind, assume, assumed)
    ok, parsed = _check_value(path, tag, raw["v"], problems)
    if not ok:
        return None
    return Leaf(path, parsed, None, src, note, tuple(alt), kind)


# ------------------------------------------------------------------------------- whole file ----
def _sources(raw: Any, problems: list[str]) -> dict[str, Source]:
    if not isinstance(raw, Mapping) or not raw:
        problems.append("sources: expected a non-empty table")
        return {}
    out: dict[str, Source] = {}
    for key, s in raw.items():
        where = f"sources.{key}"
        if not isinstance(s, Mapping):
            problems.append(f"{where}: expected a table")
            continue
        extra = sorted(set(s) - SOURCE_KEYS)
        if extra:
            problems.append(f"{where}: unknown keys {extra}")
        if ("url" in s) == ("record" in s):
            problems.append(f"{where}: exactly one of url and record is required")
        if not _valid_date(s.get("looked_up")):
            problems.append(f"{where}: looked_up must be an ISO date")
        if not isinstance(s.get("title"), str) or not s.get("title"):
            problems.append(f"{where}: title is required")
        if "url" in s and not isinstance(s.get("quote"), str):
            problems.append(f"{where}: a url source needs a quote")
        looked = s.get("looked_up")
        out[key] = Source(
            key,
            s.get("url"),
            s.get("record"),
            looked.isoformat() if isinstance(looked, dt.date) else str(looked),
            str(s.get("title")),
            s.get("quote"),
        )
    return out


def _top(doc: Mapping[str, Any], path: Path, problems: list[str]) -> None:
    missing = [k for k in TOP_KEYS if k not in doc]
    extra = sorted(set(doc) - set(TOP_KEYS))
    if missing:
        problems.append(f"missing top-level keys {missing}")
    if extra:
        problems.append(f"unknown top-level keys {extra}")
    if doc.get("schema") != SCHEMA:
        problems.append(f"schema {doc.get('schema')!r} != {SCHEMA!r}")
    if doc.get("firm") not in FIRMS:
        problems.append(f"firm {doc.get('firm')!r} not in {FIRMS}")
    if doc.get("status") not in STATUSES:
        problems.append(f"status {doc.get('status')!r} not in {STATUSES}")
    for key in ("plan", "size", "label"):
        if not isinstance(doc.get(key), str) or not doc.get(key):
            problems.append(f"{key}: expected a non-empty string")
    stem = f"{doc.get('plan')}_{doc.get('size')}"
    if path.stem != stem:
        problems.append(f"file name {path.name!r} != {stem}.toml (plan id + size)")


def _version(raw: Any, problems: list[str]) -> tuple[int, str, str]:
    """(n, rules_read, read_in) of the ``version`` inline table; problems listed, (0, "", "") when broken."""
    if not isinstance(raw, Mapping):
        problems.append("version: expected { n = <int >= 1>, rules_read = <ISO date>, read_in = <record> }")
        return 0, "", ""
    extra = sorted(set(raw) - VERSION_KEYS)
    missing = sorted(VERSION_KEYS - set(raw))
    if extra or missing:
        problems.append(f"version: unknown keys {extra}, missing keys {missing}")
    n, read, where = raw.get("n"), raw.get("rules_read"), raw.get("read_in")
    ok = True
    if not _int_in(n, 1, None):
        problems.append(f"version.n {n!r} must be an int >= 1")
        ok = False
    if not _valid_date(read):
        problems.append(f"version.rules_read {read!r} must be an ISO date")
        ok = False
    if not isinstance(where, str) or not where.strip():
        problems.append("version.read_in must be a non-empty string")
        ok = False
    if not ok or not isinstance(n, int):
        return 0, "", ""
    return n, read.isoformat() if isinstance(read, dt.date) else str(read), str(where)


def _sections(doc: Mapping[str, Any], sources: Mapping[str, Any], problems: list[str]) -> dict[str, Leaf]:
    leaves: dict[str, Leaf] = {}
    eff = _leaf("effective_from", "date", doc.get("effective_from"), sources, problems)
    if eff is not None:
        leaves["effective_from"] = eff
    for section, fields in SECTION_FIELDS.items():
        block = doc.get(section)
        if not isinstance(block, Mapping):
            problems.append(f"[{section}]: missing or not a table")
            continue
        names = [n for n, _ in fields]
        for name in names:
            if name not in block:
                problems.append(f"{section}.{name}: missing")
        for name in sorted(set(block) - set(names)):
            problems.append(f"{section}.{name}: not a pfb-rules-1 field")
        for name, tag in fields:
            if name in block:
                leaf = _leaf(f"{section}.{name}", tag, block[name], sources, problems)
                if leaf is not None:
                    leaves[leaf.path] = leaf
    return leaves


def _known(leaves: Mapping[str, Leaf], path: str) -> tuple[bool, Any]:
    leaf = leaves.get(path)
    if leaf is None or leaf.is_unknown:
        return False, None
    return True, leaf.value


def _cross_checks(leaves: Mapping[str, Leaf], problems: list[str]) -> None:
    """Relations between fields, checked when both sides are known."""
    for sec in ("eval", "funded"):
        k1, cents = _known(leaves, f"{sec}.daily_loss_cents")
        k2, kind = _known(leaves, f"{sec}.daily_loss_kind")
        if k1 and k2 and (cents is None) != (kind == NONE):
            problems.append(f"{sec}.daily_loss_kind {kind!r} does not match daily_loss_cents {cents!r}")
        k1, dd = _known(leaves, f"{sec}.drawdown_cents")
        if k1 and dd <= 0:
            problems.append(f"{sec}.drawdown_cents must be > 0")
    k1, ctype = _known(leaves, "eval.consistency_type")
    k2, cbps = _known(leaves, "eval.consistency_bps")
    if k1 and k2 and ctype != NONE and cbps is None:
        problems.append(f"eval.consistency_type {ctype!r} needs consistency_bps")
    k1, wd = _known(leaves, "eval.min_winning_days")
    k2, thr = _known(leaves, "eval.winning_day_cents")
    if k1 and k2 and wd > 0 and thr is None:
        problems.append("eval.min_winning_days > 0 needs winning_day_cents")
    k1, wr = _known(leaves, "payout.win_days_required")
    k2, wthr = _known(leaves, "payout.win_day_cents")
    if k1 and k2 and wr > 0 and wthr is None:
        problems.append("payout.win_days_required > 0 needs win_day_cents")
    k1, wk = _known(leaves, "payout.request_weekday")
    if k1 and wk is not None and not 1 <= wk <= 7:
        problems.append(f"payout.request_weekday {wk} not an ISO weekday 1..7")
    k1, wcal = _known(leaves, "payout.window_calendar_days")
    if k1 and wcal is not None and wcal < 1:
        problems.append(f'payout.window_calendar_days {wcal} must be >= 1 (or "none")')
    k1, wcap = _known(leaves, "payout.window_cap_bps")
    if k1 and wcap is not None and wcap <= 0:
        problems.append(f'payout.window_cap_bps {wcap} must be > 0 (or "none")')
    k1, wfrom = _known(leaves, "payout.window_cap_from_payout")
    if k1 and wfrom < 1:
        problems.append(f"payout.window_cap_from_payout {wfrom} must be >= 1 (a 1-based payout index)")
    k1, billing = _known(leaves, "fees.billing")
    k2, period = _known(leaves, "fees.billing_period_calendar_days")
    if k1 and k2 and billing == "monthly" and not period:
        problems.append("fees.billing monthly needs billing_period_calendar_days > 0")


def parse_plan(doc: Mapping[str, Any], path: Path, sha256: str) -> PlanRules:
    """Validate a parsed TOML document; raise PlanConfigError listing every problem."""
    problems: list[str] = []
    _top(doc, path, problems)
    sources = _sources(doc.get("sources"), problems)
    raw_sources = doc.get("sources")
    leaves = _sections(doc, raw_sources if isinstance(raw_sources, Mapping) else {}, problems)
    _cross_checks(leaves, problems)
    version, read, where = _version(doc.get("version"), problems) if "version" in doc else (0, "", "")
    eff = leaves.get("effective_from")
    if read and eff is not None and not eff.is_unknown and str(eff.value) > read:
        problems.append(f"version.rules_read {read} is before effective_from {eff.value}")
    if problems:
        raise PlanConfigError(str(path), problems)
    return PlanRules(
        path=str(path),
        sha256=sha256,
        firm=str(doc["firm"]),
        plan=str(doc["plan"]),
        size=str(doc["size"]),
        label=str(doc["label"]),
        status=str(doc["status"]),
        leaves=MappingProxyType(leaves),
        sources=MappingProxyType(sources),
        file_sha256=sha256,
        version=version,
        rules_read=read,
        read_in=where,
    )


def load_plan(path: str | Path) -> PlanRules:
    """Read, validate and freeze one pfb-rules-1 plan file (sha256 of the file's bytes on the result)."""
    p = Path(path)
    data = p.read_bytes()
    try:
        doc = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise PlanConfigError(str(p), [f"not valid UTF-8 TOML: {exc}"]) from exc
    return parse_plan(doc, p, hashlib.sha256(data).hexdigest())


def with_values(plan: PlanRules, values: Mapping[str, Any]) -> PlanRules:
    """A new PlanRules with leaves replaced (sensitivity runs and tests); values use the TOML forms
    ("none" for the marker, lists for scaling and caps) and are type-checked like the file. The replacements
    are listed in ``overrides``; ``file_sha256`` keeps the file's hash and ``sha256`` becomes
    ``overridden_sha256(file_sha256, overrides)``, so a sensitivity plan never carries the file's hash with
    different rules (N10)."""
    problems: list[str] = []
    leaves = dict(plan.leaves)
    for path, raw in values.items():
        tag = FIELD_TAGS.get(path)
        if tag is None:
            problems.append(f"{path}: not a pfb-rules-1 field")
            continue
        ok, parsed = _check_value(path, tag, raw, problems)
        if ok:
            old = leaves[path]
            leaves[path] = Leaf(path, parsed, None, old.src or "override", "override", (), "exact")
    _cross_checks(leaves, problems)
    if problems:
        raise PlanConfigError(plan.path, problems)
    file_sha = plan.file_sha256 or plan.sha256
    overrides = (*plan.overrides, *sorted(values.items()))
    return PlanRules(
        path=plan.path,
        sha256=overridden_sha256(file_sha, overrides),
        firm=plan.firm,
        plan=plan.plan,
        size=plan.size,
        label=plan.label,
        status=plan.status,
        leaves=MappingProxyType(leaves),
        sources=plan.sources,
        overrides=overrides,
        file_sha256=file_sha,
        version=plan.version,
        rules_read=plan.rules_read,
        read_in=plan.read_in,
    )


def overridden_sha256(file_sha256: str, overrides: tuple[tuple[str, Any], ...]) -> str:
    """The hash of a plan with overrides: sha256 over the file's hash and the overrides in order (canonical
    JSON). No overrides gives the file's hash itself."""
    if not overrides:
        return file_sha256
    payload = json.dumps([file_sha256, [[k, v] for k, v in overrides]], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------- dated versions ----
@dataclass(frozen=True)
class PlanVersions:
    """Every dated version of one plan, oldest first; the newest is the plan file itself.

    ``in_force(yyyymmdd)`` is the index of the version in force on a trade date: the last one whose
    effective_from is on or before it; a date before the earliest version gets the earliest (index 0) and is
    counted by ``before_earliest`` so a result can say so. Dates are int yyyymmdd (no float, no clock)."""

    versions: tuple[PlanRules, ...]

    @property
    def newest(self) -> PlanRules:
        return self.versions[-1]

    @property
    def earliest(self) -> PlanRules:
        return self.versions[0]

    def in_force(self, yyyymmdd: int) -> int:
        index = 0
        for i, plan in enumerate(self.versions):
            if plan.effective_from_int <= int(yyyymmdd):
                index = i
        return index

    def before_earliest(self, yyyymmdd: int) -> bool:
        return int(yyyymmdd) < self.earliest.effective_from_int


def version_paths(plan_file: str | Path) -> list[Path]:
    """The plan file's older dated versions: <firm dir>/versions/<effective_from>/<same file name>."""
    p = Path(plan_file)
    vdir = p.parent / VERSIONS_DIR
    if not vdir.is_dir():
        return []
    return sorted(d / p.name for d in vdir.iterdir() if d.is_dir() and (d / p.name).is_file())


def check_versions(plans: list[PlanRules], newest_path: str) -> PlanVersions:
    """Order a plan's versions by effective_from and check them: one firm, plan and size; distinct dates;
    version.n ascending with effective_from; the plan file itself is the newest. PlanConfigError otherwise."""
    problems: list[str] = []
    ordered = sorted(plans, key=lambda r: r.effective_from)
    names = {(r.firm, r.plan, r.size) for r in ordered}
    if len(names) != 1:
        problems.append(f"versions name different plans {sorted(names)}")
    dates = [r.effective_from for r in ordered]
    if len(set(dates)) != len(dates):
        problems.append(f"two versions share an effective_from: {dates}")
    numbers = [r.version for r in ordered]
    if any(b <= a for a, b in zip(numbers, numbers[1:], strict=False)):
        problems.append(f"version.n {numbers} does not ascend with effective_from {dates}")
    if ordered and ordered[-1].path != newest_path:
        problems.append(f"the plan file is not the newest version: {ordered[-1].path} is ({dates[-1]})")
    for r in ordered:
        if Path(r.path).parent.name not in (r.effective_from, Path(newest_path).parent.name):
            problems.append(f"{r.path}: an older version lives under {VERSIONS_DIR}/{r.effective_from}/")
    if problems:
        raise PlanConfigError(newest_path, problems)
    return PlanVersions(tuple(ordered))


def load_versions(plan_file: str | Path) -> PlanVersions:
    """The plan file and its dated older versions (version_paths), validated and ordered (check_versions)."""
    p = Path(plan_file)
    plans = [load_plan(v) for v in version_paths(p)] + [load_plan(p)]
    return check_versions(plans, str(p))
