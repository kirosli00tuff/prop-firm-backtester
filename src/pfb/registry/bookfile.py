"""Book hypothesis files, schema "pfb.hypothesis/3": the test statistic is the daily net of a POOLED book of
K member configurations. Parsing, the derived pooled power inputs, and the record checks replay
applies to a book's hypothesis record and to each member's begin record.

THE COUNTING RULE (where each clause is enforced):
1. K >= 2 members, each one configuration (strategy sha256, root, params, window, contracts); budget == K;
   purpose "research": parse_book (file) and check_book_record (replay).
2. Each member runs as its own trial (one begin record, N + 1 when its config_id is new to the book): a
   begin whose (strategy sha256, root, params, window) is not exactly one declared member is refused
   (member_index, called by records._check_trial at begin and in replay); a member already counted under
   another config_id is refused too (MemberAlreadyCounted), so K slots are K members.
3. The pooled statistic adds no record and nothing to N: pfb.registry.api.book_runs is read-only and refuses
   (BookIncomplete) unless every member has an ok result.
4. The file's sha256 is pinned by the hypothesis record; a begin under a changed file is HypothesisChanged
   (records._check_trial), and a changed file registered under a new id is a new hypothesis whose config_ids
   (they include the hypothesis sha256) are all new.
5. The power check is the book's: Y is the union of the member windows' days / 365.25 and the pooled inputs
   are DERIVED from the members (derived_power); the [power] table may state a lower trades_per_year or a
   higher trade_sd_cents, never the reverse (require_not_flattered, PowerFlattered; exact Fractions, so no
   float decides the boundary). The check itself runs on the stated (never more flattering) inputs:
   pfb.registry.powercheck.book_power_at.
6. suite.neighbourhood must be false and power.sd_source "stated" (parse_book, check_book_record).
7. Members are not assumed independent (replaces clause 5's pooled sd; m is unchanged): members whose
   windows share a day AND that share a root family (a micro and its full-size sibling, by
   config/instruments.toml exposure: root_family) or a strategy path or sha256 are tied; a tie group
   (member_groups, union-find) moves together (correlation 1); across groups the correlation is the STATED
   [power] member_correlation (0 to 1) with a non-empty member_correlation_reason (_correlation,
   correlation_ok). With a_i = sqrt(m_i Y_i) s_i and A_g the sum over group g, s = sqrt((sum_g A_g^2 + rho
   sum_{g != h} A_g A_h) / sum_i m_i Y_i) (derived_power); every square root is rounded UP at 1 /
   SQRT_SCALE in integer arithmetic (sqrt_up), and the stated sd must be at least that figure
   (require_not_flattered). Registration (bookreg.book_body), replay (check_book_record, from the recorded
   members, families and rho) and so every begin (which reads the replayed record) use the same derived
   figure.

File keys, exactly: schema, id, purpose, statement, budget, no_signal_reason, cost, power, suite, secondary,
book, optional lookahead_seed. [book]: account ("<firm>/<plan file stem>"), cap ("plan"), statistic
("daily_net"), and [[book.member]] rows of exactly strategy, strategy_sha256, root, contracts (int >= 1),
params (a table of scalars), window (a table of first and last), trades_per_year (> 0), trade_sd_cents (int
>= 1). No two members share (strategy sha256, root, params, window): a begin names a member by that tuple.
What needs the strategies' SPECs, the data catalog, the ledger or the allowlist is checked at registration
(pfb.runner.register, pfb.registry.api.register_book).
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from fractions import Fraction
from functools import cache
from types import MappingProxyType
from typing import Any

from pfb.analysis.spec import SuiteSpec
from pfb.registry.chain import canonical_bytes, sha256_hex
from pfb.registry.errors import (
    BookInvalid,
    ChainBroken,
    MemberAlreadyCounted,
    NotABookMember,
    PowerFlattered,
)
from pfb.registry.hypfile import (
    BOOK_SCHEMA_NAME,
    DEFAULT_LOOKAHEAD_SEED,
    OPTIONAL_KEYS,
    PLAN_RE,
    PURPOSE_RESEARCH,
    SD_STATED,
    CostDecl,
    PowerDecl,
    SecondaryDecl,
    _cost,  # pyright: ignore[reportPrivateUsage]
    _int,  # pyright: ignore[reportPrivateUsage]
    _number,  # pyright: ignore[reportPrivateUsage]
    _power,  # pyright: ignore[reportPrivateUsage]
    _secondary,  # pyright: ignore[reportPrivateUsage]
    _string,  # pyright: ignore[reportPrivateUsage]
    _suite,  # pyright: ignore[reportPrivateUsage]
    lookahead_seed_fields,
)
from pfb.registry.inputs import ID_RE, SHA_RE, GridValue, iso_date

BOOK_SCHEMA = BOOK_SCHEMA_NAME
BOOK_KEYS = frozenset(
    {
        "schema", "id", "purpose", "statement", "budget", "no_signal_reason", "cost", "power", "suite",
        "secondary", "book",
    }
)  # fmt: skip
BOOK_TABLE_KEYS = frozenset({"account", "cap", "statistic", "member"})
MEMBER_KEYS = frozenset(
    {
        "strategy",
        "strategy_sha256",
        "root",
        "contracts",
        "params",
        "window",
        "trades_per_year",
        "trade_sd_cents",
    }
)
WINDOW_KEYS = frozenset({"first", "last"})
CAP_PLAN = "plan"
STATISTIC_DAILY_NET = "daily_net"
MIN_MEMBERS = 2
# The record: a book's hypothesis record carries these in "book"; each member as MEMBER_RECORD_KEYS.
CORRELATION_KEYS = ("member_correlation", "member_correlation_reason")  # clause 7b: in a book's [power]
BOOK_RECORD_KEYS = frozenset({"account", "cap", "statistic", "members", "derived", *CORRELATION_KEYS})
MEMBER_RECORD_KEYS = frozenset(
    {
        "strategy_path", "strategy_sha256", "root", "contracts", "params", "window", "trades_per_year",
        "trade_sd_cents", "bar_minutes", "session", "root_family",
    }
)  # fmt: skip
SCHEMA2_ONLY_NULL = ("strategy_path", "strategy_sha256", "bar_minutes", "session", "grid")


@dataclass(frozen=True)
class BookMember:
    """One member configuration as the file declares it."""

    strategy: str
    strategy_sha256: str
    root: str
    contracts: int
    params: Mapping[str, GridValue]
    window: tuple[str, str]
    trades_per_year: float
    trade_sd_cents: int

    def identity(self) -> bytes:
        return member_identity(self.strategy_sha256, self.root, self.params, self.window)


@dataclass(frozen=True)
class BookHypothesis:
    """A parsed schema-3 book file; sha256 is that of the file's bytes, budget is K = len(members)."""

    id: str
    sha256: str
    purpose: str
    statement: str
    budget: int
    no_signal_reason: str
    cost: CostDecl
    power: PowerDecl
    suite: SuiteSpec
    secondary: SecondaryDecl
    account: str
    cap: str
    statistic: str
    members: tuple[BookMember, ...]
    member_correlation: float  # clause 7b: the stated correlation across tie groups, 0 to 1
    member_correlation_reason: str
    lookahead_seed: int = DEFAULT_LOOKAHEAD_SEED
    lookahead_seed_reason: str = ""  # required when the seed is not the program's

    @property
    def windows(self) -> tuple[tuple[str, str], ...]:
        """The distinct member windows in member order (`--window K` indexes this list)."""
        return tuple(dict.fromkeys(m.window for m in self.members))

    @property
    def roots(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(m.root for m in self.members))

    @property
    def grid(self) -> Mapping[str, tuple[GridValue, ...]]:
        """A book declares no grid: its configurations are its members (an empty mapping, so a grid check
        against it refuses every params table)."""
        return MappingProxyType({})


def member_identity(
    strategy_sha256: str, root: str, params: Mapping[str, Any], window: Sequence[str]
) -> bytes:
    """What names a member at a begin: (strategy sha256, root, params, [first, last]), canonical."""
    return canonical_bytes([strategy_sha256, root, dict(params), list(window)])


def window_days(window: Sequence[str]) -> int:
    first, last = window
    return (date.fromisoformat(last) - date.fromisoformat(first)).days + 1


def union_days(windows: Sequence[Sequence[str]]) -> int:
    """Inclusive calendar days that lie in at least one window."""
    if not windows:
        raise ValueError("union_days needs at least one window")
    spans = sorted((date.fromisoformat(w[0]), date.fromisoformat(w[1])) for w in windows)
    total, (start, end) = 0, spans[0]
    for first, last in spans[1:]:
        if first > end:
            total += (end - start).days + 1
            start, end = first, last
        elif last > end:
            end = last
    return total + (end - start).days + 1


SQRT_SCALE = 10**6  # clause 7d: every square root is rounded UP to a multiple of 1 / SQRT_SCALE


@dataclass(frozen=True)
class PoolMember:
    """What the pooled power derivation reads of one member: its stated figures, window and the keys that
    tie it to another member (root family, strategy path, strategy sha256)."""

    trades_per_year: float
    trade_sd_cents: int
    window: tuple[str, str]
    root_family: str
    strategy_path: str
    strategy_sha256: str


@dataclass(frozen=True)
class DerivedPower:
    """The pooled power inputs derived from the members (clauses 5 and 7), exact: trades per year m =
    sum(m_i days_i) / union_days; per-trade sd s = sqrt((sum_g A_g^2 + rho sum_{g != h} A_g A_h) /
    sum_i m_i days_i) with a_i = sqrt(m_i days_i) s_i and A_g the sum of a_i over tie group g. The 365.25 of
    Y_i and Y cancels, so days are used directly. Every square root is rounded UP at SQRT_SCALE, so
    trade_sd_cents is never below the true figure."""

    union_days: int
    trades_per_year: Fraction
    trade_sd_cents: Fraction  # rounded up at SQRT_SCALE
    groups: tuple[tuple[int, ...], ...]  # 0-based member indexes, in order of their first member

    def to_json(self) -> dict[str, Any]:
        return {
            "union_days": self.union_days,
            "trades_per_year": float(self.trades_per_year),
            "trade_sd_cents": float(self.trade_sd_cents),
            "groups": [list(g) for g in self.groups],
        }


def sqrt_up(value: Fraction) -> Fraction:
    """The smallest multiple of 1 / SQRT_SCALE that is >= sqrt(value): integer arithmetic only."""
    if value < 0:
        raise ValueError("sqrt_up of a negative value")
    scaled = value * SQRT_SCALE * SQRT_SCALE
    target = scaled.numerator // scaled.denominator + (1 if scaled.denominator != 1 else 0)
    root = math.isqrt(target)
    if root * root < target:
        root += 1
    return Fraction(root, SQRT_SCALE)


@cache
def _exposures() -> Mapping[str, str]:
    from pfb.instruments.table import load_instruments

    table = load_instruments()
    return MappingProxyType({root: table.get(root).exposure for root in table.roots()})


def root_family(root: str) -> str:
    """Clause 7a: a micro and its full-size sibling count as one root. config/instruments.toml's own
    mapping is its `exposure` (NQ and MNQ: nasdaq100); a root the table does not know is its own family."""
    return _exposures().get(root, root)


def windows_share_a_day(a: Sequence[str], b: Sequence[str]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def tied(a: PoolMember, b: PoolMember) -> bool:
    """Clause 7a: windows that share a day AND a shared root family, strategy path or strategy sha256."""
    shared = (
        a.root_family == b.root_family
        or a.strategy_path == b.strategy_path
        or a.strategy_sha256 == b.strategy_sha256
    )
    return shared and windows_share_a_day(a.window, b.window)


def member_groups(members: Sequence[PoolMember]) -> tuple[tuple[int, ...], ...]:
    """The connected sets of tied members (union-find), each sorted, in order of their first member."""
    parent = list(range(len(members)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            if tied(members[i], members[j]):
                parent[find(j)] = find(i)
    groups: dict[int, list[int]] = {}
    for i in range(len(members)):
        groups.setdefault(find(i), []).append(i)
    return tuple(tuple(g) for g in sorted(groups.values()))


def derived_power(members: Sequence[PoolMember], rho: Fraction) -> DerivedPower:
    """Clauses 5 and 7c: exact Fractions of the stated floats; square roots rounded up (sqrt_up)."""
    if not members:
        raise ValueError("a book has members")
    if not 0 <= rho <= 1:
        raise ValueError("member_correlation lies in [0, 1]")
    weights = [Fraction(m.trades_per_year) * window_days(m.window) for m in members]
    total = sum(weights, Fraction(0))
    if total <= 0:
        raise ValueError("a book's members trade")
    a = [sqrt_up(w * m.trade_sd_cents * m.trade_sd_cents) for w, m in zip(weights, members, strict=True)]
    groups = member_groups(members)
    sums = [sum((a[i] for i in g), Fraction(0)) for g in groups]
    squares = sum((x * x for x in sums), Fraction(0))
    linear = sum(sums, Fraction(0))
    # sum_g A_g^2 + rho sum_{g != h} A_g A_h = (1 - rho) sum_g A_g^2 + rho (sum_g A_g)^2
    numerator = (1 - rho) * squares + rho * linear * linear
    days = union_days([m.window for m in members])
    return DerivedPower(
        union_days=days,
        trades_per_year=total / days,
        trade_sd_cents=sqrt_up(numerator / total),
        groups=groups,
    )


def require_not_flattered(stated_rate: float, stated_sd: int, derived: DerivedPower, where: str) -> None:
    """Clauses 5 and 7d: the stated [power] may be lower in trades per year and higher in sd, never the
    reverse; the stated sd must be at least the derived one rounded up (no float decides it)."""
    if Fraction(stated_rate) > derived.trades_per_year:
        raise PowerFlattered(
            f"{where}: power.trades_per_year {stated_rate} is above the {float(derived.trades_per_year):.6f} "
            "derived from the members (sum m_i Y_i / Y over the union of the member windows)"
        )
    if Fraction(stated_sd) < derived.trade_sd_cents:
        raise PowerFlattered(
            f"{where}: power.trade_sd_cents {stated_sd} is below the {float(derived.trade_sd_cents):.6f} "
            f"derived from the members (tie groups {[list(g) for g in derived.groups]}, member_correlation "
            "across groups; clause 7)"
        )


def _params(raw: object, what: str) -> Mapping[str, GridValue]:
    if not isinstance(raw, dict) or not all(
        isinstance(k, str) and isinstance(v, str | int | float | bool) for k, v in raw.items()
    ):
        raise BookInvalid(f"{what}: params must be a table of scalars (one value per SPEC parameter)")
    if any(isinstance(v, float) and not math.isfinite(v) for v in raw.values()):
        raise BookInvalid(f"{what}: params must be finite")
    return MappingProxyType(dict(raw))


def _window(raw: object, what: str) -> tuple[str, str]:
    if not isinstance(raw, dict) or set(raw) != WINDOW_KEYS:
        raise BookInvalid(f"{what}: window is a table of exactly first and last")
    first = iso_date(raw["first"], f"{what} window first", BookInvalid)
    last = iso_date(raw["last"], f"{what} window last", BookInvalid)
    if first > last:
        raise BookInvalid(f"{what}: window first is after last")
    return first, last


def _member(raw: object, what: str) -> BookMember:
    if not isinstance(raw, dict) or set(raw) != MEMBER_KEYS:
        raise BookInvalid(f"{what} has exactly the keys {sorted(MEMBER_KEYS)}")
    sha = _string(raw["strategy_sha256"], f"{what}: strategy_sha256")
    if not SHA_RE.fullmatch(sha):
        raise BookInvalid(f"{what}: strategy_sha256 is not a sha256 (64 lowercase hex)")
    rate = raw["trades_per_year"]
    if not _number(rate) or not math.isfinite(float(rate)) or float(rate) <= 0:
        raise BookInvalid(f"{what}: trades_per_year must be a number > 0")
    return BookMember(
        strategy=_string(raw["strategy"], f"{what}: strategy"),
        strategy_sha256=sha,
        root=_string(raw["root"], f"{what}: root"),
        contracts=_int(raw["contracts"], f"{what}: contracts", 1),
        params=_params(raw["params"], what),
        window=_window(raw["window"], what),
        trades_per_year=float(rate),
        trade_sd_cents=_int(raw["trade_sd_cents"], f"{what}: trade_sd_cents", 1),
    )


def _book_table(raw: object, where: str) -> tuple[str, str, str, tuple[BookMember, ...]]:
    if not isinstance(raw, dict) or set(raw) != BOOK_TABLE_KEYS:
        raise BookInvalid(f"{where}: [book] has exactly the keys {sorted(BOOK_TABLE_KEYS)}")
    account = _string(raw["account"], f"{where}: book.account")
    if not PLAN_RE.fullmatch(account):
        raise BookInvalid(f'{where}: book.account must be "<firm>/<plan file stem>"')
    if raw["cap"] != CAP_PLAN:
        raise BookInvalid(f'{where}: book.cap must be "{CAP_PLAN}" (the plan\'s own contract caps)')
    if raw["statistic"] != STATISTIC_DAILY_NET:
        raise BookInvalid(f'{where}: book.statistic must be "{STATISTIC_DAILY_NET}"')
    rows = raw["member"]
    if not isinstance(rows, list) or len(rows) < MIN_MEMBERS:
        raise BookInvalid(f"{where}: a book declares at least {MIN_MEMBERS} [[book.member]] rows")
    members = tuple(_member(row, f"{where}: member {i + 1}") for i, row in enumerate(rows))
    seen: dict[bytes, int] = {}
    for i, member in enumerate(members):
        other = seen.setdefault(member.identity(), i)
        if other != i:
            raise BookInvalid(
                f"{where}: members {other + 1} and {i + 1} share strategy, root, params and window; no two "
                "members are equal (a begin names a member by that tuple)"
            )
    return account, raw["cap"], raw["statistic"], members


def _header(raw: Mapping[str, Any], where: str) -> tuple[str, str]:
    hyp_id = _string(raw["id"], f"{where}: id")
    if not ID_RE.fullmatch(hyp_id):
        raise BookInvalid(f"{where}: id {hyp_id!r} must match {ID_RE.pattern}")
    if raw["purpose"] != PURPOSE_RESEARCH:
        raise BookInvalid(f'{where}: a book\'s purpose must be "{PURPOSE_RESEARCH}"')
    return hyp_id, PURPOSE_RESEARCH


def pool_members(members: Sequence[BookMember]) -> list[PoolMember]:
    """The file's members as the derivation reads them, each root mapped to its family."""
    return [
        PoolMember(m.trades_per_year, m.trade_sd_cents, m.window, root_family(m.root), m.strategy,
                   m.strategy_sha256)
        for m in members
    ]  # fmt: skip


def correlation_ok(rho: object, reason: object) -> bool:
    """Clause 7b: member_correlation a finite number from 0 to 1, member_correlation_reason non-empty."""
    return (
        _number(rho)
        and math.isfinite(float(rho))  # type: ignore[arg-type]
        and 0 <= float(rho) <= 1  # type: ignore[arg-type]
        and isinstance(reason, str)
        and bool(reason.strip())
    )


def _correlation(table: object, where: str) -> tuple[float, str]:
    if not isinstance(table, dict) or any(k not in table for k in CORRELATION_KEYS):
        raise BookInvalid(
            f"{where}: a book's [power] states member_correlation (0 to 1, the correlation across tie "
            "groups) and member_correlation_reason (clause 7b)"
        )
    rho, reason = table["member_correlation"], table["member_correlation_reason"]
    if not correlation_ok(rho, reason):
        raise BookInvalid(
            f"{where}: power.member_correlation must be a number from 0 to 1 and "
            "power.member_correlation_reason a non-empty string"
        )
    return float(rho), reason


def _book_power(
    raw: Mapping[str, Any], members: Sequence[BookMember], where: str
) -> tuple[PowerDecl, float, str]:
    rho, reason = _correlation(raw["power"], where)
    table = {k: v for k, v in raw["power"].items() if k not in CORRELATION_KEYS}
    power = _power({**raw, "power": table}, where)
    if power.sd_source != SD_STATED:
        raise BookInvalid(f'{where}: a book\'s power.sd_source must be "{SD_STATED}"')
    derived = derived_power(pool_members(members), Fraction(rho))
    require_not_flattered(power.trades_per_year, power.trade_sd_cents, derived, where)
    return power, rho, reason


def parse_book(data: bytes, where: str) -> BookHypothesis:
    """Parse and shape-check a schema-3 file (BookInvalid, PowerFlattered, HypothesisInvalid,
    ZeroSlippage)."""
    try:
        raw = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise BookInvalid(f"{where} is not UTF-8 TOML: {exc}") from exc
    if raw.get("schema") != BOOK_SCHEMA:
        raise BookInvalid(f"{where}: schema must be {BOOK_SCHEMA!r}")
    missing = sorted(BOOK_KEYS - set(raw))
    extra = sorted(set(raw) - BOOK_KEYS - OPTIONAL_KEYS)
    if missing or extra:
        raise BookInvalid(f"{where}: keys missing {missing}, extra {extra}")
    hyp_id, purpose = _header(raw, where)
    account, cap, statistic, members = _book_table(raw["book"], where)
    budget = _int(raw["budget"], f"{where}: budget", 1)
    if budget != len(members):
        raise BookInvalid(f"{where}: budget {budget} must equal the number of members K = {len(members)}")
    suite = _suite(raw, where)
    if suite.neighbourhood:
        raise BookInvalid(f"{where}: a book has no neighbourhood (suite.neighbourhood must be false)")
    power, rho, reason = _book_power(raw, members, where)
    seed, seed_reason = lookahead_seed_fields(raw, where)
    return BookHypothesis(
        id=hyp_id,
        sha256=sha256_hex(data),
        purpose=purpose,
        statement=_string(raw["statement"], f"{where}: statement"),
        budget=budget,
        no_signal_reason=_string(raw["no_signal_reason"], f"{where}: no_signal_reason", empty_ok=True),
        cost=_cost(raw, where),
        power=power,
        suite=suite,
        secondary=_secondary(raw, where),
        account=account,
        cap=cap,
        statistic=statistic,
        members=members,
        member_correlation=rho,
        member_correlation_reason=reason,
        lookahead_seed=seed,
        lookahead_seed_reason=seed_reason,
    )


# ------------------------------------------------------------------------- record checks (replay) ----
def is_book_record(record: Mapping[str, Any]) -> bool:
    return record.get("book") is not None


def member_windows(record: Mapping[str, Any]) -> list[list[str]]:
    return [list(m["window"]) for m in record["book"]["members"]]


def _member_record_ok(member: object) -> bool:
    if not isinstance(member, dict) or set(member) != MEMBER_RECORD_KEYS:
        return False
    window, rate = member["window"], member["trades_per_year"]
    return (
        all(
            isinstance(member[k], str) and member[k]
            for k in ("strategy_path", "root", "session", "root_family")
        )
        and isinstance(member["strategy_sha256"], str)
        and bool(SHA_RE.fullmatch(member["strategy_sha256"]))
        and all(_int_ok(member[k], 1) for k in ("contracts", "trade_sd_cents", "bar_minutes"))
        and isinstance(member["params"], dict)
        and isinstance(window, list)
        and len(window) == 2
        and all(isinstance(d, str) for d in window)
        and window[0] <= window[1]
        and _number(rate)
        and math.isfinite(float(rate))
        and float(rate) > 0
    )


def _int_ok(value: object, minimum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _book_record_shape(record: Mapping[str, Any], where: str) -> list[Mapping[str, Any]]:
    book = record["book"]
    members = book.get("members") if isinstance(book, dict) else None
    shaped = (
        isinstance(book, dict)
        and set(book) == BOOK_RECORD_KEYS
        and isinstance(book["account"], str)
        and bool(PLAN_RE.fullmatch(book["account"]))
        and book["cap"] == CAP_PLAN
        and book["statistic"] == STATISTIC_DAILY_NET
        and isinstance(members, list)
        and len(members) >= MIN_MEMBERS
        and all(_member_record_ok(m) for m in members)
        and all(record[k] is None for k in SCHEMA2_ONLY_NULL)
        and correlation_ok(book["member_correlation"], book["member_correlation_reason"])
    )
    if not shaped or not isinstance(members, list):
        raise ChainBroken(
            f"{where}: a book record holds account, cap plan, statistic daily_net, derived, "
            "member_correlation (0 to 1) with a non-empty member_correlation_reason and >= "
            f"{MIN_MEMBERS} members of exactly {sorted(MEMBER_RECORD_KEYS)}; {list(SCHEMA2_ONLY_NULL)} null"
        )
    return members


def _record_pool(members: Sequence[Mapping[str, Any]], where: str) -> list[PoolMember]:
    """The record's members as the derivation reads them; one root never maps to two families."""
    families: dict[str, str] = {}
    for m in members:
        if families.setdefault(m["root"], m["root_family"]) != m["root_family"]:
            raise ChainBroken(f"{where}: root {m['root']!r} is recorded in two root families")
    return [
        PoolMember(float(m["trades_per_year"]), m["trade_sd_cents"], (m["window"][0], m["window"][1]),
                   m["root_family"], m["strategy_path"], m["strategy_sha256"])
        for m in members
    ]  # fmt: skip


def check_book_record(record: Mapping[str, Any], where: str) -> None:
    """Replay of a book's hypothesis record: clauses 1, 5 and 7 (derived inputs from the members, their tie
    groups and the recorded member_correlation, never flattered) and 6, the member tuples distinct, roots and
    windows the members' own. The power figures are checked by the caller."""
    members = _book_record_shape(record, where)
    if record["purpose"] != PURPOSE_RESEARCH or record["budget"] != len(members):
        raise ChainBroken(f"{where}: a book is purpose research with budget = K = {len(members)}")
    identities = {member_identity(m["strategy_sha256"], m["root"], m["params"], m["window"]) for m in members}
    if len(identities) != len(members):
        raise ChainBroken(f"{where}: two book members share strategy, root, params and window")
    if record["roots"] != list(dict.fromkeys(m["root"] for m in members)) or record["windows"] != [
        list(w) for w in dict.fromkeys(tuple(m["window"]) for m in members)
    ]:
        raise ChainBroken(f"{where}: a book record's roots and windows are its members' own, in member order")
    suite, power = record["suite"], record["power"]
    if not isinstance(suite, dict) or suite.get("neighbourhood") is not False:
        raise ChainBroken(f"{where}: a book has no neighbourhood")
    if power.get("sd_source") != SD_STATED:
        raise ChainBroken(f'{where}: a book\'s sd_source is "{SD_STATED}"')
    derived = derived_power(_record_pool(members, where), Fraction(record["book"]["member_correlation"]))
    if record["book"]["derived"] != derived.to_json():
        raise ChainBroken(f"{where}: the book's derived power inputs differ from the members' recomputed")
    require_not_flattered(float(power["trades_per_year"]), int(power["trade_sd_cents"]), derived, where)


def member_index(begin: Mapping[str, Any], hyp: Mapping[str, Any], where: str) -> int:
    """Clause 2: the one declared member a begin names by (strategy sha256, root, params, window), also
    matched on strategy path and SPEC session; NotABookMember unless exactly one matches."""
    mine = member_identity(
        begin["strategy_sha256"], begin["root"], begin["params"], [begin["first"], begin["last"]]
    )
    found = [
        i
        for i, m in enumerate(hyp["book"]["members"])
        if member_identity(m["strategy_sha256"], m["root"], m["params"], m["window"]) == mine
        and m["strategy_path"] == begin["strategy_path"]
        and m["session"] == begin["session"]
    ]
    if len(found) != 1:
        raise NotABookMember(
            f"{where}: (strategy, root, params, window) is not exactly one declared member of book "
            f"{hyp['hypothesis_id']!r}"
        )
    return found[0]


def require_member_uncounted(
    counted: Mapping[int, str], index: int, configuration: str, hyp_id: str, where: str
) -> None:
    """Clause 2 (K slots are K members): a member already counted under another config_id is refused."""
    earlier = counted.get(index)
    if earlier is not None and earlier != configuration:
        raise MemberAlreadyCounted(
            f"{where}: member {index + 1} of book {hyp_id!r} already counted under another configuration "
            "(other code, data or calendars); a book member is counted once"
        )
