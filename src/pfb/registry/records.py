"""Record kinds, their exact key sets, configuration ids, and the replay that derives N and verifies every
link.

Kinds (ADR 13, A10): genesis, sync, hypothesis, trial, engine_check, synthetic, result. N = n_pe (upstream's
current_n
at the last genesis or sync) + n_local (configurations new to their hypothesis over all trial records). Replay
re-derives each record's n_after from the records before it and refuses any difference, so a valid file cannot
carry a wrong N. The same replay validates a record before it is appended (chain.locked_append).

Code identity. A begin record of every run kind carries code_sha256
(pfb.registry.codehash); config_id includes it, so for a trial an exact replay adds nothing only with the
same inputs AND the same code. Schema: the string stays "pfb.registry/1". Begin records written before the
field existed (LEGACY_RUN_KEYS: the repository ledger's line 2, engine check E3) remain valid as a legacy
PREFIX only: once any begin record carries code_sha256, a later begin record without it is refused. A
legacy record's config_id is the old tuple and its engine-check entry the old nine keys; a current one's
entry also declares session, seed and params.

tiers (docs/TRIALS.md section 3). Begin records: tier 1 (LEGACY_RUN_KEYS), tier 2 (+ code_sha256),
tier 3 (+ calendars_sha256, lookahead_signal, power); result records: tier 1, tier 2 (+ begin_sha256). Within
each family the tier never falls, so an older tier is valid only as a prefix. config_id includes
calendars_sha256 when the record carries it. Every trial is tier 3 and is checked against its schema-2
hypothesis record (section 4);
rule C5 binds hypothesis and engine_check records both ways. A hypothesis record's power figures and a trial's
are recomputed (pfb.registry.powercheck) and must equal what was written and pass: each declared window on its
own, the shortest one's figures stored, and a trial at its own window. An sd_source "run:<run_id>" is bound to
that run's realized power figures (and, for a synthetic run, to the hypothesis's strategy).

Books (schema "pfb.hypothesis/3"): hypothesis records gain a tier 2 with the key "book" (null for
a schema-2 hypothesis, the book object for a book; the tier never falls). A book record is checked by
pfb.registry.bookfile.check_book_record and its power at N + K over the union of its member windows; a trial
under a book is exactly one declared member (bookfile.member_index), a member is counted under one config_id
only (MemberAlreadyCounted), and its begin power is the BOOK's at n_before + K - members already counted.

Hypothesis records gain a tier 3 (lookahead_seed, lookahead_seed_reason; schema 2 and books
alike): a seed other than the program seed needs a non-empty reason (LookaheadSeedUnexplained), a trial under
such a record runs at its seed (SeedMismatch), and a purpose TEST record is refused while another TEST
hypothesis is registered and not spent (OpenTestHypothesis; spent = budget counted, _Replay.spent). Result
records gain a tier 3 (artifact_sha256: each artifact name, exactly the record's list, to its sha256). Records
of an older tier are never checked against these rules; the tier never falls, so they are a prefix only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from pfb.analysis.power import ALPHA_RULE, LEGACY_ALPHA_RULE, TRADING_DAYS_PER_YEAR
from pfb.registry.bookfile import (
    check_book_record,
    is_book_record,
    member_index,
    member_windows,
    require_member_uncounted,
)
from pfb.registry.chain import canonical_bytes, sha256_hex
from pfb.registry.errors import (
    BudgetExceeded,
    ChainBroken,
    ChecksNotPassed,
    DirtyTree,
    EngineCheckNotDeclared,
    EngineCheckStrategy,
    FinishNotPermitted,
    HypothesisChanged,
    HypothesisInvalid,
    HypothesisNotRegistered,
    LookaheadSeedUnexplained,
    NoSignalUnexplained,
    OpenTestHypothesis,
    PeLedgerInvalid,
    ResultExists,
    RunNotBegun,
    SeedMismatch,
    StrategyChanged,
    UndeclaredConfiguration,
)
from pfb.registry.genesis import PE_FILES, join_lines, pe_n_of
from pfb.registry.hypfile import (
    PURPOSE_TEST,
    TEST_BUDGET,
    first_overlap,
    overlap_text,
    seed_unexplained,
    seed_unexplained_text,
)
from pfb.registry.inputs import CHECK_KEYS, LEGACY_CHECK_KEYS, SHA_RE
from pfb.registry.powercheck import (
    BEGIN_POWER_KEYS,
    HYPOTHESIS_POWER_KEYS,
    begin_power,
    book_begin_power,
    book_hypothesis_power,
    hypothesis_power,
    power_at,
    require_each_window,
    require_power,
)

SCHEMA = "pfb.registry/1"
MIN_SOURCE_TRADES = 2  # an sd_source run's realized figures: pfb.analysis.inrun.realized_power computes
MIN_SOURCE_DAYS = 2  # its minimum edge only from two trades and two days on
GENESIS, SYNC, HYPOTHESIS, RESULT = "genesis", "sync", "hypothesis", "result"
TRIAL, ENGINE_CHECK, SYNTHETIC = "trial", "engine_check", "synthetic"
RUN_KINDS = frozenset({TRIAL, ENGINE_CHECK, SYNTHETIC})
STATUS_BEGUN, STATUS_OK, STATUS_FAILED = "begun", "ok", "failed"
EVENT_WINDOWS_OFF = "off"
EVENT_WINDOWS_ON = "on"  # a trial of a hypothesis whose cost.event_costs is "on"
EVENT_WINDOWS = (EVENT_WINDOWS_OFF, EVENT_WINDOWS_ON)
# The cost model id of event costs "on" ends in this suffix (pfb.costs.events.EVENT_MODEL_SUFFIX; not imported
# here so that replay stays free of the cost package; a test pins the two equal).
EVENTS_MODEL_SUFFIX = "+events"
CHECKS_PASSED = {"lint": "pass", "lookahead": "pass"}

_COMMON = frozenset({"schema", "kind", "time_utc", "prev_sha256", "n_before", "n_after"})
PURPOSE_STORED = {"test": "TEST", "research": "research"}
SIGNAL_SEEN, SIGNAL_NONE = "seen", "none"
LOOKAHEAD_SIGNALS = (SIGNAL_SEEN, SIGNAL_NONE)
_RUN_TIER2 = _COMMON | {
    "code_sha256", "run_id", "status", "hypothesis_id", "hypothesis_sha256", "strategy_path",
    "strategy_sha256",
    "git_commit", "git_dirty", "data_files", "instruments_sha256", "cost_model", "cost_model_id",
    "cost_config_sha256", "fees", "fees_sha256", "closed_windows_sha256", "engine_checks_sha256", "root",
    "first", "last", "session", "params", "seed", "engine_version", "configurations", "metrics", "artifacts",
    "declared_in", "check", "event_windows", "warnings", "static_checks",
}  # fmt: skip
_RESULT_TIER1 = _COMMON | {"run_id", "status", "error", "metrics", "artifacts", "warnings"}
BOOK_KEY = "book"  # hypothesis record tier 2: null for schema 2, the book object for schema 3
_HYPOTHESIS_TIER1 = _COMMON | frozenset(
    {
        "hypothesis_id",
        "hypothesis_sha256",
        "hypothesis_path",
        "purpose",
        "statement",
        "strategy_path",
        "strategy_sha256",
        "roots",
        "bar_minutes",
        "session",
        "windows",
        "grid",
        "budget",
        "no_signal_reason",
        "cost",
        "power",
        "suite",
        "secondary",
        "closed_windows_sha256",
    }
)  # fmt: skip
SEED_KEY, SEED_REASON_KEY = "lookahead_seed", "lookahead_seed_reason"  # hypothesis tier 3
ARTIFACT_SHA_KEY = "artifact_sha256"  # result tier 3: {artifact name: sha256}
_HYPOTHESIS_TIER2 = _HYPOTHESIS_TIER1 | {BOOK_KEY}
_RESULT_TIER2 = _RESULT_TIER1 | {"begin_sha256"}
KEYS: Mapping[str, frozenset[str]] = {
    GENESIS: _COMMON | {"n_pe", "pe_files"},
    SYNC: _COMMON | {"n_pe_before", "n_pe", "pe_files"},
    HYPOTHESIS: _HYPOTHESIS_TIER2 | {SEED_KEY, SEED_REASON_KEY},
    RESULT: _RESULT_TIER2 | {ARTIFACT_SHA_KEY},
    **dict.fromkeys(RUN_KINDS, _RUN_TIER2 | {"calendars_sha256", "lookahead_signal", "power"}),
}
CODE_KEY = "code_sha256"
CALENDARS_KEY = "calendars_sha256"
LEGACY_RUN_KEYS = _RUN_TIER2 - {CODE_KEY}
CODE_RUN_KEYS = _RUN_TIER2
LEGACY_RESULT_KEYS = _RESULT_TIER1
RUN_TIERS = ((LEGACY_RUN_KEYS, 1), (CODE_RUN_KEYS, 2), (KEYS[TRIAL], 3))
RESULT_TIERS = ((LEGACY_RESULT_KEYS, 1), (_RESULT_TIER2, 2), (KEYS[RESULT], 3))
HYPOTHESIS_TIERS = ((_HYPOTHESIS_TIER1, 1), (_HYPOTHESIS_TIER2, 2), (KEYS[HYPOTHESIS], 3))
COST_KEYS = frozenset(
    {"model", "fees", "model_id", "config_sha256", "fees_sha256", "event_costs", "event_stance"}
)
_BOUND_CHECK_FIELDS = ("session", "seed", "params")


def line_sha256(record: Mapping[str, Any]) -> str:
    """sha256 of the record's canonical line (what the chain hashes: the line without its newline)."""
    return sha256_hex(canonical_bytes(record))


def config_id(record: Mapping[str, Any]) -> str:
    """sha256 of the canonical configuration tuple of ADR 13 (data files enter by sha256, in read order),
    plus code_sha256 when the record carries it (every begin record since the code identity was added),
    then calendars_sha256 (tier 3)."""
    data_hashes = [f["sha256"] for f in record["data_files"]]
    parts = [
        record["hypothesis_sha256"],
        record["strategy_sha256"],
        record["params"],
        record["root"],
        [record["first"], record["last"]],
        record["session"],
        [record["cost_model_id"], record["cost_config_sha256"]],
        record["fees_sha256"],
        record["instruments_sha256"],
        record["engine_version"],
        data_hashes,
    ]
    if CODE_KEY in record:
        parts.append(record[CODE_KEY])
    if CALENDARS_KEY in record:
        parts.append(record[CALENDARS_KEY])
    return sha256_hex(canonical_bytes(parts))


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class LedgerState:
    """What the ledger says after its last record."""

    n: int
    n_pe: int
    records: int
    pe_sha256: Mapping[str, str]
    pe_lines: Mapping[str, tuple[str, ...]]
    hypotheses: Mapping[str, Mapping[str, Any]]
    configurations: Mapping[str, frozenset[str]]
    begun: Mapping[str, Mapping[str, Any]]
    finished: frozenset[str]
    results: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    begin_sha256: Mapping[str, str] = field(default_factory=dict)
    engine_check_shas: frozenset[str] = frozenset()

    @property
    def n_local(self) -> int:
        return self.n - self.n_pe

    @property
    def open_runs(self) -> tuple[str, ...]:
        """Runs with a begin record and no result: open or crashed (G7)."""
        return tuple(run_id for run_id in self.begun if run_id not in self.finished)

    def ok_daily_sharpes(self, hypothesis_id: str) -> dict[str, float]:
        """config_id -> sharpe_daily of the latest "ok" result of each trial configuration of the hypothesis
        (the DSR grid of metrics.core.summarize, A14 G8); a configuration without a numeric figure is left
        out."""
        out: dict[str, float] = {}
        for run_id, begin in self.begun.items():
            result = self.results.get(run_id)
            if begin["kind"] != TRIAL or begin["hypothesis_id"] != hypothesis_id or result is None:
                continue
            metrics = result["metrics"] if result["status"] == STATUS_OK else None
            value = metrics.get("sharpe_daily") if isinstance(metrics, dict) else None
            if isinstance(value, int | float) and not isinstance(value, bool):
                out[begin["configurations"][0]] = float(value)
        return out


@dataclass
class _Replay:
    n: int = 0
    n_pe: int = 0
    pe_sha256: dict[str, str] = field(default_factory=dict)
    pe_lines: dict[str, tuple[str, ...]] = field(default_factory=dict)
    hypotheses: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    configurations: dict[str, set[str]] = field(default_factory=dict)
    begun: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    finished: set[str] = field(default_factory=set)
    results: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    begin_sha: dict[str, str] = field(default_factory=dict)
    tiers: dict[str, int] = field(default_factory=dict)
    engine_check_shas: set[str] = field(default_factory=set)
    pinned_shas: set[str] = field(default_factory=set)
    book_counted: dict[str, dict[int, str]] = field(default_factory=dict)  # book id -> member -> config_id
    alpha_rule: int = LEGACY_ALPHA_RULE  # the newest alpha rule a hypothesis record has shown

    def _keys_ok(self, kind: str, record: Mapping[str, Any]) -> bool:
        """Exactly the kind's keys; for begin and result records one of the family's tiers, never below the
        highest tier the family has already used (an older tier is valid as a prefix only)."""
        keys = set(record)
        family = "run" if kind in RUN_KINDS else kind
        tiers = (
            RUN_TIERS
            if kind in RUN_KINDS
            else RESULT_TIERS
            if kind == RESULT
            else HYPOTHESIS_TIERS
            if kind == HYPOTHESIS
            else ()
        )
        if not tiers:
            return kind in KEYS and keys == KEYS[kind]
        for tier_keys, tier in tiers:
            if keys == tier_keys and tier >= self.tiers.get(family, 0):
                self.tiers[family] = tier
                return True
        return False

    def apply(self, record: Mapping[str, Any], index: int) -> None:
        where = f"record {index + 1}"
        kind = str(record.get("kind"))
        if record.get("schema") != SCHEMA or kind not in KEYS or not self._keys_ok(kind, record):
            raise ChainBroken(
                f"{where}: not a {SCHEMA} record of a known kind with exactly its fields (an older begin or "
                "result tier is accepted only before the first record of a newer tier)"
            )
        if (kind == GENESIS) != (index == 0):
            raise ChainBroken(f"{where}: genesis must be the first record and only the first")
        n_before, n_after = record["n_before"], record["n_after"]
        if not (_is_int(n_before) and _is_int(n_after)) or n_before != self.n:
            raise ChainBroken(f"{where}: n_before {n_before!r} does not equal the previous n_after {self.n}")
        added = _HANDLERS[kind](self, record, where)
        if n_after != n_before + added:
            raise ChainBroken(f"{where}: n_after {n_after} != n_before {n_before} + {added}")
        self.n = n_after

    def _pe_import(self, record: Mapping[str, Any], where: str, lines_key: str) -> int:
        pe_files = record["pe_files"]
        if pe_files == {} and lines_key == "lines" and not self.pe_lines:
            # a genesis without an upstream ledger: nothing imported, n_pe 0
            if record["n_pe"] != 0:
                raise ChainBroken(f"{where}: a genesis that imports no upstream ledger has n_pe 0")
            return 0
        if not isinstance(pe_files, dict) or set(pe_files) != set(PE_FILES):
            raise ChainBroken(f"{where}: pe_files must name exactly {list(PE_FILES)}")
        for name in PE_FILES:
            entry = pe_files[name]
            if not isinstance(entry, dict) or set(entry) != {"sha256", lines_key}:
                raise ChainBroken(f"{where}: pe_files[{name}] must hold sha256 and {lines_key}")
            lines = entry[lines_key]
            if not isinstance(lines, list) or not all(isinstance(x, str) for x in lines):
                raise ChainBroken(f"{where}: pe_files[{name}].{lines_key} must be a list of strings")
            full = self.pe_lines.get(name, ()) + tuple(lines)
            if sha256_hex(join_lines(full)) != entry["sha256"]:
                raise ChainBroken(f"{where}: the imported lines of {name} do not hash to the recorded sha256")
            self.pe_lines[name], self.pe_sha256[name] = full, entry["sha256"]
        try:
            n_pe = pe_n_of(self.pe_lines)
        except PeLedgerInvalid as exc:
            raise ChainBroken(f"{where}: imported upstream lines fail upstream's checks: {exc}") from exc
        if record["n_pe"] != n_pe:
            raise ChainBroken(
                f"{where}: n_pe {record['n_pe']!r} differs from upstream's N {n_pe} of the imported lines"
            )
        return n_pe

    def genesis(self, record: Mapping[str, Any], where: str) -> int:
        self.n_pe = self._pe_import(record, where, "lines")
        return self.n_pe

    def sync(self, record: Mapping[str, Any], where: str) -> int:
        if record["n_pe_before"] != self.n_pe:
            raise ChainBroken(f"{where}: n_pe_before differs from the last imported upstream N {self.n_pe}")
        before = self.n_pe
        self.n_pe = self._pe_import(record, where, "lines_added")
        if self.n_pe < before:
            raise ChainBroken(f"{where}: upstream's N fell from {before} to {self.n_pe}")
        return self.n_pe - before

    def hypothesis(self, record: Mapping[str, Any], where: str) -> int:
        hyp_id = record["hypothesis_id"]
        if hyp_id in self.hypotheses:
            raise HypothesisInvalid(f"{where}: hypothesis {hyp_id!r} is already registered; use a new id")
        _check_hypothesis_shape(record, where)
        _check_hypothesis_event_costs(record, where)
        if SEED_KEY in record:  # tier 3: the open-TEST and seed rules bind every record from here on
            _check_seed_fields(record, where)
            self._require_no_open_test(record, where)
        if is_book_record(record):
            self._book_hypothesis(record, where)
            return 0
        _check_hypothesis_rules(record, where)
        if record["strategy_sha256"] in self.engine_check_shas:
            raise EngineCheckStrategy(
                f"{where}: the strategy of {hyp_id!r} was run as an engine_check (rule C5); a hypothesis "
                "needs "
                "a strategy no engine_check has run"
            )
        self._check_sd_source(record, where)
        n_used = record["n_before"] + record["budget"]
        require_each_window(record["power"], record["windows"], n_used, where)
        self._require_power_figures(
            record,
            lambda rule: hypothesis_power(record["power"], record["windows"], n_used, rule),
            f"{where}: the power figures differ from the check recomputed at N {n_used}",
        )
        self.hypotheses[hyp_id] = record
        self.configurations[hyp_id] = set()
        self.pinned_shas.add(record["strategy_sha256"])
        return 0

    def spent(self, hypothesis_id: str) -> bool:
        """A hypothesis is finished when its budget is spent: as many configurations counted in N as its
        budget (the BudgetExceeded test). A crash after begin still counts, so an open run spends too."""
        hyp = self.hypotheses[hypothesis_id]
        return len(self.configurations.get(hypothesis_id, ())) >= hyp["budget"]

    def _require_no_open_test(self, record: Mapping[str, Any], where: str) -> None:
        """A purpose TEST record while another TEST hypothesis is not spent."""
        test = PURPOSE_STORED[PURPOSE_TEST]
        if record["purpose"] != test:
            return
        open_ids = sorted(
            hyp_id
            for hyp_id, hyp in self.hypotheses.items()
            if hyp["purpose"] == test and not self.spent(hyp_id)
        )
        if open_ids:
            raise OpenTestHypothesis(
                f"{where}: TEST hypothesis {open_ids[0]!r} is registered and its budget is not spent; at "
                "most one TEST hypothesis may be open at a time (run its trial first)"
            )

    def _require_power_figures(
        self, record: Mapping[str, Any], recompute: Callable[[int], Mapping[str, Any]], refusal: str
    ) -> None:
        """The written power object equals the check recomputed under the current alpha rule, or
        under the legacy one (and D: only min_edge_cents_alpha differs) while no earlier record needs
        the current rule: the legacy rule is valid as a prefix only. The gate figure is one under both."""
        written = record["power"]
        is_legacy = written == recompute(LEGACY_ALPHA_RULE)
        if written == recompute(ALPHA_RULE):
            if not is_legacy:
                self.alpha_rule = ALPHA_RULE
            return
        if not (is_legacy and self.alpha_rule == LEGACY_ALPHA_RULE):
            raise ChainBroken(refusal)

    def _book_hypothesis(self, record: Mapping[str, Any], where: str) -> None:
        """A book's hypothesis record: clauses 1, 5 and 6 (bookfile.check_book_record), rule
        C5 for every member, and the BOOK's power figures at N + K over the union of the member windows."""
        hyp_id = record["hypothesis_id"]
        check_book_record(record, where)
        members = record["book"]["members"]
        if any(m["strategy_sha256"] in self.engine_check_shas for m in members):
            raise EngineCheckStrategy(
                f"{where}: a member strategy of book {hyp_id!r} was run as an engine_check (rule C5)"
            )
        n_used = record["n_before"] + record["budget"]
        self._require_power_figures(
            record,
            lambda rule: book_hypothesis_power(record["power"], member_windows(record), n_used, where, rule),
            f"{where}: the book's power figures differ from the check recomputed at N {n_used}",
        )
        self.hypotheses[hyp_id] = record
        self.configurations[hyp_id] = set()
        self.book_counted[hyp_id] = {}
        self.pinned_shas.update(m["strategy_sha256"] for m in members)

    def _check_sd_source(self, record: Mapping[str, Any], where: str) -> None:
        """TRIALS.md section 2: a "run:<run_id>" source is an ok synthetic or engine_check
        run with realized power figures that bind the stated ones; a synthetic one runs this strategy."""
        power = record["power"]
        source = power["sd_source"]
        if not source.startswith("run:"):
            return
        run_id = source[len("run:") :]
        begin = self.begun.get(run_id)
        result = self.results.get(run_id)
        if (
            begin is None
            or begin["kind"] not in (SYNTHETIC, ENGINE_CHECK)
            or result is None
            or result["status"] != STATUS_OK
        ):
            raise HypothesisInvalid(
                f"{where}: sd_source names {run_id!r}, which is not a finished ok synthetic or "
                "engine_check run"
            )
        realized = realized_power_figures(result["metrics"])
        if realized is None:
            raise HypothesisInvalid(
                f"{where}: sd_source run {run_id!r} has no realized power figures (metrics.suite.power with "
                f"trades >= {MIN_SOURCE_TRADES}, days >= {MIN_SOURCE_DAYS} and trade_sd_cents >= 1)"
            )
        sd, trades, days = realized
        if sd > int(power["trade_sd_cents"]):
            raise HypothesisInvalid(
                f"{where}: sd_source run {run_id!r} has a realized per-trade sd of {sd} cents, above the "
                f"stated trade_sd_cents {power['trade_sd_cents']}"
            )
        if Fraction(trades * TRADING_DAYS_PER_YEAR, days) < Fraction(float(power["trades_per_year"])):
            raise HypothesisInvalid(
                f"{where}: sd_source run {run_id!r} realized {trades} trades in {days} days, below the "
                f"stated trades_per_year {power['trades_per_year']}"
            )
        if begin["kind"] == SYNTHETIC and begin["strategy_sha256"] != record["strategy_sha256"]:
            raise HypothesisInvalid(
                f"{where}: sd_source run {run_id!r} is a synthetic run of another strategy (its strategy "
                "sha256 is not the hypothesis's)"
            )

    def run(self, record: Mapping[str, Any], where: str) -> int:
        run_id = record["run_id"]
        if not isinstance(run_id, str) or not run_id or run_id in self.begun:
            raise ChainBroken(f"{where}: run_id {run_id!r} is empty or repeats")
        _check_run_common(record, where)
        if CODE_KEY in record and (
            not isinstance(record[CODE_KEY], str) or not SHA_RE.fullmatch(record[CODE_KEY])
        ):
            raise ChainBroken(f"{where}: {CODE_KEY} is not a sha256 (64 lowercase hex)")
        kind = record["kind"]
        if CALENDARS_KEY in record:
            _check_tier3(record, where)
        elif kind == TRIAL:
            raise ChainBroken(
                f"{where}: a trial begin record carries {CALENDARS_KEY}, lookahead_signal, power"
            )
        added = _check_trial(self, record, where) if kind == TRIAL else 0
        if kind == ENGINE_CHECK:
            _check_engine_check(record, where)
            if record["strategy_sha256"] in self.pinned_shas:
                raise EngineCheckStrategy(
                    f"{where}: the strategy is pinned by a registered hypothesis; it cannot run as an "
                    "engine_check (rule C5)"
                )
            self.engine_check_shas.add(record["strategy_sha256"])
        elif kind == SYNTHETIC and record["data_files"]:
            raise ChainBroken(f"{where}: a synthetic run reads no data file")
        self.begun[run_id] = record
        self.begin_sha[run_id] = line_sha256(record)
        return added

    def result(self, record: Mapping[str, Any], where: str) -> int:
        run_id = record["run_id"]
        if run_id not in self.begun:
            raise RunNotBegun(f"{where}: result for run {run_id!r}, which has no begin record")
        if run_id in self.finished:
            raise ResultExists(f"{where}: run {run_id!r} already has its result record")
        status, error = record["status"], record["error"]
        if status not in (STATUS_OK, STATUS_FAILED) or (status == STATUS_FAILED) != isinstance(error, str):
            raise ChainBroken(f"{where}: status must be ok (error null) or failed (error the class name)")
        if not isinstance(record["artifacts"], list):
            raise ChainBroken(f"{where}: artifacts must be a list of planned paths")
        if "begin_sha256" in record and record["begin_sha256"] != self.begin_sha[run_id]:
            raise FinishNotPermitted(
                f"{where}: begin_sha256 is not the sha256 of run {run_id!r}'s begin line"
            )
        if ARTIFACT_SHA_KEY in record:
            _check_artifact_shas(record, where)
        self.finished.add(run_id)
        self.results[run_id] = record
        return 0


def realized_power_figures(metrics: object) -> tuple[int, int, int] | None:
    """(trade_sd_cents, trades, days) of a result's metrics.suite.power (pfb.analysis.inrun.realized_power);
    None unless all three are ints with at least two trades, two days and a positive sd."""
    suite = metrics.get("suite") if isinstance(metrics, dict) else None
    power = suite.get("power") if isinstance(suite, dict) else None
    if not isinstance(power, dict):
        return None
    sd, trades, days = power.get("trade_sd_cents"), power.get("trades"), power.get("days")
    if not (isinstance(sd, int) and isinstance(trades, int) and isinstance(days, int)):
        return None
    if any(isinstance(v, bool) for v in (sd, trades, days)):
        return None
    if sd < 1 or trades < MIN_SOURCE_TRADES or days < MIN_SOURCE_DAYS:
        return None
    return sd, trades, days


def _check_hypothesis_shape(record: Mapping[str, Any], where: str) -> None:
    grid, budget, roots, windows = record["grid"], record["budget"], record["roots"], record["windows"]
    grid_ok = (  # a book declares no grid (null); its members carry their params
        grid is None
        if is_book_record(record)
        else isinstance(grid, dict) and bool(grid) and all(isinstance(v, list) and v for v in grid.values())
    )
    shaped = (
        isinstance(record["hypothesis_id"], str)
        and grid_ok
        and _is_int(budget)
        and budget >= 1
        and isinstance(roots, list)
        and bool(roots)
        and all(isinstance(r, str) for r in roots)
        and isinstance(windows, list)
        and bool(windows)
        and all(isinstance(w, list) and len(w) == 2 and all(isinstance(d, str) for d in w) for w in windows)
        and isinstance(record["cost"], dict)
        and set(record["cost"]) == COST_KEYS
        and isinstance(record["power"], dict)
        and set(record["power"]) == HYPOTHESIS_POWER_KEYS
        and record["purpose"] in PURPOSE_STORED.values()
        and isinstance(record["suite"], dict)
        and isinstance(record["secondary"], dict)
        and isinstance(record["no_signal_reason"], str)
    )
    if not shaped:
        raise ChainBroken(
            f"{where}: a hypothesis record needs an id, roots, windows, a grid of non-empty lists, a budget "
            "and the cost, power, suite and secondary objects"
        )


def _check_hypothesis_rules(record: Mapping[str, Any], where: str) -> None:
    """The record-only parser rules, so a record an older checkout wrote is refused too:
    purpose TEST has budget 1, and no two windows share a day."""
    if record["purpose"] == PURPOSE_STORED[PURPOSE_TEST] and record["budget"] != TEST_BUDGET:
        raise ChainBroken(
            f"{where}: a TEST hypothesis record has budget {TEST_BUDGET}, not {record['budget']}"
        )
    pair = first_overlap([(w[0], w[1]) for w in record["windows"]])
    if pair is not None:
        raise ChainBroken(f"{where}: {overlap_text(pair)}")


def _check_seed_fields(record: Mapping[str, Any], where: str) -> None:
    """Hypothesis tier 3: lookahead_seed an int >= 0, lookahead_seed_reason a
    string, non-empty when the seed is not the program seed."""
    seed, reason = record[SEED_KEY], record[SEED_REASON_KEY]
    if not (_is_int(seed) and seed >= 0 and isinstance(reason, str)):
        raise ChainBroken(f"{where}: {SEED_KEY} is an int >= 0 and {SEED_REASON_KEY} a string")
    if seed_unexplained(seed, reason):
        raise LookaheadSeedUnexplained(f"{where}: {seed_unexplained_text(seed)}")


def _check_artifact_shas(record: Mapping[str, Any], where: str) -> None:
    """Result tier 3: artifact_sha256 names exactly the record's artifacts, each with
    a sha256 (64 lowercase hex); {} when there are none."""
    shas, names = record[ARTIFACT_SHA_KEY], record["artifacts"]
    if (
        not isinstance(shas, dict)
        or set(shas) != set(names)
        or len(names) != len(set(names))
        or not all(isinstance(v, str) and SHA_RE.fullmatch(v) for v in shas.values())
    ):
        raise ChainBroken(
            f"{where}: {ARTIFACT_SHA_KEY} maps each of the record's artifacts, exactly, to its sha256"
        )


def _check_tier3(record: Mapping[str, Any], where: str) -> None:
    calendars = record[CALENDARS_KEY]
    if not isinstance(calendars, str) or not SHA_RE.fullmatch(calendars):
        raise ChainBroken(f"{where}: {CALENDARS_KEY} is not a sha256 (64 lowercase hex)")
    if record["lookahead_signal"] not in LOOKAHEAD_SIGNALS:
        raise ChainBroken(f"{where}: lookahead_signal must be one of {list(LOOKAHEAD_SIGNALS)}")
    power = record["power"]
    if record["kind"] == TRIAL:
        if not isinstance(power, dict) or set(power) != BEGIN_POWER_KEYS:
            raise ChainBroken(f"{where}: a trial's power holds exactly {sorted(BEGIN_POWER_KEYS)}")
    elif power is not None:
        raise ChainBroken(f"{where}: power is null for a {record['kind']} run")


def events_id_matches(event_costs: object, model_id: object) -> bool:
    """event costs "on" exactly when the cost model id ends in "+events"."""
    return isinstance(model_id, str) and model_id.endswith(EVENTS_MODEL_SUFFIX) == (
        event_costs == EVENT_WINDOWS_ON
    )


def _check_event_windows(record: Mapping[str, Any], where: str) -> None:
    """A begin's event_windows is "off" or "on"; "on" only for a trial (no switch reaches another kind) and
    exactly when its cost_model_id ends in "+events". Whether it equals the hypothesis's cost.event_costs is
    _check_cost_bound's. Every begin written says "off" under a plain id."""
    windows = record["event_windows"]
    if windows not in EVENT_WINDOWS:
        raise ChainBroken(f"{where}: event_windows must be one of {list(EVENT_WINDOWS)}")
    if windows == EVENT_WINDOWS_ON and record["kind"] != TRIAL:
        raise ChainBroken(f'{where}: event_windows "on" is a trial\'s only (its hypothesis declares it)')
    if not events_id_matches(windows, record["cost_model_id"]):
        raise ChainBroken(
            f'{where}: event_windows {windows!r} with cost_model_id {record["cost_model_id"]!r}: "on" needs '
            f'an id ending in "{EVENTS_MODEL_SUFFIX}" and "off" one that does not'
        )


def _check_hypothesis_event_costs(record: Mapping[str, Any], where: str) -> None:
    """A hypothesis (or book) record's cost: event_costs "off" or "on", "on" exactly when model_id ends in
    "+events" (registration identifies the model with its event costs)."""
    events, model_id = record["cost"]["event_costs"], record["cost"]["model_id"]
    if events not in EVENT_WINDOWS or not events_id_matches(events, model_id):
        raise ChainBroken(
            f"{where}: cost.event_costs {events!r} with model_id {model_id!r}: "
            f'"on" needs an id ending in "{EVENTS_MODEL_SUFFIX}" and "off" one that does not'
        )


def _check_run_common(record: Mapping[str, Any], where: str) -> None:
    if record["status"] != STATUS_BEGUN or record["metrics"] is not None or record["artifacts"] != []:
        raise ChainBroken(f"{where}: a begin record has status begun, metrics null and no artifacts")
    if record["static_checks"] != CHECKS_PASSED:
        raise ChecksNotPassed(f"{where}: lint and look-ahead must both pass before a run is begun")
    _check_event_windows(record, where)
    files = record["data_files"]
    if not isinstance(files, list) or not all(
        isinstance(f, dict) and set(f) == {"path", "sha256"} for f in files
    ):
        raise ChainBroken(f"{where}: data_files must be a list of {{path, sha256}}")
    if record["configurations"] != [config_id(record)]:
        raise ChainBroken(f"{where}: configurations does not hold this run's config_id")


def _check_engine_check(record: Mapping[str, Any], where: str) -> None:
    check = record["check"]
    fields = ("strategy", "strategy_sha256", "root", "first", "last", "fees", "cost_model")
    mine = (
        record["strategy_path"],
        record["strategy_sha256"],
        record["root"],
        record["first"],
        record["last"],
        record["fees"],
        record["cost_model"],
    )
    expected_keys = CHECK_KEYS if CODE_KEY in record else LEGACY_CHECK_KEYS
    if (
        record["engine_checks_sha256"] is None
        or not isinstance(check, dict)
        or set(check) != expected_keys
        or tuple(check.get(k) for k in fields) != mine
        or record["declared_in"] != check.get("declared_in")
        or any(
            canonical_bytes(check[k]) != canonical_bytes(record[k]) for k in _BOUND_CHECK_FIELDS if k in check
        )
    ):
        raise EngineCheckNotDeclared(
            f"{where}: an engine_check must equal its allowlist entry in every field"
        )
    if not record["data_files"]:
        raise ChainBroken(f"{where}: an engine_check runs on real data and names its data files")


def in_grid(params: object, grid: Mapping[str, Sequence[object]]) -> bool:
    if not isinstance(params, dict) or set(params) != set(grid):
        return False
    return all(canonical_bytes(params[k]) in {canonical_bytes(v) for v in grid[k]} for k in grid)


def _check_bound_to_hypothesis(record: Mapping[str, Any], hyp: Mapping[str, Any], where: str) -> None:
    """Root, window, session, cost model, fees and their hashes, params: as the hypothesis record declares."""
    hyp_id = hyp["hypothesis_id"]
    if record["root"] not in hyp["roots"]:
        raise UndeclaredConfiguration(f"{where}: root {record['root']!r} is not one of {hyp_id!r}'s roots")
    if [record["first"], record["last"]] not in hyp["windows"]:
        raise UndeclaredConfiguration(f"{where}: window {record['first']}..{record['last']} is not declared")
    if record["session"] != hyp["session"]:
        raise UndeclaredConfiguration(f"{where}: session differs from hypothesis {hyp_id!r}")
    _check_cost_bound(record, hyp, where)
    if not in_grid(record["params"], hyp["grid"]):
        raise UndeclaredConfiguration(f"{where}: params outside the declared grid of {hyp_id!r}")


def _check_cost_bound(record: Mapping[str, Any], hyp: Mapping[str, Any], where: str) -> None:
    cost = hyp["cost"]
    mine = tuple(
        record[k] for k in ("cost_model", "cost_model_id", "cost_config_sha256", "fees", "fees_sha256")
    )
    declared = tuple(cost[k] for k in ("model", "model_id", "config_sha256", "fees", "fees_sha256"))
    if mine != declared:
        raise UndeclaredConfiguration(
            f"{where}: cost model, fees or their hashes differ from {hyp['hypothesis_id']!r}"
        )
    if record["event_windows"] != cost["event_costs"]:
        raise UndeclaredConfiguration(
            f"{where}: event_windows {record['event_windows']!r} differs from the event_costs "
            f"{cost['event_costs']!r} of {hyp['hypothesis_id']!r}"
        )


def _check_bound_to_book(record: Mapping[str, Any], hyp: Mapping[str, Any], where: str) -> int:
    """Clause 2: the begin is exactly one declared member (strategy, root, params, window, session) at the
    book's cost model and fees; returns that member's index."""
    index = member_index(record, hyp, where)
    _check_cost_bound(record, hyp, where)
    return index


def _trial_power(
    record: Mapping[str, Any], hyp: Mapping[str, Any], n_used: int, where: str
) -> dict[str, Any]:
    """The begin's power object: a book's at the union of its member windows, else the trial's
    own window."""
    if is_book_record(hyp):
        return book_begin_power(hyp["power"], member_windows(hyp), n_used, where)
    window = [[record["first"], record["last"]]]
    require_power(power_at(hyp["power"], window, n_used), where)
    return begin_power(hyp["power"], window, n_used)


def _check_trial(state: _Replay, record: Mapping[str, Any], where: str) -> int:
    hyp_id = record["hypothesis_id"]
    hyp = state.hypotheses.get(hyp_id)
    if hyp is None:
        raise HypothesisNotRegistered(f"{where}: hypothesis {hyp_id!r} is not registered")
    if record["hypothesis_sha256"] != hyp["hypothesis_sha256"]:
        raise HypothesisChanged(f"{where}: the hypothesis file differs from the registered one")
    if SEED_KEY in hyp and record["seed"] != hyp[SEED_KEY]:  # a tier-3 hypothesis record stores its seed
        raise SeedMismatch(
            f"{where}: a trial runs at its hypothesis's {SEED_KEY} {hyp[SEED_KEY]}, not {record['seed']!r}"
        )
    book = is_book_record(hyp)
    if not book and (record["strategy_path"], record["strategy_sha256"]) != (
        hyp["strategy_path"],
        hyp["strategy_sha256"],
    ):
        raise StrategyChanged(f"{where}: the strategy differs from the one hypothesis {hyp_id!r} pins")
    if record["git_dirty"] is not False:
        raise DirtyTree(f"{where}: a trial needs a clean git tree")
    member = _check_bound_to_book(record, hyp, where) if book else None
    if member is None:
        _check_bound_to_hypothesis(record, hyp, where)
    else:
        require_member_uncounted(
            state.book_counted[hyp_id], member, record["configurations"][0], hyp_id, where
        )
    if not record["data_files"]:
        raise ChainBroken(f"{where}: a trial runs on real data and names its data files")
    if record["lookahead_signal"] == SIGNAL_NONE and not hyp["no_signal_reason"].strip():
        raise NoSignalUnexplained(
            f"{where}: the step-1 look-ahead check saw no signal and {hyp_id!r} gives no no_signal_reason"
        )
    seen = state.configurations[hyp_id]
    new = [c for c in record["configurations"] if c not in seen]
    if len(seen) + len(new) > hyp["budget"]:
        raise BudgetExceeded(
            f"{where}: {len(seen) + len(new)} distinct configurations exceed the budget "
            f"{hyp['budget']} of {hyp_id!r}"
        )
    n_used = record["n_before"] + hyp["budget"] - len(seen)
    if record["power"] != _trial_power(record, hyp, n_used, where):
        raise ChainBroken(
            f"{where}: the trial's power figures differ from the check recomputed at N {n_used}"
        )
    seen.update(new)
    if member is not None:
        state.book_counted[hyp_id][member] = record["configurations"][0]
    return len(new)


_HANDLERS = {
    GENESIS: _Replay.genesis,
    SYNC: _Replay.sync,
    HYPOTHESIS: _Replay.hypothesis,
    RESULT: _Replay.result,
    **dict.fromkeys(RUN_KINDS, _Replay.run),
}


def replay(records: Sequence[Mapping[str, Any]]) -> LedgerState:
    """Verify every record's kind, fields and N link in order; return the state after the last one."""
    if not records:
        raise ChainBroken("the ledger is empty: genesis is missing")
    state = _Replay()
    for index, record in enumerate(records):
        state.apply(record, index)
    return LedgerState(
        n=state.n,
        n_pe=state.n_pe,
        records=len(records),
        pe_sha256=dict(state.pe_sha256),
        pe_lines=dict(state.pe_lines),
        hypotheses=dict(state.hypotheses),
        configurations={k: frozenset(v) for k, v in state.configurations.items()},
        begun=dict(state.begun),
        finished=frozenset(state.finished),
        results=dict(state.results),
        begin_sha256=dict(state.begin_sha),
        engine_check_shas=frozenset(state.engine_check_shas),
    )
