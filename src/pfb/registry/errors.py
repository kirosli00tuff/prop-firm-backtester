"""Registry errors. Every refusal is a RegistryError subclass; nothing is defaulted or filtered silently."""

from __future__ import annotations


class RegistryError(RuntimeError):
    """Base class: the registry refused an append or found the ledger invalid."""


class RegistryMissing(RegistryError):
    """ledger/registry.jsonl does not exist (genesis has not been written)."""


class GenesisExists(RegistryError):
    """Genesis is written once, into an empty ledger only."""


class ChainBroken(RegistryError):
    """A line does not parse, is not canonical, breaks the hash chain or an n_before/n_after link."""


class LedgerAnchorBroken(RegistryError):
    """HEAD tracks the ledger and its committed bytes are not a byte prefix of the working file: lines were
    cut off or the file was re-created. Lines appended after the last commit are not anchored."""


class PeLedgerInvalid(RegistryError):
    """An upstream ledger file fails its own checks or cannot be imported byte for byte."""


class SyncRequired(RegistryError):
    """Upstream's ledger changed since the last genesis or sync: run `pfb registry sync` first."""


class SyncPrefixChanged(RegistryError):
    """Upstream's ledger no longer starts with the lines already imported: refused loudly."""


class HypothesisInvalid(RegistryError):
    """A hypothesis file is malformed, or its id is already registered."""


class HypothesisNotRegistered(RegistryError):
    """A trial names a hypothesis that has no hypothesis record."""


class HypothesisChanged(RegistryError):
    """The hypothesis file differs from the registered one."""


class StrategyChanged(RegistryError):
    """The strategy file differs from the one the hypothesis or allowlist entry pins."""


class UndeclaredConfiguration(RegistryError):
    """A parameter, root or window lies outside the hypothesis's declared grid."""


class BudgetExceeded(RegistryError):
    """The run would take the hypothesis past its budget of distinct configurations."""


class DirtyTree(RegistryError):
    """A counted run (trial) on a git tree with changes other than ledger/registry.jsonl."""


class InputNotCommitted(RegistryError):
    """An input file is untracked, changed against HEAD, or (declared_in) absent at HEAD (ADR A14 a)."""


class EngineCheckNotDeclared(RegistryError):
    """An engine_check that is not on config/engine_checks.toml or differs from its entry (ADR A2)."""


class ChecksNotPassed(RegistryError):
    """The strategy's lint or look-ahead check did not pass before begin."""


class RunNotBegun(RegistryError):
    """A result names a run_id that has no begin record."""


class ResultExists(RegistryError):
    """A run already has its result record."""


class BadRequest(RegistryError):
    """A request or record field is malformed."""


class GitError(RegistryError):
    """A git command needed by the registry failed."""


class SpecMismatch(RegistryError):
    """A hypothesis's roots, bar_minutes, session or grid names differ from its strategy's SPEC (TRIALS 2)."""


class ClosedWindowTouched(RegistryError):
    """A declared window touches a closed window for one of the hypothesis's roots (TRIALS 2)."""


class NoDataFile(RegistryError):
    """The catalog names no data file for a declared root and window (TRIALS 2, 4)."""


class ZeroSlippage(RegistryError):
    """A hypothesis or trial cost model without slippage (fixed:0 or a stress of it by 0 ticks)."""


class PowerInsufficient(RegistryError):
    """The power check fails: the target edge is below the minimum detectable edge, or none is (TRIALS 5)."""


class EngineCheckStrategy(RegistryError):
    """Rule C5: a strategy is both an engine_check's and a registered hypothesis's."""


class NoSignalUnexplained(RegistryError):
    """A trial whose step-1 look-ahead check saw no signal, under a hypothesis with no no_signal_reason."""


class LedgerNotTracked(RegistryError):
    """A trial needs the ledger tracked at git HEAD inside the repository (TRIALS 4)."""


class FinishNotPermitted(RegistryError):
    """finish was given a receipt whose record_sha256 is not the begin line of its run (TRIALS 4)."""


class SuiteMismatch(RegistryError):
    """A trial's suite, secondary outputs, fees or cost model differ from its hypothesis (TRIALS C4)."""


class BookInvalid(HypothesisInvalid):
    """A schema-3 book file breaks a shape rule or clause 1 or 6 of the counting rule."""


class PowerFlattered(HypothesisInvalid):
    """A book's [power] states more trades per year or a smaller sd than the members derive."""


class NotABookMember(UndeclaredConfiguration):
    """A begin under a book whose (strategy, root, params, window) is not exactly one declared member."""


class MemberAlreadyCounted(UndeclaredConfiguration):
    """A begin of a book member already counted under another config_id (a member is counted once)."""


class BookIncomplete(RegistryError):
    """The pooled statistic of a book is refused until every member has an ok result."""


class SeedMismatch(SuiteMismatch):
    """A trial at a seed other than its hypothesis's lookahead_seed: the runner refuses it before
    step 1, registry begin refuses it again (one class for both layers)."""


class OpenTestHypothesis(HypothesisInvalid):
    """A purpose "test" hypothesis while another one is registered and its budget not yet spent: at most one
    open TEST hypothesis at a time."""


class LookaheadSeedUnexplained(HypothesisInvalid):
    """A hypothesis whose lookahead_seed is not the program seed and whose lookahead_seed_reason is empty
    """


class ArtifactMismatch(RegistryError):
    """A run folder's file is missing or does not hash to its result record's artifact_sha256: the folder is
    refused as an input."""
