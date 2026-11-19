"""Probability of backtest overfitting (PBO) by combinatorially symmetric cross-validation (CSCV).

Definition (Bailey, Borwein, Lopez de Prado and Zhu; worked from the definition, no text fetched):
take the T x K matrix of per-period net results of K configurations, cut its rows into S equal contiguous
blocks, and for every choice of S/2 blocks as the in-sample set (the other S/2 out of sample, C(S, S/2)
splits) pick the configuration with the best in-sample performance, take its relative rank w = rank / (K + 1)
among the K out-of-sample performances (rank 1 = worst, K = best) and its logit ln(w / (1 - w)). PBO is the
share of splits whose logit is <= 0: the in-sample best ranks at or below the out-of-sample median.

THE CALLER OWES THE WHOLE DECLARED GRID. A PBO over a subset of the configurations a hypothesis declared
understates overfitting (the configurations left out are exactly the ones a search would have rejected), so
the array functions take a matrix whose K columns are every configuration of the grid and refuse K < 2;
the registry entry `for_hypothesis` builds that matrix from ALL the hypothesis's declared grid points, each
with an ok trial result, or refuses (GridIncomplete). Nothing here reads a bar, appends to the ledger or adds
to N; the result is an output about a hypothesis, never an input to choosing, adding or ordering
configurations.

Conventions (each a keyword argument with the default named here, and echoed in the result's "conventions"):
- metric="sharpe": per-period (daily) Sharpe of a column over the rows of a set of blocks, mean / population
  sd as pfb.metrics.stats.moments (sd 0 gives 0), not annualized (a monotone rescaling, ranks unchanged);
  "mean" is the mean net per period. The rows of the in-sample blocks are pooled (one Sharpe over them all).
- n_blocks=16 (S; even, >= 2): the value Bailey et al. use in their examples; C(16, 8) = 12,870 splits.
- min_block_rows=2: refuse when T // S is below it (a Sharpe needs two rows per block at least).
- remainder="drop_first": when S does not divide T, the T mod S OLDEST rows are left out (the blocks stay
  equal, as the symmetry of CSCV needs, and the most recent days are kept); "drop_last" leaves out the newest;
  "refuse" refuses. The result reports rows_used and rows_dropped.
- is_tie="worst": several configurations tie for the best in-sample value; the conservative reading picks the
  one of them with the WORST out-of-sample value (more overfitting reported). "first" picks the lowest column.
- oos_tie="min": the chosen configuration ties others out of sample; its rank is 1 + the number strictly below
  it (the lowest rank of the tie, more overfitting reported). "average" gives the mean rank of the tie.
- A logit of exactly 0 (rank (K + 1) / 2, possible for odd K) counts as overfit (logit <= 0, as defined).

Also reported: prob_oos_loss (share of splits whose chosen configuration has out-of-sample performance
< 0) and the performance degradation line (least squares of the chosen one's out-of-sample performance on
its in-sample performance).

FIXED FOR A HYPOTHESIS. The array function pbo_cscv keeps its keyword
conventions; the registry entry for_hypothesis takes NONE: it applies the defaults above (Conventions(), the
module's constants) and echoes them in "conventions", so a reader cannot pick the kindest of many PBO
figures after the runs. A hypothesis key that declares other conventions before the runs is the only way to
change them (not built). grid_matrix checks every run's daily file against the run's result record in the
ledger and refuses a mismatch by name (GridFileMismatch): runs/ is not pinned by the ledger. A record that
carries artifact_sha256 (result record tier 3) pins the file by its sha256, compared
through pfb.registry.artifacts (file_in_folder, folder_problems): the record must name the daily file and the
file must hash to it. A record written before that key (tier 1 or 2) is checked by the sum of net_cents
against metrics.net_cents and the row count against metrics.n_days (check_against_record), and its run is
listed in sum_checked_run_ids. cpcv.py and family_tests.py read run files only through grid_matrix.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pfb.analysis.walkforward import _check_runs  # pyright: ignore[reportPrivateUsage]
from pfb.registry.api import RegistryConfig, read_state
from pfb.registry.artifacts import file_in_folder, folder_problems, recorded_shas
from pfb.registry.bookfile import is_book_record
from pfb.registry.errors import HypothesisNotRegistered
from pfb.registry.records import STATUS_OK, TRIAL, LedgerState

METRICS = ("sharpe", "mean")
IS_TIES = ("worst", "first")
OOS_TIES = ("min", "average")
REMAINDERS = ("drop_first", "drop_last", "refuse")
DEFAULT_BLOCKS = 16
DEFAULT_MIN_BLOCK_ROWS = 2
MIN_CONFIGURATIONS = 2
SPLIT_CHUNK = 512  # splits evaluated per numpy batch (memory: chunk x S x K floats)
DAILY_FILE = "daily.parquet"
DAILY_COLUMNS = ("trade_date", "net_cents")
# the result record's metrics (the run's summary, pfb.metrics.core) a daily file is checked against
RECORD_NET_KEY = "net_cents"
RECORD_DAYS_KEY = "n_days"


class PboError(ValueError):
    """A matrix or a convention the PBO and CPCV functions refuse."""


class GridIncomplete(PboError):
    """The hypothesis's declared grid does not have an ok trial result for every configuration."""


class GridFileMismatch(PboError):
    """A run's daily file does not match its result record (its artifact_sha256, or for a record before that
    key the sum of net_cents and the row count), or the record lacks what to check it against (a sha256 map
    that does not name the daily file, or no integer figures): the file is not the one the trial wrote."""


# --------------------------------------------------------------------------- conventions and checks


@dataclass(frozen=True)
class Conventions:
    metric: str = "sharpe"
    n_blocks: int = DEFAULT_BLOCKS
    min_block_rows: int = DEFAULT_MIN_BLOCK_ROWS
    remainder: str = "drop_first"
    is_tie: str = "worst"
    oos_tie: str = "min"

    def __post_init__(self) -> None:
        _choice("metric", self.metric, METRICS)
        _choice("is_tie", self.is_tie, IS_TIES)
        _choice("oos_tie", self.oos_tie, OOS_TIES)
        _choice("remainder", self.remainder, REMAINDERS)
        if not _is_int(self.n_blocks) or self.n_blocks < 2 or self.n_blocks % 2:
            raise PboError(f"n_blocks {self.n_blocks!r} must be an even int >= 2")
        if not _is_int(self.min_block_rows) or self.min_block_rows < 1:
            raise PboError(f"min_block_rows {self.min_block_rows!r} must be an int >= 1")

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "n_blocks": self.n_blocks,
            "min_block_rows": self.min_block_rows,
            "remainder": self.remainder,
            "is_tie": self.is_tie,
            "oos_tie": self.oos_tie,
            "overfit_when": "logit <= 0",
        }

    def kwargs(self) -> dict[str, Any]:
        """pbo_cscv's keyword arguments."""
        return {
            "metric": self.metric,
            "n_blocks": self.n_blocks,
            "min_block_rows": self.min_block_rows,
            "remainder": self.remainder,
            "is_tie": self.is_tie,
            "oos_tie": self.oos_tie,
        }


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _choice(name: str, value: object, allowed: Sequence[str]) -> None:
    if value not in allowed:
        raise PboError(f"{name} {value!r} must be one of {tuple(allowed)}")


# the only conventions for_hypothesis applies: the defaults, until a hypothesis key
# declares others before its runs
HYPOTHESIS_CONVENTIONS = Conventions()


def check_matrix(matrix: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """The T x K matrix as float64; refuses a non-2-D array, no rows, K < 2 or a value that is not finite.

    K must be the WHOLE declared grid of one hypothesis (module docstring); this function cannot see that.
    """
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise PboError(f"the matrix must be 2-D (T rows x K configurations), got {arr.ndim}-D")
    if arr.shape[1] < MIN_CONFIGURATIONS:
        raise PboError(f"PBO needs at least 2 configurations (the whole declared grid), K = {arr.shape[1]}")
    if arr.shape[0] == 0:
        raise PboError("the matrix has no rows")
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.number):
        raise PboError(f"the matrix must be numeric, got {arr.dtype}")
    out = arr.astype(np.float64)
    if not np.isfinite(out).all():
        raise PboError("every value must be finite (no NaN or inf)")
    return out


def block_layout(t: int, n_blocks: int, remainder: str, min_block_rows: int) -> tuple[int, int]:
    """(first row used, rows per block) for T rows in n_blocks equal blocks."""
    rows = t // n_blocks
    if rows < min_block_rows:
        raise PboError(f"{t} rows give {rows} rows per block for {n_blocks} blocks; {min_block_rows} needed")
    extra = t - rows * n_blocks
    if extra and remainder == "refuse":
        raise PboError(f"T = {t} is not a multiple of {n_blocks} blocks (remainder 'refuse')")
    return (extra if remainder == "drop_first" else 0), rows


def split_masks(n_blocks: int) -> np.ndarray:
    """C(S, S/2) x S booleans, True where a block is in sample; lexicographic order of the in-sample sets."""
    combos = list(itertools.combinations(range(n_blocks), n_blocks // 2))
    masks = np.zeros((len(combos), n_blocks), dtype=bool)
    for i, combo in enumerate(combos):
        masks[i, list(combo)] = True
    return masks


# --------------------------------------------------------------------------- performance


def column_performance(rows: np.ndarray, metric: str) -> np.ndarray:
    """Per-column performance of a rows x K block: mean, or Sharpe (population sd; a constant column: 0)."""
    _choice("metric", metric, METRICS)
    vals = np.asarray(rows, dtype=np.float64)
    mean = vals.mean(axis=0)
    if metric == "mean":
        return mean
    sd = vals.std(axis=0)
    const = vals.max(axis=0) == vals.min(axis=0)
    return np.divide(mean, sd, out=np.zeros_like(mean), where=~const & (sd > 0))


@dataclass(frozen=True)
class _Blocks:
    """Per-block sums of each column (centred on the column mean for precision), lows and highs."""

    count: np.ndarray  # (S,)
    sums: np.ndarray  # (S, K)
    squares: np.ndarray  # (S, K)
    lows: np.ndarray  # (S, K)
    highs: np.ndarray  # (S, K)
    centre: np.ndarray  # (K,)


def _blocks(used: np.ndarray, n_blocks: int, rows: int) -> _Blocks:
    centre = used.mean(axis=0)
    cube = (used - centre).reshape(n_blocks, rows, used.shape[1])
    raw = used.reshape(n_blocks, rows, used.shape[1])
    return _Blocks(
        count=np.full(n_blocks, float(rows)),
        sums=cube.sum(axis=1),
        squares=(cube * cube).sum(axis=1),
        lows=raw.min(axis=1),
        highs=raw.max(axis=1),
        centre=centre,
    )


def _pooled(b: _Blocks, mask: np.ndarray, metric: str) -> np.ndarray:
    """(splits, K) performance over the pooled rows of the blocks marked in mask (splits, S)."""
    weights = mask.astype(np.float64)
    n = (weights @ b.count)[:, None]
    mean_c = (weights @ b.sums) / n
    mean = b.centre[None, :] + mean_c
    if metric == "mean":
        return mean
    var = (weights @ b.squares) / n - mean_c * mean_c
    low = np.where(mask[:, :, None], b.lows[None], np.inf).min(axis=1)
    high = np.where(mask[:, :, None], b.highs[None], -np.inf).max(axis=1)
    sd = np.sqrt(np.maximum(var, 0.0))
    ok = (high > low) & (sd > 0)
    return np.divide(mean, sd, out=np.zeros_like(mean), where=ok)


def choose(is_perf: np.ndarray, oos_perf: np.ndarray, is_tie: str) -> np.ndarray:
    """Index of the in-sample best per split; a tie goes to the worst out of sample ("worst") or the first."""
    best = is_perf.max(axis=1, keepdims=True)
    tied = is_perf == best
    if is_tie == "first":
        return tied.argmax(axis=1)
    return np.where(tied, oos_perf, np.inf).argmin(axis=1)


def oos_rank(oos_perf: np.ndarray, chosen: np.ndarray, oos_tie: str) -> np.ndarray:
    """Rank 1..K (1 = worst) of the chosen column among the out-of-sample values of its split."""
    mine = oos_perf[np.arange(len(chosen)), chosen][:, None]
    below = (oos_perf < mine).sum(axis=1).astype(np.float64)
    if oos_tie == "min":
        return below + 1.0
    equal = (oos_perf == mine).sum(axis=1).astype(np.float64)
    return below + (equal + 1.0) / 2.0


# --------------------------------------------------------------------------- PBO


@dataclass(frozen=True)
class PboResult:
    pbo: float
    n_splits: int
    k: int
    t: int
    rows_used: int
    rows_dropped: int
    block_rows: int
    prob_oos_loss: float
    degradation_slope: float | None
    degradation_intercept: float | None
    chosen: tuple[int, ...]
    is_perf: tuple[float, ...]
    oos_perf: tuple[float, ...]
    oos_rank: tuple[float, ...]
    logits: tuple[float, ...]
    conventions: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self, detail: bool = False) -> dict[str, object]:
        """JSON-ready summary; detail=True adds the per-split lists (chosen, perf in/out, ranks, logits)."""
        out: dict[str, object] = {
            "pbo": self.pbo,
            "n_splits": self.n_splits,
            "k": self.k,
            "t": self.t,
            "rows_used": self.rows_used,
            "rows_dropped": self.rows_dropped,
            "block_rows": self.block_rows,
            "prob_oos_loss": self.prob_oos_loss,
            "degradation_slope": self.degradation_slope,
            "degradation_intercept": self.degradation_intercept,
            "logit_median": float(np.median(self.logits)),
            "conventions": dict(self.conventions),
        }
        if detail:
            out |= {
                "chosen": list(self.chosen),
                "is_perf": list(self.is_perf),
                "oos_perf": list(self.oos_perf),
                "oos_rank": list(self.oos_rank),
                "logits": list(self.logits),
            }
        return out


def _degradation(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    if len(x) < 2 or float(np.ptp(x)) == 0.0:
        return None, None
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def pbo_cscv(
    matrix: np.ndarray | Sequence[Sequence[float]],
    *,
    metric: str = "sharpe",
    n_blocks: int = DEFAULT_BLOCKS,
    min_block_rows: int = DEFAULT_MIN_BLOCK_ROWS,
    remainder: str = "drop_first",
    is_tie: str = "worst",
    oos_tie: str = "min",
) -> PboResult:
    """PBO of a T x K matrix (rows = periods in time order, columns = EVERY configuration of the declared
    grid; the caller owes the whole grid, K >= 2). Conventions: module docstring."""
    conv = Conventions(metric, n_blocks, min_block_rows, remainder, is_tie, oos_tie)
    arr = check_matrix(matrix)
    t, k = arr.shape
    start, rows = block_layout(t, n_blocks, remainder, min_block_rows)
    used = arr[start : start + rows * n_blocks]
    blocks = _blocks(used, n_blocks, rows)
    masks = split_masks(n_blocks)
    parts: list[tuple[np.ndarray, ...]] = []
    for lo in range(0, len(masks), SPLIT_CHUNK):
        chunk = masks[lo : lo + SPLIT_CHUNK]
        ins, outs = _pooled(blocks, chunk, metric), _pooled(blocks, ~chunk, metric)
        picked = choose(ins, outs, is_tie)
        idx = np.arange(len(picked))
        parts.append((picked, ins[idx, picked], outs[idx, picked], oos_rank(outs, picked, oos_tie)))
    chosen, is_best, oos_best, ranks = (np.concatenate(col) for col in zip(*parts, strict=True))
    w = ranks / (k + 1)
    logits = np.log(w / (1.0 - w))
    slope, intercept = _degradation(is_best, oos_best)
    return PboResult(
        pbo=float((logits <= 0).mean()),
        n_splits=len(masks),
        k=k,
        t=t,
        rows_used=rows * n_blocks,
        rows_dropped=t - rows * n_blocks,
        block_rows=rows,
        prob_oos_loss=float((oos_best < 0).mean()),
        degradation_slope=slope,
        degradation_intercept=intercept,
        chosen=tuple(int(v) for v in chosen),
        is_perf=tuple(float(v) for v in is_best),
        oos_perf=tuple(float(v) for v in oos_best),
        oos_rank=tuple(float(v) for v in ranks),
        logits=tuple(float(v) for v in logits),
        conventions=conv.to_dict(),
    )


# ------------------------------------------------------------------------- the declared grid of a hypothesis


@dataclass(frozen=True)
class GridMatrix:
    """The T x K daily net matrix of one hypothesis's WHOLE declared grid on one root and window."""

    hypothesis_id: str
    root: str
    window: tuple[str, str]
    n: int  # the live N when read (report N with every figure)
    params: tuple[Mapping[str, Any], ...]  # the declared grid points, in declared (record key) order
    run_ids: tuple[str, ...]  # the latest ok trial run of each grid point
    superseded_run_ids: tuple[str, ...]  # earlier ok runs of a grid point (re-runs), not in the matrix
    trade_dates: tuple[date, ...]
    matrix: np.ndarray  # T x K int64 cents
    # runs whose result record predates artifact_sha256: their daily file was checked by sum and row count
    sum_checked_run_ids: tuple[str, ...] = ()

    def describe(self) -> dict[str, object]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "root": self.root,
            "window": list(self.window),
            "n": self.n,
            "k": len(self.run_ids),
            "t": len(self.trade_dates),
            "params": [dict(p) for p in self.params],
            "run_ids": list(self.run_ids),
            "superseded_run_ids": list(self.superseded_run_ids),
            "sum_checked_run_ids": list(self.sum_checked_run_ids),
        }


def declared_points(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Every configuration of a declared grid: the Cartesian product, in the record's key order."""
    names = list(grid)
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*(grid[n] for n in names))]


def _key(params: Mapping[str, Any]) -> str:
    return json.dumps(dict(params), sort_keys=True)


def _only(name: str, given: Any, declared: Sequence[Any], hypothesis_id: str) -> Any:
    if given is not None:
        if given not in declared:
            raise PboError(f"{name} {given!r} is not one of hypothesis {hypothesis_id!r}'s {list(declared)}")
        return given
    if len(declared) != 1:
        raise PboError(f"hypothesis {hypothesis_id!r} declares {len(declared)} {name}s: name one")
    return declared[0]


def _window(window: tuple[str | date, str | date] | None) -> list[str] | None:
    if window is None:
        return None
    return [w.isoformat() if isinstance(w, date) else str(w) for w in window]


def _latest_ok_runs(
    state: LedgerState, hypothesis_id: str, root: str, window: list[str]
) -> tuple[dict[str, str], list[str]]:
    """params key -> latest ok trial run on root and window (ledger order), and the superseded run ids."""
    latest: dict[str, str] = {}
    superseded: list[str] = []
    for run_id, begin in state.begun.items():
        result = state.results.get(run_id)
        if begin["kind"] != TRIAL or begin["hypothesis_id"] != hypothesis_id or result is None:
            continue
        if (
            result["status"] != STATUS_OK
            or begin["root"] != root
            or [begin["first"], begin["last"]] != window
        ):
            continue
        key = _key(begin["params"])
        if key in latest:
            superseded.append(latest[key])
        latest[key] = run_id
    return latest, superseded


def _read_daily(runs_dir: Path, run_id: str) -> pl.DataFrame:
    path = runs_dir / run_id / DAILY_FILE
    if not path.is_file():
        raise GridIncomplete(f"run {run_id}: {DAILY_FILE} is missing under {runs_dir}")
    frame = pl.read_parquet(path)
    missing = [c for c in DAILY_COLUMNS if c not in frame.columns]
    if missing:
        raise PboError(f"run {run_id}: {DAILY_FILE} lacks {missing}")
    return frame.select(DAILY_COLUMNS)


def _record_int(run_id: str, metrics: Any, key: str) -> int:
    value = metrics.get(key) if isinstance(metrics, Mapping) else None
    if not isinstance(value, int) or isinstance(value, bool):
        raise GridFileMismatch(
            f"run {run_id}: the result record has no integer metrics.{key} to check {DAILY_FILE} against"
        )
    return value


def check_against_record(run_id: str, frame: pl.DataFrame, metrics: Any) -> None:
    """For a result record written before artifact_sha256: the daily file is the one the trial wrote when
    sum(net_cents) == metrics.net_cents and its row count == metrics.n_days of the record, or
    GridFileMismatch. A record with the key is checked by the hash instead (check_daily_file)."""
    want_net = _record_int(run_id, metrics, RECORD_NET_KEY)
    want_days = _record_int(run_id, metrics, RECORD_DAYS_KEY)
    got_net = int(frame["net_cents"].cast(pl.Int64).sum())
    if got_net != want_net or frame.height != want_days:
        raise GridFileMismatch(
            f"run {run_id}: {DAILY_FILE} has {frame.height} days netting {got_net} cents, its result record "
            f"{want_days} days netting {want_net} cents; the file is not the one the trial wrote"
        )


def check_daily_file(run_dir: Path, run_id: str, frame: pl.DataFrame, result: Mapping[str, Any]) -> bool:
    """The run's daily file is the one the trial wrote, or GridFileMismatch. A result record carrying
    artifact_sha256 must name the daily file (<base>/<run_id>/daily.parquet) and the file in run_dir must hash
    to it (pfb.registry.artifacts.folder_problems on that one entry); returns True. A record written before
    the key falls back to check_against_record (sum and row count); returns False."""
    shas = recorded_shas(result)
    if shas is None:
        check_against_record(run_id, frame, result.get("metrics"))
        return False
    named = {name: sha for name, sha in shas.items() if file_in_folder(name, run_id) == DAILY_FILE}
    if not named:
        raise GridFileMismatch(
            f"run {run_id}: its result record's artifact_sha256 does not name {DAILY_FILE} "
            f"({len(shas)} artifact(s) named); the file cannot be the one the trial wrote"
        )
    problems = folder_problems(run_dir, run_id, named)
    if problems:
        raise GridFileMismatch(
            f"run {run_id}: {'; '.join(problems)}; the file is not the one the trial wrote"
        )
    return True


def grid_matrix(
    config: RegistryConfig,
    hypothesis_id: str,
    *,
    runs_dir: Path | None = None,
    root: str | None = None,
    window: tuple[str | date, str | date] | None = None,
) -> GridMatrix:
    """Read-only: the daily nets of the latest ok trial of EVERY declared grid point, or GridIncomplete.

    root and window may be left out when the hypothesis declares exactly one. A book hypothesis is refused
    (its members are not a grid). Each daily file is checked against its run's result record
    (check_daily_file; GridFileMismatch): by the record's artifact_sha256, or by the net sum and row count for
    a record written before that key (those runs are sum_checked_run_ids). No ledger line is written; N is
    reported, not changed.
    """
    state = read_state(config)
    hyp = state.hypotheses.get(hypothesis_id)
    if hyp is None:
        raise HypothesisNotRegistered(f"hypothesis {hypothesis_id!r} is not registered")
    if is_book_record(hyp):
        raise PboError(f"hypothesis {hypothesis_id!r} is a book: its members are not a declared grid")
    the_root = _only("root", root, hyp["roots"], hypothesis_id)
    the_window = _only("window", _window(window), hyp["windows"], hypothesis_id)
    points = declared_points(hyp["grid"])
    if len(points) < MIN_CONFIGURATIONS:
        raise PboError(f"hypothesis {hypothesis_id!r} declares {len(points)} configuration(s); PBO needs 2")
    latest, superseded = _latest_ok_runs(state, hypothesis_id, the_root, the_window)
    missing = [p for p in points if _key(p) not in latest]
    if missing:
        raise GridIncomplete(
            f"hypothesis {hypothesis_id!r} on {the_root} {the_window}: {len(missing)} of {len(points)} "
            f"declared configurations have no ok trial result (budget {hyp['budget']}), first {missing[0]}; "
            "PBO over a subset of the grid would understate overfitting and is refused"
        )
    run_ids = [latest[_key(p)] for p in points]
    folder = runs_dir if runs_dir is not None else config.repo_root / "runs"
    frames = {run_id: _read_daily(folder, run_id) for run_id in run_ids}
    _check_runs(frames)  # integer nets, no nulls, increasing dates, the same days in every run
    sum_checked = tuple(
        run_id
        for run_id in run_ids
        if not check_daily_file(folder / run_id, run_id, frames[run_id], state.results[run_id])
    )
    matrix = np.column_stack([frames[r]["net_cents"].to_numpy().astype(np.int64) for r in run_ids])
    return GridMatrix(
        hypothesis_id=hypothesis_id,
        root=the_root,
        window=(the_window[0], the_window[1]),
        n=state.n,
        params=tuple(points),
        run_ids=tuple(run_ids),
        superseded_run_ids=tuple(superseded),
        trade_dates=tuple(frames[run_ids[0]]["trade_date"].to_list()),
        matrix=matrix,
        sum_checked_run_ids=sum_checked,
    )


def for_hypothesis(
    hypothesis_id: str,
    *,
    config: RegistryConfig,
    runs_dir: Path | None = None,
    root: str | None = None,
    window: tuple[str | date, str | date] | None = None,
    detail: bool = False,
) -> dict[str, object]:
    """PBO of one registered hypothesis's whole declared grid (grid_matrix), with N and the run ids.

    NO convention argument: the module's defaults (Conventions()) always, echoed under pbo.conventions
    detail only adds the per-split lists. Read-only on the registry: nothing is appended
    and N is unchanged.
    """
    grid = grid_matrix(config, hypothesis_id, runs_dir=runs_dir, root=root, window=window)
    result = pbo_cscv(grid.matrix, **HYPOTHESIS_CONVENTIONS.kwargs())
    warnings = []
    if grid.superseded_run_ids:
        warnings.append(
            f"{len(grid.superseded_run_ids)} earlier ok run(s) of a grid point were superseded by a later "
            "run of the same point and are not in the matrix"
        )
    if grid.sum_checked_run_ids:
        warnings.append(
            f"{len(grid.sum_checked_run_ids)} run(s) have a result record written before artifact_sha256: "
            "their daily files were checked by net sum and row count only (sum_checked_run_ids)"
        )
    return {**grid.describe(), "pbo": result.to_dict(detail=detail), "warnings": warnings}
