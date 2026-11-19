"""SPA, Reality Check and StepM across the whole declared family of one hypothesis.

Wraps arch.bootstrap (arch 9.0.0) SPA, RealityCheck and StepM over a T x K matrix of daily NET results
(integer cents, rows in time order, one column per configuration) against the ZERO benchmark: "does any
configuration of the family earn more than nothing, after the data-snooping correction for all K?"

SIGN CONVENTION (the one place it can go wrong). arch takes LOSSES (lower is better) and forms the
loss differential d = benchmark_loss - model_loss (arch.bootstrap.multiple_comparison.SPA.__init__:
"self._loss_diff = benchmark_arr - models_arr"); a model is superior when E[d] > 0. Our series are
nets (higher is better), so
    model_loss = -net_cents,  benchmark_loss = 0  =>  d = 0 - (-net) = net.
A configuration is superior exactly when its expected daily net is above zero. The tests in
tests/analysis/test_family_tests.py check that arch's loss differential equals the net, that a planted
positive edge is named by StepM with small p-values, and that the same edge negated is not found.

THE CALLER OWES THE WHOLE FAMILY. A family test over a subset of the configurations a hypothesis
declared understates the data-snooping correction (the max over fewer statistics is smaller). The array
functions refuse K < 2 but cannot see which columns were left out of a larger family; the entry for a
hypothesis is `for_hypothesis`, which builds the matrix with pfb.analysis.pbo.grid_matrix: EVERY declared
grid point with an ok trial result, or a refusal (GridIncomplete). Nothing here reads a bar, appends to
the ledger or adds to N; the result is an output about a hypothesis, never an input to choosing
configurations.

FIXED BOOTSTRAP (no argument a caller can turn for a smaller p-value):
- bootstrap: the stationary bootstrap of Politis and Romano (arch "stationary").
- block length: BLOCK RULE "ceil(T^(1/3))", the smallest integer b with b**3 >= T (integer arithmetic):
  the rate-optimal order of the expected block length for the stationary bootstrap; T = 2,600 days
  gives 14. It depends on T only, never on the data's values.
- replications: REPS = 1000 (arch's default). Seed: BOOTSTRAP_SEED = 20261117, a module constant (the
  look-ahead seed is a per-hypothesis value in the hypothesis file, not a program constant); each of the
  three tests is built with seed=BOOTSTRAP_SEED, so the result is a pure function of the matrix.
- SPA (Hansen 2005): studentized (arch's kernel variance, nested=False); the lower, consistent and upper
  p-values are reported, the consistent one is Hansen's.
- Reality Check (White 2000): arch's RealityCheck (a shallow clone of SPA) with studentize=False, and
  its UPPER p-value: every bootstrap column is re-centred by its own sample mean (arch
  SPA._simulate_values: "Upper always re-centers"), the least-favourable null mu = 0 for all K, which is
  White's non-studentized test.
- StepM (Romano and Wolf 2005): studentized, consistent re-centring, family-wise error rate ALPHA = 0.05.
- A column that is constant over the rows (for example a configuration that never traded) has no
  variance: arch warns that such a model should be removed ("Remove these models before testing"), and
  removing it would be a subset, so it is REFUSED. Every arch warning, raised while a test is built or
  computed, is a refusal (FamilyTestError), never a result printed beside a warning.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
from arch.bootstrap import SPA, RealityCheck, StepM

from pfb.analysis.pbo import grid_matrix
from pfb.registry.api import RegistryConfig

BOOTSTRAP = "stationary"
BLOCK_RULE = "ceil(T^(1/3))"
REPS = 1000
BOOTSTRAP_SEED = 20261117
ALPHA = 0.05  # StepM's family-wise error rate; also echoed so a reader compares p-values with it
MIN_CONFIGURATIONS = 2
MIN_ROWS = 3  # arch's own minimum number of observations
PVALUE_KEYS = ("lower", "consistent", "upper")
SIGN = "model_loss = -net_cents, benchmark_loss = 0, arch loss differential = benchmark - model = net"


class FamilyTestError(ValueError):
    """A matrix the family tests refuse."""


# --------------------------------------------------------------------------- checks and the fixed rules


def block_length(t: int) -> int:
    """The stated block rule: the smallest b >= 1 with b**3 >= t, i.e. ceil(t^(1/3)) without floats."""
    if not isinstance(t, int) or isinstance(t, bool) or t < 1:
        raise FamilyTestError(f"T must be an int >= 1, got {t!r}")
    b = max(1, round(t ** (1 / 3)))
    while b**3 < t:
        b += 1
    while b > 1 and (b - 1) ** 3 >= t:
        b -= 1
    return b


def check_family(matrix: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """The T x K matrix of daily nets as float64, or FamilyTestError.

    Refuses a non-2-D or non-numeric array, K < 2 (a family test over fewer than two configurations is not a
    family test, and K must be the WHOLE declared family: this function cannot see that, the caller owes it),
    T < 3, a value that is not finite, and a column that is constant over the rows.
    """
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise FamilyTestError(f"the matrix must be 2-D (T rows x K configurations), got {arr.ndim}-D")
    if arr.shape[1] < MIN_CONFIGURATIONS:
        raise FamilyTestError(
            f"a family test needs at least {MIN_CONFIGURATIONS} configurations (the whole declared family), "
            f"K = {arr.shape[1]}"
        )
    if arr.shape[0] < MIN_ROWS:
        raise FamilyTestError(f"a family test needs at least {MIN_ROWS} rows, T = {arr.shape[0]}")
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.number):
        raise FamilyTestError(f"the matrix must be numeric, got {arr.dtype}")
    out = arr.astype(np.float64)
    if not np.isfinite(out).all():
        raise FamilyTestError("every value must be finite (no NaN or inf)")
    constant = [int(j) for j in np.flatnonzero(np.ptp(out, axis=0) == 0)]
    if constant:
        raise FamilyTestError(
            f"configuration column(s) {constant} are constant over all {out.shape[0]} rows: no variance, "
            "so no studentized statistic; dropping them would test a subset of the family, so the matrix "
            "is refused"
        )
    return out


def _losses(net: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(benchmark losses, model losses) in arch's convention: zero benchmark, model loss = -net."""
    return np.zeros(net.shape[0]), -net


# --------------------------------------------------------------------------- the three tests


def _computed[M: (SPA, StepM)](build: Callable[[], M]) -> M:
    """Build and compute with arch's warnings raised as errors.

    arch warns of a zero-variance column in SPA.__init__, so the constructor runs inside the filter too:
    a warning means the input was not as checked, and it is a refusal.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            test = build()
            test.compute()
        except Warning as w:
            raise FamilyTestError(f"arch refused the matrix: {w}") from w
    return test


def _spa(net: np.ndarray, *, studentize: bool, cls: type[SPA] = SPA) -> SPA:
    bench, models = _losses(net)
    return _computed(
        lambda: cls(
            bench,
            models,
            block_size=block_length(net.shape[0]),
            reps=REPS,
            bootstrap=BOOTSTRAP,
            studentize=studentize,
            seed=BOOTSTRAP_SEED,
        )
    )


def spa_pvalues(matrix: np.ndarray | Sequence[Sequence[float]]) -> dict[str, float]:
    """Hansen's SPA p-values (lower, consistent, upper) of the family against the zero benchmark."""
    test = _spa(check_family(matrix), studentize=True)
    return {key: _pvalue(test, key) for key in PVALUE_KEYS}


def reality_check_pvalue(matrix: np.ndarray | Sequence[Sequence[float]]) -> float:
    """White's Reality Check p-value: non-studentized, every column re-centred (arch's upper p-value)."""
    test = _spa(check_family(matrix), studentize=False, cls=RealityCheck)
    return _pvalue(test, "upper")


def _pvalue(test: SPA, key: str) -> float:
    return float(cast("float", test.pvalues[key]))


def stepm_superior(matrix: np.ndarray | Sequence[Sequence[float]]) -> tuple[int, ...]:
    """Column indices StepM finds superior to the zero benchmark at family-wise error rate ALPHA."""
    net = check_family(matrix)
    bench, models = _losses(net)
    test = _computed(
        lambda: StepM(
            bench,
            models,
            size=ALPHA,
            block_size=block_length(net.shape[0]),
            reps=REPS,
            bootstrap=BOOTSTRAP,
            studentize=True,
            seed=BOOTSTRAP_SEED,
        )
    )
    # models is an ndarray, so arch returns column positions (ints), not labels
    return tuple(sorted(int(cast("int", i)) for i in test.superior_models))


def conventions(t: int) -> dict[str, object]:
    """The fixed bootstrap and the sign convention, echoed with every result."""
    return {
        "benchmark": "zero",
        "sign": SIGN,
        "bootstrap": BOOTSTRAP,
        "block_rule": BLOCK_RULE,
        "block_size": block_length(t),
        "reps": REPS,
        "seed": BOOTSTRAP_SEED,
        "alpha": ALPHA,
        "spa": "studentized, kernel variance (nested=False); consistent p-value is Hansen's",
        "reality_check": "arch RealityCheck, studentize=False, upper p-value (all columns re-centred)",
        "stepm": "studentized, consistent re-centring, size=alpha",
        "arch_version": version("arch"),
    }


@dataclass(frozen=True)
class FamilyResult:
    k: int
    t: int
    mean_net_cents: tuple[float, ...]  # per configuration, the observed mean daily net
    spa_pvalues: Mapping[str, float]
    reality_check_pvalue: float
    stepm_superior: tuple[int, ...]
    conventions: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "k": self.k,
            "t": self.t,
            "mean_net_cents": list(self.mean_net_cents),
            "spa_pvalues": dict(self.spa_pvalues),
            "reality_check_pvalue": self.reality_check_pvalue,
            "stepm_superior": list(self.stepm_superior),
            "conventions": dict(self.conventions),
        }


def family_tests(matrix: np.ndarray | Sequence[Sequence[float]]) -> FamilyResult:
    """SPA, Reality Check and StepM of a T x K net matrix (EVERY configuration of the family, K >= 2)."""
    net = check_family(matrix)
    t, k = net.shape
    return FamilyResult(
        k=k,
        t=t,
        mean_net_cents=tuple(float(v) for v in net.mean(axis=0)),
        spa_pvalues=spa_pvalues(net),
        reality_check_pvalue=reality_check_pvalue(net),
        stepm_superior=stepm_superior(net),
        conventions=conventions(t),
    )


# ----------------------------------------------------------------------- the declared family of a hypothesis


def for_hypothesis(
    hypothesis_id: str,
    *,
    config: RegistryConfig,
    runs_dir: Path | None = None,
    root: str | None = None,
    window: tuple[str | date, str | date] | None = None,
) -> dict[str, object]:
    """The three family tests over one registered hypothesis's WHOLE declared grid, with N and the run ids.

    The matrix is pfb.analysis.pbo.grid_matrix's: every declared grid point's latest ok trial on one root and
    window, or GridIncomplete (a PboError); each daily file checked against its run's result record, or
    GridFileMismatch (by the sha256 in the record's artifact_sha256 where it
    has one; by the sum of net cents and the row count for an older record). NO convention argument:
    the bootstrap is the module's fixed constants, echoed
    under family_tests.conventions. Read-only on the registry: nothing is appended and N is unchanged.
    """
    grid = grid_matrix(config, hypothesis_id, runs_dir=runs_dir, root=root, window=window)
    result = family_tests(grid.matrix)
    warn: list[str] = []
    if grid.superseded_run_ids:
        warn.append(
            f"{len(grid.superseded_run_ids)} earlier ok run(s) of a grid point were superseded by a later "
            "run of the same point and are not in the matrix"
        )
    superior: list[dict[str, Any]] = [dict(grid.params[i]) for i in result.stepm_superior]
    return {
        **grid.describe(),
        "family_tests": result.to_dict(),
        "stepm_superior_params": superior,
        "warnings": warn,
    }
