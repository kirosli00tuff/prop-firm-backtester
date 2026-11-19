"""SuiteSpec: which in-run analyses a run computes (docs/TRIALS.md section 2, the [suite] table).

A trial takes its SuiteSpec from the registered hypothesis; nothing here reads the command line. A part with
an empty or zero setting is off: no cost-stress ticks, zero baseline draws, zero walk-forward folds.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

MIN_BASELINE_DRAWS = 200
MIN_FOLDS = 2
MAX_STRESS_TICKS = 100

SUITE_KEYS: tuple[str, ...] = (
    "cost_stress_ticks",
    "baseline_draws",
    "baseline_seed",
    "walk_forward_folds",
    "neighbourhood",
    "sizing_plan",
)


class SuiteError(ValueError):
    """An in-run analysis cannot be computed as specified (the run itself is not affected)."""


class SuiteIntegrityError(RuntimeError):
    """A suite cross-check of the base run's own trades, days or nets failed: not a part that
    cannot be computed but a base run that disagrees with itself. Not one of inrun.SUITE_PART_ERRORS, so the
    run fails."""


def _strict_int(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"suite.{key} must be an integer, got {value!r}")
    return value


@dataclass(frozen=True)
class SuiteSpec:
    """The [suite] table of a hypothesis. Frozen; validated on construction."""

    cost_stress_ticks: tuple[int, ...] = (1, 2)
    baseline_draws: int = 200
    baseline_seed: int = 7
    walk_forward_folds: int = 5
    neighbourhood: bool = False
    sizing_plan: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.cost_stress_ticks, tuple):
            raise ValueError("suite.cost_stress_ticks must be a tuple of integers")
        seen: set[int] = set()
        for k in self.cost_stress_ticks:
            k = _strict_int(k, "cost_stress_ticks")
            if k < 1 or k > MAX_STRESS_TICKS:
                raise ValueError(f"suite.cost_stress_ticks entries must be in 1..{MAX_STRESS_TICKS}, got {k}")
            if k in seen:
                raise ValueError(f"suite.cost_stress_ticks has a duplicate entry {k}")
            seen.add(k)
        draws = _strict_int(self.baseline_draws, "baseline_draws")
        if draws != 0 and draws < MIN_BASELINE_DRAWS:
            raise ValueError(
                f"suite.baseline_draws must be 0 (off) or at least {MIN_BASELINE_DRAWS}, got {draws}"
            )
        _strict_int(self.baseline_seed, "baseline_seed")
        if self.baseline_seed < 0:
            raise ValueError(f"suite.baseline_seed must be >= 0, got {self.baseline_seed}")
        folds = _strict_int(self.walk_forward_folds, "walk_forward_folds")
        if folds != 0 and folds < MIN_FOLDS:
            raise ValueError(f"suite.walk_forward_folds must be 0 (off) or at least {MIN_FOLDS}, got {folds}")
        if not isinstance(self.neighbourhood, bool):
            raise ValueError(f"suite.neighbourhood must be true or false, got {self.neighbourhood!r}")
        if not isinstance(self.sizing_plan, str):
            raise ValueError(f"suite.sizing_plan must be a string, got {self.sizing_plan!r}")
        if self.sizing_plan and self.sizing_plan.count("/") != 1:
            raise ValueError(f"suite.sizing_plan must be '<firm>/<plan>' or '', got {self.sizing_plan!r}")

    @classmethod
    def off(cls) -> SuiteSpec:
        """Every part off: no stress ticks, no baseline draws, no folds, no neighbourhood, no sizing plan."""
        return cls(
            cost_stress_ticks=(),
            baseline_draws=0,
            baseline_seed=0,
            walk_forward_folds=0,
            neighbourhood=False,
            sizing_plan="",
        )

    @classmethod
    def from_mapping(cls, table: Mapping[str, object]) -> SuiteSpec:
        """Build from a parsed [suite] table: exactly SUITE_KEYS; missing or extra keys raise ValueError."""
        if not isinstance(table, Mapping):
            raise ValueError("suite must be a table")
        keys = set(table)
        missing = sorted(set(SUITE_KEYS) - keys)
        extra = sorted(keys - set(SUITE_KEYS))
        if missing or extra:
            raise ValueError(f"suite keys wrong: missing {missing}, extra {extra}")
        ticks = table["cost_stress_ticks"]
        if not isinstance(ticks, list | tuple):
            raise ValueError(f"suite.cost_stress_ticks must be a list of integers, got {ticks!r}")
        return cls(
            cost_stress_ticks=tuple(ticks),  # type: ignore[arg-type]  # validated in __post_init__
            baseline_draws=table["baseline_draws"],  # type: ignore[arg-type]
            baseline_seed=table["baseline_seed"],  # type: ignore[arg-type]
            walk_forward_folds=table["walk_forward_folds"],  # type: ignore[arg-type]
            neighbourhood=table["neighbourhood"],  # type: ignore[arg-type]
            sizing_plan=table["sizing_plan"],  # type: ignore[arg-type]
        )

    def to_json(self) -> dict[str, object]:
        """The [suite] table as JSON-ready values, keys in SUITE_KEYS order."""
        return {
            "cost_stress_ticks": list(self.cost_stress_ticks),
            "baseline_draws": self.baseline_draws,
            "baseline_seed": self.baseline_seed,
            "walk_forward_folds": self.walk_forward_folds,
            "neighbourhood": self.neighbourhood,
            "sizing_plan": self.sizing_plan,
        }
