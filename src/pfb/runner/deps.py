"""What the runner needs from the other packages, as injectable callables (tests pass fakes).

`default_deps()` wires the real modules. The engine modules are imported by name at call time (importlib)
so that
the runner imports and type-checks while pfb.engine is still being written; a missing name fails loudly there.
"""

from __future__ import annotations

import functools
import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from pfb.registry.api import RegistryConfig
from pfb.registry.codehash import calendars_sha256
from pfb.runner.secondary import grid_dsr, prop_secondary

DataFiles = tuple[tuple[str, str], ...]


def event_coverage(first: date, last: date, paths_toml: Path | None = None) -> None:
    """The pinned release calendar (config/costs.toml [events]) covers first..last, else
    pfb.costs.events.EventCoverageError."""
    from pfb.costs.events import load_pinned_calendar
    from pfb.data.paths import DEFAULT_PATHS_TOML, load_paths

    calendar = load_paths(paths_toml or DEFAULT_PATHS_TOML).release_calendar
    load_pinned_calendar(calendar).require_covers(first, last)


@dataclass(frozen=True)
class RunnerDeps:
    load_strategy: Callable[[Path], Any]
    lint_strategy: Callable[[Path], Sequence[object]]
    lookahead_check: Callable[[Any, Mapping[str, Any], int], Sequence[str] | None]
    run_strategy: Callable[..., Any]
    make_permit: Callable[..., Any]
    engine_version: Callable[[], str]
    instruments_sha256: Callable[[], str]
    # (cost spec, event_costs) -> (model id, config sha256): "off" is the spec's own model, "on" its
    # EventCostModel (id <spec>+events)
    cost_model_ids: Callable[[str, str], tuple[str, str]]
    fees_sha256: Callable[[str], str]
    closed_windows_sha256: Callable[[], str]
    data_files: Callable[[str, date, date], DataFiles]
    provenance_files: Callable[[Any], DataFiles]
    registry_config: Callable[[], RegistryConfig]
    store_roots: Callable[[], Mapping[str, Path]]  # the bar-store roots the engine reads
    # (docs/TRIALS.md); defaults are the real functions, tests pass fakes
    calendars_sha256: Callable[[], str] = calendars_sha256
    grid_dsr: Callable[[Any, int, Sequence[float]], Mapping[str, Any]] = grid_dsr
    prop_secondary: Callable[[Path, str, Mapping[str, Any]], Mapping[str, Any]] = prop_secondary
    # raises (pfb.costs.events.EventCoverageError) when the pinned release calendar does not
    # cover first..last; called for event_costs "on" at registration (every declared window) and before
    # each begin
    event_coverage: Callable[[date, date], None] = event_coverage


def _engine(module: str, name: str) -> Any:
    return getattr(importlib.import_module(f"pfb.engine.{module}"), name)


def store_relative(path: str, roots: Mapping[str, Path]) -> str:
    """'<store>/<path inside it>' (research, step2 or hist): no home path enters a tracked record."""
    candidate = Path(path)
    for name, root in roots.items():
        for base in (root, root.resolve()):
            for target in (candidate, candidate.resolve()):
                if target.is_relative_to(base):
                    return f"{name}/{target.relative_to(base).as_posix()}"
    raise ValueError(f"data file {candidate.name} lies under no configured store root")


def _field(item: Any, key: str) -> str:
    value = item.get(key) if isinstance(item, Mapping) else getattr(item, key)
    if not isinstance(value, str):
        raise TypeError(f"provenance item has no string {key!r}")
    return value


def provenance_pairs(provenance: Any, roots: Mapping[str, Path]) -> DataFiles:
    """(path, sha256) of every data file a RunResult's provenance names, in its order."""
    items = provenance
    if isinstance(provenance, Mapping):
        items = provenance.get("data_files", provenance.get("files", ()))
    return tuple((store_relative(_field(i, "path"), roots), _field(i, "sha256")) for i in items)


def _lookahead(strategy: Any, params: Mapping[str, Any], seed: int) -> tuple[str, ...]:
    """Step 1's check on synthetic bars (pfb.engine.lookahead.lookahead_check): 20 seeded
    random cuts, cuts after entry then exit bars, a cut after every bar of SWEEP_SESSIONS sessions spread
    evenly, the full call last. Returns the check's warnings ("look-ahead check saw no signal") for the
    result; raises LookAheadDetected."""
    bars = _engine("synthetic", "synthetic_bars")(strategy.spec.root, seed=seed)
    sessions = _engine("lookahead", "SWEEP_SESSIONS")
    check = _engine("lookahead", "lookahead_check")(strategy.generate, bars, dict(params), seed,
                                                    sweep_sessions=sessions)  # fmt: skip
    return tuple(check.warnings)


def default_deps(paths_toml: Path | None = None) -> RunnerDeps:
    """The real modules: instruments, costs, data (gateway files_for) and the engine. config/paths.toml is
    read on first use (without it, the demo store's paths); a broken paths file raises PathsConfigError at
    the first step that needs it (registry config, store roots, data files, costs, coverage)."""
    from pfb.costs.events import load_cost_model_for, load_pinned_calendar
    from pfb.costs.fees import load_fees
    from pfb.data.catalog import load_catalog
    from pfb.data.gateway import DataGateway
    from pfb.data.paths import DEFAULT_PATHS_TOML, StorePaths, load_paths
    from pfb.data.windows import load_windows
    from pfb.instruments.table import load_instruments

    instruments = load_instruments()

    @functools.cache
    def paths() -> StorePaths:
        return load_paths(paths_toml or DEFAULT_PATHS_TOML)

    def roots() -> Mapping[str, Path]:
        return paths().store_roots()

    def data_files(root: str, first: date, last: date) -> DataFiles:
        gateway = DataGateway(
            paths(), load_windows(), load_catalog(), instruments.tick_fixed_of, instruments.price_factor_of
        )
        named = roots()
        return tuple((store_relative(e.path, named), e.sha256) for e in gateway.files_for(root, first, last))

    def cost_ids(spec: str, event_costs: str) -> tuple[str, str]:
        model = load_cost_model_for(spec, event_costs, instruments, paths())
        return model.model_id, model.config_sha256

    def coverage(first: date, last: date) -> None:
        load_pinned_calendar(paths().release_calendar).require_covers(first, last)

    return RunnerDeps(
        load_strategy=lambda p: _engine("strategy", "load_strategy")(p),
        lint_strategy=lambda p: _engine("lint", "lint_strategy")(p),
        lookahead_check=_lookahead,
        run_strategy=lambda *a, **k: _engine("run", "run_strategy")(*a, **k),
        make_permit=lambda **k: _engine("permit", "RunPermit")(**k),
        engine_version=lambda: str(importlib.import_module("pfb.engine").ENGINE_VERSION),
        instruments_sha256=lambda: instruments.sha256,
        cost_model_ids=cost_ids,
        fees_sha256=lambda firm: load_fees(firm, instruments=instruments).sha256,
        closed_windows_sha256=lambda: load_windows().sha256,
        data_files=data_files,
        provenance_files=lambda result: provenance_pairs(result.provenance, roots()),
        registry_config=lambda: RegistryConfig.for_repo(paths().upstream_root),
        store_roots=roots,
        event_coverage=coverage,
    )
