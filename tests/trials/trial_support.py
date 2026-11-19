"""A fake engine and fake gateway for trial-flow tests: records every call, reads no bar."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
from registry_env import Env, git  # pyright: ignore[reportMissingImports]

from pfb.data.errors import GatewayRefusal
from pfb.registry.chain import read_ledger, sha256_hex
from pfb.runner.deps import RunnerDeps

DATA = (("research/ohlcv-1m_MNQ_research.parquet", "d" * 64),)
CALENDARS = "e" * 64
EVENTS_SHA = "f" * 64  # the fake "+events" model's config sha256 (differs from the base's "b" * 64)


def cost_ids(spec: str, event_costs: str) -> tuple[str, str]:
    """The fake cost identity: "off" -> (spec, b...), "on" -> (spec+events, f...), shaped as the real one."""
    return (spec, "b" * 64) if event_costs == "off" else (f"{spec}+events", EVENTS_SHA)


@dataclass
class FakeTrialEngine:
    """Engine, gateway and prop fakes. `summaries` maps a param value of `a` to the summary returned."""

    calls: list[tuple[str, Any]] = field(default_factory=list)
    crash: bool = False
    lookahead_warnings: tuple[str, ...] = ()
    summaries: Mapping[Any, dict[str, Any]] = field(default_factory=dict)
    data_error: GatewayRefusal | None = None
    no_files: bool = False
    extras: bool = False
    prop_error: Exception | None = None
    coverage_error: Exception | None = None
    run_error: Exception | None = None  # run_strategy raises it after begin (LockedExitThroughSessionEnd)

    def strategy(self, path: Path) -> Any:
        params = {"a": SimpleNamespace(default=1, grid=(1, 2, 3))}
        spec = SimpleNamespace(
            root="MNQ", session="globex", bar_minutes=1, hypothesis_id="H1", params=params, contracts=1
        )
        return SimpleNamespace(spec=spec, sha256=sha256_hex(path.read_bytes()), generate=None, path=path)

    def lookahead(self, strategy: Any, params: Mapping[str, Any], seed: int) -> tuple[str, ...]:
        self.calls.append(("lookahead", dict(params)))
        return self.lookahead_warnings

    def summary_for(self, params: Mapping[str, Any]) -> dict[str, Any]:
        default = {"net_cents": 100, "sharpe_annual": 1.0, "sharpe_daily": 0.06, "warnings": []}
        return dict(self.summaries.get(params.get("a"), default))

    def run_strategy(self, path: Path, **kwargs: Any) -> Any:
        self.calls.append(("run_strategy", kwargs))
        if self.crash:
            raise ZeroDivisionError("engine crashed")
        if self.run_error is not None:
            raise self.run_error
        frame = pl.DataFrame({"x": [1]})
        summary = self.summary_for(kwargs["params"])
        if "suite" in kwargs:
            summary["suite"] = {"cost_stress": None, "walk_forward": None}
        extras = {"rolling_sharpe": frame} if self.extras else {}
        return SimpleNamespace(trades=frame, equity_minute=frame, daily=frame, summary=summary,
                               provenance=DATA, extras=extras)  # fmt: skip

    def data_files(self, root: str, first: date, last: date) -> tuple[tuple[str, str], ...]:
        self.calls.append(("data_files", (root, first, last)))
        if self.data_error is not None:
            raise self.data_error
        return () if self.no_files else DATA

    def grid_dsr(self, result: Any, n_trials: int, grid: Sequence[float]) -> dict[str, Any]:
        self.calls.append(("grid_dsr", (n_trials, list(grid))))
        return {"dsr_n_grid": len(grid)}

    def prop_secondary(self, run_dir: Path, root: str, secondary: Mapping[str, Any]) -> dict[str, Any]:
        self.calls.append(("prop_secondary", (run_dir.name, root, dict(secondary))))
        if self.prop_error is not None:
            raise self.prop_error
        return {"rows": [], "declared": list(secondary["prop_plans"])}

    def deps(self, env: Env) -> RunnerDeps:
        base = env.repo.parent / "synthetic_stores"
        return RunnerDeps(
            load_strategy=self.strategy,
            lint_strategy=lambda p: [],
            lookahead_check=self.lookahead,
            run_strategy=self.run_strategy,
            make_permit=lambda **k: SimpleNamespace(**k),
            engine_version=lambda: "pfb-engine/test",
            instruments_sha256=lambda: "a" * 64,
            cost_model_ids=cost_ids,
            fees_sha256=lambda firm: "a" * 64,
            closed_windows_sha256=lambda: "b" * 64,
            data_files=self.data_files,
            provenance_files=lambda result: result.provenance,
            registry_config=lambda: env.config,
            store_roots=lambda: {name: base / name for name in ("research", "step2", "hist")},
            calendars_sha256=lambda: CALENDARS,
            grid_dsr=self.grid_dsr,
            prop_secondary=self.prop_secondary,
            event_coverage=self.event_coverage,
        )

    def event_coverage(self, first: date, last: date) -> None:
        """The calendar check of event_costs "on": recorded; refused when coverage_error."""
        self.calls.append(("event_coverage", (first, last)))
        if self.coverage_error is not None:
            raise self.coverage_error

    def called(self, name: str) -> list[Any]:
        return [args for call, args in self.calls if call == name]


def kinds(env: Env) -> list[str]:
    return [line.record["kind"] for line in read_ledger(env.config.registry_path)]


def n_of(env: Env) -> int:
    return read_ledger(env.config.registry_path)[-1].record["n_after"]


def clean(env: Env) -> bool:
    """Nothing differs from HEAD, the ledger included."""
    return git(env.repo, "status", "--porcelain", "--untracked-files=all") == ""


def temp_paths_toml(base: Path) -> Path:
    """A paths file naming temporary store roots under `base` (empty directories; no bar is read from them),
    for tests that build the real default_deps without config/paths.toml."""
    base.mkdir(parents=True, exist_ok=True)
    roots = {name: base / name for name in ("research", "step2", "hist")}
    for root in roots.values():
        root.mkdir(exist_ok=True)
    toml = base / "paths.toml"
    toml.write_text(
        "[stores]\n"
        + "".join(f'{name} = "{root}"\n' for name, root in roots.items())
        + f'[upstream]\nroot = "{base / "pe"}"\n'
        + f'[costs]\nbook_table = "{base / "pe" / "costs.json"}"\n',
        encoding="utf-8",
    )
    return toml


def without_repo_paths(monkeypatch: Any, base: Path) -> None:
    """Point the runner's real-store guard and the paths default at a file that does not exist, so a test
    behaves the same on a checkout without config/paths.toml (the guard then knows no real store)."""
    import pfb.data.paths as data_paths
    import pfb.runner.api as runner_api

    absent = base / "absent_paths.toml"
    monkeypatch.setattr(runner_api, "PATHS_TOML", absent)
    monkeypatch.setattr(data_paths, "DEFAULT_PATHS_TOML", absent)
