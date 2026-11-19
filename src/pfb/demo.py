"""`pfb demo`: the whole pipeline on synthetic data, in well under two minutes, with no vendor data.

Steps: build (or reuse) the demo store (pfb.data.demo) and its catalog; read it through the public
DataGateway constructor, once on an open range and once on a range that touches a closed window (refused);
`pfb check` on the reference strategy; one synthetic run with its analysis suite and a prop Monte Carlo on one
firm plan at a fixed seed. A synthetic run is logged in the ledger and adds nothing to N. The ledger gets its
genesis record first when it is absent or empty.

Nothing here reads a real store. Without config/paths.toml the demo store is also the default store, so the
catalog is written to runs/catalog.json (the dashboard's data view reads it); with a paths file the catalog of
your own stores is left alone.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

from pfb import app as service
from pfb.data.catalog import DEFAULT_CATALOG_JSON
from pfb.data.demo import DEMO_ROOTS, build_demo_store, demo_catalog
from pfb.data.errors import ClosedWindowRefused
from pfb.data.gateway import DataGateway
from pfb.data.paths import DEFAULT_PATHS_TOML, DEMO_STORE_ROOT, load_paths
from pfb.data.windows import load_windows
from pfb.instruments import load_instruments
from pfb.registry.api import LEDGER_REL, REPO_ROOT, RegistryConfig, create_genesis, status
from pfb.registry.hypfile import PROGRAM_LOOKAHEAD_SEED

REFERENCE_STRATEGY = REPO_ROOT / "strategies" / "reference_breakout.py"
DEFAULT_FIRM = "topstep"
DEFAULT_PLAN = "combine_xfa_standard_50k"
DEFAULT_MC = 200
GATEWAY_ROOT = "MNQ"
OPEN_RANGE = (date(2026, 4, 27), date(2026, 5, 8))  # open by label
CLOSED_RANGE = (date(2026, 5, 4), date(2026, 5, 15))  # touches the embargo of config/closed_windows.toml
RUN_KEYS = (
    "run_id", "kind", "status", "root", "window", "n_added", "net_cents", "trades", "sharpe",
    "max_drawdown_cents",
)  # fmt: skip


@dataclass(frozen=True)
class DemoLocations:
    """Where the demo writes: the store, the catalog (None: not written), the runs and the ledger."""

    store_dir: Path = DEMO_STORE_ROOT
    catalog_json: Path | None = DEFAULT_CATALOG_JSON
    runs_dir: Path = REPO_ROOT / "runs"
    ledger: Path = REPO_ROOT / LEDGER_REL

    @classmethod
    def default(cls) -> DemoLocations:
        """The repository's locations; the catalog only when no config/paths.toml names other stores."""
        return cls(catalog_json=None if DEFAULT_PATHS_TOML.is_file() else DEFAULT_CATALOG_JSON)


def _registry(ledger: Path) -> RegistryConfig:
    config = RegistryConfig(registry_path=ledger, pe_root=load_paths().upstream_root, repo_root=REPO_ROOT)
    if not ledger.is_file() or ledger.stat().st_size == 0:
        create_genesis(config)
    return config


def _store_step(locations: DemoLocations) -> dict[str, Any]:
    store = build_demo_store(locations.store_dir)
    windows = load_windows()
    catalog = demo_catalog(store, windows.sha256, locations.catalog_json)
    instruments = load_instruments()
    gateway = DataGateway(
        store.paths, windows, catalog, instruments.tick_fixed_of, instruments.price_factor_of
    )
    frame = gateway.bars(GATEWAY_ROOT, *OPEN_RANGE)
    try:
        gateway.bars(GATEWAY_ROOT, *CLOSED_RANGE)
        refusal = None
    except ClosedWindowRefused as exc:
        refusal = str(exc)
    return {
        "store_dir": str(store.paths.research.parent),
        "built": store.built,
        "roots": list(DEMO_ROOTS),
        "files": len(catalog.files),
        "rows": sum(e.num_rows for e in catalog.files),
        "catalog": None if locations.catalog_json is None else str(locations.catalog_json),
        "open_read": {"root": GATEWAY_ROOT, "first": OPEN_RANGE[0].isoformat(),
                      "last": OPEN_RANGE[1].isoformat(), "rows": frame.height},  # fmt: skip
        "closed_read": {"first": CLOSED_RANGE[0].isoformat(), "last": CLOSED_RANGE[1].isoformat(),
                        "refused": refusal},  # fmt: skip
    }


def run_demo(
    locations: DemoLocations | None = None,
    *,
    strategy: Path = REFERENCE_STRATEGY,
    firm: str = DEFAULT_FIRM,
    plan: str = DEFAULT_PLAN,
    mc: int = DEFAULT_MC,
    seed: int = PROGRAM_LOOKAHEAD_SEED,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run every demo step; returns a summary (the run's own contract is saved under runs/<run_id>/)."""
    where = locations or DemoLocations.default()
    say = progress or (lambda _text: None)
    say("1/4 demo store: build or reuse, then read it through the DataGateway")
    store = _store_step(where)
    config = _registry(where.ledger)
    from pfb.runner.deps import default_deps

    deps = replace(default_deps(), registry_config=lambda: config)
    with service.using(deps=deps, runs_dir=where.runs_dir, ledger=where.ledger):
        say("2/4 pfb check: lint and look-ahead check on synthetic bars")
        check = service.validate_strategy(str(strategy), seed=seed)
        say(f"3/4 synthetic run with prop Monte Carlo ({firm} {plan}, {mc} paths, seed {seed})")
        run = service.run_backtest(str(strategy), kind="synthetic", seed=seed, firm=firm, plan=plan, mc=mc)
    say("4/4 ledger status")
    report = status(config)
    return {
        "store": store,
        "check": {"strategy": strategy.name, "seed": seed, "warnings": list(check.get("warnings", []))},
        "run": {key: run.get(key) for key in RUN_KEYS},
        "prop": run.get("prop"),
        "runs_dir": str(where.runs_dir),
        "ledger": {"path": str(where.ledger), "n": report.n, "records": report.records},
    }
