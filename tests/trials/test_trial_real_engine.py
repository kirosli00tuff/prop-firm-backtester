"""`pfb run --kind trial` through the CLI with the REAL engine,
reading synthetic bars through a test DataGateway (DataGateway.for_synthetic_stores on a store in tmp_path), a
temporary ledger in a temporary git repository and a test hypothesis registered there.

Only the engine's gateway factory is swapped (pfb.engine.run builds its gateway from DataGateway, load_paths,
load_windows and load_catalog; the test points those at the synthetic store); the runner, registry, permit,
engine, suite, contract and CLI are the real ones. This repository's ledger/registry.jsonl is hashed before
and after and must be byte-identical. Every bar is generated here; no real row is read."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from datetime import date, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from registry_env import HYPOTHESIS_REL, STRATEGY_REL, Env  # pyright: ignore[reportMissingImports]
from trial_support import clean, kinds, n_of, temp_paths_toml, without_repo_paths  # pyright: ignore[reportMissingImports]
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from gateway_synth import Row, SynthStores, ct_ns, research_name, write_store  # noqa: E402  # pyright: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parents[2]
REAL_LEDGER = REPO / "ledger" / "registry.jsonl"
ROOT = "MNQ"
FIRST, LAST = date(2021, 1, 19), date(2021, 2, 12)  # no holiday; open in the synthetic and real window tables
SESSION_OPEN, BARS_PER_DAY = time(8, 30), 390
MNQ_TICK_FIXED = 250_000_000  # 0.25 index points, 1e-9 fixed point
BASE_TICKS = 40_000
WINDOWS_TOML = f"[[window]]\nfirst = {FIRST.isoformat()}\nlast = {LAST.isoformat()}\n"

STRATEGY = """import numpy as np

SPEC = {
    "hypothesis_id": "real-engine-trial",
    "root": "MNQ",
    "bar_minutes": 1,
    "session": "globex",
    "contracts": 1,
    "params": {"a": {"default": 1, "grid": [1, 2, 3]}},
}


def generate(bars, params):
    n = len(bars["close"])
    entry = np.zeros(n, dtype=bool)
    entry[:: 97 * params["a"]] = True
    flat = np.zeros(n, dtype=bool)
    ticks = np.full(n, 8, dtype=np.int64)
    return {"entry_long": entry, "entry_short": flat, "exit": flat,
            "stop_ticks": ticks, "target_ticks": ticks}
"""


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _rows() -> list[Row]:
    """390 one-minute bars from 08:30 CT on every weekday of the window, a seeded random walk in ticks."""
    rng = np.random.default_rng(20261116)
    days = [FIRST + timedelta(days=i) for i in range((LAST - FIRST).days + 1)]
    rows: list[Row] = []
    price = BASE_TICKS
    for day in (d for d in days if d.weekday() < 5):
        start = ct_ns(day, SESSION_OPEN)
        for k in range(BARS_PER_DAY):
            price += int(rng.integers(-3, 4))
            rows.append(Row(start + k * 60_000_000_000, day.isoformat(), price))
    return rows


@dataclass(frozen=True)
class SyntheticStore:
    """The synthetic store and the gateway factory the engine is pointed at."""

    stores: SynthStores
    engine_reads: list[int] = field(default_factory=list)  # one entry per gateway the engine built

    def engine_gateway(self, *args: object, **kwargs: object) -> Any:
        self.engine_reads.append(1)
        return self.gateway()

    def gateway(self) -> Any:
        from pfb.data.gateway import DataGateway
        from pfb.instruments.table import load_instruments

        table = load_instruments()
        return DataGateway.for_synthetic_stores(self.stores.paths, self.stores.windows, self.stores.catalog(),
                                                table.tick_fixed_of, table.price_factor_of)  # fmt: skip


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SyntheticStore:
    stores = SynthStores(tmp_path / "stores")
    write_store(stores.paths.research / ROOT / research_name(ROOT), _rows(), MNQ_TICK_FIXED, MNQ_TICK_FIXED)
    test_store = SyntheticStore(stores)
    import pfb.engine.run as engine_run

    monkeypatch.setattr(engine_run, "DataGateway", test_store.engine_gateway)
    for name in ("load_paths", "load_windows", "load_catalog"):  # the real config is not read for bars
        monkeypatch.setattr(engine_run, name, lambda: None)
    return test_store


def _deps(env: Env, store: SyntheticStore) -> Any:
    """The real runner deps with the data side (files_for, provenance, store roots) on the synthetic store."""
    from pfb.runner.deps import default_deps, provenance_pairs, store_relative

    roots = store.stores.paths.store_roots()

    def data_files(root: str, first: date, last: date) -> tuple[tuple[str, str], ...]:
        entries = store.gateway().files_for(root, first, last)
        return tuple((store_relative(e.path, roots), e.sha256) for e in entries)

    toml = temp_paths_toml(env.repo.parent / "paths")  # config/paths.toml is not needed
    return replace(default_deps(toml), registry_config=lambda: env.config, data_files=data_files,
                   provenance_files=lambda result: provenance_pairs(result.provenance, roots),
                   store_roots=lambda: roots)  # fmt: skip


@pytest.fixture
def registered(
    genesis_env: Env, store: SyntheticStore, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Env, SyntheticStore]]:
    """The strategy and a test hypothesis (budget 1, a = [1], fixed:1 costs) committed and registered in the
    temporary ledger, which is checkpointed; pfb.app pointed at that ledger and the repository's runs/."""
    from pfb import app
    from pfb.runner.register import register

    env = genesis_env
    without_repo_paths(monkeypatch, env.repo.parent)
    env.write(STRATEGY_REL, STRATEGY)
    env.write(HYPOTHESIS_REL, env.hypothesis_text(budget=1, grid="a = [1]", cost__model='"fixed:1"',
                                                  windows=WINDOWS_TOML))  # fmt: skip
    env.commit()
    deps = _deps(env, store)
    register(env.repo / HYPOTHESIS_REL, deps)
    env.checkpoint()
    with app.using(deps=deps, runs_dir=env.repo / "runs", ledger=None):
        yield env, store


def test_a_trial_through_the_cli_with_the_real_engine_counts_once_in_the_temporary_ledger(
    registered: tuple[Env, SyntheticStore],
) -> None:
    # Arrange
    from pfb.cli.main import app as cli_app
    from pfb.contract import validate

    env, store = registered
    real_before = _sha256(REAL_LEDGER)
    n_before = n_of(env)
    assert kinds(env) == ["genesis", "hypothesis"] and clean(env)
    args = ["run", str(env.repo / STRATEGY_REL), "--kind", "trial", "--hypothesis",
            str(env.repo / HYPOTHESIS_REL), "--json"]  # fmt: skip
    # Act
    res = CliRunner().invoke(cli_app, args)
    # Assert: the contract
    assert res.exit_code == 0, res.output
    doc = json.loads(res.stdout)
    validate(doc)
    assert (doc["kind"], doc["status"], doc["purpose"], doc["n_added"]) == ("trial", "ok", "TEST", 1)
    assert doc["window"] == [FIRST.isoformat(), LAST.isoformat()] and doc["params"] == {"a": 1}
    assert doc["trades"] > 0 and doc["walk_forward"]["folds"] == 5 and doc["baseline"]["draws"] == 200
    assert set(doc["cost_stress"]) == {"1", "2"} and doc["power"]["min_edge_cents"] is not None
    assert not [w for w in doc["warnings"] if w.startswith("suite: ")]
    # the temporary ledger: one begin and one result, N up by exactly 1, checkpointed
    assert kinds(env) == ["genesis", "hypothesis", "trial", "result"]
    assert n_of(env) == n_before + 1 == doc["n"]
    from pfb.registry.chain import read_ledger

    begin, result = (line.record for line in read_ledger(env.config.registry_path)[-2:])
    assert begin["run_id"] == result["run_id"] == doc["run_id"] and result["status"] == "ok"
    assert begin["n_after"] == begin["n_before"] + 1 and begin["data_files"][0]["path"].startswith(
        "research/"
    )
    assert clean(env)  # the ledger is committed at the end
    # the bars came through the test gateway (one engine read), the files named before the run
    assert store.engine_reads == [1]
    summary = json.loads((env.repo / "runs" / doc["run_id"] / "summary.json").read_text())
    assert summary["n_trades"] == doc["trades"]
    # this repository's ledger is untouched
    assert _sha256(REAL_LEDGER) == real_before
