"""A book through the runner and the CLI (fake engine and gateway; temporary ledger and git
repository): `pfb hypothesis register` of a book and its --dry-run, each member run as its own trial, and the
refusals the runner makes before anything is logged."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from book_support import (  # pyright: ignore[reportMissingImports]
    BOOK_ID,
    BOOK_REL,
    LATE_WINDOW,
    book_text,
    ledger_bytes,
    member_toml,
    write_book,
)
from registry_env import STRATEGY_REL, Env  # pyright: ignore[reportMissingImports]
from trial_support import FakeTrialEngine, kinds, n_of  # pyright: ignore[reportMissingImports]
from typer.testing import CliRunner

from pfb.data.errors import ClosedWindowRefused
from pfb.registry.api import read_state
from pfb.registry.errors import NotABookMember, SuiteMismatch
from pfb.runner.api import run
from pfb.runner.neighbours import run_with_neighbours
from pfb.runner.register import dry_run, register


def _register(env: Env, engine: FakeTrialEngine, text: str | None = None) -> None:
    path = write_book(env, text if text is not None else book_text(env.strategy_sha))
    register(path, engine.deps(env))
    env.checkpoint()


def _run(env: Env, engine: FakeTrialEngine, a: int, **kwargs: object):
    return run(env.repo / STRATEGY_REL, kind="trial", hypothesis_path=Path(BOOK_REL), deps=engine.deps(env),
               runs_dir=env.repo / "runs", params={"a": a}, **kwargs)  # type: ignore[arg-type]  # fmt: skip


def test_a_book_registers_through_the_runner_with_n_unchanged(genesis_env: Env) -> None:
    engine = FakeTrialEngine()
    n0 = n_of(genesis_env)
    _register(genesis_env, engine)
    assert kinds(genesis_env) == ["genesis", "hypothesis"] and n_of(genesis_env) == n0
    assert engine.called("data_files") == [("MNQ", *(d for d in _dates()))]  # one (root, window) pair


def _dates():
    from datetime import date

    return (date(2025, 4, 1), date(2025, 4, 30))


def test_each_member_runs_as_its_own_trial_and_n_rises_by_k(genesis_env: Env) -> None:
    engine = FakeTrialEngine()
    _register(genesis_env, engine)
    n0 = n_of(genesis_env)
    outcomes = [_run(genesis_env, engine, a) for a in (1, 2)]
    assert [o.begin_record["n_after"] - o.begin_record["n_before"] for o in outcomes] == [1, 1]
    assert n_of(genesis_env) == n0 + 2 and read_state(genesis_env.config).n == n0 + 2
    assert all(o.begin_record["hypothesis_id"] == BOOK_ID for o in outcomes)


def test_a_non_member_is_refused_before_anything_is_logged(genesis_env: Env) -> None:
    engine = FakeTrialEngine()
    _register(genesis_env, engine)
    before = ledger_bytes(genesis_env)
    with pytest.raises(NotABookMember):
        _run(genesis_env, engine, 3)
    assert ledger_bytes(genesis_env) == before and not engine.called("run_strategy")


def test_a_neighbourhood_request_on_a_book_is_refused(genesis_env: Env) -> None:
    engine = FakeTrialEngine()
    _register(genesis_env, engine)
    before = ledger_bytes(genesis_env)
    with pytest.raises(SuiteMismatch, match="neighbourhood"):
        run_with_neighbours(genesis_env.repo / STRATEGY_REL, kind="trial", hypothesis_path=Path(BOOK_REL),
                            deps=engine.deps(genesis_env), runs_dir=genesis_env.repo / "runs",
                            params={"a": 1})  # fmt: skip
    assert ledger_bytes(genesis_env) == before


def test_a_member_window_is_chosen_by_index_over_the_distinct_member_windows(genesis_env: Env) -> None:
    sha = genesis_env.strategy_sha
    rows = [member_toml(sha, a=1), member_toml(sha, a=1, window=LATE_WINDOW)]
    engine = FakeTrialEngine()
    _register(genesis_env, engine, book_text(sha, members=rows, power__trades_per_year="6666.0"))
    late = _run(genesis_env, engine, 1, window=2)
    assert (late.begin_record["first"], late.begin_record["last"]) == LATE_WINDOW
    with pytest.raises(NotABookMember):
        _run(genesis_env, engine, 2, window=1)


def test_the_dry_run_lists_each_closed_member_window_and_appends_nothing(genesis_env: Env) -> None:
    sha = genesis_env.strategy_sha
    rows = [member_toml(sha, a=1), member_toml(sha, a=2, window=LATE_WINDOW)]
    path = write_book(genesis_env, book_text(sha, members=rows, power__trades_per_year="6666.0"))
    engine = FakeTrialEngine(data_error=ClosedWindowRefused("W1", "MNQ is sealed here"))
    before = ledger_bytes(genesis_env)
    found = [type(e).__name__ for e in dry_run(path, engine.deps(genesis_env))]
    assert found == [
        "ClosedWindowTouched",
        "ClosedWindowTouched",
    ]  # one per member pair; registry checks pass
    assert ledger_bytes(genesis_env) == before


def test_the_cli_dry_run_prints_the_refusals_as_json(
    genesis_env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pfb.cli.registry_cmd as registry_cmd
    from pfb.cli.main import app as cli_app

    path = write_book(genesis_env, book_text(genesis_env.strategy_sha, power__trades_per_year="20000.0"))
    engine = FakeTrialEngine()
    monkeypatch.setattr(registry_cmd, "default_deps", lambda *a, **k: engine.deps(genesis_env))
    before = ledger_bytes(genesis_env)
    res = CliRunner().invoke(cli_app, ["hypothesis", "register", str(path), "--dry-run"])
    assert res.exit_code == 2
    doc = json.loads(res.stdout)
    assert doc["ok"] is False and doc["appended"] is False
    assert [r["code"] for r in doc["refusals"]] == ["PowerFlattered"]
    write_book(genesis_env, book_text(genesis_env.strategy_sha))
    res = CliRunner().invoke(cli_app, ["hypothesis", "register", str(path), "--dry-run"])
    assert res.exit_code == 0 and json.loads(res.stdout)["refusals"] == []
    assert ledger_bytes(genesis_env) == before
