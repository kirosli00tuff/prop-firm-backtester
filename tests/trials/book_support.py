"""Book (schema "pfb.hypothesis/3") file builders tests: synthetic members only, on the
temporary repository's strategies/s.py (SPEC parameter a, grid 1, 2, 3; contracts 1)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from registry_env import STRATEGY_REL, Env  # pyright: ignore[reportMissingImports]

from pfb.registry.chain import read_ledger

BOOK_REL = "hypotheses/b1.toml"
BOOK_ID = "B1"
WINDOW = ("2025-04-01", "2025-04-30")
LATE_WINDOW = ("2025-04-16", "2025-05-15")  # overlaps WINDOW: union 2025-04-01..2025-05-15 = 45 days
ACCOUNT = "topstep/combine_xfa_standard_50k"


def member_toml(
    sha: str,
    *,
    a: int = 1,
    root: str = "MNQ",
    contracts: int = 1,
    window: Sequence[str] = WINDOW,
    rate: str = "5000.0",
    sd: str = "1000",
    strategy: str = STRATEGY_REL,
) -> str:
    """One [[book.member]] row (raw TOML values for rate and sd)."""
    return (
        f'[[book.member]]\nstrategy = "{strategy}"\nstrategy_sha256 = "{sha}"\nroot = "{root}"\n'
        f"contracts = {contracts}\nparams = {{ a = {a} }}\n"
        f"window = {{ first = {window[0]}, last = {window[1]} }}\n"
        f"trades_per_year = {rate}\ntrade_sd_cents = {sd}\n"
    )


def book_text(
    sha: str, *, members: Sequence[str] | None = None, book_id: str = BOOK_ID, **override: str
) -> str:
    """A book of two members (a = 1 and a = 2, one window, 5000 trades a year each, sd 1000). They share a
    strategy and a root over one window, so they are one tie group: the derived pooled inputs are
    10000 trades a year and sd sqrt(2) * 1000 = 1414.2, and [power] states 10000 and 1415. `override` sets a
    key by name (raw TOML; "table__key" inside a table, "<drop>" removes it)."""
    rows = list(members) if members is not None else [member_toml(sha, a=1), member_toml(sha, a=2)]
    top = {
        "schema": '"pfb.hypothesis/3"',
        "id": f'"{book_id}"',
        "purpose": '"research"',
        "statement": '"synthetic book"',
        "budget": str(len(rows)),
        "no_signal_reason": '""',
    }
    tables = {
        "cost": {
            "model": '"book"',
            "fees": '"topstep"',
            "event_costs": '"off"',
            "event_stance": '"synthetic"',
        },
        "power": {
            "trades_per_year": "10000.0",
            "trade_sd_cents": "1415",
            "trade_skew": "0.0",
            "trade_kurtosis": "3.0",
            "sd_source": '"stated"',
            "target_edge_cents": "1000",
            "member_correlation": "0.0",
            "member_correlation_reason": '"synthetic: one tie group, so rho is not used"',
        },
        "suite": {
            "cost_stress_ticks": "[1, 2]",
            "baseline_draws": "200",
            "baseline_seed": "7",
            "walk_forward_folds": "5",
            "neighbourhood": "false",
            "sizing_plan": '""',
        },
        "secondary": {"prop_plans": "[]", "prop_mc": "20000", "prop_seed": "20261116"},
        "book": {"account": f'"{ACCOUNT}"', "cap": '"plan"', "statistic": '"daily_net"'},
    }
    for key, value in override.items():
        table, _, name = key.rpartition("__")
        target = tables[table] if table else top
        if value == "<drop>":
            target.pop(name)
        else:
            target[name] = value
    text = "".join(f"{k} = {v}\n" for k, v in top.items())
    for name, body in tables.items():
        text += f"[{name}]\n" + "".join(f"{k} = {v}\n" for k, v in body.items())
    return text + "".join(rows)


def write_book(env: Env, text: str, rel: str = BOOK_REL) -> Path:
    path = env.write(rel, text)
    env.commit()
    return path


def ledger_bytes(env: Env) -> bytes:
    return env.config.registry_path.read_bytes()


def records(env: Env) -> list[dict]:
    return [dict(line.record) for line in read_ledger(env.config.registry_path)]
