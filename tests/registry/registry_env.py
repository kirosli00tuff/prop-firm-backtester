"""Helpers for the registry tests: a temporary git repository, a synthetic upstream ledger and request
builders."""

from __future__ import annotations

import ast
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

from pfb.registry.api import HypothesisFacts, Receipt, RegistryConfig, RunRequest, begin
from pfb.registry.chain import sha256_hex

PE_REG = "ledger/trial_registrations.jsonl"
PE_AMD = "ledger/trial_amendments.jsonl"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
DECLARED_IN = "docs/checks/DECLARED.md"
STRATEGY_REL = "strategies/s.py"
HYPOTHESIS_REL = "hypotheses/h1.toml"
CODE_REL = "src/pfb"
ENGINE_REL = f"{CODE_REL}/engine.py"
DEFAULT_CHECK_PARAMS = {"a": 1}


def pe_line(entry_id: str, test: str | None, ids: list[str], n_before: int | None, n_after: int) -> str:
    record = {
        "schema": "trial_registration/1",
        "entry_id": entry_id,
        "test": test,
        "test_ids": ids,
        "freeze_path": None,
        "freeze_sha256": None,
        "harness_sha256": None,
        "n_before": n_before,
        "n_after": n_after,
        "time_local": "2026-01-01T00:00:00-08:00",
        "source": None,
        "note": "synthetic",
    }
    return json.dumps(record, sort_keys=True)


PE_BASE_LINES = [
    pe_line("baseline", None, [], None, 300),
    pe_line("r001-T1", "T1", ["T1-a", "T1-b"], 300, 302),
]
PE_AMENDMENT = json.dumps(
    {"amendment_id": "a001-T1", "n_before": 302, "n_after": 302, "configurations_added": 0}
)


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True, text=True)
    return done.stdout


class Env:
    """A temporary repository with a registry config and a synthetic upstream checkout."""

    def __init__(self, tmp_path: Path) -> None:
        self.repo = tmp_path / "repo"
        self.pe = tmp_path / "pe"
        self.repo.mkdir()
        (self.pe / "ledger").mkdir(parents=True)
        self.write_pe(PE_BASE_LINES, [PE_AMENDMENT])
        git(self.repo, "init", "-q", "-b", "main")
        git(self.repo, "config", "user.email", "test@example.com")
        git(self.repo, "config", "user.name", "test")
        self.write(".gitignore", "runs/\n")
        self.write(STRATEGY_REL, "x = 1\n")
        self.write(DECLARED_IN, "declared here\n")
        self.write(ENGINE_REL, "v = 1\n")
        self.commit()
        # The code identity (code_sha256) of runs in this repository is its own src/pfb, never the live one.
        self.config = replace(RegistryConfig.for_repo(self.pe, self.repo), code_root=self.repo / CODE_REL)

    def write_pe(self, registrations: list[str], amendments: list[str]) -> None:
        (self.pe / PE_REG).write_text("".join(x + "\n" for x in registrations))
        (self.pe / PE_AMD).write_text("".join(x + "\n" for x in amendments))

    def write(self, rel: str, text: str) -> Path:
        path = self.repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def commit(self) -> None:
        git(self.repo, "add", "-A", "--", ".", ":!ledger/registry.jsonl")
        git(self.repo, "commit", "-q", "-m", "test", "--allow-empty")

    @property
    def strategy_sha(self) -> str:
        return sha256_hex((self.repo / STRATEGY_REL).read_bytes())

    def hypothesis_text(
        self,
        *,
        budget: int = 2,
        hyp_id: str = "H1",
        grid: str = "a = [1, 2, 3]",
        **override: str,
    ) -> str:
        """A schema-2 hypothesis on strategies/s.py; `override` sets a key by name (raw TOML). The purpose is
        "test" at budget 1 and "research" otherwise (a test hypothesis has budget 1)."""
        top = {
            "schema": '"pfb.hypothesis/2"',
            "id": f'"{hyp_id}"',
            "purpose": '"test"' if budget == 1 else '"research"',
            "statement": '"synthetic"',
            "strategy": f'"{STRATEGY_REL}"',
            "strategy_sha256": f'"{self.strategy_sha}"',
            "roots": '["MNQ"]',
            "bar_minutes": "1",
            "session": '"globex"',
            "budget": str(budget),
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
                "trades_per_year": "5000.0",
                "trade_sd_cents": "1000",
                "trade_skew": "0.0",
                "trade_kurtosis": "3.0",
                "sd_source": '"stated"',
                "target_edge_cents": "1000",
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
        }
        windows = override.pop("windows", "[[window]]\nfirst = 2025-04-01\nlast = 2025-04-30\n")
        for key, value in override.items():
            table, _, name = key.rpartition("__")
            target = tables[table] if table else top
            if value == "<drop>":
                target.pop(name)
            else:
                target[name] = value
        text = "".join(f"{k} = {v}\n" for k, v in top.items()) + windows + f"[grid]\n{grid}\n"
        for name, body in tables.items():
            text += f"[{name}]\n" + "".join(f"{k} = {v}\n" for k, v in body.items())
        return text

    def facts(self, **override: Any) -> HypothesisFacts:
        """What pfb.runner would resolve for strategies/s.py (SPEC root MNQ, globex, parameter a)."""
        base = HypothesisFacts(
            spec_root="MNQ",
            spec_bar_minutes=1,
            spec_session="globex",
            spec_params=("a",),
            cost_model_id="book",
            cost_config_sha256=SHA_B,
            fees_sha256=SHA_A,
            closed_windows_sha256=SHA_B,
            spec_grids={"a": (1, 2, 3)},
        )
        return replace(base, **override)

    def checkpoint(self) -> None:
        """Commit the ledger alone, as `pfb registry checkpoint` does (a trial needs it at HEAD)."""
        if not git(self.repo, "status", "--porcelain", "--", "ledger/registry.jsonl"):
            return
        git(self.repo, "add", "--", "ledger/registry.jsonl")
        git(self.repo, "commit", "-q", "--only", "-m", "ledger", "--", "ledger/registry.jsonl")

    def begin_trial(self, **override: Any) -> Receipt:
        """Checkpoint the ledger, then begin a trial of H1 (the order the trial runner follows, C6)."""
        self.checkpoint()
        return begin(self.request("trial", **override), config=self.config)

    def spec_defaults(self) -> dict[str, Any]:
        """The strategy file's SPEC param defaults (read with ast, never executed); {"a": 1} without SPEC."""
        tree = ast.parse((self.repo / STRATEGY_REL).read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "SPEC" for t in node.targets
            ):
                spec = ast.literal_eval(node.value)
                return {name: p["default"] for name, p in spec["params"].items()}
        return dict(DEFAULT_CHECK_PARAMS)

    def engine_checks_text(
        self, *, seed: str = "0", params: dict[str, Any] | None = None, **override: str
    ) -> str:
        """One [[check]] E3; params default to the strategy's SPEC defaults (raw TOML for seed)."""
        chosen = self.spec_defaults() if params is None else params
        inline = ", ".join(f"{k} = {json.dumps(v)}" for k, v in chosen.items())
        entry = {
            "id": "E3",
            "declared_in": DECLARED_IN,
            "strategy": STRATEGY_REL,
            "strategy_sha256": self.strategy_sha,
            "root": "MNQ",
            "first": "2025-04-01",
            "last": "2025-04-30",
            "fees": "topstep",
            "cost_model": "book",
            "session": "globex",
            **override,
        }
        body = "".join(f'{k} = "{v}"\n' for k, v in entry.items())
        return f'schema = "pfb.engine_checks/1"\n[[check]]\n{body}seed = {seed}\nparams = {{ {inline} }}\n'

    def request(self, kind: str, **override: Any) -> RunRequest:
        base = RunRequest(
            kind=kind,
            strategy_path=self.repo / STRATEGY_REL,
            strategy_sha256=self.strategy_sha,
            hypothesis_id="synthetic-check",
            root="MNQ",
            session="globex",
            params={"a": 1},
            seed=0,
            engine_version="0.1.0",
            instruments_sha256=SHA_A,
            cost_model="book",
            cost_model_id="book",
            cost_config_sha256=SHA_B,
            fees="topstep",
            fees_sha256=SHA_A,
            closed_windows_sha256=SHA_B,
            lint_passed=True,
            lookahead_passed=True,
            calendars_sha256=SHA_C,
        )
        if kind == "trial":
            base = replace(
                base,
                hypothesis_id="H1",
                hypothesis_path=self.repo / HYPOTHESIS_REL,
                first="2025-04-01",
                last="2025-04-30",
                data_files=(("research/f.parquet", SHA_A),),
            )
        elif kind == "engine_check":
            base = replace(
                base,
                check_id="E3",
                first="2025-04-01",
                last="2025-04-30",
                data_files=(("research/f.parquet", SHA_A),),
            )
        return replace(base, **override)
