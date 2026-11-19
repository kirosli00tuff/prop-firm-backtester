"""The dashboard imports no writer: no registry begin, finish, register, genesis, sync or checkpoint, no
runner, engine, gateway, app service layer or CLI; checked in the source (AST) and in a fresh interpreter.

item 2: src/pfb/dash/actions.py alone may write files (the named calls in ACTIONS_WRITES) and spawn
the pfb CLI in a child (the modules in SPAWN_MODULES); every other dash module keeps the old rules and imports
no spawn module, and view_strategy.py reaches the writer only through pfb.dash.actions."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pfb.dash

DASH_DIR = Path(pfb.dash.__file__).resolve().parent
ALLOWED_REGISTRY_API = {"REPO_ROOT", "LEDGER_REL", "RegistryConfig", "read_state", "status"}
WRITER_NAMES = {
    "begin",
    "finish",
    "register_hypothesis",
    "create_genesis",
    "sync",
    "checkpoint",
    "locked_append",
    "commit_only",
    "commit_paths",
    "plan_configurations",
    "run_with_neighbours",
    "run_strategy",
    "write_parquet",
    "write_text",
    "write_bytes",
    "mkdir",
    "unlink",
    "rmtree",
    "open",
}
FORBIDDEN_MODULES = (
    "pfb.runner",
    "pfb.engine",
    "pfb.data.gateway",
    "pfb.app",
    "pfb.cli",
    "pfb.prop.mc",
    "pfb.prop.matrix",
    "pfb.registry.chain.locked_append",
)
ACTIONS = DASH_DIR / "actions.py"
# uploads/<sha256>.py, the machine lock and the commit button
ACTIONS_WRITES = {"mkdir", "write_bytes", "open", "commit_paths"}
SPAWN_MODULES = ("subprocess", "multiprocessing", "fcntl", "resource", "pty", "signal")
SPAWN_NAMES = {"Popen", "system", "popen", "fork", "forkpty", "killpg", "kill", "execv", "execve", "spawnv"}
VIEW_STRATEGY = DASH_DIR / "view_strategy.py"
VIEW_STRATEGY_PFB_IMPORTS = {"pfb.dash.actions", "pfb.dash.views", "pfb.registry.api"}


def _sources() -> list[Path]:
    return sorted(DASH_DIR.rglob("*.py"))


def _imported(tree: ast.AST) -> list[str]:
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.append(node.module)
        elif isinstance(node, ast.Import):
            out.extend(a.name for a in node.names)
    return out


def _called(tree: ast.AST) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            out.append(
                (node.lineno, func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", ""))
            )
    return out


def test_dash_modules_import_only_the_registry_readers() -> None:
    for path in _sources():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "pfb.registry.api":
                names = {a.name for a in node.names}
                assert names <= ALLOWED_REGISTRY_API, (path.name, names - ALLOWED_REGISTRY_API)
            modules: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            for module in modules:
                assert not module.startswith(FORBIDDEN_MODULES), (path.name, module)


def test_dash_modules_call_no_writer() -> None:
    for path in _sources():
        allowed = ACTIONS_WRITES if path == ACTIONS else set()
        for lineno, name in _called(ast.parse(path.read_text())):
            assert name not in WRITER_NAMES - allowed, (path.name, lineno, name)


def test_only_actions_spawns_a_process() -> None:
    assert ACTIONS.is_file()
    for path in _sources():
        if path == ACTIONS:
            continue
        tree = ast.parse(path.read_text())
        spawning = [m for m in _imported(tree) if m.split(".")[0] in SPAWN_MODULES]
        assert spawning == [], (path.name, spawning)
        calls = [(n, name) for n, name in _called(tree) if name in SPAWN_NAMES]
        assert calls == [], (path.name, calls)


def test_actions_writes_only_through_its_named_calls() -> None:
    tree = ast.parse(ACTIONS.read_text())
    used = {name for _, name in _called(tree)} & WRITER_NAMES
    assert used <= ACTIONS_WRITES, used - ACTIONS_WRITES


def test_view_strategy_imports_no_writer_module() -> None:
    tree = ast.parse(VIEW_STRATEGY.read_text())
    modules = _imported(tree)
    pfb_modules = {m for m in modules if m.startswith("pfb")}
    assert pfb_modules <= VIEW_STRATEGY_PFB_IMPORTS, pfb_modules - VIEW_STRATEGY_PFB_IMPORTS
    assert [m for m in modules if m.split(".")[0] in SPAWN_MODULES] == []
    assert not [m for m in modules if m.startswith(FORBIDDEN_MODULES)]


def test_a_fresh_interpreter_loads_no_writer_module_with_the_dash() -> None:
    # Arrange
    code = (
        "import json, sys, runpy\n"
        "import pfb.dash.data, pfb.dash.views, pfb.dash.view_runs, pfb.dash.view_detail, pfb.dash.view_prop\n"
        "import pfb.dash.actions, pfb.dash.view_strategy\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith('pfb'))))\n"
    )
    # Act
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    loaded = json.loads(done.stdout.strip().splitlines()[-1])
    # Assert
    assert "pfb.dash.data" in loaded
    assert "pfb.dash.view_strategy" in loaded
    bad = [m for m in loaded if m.startswith(FORBIDDEN_MODULES)]
    assert bad == []
