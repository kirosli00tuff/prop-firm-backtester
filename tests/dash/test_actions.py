"""pfb.dash.actions, the dashboard's one writer. Uploads, the capped child (memory cap read
back in the child, timeout kills the group, the machine lock gives "busy"), and the check on a passing copy of
the template and on a look-ahead failure, in a temporary repository and ledger (this repository's ledger and
runs/ are never touched)."""

from __future__ import annotations

import hashlib
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
from registry_env import Env  # pyright: ignore[reportMissingImports]

from pfb.dash import actions
from pfb.dash.actions import (
    CHECK_LINT,
    CHECK_SYNTHETIC,
    STATE_BUSY,
    STATE_DONE,
    STATE_TIMEOUT,
    UPLOADS_REL,
    Upload,
    UploadRefused,
    check,
    check_strategy,
    machine_lock,
    read_spec,
    run_capped,
    run_cli,
    save_upload,
)
from pfb.dash.data import DashConfig
from pfb.registry.api import LEDGER_REL, create_genesis, read_state

REPO = Path(__file__).resolve().parents[2]
TEMPLATE_TEXT = (REPO / "strategies" / "_template.py").read_text()
PATHS_TOML = REPO / "config" / "paths.toml"
SMALL_CAP = 3 * actions.GIB
SHORT_TIMEOUT_S = 2.0
SLEEP_S = 60
PEEK = (
    "\n\ndef generate(bars, params):\n"
    "    view = {k: np.concatenate((bars[k][1:], bars[k][-1:])) for k in bars}\n"
    "    return _honest(view, params)\n"
)
LEAKY_TEXT = TEMPLATE_TEXT.replace("def generate(bars, params):", "def _honest(bars, params):", 1) + PEEK



def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


@pytest.fixture
def lock(tmp_path: Path) -> Path:
    return tmp_path / "lock" / "run.lock"


@pytest.fixture(scope="module")
def temp_env(tmp_path_factory: pytest.TempPathFactory) -> tuple[Env, DashConfig]:
    env = Env(tmp_path_factory.mktemp("actions"))
    create_genesis(env.config)
    config = DashConfig(
        repo_root=env.repo, ledger_path=env.config.registry_path, runs_dir=env.repo / "runs", pe_root=env.pe
    )
    return env, config


# ------------------------------------------------------------------------------------ upload ----
def test_bytes_and_text_give_one_file_and_one_sha(tmp_path: Path) -> None:
    # Arrange
    text = TEMPLATE_TEXT
    # Act
    from_text = save_upload(text, tmp_path)
    from_bytes = save_upload(text.encode("utf-8"), tmp_path)
    # Assert
    assert from_text == from_bytes
    assert from_text.sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert from_text.rel_path == f"{UPLOADS_REL}/{from_text.sha256}.py"
    files = sorted(p.name for p in (tmp_path / UPLOADS_REL).iterdir())
    assert files == [f"{from_text.sha256}.py"]
    assert (tmp_path / from_text.rel_path).read_bytes() == text.encode("utf-8")


def test_save_upload_is_idempotent_and_leaves_an_existing_file_alone(tmp_path: Path) -> None:
    # Arrange
    first = save_upload("x = 1\n", tmp_path)
    mtime = (tmp_path / first.rel_path).stat().st_mtime_ns
    # Act
    second = save_upload(b"x = 1\n", tmp_path)
    # Assert
    assert second == first
    assert (tmp_path / first.rel_path).stat().st_mtime_ns == mtime


@pytest.mark.parametrize(
    ("source", "words"),
    [
        (b"", "empty"),
        ("  \n\t\n", "empty"),
        (b"\xff\xfe x = 1", "not UTF-8"),
        (b"#" * (actions.MAX_UPLOAD_BYTES + 1), "limit"),
    ],
)
def test_save_upload_refuses_blank_non_utf8_and_oversized(
    tmp_path: Path, source: bytes | str, words: str
) -> None:
    # Act
    with pytest.raises(UploadRefused, match=words):
        save_upload(source, tmp_path)
    # Assert
    assert not (tmp_path / UPLOADS_REL).exists()


def test_read_spec_reads_the_literal_without_running_the_file() -> None:
    # Act
    spec = read_spec(TEMPLATE_TEXT)
    # Assert
    assert spec is not None
    assert spec["root"] == "MNQ"
    assert spec["params"]["buffer_ticks"] == {"default": 4, "grid": [2, 4, 6]}
    assert read_spec("SPEC = make()\n") is None
    assert read_spec("def (:\n") is None


# ------------------------------------------------------------------------------- the child ----
def test_the_memory_cap_is_set_in_the_child(tmp_path: Path) -> None:
    # Arrange
    code = "import resource; print(resource.getrlimit(resource.RLIMIT_AS))"
    # Act
    done = run_capped([sys.executable, "-c", code], env=os.environ, cwd=tmp_path, timeout_s=30,
                      mem_bytes=SMALL_CAP)  # fmt: skip
    # Assert
    assert done.state == STATE_DONE
    assert done.stdout.strip() == f"({SMALL_CAP}, {SMALL_CAP})"


def test_an_allocation_past_the_cap_fails_in_the_child_only(tmp_path: Path) -> None:
    # Arrange
    code = f"b = bytearray({2 * SMALL_CAP})"
    # Act
    done = run_capped([sys.executable, "-c", code], env=os.environ, cwd=tmp_path, timeout_s=30,
                      mem_bytes=SMALL_CAP)  # fmt: skip
    # Assert
    assert done.returncode != 0
    assert "MemoryError" in done.stderr


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_a_timeout_kills_the_whole_process_group(tmp_path: Path) -> None:
    # Arrange: the child starts a grandchild that would outlive it, writes its pid, then sleeps
    pid_file = tmp_path / "grandchild.pid"
    code = (
        "import subprocess, sys, time\n"
        f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep({SLEEP_S})'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(p.pid))\n"
        f"time.sleep({SLEEP_S})\n"
    )
    # Act
    started = time.monotonic()
    done = run_capped([sys.executable, "-c", code], env=os.environ, cwd=tmp_path, timeout_s=SHORT_TIMEOUT_S,
                      mem_bytes=actions.DEFAULT_MEM_BYTES)  # fmt: skip
    elapsed = time.monotonic() - started
    # Assert
    assert done.state == STATE_TIMEOUT
    assert done.json is None
    assert "timeout" in done.stderr
    assert elapsed < SLEEP_S / 2
    grandchild = int(pid_file.read_text())
    deadline = time.monotonic() + SHORT_TIMEOUT_S
    while _alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(grandchild)


def test_the_lock_is_exclusive_and_never_waits(lock: Path) -> None:
    # Act / Assert
    with machine_lock(lock) as first, machine_lock(lock) as second:
        assert first is True
        assert second is False
    with machine_lock(lock) as again:
        assert again is True


def test_run_cli_is_busy_while_the_lock_is_held(lock: Path, tmp_path: Path) -> None:
    # Arrange
    config = DashConfig(repo_root=tmp_path, ledger_path=tmp_path / "l.jsonl", runs_dir=tmp_path / "runs")
    upload = Upload(rel_path="strategies/uploads/x.py", sha256="0" * 64)
    # Act
    with machine_lock(lock) as held:
        assert held
        started = time.monotonic()
        busy = run_cli(["version"], config=config, lock_path=lock)
        items = check(upload, config=config, lock_path=lock)
        elapsed = time.monotonic() - started
    # Assert
    assert busy.state == STATE_BUSY
    assert busy.returncode is None
    assert [(i.name, i.ok) for i in items] == [(CHECK_LINT, False), (CHECK_SYNTHETIC, False)]
    assert all("another dashboard action" in i.message for i in items)
    assert elapsed < SHORT_TIMEOUT_S


def test_run_cli_with_the_default_config_runs_the_plain_cli(lock: Path) -> None:
    # Arrange: `pfb version` writes nothing
    config = DashConfig.from_env({})
    # Act
    done = run_cli(["version"], config=config, lock_path=lock, timeout_s=120)
    # Assert
    assert done.state == STATE_DONE
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip()


# ------------------------------------------------------------------------------------ check ----
def test_check_passes_a_template_copy_with_n_unchanged(temp_env: tuple[Env, DashConfig], lock: Path) -> None:
    # Arrange
    env, config = temp_env
    repo_ledger = REPO / LEDGER_REL
    repo_ledger_sha = _sha(repo_ledger)
    n_before = read_state(env.config).n
    lines_before = len(config.ledger_path.read_bytes().splitlines())
    upload = save_upload(TEMPLATE_TEXT, config.repo_root)
    # Act
    state = check_strategy(upload, TEMPLATE_TEXT, config=config, lock_path=lock)
    # Assert
    assert state.passed, state.checks
    assert [i.name for i in state.checks] == [CHECK_LINT, CHECK_SYNTHETIC]
    assert state.spec is not None and state.spec["root"] == "MNQ"
    synthetic = state.checks[1].message
    run_id = synthetic.split("run ", 1)[1].split(",", 1)[0]
    assert run_id.startswith("synthetic-")
    assert "(+0)" in synthetic
    assert (config.runs_dir / run_id).is_dir()
    assert read_state(env.config).n == n_before
    # a synthetic run is logged in the ledger it is pointed at (begin and result), N + 0
    assert len(config.ledger_path.read_bytes().splitlines()) > lines_before
    assert _sha(repo_ledger) == repo_ledger_sha


def test_check_fails_a_look_ahead_leak_with_the_exact_reason(
    temp_env: tuple[Env, DashConfig], lock: Path
) -> None:
    # Arrange
    _, config = temp_env
    lines_before = len(config.ledger_path.read_bytes().splitlines())
    upload = save_upload(LEAKY_TEXT, config.repo_root)
    # Act
    items = check(upload, config=config, lock_path=lock)
    # Assert
    lint, synthetic = items
    assert lint.name == CHECK_LINT and not lint.ok
    assert lint.message.startswith("refused: RunRefused: look-ahead check failed: LookAheadDetected")
    assert synthetic.name == CHECK_SYNTHETIC and not synthetic.ok
    assert synthetic.message.startswith("not run")
    assert len(config.ledger_path.read_bytes().splitlines()) == lines_before


def test_the_child_env_carries_the_config_and_the_arena_cap(tmp_path: Path) -> None:
    # Arrange
    config = DashConfig(repo_root=tmp_path, ledger_path=tmp_path / "l.jsonl", runs_dir=tmp_path / "r")
    # Act
    plain = actions._child_env(config, None)  # pyright: ignore[reportPrivateUsage]
    overridden = actions._child_env(config, {"MALLOC_ARENA_MAX": "8"})  # pyright: ignore[reportPrivateUsage]
    # Assert
    assert DashConfig.from_env(plain) == replace(config, pe_root=DashConfig.from_env(plain).pe_root)
    assert plain["MALLOC_ARENA_MAX"] == "2"
    assert overridden["MALLOC_ARENA_MAX"] == "8"


def _strategy_page() -> None:
    from pfb.dash import view_strategy

    view_strategy.render()


def test_view_strategy_shows_sha_spec_and_the_check_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: the check itself is replaced (no child here); the upload is real, under tmp_path
    from streamlit.testing.v1 import AppTest

    from pfb.dash import view_strategy

    monkeypatch.setenv("PFB_DASH_REPO", str(tmp_path))
    items = (
        actions.CheckItem(CHECK_LINT, True, "no warnings"),
        actions.CheckItem(CHECK_SYNTHETIC, False, "x <b>"),
    )

    def fake(upload: Upload, source: bytes, *, config: DashConfig) -> actions.StrategyState:
        return actions.StrategyState(upload.rel_path, upload.sha256, read_spec(source), items, False)

    monkeypatch.setattr(view_strategy, "check_strategy", fake)
    sha = hashlib.sha256(TEMPLATE_TEXT.encode("utf-8")).hexdigest()
    at = AppTest.from_function(_strategy_page, default_timeout=60)
    # Act
    at.run()
    at.text_area(key=view_strategy.TEXT_KEY).input(TEMPLATE_TEXT)
    at.run()
    at.button(key=view_strategy.CHECK_KEY).click()
    at.run()
    # Assert
    assert not at.exception, [e.value for e in at.exception]
    shown = " ".join(m.value for m in at.markdown)
    assert sha in shown
    assert "`root`: MNQ" in shown
    assert "(pass)" in shown and "(fail)" in shown
    assert "x &lt;b&gt;" in shown
    assert (tmp_path / UPLOADS_REL / f"{sha}.py").is_file()
