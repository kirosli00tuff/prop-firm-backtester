"""Store roots (config/paths.toml) and the exact-name rule for every file the gateway may open.

A file is accepted only when, after resolving symlinks, it lies under one of the three store roots, no part of
its path is a forbidden directory, its suffix is .parquet and its name matches its era's pattern with <ROOT>
equal to the parent directory. Anything else raises PathRefused and is never opened.

Without config/paths.toml the roots are the synthetic demo store under data/demo/ (built by `pfb demo`); the
cost tables are then the ones config/costs.toml names inside the repository.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from pfb.data.errors import PathRefused, PathsConfigError

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATHS_TOML = REPO_ROOT / "config" / "paths.toml"
DEMO_STORE_ROOT = REPO_ROOT / "data" / "demo"
DEMO_PINS = "pins.json"

FORBIDDEN_PARTS = frozenset({"sealed", "vendor", "derived", "superseded"})
PARQUET_SUFFIX = ".parquet"
HIST_PLANS = ("plan_a", "plan_b")
HIST_LAST = "2018-12-31"
# The made-up store layout of the public copy: the era spans in the file names (the research era is the demo
# store's range; the step 2 era ends on the last open day before the demo table's older embargo).
RESEARCH_FIRST, RESEARCH_LAST = "2025-03-03", "2026-05-15"
STEP2_FIRST, STEP2_LAST = "2019-01-02", "2022-01-31"

ERA_RESEARCH = "research"
ERA_MES_CONFIRMATION = "mes_confirmation"
ERA_STEP2 = "step2"
ERA_HIST = "hist"
ERAS = (ERA_HIST, ERA_STEP2, ERA_MES_CONFIRMATION, ERA_RESEARCH)

_ROOT_RE = r"[A-Z0-9]{1,5}"
_RESEARCH_NAME = re.compile(
    rf"ohlcv-1m_(?P<root>{_ROOT_RE})_v_0_{RESEARCH_FIRST}_{RESEARCH_LAST}_research\.parquet"
)
_MES_CONFIRMATION_NAME = f"ohlcv-1m_MES_v_0_{STEP2_FIRST}_{STEP2_LAST}_confirmation.parquet"
_STEP2_NAME = re.compile(rf"ohlcv-1m_(?P<root>{_ROOT_RE})_v_0_{STEP2_FIRST}_{STEP2_LAST}_step2\.parquet")
_HIST_NAME = re.compile(
    rf"ohlcv-1m_(?P<root>{_ROOT_RE})_v_0_(?P<first>\d{{4}}-\d{{2}}-\d{{2}})_{HIST_LAST}_(?P<plan>plan_[ab])\.parquet"
)


@dataclass(frozen=True)
class StorePaths:
    """The three bar-store roots plus optional overrides other modules read later.

    upstream_root: an optional earlier research checkout whose append-only trial ledger is imported into N
    (pfb.registry.genesis); None when there is none. book_table and release_calendar override the cost
    files config/costs.toml names (None: the repository's own). pins_json: the producer's sha256 per store
    file (pfb.data.catalog.load_pins), or None.
    """

    research: Path
    step2: Path
    hist: Path
    upstream_root: Path | None = None
    book_table: Path | None = None
    release_calendar: Path | None = None
    pins_json: Path | None = None

    def store_roots(self) -> dict[str, Path]:
        return {"research": self.research, "step2": self.step2, "hist": self.hist}


def demo_paths(root: Path = DEMO_STORE_ROOT) -> StorePaths:
    """The synthetic demo store's roots (data/demo/{research,step2,hist}) and its pins file."""
    return StorePaths(
        research=root / "research", step2=root / "step2", hist=root / "hist", pins_json=root / DEMO_PINS
    )


@dataclass(frozen=True)
class StoreFileId:
    """What an accepted path is: its root, era and (hist only) plan, and the resolved path to open."""

    root: str
    era: str
    plan: str | None
    resolved: Path


def _required(table: dict[str, object], section: str, key: str, source: Path) -> Path:
    block = table.get(section)
    if not isinstance(block, dict):
        raise PathsConfigError(f"{source}: missing [{section}] table")
    value = block.get(key)
    if not isinstance(value, str) or not value:
        raise PathsConfigError(f"{source}: [{section}] {key} must be a non-empty string")
    return Path(value)


def _optional(table: dict[str, object], section: str, key: str, source: Path) -> Path | None:
    block = table.get(section)
    if block is None:
        return None
    if not isinstance(block, dict):
        raise PathsConfigError(f"{source}: [{section}] must be a table")
    value = block.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise PathsConfigError(f"{source}: [{section}] {key} must be a non-empty string")
    return Path(value)


def load_paths(path: Path = DEFAULT_PATHS_TOML) -> StorePaths:
    """Read config/paths.toml. The default file, when absent, gives the demo store (demo_paths); any other
    absent, unreadable or malformed file raises PathsConfigError."""
    if not path.is_file():
        if path == DEFAULT_PATHS_TOML and not path.is_symlink():
            return demo_paths()
        raise PathsConfigError(f"{path}: not found (copy config/paths.example.toml and fill it in)")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise PathsConfigError(f"{path}: not UTF-8 text: {exc.reason}") from exc
    except OSError as exc:
        raise PathsConfigError(f"{path}: cannot be read: {exc.strerror}") from exc
    try:
        table = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise PathsConfigError(f"{path}: not valid TOML: {exc}") from exc
    return StorePaths(
        research=_required(table, "stores", "research", path),
        step2=_required(table, "stores", "step2", path),
        hist=_required(table, "stores", "hist", path),
        upstream_root=_optional(table, "upstream", "root", path),
        book_table=_optional(table, "costs", "book_table", path),
        release_calendar=_optional(table, "costs", "release_calendar", path),
        pins_json=_optional(table, "stores", "pins", path),
    )


def _forbidden_part(parts: tuple[str, ...]) -> str | None:
    for part in parts:
        if part in FORBIDDEN_PARTS:
            return part
    return None


def _relative_to_root(resolved: Path, paths: StorePaths) -> tuple[str, tuple[str, ...]]:
    for store, root in paths.store_roots().items():
        root_resolved = root.resolve()
        if resolved.is_relative_to(root_resolved):
            return store, resolved.relative_to(root_resolved).parts
    raise PathRefused(f"{resolved}: not under any store root")


def _classify_research(rel: tuple[str, ...]) -> tuple[str, str, str | None] | None:
    if len(rel) != 2:
        return None
    parent, name = rel
    if parent == "MES" and name == _MES_CONFIRMATION_NAME:
        return "MES", ERA_MES_CONFIRMATION, None
    match = _RESEARCH_NAME.fullmatch(name)
    if match and match["root"] == parent:
        return parent, ERA_RESEARCH, None
    return None


def _classify_step2(rel: tuple[str, ...]) -> tuple[str, str, str | None] | None:
    if len(rel) != 2:
        return None
    parent, name = rel
    match = _STEP2_NAME.fullmatch(name)
    if match and match["root"] == parent:
        return parent, ERA_STEP2, None
    return None


def _valid_iso(text: str) -> bool:
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


def _classify_hist(rel: tuple[str, ...]) -> tuple[str, str, str | None] | None:
    if len(rel) != 3:
        return None
    plan_dir, parent, name = rel
    match = _HIST_NAME.fullmatch(name)
    if not match or plan_dir not in HIST_PLANS:
        return None
    if match["root"] != parent or match["plan"] != plan_dir or not _valid_iso(match["first"]):
        return None
    return parent, ERA_HIST, plan_dir


_CLASSIFIERS = {"research": _classify_research, "step2": _classify_step2, "hist": _classify_hist}


def check_path(path: Path, paths: StorePaths) -> StoreFileId:
    """Accept a store file or raise PathRefused. The file is not opened."""
    given = Path(path)
    if not given.is_absolute():
        raise PathRefused(f"{given}: store paths must be absolute")
    bad = _forbidden_part(given.parts)
    if bad is not None:
        raise PathRefused(f"{given}: path part {bad!r} is forbidden")
    resolved = given.resolve()
    bad = _forbidden_part(resolved.parts)
    if bad is not None:
        raise PathRefused(f"{given}: resolves into forbidden part {bad!r} ({resolved})")
    if resolved.suffix != PARQUET_SUFFIX:
        raise PathRefused(f"{given}: suffix {resolved.suffix!r} is not {PARQUET_SUFFIX}")
    store, rel = _relative_to_root(resolved, paths)
    found = _CLASSIFIERS[store](rel)
    if found is None:
        raise PathRefused(f"{given}: name does not match the {store} store pattern ({'/'.join(rel)})")
    root, era, plan = found
    return StoreFileId(root=root, era=era, plan=plan, resolved=resolved)
