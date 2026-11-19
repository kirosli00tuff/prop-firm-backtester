"""The fictional cost tables shipped in config/demo/: reproducible from pfb.costs.demo_tables and pinned by
config/costs.toml."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pfb.costs.demo_tables import BOOK_FILE, CALENDAR_FILE, DEMO_DIR, write
from pfb.costs.events import load_cost_model_for, load_events_config, load_pinned_calendar
from pfb.costs.model import load_costs_config
from pfb.data.paths import demo_paths
from pfb.instruments import load_instruments


def test_the_shipped_tables_equal_a_fresh_generation(tmp_path: Path) -> None:
    written = write(tmp_path)

    for name in (BOOK_FILE, CALENDAR_FILE):
        assert written[name].read_bytes() == (DEMO_DIR / name).read_bytes(), name


def test_config_costs_toml_pins_the_shipped_tables() -> None:
    book = hashlib.sha256((DEMO_DIR / BOOK_FILE).read_bytes()).hexdigest()
    calendar = hashlib.sha256((DEMO_DIR / CALENDAR_FILE).read_bytes()).hexdigest()

    assert load_costs_config()["sha256"] == book
    assert load_events_config().sha256 == calendar


def test_the_book_and_event_models_load_from_the_repository_without_a_paths_file() -> None:
    instruments = load_instruments()

    model = load_cost_model_for("book", "on", instruments, demo_paths())

    assert model.model_id == "book+events"
    admissible = {r for r in instruments.roots() if instruments.get(r).admissible}
    assert set(model.roots()) == admissible
    table = model.slippage_cents("MNQ", 1)
    assert table.shape == (1440, 2) and int(table.min()) > 0
    assert not bool(model.calibrated("MNQ")[960:1020].any())  # the hour in no bucket: R-B2 fallback
    config = load_events_config()
    assert load_pinned_calendar().covers(config.coverage_first, config.coverage_last)
