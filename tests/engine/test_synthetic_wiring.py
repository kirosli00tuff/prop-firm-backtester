"""Synthetic bars and the run_strategy wiring (instruments, sessions, costs, fees, metrics). No real bar."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pfb.data.gateway import OUTPUT_COLUMNS
from pfb.engine import ENGINE_VERSION
from pfb.engine.lookahead import truncation_check
from pfb.engine.run import EVENT_WARNING, STANDING_WARNING, run_strategy
from pfb.engine.strategy import load_strategy, resolve_params
from pfb.engine.synthetic import SYNTHETIC_ERA, frame_from_intervals, synthetic_bars, synthetic_frame
from pfb.metrics.core import TRADE_COLUMNS

REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / "strategies" / "reference_breakout.py"
NS_MIN = 60_000_000_000
APRIL = (date(2025, 4, 1), date(2025, 4, 30))


def _intervals() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.array([0, 1000 * NS_MIN], dtype=np.int64)
    return starts, starts + 300 * NS_MIN, np.array([1, 2], dtype=np.int32)


def test_frame_from_intervals_is_gateway_shaped_seeded_and_one_era() -> None:
    a = frame_from_intervals("MNQ", *_intervals(), seed=9)
    b = frame_from_intervals("MNQ", *_intervals(), seed=9)
    c = frame_from_intervals("MNQ", *_intervals(), seed=10)
    assert a.columns == list(OUTPUT_COLUMNS)
    assert a.equals(b)
    assert not a.equals(c)
    assert a["era"].unique().to_list() == [SYNTHETIC_ERA]
    assert bool((a["ts_event"].diff().drop_nulls() > 0).all())
    assert a.select((pl.col("low") <= pl.min_horizontal("open", "close")).all()).item()
    assert a.select((pl.col("high") >= pl.max_horizontal("open", "close")).all()).item()
    assert 580 <= a.height < 600  # about 0.5 % of minutes are absent


def test_synthetic_frame_follows_the_calendar() -> None:
    frame = synthetic_frame("MNQ", *APRIL, seed=1)
    days = frame["trade_date"].unique().sort().to_list()
    assert days[0] == date(2025, 4, 1) and days[-1] == date(2025, 4, 30)
    assert all(d.weekday() < 5 for d in days)


def test_reference_is_truncation_invariant_on_synthetic_bars() -> None:
    strategy = load_strategy(REFERENCE)
    truncation_check(
        strategy.generate, synthetic_bars("MNQ", seed=5), resolve_params(strategy.spec, None), seed=5
    )


def test_synthetic_run_end_to_end_with_a_fixed_cost_model() -> None:
    # Act
    result = run_strategy(REFERENCE, source="synthetic", cost_model="fixed:1", seed=3, n_trials=1000)
    # Assert
    assert tuple(result.trades.columns) == TRADE_COLUMNS
    assert result.trades.height > 0
    assert int(result.trades["net_cents"].sum()) == result.equity_minute["eq_close"][-1]
    assert result.summary["event_windows"] == "off"
    # the run's first date has no prior settlement in
    # its own bars, but no position was open there (entries refused), so the standing warning drops and only
    # that count stays, then the event warning
    warnings = result.summary["warnings"]
    assert isinstance(warnings, list) and len(warnings) == 2 and STANDING_WARNING not in warnings
    assert warnings[0].startswith("price-limit rule: entries refused on 1 of ")
    assert warnings[1] == EVENT_WARNING
    assert result.summary["price_limits"]["kind"] == {"MNQ": "equity_bands"}  # type: ignore[index]
    assert result.provenance["engine_version"] == ENGINE_VERSION
    assert result.provenance["files"] == []
    assert result.provenance["cost_model_id"] == "fixed:1"
    json.dumps(result.summary)  # JSON-ready
    assert result.counters["signals_ignored"] >= 0


def test_synthetic_runs_are_reproducible_per_seed() -> None:
    a = run_strategy(REFERENCE, source="synthetic", cost_model="fixed:1", seed=3)
    b = run_strategy(REFERENCE, source="synthetic", cost_model="fixed:1", seed=3)
    assert a.trades.equals(b.trades)
    assert a.summary == b.summary


def test_synthetic_run_on_another_root_with_the_book_model(paths_toml: Path) -> None:
    result = run_strategy(REFERENCE, source="synthetic", root="NQ", seed=2)
    assert result.provenance["root"] == "NQ"
    assert result.provenance["cost_model_id"] == "book"
    assert result.summary["net_cents"] == int(result.trades["net_cents"].sum())


def test_a_root_without_a_fee_row_is_refused_not_defaulted() -> None:
    from pfb.costs.fees import load_fees
    from pfb.instruments.table import load_instruments

    listed = set(load_fees("topstep").roots())
    unlisted = sorted(set(load_instruments().roots()) - listed)
    if not unlisted:
        pytest.skip("every root has a topstep fee row")
    with pytest.raises(LookupError):
        run_strategy(REFERENCE, source="synthetic", root=unlisted[0], cost_model="fixed:1", seed=1)
