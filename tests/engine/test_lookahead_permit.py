"""The truncation check (ADR-001 section 12) and the run permit (A1)."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from pfb.engine.lookahead import TRUNCATION_CUTS, LookAheadDetected, _cuts, truncation_check
from pfb.engine.permit import PermitRequest, PermitRequired, RunPermit, check_permit
from pfb.engine.run import run_strategy
from pfb.engine.strategy import bars_from_arrays, load_strategy, params_json, resolve_params

REFERENCE = Path(__file__).resolve().parents[2] / "strategies" / "reference_breakout.py"


def _bars(n: int):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(5)
    close = 1000 + np.cumsum(rng.integers(-3, 4, n))
    return bars_from_arrays(
        {
            "ts_event": np.arange(n) * 60_000_000_000,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.ones(n, dtype=np.int64),
            "trade_day": np.full(n, 20_000),
            "ct_minute": np.arange(n) % 1440,
        }
    )


def _out(entry: np.ndarray) -> dict[str, np.ndarray]:
    n = len(entry)
    zeros = np.zeros(n, dtype=np.int64)
    return {
        "entry_long": entry,
        "entry_short": np.zeros(n, dtype=bool),
        "exit": np.zeros(n, dtype=bool),
        "stop_ticks": zeros,
        "target_ticks": zeros,
    }


def test_a_causal_strategy_passes() -> None:
    def momentum(bars, params):  # type: ignore[no-untyped-def]
        close = bars["close"]
        prev = np.concatenate((close[:1], close[:-1]))
        return _out(close > prev)

    truncation_check(momentum, _bars(300), {}, seed=11)


def test_a_strategy_that_reads_the_next_bar_is_detected() -> None:
    def peek(bars, params):  # type: ignore[no-untyped-def]
        close = bars["close"]
        nxt = np.concatenate((close[1:], close[-1:]))
        return _out(nxt > close)

    with pytest.raises(LookAheadDetected):
        truncation_check(peek, _bars(300), {}, seed=11)


def test_a_strategy_using_a_full_sample_statistic_is_detected() -> None:
    def demeaned(bars, params):  # type: ignore[no-untyped-def]
        close = bars["close"]
        return _out(close > close.mean())

    with pytest.raises(LookAheadDetected):
        truncation_check(demeaned, _bars(300), {}, seed=2)


def test_cuts_are_twenty_distinct_seeded_points() -> None:
    cuts = _cuts(500, seed=4)
    assert len(cuts) == TRUNCATION_CUTS == len(set(cuts))
    assert cuts == _cuts(500, seed=4)
    assert all(1 <= k < 500 for k in cuts)
    assert _cuts(1, seed=4) == []


# ------------------------------------------------------------------ permit ----
def _permit(**changes: object) -> RunPermit:
    base = {
        "run_id": "r1",
        "kind": "engine_check",
        "record_sha256": "a" * 64,
        "strategy_sha256": "b" * 64,
        "root": "MNQ",
        "first": date(2025, 4, 1),
        "last": date(2025, 4, 30),
        "params_json": "{}",
        "fees": "topstep",
        "cost_model": "book",
    }
    return RunPermit(**(base | changes))  # type: ignore[arg-type]


def _request(permit: RunPermit) -> PermitRequest:
    return PermitRequest(
        permit.strategy_sha256,
        permit.root,
        permit.first,
        permit.last,
        permit.params_json,
        permit.fees,
        permit.cost_model,
    )


def test_a_matching_permit_passes_and_a_missing_one_raises() -> None:
    permit = _permit()
    assert check_permit(permit, _request(permit)) is permit
    with pytest.raises(PermitRequired):
        check_permit(None, _request(permit))


@pytest.mark.parametrize(
    "field", ["strategy_sha256", "root", "first", "last", "params_json", "fees", "cost_model"]
)
def test_a_permit_differing_from_the_call_in_any_field_raises(field: str) -> None:
    permit = _permit()
    other = {
        "strategy_sha256": "c" * 64,
        "root": "NQ",
        "first": date(2025, 4, 2),
        "last": date(2025, 4, 29),
        "params_json": '{"k":1}',
        "fees": "lucid",
        "cost_model": "fixed:1",
    }[field]
    with pytest.raises(PermitRequired):
        check_permit(permit, replace(_request(permit), **{field: other}))


@pytest.mark.parametrize(
    "changes", [{"kind": "trial_x"}, {"record_sha256": "XYZ"}, {"first": "2025-04-01"}, {"run_id": ""}]
)
def test_a_malformed_permit_is_refused(changes: dict[str, object]) -> None:
    with pytest.raises(PermitRequired):
        _permit(**changes)


def test_run_strategy_on_real_data_needs_a_permit_before_reading_anything() -> None:
    with pytest.raises(PermitRequired):
        run_strategy(REFERENCE, source="real", root="MNQ", first=date(2025, 4, 1), last=date(2025, 4, 30))


def test_run_strategy_refuses_a_permit_for_another_window_or_a_synthetic_permit_on_real_data() -> None:
    strategy = load_strategy(REFERENCE)
    pj = params_json(resolve_params(strategy.spec, None))
    good = _permit(strategy_sha256=strategy.sha256, params_json=pj)
    with pytest.raises(PermitRequired):
        run_strategy(REFERENCE, source="real", first=date(2025, 4, 1), last=date(2025, 4, 29), permit=good)
    with pytest.raises(PermitRequired):
        run_strategy(
            REFERENCE,
            source="real",
            first=date(2025, 4, 1),
            last=date(2025, 4, 30),
            permit=replace(good, kind="synthetic"),
        )
