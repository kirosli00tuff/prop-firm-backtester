"""Strategy interface v0 (ADR-001 section 7, A4, A9): SPEC, parameters, the bars view and the signals."""

from __future__ import annotations

import copy
import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pfb.engine.strategy import (
    SignalError,
    StrategyError,
    bars_from_arrays,
    load_strategy,
    make_bars,
    normalize_signals,
    params_json,
    resolve_params,
    slice_bars,
    validate_spec,
)

GOOD_SPEC = {
    "hypothesis_id": "H-test",
    "root": "MNQ",
    "bar_minutes": 1,
    "session": "rth",
    "contracts": 2,
    "params": {"k": {"default": 3, "grid": [1, 3]}, "mode": {"default": "a", "grid": ["a", "b"]}},
}


def _spec_with(**changes: object) -> dict[str, object]:
    spec = copy.deepcopy(GOOD_SPEC)
    spec.update(changes)
    return spec


def test_a_valid_spec_is_frozen_and_readable_both_ways() -> None:
    spec = validate_spec(GOOD_SPEC)
    assert (spec.root, spec["root"], spec.contracts, spec.session) == ("MNQ", "MNQ", 2, "rth")
    assert spec.as_dict()["params"] == {
        "k": {"default": 3, "grid": [1, 3]},
        "mode": {"default": "a", "grid": ["a", "b"]},
    }


@pytest.mark.parametrize(
    "bad",
    [
        _spec_with(bar_minutes=5),
        _spec_with(session="overnight"),
        _spec_with(contracts=0),
        _spec_with(contracts=True),
        _spec_with(root="mnq"),
        _spec_with(hypothesis_id=" "),
        _spec_with(params={"k": {"default": 2, "grid": [1, 3]}}),
        _spec_with(params={"k": {"default": 1, "grid": []}}),
        _spec_with(params={"k": {"default": 1, "grid": [1, 1]}}),
        _spec_with(params={"k": {"default": 1, "grid": [1], "x": 0}}),
        _spec_with(params={"k": {"default": [1], "grid": [[1]]}}),
        {k: v for k, v in GOOD_SPEC.items() if k != "session"},
        _spec_with(extra=1),
    ],
)
def test_an_invalid_spec_is_refused(bad: dict[str, object]) -> None:
    with pytest.raises(StrategyError):
        validate_spec(bad)


def test_resolve_params_overlays_defaults_and_refuses_off_grid_or_unknown() -> None:
    spec = validate_spec(GOOD_SPEC)
    assert resolve_params(spec, None) == {"k": 3, "mode": "a"}
    assert resolve_params(spec, {"k": 1}) == {"k": 1, "mode": "a"}
    with pytest.raises(StrategyError):
        resolve_params(spec, {"k": 2})
    with pytest.raises(StrategyError):
        resolve_params(spec, {"k": 1.0})  # type-strict: 1.0 is not the grid's 1
    with pytest.raises(StrategyError):
        resolve_params(spec, {"z": 1})


def test_params_json_is_canonical() -> None:
    assert params_json({"mode": "a", "k": 3}) == '{"k":3,"mode":"a"}'


def _arrays(n: int) -> dict[str, np.ndarray]:
    base = np.arange(n, dtype=np.int64)
    return {
        "ts_event": base * 60_000_000_000,
        "open": base,
        "high": base + 1,
        "low": base - 1,
        "close": base,
        "volume": base,
        "trade_day": np.full(n, 20_000),
        "ct_minute": base % 1440,
    }


def test_bars_view_is_read_only_typed_and_sliceable() -> None:
    bars = bars_from_arrays(_arrays(5))
    with pytest.raises(ValueError):
        bars["close"][0] = 7
    with pytest.raises(TypeError):
        bars["close"] = np.zeros(5)  # type: ignore[index]
    assert (bars["trade_day"].dtype, bars["ct_minute"].dtype) == (np.int32, np.int16)
    assert len(slice_bars(bars, 3)["open"]) == 3


def test_make_bars_turns_trade_date_into_trade_day() -> None:
    frame = pl.DataFrame({k: v for k, v in _arrays(2).items() if k != "trade_day"}).with_columns(
        pl.Series("trade_date", ["1970-01-02", "2025-04-07"]).str.to_date()
    )
    assert make_bars(frame)["trade_day"].tolist() == [1, (date(2025, 4, 7) - date(1970, 1, 1)).days]


def _signals(n: int, **override: object) -> dict[str, object]:
    out: dict[str, object] = {k: np.zeros(n, dtype=bool) for k in ("entry_long", "entry_short", "exit")}
    out |= {"stop_ticks": np.zeros(n, dtype=np.int64), "target_ticks": np.zeros(n, dtype=np.int32)}
    return out | override


def test_normalize_signals_accepts_the_five_arrays() -> None:
    sig = normalize_signals(_signals(3), 3)
    assert sig.target_ticks.dtype == np.int64


@pytest.mark.parametrize(
    "raw",
    [
        _signals(3, stop_ticks=np.array([0, -1, 0])),
        _signals(3, exit=np.zeros(3, dtype=np.int64)),
        _signals(3, entry_long=np.zeros(2, dtype=bool)),
        _signals(3, stop_ticks=np.zeros(3)),
        {k: v for k, v in _signals(3).items() if k != "exit"},
        _signals(3, extra=np.zeros(3, dtype=bool)),
        [1, 2, 3],
    ],
)
def test_normalize_signals_refuses_bad_output(raw: object) -> None:
    with pytest.raises(SignalError):
        normalize_signals(raw, 3)


def test_load_strategy_hashes_the_bytes_and_validates(tmp_path: Path) -> None:
    path = tmp_path / "s.py"
    path.write_text(
        "import numpy as np\nSPEC = {'hypothesis_id': 'H', 'root': 'ES', 'bar_minutes': 1,\n"
        "        'session': 'globex',\n"
        "        'contracts': 1, 'params': {}}\n\ndef generate(bars, params):\n    return {}\n",
        encoding="utf-8",
    )
    strategy = load_strategy(path)
    assert strategy.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert strategy.spec.root == "ES"


def test_load_strategy_refuses_a_file_without_generate(tmp_path: Path) -> None:
    path = tmp_path / "s.py"
    path.write_text(
        "SPEC = {'hypothesis_id': 'H', 'root': 'ES', 'bar_minutes': 1, 'session': 'globex', 'contracts': 1,"
        " 'params': {}}\n",
        encoding="utf-8",
    )
    with pytest.raises(StrategyError):
        load_strategy(path)
