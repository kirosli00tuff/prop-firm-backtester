"""E1: no state survives a generate call.

Each leaky file below remembers the longest close array it has seen and answers a truncated call from that
memory (a module-level cache and four more channels of the same peek). Before the fix the check called
generate on the FULL bars first, so the memory held the future and every cut agreed. After it, every call runs
on a freshly executed module with its own read-only copies, the truncated calls come first and the full call
last. Synthetic bars only."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from pfb.engine import run as engine_run
from pfb.engine.canaries import zero_cost_inputs
from pfb.engine.lint import lint_source, lint_strategy
from pfb.engine.lookahead import CheckResult, LookAheadDetected, lookahead_check
from pfb.engine.run import run_strategy, simulate
from pfb.engine.strategy import LintFailed, Strategy, StrategySpec, load_strategy, resolve_params
from pfb.engine.synthetic import synthetic_bars, synthetic_frame
from pfb.runner.deps import _lookahead

HEAD = '''"""Leaky on purpose (test only): a remembered close array answers truncated calls."""

from __future__ import annotations

import numpy as np

SPEC = {
    "hypothesis_id": "test:state_leak",
    "root": "MNQ",
    "bar_minutes": 1,
    "session": "globex",
    "contracts": 1,
    "params": {"use_cache": {"default": 1, "grid": [1, 0]}},
}

HOLD = 6
EDGE = 6
'''

PEEK = """

def generate(bars, params):
    close = bars["close"]
    n = len(close)
    full = _full_close(close, params)
    idx = np.arange(n)
    ahead = full[np.minimum(idx + HOLD, max(len(full) - 1, 0))] if n else close
    move = ahead - close
    slot = idx % HOLD
    zeros = np.zeros(n, dtype=np.int64)
    return {
        "entry_long": (slot == 0) & (move >= EDGE),
        "entry_short": (slot == 0) & (move <= -EDGE),
        "exit": slot == HOLD - 1,
        "stop_ticks": zeros,
        "target_ticks": zeros.copy(),
    }
"""

# a module-level dict
MODULE_DICT = """
_SEEN = {}


def _full_close(close, params):
    n = len(close)
    if params["use_cache"] == 0 or n == 0:
        return close
    known = "c" in _SEEN and len(_SEEN["c"]) >= n and np.array_equal(_SEEN["c"][:n], close)
    if not known:
        _SEEN["c"] = close.copy()
    return _SEEN["c"]
"""

LONGEST = """
    best = close
    for old in {store}:
        if len(old) > len(best) and np.array_equal(old[: len(close)], close):
            best = old
    {store}.append(close.copy())
    return best
"""

# lint-clean channels: a mutation through a method call is not a store
MODULE_LIST = "\n_SEEN = []\n\n\ndef _full_close(close, params):" + LONGEST.format(store="_SEEN")
CLASS_ATTRIBUTE = "\nclass _Box:\n    T = []\n\n\ndef _full_close(close, params):" + LONGEST.format(
    store="_Box.T"
)
CLOSURE = (
    "\ndef _memory():\n    seen = []\n\n    def full_close(close, params):"
    + LONGEST.format(store="seen").replace("\n    ", "\n        ")
    + "\n    return full_close\n\n\n_full_close = _memory()\n"
)
# refused by the lint (mutable default), and stopped by the fresh module when the lint is bypassed
MUTABLE_DEFAULT = "\ndef _full_close(close, params, seen=[]):" + LONGEST.format(store="seen")

CHANNELS = {
    "module_dict": MODULE_DICT,
    "module_list_append": MODULE_LIST,
    "class_attribute_append": CLASS_ATTRIBUTE,
    "closure_append": CLOSURE,
    "mutable_default": MUTABLE_DEFAULT,
}
LINT_CLEAN = ("module_list_append", "class_attribute_append", "closure_append")
LINT_REFUSED = {"module_dict": "item_store_not_local", "mutable_default": "mutable_default"}

WRITE_INPUT = (
    HEAD
    + """

def generate(bars, params):
    close = bars["close"]
    n = len(close)
    vol = bars["volume"]
    np.maximum(vol, close, out=vol)
    zeros = np.zeros(n, dtype=np.int64)
    flags = np.zeros(n, dtype=np.bool_)
    return {"entry_long": flags, "entry_short": flags.copy(), "exit": flags.copy(), "stop_ticks": zeros,
            "target_ticks": zeros.copy()}
"""
)


def _write(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    return path


def _leaky(tmp_path: Path, channel: str) -> Path:
    return _write(tmp_path, channel, HEAD + CHANNELS[channel] + PEEK)


@pytest.fixture
def no_lint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass the lint so that the isolation alone is tested (the lint is a second, partial barrier)."""
    monkeypatch.setattr("pfb.engine.strategy.lint_source", lambda source, filename="": [])


# ---------------------------------------------------------------- the lint ----
@pytest.mark.parametrize("channel", LINT_CLEAN)
def test_the_method_call_channels_pass_the_lint(tmp_path: Path, channel: str) -> None:
    assert lint_strategy(_leaky(tmp_path, channel)) == []


@pytest.mark.parametrize("channel", sorted(LINT_REFUSED))
def test_leak_cache_and_the_mutable_default_are_refused_by_the_lint_at_step_1_and_in_the_run(
    tmp_path: Path, channel: str
) -> None:
    # Arrange
    path = _leaky(tmp_path, channel)
    # Act / Assert: step 1 lints first; run_strategy loads through the same lint
    assert LINT_REFUSED[channel] in {v.rule for v in lint_strategy(path)}
    with pytest.raises(LintFailed):
        load_strategy(path)
    with pytest.raises(LintFailed):
        run_strategy(path, source="synthetic", cost_model="fixed:1", seed=11)


# ------------------------------------------------------------ the isolation ----
@pytest.mark.parametrize("channel", sorted(CHANNELS))
def test_every_state_channel_is_caught_at_step_1(tmp_path: Path, channel: str, no_lint: None) -> None:
    # Arrange: the lint is bypassed, so only the fresh module and the call order stand in the way
    strategy = load_strategy(_leaky(tmp_path, channel))
    params = resolve_params(strategy.spec, None)
    # Act / Assert: step 1's check (synthetic bars, the K-session sweep included)
    with pytest.raises(LookAheadDetected):
        _lookahead(strategy, params, 3)


@pytest.mark.parametrize("channel", sorted(CHANNELS))
def test_every_state_channel_is_caught_in_the_run(tmp_path: Path, channel: str, no_lint: None) -> None:
    with pytest.raises(LookAheadDetected):
        run_strategy(_leaky(tmp_path, channel), source="synthetic", cost_model="fixed:1", seed=11)


def test_without_its_cache_the_reviewers_strategy_is_caught_too(tmp_path: Path, no_lint: None) -> None:
    strategy = load_strategy(_leaky(tmp_path, "module_dict"))
    with pytest.raises(LookAheadDetected):
        _lookahead(strategy, {"use_cache": 0}, 3)


def test_a_write_into_an_input_array_is_refused(tmp_path: Path) -> None:
    # Arrange: lint-clean; tries to stash the close in the volume array it was handed
    path = _write(tmp_path, "write_input", WRITE_INPUT)
    assert lint_strategy(path) == []
    strategy = load_strategy(path)
    # Act / Assert: every call's arrays are read-only, at step 1 and in the run
    with pytest.raises(ValueError, match="read-only"):
        _lookahead(strategy, resolve_params(strategy.spec, None), 3)
    with pytest.raises(ValueError, match="read-only"):
        run_strategy(path, source="synthetic", cost_model="fixed:1", seed=11)


# ------------------------------------------------------- copies and order ----
class _Recorder:
    """A plain generate that records what each call was handed (honest momentum output)."""

    def __init__(self) -> None:
        self.calls: list[Mapping[str, np.ndarray]] = []
        self.params: list[object] = []
        self.outputs: list[dict[str, np.ndarray]] = []

    def __call__(self, bars: Mapping[str, np.ndarray], params: Mapping[str, object]) -> dict[str, np.ndarray]:
        self.calls.append(bars)
        self.params.append(params)
        close = bars["close"]
        n = len(close)
        prev = np.concatenate((close[:1], close[:-1]))
        zeros = np.zeros(n, dtype=np.int64)
        out = {"entry_long": close > prev + 2, "entry_short": close < prev - 2, "exit": close == prev,
               "stop_ticks": zeros, "target_ticks": zeros.copy()}  # fmt: skip
        self.outputs.append(out)
        return out


def test_truncated_calls_come_first_and_the_full_call_last() -> None:
    # Arrange
    bars = synthetic_bars("MNQ", seed=4)
    n = len(bars["close"])
    recorder = _Recorder()
    # Act
    result = lookahead_check(recorder, bars, {}, 4, sweep_sessions=5)
    # Assert
    sizes = [len(c["close"]) for c in recorder.calls]
    assert sizes[-1] == n and max(sizes[:-1]) == n - 1
    assert result.calls == len(sizes)
    # the random and sweep cuts precede the n-1 call, strictly ascending
    before = sizes[: sizes.index(n - 1)]
    assert len(before) > 1000 and all(a < b for a, b in zip(before, before[1:], strict=False))
    for key, value in recorder.outputs[-1].items():
        assert np.array_equal(result.signals.get(key), value)


def test_each_call_gets_its_own_read_only_copies_and_fresh_params() -> None:
    # Arrange
    bars = synthetic_bars("MNQ", seed=5)
    recorder = _Recorder()
    # Act
    lookahead_check(recorder, bars, {"k": 1}, 5)
    # Assert: no array is a view of the caller's buffers or of another call's
    for call in recorder.calls:
        for key, arr in call.items():
            assert not arr.flags.writeable and arr.base is None
            assert not np.shares_memory(arr, bars[key])
    first, second = recorder.calls[0], recorder.calls[1]
    assert not any(np.shares_memory(first[k], second[k]) for k in first)
    assert recorder.params[0] is not recorder.params[1]
    with pytest.raises(TypeError):
        recorder.params[0]["k"] = 2  # type: ignore[index]


def test_simulate_hands_the_kernel_the_signals_of_the_last_call(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    frame = synthetic_frame("MNQ", date(2025, 4, 1), date(2025, 4, 4), seed=6)
    recorder = _Recorder()
    spec = StrategySpec("test:recorder", "MNQ", 1, "globex", 1, {})
    strategy = Strategy(spec, recorder, "0" * 64, Path("r.py"))
    seen: dict[str, object] = {}
    real_check, real_kernel = engine_run.lookahead_check, engine_run.K.run_kernel

    def spy_check(*args: object, **kwargs: object) -> CheckResult:
        seen["check"] = real_check(*args, **kwargs)  # type: ignore[arg-type]
        return seen["check"]  # type: ignore[return-value]

    def spy_kernel(inputs):  # type: ignore[no-untyped-def]
        seen["kernel"] = inputs
        return real_kernel(inputs)

    monkeypatch.setattr(engine_run, "lookahead_check", spy_check)
    monkeypatch.setattr(engine_run.K, "run_kernel", spy_kernel)
    # Act
    simulate(strategy, frame, {}, zero_cost_inputs("MNQ", frame), seed=6)
    # Assert: no generate call after the check; the kernel reads the check's own arrays
    check, kernel_in = seen["check"], seen["kernel"]
    assert isinstance(check, CheckResult)
    assert len(recorder.calls) == check.calls
    assert kernel_in.entry_long is check.signals.entry_long  # type: ignore[attr-defined]
    assert kernel_in.exit is check.signals.exit  # type: ignore[attr-defined]


def test_the_lint_refuses_the_cache_store_of_leak_cache_by_rule() -> None:
    rules = {v.rule for v in lint_source(HEAD + MODULE_DICT + PEEK)}
    assert rules == {"item_store_not_local"}
