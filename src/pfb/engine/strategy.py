"""Strategy interface v0 (ADR-001 section 7, A4, A9): SPEC, generate, the bars view and the signals."""

from __future__ import annotations

import hashlib
import inspect
import json
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from pfb.engine.lint import lint_source

SPEC_KEYS = ("hypothesis_id", "root", "bar_minutes", "session", "contracts", "params")
SESSIONS = ("globex", "rth")
SUPPORTED_BAR_MINUTES = 1
BAR_DTYPES: dict[str, type[np.integer]] = {
    "ts_event": np.int64,
    "open": np.int64,
    "high": np.int64,
    "low": np.int64,
    "close": np.int64,
    "volume": np.int64,
    "trade_day": np.int32,
    "ct_minute": np.int16,
}
SIGNAL_FLAGS = ("entry_long", "entry_short", "exit")
SIGNAL_DISTANCES = ("stop_ticks", "target_ticks")
SIGNAL_KEYS = SIGNAL_FLAGS + SIGNAL_DISTANCES

ParamValue = int | float | str | bool
Bars = Mapping[str, np.ndarray]
GenerateFn = Callable[[Bars, Mapping[str, ParamValue]], object]


class StrategyError(ValueError):
    """A strategy file, its SPEC, its parameters or its signals are refused."""


class LintFailed(StrategyError):
    """The strategy file fails the static lint; it is never executed."""


class SignalError(StrategyError):
    """generate returned something other than the five signal arrays of the bars' length."""


# ------------------------------------------------------------------ bars ----
def _frozen(array: np.ndarray, dtype: type[np.integer]) -> np.ndarray:
    out = np.array(array, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


def bars_from_arrays(arrays: Mapping[str, np.ndarray]) -> Bars:
    """A read-only view over copies of the eight bar arrays (equal length, integer dtypes)."""
    missing = [k for k in BAR_DTYPES if k not in arrays]
    if missing:
        raise StrategyError(f"bars lack {missing}")
    lengths = {len(arrays[k]) for k in BAR_DTYPES}
    if len(lengths) != 1:
        raise StrategyError(f"bar arrays differ in length: {sorted(lengths)}")
    for key in BAR_DTYPES:
        if np.asarray(arrays[key]).dtype.kind not in "iu":
            raise StrategyError(f"bar array {key} is not integer")
    return types.MappingProxyType({k: _frozen(arrays[k], t) for k, t in BAR_DTYPES.items()})


def make_bars(frame: pl.DataFrame) -> Bars:
    """The strategy's view of a gateway-shaped frame: trade_date becomes trade_day (int32 days, A9)."""
    arrays = {k: frame[k].to_numpy() for k in BAR_DTYPES if k != "trade_day"}
    arrays["trade_day"] = frame["trade_date"].cast(pl.Int32).to_numpy()
    return bars_from_arrays(arrays)


def bar_count(bars: Bars) -> int:
    return len(bars["ts_event"])


def slice_bars(bars: Bars, stop: int) -> Bars:
    """The first `stop` bars, as read-only views."""
    return types.MappingProxyType({k: bars[k][:stop] for k in BAR_DTYPES})


def owned_bars(bars: Bars) -> Bars:
    """Read-only copies of the eight bar arrays that own their memory: never a view of the caller's buffer,
    so nothing a call holds on to is shared with another call (E1 c)."""
    return types.MappingProxyType({k: _frozen(bars[k], t) for k, t in BAR_DTYPES.items()})


def frozen_params(params: Mapping[str, ParamValue]) -> Mapping[str, ParamValue]:
    """A fresh read-only mapping of immutable scalars (int, float, str, bool) for one generate call."""
    for name, value in params.items():
        if not isinstance(value, bool | int | float | str):
            kind = type(value).__name__
            raise StrategyError(f"parameter {name} must be an int, float, str or bool, got {kind}")
    return types.MappingProxyType(dict(params))


# ------------------------------------------------------------------ spec ----
@dataclass(frozen=True)
class ParamSpec:
    default: ParamValue
    grid: tuple[ParamValue, ...]


@dataclass(frozen=True)
class StrategySpec:
    hypothesis_id: str
    root: str
    bar_minutes: int
    session: str
    contracts: int
    params: Mapping[str, ParamSpec]

    def __getitem__(self, key: str) -> object:
        """SPEC-dict style access (spec["root"]) for callers written against ADR section 7."""
        if key not in SPEC_KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {k: getattr(self, k) for k in SPEC_KEYS if k != "params"}
        out["params"] = {k: {"default": p.default, "grid": list(p.grid)} for k, p in self.params.items()}
        return out


def _is_param_value(value: object) -> bool:
    return isinstance(value, bool | int | str) or (isinstance(value, float) and np.isfinite(value))


def _same_value(a: ParamValue, b: ParamValue) -> bool:
    return type(a) is type(b) and a == b


def _param_spec(name: object, raw: object) -> ParamSpec:
    if not isinstance(name, str) or not name.isidentifier():
        raise StrategyError(f"parameter name {name!r} is not an identifier")
    if not isinstance(raw, Mapping) or set(raw) != {"default", "grid"}:
        raise StrategyError(f"parameter {name}: needs exactly the keys default and grid")
    default, grid = raw["default"], raw["grid"]
    if not isinstance(grid, list | tuple) or not grid:
        raise StrategyError(f"parameter {name}: grid must be a non-empty list")
    values = (default, *grid)
    if not all(_is_param_value(v) for v in values):
        raise StrategyError(f"parameter {name}: values must be int, finite float, str or bool")
    if any(_same_value(a, b) for i, a in enumerate(grid) for b in grid[i + 1 :]):
        raise StrategyError(f"parameter {name}: grid repeats a value")
    if not any(_same_value(default, g) for g in grid):
        raise StrategyError(f"parameter {name}: default {default!r} is not in the grid")
    return ParamSpec(default=default, grid=tuple(grid))


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise StrategyError(message)


def validate_spec(raw: object) -> StrategySpec:
    """Check a SPEC dict against ADR section 7 and return it frozen."""
    _require(isinstance(raw, Mapping), "SPEC must be a dict")
    assert isinstance(raw, Mapping)
    _require(set(raw) == set(SPEC_KEYS), f"SPEC keys must be exactly {list(SPEC_KEYS)}")
    hyp, root, minutes, session = raw["hypothesis_id"], raw["root"], raw["bar_minutes"], raw["session"]
    contracts, params = raw["contracts"], raw["params"]
    _require(isinstance(hyp, str) and bool(hyp.strip()), "SPEC hypothesis_id must be a non-empty string")
    _require(
        isinstance(root, str) and root.isalnum() and root.isupper(), "SPEC root must be an upper-case root"
    )
    _require(
        type(minutes) is int and minutes == SUPPORTED_BAR_MINUTES, "SPEC bar_minutes must be 1"
    )
    _require(session in SESSIONS, f"SPEC session must be one of {SESSIONS}")
    _require(type(contracts) is int and contracts >= 1, "SPEC contracts must be a whole number >= 1")
    _require(isinstance(params, Mapping), "SPEC params must be a dict")
    assert isinstance(params, Mapping)
    frozen = types.MappingProxyType({str(k): _param_spec(k, v) for k, v in params.items()})
    return StrategySpec(str(hyp), str(root), int(minutes), str(session), int(contracts), frozen)


def resolve_params(spec: StrategySpec, params: Mapping[str, ParamValue] | None) -> dict[str, ParamValue]:
    """SPEC defaults overlaid with `params`; an unknown name or an off-grid value is refused."""
    given = dict(params or {})
    unknown = sorted(set(given) - set(spec.params))
    if unknown:
        raise StrategyError(f"unknown parameter(s) {unknown}")
    out: dict[str, ParamValue] = {}
    for name, pspec in spec.params.items():
        value = given.get(name, pspec.default)
        if not any(_same_value(value, g) for g in pspec.grid):
            raise StrategyError(f"parameter {name}={value!r} is outside the declared grid {list(pspec.grid)}")
        out[name] = value
    return out


def params_json(params: Mapping[str, ParamValue]) -> str:
    """Canonical JSON of resolved parameters: sorted keys, separators ',' and ':', ASCII."""
    return json.dumps(dict(params), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


# -------------------------------------------------------------- strategy ----
@dataclass(frozen=True)
class Strategy:
    spec: StrategySpec
    generate: GenerateFn
    sha256: str
    path: Path


def _execute(code: types.CodeType, module_name: str) -> dict[str, object]:
    """A freshly executed module of the linted source: nothing of an earlier execution is reachable."""
    module = types.ModuleType(module_name)
    exec(code, module.__dict__)  # noqa: S102 - compiled from the bytes that passed the lint
    return module.__dict__


def _generate_of(namespace: Mapping[str, object], name: str) -> Callable[..., object]:
    generate = namespace.get("generate")
    if not callable(generate) or len(inspect.signature(generate).parameters) != 2:
        raise StrategyError(f"{name}: generate(bars, params) is missing")
    return generate


@dataclass(frozen=True)
class IsolatedGenerate:
    """A loaded strategy's generate (c). Every call executes the linted source again in a new module,
    so no module, class, closure or default-argument state survives a call, and hands that module's generate
    read-only copies of the bars (never views of a shared buffer) and a fresh read-only params mapping."""

    code: types.CodeType
    module_name: str
    file_name: str

    def __call__(self, bars: Bars, params: Mapping[str, ParamValue]) -> object:
        generate = _generate_of(_execute(self.code, self.module_name), self.file_name)
        return generate(owned_bars(bars), frozen_params(params))


def load_strategy(path: Path | str) -> Strategy:
    """Lint, then execute the file once (from the same bytes that were hashed) and validate SPEC; generate
    runs in a fresh module of those bytes at every call (IsolatedGenerate)."""
    file_path = Path(path).resolve()
    raw = file_path.read_bytes()
    source = raw.decode("utf-8")
    violations = lint_source(source, filename=str(file_path))
    if violations:
        listed = "; ".join(f"{v.rule} {v.name} (line {v.line})" for v in violations[:5])
        raise LintFailed(f"{file_path.name}: {len(violations)} lint violation(s): {listed}")
    digest = hashlib.sha256(raw).hexdigest()
    code = compile(source, str(file_path), "exec")
    module_name = f"pfb_strategy_{digest[:12]}"
    namespace = _execute(code, module_name)
    spec = validate_spec(namespace.get("SPEC"))
    _generate_of(namespace, file_path.name)
    generate = IsolatedGenerate(code=code, module_name=module_name, file_name=file_path.name)
    return Strategy(spec=spec, generate=generate, sha256=digest, path=file_path)


# --------------------------------------------------------------- signals ----
@dataclass(frozen=True)
class Signals:
    entry_long: np.ndarray
    entry_short: np.ndarray
    exit: np.ndarray
    stop_ticks: np.ndarray
    target_ticks: np.ndarray

    def get(self, key: str) -> np.ndarray:
        if key not in SIGNAL_KEYS:
            raise KeyError(key)
        return getattr(self, key)


def _signal_array(raw: Mapping[str, object], key: str, n: int) -> np.ndarray:
    value = raw[key]
    if not isinstance(value, np.ndarray) or value.ndim != 1 or len(value) != n:
        raise SignalError(f"generate output {key} must be a 1-d numpy array of length {n}")
    if key in SIGNAL_FLAGS:
        if value.dtype != np.bool_:
            raise SignalError(f"generate output {key} must be bool, got {value.dtype}")
        return _frozen(value, np.bool_)  # type: ignore[arg-type]
    if value.dtype.kind not in "iu":
        raise SignalError(f"generate output {key} must be integer ticks, got {value.dtype}")
    if n and int(value.min()) < 0:
        raise SignalError(f"generate output {key} has a negative distance (ADR A4): run refused")
    return _frozen(value, np.int64)


def normalize_signals(raw: object, n: int) -> Signals:
    """Validate generate's dict (exactly the five keys, length n) and return frozen typed arrays."""
    if not isinstance(raw, Mapping) or set(raw) != set(SIGNAL_KEYS):
        raise SignalError(f"generate must return a dict with exactly the keys {list(SIGNAL_KEYS)}")
    arrays = {k: _signal_array(raw, k, n) for k in SIGNAL_KEYS}
    return Signals(**arrays)


__all__ = [
    "BAR_DTYPES",
    "Bars",
    "IsolatedGenerate",
    "LintFailed",
    "ParamSpec",
    "SIGNAL_KEYS",
    "SignalError",
    "Signals",
    "Strategy",
    "StrategyError",
    "StrategySpec",
    "bar_count",
    "bars_from_arrays",
    "frozen_params",
    "load_strategy",
    "make_bars",
    "normalize_signals",
    "owned_bars",
    "params_json",
    "resolve_params",
    "slice_bars",
    "validate_spec",
]
