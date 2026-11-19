"""S7 sizing table: the run's trades replayed at other sizes (an approximation, stated in every output).

Approximation: a trade at q contracts nets floor(net x q / c), c the SPEC contracts, so commission and
slippage scale linearly with size (the book table's per-quantity rounding and any size-dependent impact are
ignored).
Equity is closed-trade equity from 0; the floor trails the highest end-of-day equity (at least 0) by the
plan's drawdown and moves only at day ends (no lock modelled); distance = equity - floor before the trade, and
a trade sized to 0 is not taken; a distance rule takes no trade once the floor is reached (fixed keeps
trading and reports floor_breached). Contracts are whole, floored, capped by the plan's contract limit
(minis x 10 on a micro root). Integer arithmetic decides every size: fractions in basis points, Kelly from
exact sums.

Rules: fixed (the SPEC size); fixed fraction f of the distance over the per-contract risk (the trade's stop
when every trade has one, else the 95th percentile of per-contract MAE); volatility target (target / 20-day
trailing population sd of per-contract daily net, SPEC size until 20 days exist or while that sd is 0;
target = daily loss limit / 3, or drawdown / 10 without one); quarter and half Kelly (k x mean / variance of
per-contract trade net x distance), an upper bound fitted in-sample. Without a plan only fixed and Kelly (as
fractions, not replayed) are reported.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from pfb.analysis.breakdowns import sharpe_annual
from pfb.analysis.spec import SuiteError

REPO_ROOT = Path(__file__).resolve().parents[3]
FIRMS_DIR = REPO_ROOT / "config" / "firms"
BPS = 10_000
CC_PER_CENT = 100
FIXED_FRACTIONS_BPS = (500, 1000, 2000)
KELLY = (("kelly_quarter", 4), ("kelly_half", 2))
VOL_DAYS = 20
DLL_DIVISOR, DRAWDOWN_DIVISOR = 3, 10
MAE_RISK_QUANTILE = 0.95
KELLY_FLAG = "upper bound, in-sample"
APPROXIMATION = "per-contract net = trade net / contracts; costs scale linearly with size"
CONTRACTS_PER_MINI = {"mini": 1, "micro": 10}


@dataclass(frozen=True)
class SizingPlan:
    name: str
    drawdown_cents: int
    daily_loss_cents: int | None
    cap_contracts: int


@dataclass(frozen=True)
class SizingInputs:
    net: np.ndarray  # int64 trade net cents at the SPEC size, ledger order
    day: np.ndarray  # int64 index of each trade's day in the run's days
    n_days: int
    contracts: int  # SPEC contracts
    risk_cc: np.ndarray  # int64 per-contract risk per trade in 1e-4 USD (0: unknown)
    risk_source: str


def load_sizing_plan(name: str, contract_type: str, firms_dir: Path = FIRMS_DIR) -> SizingPlan:
    """'<firm>/<plan>' -> eval drawdown, daily loss limit and contract cap (RuleUnknown is a SuiteError)."""
    from pfb.prop.config import PlanConfigError, load_plan
    from pfb.prop.types import RuleUnknown

    firm, _, plan = name.partition("/")
    path = firms_dir / firm / f"{plan}.toml"
    if not path.is_file():
        raise SuiteError(f"sizing plan {name!r}: no file {path.relative_to(REPO_ROOT)}")
    if contract_type not in CONTRACTS_PER_MINI:
        raise SuiteError(f"sizing plan: contract type {contract_type!r} has no stated mini equivalent")
    try:
        rules = load_plan(path)
    except PlanConfigError as exc:
        raise SuiteError(f"sizing plan {name!r}: {exc}") from exc
    try:
        drawdown = int(rules.value("eval.drawdown_cents"))
        cap = int(rules.value("eval.max_contracts_minis")) * CONTRACTS_PER_MINI[contract_type]
        dll_kind = rules.value("eval.daily_loss_kind")
        dll = rules.value("eval.daily_loss_cents") if dll_kind != "none" else None
    except RuleUnknown as exc:
        raise SuiteError(f"sizing plan {name!r}: {exc}") from exc
    return SizingPlan(name, drawdown, None if dll is None else int(dll), cap)


def risk_per_contract(mae_ticks: np.ndarray, stop_ticks: np.ndarray | None,
                      tick_value_cc: int) -> tuple[np.ndarray, str]:  # fmt: skip
    """Each trade's stop distance when every trade has one, else the 95th percentile of per-contract MAE."""
    n = len(mae_ticks)
    if stop_ticks is not None and n and bool(np.all(stop_ticks > 0)):
        return stop_ticks.astype(np.int64) * tick_value_cc, "stop"
    if n == 0:
        return np.zeros(0, dtype=np.int64), "none"
    p95 = int(np.quantile(np.abs(mae_ticks), MAE_RISK_QUANTILE, method="inverted_cdf"))
    return np.full(n, p95 * tick_value_cc, dtype=np.int64), "mae_p95"


# ------------------------------------------------------------------------------------ replay ----
SizeFn = Callable[[int, int, int], int]  # (trade index, day index, distance cents) -> uncapped contracts


def _replay(inputs: SizingInputs, size: SizeFn, plan: SizingPlan | None, *,
            stops_at_floor: bool = True) -> dict[str, object]:  # fmt: skip
    equity, eod_peak, day_now = 0, 0, -1
    floor = None if plan is None else -plan.drawdown_cents
    daily = np.zeros(inputs.n_days, dtype=np.int64)
    marks, sizes, capped, breached = [0], [], 0, False
    for t in range(len(inputs.net)):
        d = int(inputs.day[t])
        if d != day_now:
            eod_peak = max(eod_peak, equity)  # equity at the end of the last traded day
            floor = None if plan is None else eod_peak - plan.drawdown_cents
            day_now = d
        distance = 0 if floor is None else equity - floor
        breached |= floor is not None and distance <= 0
        live = floor is None or distance > 0 or not stops_at_floor
        q = max(size(t, d, distance), 0) if live else 0
        if plan is not None and q > plan.cap_contracts:
            q, capped = plan.cap_contracts, capped + 1
        net = int(inputs.net[t]) * q // inputs.contracts
        equity += net
        daily[d] += net
        marks.append(equity)
        sizes.append(q)
    peak = np.maximum.accumulate(np.asarray(marks, dtype=np.int64))
    n = len(sizes)
    return {
        "net_cents": int(daily.sum()), "sharpe_annual": sharpe_annual(daily),
        "max_drawdown_cents": int((peak - np.asarray(marks)).max()), "max_contracts": max(sizes, default=0),
        "trades_taken": sum(1 for q in sizes if q > 0), "share_capped": None if plan is None or n == 0
        else capped / n, "floor_breached": None if plan is None else breached,
    }  # fmt: skip


def _kelly_terms(inputs: SizingInputs) -> tuple[int, int]:
    """(numerator, denominator) of mean / variance of per-contract trade net per cent: c n S / (n Q - S^2)."""
    s = int(inputs.net.sum())
    q = sum(int(v) * int(v) for v in inputs.net.tolist())
    n = len(inputs.net)
    return inputs.contracts * n * s, n * q - s * s


def _vol_size(inputs: SizingInputs, target_cents: int) -> SizeFn:
    spec_daily = np.zeros(inputs.n_days, dtype=np.int64)
    np.add.at(spec_daily, inputs.day, inputs.net)
    values = [int(v) for v in spec_daily.tolist()]

    def size(_t: int, d: int, _distance: int) -> int:
        if d < VOL_DAYS:
            return inputs.contracts
        window = values[d - VOL_DAYS : d]
        s, q = sum(window), sum(v * v for v in window)
        var_n2 = VOL_DAYS * q - s * s  # n^2 x population variance of the SPEC-size daily net
        if var_n2 <= 0:
            return inputs.contracts
        # q* = floor(target x c / sd) with sd = sqrt(var_n2) / n, exactly in integers
        return math.isqrt((target_cents * inputs.contracts * VOL_DAYS) ** 2 // var_n2)

    return size


def sizing_table(inputs: SizingInputs, plan: SizingPlan | None) -> dict[str, object]:
    """Every rule's replay (or, without a plan, fixed and the Kelly fractions)."""
    fixed = _replay(inputs, lambda _t, _d, _x: inputs.contracts, plan, stops_at_floor=False)
    rules: dict[str, object] = {"fixed": fixed}
    num, den = _kelly_terms(inputs)
    f_star = None if den <= 0 else num / den
    kelly: dict[str, object] = {"flag": KELLY_FLAG, "f_star_contracts_per_cent": f_star}
    if plan is not None:
        risk = inputs.risk_cc
        for bps in FIXED_FRACTIONS_BPS:
            if risk.size and bool(np.all(risk > 0)):
                rules[f"fixed_fraction_{bps / BPS:.2f}"] = _replay(
                    inputs, lambda t, _d, x, b=bps: x * CC_PER_CENT * b // (BPS * int(risk[t])), plan
                )
            else:
                rules[f"fixed_fraction_{bps / BPS:.2f}"] = None
        target = (plan.daily_loss_cents // DLL_DIVISOR if plan.daily_loss_cents
                  else plan.drawdown_cents // DRAWDOWN_DIVISOR)  # fmt: skip
        rules["vol_target"] = _replay(inputs, _vol_size(inputs, target), plan) | {"target_cents": target}
    for name, divisor in KELLY:
        if plan is None or den <= 0 or num <= 0:
            kelly[name] = None if f_star is None else {"fraction": f_star / divisor, "replayed": False}
            continue
        result = _replay(inputs, lambda _t, _d, x, k=divisor: num * x // (k * den), plan)
        fraction = f_star / divisor if f_star is not None else None
        kelly[name] = result | {"fraction": fraction, "replayed": True}
    return {
        "approximation": APPROXIMATION, "plan": None if plan is None else plan.name,
        "drawdown_cents": None if plan is None else plan.drawdown_cents,
        "daily_loss_cents": None if plan is None else plan.daily_loss_cents,
        "cap_contracts": None if plan is None else plan.cap_contracts, "risk_source": inputs.risk_source,
        "rules": rules, "kelly": kelly,
    }  # fmt: skip


def sizing_inputs(trades: pl.DataFrame, days: pl.Series, contracts: int, tick_value_cc: int,
                  stop_ticks: np.ndarray | None) -> SizingInputs:  # fmt: skip
    """From the ledger: net, day index, per-contract risk."""
    day_int = days.cast(pl.Int32).to_numpy()
    day = np.searchsorted(day_int, trades["trade_date"].cast(pl.Int32).to_numpy()).astype(np.int64)
    risk, source = risk_per_contract(trades["mae_ticks"].to_numpy().astype(np.int64), stop_ticks,
                                     tick_value_cc)  # fmt: skip
    return SizingInputs(trades["net_cents"].to_numpy().astype(np.int64), day, len(day_int), contracts, risk,
                        source)  # fmt: skip
