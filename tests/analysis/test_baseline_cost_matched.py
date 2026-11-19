"""The cost-matched random baseline (uniform placements charged the strategy's own costs).

Synthetic inputs only. (1) The mode charges each placement the commission + slippage the engine
charged the real trade it replaces, in integer cents; (2) it always runs beside the other modes with the same
seed and draw count as uniform, there is no switch, and the reported mode is fixed in code and never easier
than the two-mode rule; (3) zero-edge uniformity over seeds, exact agreement at zero slippage, integer cents.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import fields, replace
from datetime import date
from pathlib import Path

import jsonschema
import numpy as np
import polars as pl
import pytest
from suite_planting import HASH_RANDOM, REFERENCE

import pfb.analysis.inrun as inrun
from pfb import app
from pfb.analysis import baseline as B
from pfb.analysis.spec import SUITE_KEYS, SuiteIntegrityError, SuiteSpec
from pfb.engine.run import prepare_bars, run_strategy

DRAWS = 200
PER_DAY = 100
N_DAYS = 20
CHEAP_UNTIL = 50  # bars 0..49 of a day are cheap, 50..99 dear
DEAR = 1500  # cents per side on dear bars in the cheap/dear tests (a cost gap near one sd of a run's gross)
COMMISSION = 200
TICK_VALUE_CC = 125_000  # USD 12.50 per tick in 1e-4 USD
WINDOW = (date(2025, 4, 1), date(2025, 5, 30))
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "docs" / "result.schema.json"


# ------------------------------------------------------------------------------ toy market ----
def _toy_market(seed: int, *, cheap: int = 0, dear: int = 500) -> B.BaselineMarket:
    """A zero-drift random walk (integer ticks), N_DAYS sessions of PER_DAY bars, slippage by bar of day."""
    rng = np.random.default_rng(seed)
    n = N_DAYS * PER_DAY
    inc = rng.integers(-2, 3, n).astype(np.int64)
    open_ = 10_000 + np.concatenate([[0], np.cumsum(inc)[:-1]]).astype(np.int64)
    j = np.arange(n, dtype=np.int64) % PER_DAY
    slip = np.where(j < CHEAP_UNTIL, cheap, dear).astype(np.int64)
    session_end = j == PER_DAY - 1
    ones = np.ones(n, dtype=np.bool_)
    return B.BaselineMarket(
        open=open_, close=open_ + inc, slip_buy=slip, slip_sell=slip.copy(), ct_minute=600 + j,
        day_idx=np.arange(n, dtype=np.int64) // PER_DAY, entry_ok=B.entry_ok_mask(session_end, ones, ones),
        room=B.segment_room(session_end, np.zeros(n, dtype=np.bool_)), n_days=N_DAYS,
        commission_cents=COMMISSION, tick_value_cc=TICK_VALUE_CC, contracts=1,
    )  # fmt: skip


def _place(rng: np.random.Generator, market: B.BaselineMarket, hold: np.ndarray, where: str) -> np.ndarray:
    """The strategy's own entries, drawn as the uniform mode draws (uniform over eligible bars with room for
    the hold, overlap redrawn): "any" bar, or only bars whose entry and exit are both "cheap" or "dear"."""
    j = np.arange(len(market.open)) % PER_DAY
    occupied = np.zeros(len(market.open), dtype=np.bool_)
    entries = np.empty(len(hold), dtype=np.int64)
    for t, h in enumerate(hold.tolist()):
        ok = market.entry_ok & (market.room >= h)
        if where == "cheap":
            ok &= j + h < CHEAP_UNTIL
        elif where == "dear":
            ok &= j >= CHEAP_UNTIL
        cand = np.flatnonzero(ok)
        e = int(cand[rng.integers(0, len(cand))])
        for _ in range(B.MAX_TRIES - 1):
            if not occupied[e : e + h + 1].any():
                break
            e = int(cand[rng.integers(0, len(cand))])
        occupied[e : e + h + 1] = True
        entries[t] = e
    return entries


def _strategy(
    market: B.BaselineMarket, seed: int, *, n_trades: int = 30, where: str = "any"
) -> tuple[B.BaselineTrades, np.ndarray]:
    """Trades with no edge (entries independent of the walk) and their daily nets as the engine would book
    them; own_cost is what the fills cost at the trade's own bars (2 commissions + entry and exit slip)."""
    rng = np.random.default_rng(10_000 + seed)
    hold = rng.integers(0, 6, n_trades).astype(np.int64)
    side = np.where(rng.integers(0, 2, n_trades) == 0, -1, 1).astype(np.int64)
    entry = _place(rng, market, hold, where)
    x = entry + hold
    slip = np.where(side > 0, market.slip_buy[entry], market.slip_sell[entry]) + np.where(
        side > 0, market.slip_sell[x], market.slip_buy[x]
    )
    trades = B.BaselineTrades(side=side, entry_bar=entry, hold=hold, exit_slip=np.ones(n_trades, np.bool_),
                              own_cost=(2 * COMMISSION + slip).astype(np.int64))  # fmt: skip
    nets = B.placement_nets(market, trades, entry)
    daily = np.zeros(market.n_days, dtype=np.int64)
    np.add.at(daily, market.day_idx[x], nets)
    return trades, daily


def _without_mode(block: object) -> dict[str, object]:
    assert isinstance(block, dict)
    return {k: v for k, v in block.items() if k != "mode"}


def _baseline(market: B.BaselineMarket, trades: B.BaselineTrades, daily: np.ndarray, seed: int) -> dict:
    out, _ = B.random_baseline(market, trades, daily, draws=DRAWS, seed=seed)
    return out


# ------------------------------------------------------------------- known answers ----
ZERO_EDGE_SEEDS = range(400)
KS_TOLERANCE = 0.09  # 1 % two-sided KS critical value at n = 400 (1.63 / 20) plus the 1/200 grid of draws
REJECT_RANGE = (0.02, 0.09)  # share of seeds with p_net <= 0.05; 0.05 +- about 3.5 binomial sd at n = 400


def _ks_uniform(values: list[float]) -> float:
    x = np.sort(np.asarray(values))
    n = len(x)
    return float(max((np.arange(1, n + 1) / n - x).max(), (x - np.arange(n) / n).max()))


def test_a_zero_edge_strategy_has_a_uniform_cost_matched_percentile_over_seeds() -> None:
    percentiles, rejected = [], 0
    for seed in ZERO_EDGE_SEEDS:
        market = _toy_market(seed)
        trades, daily = _strategy(market, seed)
        part = _baseline(market, trades, daily, seed)[B.COST_MATCHED]
        percentiles.append(float(part["net"]["percentile"]))
        rejected += part["net"]["p"] <= 0.05
    assert _ks_uniform(percentiles) < KS_TOLERANCE, _ks_uniform(percentiles)
    share = rejected / len(ZERO_EDGE_SEEDS)
    assert REJECT_RANGE[0] <= share <= REJECT_RANGE[1], share


def test_with_zero_slippage_the_cost_matched_and_uniform_modes_agree_exactly() -> None:
    for seed in range(5):
        market = _toy_market(seed, cheap=0, dear=0)
        trades, daily = _strategy(market, seed)
        assert set(trades.own_cost.tolist()) == {2 * COMMISSION}
        out, frame = B.random_baseline(market, trades, daily, draws=DRAWS, seed=seed)
        assert _without_mode(out["uniform"]) == _without_mode(out[B.COST_MATCHED])
        a = frame.filter(pl.col("mode") == "uniform").drop("mode")
        b = frame.filter(pl.col("mode") == B.COST_MATCHED).drop("mode")
        assert a.equals(b)


def test_with_zero_slippage_through_the_engine_the_two_modes_agree_exactly() -> None:
    suite = SuiteSpec(cost_stress_ticks=(), baseline_draws=DRAWS, baseline_seed=7, walk_forward_folds=0)
    result = run_strategy(HASH_RANDOM, source="synthetic", cost_model="fixed:0", seed=3, first=WINDOW[0],
                          last=WINDOW[1], suite=suite)  # fmt: skip
    assert result.trades.height > 0 and result.trades["slippage_cents"].sum() == 0
    base = result.summary["suite"]["baseline"]  # type: ignore[index]
    assert _without_mode(base["uniform"]) == _without_mode(base[B.COST_MATCHED])


def test_every_value_is_integer_cents() -> None:
    market = _toy_market(1)
    trades, daily = _strategy(market, 1)
    out, frame = B.random_baseline(market, trades, daily, draws=DRAWS, seed=1)
    for mode in B.MODES:
        net = out[mode]["net"]  # type: ignore[index]
        for key in ("strategy", "q05", "q50", "q95"):
            assert type(net[key]) is int, (mode, key)
    assert frame.schema["net_cents"] == pl.Int64
    run = B.run_mode(market, trades, B.COST_MATCHED, DRAWS, 1)
    assert run.nets.dtype == np.int64


@pytest.mark.parametrize(
    ("column", "values", "match"),
    [
        ("slippage_cents", [1.5, 2.0], "integer cents"),
        ("slippage_cents", [None, 2], "integer cents"),
        ("commission_cents", [-1, 2], "negative"),
    ],
)
def test_own_costs_refuse_a_ledger_that_is_not_integer_non_negative_cents(
    column: str, values: list[object], match: str
) -> None:
    ledger = pl.DataFrame({"commission_cents": [4, 4], "slippage_cents": [0, 2]}).with_columns(
        pl.Series(column, values)
    )
    with pytest.raises(SuiteIntegrityError, match=match):
        B.own_costs(ledger)


def test_own_costs_refuse_a_ledger_without_cost_columns() -> None:
    with pytest.raises(SuiteIntegrityError, match="lacks"):
        B.own_costs(pl.DataFrame({"commission_cents": [4]}))


# ---------------------------------------------------------- own costs, as charged ----
def _capture(monkeypatch: pytest.MonkeyPatch) -> list[inrun.RunContext]:
    seen: list[inrun.RunContext] = []
    real = inrun.compute_suite

    def spy(ctx: inrun.RunContext, spec: SuiteSpec):  # type: ignore[no-untyped-def]
        seen.append(ctx)
        return real(ctx, spec)

    monkeypatch.setattr(inrun, "compute_suite", spy)
    return seen


def test_own_cost_is_each_ledger_rows_commission_plus_slippage(monkeypatch: pytest.MonkeyPatch) -> None:
    result = run_strategy(REFERENCE, source="synthetic", cost_model="book", seed=3, first=WINDOW[0],
                          last=WINDOW[1], suite=SuiteSpec.off())  # fmt: skip
    trades = B.trades_of(result.trades)
    want = (result.trades["commission_cents"] + result.trades["slippage_cents"]).to_list()
    assert trades.own_cost.tolist() == want  # per trade as charged, not an average
    assert len(set(want)) > 1  # the book charges by minute: the costs differ across trades


def test_on_flat_prices_every_cost_matched_draw_pays_exactly_the_strategys_costs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _capture(monkeypatch)
    result = run_strategy(REFERENCE, source="synthetic", cost_model="book", seed=3, first=WINDOW[0],
                          last=WINDOW[1], suite=SuiteSpec.off())  # fmt: skip
    market = inrun.baseline_market(seen[0], prepare_bars(seen[0].frame, seen[0].inputs))
    flat = replace(market, open=np.zeros_like(market.open), close=np.zeros_like(market.close))
    trades = B.trades_of(result.trades)
    own_total = int(result.trades["commission_cents"].sum()) + int(result.trades["slippage_cents"].sum())
    cm = B.run_mode(flat, trades, B.COST_MATCHED, DRAWS, 5)
    assert set(cm.nets.tolist()) == {-own_total}
    uni = B.run_mode(flat, trades, "uniform", DRAWS, 5)
    assert len(set(uni.nets.tolist())) > 1  # the uniform mode pays the drawn bars' slippage instead


def test_cost_matched_makes_exactly_the_uniform_placements() -> None:
    """Same seed, same placements: cost_matched = uniform - uniform on flat prices (its costs) - own costs,
    draw by draw, with slippage that differs by bar."""
    market = _toy_market(3)
    trades, _ = _strategy(market, 3)
    flat = replace(market, open=np.zeros_like(market.open), close=np.zeros_like(market.close))
    uni = B.run_mode(market, trades, "uniform", DRAWS, 9)
    uni_cost = B.run_mode(flat, trades, "uniform", DRAWS, 9)
    cm = B.run_mode(market, trades, B.COST_MATCHED, DRAWS, 9)
    assert np.array_equal(cm.nets, uni.nets - uni_cost.nets - int(trades.own_cost.sum()))
    assert np.array_equal(cm.overlaps, uni.overlaps)
    assert not np.array_equal(cm.nets, uni.nets)


# ------------------------------------------------- no choice, never an easier bar ----
def test_cheap_fills_earn_a_zero_edge_strategy_nothing_in_the_cost_matched_mode() -> None:
    uniform_pct, cm_pct, reported_cm_or_tod = [], [], 0
    for seed in range(60):
        market = _toy_market(seed, dear=DEAR)
        trades, daily = _strategy(market, seed, where="cheap")
        out = _baseline(market, trades, daily, seed)
        uniform_pct.append(out["uniform"]["net"]["percentile"])
        cm_pct.append(out[B.COST_MATCHED]["net"]["percentile"])
        reported_cm_or_tod += out["mode"] in (B.COST_MATCHED, "tod")
    assert np.mean(uniform_pct) > 0.75  # the uniform mode credits the cheap fills as an edge
    assert 0.38 < np.mean(cm_pct) < 0.62  # the cost-matched one does not
    assert reported_cm_or_tod == 60  # and the reported figure is never the flattering uniform one


def test_dear_fills_make_the_cost_matched_mode_the_easier_bar_so_it_is_not_reported_then() -> None:
    """The cost-matched mode is not always the harder bar: a strategy paying more than the average bar's cost
    is flattered by it relative to uniform. The reported mode (largest p_net) then stays the harder one."""
    easier = 0
    for seed in range(60):
        market = _toy_market(seed, dear=DEAR)
        trades, daily = _strategy(market, seed, where="dear")
        out = _baseline(market, trades, daily, seed)
        p = {m: out[m]["net"]["p"] for m in B.MODES}
        easier += p[B.COST_MATCHED] < p["uniform"]
        assert out["p_net"] == max(p.values())
    assert easier > 30  # the literal "always cost_matched" rule would have reported an easier bar here


def test_the_reported_p_net_is_never_below_the_two_mode_rule_and_the_old_modes_are_unchanged() -> None:
    for seed in range(40):
        for where in ("any", "cheap", "dear"):
            market = _toy_market(seed)
            trades, daily = _strategy(market, seed, where=where)
            out = _baseline(market, trades, daily, seed)
            two_mode_p = max(out["uniform"]["net"]["p"], out["tod"]["net"]["p"])
            assert out["p_net"] >= two_mode_p
            assert out["p_net"] == out[out["mode"]]["net"]["p"]
            # the uniform and tod blocks are the pre- definitions: seeds seed and seed + 1
            net, sharpe = int(daily.sum()), float(B.sharpe_rows(daily.reshape(1, -1))[0])
            for mode in ("uniform", "tod"):
                run = B.run_mode(market, trades, mode, DRAWS, seed + B.SEED_OFFSET[mode])
                assert out[mode] == B.mode_json(run, net, sharpe)
            assert out[B.COST_MATCHED]["seed"] == out["uniform"]["seed"] == seed
            assert out[B.COST_MATCHED]["draws"] == out["uniform"]["draws"] == out["tod"]["draws"] == DRAWS


def test_there_is_no_switch_for_the_mode_or_the_reported_mode() -> None:
    assert list(inspect.signature(B.random_baseline).parameters) == [
        "market", "trades", "strategy_daily", "draws", "seed"
    ]  # fmt: skip
    assert [k for k in SUITE_KEYS if k.startswith("baseline")] == ["baseline_draws", "baseline_seed"]
    assert {f.name for f in fields(SuiteSpec)} == set(SUITE_KEYS)
    table = {**SuiteSpec().to_json(), "baseline_mode": "uniform"}
    with pytest.raises(ValueError, match="extra"):
        SuiteSpec.from_mapping(table)
    run_params = [p for p in inspect.signature(app.run_backtest).parameters if "baseline" in p]
    assert run_params == ["baseline"]  # on or off for a synthetic run only; no mode


def test_the_schema_accepts_the_cost_matched_mode_and_nothing_else_new() -> None:
    schema = json.loads(SCHEMA_PATH.read_text())
    part = schema["$defs"]["result"]["properties"]["baseline"]["oneOf"][1]
    assert part["properties"]["mode"]["enum"] == ["tod", "uniform", "cost_matched"]
    doc = {"draws": 200, "mode": "cost_matched", "net_percentile": 0.5, "sharpe_percentile": 0.5,
           "p_net": 0.5, "p_sharpe": 0.5}  # fmt: skip
    jsonschema.validate(doc, part, cls=jsonschema.Draft202012Validator)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**doc, "mode": "kinder"}, part, cls=jsonschema.Draft202012Validator)
