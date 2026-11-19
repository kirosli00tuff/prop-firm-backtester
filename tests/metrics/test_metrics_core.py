"""pfb.metrics.core.summarize: one hand-computed fixture per metric (synthetic frames built in code)."""

import json
import math
from datetime import date

import polars as pl
import pytest

from pfb.metrics import core
from pfb.metrics import stats as S

MIN = 60_000_000_000

# Three trades over four trade dates (the second date has no trade).
#   T1 long 1 @100 -> 110: +10 ticks, gross 5000, costs 100 + 50, net 4850, hold 10 min, mae 2, mfe 12
#   T2 short 2 @200 -> 205: -10 ticks, gross -2500, costs 200 + 100, net -2800, hold 20 min, mae 6, mfe 1
#   T3 short 1 @50 -> 51: -1 tick, gross -500, costs 100 + 0, net -600, hold 30 min, mae 3, mfe 4
DAILY_NET = [4850, 0, -2800, -600]
EQ_CLOSE = [-150, 4850, 4850, 3500, 2050, 1900, 1450]
EQ_LOW = [-350, 4600, 4850, 3000, 1900, 1700, 1400]
EQ_HIGH = [200, 5100, 4850, 4900, 3600, 2050, 1950]
POSITION = [1, 0, 0, -2, 0, -1, 0]


def _trades() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trade_date": [date(2025, 4, 1), date(2025, 4, 3), date(2025, 4, 4)],
            "side": [1, -1, -1],
            "contracts": [1, 2, 1],
            "entry_ts": [0, 100 * MIN, 200 * MIN],
            "exit_ts": [10 * MIN, 120 * MIN, 230 * MIN],
            "entry_price": [100, 200, 50],
            "exit_price": [110, 205, 51],
            "gross_cents": [5000, -2500, -500],
            "commission_cents": [100, 200, 100],
            "slippage_cents": [50, 100, 0],
            "net_cents": [4850, -2800, -600],
            "mae_ticks": [2, 6, 3],
            "mfe_ticks": [12, 1, 4],
            "entry_bar": [0, 100, 200],
            "exit_bar": [10, 120, 230],
            "exit_reason": ["TARGET", "STOP", "SESSION_END"],
        },
        schema_overrides={"entry_ts": pl.Int64, "exit_ts": pl.Int64},
    )


def _equity() -> pl.DataFrame:
    return pl.DataFrame({"position": POSITION, "eq_close": EQ_CLOSE, "eq_low": EQ_LOW, "eq_high": EQ_HIGH})


def _daily(nets: list[int] | None = None) -> pl.DataFrame:
    vals = DAILY_NET if nets is None else nets
    return pl.DataFrame({"trade_date": [date(2025, 4, d + 1) for d in range(len(vals))], "net_cents": vals})


def _summary(**kw: object) -> dict[str, object]:
    return core.summarize(_trades(), _equity(), _daily(), n_trials=kw.pop("n_trials", None), **kw)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- P&L block


def test_net_and_cost_totals_are_exact_integer_cents() -> None:
    s = _summary()
    assert (s["net_cents"], s["gross_cents"]) == (1450, 2000)
    assert (s["commission_cents"], s["slippage_cents"]) == (400, 150)
    assert all(type(s[k]) is int for k in ("net_cents", "gross_cents", "commission_cents", "slippage_cents"))


def test_counts() -> None:
    s = _summary()
    assert (s["n_trades"], s["n_wins"], s["n_losses"], s["n_days"], s["n_minutes"]) == (3, 1, 2, 4, 7)


def test_win_rate() -> None:
    assert _summary()["win_rate"] == pytest.approx(1 / 3, abs=1e-15)


def test_profit_factor() -> None:
    assert _summary()["profit_factor"] == pytest.approx(4850 / 3400, abs=1e-15)


def test_expectancy_in_cents_and_gross_ticks() -> None:
    s = _summary()
    assert s["expectancy_cents"] == pytest.approx(1450 / 3, abs=1e-12)
    assert s["expectancy_gross_ticks"] == pytest.approx((10 - 10 - 1) / 3, abs=1e-15)


def test_payoff_ratio_and_average_win_and_loss() -> None:
    s = _summary()
    assert s["avg_win_cents"] == 4850.0 and s["avg_loss_cents"] == -1700.0
    assert s["payoff_ratio"] == pytest.approx(4850 / 1700, abs=1e-15)


def test_mae_and_mfe() -> None:
    s = _summary()
    assert s["mae_ticks_mean"] == pytest.approx(11 / 3, abs=1e-15) and s["mae_ticks_worst"] == 6
    assert s["mfe_ticks_mean"] == pytest.approx(17 / 3, abs=1e-15) and s["mfe_ticks_best"] == 12


# --------------------------------------------------------------------------- daily block


def test_sharpe_daily_and_annual_use_population_sd() -> None:
    # mean 362.5; deviations 4487.5, -362.5, -3162.5, -962.5; pop var = 31196875 / 4
    sd = math.sqrt(31196875 / 4)
    s = _summary()
    assert s["mean_daily_net_cents"] == 362.5
    assert s["sd_daily_net_cents"] == pytest.approx(sd, rel=1e-14)
    assert s["sharpe_daily"] == pytest.approx(362.5 / sd, rel=1e-14)
    assert s["sharpe_annual"] == pytest.approx(362.5 / sd * math.sqrt(252), rel=1e-14)


def test_skew_and_raw_kurtosis_from_population_moments() -> None:
    dev = [4487.5, -362.5, -3162.5, -962.5]
    sd = math.sqrt(31196875 / 4)
    s = _summary()
    assert s["skewness_daily"] == pytest.approx(sum(d**3 for d in dev) / 4 / sd**3, rel=1e-12)
    assert s["kurtosis_daily"] == pytest.approx(sum(d**4 for d in dev) / 4 / sd**4, rel=1e-12)


def test_sortino() -> None:
    downside = math.sqrt((2800**2 + 600**2) / 4)  # zero-trade and positive days count as zero
    assert _summary()["sortino_annual"] == pytest.approx(362.5 / downside * math.sqrt(252), rel=1e-14)


def test_best_day_share() -> None:
    s = _summary()
    assert s["best_day_cents"] == 4850
    assert s["best_day_share"] == pytest.approx(4850 / 1450, abs=1e-15)


def test_annualized_net_and_calmar_on_the_intrabar_drawdown() -> None:
    s = _summary()
    assert s["annualized_net_cents"] == 362.5 * 252
    assert s["calmar"] == pytest.approx(362.5 * 252 / 3700, abs=1e-12)


# --------------------------------------------------------------------------- drawdowns and exposure


def test_max_drawdown_on_eq_close_from_a_zero_start() -> None:
    # peak 4850 (row 1), trough 1450 (row 6)
    assert _summary()["max_drawdown_cents"] == 3400


def test_max_drawdown_intrabar_running_peak_of_high_against_low() -> None:
    # running max eq_high = 200, 5100, ...; 5100 - 1400 at the last row
    assert _summary()["max_drawdown_intrabar_cents"] == 3700


def test_drawdown_counts_the_first_bar_loss_below_the_zero_start() -> None:
    eq = pl.DataFrame({"position": [1, 0], "eq_close": [-300, 0], "eq_low": [-400, -300], "eq_high": [0, 0]})
    s = core.summarize(_trades().head(0), eq, _daily([0]), n_trials=None)
    assert s["max_drawdown_cents"] == 300 and s["max_drawdown_intrabar_cents"] == 400


def test_exposure() -> None:
    s = _summary()
    assert s["trades_per_day"] == 0.75
    assert s["avg_hold_minutes"] == 20.0
    assert s["share_minutes_in_position"] == pytest.approx(3 / 7, abs=1e-15)


# --------------------------------------------------------------------------- PSR, DSR, MinBTL


def _daily_moments() -> dict[str, float | int]:
    return S.moments([float(v) for v in DAILY_NET])


def test_psr_against_zero_is_always_reported() -> None:
    m = _daily_moments()
    expected = S.probabilistic_sharpe_ratio(float(m["sharpe"]), 4, float(m["skew"]), float(m["kurtosis"]))
    s = _summary()
    assert s["psr"] == pytest.approx(expected, abs=1e-15)
    assert s["dsr"] is None and s["minbtl_years"] is None and s["n_trials"] is None


def test_dsr_without_a_grid_uses_the_null_floor() -> None:
    m = _daily_moments()
    sr0 = math.sqrt(1 / 3) * S.bracket(10)
    expected = S.probabilistic_sharpe_ratio(
        float(m["sharpe"]), 4, float(m["skew"]), float(m["kurtosis"]), sr0
    )
    s = _summary(n_trials=10)
    assert s["dsr_variance_floor"] == pytest.approx(1 / 3, abs=1e-15)
    assert s["dsr_variance_grid"] is None and s["dsr_n_grid"] == 0
    assert s["dsr_variance"] == s["dsr_variance_floor"]
    assert s["dsr_sr0_daily"] == pytest.approx(sr0, abs=1e-15)
    assert s["dsr"] == pytest.approx(expected, abs=1e-15) and s["dsr_floor"] == s["dsr"]
    assert s["dsr_bracket"] == S.bracket(10)


def test_dsr_with_a_wide_grid_uses_the_larger_cross_configuration_variance() -> None:
    grid = [0.0, 2.0]  # population variance 1.0 > 1/3
    m = _daily_moments()
    expected = S.probabilistic_sharpe_ratio(
        float(m["sharpe"]), 4, float(m["skew"]), float(m["kurtosis"]), 1.0 * S.bracket(10)
    )
    s = _summary(n_trials=10, grid_daily_sharpes=grid)
    assert s["dsr_variance_grid"] == 1.0 and s["dsr_n_grid"] == 2 and s["dsr_variance"] == 1.0
    assert s["dsr"] == pytest.approx(expected, abs=1e-15)
    assert s["dsr_grid"] == s["dsr"] and s["dsr_floor"] != s["dsr"]


def test_dsr_with_a_narrow_grid_keeps_the_floor_and_reports_both() -> None:
    s = _summary(n_trials=10, grid_daily_sharpes=[0.1, 0.2])
    assert s["dsr_variance_grid"] == pytest.approx(0.0025, abs=1e-15)
    assert s["dsr_variance"] == s["dsr_variance_floor"] and s["dsr"] == s["dsr_floor"]


def test_one_configuration_grid_applies_only_the_floor() -> None:
    s = _summary(n_trials=10, grid_daily_sharpes=[5.0])
    assert s["dsr_variance_grid"] is None and s["dsr_n_grid"] == 1
    assert s["dsr_variance"] == pytest.approx(1 / 3)


def test_min_btl_years() -> None:
    s = _summary(n_trials=1000)
    sharpe = s["sharpe_annual"]
    assert isinstance(sharpe, float)
    assert s["minbtl_years"] == pytest.approx(S.bracket(1000) ** 2 / sharpe**2, rel=1e-14)


def test_e8_min_btl_years_is_none_when_the_annual_sharpe_is_not_above_zero() -> None:
    # Arrange: one losing trade (T3, -600) over three days whose nets sum to it: a negative Sharpe
    trades = _trades().filter(pl.col("net_cents") == -600)
    eq = pl.DataFrame({"position": [0], "eq_close": [-600], "eq_low": [-600], "eq_high": [0]})
    # Act
    s = core.summarize(trades, eq, _daily([100, -700, 0]), n_trials=1000)
    # Assert: the square in bracket^2 / SR^2 would have hidden the sign
    assert isinstance(s["sharpe_annual"], float) and s["sharpe_annual"] < 0
    assert s["minbtl_years"] is None


def test_capital_base_is_carried_and_does_not_enter_the_sharpe() -> None:
    a, b = _summary(), _summary(capital_base_cents=5_000_000)
    assert b["capital_base_cents"] == 5_000_000 and a["capital_base_cents"] is None
    assert a["sharpe_annual"] == b["sharpe_annual"]


# --------------------------------------------------------------------------- zero denominators and errors


def test_zero_denominators_are_null_not_infinite() -> None:
    trades = _trades().filter(pl.col("net_cents") > 0)
    eq = pl.DataFrame(
        {"position": [0, 0], "eq_close": [4850, 4850], "eq_low": [4850, 4850], "eq_high": [4850, 4850]}
    )
    s = core.summarize(trades, eq, _daily([4850, 0]), n_trials=5)
    assert s["profit_factor"] is None and s["payoff_ratio"] is None and s["avg_loss_cents"] is None
    assert s["sortino_annual"] is None
    assert s["calmar"] is None  # max_drawdown_intrabar_cents is 0


def test_flat_daily_series_has_null_sharpe_psr_dsr_minbtl() -> None:
    eq = pl.DataFrame({"position": [0], "eq_close": [0], "eq_low": [0], "eq_high": [0]})
    s = core.summarize(_trades().head(0), eq, _daily([0, 0, 0]), n_trials=7)
    assert s["sharpe_daily"] is None and s["sharpe_annual"] is None
    assert s["psr"] is None and s["dsr"] is None and s["minbtl_years"] is None
    assert s["win_rate"] is None and s["expectancy_cents"] is None and s["avg_hold_minutes"] is None
    assert s["best_day_share"] is None and s["n_trials"] == 7


def test_empty_run_is_all_null_and_json_ready() -> None:
    empty_eq = pl.DataFrame(
        schema={"position": pl.Int64, "eq_close": pl.Int64, "eq_low": pl.Int64, "eq_high": pl.Int64}
    )
    empty_daily = pl.DataFrame(schema={"trade_date": pl.Date, "net_cents": pl.Int64})
    s = core.summarize(_trades().head(0), empty_eq, empty_daily, n_trials=3)
    assert s["net_cents"] == 0 and s["sharpe_annual"] is None and s["max_drawdown_cents"] == 0
    assert s["trades_per_day"] is None and s["share_minutes_in_position"] is None
    json.dumps(s, allow_nan=False)


def test_summary_is_json_ready_with_scalar_values_only() -> None:
    s = _summary(n_trials=10, grid_daily_sharpes=[0.0, 2.0], capital_base_cents=100)
    json.dumps(s, allow_nan=False)
    assert all(v is None or type(v) in (int, float, str, bool) for v in s.values())


def test_degenerate_shape_gives_null_psr_dsr_and_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    # With population moments g4 >= g3^2 + 1, so the PSR variance term is >= (1 - g3 SR / 2)^2 >= 0 and a
    # real series reaches <= 0 only by rounding; the stats function's refusal is injected here.
    def refuse(*_a: object, **_k: object) -> float:
        raise ValueError("degenerate variance term")

    assert _summary()["metrics_warnings"] is None
    monkeypatch.setattr(core.stats, "probabilistic_sharpe_ratio", refuse)
    s = _summary(n_trials=10)
    assert s["psr"] is None and s["dsr"] is None and s["dsr_floor"] is None
    assert "degenerate" in str(s["metrics_warnings"])


def test_missing_column_raises() -> None:
    with pytest.raises(core.MetricsInputError, match="net_cents"):
        core.summarize(_trades().drop("net_cents"), _equity(), _daily(), n_trials=None)


def test_daily_net_not_equal_to_ledger_net_raises() -> None:
    with pytest.raises(core.MetricsInputError, match="daily"):
        core.summarize(_trades(), _equity(), _daily([4850, 0, -2800, -500]), n_trials=None)


def test_last_eq_close_not_equal_to_ledger_net_raises() -> None:
    eq = _equity().with_columns(pl.Series("eq_close", EQ_CLOSE[:-1] + [1451]))
    with pytest.raises(core.MetricsInputError, match="eq_close"):
        core.summarize(_trades(), eq, _daily(), n_trials=None)


def test_null_in_a_money_column_raises() -> None:
    daily = pl.DataFrame(
        {"trade_date": [date(2025, 4, 1)], "net_cents": [None]}, schema_overrides={"net_cents": pl.Int64}
    )
    with pytest.raises(core.MetricsInputError, match="null"):
        core.summarize(_trades().head(0), _equity().head(0), daily, n_trials=None)


def test_float_money_column_raises() -> None:
    daily = _daily().with_columns(pl.col("net_cents").cast(pl.Float64))
    with pytest.raises(core.MetricsInputError, match="integer"):
        core.summarize(_trades(), _equity(), daily, n_trials=None)


@pytest.mark.parametrize("bad", [0, -3, True])
def test_bad_n_trials_raises(bad: object) -> None:
    with pytest.raises(core.MetricsInputError, match="n_trials"):
        core.summarize(_trades(), _equity(), _daily(), n_trials=bad)  # type: ignore[arg-type]


def test_non_finite_grid_sharpe_raises() -> None:
    with pytest.raises(core.MetricsInputError, match="grid"):
        _summary(n_trials=10, grid_daily_sharpes=[0.1, math.nan])


def test_non_positive_capital_base_raises() -> None:
    with pytest.raises(core.MetricsInputError, match="capital_base"):
        _summary(capital_base_cents=0)


# --------------------------------------------------------------------------- ledger labels


def test_trade_columns_hold_the_four_ledger_labels() -> None:
    assert core.TRADE_COLUMNS[-4:] == ("entry_bar", "exit_bar", "exit_reason", "trade_date")
    assert set(_trades().columns) == set(core.TRADE_COLUMNS)


@pytest.mark.parametrize("column", ["entry_bar", "exit_bar", "exit_reason", "trade_date"])
def test_a_missing_ledger_label_raises(column: str) -> None:
    with pytest.raises(core.MetricsInputError, match=column):
        core.summarize(_trades().drop(column), _equity(), _daily(), n_trials=None)


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("exit_reason", ["TARGET", "STOP", "MARGIN_CALL"]),
        ("exit_reason", [1, 2, 3]),
        ("trade_date", ["2025-04-01", "2025-04-03", "2025-04-04"]),
        ("entry_bar", [0.0, 100.0, 200.0]),
    ],
)
def test_a_malformed_ledger_label_raises(column: str, values: list[object]) -> None:
    trades = _trades().with_columns(pl.Series(column, values))
    with pytest.raises(core.MetricsInputError, match=column):
        core.summarize(trades, _equity(), _daily(), n_trials=None)


def test_ledger_labels_do_not_change_any_metric() -> None:
    relabelled = _trades().with_columns(
        pl.Series("exit_reason", ["SIGNAL"] * 3), pl.Series("entry_bar", [5, 6, 7])
    )
    assert core.summarize(relabelled, _equity(), _daily(), n_trials=None) == _summary()
