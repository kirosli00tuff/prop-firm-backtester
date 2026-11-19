# The in-run analysis suite (S1-S7)

Assembled results of the first build wave; the binding rules are docs/TRIALS.md. The suite runs inside the engine on the bars the run already loaded (no second read).
Its parts are outputs of the one configuration (C4): a trial's hypothesis fixes them in [suite]; no switch
changes them for a trial; nothing reads them to choose, add or order configurations. A synthetic or
engine_check run chooses them with --no-stress, --no-baseline, --no-walk-forward and --neighbourhood.
Numbers: summary.json "suite" (= suite.json and the result record's metrics), baseline_draws.parquet,
rolling_sharpe.parquet, runs/<centre>/neighbourhood.json, and the contract's headline keys. A part off or
not applicable is null; NaN and infinities are null.

| part | contract key | suite key | switch (non-trial) |
|---|---|---|---|
| S1 cost stress | cost_stress | cost_stress | --no-stress |
| S2 random-entry baseline | baseline | baseline | --no-baseline |
| S3 power | power | power (realized half) | on with any suite |
| S4 walk-forward | walk_forward | walk_forward | --no-walk-forward |
| S5 neighbourhood | neighbourhood | (neighbourhood.json) | --neighbourhood |
| S6 breakdowns | (suite.json) | breakdowns | on with any suite |
| S7 sizing | (suite.json) | sizing | on with any suite |

S1 Cost stress. For each k in cost_stress_ticks the kernel runs again on the same bars and signals with
cost model stress:<run's model>:<k>: the run's slippage plus k ticks per side per fill (rounded up per
fill). Same trades (a different count is an error). Per k: cost_model, net_cents, sharpe_annual, trades,
slippage_cents; each level equals a separate run with that cost model (tested). Contract:
{"k": {net_cents, sharpe}}.

S2 Random-entry baseline. One draw re-places every real trade (side, contracts, hold length in bars kept)
at an entry bar drawn uniformly from the bars where the engine would fill an entry (decision bar not a
session end and inside the entry mask; entry bar and its next minute outside every must-be-flat interval)
whose hold ends by the last bar of its session segment; an overlap is redrawn up to 50 times, then kept
and counted. Fills mirror the kernel: entry at the open with that minute's slippage, exit at the close of
the last held bar with its slippage (none for a real target exit), one commission per fill, gross floored
to cents; a real exit at an open is mirrored by the previous close (at most the open-close gap per
trade). Daily nets over the run's own days; Sharpe = mean / population sd x sqrt(252). Modes: uniform
and tod (keeps each trade's CT minute, so the costs by time of day are the strategy's, and draws the day;
seed + 1; degenerates when trades fill most minute-day cells: its draws then resample the strategy's own
cells). Per mode for net and Sharpe: value, percentile (a fraction in [0, 1]: share below plus half the
ties), one-sided p = (1 + #draws >= value) / (draws + 1), mean, sd, 5/50/95 quantiles. The top-level
figures come from ONE mode, named in `mode`: the one with the larger p_net, tod on a tie (TRIALS.md
section 7: uniform draws pay the slippage of any minute, so a rule trading only in cheaper
hours sat high in the uniform distribution with no edge; the larger of two p-values is conservative).
Both modes stay in suite.json. A pure time-of-day edge is therefore reported in mode tod, where it is not
distinguished from random entry at the same minutes (tested). At least 200 draws; every real trade must
itself be an eligible placement. Contract: draws, mode, net_percentile, sharpe_percentile, p_net,
p_sharpe; a warning when every draw of the reported mode has the same net (draw sd 0).
Known answers (synthetic, tests/analysis/test_suite_baseline.py): a zero-edge time-of-day rule under the
book costs is inside the reported 5-95 % range in 17 of 20 runs (uniform alone: 9 of 20); an edge planted on
even days at fixed minutes is above every draw of both modes.

S3 Power. min_edge(m, sd, years, N, target, skew, kurtosis) per TRIALS.md section 5: the smallest net edge
per trade (cents, rounded up) detected with 80 % power at one-sided 5 % after deflation for N (DSR at the
floor variance); min_edge_cents_alpha without deflation; compound-Poisson trades with stated skew and raw
kurtosis; k >= 1. Known answers: (252, 10000, 1 y, N 1) -> 1615/1615; (220, 21000, 10 y, N 1000) ->
2601/1125; skew -2 kurt 9 -> 2744; skew +2 -> 2591. Monte Carlo power 0.81-0.94, never under 0.80 - 3 SE.
Stated half (trial): checked at registration (N + budget) and begin (n_used); the begin record holds
n_used, min_edge_cents, target_edge_cents; the contract adds alpha at n_used. Realized half (any suite):
the run's trades per year, per-trade sd, skew, kurtosis, days / 252, its N, target = floor mean trade net;
passed = realized edge >= realized minimum (False if undetectable, None under two trades or zero sd): the
realized test, not the begin gate (a trial always passed the gate). Y is one window (TRIALS.md section 5):
the contract's alpha twin uses the trial's own window. The contract warns when the realized minimum exceeds
the stated one, and when a minimum is stated and the realized one is None (with the reason).

S4 Walk-forward report. fold_report(daily, folds >= 2): contiguous blocks of days // folds trade days
(zero-trade days included; remainder to the last). Per fold: dates, days, trades, net_cents, annual
Sharpe. Then sharpe_spread (population sd), min, max, positive_folds, and anchored and rolling pair tables
(share of same-sign Sharpe pairs; mean OOS / IS Sharpe). Fold nets sum to the run's net. A report of one
configuration, never an input; select() reads recorded runs only. Contract: folds (count),
sharpe_spread, positive_folds.

S5 Neighbourhood. neighbours(grid, centre): per strictly increasing numeric grid, one step down and up
(one at an edge). Each neighbour is its own run; for a trial a counted trial in the grid and budget, the
whole set planned first and refused whole when the budget cannot cover it. verdict: r = mean neighbour
Sharpe / centre Sharpe; n/a without neighbours or centre Sharpe <= 0; plateau if r >= 0.7 and all nets
share the centre's sign; spike if r < 0.5 or half the nets have the other sign; else mixed (missing
Sharpe = 0; a zero net has no sign). Contract: verdict, ratio, run_ids.

S6 Breakdowns. Net, trades, Sharpe by year, by volatility tercile of the trade date (same-day range, so
descriptive only), by CT entry hour and by weekday; each sums to the run's net. Best day and its share,
rolling 63/252-day Sharpe (series in rolling_sharpe.parquet), MAE and MFE quantiles (all, winners,
losers), hold minutes, trades per day.

S7 Sizing table. Trades replayed at other sizes, net scaled floor(net x q / c) (costs linear, stated).
Closed-trade equity; floor = highest end-of-day equity (>= 0) minus the plan's eval drawdown, moving at
day ends; whole contracts capped by the plan (x 10 on micros). Rules: fixed; fixed fraction 0.05/0.10/0.20
of distance over per-contract risk (stop, else 95th percentile MAE); vol target (DLL / 3 or drawdown /
10 over the 20-day sd); quarter and half Kelly ("upper bound, in-sample"); integer decisions. Without a
sizing_plan: fixed only and Kelly fractions.

Cost: the default suite adds about 2.7 s to a 9 s 5-year synthetic run (peak RSS about 1.2 GB).
