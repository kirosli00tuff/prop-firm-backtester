# PropFirmBacktester design brief (2026-11-14)

Condensed from a survey of methods and existing tools (sources at the end) and of the earlier research codebase
("upstream") whose conventions pfb ports. The ADR turns this into binding design.

## 1. Purpose and users

- A backtester for intraday CME futures strategies that also prices the prop-firm side: pass probability, days to
  pass, cost to get funded, payouts and expected value per account purchase, per firm and plan.
- Two users: a person, through a dashboard; AI coding agents in CLI sessions, through one command and a compact JSON
  result, so a session writes one strategy file and never rewrites loaders, costs or rules.
- Built for one laptop (about 20 threads, 16 GB RAM), licensed data read in place, and a public build on synthetic data.

## 2. Architecture (no framework adopted whole)

    Bar stores (parquet, read-only) -> DataGateway (closed-window and bound refusals, int ticks)
      -> strategy file (SPEC + generate) -> bar engine (Numba) -> trade ledger + per-minute day equity
      -> metrics and statistics -> prop-rule state machine + day-path Monte Carlo
      -> append-only registry (global N) -> CLI JSON, MCP server, Python API, dashboard
    Later: NautilusTrader as a second engine on MBP-1 and MBO data for strategies that survive (version 3).

Why custom: no open-source engine models trailing drawdowns that count open P&L, consistency caps, payout gates
or fees; NautilusTrader (LGPL-3.0, native market-data adapters, L3 book fills) is the right second engine, too heavy
for screening many variants. vectorbt's free edition is source-available with a Commons Clause and weak on
futures; Backtrader is unmaintained since 2024; LEAN needs Docker and is heavy for this laptop.
A 2026 study ran 15 strategies through 5 engines and found up to 3.71% spread in total return, and recommends at
least two independent validators: the new engine is checked against upstream's own engine first.

## 3. Version plan

- Version 1: repo, DataGateway, instruments and costs, bar engine with parity, metrics, look-ahead
  suite, global registry; dated rule configs for five firms, prop Monte Carlo from the strategy's own
  day paths, several strategies in one account, firm-by-plan EV matrix, firm rule refresh from the firms' own
  pages; analysis suite, strategy template, CLI with JSON, MCP server, dashboard, agent docs.
- Version 2: CPCV and PBO for grids, SPA and Reality Check (the arch package), rules in force by date,
  correlated multi-account mode, pessimistic limit-order fills, planted-edge recovery tests.
- Version 3: NautilusTrader validator on MBP-1 and MBO, latency sensitivity, public synthetic build, docs site.

## 4. Metric definitions (daily figures include zero-trade days)

| Metric | Definition |
|---|---|
| Net P&L | sum of trade P&L minus commissions, exchange fees and slippage |
| Sharpe (annual) | mean(daily net) / sd(daily net) x sqrt(252); capital base declared (default: distance to the drawdown floor) |
| Sortino | mean(daily) / sqrt(mean(min(daily, 0)^2)) x sqrt(252) |
| Win rate | trades with net > 0 / trades |
| Profit factor | sum of winning net / abs(sum of losing net) |
| Expectancy | mean net per trade, in USD and ticks |
| Payoff ratio | average win / abs(average loss) |
| Max drawdown | max over t of (running peak equity - equity), at minute resolution |
| Calmar | annualized net / max drawdown |
| MAE, MFE | worst and best open excursion per trade, in ticks |
| Best-day share | largest daily profit / total profit (the consistency-rule input) |
| Exposure | trades per day, average hold, share of minutes in a position |
| PSR, DSR, MinBTL | section 5 |

## 5. Statistics (per-period, not annualized, unless stated)

- PSR(SR*) = Phi( (SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + ((g4 - 1)/4) SR^2) ), g3 skewness, g4 raw kurtosis.
- DSR = PSR(SR0), SR0 = sqrt(V[SR_n]) ((1 - gamma) Phi^-1(1 - 1/N) + gamma Phi^-1(1 - 1/(N e))),
  gamma = 0.5772 (Euler-Mascheroni), V[SR_n] the variance of Sharpe across trials (Bailey and Lopez de Prado 2014).
- MinBTL (years) ~ ((1 - gamma) Phi^-1(1 - 1/N) + gamma Phi^-1(1 - 1/(N e)))^2 / SR^2, SR annualized.
  At N = 1000 the bracket is about 3.26, so MinBTL ~ 10.6 / SR^2 years (SR 1: about 10.6 years; SR 2: 2.6).
  Shown on every run next to the Sharpe as the N-adjusted bar.
- PBO by CSCV (Bailey, Borwein, Lopez de Prado, Zhu 2017); SPA, Reality Check and StepM from the arch package
  (stationary bootstrap, block size sqrt(T) by default). Both version 2.

## 6. Prop-firm Monte Carlo

- Resampling unit: one trading day with its per-minute equity path, so intraday-trailing plans (Bulenox Option 1,
  MFFU Rapid funded) breach on open-profit peaks exactly, and daily loss limits, flat-by times and best-day
  shares apply exactly.
- Paths: stationary block bootstrap of days (default), randomized historical start dates, cost-stressed variants.
- Rule state per firm, plan and size: balance, peak (end of day or intraday), floor, lock, day count, winning
  days, best day, payout count, buffer. Rules live in dated configs with source and lookup date, never in code.
- Outputs: P(pass); days to pass (median, 90th percentile); E[cost to funded] = E[cost per attempt]/P(pass) plus
  activation; funded EV; P(breach before first payout); EV per purchase; a firm-by-plan matrix per strategy.
- Several strategies in one account: the combined day path, so rare-signal strategies can be pooled.
- Several accounts on one signal: copies do not diversify; version 2.

## 7. Agent contract

- Strategy file: SPEC (hypothesis_id, root, bar size, session, params with declared grid) and a pure
  generate(bars, params) returning entry_long, entry_short, exit, stop_ticks, target_ticks. Imports limited to
  numpy, polars and math; no file or network access.
- One command: `pfb run <strategy.py> --firm <f> --plan <p> --mc <n> --json`; stdout carries about 30 keys
  (net, trades, win rate, PF, Sharpe, Sortino, max DD, PSR, DSR, N, MinBTL bar, cost stress +1 and +2 ticks,
  P(pass) and EV per firm and plan, warnings, artifact paths); full artifacts under runs/<run_id>/.
- MCP server (official Python SDK, FastMCP, stdio): list_instruments, list_firms, describe_strategy_template,
  validate_strategy, run_backtest, run_prop_mc, get_run, compare_runs, trial_count.

## 8. Main risks

- Cost model error: cost stress on every run; real fills from a prop platform simulator,
  exported by the user, to check the model later.
- Look-ahead: next-bar entries by default, the truncation-invariance check and planted-future canaries.
- Overfitting: N rises with every counted run by design; a trial budget per hypothesis keeps agents from
  inflating it.
- Memory: never load MBO whole; stream per day (version 3).
- Rule drift: firms changed rules several times in 2026; dated configs and a refresh step before any use.

## Sources (primary where possible)

- NautilusTrader docs: nautilustrader.io/docs/latest/concepts/backtesting/data-and-venues/
- Implementation risk across engines: arxiv.org/pdf/2603.20319
- Deflated Sharpe Ratio: papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- arch multiple comparisons: arch.readthedocs.io/en/latest/multiple-comparison/multiple-comparison-reference.html
- DuckDB vs Polars memory: codecentric.de/en/knowledge-hub/blog/duckdb-vs-polars-performance-and-memory-with-massive-parquet-data
- MCP server guide: modelcontextprotocol.io/docs/develop/build-server
- Firm pages: help.topstep.com/en/articles/8284197-trading-combine-parameters,
  topstep.com/express-funded-account-rules, support.lucidtrading.com (LucidFlex evaluation),
  help.myfundedfutures.com (Rapid 50K), help.tradeify.co (Select, Growth, drawdowns, DLL), bulenox.com
