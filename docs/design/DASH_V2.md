# Dashboard v2: design (2026-11-17)

Brief: a simple, functional and minimalist dark design, functional over pretty: a box where code can be
uploaded as text or as a file; as it backtests, a dashboard opens with the performance of that script against
the data.

## Look
- Dark only. Background #0E1014, panels #161A20, text #E6E8EB, muted #8A9099, one accent #4C9AFF, gain
  #3FB950, loss #F85149. One sans font (system), monospace for code and numbers. No gradients, no icons beyond
  status dots. 8 px spacing grid. Wide layout, max content width 1280 px.
- Two pages in a left nav: Backtest (default), Runs.

## Backtest page: one column, five steps top to bottom (each step unlocks the next)
1. Strategy. A code box (paste) and a file drop (.py), same result. Shows the file's sha256 and the parsed SPEC
   (root, bar minutes, session, contracts, params). A "Template" link copies strategies/_template.py.
2. Check (free, N unchanged). Lint, look-ahead check and a synthetic run. A pass/fail list with the exact
   message for each failure. Nothing touches real bars.
3. Data. For the SPEC's root: which windows are open (with dates), which are closed (greyed, with the
   reason), what is on disk (from runs/catalog.json, built by `pfb data catalog`). The dashboard fetches no data.
4. Run. Pick open windows and firm plans (none preselected). The N counter shows
   "N k -> k + 1" and the run's hypothesis is built from the SPEC and shown before the confirm box. One run
   at a time, in a subprocess with a memory cap and a timeout. Progress bar.
5. Results (opens when the run ends). KPI row: net USD, trades, win rate, average trade, Sharpe and deflated
   Sharpe, max drawdown. Equity curve with drawdown band. Per-plan prop Monte Carlo table: pass rate, EV per
   purchase, payout count, main breach reason. Costs (fees, slippage, event costs). Warnings. JSON export.

## Runs page
- One table of every run (date, strategy, kind, root, net, trades, verdict), click opens its Results view.

## Fixed by the program's rules (not design choices)
- Every real-data run is a trial: N + 1, a registered hypothesis, closed windows refused, crash still counts.
- Same execution path as the CLI (pfb.runner); the dashboard adds no second engine.
- Bound to 127.0.0.1 only. Uploaded files go to strategies/uploads/<sha256>.py.
- No network call and no data purchase from the dashboard.

## Decision
A real-data run from the dashboard counts as a trial (N + 1), as above; steps 1-2 give a free path. The
alternative (free exploratory runs) would break the global N.
