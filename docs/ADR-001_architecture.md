# ADR-001: architecture of the PropFirmBacktester core

Status: binding; section 16 amends sections 1 to 15 and wins where they disagree, and section 17 amends section 6.
Written on 2026-11-15 from docs/DESIGN_BRIEF.md. "Upstream" means the earlier research codebase whose data
conventions, cost expressions and statistics pfb ported and was checked against during development; the public
copy contains no upstream code, data or parity tests. Where the public copy differs (the synthetic demo store and
its pins file, the fictional cost table and release calendar in config/demo/, the optional upstream ledger), a
"Public copy" note says so. A later ADR may amend this one; code never does so silently.

## 1. Modules and the dependency rule

| Package | Owns | May import from pfb |
|---|---|---|
| pfb.instruments | table.py (config/instruments.toml), calendar.py (config/calendars/<group>.json), sessions.py | nothing |
| pfb.data | errors.py, paths.py, windows.py, catalog.py, gateway.py: the only code that reads bars | nothing (the tick lookup is injected) |
| pfb.costs | model.py (cost models), fees.py (config/fees/<firm>.toml) | instruments |
| pfb.metrics | core.py (section 4 of the design brief), stats.py (section 5 and the upstream ports) | nothing |
| pfb.engine | strategy.py, lint.py, kernel.py (Numba), run.py, synthetic.py, lookahead.py | instruments, costs, data, metrics |
| pfb.registry | chain.py, records.py, genesis.py, sync.py, api.py | nothing |
| pfb.cli | main.py plus one <name>_cmd.py per package | everything |

Nothing imports pfb.cli. No module imports upstream code at run time; the development-time parity tests against upstream are not
part of the public copy. Config lives in config/ (TOML or JSON, each file's sha256 goes into registry records).

## 2. Data flow

    Bar stores (parquet, read in place, read-only; the synthetic demo store without config/paths.toml)
      -> DataGateway: path, hash, closed-window and bound refusals; float prices -> int64 vendor ticks
      -> engine.run: session masks (instruments.sessions), cost tables (costs), signals (strategy generate)
      -> engine.kernel (Numba): trade ledger, per-minute equity
      -> metrics: summary -> registry: one hash-chained record -> CLI: JSON on stdout, artifacts under runs/<run_id>/

## 3. Units, time and the bar convention

- Prices: int64 VENDOR ticks. vendor_tick_fixed = tick_size x vendor_price_factor x 1e9 (whole; factor 100 for ZC,
  ZW, ZS, ZL, HE, LE, which the vendor quotes in cents). ticks = rint(price x 1e9) // vendor_tick_fixed;
  a price off the grid raises, nothing is snapped (as upstream).
- Money: integer cents everywhere outside the kernel. Five roots have a tick value that is not a whole cent (ZT, ZF
  781.25 c; ZN, TN 1562.5 c; M6B 62.5 c). Inside the kernel gross P&L is exact in int64 units of 1e-4 USD
  (tick_value_cc); each amount that leaves the kernel (a trade's gross, an equity mark) is floored to cents (toward
  minus infinity), once. For every other root this is exact. Upstream keeps exact Fractions here; the floor costs at most
  one cent per trade and never favours the strategy. A float never carries money and never decides a boundary.
- Time: ts_event is int64 UTC nanoseconds. The CME trade date is the stored label (the gateway checks it, section 5).
  The CT session clock is ct_minute, the minute of the America/Chicago day (0..1439), computed with zoneinfo.
- Bar: ts_event is the bar OPEN; the bar closes 60 s later. A decision is made at a bar's close and acts at the
  open of the next bar. Nothing is forward filled: a missing minute is simply absent.

## 4. Instruments, calendars and sessions (I2)

- config/instruments.toml: schema "pfb.instruments/1", one [[instrument]] per root (47, ES and MES included):
  root, group, subgroup, exposure, contract_type, tick_size, tick_value_usd, multiplier_usd (decimal STRINGS),
  vendor_price_factor (int), admissible (bool), tick_source, weight_source. No commission here (fees are per firm).
  `load_instruments(path=None) -> InstrumentTable`; `.get(root) -> Instrument` (frozen; Decimal fields plus derived
  `vendor_tick_fixed: int`, `tick_value_cc: int` (1e-4 USD, whole), `tick_value_cents: Fraction`), `.roots()`,
  `.tick_fixed_of(root) -> int`, `.sha256`. An unknown root raises UnknownRoot.
- Calendars: PORTED as data, one config/calendars/<group>.json per group (8 groups), produced once by a read-only
  extraction from upstream's calendars (a recent era and a historical era; crypto has the recent era only). Only dates, kinds and clock times are ported
  (holidays, early halts, late opens, session regimes, crypto's booked-forward days, unsourced dates, and the
  per-date flat-by exceptions); none of upstream's citation text. Why port and not adapt: run-time code may never import
  Upstream, the public build needs calendars, the files are small, and their hashes belong in registry records.
- sessions.py, all in integer nanoseconds (no float division):
  `trade_dates_of(root, ts_ns) -> int32 ordinals` (-1 where the calendar books nothing; upstream's close-minute rule),
  `open_intervals(root, first, last) -> (starts_ns, ends_ns, day_ordinals)`,
  `flat_by_instants_ns(root, trade_dates) -> int64[]` (the instant by which a position must be flat on each trade
  date, as upstream: 15:08 CT for equity, rates, fx, energy, metals, crypto; 13:18 grains; 13:03
  livestock; on an early halt the earlier of that and 15 minutes before the halt; plus the per-date exceptions),
  `session_window_ns(root, trade_date, name) -> (start_ns, end_ns)` for name "rth" (the group's day session) or
  "globex" (the whole session). Outside calendar coverage every function raises CalendarCoverageError.
  During development each function was compared with upstream's on every date of the coverage, not a sample;
  the public tests are hand-computed cases.

## 5. Data gateway

- Three roots from config/paths.toml (local): research, step2, hist. Readable files, by exact name only: one
  research file per root, plus optional step2 and hist files for the roots that have them. A micro (MNQ and so
  on) has only its research file: its earlier history is the full-size root's, requested by name; no price-path
  substitution. Sealed, vendor, derived and raw-vendor trees are never opened.
  Public copy: the demo store holds one research-era file per demo root (MNQ, NQ), and
  config/paths.example.toml shows the three roots.
- config/closed_windows.toml is binding and its sha256 is pinned in pfb/data/windows.py. A row is closed when its
  stored trade_date label lies in a window's label range OR its ts_event lies in the window's UTC span (so a
  mislabelled row cannot pass). The catalog also records the sha256 the data producer pinned for each file (a pins file,
  schema pfb.pins/1: config/paths.toml [stores] pins, or data/demo/pins.json for the demo); the producer
  re-booked every row of those exact bytes on its group calendars, which is why the stored label can be trusted.
- `DataGateway.bars(root, first, last, session=ALL) -> polars.DataFrame`. It raises, never filters, on: a request
  touching a closed label range; a path outside the roots or not matching its era's name; a file hash differing
  from the catalog; a returned row with a label outside the request, a ts_event at or after the bound (17:00 CT on
  the last requested day), a label or time inside a closed window, a duplicate or non-increasing time, or a price
  off the tick grid. Columns: ts_event, open, high, low, close (ticks), volume, trade_date, ct_minute,
  is_roll_session, in_flatten_window, in_no_new_positions_window, in_scheduled_closure, vendor_degraded_day,
  early_halt_ct, gap_before_minutes, instrument_id, raw_symbol, era.
- `pfb data catalog` writes runs/catalog.json from parquet footers and the trade_date column only.

## 6. Costs and fees

- A fill pays, per side: commission (qty x commission_side_cents of the chosen firm's fee file) plus slippage.
- `CostModel` (pfb.costs.model): `model_id: str`, `config_sha256: str`,
  `slippage_cents(root, contracts) -> int64[48, 2]` (the cents one MARKET fill of `contracts` pays, by 30-minute CT
  bucket = ct_minute // 30 and side 0 buy, 1 sell), `calibrated(root) -> bool[48]`,
  `event_slippage_cents(root, contracts) -> int64[2]`. `load_cost_model(spec, instruments, paths) -> CostModel`,
  spec "book", "fixed:<ticks>" or "stress:<base spec>:<k>".
  (a) book: a frozen per-bucket slippage table (JSON), sha256 pinned in config/costs.toml. Public copy: the
  repository ships a fictional table, config/demo/book_costs.json (generated by `python -m
  pfb.costs.demo_tables`); config/paths.toml [costs] book_table may name your own, with its pin updated. Cents = `math.ceil(round(qty * slip_ticks * float(tick_value_cents), 6))`,
  Upstream's own expression, evaluated once in Python when the table is built, so
  the kernel sees integers only and the result is cent-identical to upstream's lookup. The ceil is once per fill on the
  whole quantity. A bucket with no calibrated value takes the root's largest per-side slippage and
  is marked uncalibrated. The table has 45 roots (no MES, no ES) and no date axis (five sample dates, 2025-2026).
  (b) fixed: ceil(qty x k x tick_value_cents) in exact rational arithmetic, k a decimal string.
  (c) stress: the base model plus ceil(qty x k x tick_value_cents) per side.
- A target fill is a resting limit: commission only, no slippage (as upstream).
- Event windows (upstream charges a larger slippage for 30 minutes after scheduled releases) are carried in the
  interface but OFF: the release calendar is not ported. This understates cost on release minutes. (Section 17
  later turned them on.)
- config/fees/<firm>.toml (topstep, bulenox, lucid, mffu, tradeify): schema "pfb.fees/1", firm, lookup_date,
  source_file, and per allowed root commission_side_cents and commission_rt_cents (integers; an odd round turn is
  rounded UP per side and noted). Taken from upstream's fee records; no page text is copied.
  `load_fees(firm) -> FeeSchedule` with `.commission_side_cents(root)` (raises for a root the firm does not list),
  `.sha256`, `.lookup_date`. Default firm: topstep.

## 7. Strategy interface v0

One file per strategy. Imports limited to numpy, polars, math (and `from __future__ import annotations`).

    SPEC = {
        "hypothesis_id": "...",            # the registered hypothesis, or an engine-check label
        "root": "MNQ",
        "bar_minutes": 1,                  # only 1 is accepted
        "session": "globex",               # or "rth": which bars generate() receives and may trade
        "contracts": 1,                    # whole contracts, one position at a time
        "params": {"name": {"default": v, "grid": [v, ...]}, ...},   # the declared grid
    }
    def generate(bars, params) -> dict     # keys entry_long, entry_short, exit (bool[n]); stop_ticks, target_ticks (int[n])

`bars` is a frozen view of read-only numpy arrays of length n: ts_event, open, high, low, close (int64 ticks),
volume, trade_date (int32 ordinal), ct_minute (int16). generate is pure and causal: the value at bar i may use
bars 0..i only. stop_ticks and target_ticks are distances from the entry fill price, read at the entry signal bar;
0 means none. `load_strategy(path) -> Strategy(spec, generate, sha256, path)` validates SPEC and the grid.
`lint_strategy(path) -> list[LintViolation]` is a static AST check (nothing is executed): any other import, star
import, dunder attribute, banned name (open, eval, exec, compile, __import__, input, breakpoint, globals, locals,
getattr, setattr, delattr, vars) or file or network attribute (numpy and polars readers, writers, scans, sinks,
memmap, fromfile, tofile, save, load) is a violation. strategies/_template.py is the starting file.

## 8. Bar engine: fill conventions. Each is chosen so that it cannot flatter a result.

Inputs to the kernel are plain arrays over the USABLE bars of one root (bars flagged in_scheduled_closure are
removed first) and integers; no float enters the kernel. run.py derives, per bar i:
- `session_end[i]`: bar i is the last usable bar before a trade-date change, a removed closure bar, an
  instrument_id change (a roll splice) or the end of the data;
- `flat_by[i]`: ts[i] >= F(trade date), F from `flat_by_instants_ns`; `entry_fill_ok[i]`: ts[i] + 60 s < F, so
  neither the fill minute nor the next is must-be-flat (as upstream).
Order inside bar i:
1. Open. A pending exit fills at open[i]. Then, if flat and entry_fill_ok[i], a pending entry fills at open[i]
   (stop = fill price -/+ stop_ticks, target = fill price +/- target_ticks). A pending entry that cannot fill on
   this bar is dropped, not carried.
2. Flat-by. If flat_by[i] and a position is open, it exits at open[i] (reason FLAT_BY).
3. Inside the bar, for an open position (also one opened at this open). Long: the stop is touched when
   low[i] <= stop; the target is filled only when high[i] >= target + 1 tick (upstream's rule for resting limits:
   the price must trade one tick through). Short: mirrored. Stop touched: exit at the
   worse of the stop price and open[i] (a gap fills at the open), a market fill with slippage. Otherwise target
   filled: exit at the target price exactly, never better, commission only. Both in one bar: the stop.
4. Close (the decision). If session_end[i]: an open position exits at close[i] (reason SESSION_END, a market fill
   priced in bar i's cost bucket) and the bar's signals are ignored, so no position and no order crosses a session,
   a roll or a scheduled closure. Otherwise: in a position, exit[i] queues an exit for the next open; flat, exactly
   one of entry_long[i], entry_short[i] queues an entry (both set: ignored and counted). An entry signal while in a
   position is ignored; there is no reversal in v0.
Every market fill pays commission plus the slippage of the fill bar's bucket and side (also a close fill, as
upstream). One position per strategy, whole contracts, size SPEC.contracts.
Not in the engine (firm rules): loss limits, upstream's 2-minute minimum hold and 20 entries a day, roll
blackout dates. The caller may pass an `entry_allowed` bool mask (default all true) for such rules.

## 9. Engine outputs

- Trade ledger, one row per round trip: entry and exit ts_event (ns) and bar index, entry and exit price (ticks),
  side (+1, -1), contracts, gross_cents, commission_cents, slippage_cents, net_cents, mae_ticks, mfe_ticks (worst
  and best open excursion from the entry price, from bar extremes while the position was open), exit_reason
  (SIGNAL, STOP, TARGET, FLAT_BY, SESSION_END), trade_date.
- Per-minute equity, one row per usable bar: ts_event, trade_date, position, eq_close, eq_low, eq_high (cents).
  Equity = realized net so far + open gross at the mark. eq_close marks at close[i]. eq_low and eq_high bound the
  equity path inside the bar: the minimum and maximum over the equity at the open, after each fill, at the bar's
  adverse and favourable extreme while a position was open, and at the close. The order inside a bar is unknown,
  so a reader (the prop rules' trailing drawdown) must take the worst case.
- Daily rows, one per trade date with at least one usable bar (zero-trade days included): open, low, high, close
  equity and net_cents. Positions never cross a session, so a day's open equity is the previous day's close.
- `RunResult(trades, equity_minute, daily, summary, counters, provenance, spec, params)`: three polars frames, the
  metrics summary, counters (signals ignored, entries dropped, conflicts, uncalibrated fills) and provenance
  (files and hashes read, config hashes, engine version `pfb.engine.ENGINE_VERSION`). Artifacts go to
  runs/<run_id>/ (local, ignored; never committed: they derive row by row from market data).
- `run_strategy(strategy_path, *, source, root=None, first=None, last=None, params=None, fees="topstep",
  cost_model="book", seed=0, entry_allowed=None) -> RunResult`, source "real" (through the gateway) or "synthetic"
  (pfb.engine.synthetic: seeded bars built in code, any root in the instrument table).

## 10. Parity with upstream's generalized engine

Development history: the parity runs below needed upstream and are not in the public copy; the reference
strategy and the engine-check allowlist remain.
- Reference strategy: strategies/reference_breakout.py, a port of upstream's breakout template at its default
  parameters (range 15 minutes from the day-session open, buffer 4 ticks, exit 75 minutes after the entry DECISION
  or from 2 minutes before the day-session close, one entry per trade date, size upstream's frozen q_c), session
  "globex", no stop and no target (upstream's engine has no stop order type). No parameter is tuned.
- Upstream's engine runs read-only in its own environment on (a) synthetic bars with known answers and (b) MNQ for
  one open month (fixed before any run; the month has no index roll), exported through OUR gateway to
  a local scratch folder (never committed), with an empty release calendar. Trades and net P&L must match to
  the cent. Each difference is traced to a named convention and ruled on in review. Known candidates: Upstream's
  2-minute minimum hold, upstream's account rules (loss limit), and the flat-by trigger when a minute is missing.
- Parity and benchmark runs are engine checks: logged in the registry, N unchanged.

## 11. Metrics

- pfb.metrics.core implements every row of design brief section 4 from the ledger, the daily rows (zero-trade
  days included) and the per-minute equity. One Sharpe convention everywhere, upstream's `moments()`: per-day Sharpe = mean / population sd of daily net; skewness and RAW kurtosis from
  population moments; annual Sharpe = daily x sqrt(252). Max drawdown is on eq_close at minute resolution.
  Ratios with a zero denominator are null, never infinity. The capital base is carried as a declared field and
  does not enter the Sharpe (it cancels).
- pfb.metrics.stats: ports of upstream's multiple-comparison and statistics code with its pinned known
  answers, plus PSR (new: Upstream has none) and MinBTL as in design brief section 5.
  DSR at the live N = PSR(SR0), SR0 = sqrt(V) x bracket(N), bracket(N) = (1 - g) ppf(1 - 1/N) + g ppf(1 - 1/(N e)).
  V is the variance of the per-day Sharpe across trials: when a run has no grid of its own, V = 1/(T - 1), the
  null sampling variance (as upstream);
  with a grid of two or more configurations the cross-configuration population variance is reported beside it.
  for example bracket(1000) = 3.2551215; MinBTL years = bracket(N)^2 / (annual Sharpe)^2. N is read from the registry.

## 12. Look-ahead suite

- Truncation invariance: for 20 seeded random cuts k, generate(bars[:k]) must equal generate(bars)[:k] on all five
  outputs. It runs in `pfb check` (synthetic bars) and inside run_strategy on the run's own bars before the kernel;
  a failure refuses the run.
- Canaries, ported in idea from upstream's leakage canaries to this engine, on synthetic bars: a planted future
  jump with a marker, on which a marker-reading strategy must show NO edge through the real engine; and two leaky
  mutants that must be DETECTED: same-bar fill (the fill takes the decision bar's own open) and peek-next-bar
  (signals shifted one bar back). Mutants live in the test suite only; the kernel has no switch for them.

## 13. Global registry

- ledger/registry.jsonl, tracked, append-only. One record per line in canonical JSON (sorted keys, separators
  "," and ":", ASCII). `prev_sha256` is the sha256 of the previous line's bytes without its newline; the first
  record uses 64 zeros. Reading verifies the whole chain and every n_before/n_after link, and refuses on any break.
- Kinds: genesis (once), sync, hypothesis, trial, engine_check, synthetic.
  genesis: with an [upstream] root in config/paths.toml it embeds upstream's ledger/trial_registrations.jsonl
  and ledger/trial_amendments.jsonl lines unchanged (as strings) with each file's sha256, and sets N to upstream's
  current_n (the last line's n_after after upstream's chain checks, ported). Upstream's files are never written.
  Public copy: without an upstream (the default) genesis imports nothing and N starts at 0.
  sync (only with an upstream): before ANY append the registry hashes upstream's two ledger files; if either differs from the last genesis or
  sync record the append is refused (SyncRequired) until `pfb registry sync` imports the new lines (the old lines
  must be an unchanged prefix, else it refuses loudly) and adds upstream's N increase.
  hypothesis: registers a hypothesis file (TOML: id, statement, strategy path and sha256, root, window, grid,
  budget) by sha256. N unchanged. None is registered.
  trial: a counted run on real data. N rises by the number of configurations in the run that are NEW for that
  hypothesis; a configuration is its config_id, the sha256 of the canonical tuple (hypothesis sha256, strategy
  sha256, params, root, window, session, cost model id and hash, fee hash, instrument hash, engine version, data
  file hashes). An exact replay adds nothing and is still logged. Refused: no registered hypothesis or a changed
  hypothesis or strategy file; a configuration outside the declared grid; more distinct configurations than the
  budget; a dirty git tree (ledger/registry.jsonl itself is exempt); an unsynced upstream ledger; a failed lint or
  look-ahead check.
  engine_check: a fixed strategy at its SPEC defaults, declared in a design document (the record names it),
  no tuning, no selection: logged, N unchanged. synthetic: synthetic data only: logged, N unchanged.
- N = n_pe (upstream's current_n at the last genesis or sync; 0 without an upstream) + n_local (new configurations over all trial records).
- Record fields: run_id, kind, time_utc, hypothesis_id, hypothesis_sha256, strategy_sha256, git_commit,
  git_dirty, data_files (path, sha256 from the catalog), instruments_sha256, cost_model_id, cost_config_sha256,
  fees_sha256, closed_windows_sha256, params, seed, engine_version, n_before, n_after, configurations,
  metrics (the summary), artifacts, declared_in, prev_sha256, schema "pfb.registry/1".

## 14. CLI and what is deferred

`pfb data catalog`; `pfb registry status` (N, chain state, upstream sync state) and `pfb registry sync`;
`pfb check <strategy.py>` (lint plus truncation check on synthetic bars); `pfb run <strategy.py> --kind
engine_check|synthetic --json [--root R --first D --last D --fees F --cost-model S --declared-in P]`, which prints
the metrics summary as one JSON object and appends one registry record. `--kind trial` does not exist yet.
Deferred: firm rules and the Monte Carlo; the release calendar for event-window costs; the full CLI
contract, MCP server and dashboard; price paths for micros; bar sizes other than 1 minute.

## 15. Rulings made here that may be reversed later

1. A closed window closes every root unless its row says otherwise. (Public copy: config/closed_windows.toml
holds made-up windows for the demo.)
4. Fractional-cent roots floor each trade's gross to the cent. 5. A target needs a one-tick trade-through.
6. No position crosses a session. 7. Sharpe uses the population sd. 8. DSR's default variance is 1/(T - 1).
9. N counts distinct configurations; an exact replay adds nothing. 10. Calendars are ported as data.

## 16. Amendments of 2026-11-15, after the design check. This section wins over 1 to 15.

A1 Counted runs. New package pfb.runner (api.py; it may import every package but cli; only pfb.cli and
tests import it). It is the one entry that runs a strategy and logs it:
`pfb.runner.run(strategy_path, *, kind, check_id=None, root=None, first=None, last=None, params=None,
fees="topstep", cost_model="book", seed=0) -> RunOutcome(result, begin_record, result_record)`. Order:
(1) load and lint the strategy, truncation check on synthetic bars; (2) resolve the configuration and every hash
(data file hashes come from the catalog; no bar is read); (3) the kind's preconditions; (4)
`pfb.registry.api.begin(...)` appends and fsyncs the run's record (kind trial, engine_check or synthetic; a
trial's N rises HERE) and returns a receipt; (5) the engine runs; (6) `pfb.registry.api.finish(run_id, status,
metrics, artifacts)` appends a record of kind `result` (N unchanged; status "ok", or "failed" with the error
class) BEFORE anything is printed or written; (7) artifacts are written, then the CLI prints. A crash after step
4 still counts; `pfb registry status` lists begun runs that have no result.
In pfb.engine, `run_strategy(..., source="real")` needs `permit: RunPermit` (pfb/engine/permit.py, frozen: run_id,
kind, record_sha256, strategy_sha256, root, first, last, params_json, fees, cost_model), built by pfb.runner from
the receipt. A missing permit, or one that differs from the call, raises PermitRequired. `run_strategy` also
takes `n_trials: int | None` (the registry's N after the begin record; None leaves the DSR fields null).
The permit stops an uncounted run by accident, not by intent: no Python check stops a session that reads parquet
itself, which the project rules forbid. At this point pfb.runner accepted the kinds engine_check (real data) and
synthetic only; trial and hypothesis were wired later (docs/TRIALS.md).

A2 Engine-check allowlist. config/engine_checks.toml (tracked, changed only through review, schema "pfb.engine_checks/1", its
sha256 in every record as engine_checks_sha256): one [[check]] with id, declared_in, strategy, strategy_sha256,
root, first, last, fees, cost_model. An engine_check always runs on real data, names its entry (`--check-id`),
must equal the entry in every field and runs at the SPEC default params; otherwise it is refused
(EngineCheckNotDeclared). The record stores the entry. It lists four: the reference breakout on MNQ
for one open month and on NQ for three open benchmark windows of 1, 5 and 10 years.

A3 Sessions API. It replaces the list in section 4; arrays carry trade_day (A9):
`trade_days_of(root, ts_ns) -> int32[]` (-1 where the calendar books nothing), `open_intervals(root, first, last)
-> (starts_ns, ends_ns, trade_days)`, `session_window_ns(root, trade_date, name) -> (start_ns, end_ns)`,
`unsourced_days(root) -> int32[]`, and `must_be_flat_intervals_ns(root, first, last) -> (starts_ns, ends_ns)`:
half-open, sorted, merged. An instant is inside exactly when upstream requires a flat account at that
minute: from each trade date's flat-by time (the times of section 4) to the next session open, the grain pause
07:43-08:30 CT, and upstream's late-open rules (the evening session before a late open, and the period before it).
`flat_by_instants_ns` does not exist. In section 8: flat_by[i] = ts[i] lies in an interval; entry_fill_ok[i] =
neither ts[i] nor ts[i] + 60 s lies in one.
Tests: against upstream's own function on every booked minute of every trade date of the recent era, for one
root per group and every root with an exception of its own. Before the recent era upstream has no flat-by function: the hist
era applies the same rule to the hist calendars and is tested by hand-computed cases only. A real-store test
(skipped without paths) checks that trade_days_of equals the stored label on every usable row of one open month
per group and era; a mismatch is reported, never patched.
Unsourced trade dates (upstream marks some livestock dates of the hist era): pfb.engine.run drops their bars before the
kernel and counts them (counters bars_dropped_unsourced, days_dropped_unsourced).

A4 Lint by ALLOWLIST. It replaces the ban list of section 7. pfb/engine/lint_allow.py holds four
frozensets: numpy names, polars names, math names, builtins. In a strategy file every attribute name and every
called name must be on a list or be defined in the file itself; anything else is a violation. No listed name
takes a path, a URL, SQL text, a code string or a foreign pointer, and none draws random numbers. Also
violations: any import other than `import numpy`, `import polars`, `import math` (with or without `as`), `from`
imports of listed names out of those three, and `from __future__ import annotations`; a submodule import; a star
import; any dunder name or attribute; `global` and `nonlocal`. Nothing is executed. The test holds one refused
snippet per escape (pl.sql, SQLContext, DataFrame.sql, every read_, scan_, write_ and sink_, deserialize, Config,
np.load, save, loadtxt, genfromtxt, fromregex, fromfile, tofile, memmap, DataSource, lib, ctypeslib, ctypes, f2py,
random, open, eval, exec, getattr, a dunder), and the reference strategy passes.
generate must return stop_ticks and target_ticks >= 0; a negative value refuses the run.

A5  Gateway. (a) A catalog entry whose pe_pin_status is "mismatch" is
refused; "none" (some step2 files) is read and flagged in provenance and in the summary warnings. (b) The
MES confirmation era is refused (EraNotSupported; it stays in the catalog): Upstream re-books its
scheduled-closure prints with pinned rulings that are not ported, so section 5's "upstream re-booked every row of those
exact bytes" holds for the pinned files only. (c) Order of checks: refuse the request on a label-range touch;
scan with the label range pushed down; then raise on any selected row whose label or time is closed, outside the
request, or at or after the bound. Rows the predicate drops are never returned and never checked, which is why a
research file that also holds embargo rows stays readable. (d) The tick check is exact: `DataGateway(paths,
windows, catalog, tick_fixed_of, price_factor_of)`; the file's metadata tick x price_factor_of(root) must equal
tick_fixed_of(root). InstrumentTable gains `.price_factor_of(root) -> int`. (e) The conversion keeps both upstream
guards (the remainder test and |fixed / 1e9 - price| <= 1e-6). (f) Paths are resolved through symlinks before the
checks; any part named sealed, vendor, derived or superseded is refused. (g) The gateway's `session` argument is
a plain CT clock filter (ALL or a ClockWindow); the SPEC names "globex" and "rth" are resolved in pfb.engine.run
with `session_window_ns`, per trade date.

A6 Kernel accounting. Costs are charged at the fill. realized = the sum of net_cents of
closed trades minus the entry costs of the open position; equity = realized + floor(open gross in 1e-4 USD / 100).
A closing trade books gross_cents = floor(gross / 100) and net_cents = gross_cents - commission_cents -
slippage_cents (both sides), so the last eq_close equals the ledger's sum of net_cents exactly (a test asserts it).
Exposure inside a bar, the same for mae_ticks, mfe_ticks, eq_low and eq_high: a position opened at the bar's open
is exposed to the bar's whole range; an exit at the open (SIGNAL, FLAT_BY) contributes the open only; a STOP
exit: adverse = the stop fill price, favourable = the bar's favourable extreme; a TARGET exit: favourable = the
target price, adverse = the bar's adverse extreme; a SESSION_END exit: the whole range.
The stored in_flatten_window and in_no_new_positions_window flags are not used by the engine (flat-by comes from
A3 only). A stop fills at the worse of stop and open plus the normal slippage of its minute, which is optimistic
for tight stops in fast markets: a strategy with stops is also run under a stress cost model before anyone relies
on it. A roll splice (at UTC midnight, mid-session) forces a SESSION_END exit and so an extra round trip:
conservative, and documented here.

A7 Costs and fees. Cost tables are per MINUTE of the CT day: `slippage_cents(root,
contracts) -> int64[1440, 2]`, `calibrated(root) -> bool[1440]`, built from each upstream bucket's own start_min and
end_min; a minute in no bucket takes the fallback of section 6 (the root's largest per-side slippage) and is uncalibrated (12 upstream buckets are shorter than 30
minutes; `ct_minute // 30` is not used). The book expression of section 6 is the only float in a money path: it
runs when the table is built, is pinned to upstream and is tested cent for cent.
Fee files: one [[fee]] per root with root, commission_side_cents, commission_rt_cents, source_file, lookup_date;
cents = Decimal(str(usd)) x 100, which must be whole or the build raises; `FeeSchedule.entry(root)`, `.roots()`.
A root that the firm's fee file or the book table does not list (15 roots have no fee row; MES and ES have no
book row) cannot run with that firm or that model: the run is refused, never defaulted.
The book table has no date axis (a few sample dates): the summary of a run outside the table's sample range
warns of it. Event windows are off: every record and summary carries event_windows = "off" with a warning, and a
hypothesis file must state event_costs = "off" to be registrable until the release calendar is ported.

A8 Metrics. max_drawdown_cents is on eq_close. max_drawdown_intrabar_cents = the maximum over t of
(the maximum over s <= t of eq_high[s]) - eq_low[t], the worst case since the order inside a bar is unknown.
Calmar = annualized net / max_drawdown_intrabar_cents, annualized net = mean daily net x 252. A Sharpe with zero
sd is null (upstream returns 0). The headline DSR uses V = max(1 / (T - 1), the cross-configuration population
variance); both are reported. N is the registry's N after the run's own begin record.

A9 Trade date and clock. In a frame the trade date is the polars Date column trade_date. In every numpy
array, kernel input and sessions call it is trade_day: int32 days since 1970-01-01 (numpy datetime64[D]), -1 for
none; the strategy's bars view exposes trade_day, not trade_date. ct_minute comes from polars' time-zone
conversion of the integer timestamp (tested equal to zoneinfo across DST changes).

A10 Registry. Kinds: genesis, sync, hypothesis, trial, engine_check, synthetic, result. A run has one
begin record of its kind and at most one result record with the same run_id. Fields added to section 13's list:
status, engine_checks_sha256, event_windows, check (the allowlist entry), warnings. Dirty tree = any line of
`git status --porcelain --untracked-files=all` other than ledger/registry.jsonl; a trial's strategy and
hypothesis files must lie inside the repository, so a clean tree means they are committed. Only a trial refuses
a dirty tree; engine_check and synthetic records carry git_dirty. The budget is cumulative over the hypothesis.

A11 CLI (section 14). `pfb run <strategy.py> --kind engine_check --check-id ID --json` and `pfb run <strategy.py>
--kind synthetic --json [--root R --seed N --fees F --cost-model S]`. Both call pfb.runner.run and print the
summary of the result record. `--declared-in` does not exist (the allowlist entry carries it).

A12 Parity candidates add: interface v0 cannot see fills, so "one entry per trade date" cannot react to a
dropped entry.

A13 Rulings added to section 15, each reversible. 12. N rises at begin, so a crashed run still
counts. 13. A real-data engine check must be on the allowlist. 14. The MES confirmation era is refused until upstream's
closure rulings are ported. 15. Unsourced hist dates are dropped and counted, not refused. 16. The headline DSR
uses the larger variance. 17. A hypothesis must state event_costs = "off". 18. The exposure rule of A6 (worst
case on stop and target bars).

Not applied: N3 (eq_high floors toward minus infinity, under one cent).

A14 Rulings after the design recheck (G1-G11 are its
ids). They bind like A1-A13.
(a) G1, G2 Committed inputs. `pfb.registry.api.begin` refuses a trial unless its strategy and hypothesis files,
and an engine_check unless config/engine_checks.toml and the entry's strategy file, are tracked and unchanged
against HEAD: `git ls-files --error-unmatch <path>` succeeds and `git diff --quiet HEAD -- <path>` exits 0 (error
InputNotCommitted). An engine_check's declared_in file must exist at HEAD (`git cat-file -e HEAD:<path>`). This
adds to A10: still only a trial refuses a dirty tree as a whole, and a synthetic run needs none of it.
(b) G6 One writer at a time. ledger/registry.jsonl is only ever appended to, never rewritten or renamed. Every
append (genesis, sync, hypothesis, begin, finish) takes `fcntl.flock(fd, LOCK_EX)` on that file itself (a lock
file would dirty the tree), then re-reads and verifies the tail, derives the previous hash, n_before, n_after and
the budget already used INSIDE the lock, appends, fsyncs and unlocks. A test lets two processes append at the
same time and then verifies the whole chain.
(c) G3 Entry and exit in one bar (A6). The exit's rule decides both marks. STOP: adverse = the stop fill price,
favourable = the bar's favourable extreme. TARGET: favourable = the target price, adverse = the bar's adverse
extreme. SESSION_END: the whole range. "Exposed to the bar's whole range" holds only for a bar the position
stays open through. Tests: one hand-computed case per exit reason for eq_low, eq_high, mae_ticks and mfe_ticks.
(d) G4 Eras. `bars()` over several eras returns the rows of every selected file joined in read order (era order,
then first trade date, then path), as the gateway already does. ts must be strictly increasing over the joined
frame, or the read raises (nothing is de-duplicated silently). New: `DataGateway.files_for(root, first, last) ->
tuple[CatalogEntry, ...]` returns, without reading a bar, exactly the entries bars() would read, in read order
(bars() itself selects through it). pfb.runner takes the data hashes of A1 step 2 from it, and after the run the
provenance must equal it in path and sha256. In section 8, session_end[i] is also true when era[i + 1] != era[i]
(the frame's era column): an era change forces a flat position, like an instrument_id change. A synthetic frame
carries one era.
(e) S1 The real-store label test of A3 also covers every usable open row of each file whose pe_pin_status is
"none" (it reads through the gateway; a mismatch is reported, never patched).
(f) Notes. G5: a module attribute (np.X, pl.X, math.X, under any alias) must be on that module's list; any other
attribute name must be on the union of the numpy and polars lists. G7: the artifacts in a result record are
planned paths; `pfb registry status` lists a begun run without a result as "open or crashed". G8: the
cross-configuration variance of A8 is over every configuration of the same hypothesis with an "ok" result
record, this run included; with fewer than two, and for an engine check, only the floor 1 / (T - 1) applies. The
caller passes the grid's daily Sharpes in (`grid_daily_sharpes`) and `summarize` computes the variance: pfb.metrics
never reads the registry. G9: Decimal(str(usd)) x 100 must be whole; a
side derived from a round turn alone is ceil(rt_cents / 2), as in section 6. G10 (reversed by A15 (f)): the
upstream-side parity run is logged as an engine_check under an allowlist entry of its own (id E3_PE, strategy = the
tracked parity script). G11: pfb.runner is an eighth package beside the seven of section 1. S7 (whether a trial may run with
event costs off) was settled later by docs/TRIALS.md and section 17.
(g) Added to the rulings of A13, each reversible: 19. A trial and an engine check need their input
files committed. 20. An era change forces a flat position. 21. A stop or target in the entry bar uses the exit's
marks.
(h) A5 (d) for a file without a metadata tick (only a few store files have none): the metadata comparison is skipped and the
price-grid guards still run; the provenance record carries tick_metadata = "absent" and a warning names the file;
a local real-store test pins which files these are.
(i) Metrics interface (ruled after review). The trade ledger's columns are defined once, in
pfb.metrics.core.TRADE_COLUMNS; the engine imports it and a test holds the engine's ledger to it. mae_ticks and
mfe_ticks are non-negative magnitudes. `summarize` raises when the last eq_close or the daily net differs from
the ledger's sum of net_cents: it never reports on inconsistent input. Expectancy in ticks is gross, because
pfb.metrics does not know the tick value. Drawdowns start from equity 0 before the first bar. There is no scipy:
the Student-t tail is our own code, pinned by tests.
(j) Costs (ruled after review). `event_slippage_cents` raises EventCostsOff (A7: event windows are off). The
stress model rounds up to whole cents on fractional-cent roots: at most one cent per side more than upstream, never in
the strategy's favour. A root outside the instrument table has no fee row (MET today), so it cannot run.
(k) Sessions (ruled after review). Hist era, every group: must-be-flat ALSO covers every same-date gap of
the calendar, [end of the segment before the gap - 2 min, start of the next segment), in addition to the literal
Upstream rule of A3. Reason: for grains from 2010-06 to 2012-05 the CME pause was 07:15-09:30 CT, the constant
07:43-08:30 held no bar, and a position could be carried through the closure. The stricter side effects of the
literal rule on old regimes stay (grains 18:00-19:00 CT, livestock before 08:30 CT): conservative. The cme2019
era is upstream's rule unchanged. A closure minute beyond the close keeps trade_day -1 (upstream books it to the next
session; the engine removes such bars). `session_window_ns` for "rth" is the day session clipped to the trade
date's globex span, so an early halt cuts it. From a dated rule change every weekend MBT bar is must-be-flat, as in upstream.
(l) Registry (ruled after review). With an upstream configured, the sync rule ("before any append") is applied
literally: a change of upstream's ledger blocks every append, `finish` included, so a run that upstream overtakes
stays "open or crashed" until a sync (docs/TRIALS.md later lets finish proceed). A line torn by a crash is not
repaired: the ledger fails closed until it is fixed by hand. The genesis record holds upstream's two ledger files
byte for byte.
(m) Engine (ruled after review). TRADE_COLUMNS also holds exit_reason, trade_date and the entry and exit bar
index, as section 9 requires. G5 is read strictly: an attribute name gets no "defined in the file" exemption,
and a builtin that is not on the list can be neither shadowed nor aliased. On a bar that gaps beyond a target,
the open is a real mark of the open position and counts toward eq_high and mfe_ticks.

A15 Rulings after the first parity phase (2026-11-15). The parity items, (c) and (f) to (i), describe development
runs that needed upstream; they are kept as the record of how the engine was checked.
(a) Look-ahead. The automatic check (`lookahead_check` in src/pfb/engine/lookahead.py, called by the runner
before files_for and before begin) runs the 20 seeded random cuts and then cuts right after signal bars (entry
or exit flag), at most 200 per run: all of them when fewer, else evenly spaced with the first and the last.
(b) A paths file that exists but cannot be read fails loudly; it never skips.
(c) The parity script stays ONE file although it exceeds the 800-line style limit: the parity report pins the
script by one sha256.
(d) E3 keeps fees "topstep" and cost_model "book": on MNQ, with upstream's table, they equal upstream's costs cent for cent.
(e) No ledger record may say that lint or look-ahead passed when the check did not run.
(f) G10 of A14 is REVERSED by (e). There is no E3_PE record and no fifth allowlist entry. The upstream-side run is
part of the ONE engine check E3, whose lint and look-ahead really ran on strategies/reference_breakout.py. E3
runs through pfb.runner first; the upstream-side gateway read follows and must show the same (path, sha256) pairs as
E3's begin record. Real mode refuses unless the script, the reference strategy and config/engine_checks.toml are
tracked and unchanged against HEAD. The parity report names both repositories' HEAD, the script's sha256 and
E3's run_id.
(g) Parity classes fixed before the real run (section 10): pe_min_hold (upstream's exit one minute after ours when our
exit was decided within 2 minutes of the entry fill) and missing_minute_flat_by (upstream's forced exit 60 s after
ours, same price and net). The new engine adopts neither convention.
(h) The window's first trade date is outside the cent match (pe_first_day_no_settlement): Upstream's price-limit rule
needs the settlement of the day before the window and refuses every entry that day. The class applies only on
Upstream's own evidence of the refusal. Every other difference is unclassified, fails the run and goes to review.
(i) Ruled after the first real E3 run: the fourth class pe_price_limit_exit: same side, entry time,
entry price and quantity, equal commission and entry-side slippage, upstream's exit_reason exactly PE_PRICE_LIMIT,
Upstream's exit earlier than ours, our exit reason SIGNAL at the open of that bar. The final run asks upstream twice on the
same exported bars, in one upstream subprocess and with no second gateway read: OFF (rules.price_limits.
HARD_LIMIT_PRODUCTS emptied inside our subprocess only, restored after; nothing is written in upstream) and ON (upstream as
it is). Comparison OFF: every trade equal to the cent on every field, only pe_min_hold and
missing_minute_flat_by may take a pair out. Comparison ON: all four classes are admitted, but a pe_min_hold or
missing_minute_flat_by pair must be the same pair in the same class on OFF, and pe_first_day_no_settlement and
pe_price_limit_exit counts must equal upstream's own counters (engine_price_limit_reference_unavailable,
price_limit_exit). Verdict PASS needs both comparisons; the report gives both (the parity script's
CLASSES, ON_CLASSES).

A16 Later rulings. They bind like A1-A15.
(a) Flat-by over an empty interval: when bar i lies before a must-be-flat interval's start and bar i + 1
at or after its end, the interval holds no bar and bar i ends the session (src/pfb/engine/run.py flat_gap_ends, ORed into session_end; counter
session_ends_at_flat_gap). The position closes at that bar's close as SESSION_END; an entry decided there does
not fill (entry_signals_at_session_end). Signals that cannot act at a session end are counted
(entry_signals_at_session_end, exit_signals_at_session_end) and are part of signals_ignored. Engine version pfb-engine/0.2.0.
(b) Ledger anchor: while HEAD tracks ledger/registry.jsonl, its bytes at HEAD must be a byte prefix of the
working file before every append and in status (src/pfb/registry/chain.py require_extends, gitstate.py
head_blob), else LedgerAnchorBroken and nothing is appended; genesis is refused while HEAD tracks a ledger. No
anchor outside the repo or without HEAD. LIMIT: lines not yet committed are not anchored.
(c) Code identity: every begin record of every kind carries code_sha256, one sha256 over every *.py under
src/pfb (src/pfb/registry/codehash.py code_canonical_bytes), and it is part of config_id for every run kind
(src/pfb/registry/records.py), so an exact replay that adds nothing means same inputs AND same code; a replay
under new code counts as a new trial. The schema string stays pfb.registry/1: a begin record without
code_sha256 is accepted only before the first begin record that carries one (the three lines of 2026-11-15).
(d) Engine checks are bound: each entry of config/engine_checks.toml declares session, seed and params;
begin refuses (EngineCheckNotDeclared) an engine_check whose session, seed or params differ from its entry
(canonical JSON, so 1.0 != 1). The four entries carry session "globex", seed 0 and the SPEC defaults of
strategies/reference_breakout.py.
(e) Pinned window table: the gateway's constructor reloads config/closed_windows.toml itself, refuses a
table whose sha256 field is not the pin (src/pfb/data/guards.py pinned_window_table), and checks with its own
table, never a caller's object. A configured upstream root counts as a real store location.
(f) Look-ahead (E1-E4), replacing A15 (a) where they differ: every generate call runs the linted code in a
fresh module with read-only copies of the bars and frozen params (src/pfb/engine/strategy.py
IsolatedGenerate). Order of calls: random cuts and the sweep cuts ascending, then the call on n-1 bars (it
chooses the signal cuts), then up to ENTRY_BAR_CUTS = 200 cuts after entry bars and EXIT_BAR_CUTS = 100 after
exit-only bars, the full call last; each output must equal the next longer one on its prefix. At step 1 the
check also cuts after every bar of SWEEP_SESSIONS = 5 sessions spread evenly (about 3 s on a month). The run
itself repeats the check without the sweep and trades the checked full output (about 0.1 s on a month). The
lint refuses every attribute store, item stores into non-local names, mutable defaults and uninitialized
arrays, and every load of the name ndarray outside annotations (an alias of
numpy.ndarray hands out uninitialized memory holding freed copies of longer bar arrays). When the full output
has no entry, the warning "look-ahead check saw no signal" goes into the result's warnings.
LIMIT (kept for cost): the in-run check has no sweep, so a strategy that peeks only on bars
unlike step 1's synthetic bars, and only to suppress some entries, is caught in the run only when a random cut
lands right after such a bar. Open.
(g) Real stores need the repo ledger: pfb.runner.api.run refuses, before begin, a run whose store roots
overlap a real store location (src/pfb/data/guards.py real_store_locations) while its ledger is not the repo's
ledger/registry.jsonl. Kind synthetic opens no store and is not checked.
(h) config/calendars/*.json entered no run record and no config_id at this point; docs/TRIALS.md
C1 adds the calendars sha256 to config_id.

## 17. Amendment of 2026-11-17: a hypothesis may state event_costs "on". This section wins over section 6 and over ruling 17 of section 15.

Section 6 and ruling 17 said a hypothesis must state event_costs = "off" until the release calendar was ported.
pfb.costs.events now reads a frozen release calendar (schema pfb.release_calendar/1), pinned by sha256 in
config/costs.toml [events]. Public copy: the repository ships a small fictional calendar,
config/demo/release_calendar.json (2025-01-01..2026-06-30); config/paths.toml [costs] release_calendar may name
your own. Since then a hypothesis states event_costs "off" or "on":
- "on" needs a book-based cost model (book, or stress over book). A market fill whose bar opens in
  [release, release + 30 min) of a release concerning its root pays the book's event table; a micro root, and
  the mini-sized QM and QG, are charged by the union of their own releases and their full-size
  sibling's. The cost model id is <model>+events, its config sha256 covers the calendar's sha256, the window
  and the sibling pairs, and records and summaries carry event_windows = "on".
- "off" stays valid and unchanged: event_windows = "off" with the warning that event-window costs are not
  modelled, and a non-empty event_stance.
- A declared window the calendar does not cover is refused at registration and before begin. Every run of a
  hypothesis (neighbours, book members) uses the hypothesis's value; there is no switch.
- Not ported under "on": Upstream's 2-minute fill guard and the CPI opening rule (the "on" warning names both), and
  the pre-2019 rows (pfb.costs.events).
The rules in force are docs/TRIALS.md sections 2 and 4 and AGENTS.md; this note only lifts the older sentence.
