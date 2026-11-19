# AGENTS.md: a guide for contributors and AI coding agents

pfb is a futures backtester with prop-firm economics. Every configuration evaluated on real data is a counted
trial in ledger/registry.jsonl (one global N). This page is the quick start; the binding rules are
docs/TRIALS.md, the result keys docs/result.schema.json, the suite docs/ANALYSIS_SUITE.md. The CLI, the MCP
server (`.mcp.json`, `python -m pfb.mcp`) and the dashboard all call one service layer, `pfb.app`.

## Setting up

    uv sync                       # Python 3.12, dependencies from uv.lock
    uv run pfb demo               # synthetic demo store, check, run and prop MC; N unchanged
    uv run pfb demo --dash        # then the dashboard on http://127.0.0.1:8501
    uv run ruff check . && uv run pyright && uv run pytest -q   # what CI runs

Without config/paths.toml every command uses the synthetic demo store (data/demo/, built by `pfb demo`) and
the fictional cost tables in config/demo/. To use your own licensed bar data, copy config/paths.example.toml to
config/paths.toml (git-ignored), point it at your stores and run `pfb data catalog`. Never commit market data,
anything derived from it row by row, runs/ or data/; a hygiene test fails on tracked parquet, dbn or npy files.
An MCP client finds the server in .mcp.json; its tools are list_instruments, list_firms,
describe_strategy_template, validate_strategy, register_hypothesis, run_backtest, run_prop_mc, get_run,
compare_runs and trial_count, each returning the same objects as the CLI's --json.

## Quick start

    pfb new my_orb                                # strategies/my_orb.py from strategies/_template.py
    pfb check strategies/my_orb.py                # lint + look-ahead check on synthetic bars; logs nothing
    pfb run strategies/my_orb.py --kind synthetic --json        # synthetic bars, N unchanged
    git add strategies/my_orb.py hypotheses/h_my_orb.toml && git commit   # a trial's inputs must be committed
    pfb hypothesis register hypotheses/h_my_orb.toml            # schema "pfb.hypothesis/2"; N unchanged; --json prints the record
    pfb run strategies/my_orb.py --kind trial --hypothesis hypotheses/h_my_orb.toml --json   # N + 1
    pfb prop <run_id> --firm topstep --plan combine_xfa_standard --size 50k --mc 20000 --seed 1 --json
    pfb registry status                           # N, open runs, hypotheses

A strategy is one file: SPEC (root, bar_minutes, session, contracts, params with default and grid) and
`generate(bars, params)` returning entry_long, entry_short, exit (bool) and stop_ticks, target_ticks (int64
ticks). A signal on bar i fills at the open of bar i + 1; use bars 0..i only. A hypothesis file pins the
strategy's sha256, roots, windows, grid, budget, cost model and event costs, power inputs, [suite] and [secondary]
(TRIALS.md section 2), and may set `lookahead_seed` (int >= 0, default 0): the seed of a trial's step-1
look-ahead check. A trial runs at it; a `--seed` (MCP/app `seed`) that differs is refused (SeedMismatch).
The program look-ahead seed is 0 (`pfb check` uses it); another lookahead_seed needs a non-empty
`lookahead_seed_reason` (LookaheadSeedUnexplained), and a `pfb check --seed` other than 0 is marked as
not the program seed. At most one purpose "test" hypothesis may be open (budget not yet counted) at a time
(OpenTestHypothesis).
purpose "test" takes budget 1 only (PurposeTestBudget); two windows may not share a day (WindowsOverlap;
a window ending on day d and one starting on d + 1 are fine). Synthetic runs choose suite parts with
`--no-stress --no-baseline --no-walk-forward`, `--neighbourhood`, and a prop plan with
`--firm --plan <file stem> --mc`; a trial takes all of them from its hypothesis and refuses those switches.
`pfb prop --copies N` (one plan; refused with the matrix and on a trial, TrialNotDeclared) runs N accounts that
copy the signal beside N independent ones on the same day paths. Its row gains an optional "copies" object
(n_accounts, copies, independent, not_modelled), and the artifact
runs/<run_id>/prop/<firm>_<plan>_<seed>_copies<N>.json shows, for both columns, P(all pass), P(all fail), P(any pass), the correlation of two accounts and the
mean, SD and quantiles of the summed payouts and net. The document names what it leaves out: the account
count limit, copy-trading and aggregate contract cap rules (no plan field holds them), resets per account,
staggered starts and per-account fills. It is descriptive, never a trial output, and adds nothing to N. The
MCP tool `run_prop_mc` takes `copies` (int, default null) the same way, with the same refusals.

`--json` prints one object and nothing else on stdout. A refusal is
`{"schema": "pfb.result/1", "status": "refused", "error": {"code", "reason"}}` with exit 2 (and
`refused: <code>: <reason>` on stderr); an unexpected error after begin has status "error" and exit 1.

A book (schema "pfb.hypothesis/3", docs/TRIALS.md section 8) pools K >= 2 member configurations into one
account; budget = K. Register it like any hypothesis (N unchanged); `--dry-run` lists every refusal and
appends nothing. Run each member as its own trial with its params (and `--window K` over the distinct
member windows when there are several): N + 1 per member, a non-member is refused. Its [power] is the
BOOK's and may never be more flattering than the members derive (PowerFlattered). Members are not
assumed independent: members on one root (MNQ and NQ count as one), or on one strategy file, whose windows
overlap move together, so their SD adds linearly; [power] must state member_correlation (0 to 1, across
the remaining groups) and a non-empty member_correlation_reason. State trade_sd_cents at or above the
derived figure rounded up to the cent (the dry run's PowerFlattered message prints it).

## The result contract ("pfb.result/1"; each run's runs/<run_id>/result.json)

| key | meaning |
|---|---|
| schema, run_id, kind, status | "pfb.result/1"; the run id; trial, engine_check or synthetic; ok, failed or open |
| hypothesis_id, purpose | the SPEC's or hypothesis's id; a trial's purpose (research or TEST), else null |
| strategy, root, window, params | repo-relative strategy path; the root; [first, last] (null for synthetic); the params |
| n, n_added | N right after this run's begin, not the live N (`pfb registry status`); `pfb prop` on an old run repeats that old n; what this run added (1 for a new trial configuration, else 0) |
| net_cents, trades, win_rate | net P&L in cents; number of trades; share of winning trades |
| profit_factor, expectancy_cents | gross wins / gross losses; mean net per trade (cents) |
| sharpe, sortino | annualized daily Sharpe and Sortino |
| max_drawdown_cents | closed-bar peak-to-trough drawdown |
| psr, dsr, minbtl_years | probabilistic and deflated Sharpe (N trials); minimum backtest length |
| cost_stress | {"k": {net_cents, sharpe}} with k extra ticks per side (S1) |
| baseline | random-entry baseline (S2): draws; `mode`, the one mode the four figures come from (of uniform, tod and cost_matched the one with the largest p_net, ties to cost_matched then tod; older records have tod and uniform only); net_percentile and sharpe_percentile, fractions in [0, 1] (not percents); p_net and p_sharpe, one-sided p |
| power | stated min edge (trial, over the trial's own window), its alpha-only twin, the target, realized minimum and edge, and passed: the REALIZED test (realized edge >= realized minimum; false when undetectable, null when not computable), not the begin gate, which every trial passed (S3) |
| walk_forward | folds, Sharpe spread across folds, folds with net > 0 (S4) |
| neighbourhood | verdict (plateau, spike, mixed, n/a), mean neighbour / centre Sharpe, neighbour run ids (S5) |
| prop | rows: firm, plan, p_pass and SE, EV per purchase and SE (cents), status, n_paths, seed, soft_daily_loss, plan_version, effective_from (the firm's own in-force date of that rules version), plan_file_sha256, rules_in_force (these four null on a skipped row; absent on rows written by older versions) |
| warnings, artifacts | every warning; the files written (repo-relative) |
| artifact_sha256 | the result record's map of each of its artifacts to the sha256 of the file written (null for an open run or an older record) |
| ledger | sha256 of the run's begin and result lines in the ledger |

A part turned off or not applicable is null, never absent. Every value comes from the run's ledger records
or its artifacts (pfb.contract docstring lists each source). Warnings to read: a realized minimum edge above
the stated one, or None while one is stated (with the reason); every draw of the reported baseline mode
equal (draw sd 0: the percentile says nothing).
A prop record's warnings starting "rule unknown, assumed:" name a rule value the firm does not publish that the
run assumed (the figure holds only under it); "rule not modelled:" names a firm feature the figure leaves out.

## Worked examples (synthetic data; real output from this tree, trimmed)

Produced in a fresh scratch copy of this tree whose ledger held only its genesis line (N 0) and whose runs/
directory was empty, so no number comes from any earlier ledger. Run ids and timestamps differ on every run.

1. A synthetic run with the default suite:

        $ pfb run strategies/_template.py --kind synthetic --cost-model fixed:1 --json
        {"run_id": "synthetic-20261119T164951964135Z-eb5d1ae4", "kind": "synthetic", "status": "ok",
         "n": 0, "n_added": 0, "net_cents": 1110, "trades": 20, "win_rate": 0.45, "sharpe": 0.3863,
         "psr": 0.5436, "dsr": 0.5436, "max_drawdown_cents": 9476,
         "cost_stress": {"1": {"net_cents": -790, ...}, "2": {"net_cents": -2690, ...}},
         "baseline": {"draws": 200, "mode": "cost_matched", "net_percentile": 0.72, "p_net": 0.2836, ...},
         "power": {"min_edge_cents": null, "realized_edge_cents": 55, "realized_min_edge_cents": 1597,
                   "passed": false, ...},
         "walk_forward": {"folds": 5, "positive_folds": 3, ...}, "neighbourhood": null, "prop": null, ...}

   Read it as: 20 trades are far too few for the realized edge (55 cents) to reach the realized minimum
   (1597 cents; passed false); one extra tick per side turns the net negative. The baseline reports mode
   cost_matched because its p_net (0.2836) ties uniform's and is above tod's (0.2388), and ties go to
   cost_matched; 0.72 is a fraction of draws, not a percent. At N 0 the DSR equals the PSR (nothing to
   deflate for).

2. Its neighbourhood (each neighbour is its own run with its own result.json):

        $ pfb run strategies/_template.py --kind synthetic --cost-model fixed:1 --neighbourhood --no-baseline --json
        {..., "baseline": null, "neighbourhood": {"verdict": "spike", "ratio": -0.2341,
         "run_ids": ["synthetic-...-be16a229", "synthetic-...-1f3bd1db", "synthetic-...-3e6aba06"]}, ...}

3. The prop Monte Carlo of a saved run:

        $ pfb prop synthetic-20261119T164951964135Z-eb5d1ae4 --firm topstep --plan combine_xfa_standard \
              --size 50k --mc 2000 --seed 1 --json
        {..., "prop": [{"firm": "topstep", "plan": "combine_xfa_standard_50k", "status": "ok",
         "p_pass": 0.0, "p_pass_se": 0.0, "ev_per_purchase_cents": -76200.0, "ev_se_cents": 0.0,
         "n_paths": 2000, "seed": 1, "soft_daily_loss": "limit", ...}],
         "artifacts": [..., "runs/synthetic-...-eb5d1ae4/prop/topstep_combine_xfa_standard_50k_1.json"]}

   And a refusal: `pfb run ... --kind trial --hypothesis hypotheses/h_template.toml --no-stress --json` prints
   `{"error": {"code": "SuiteMismatch", "reason": "a trial's suite and prop outputs are fixed by its
   hypothesis; leave ['stress'] at their defaults"}, "schema": "pfb.result/1", "status": "refused"}`, exit 2.

## Rules

- Never read a closed window: every bar read goes through the DataGateway; never open a bar store, a sealed
  store or a raw .dbn file yourself. A hypothesis whose window touches a closed window is refused.
- Every trial counts: a trial's begin raises N before the engine runs, and a crash still counts. Use
  synthetic runs to develop; run a trial only for a registered hypothesis, inside its grid and budget.
- The suite is fixed by the hypothesis: cost stress, baseline, walk-forward, neighbourhood, sizing and the
  prop plans are declared before any trial and are outputs of the one configuration (TRIALS.md C4).
- Event costs: `event_costs = "on"` in a hypothesis's [cost] (book-based models only) prices every fill
  inside a release window from the pinned release calendar; the run's cost model id is <model>+events,
  the result's event_windows is "on", and the warning names what is still not modelled. Every run of
  that hypothesis (neighbours and book members included) runs "on"; there is no switch. A window the
  calendar does not cover is refused at registration and before begin. Under "off" the result carries
  the warning that event-window costs are not modelled.
- Rule C5 (an engine_check strategy is never a hypothesis's) compares file sha256s: a one-byte edit gives a
  new hash and passes it. The control is the review of config/engine_checks.toml, not the hash.
- No result is read to pick a configuration: no prop figure, suite figure or comparison (`compare_runs`)
  chooses, adds or orders configurations. The DSR deflates for N; report N with every figure.
- `pfb registry verify` hashes every run folder against its result record; `pfb prop` refuses a folder
  that does not match (ArtifactMismatch) and, on a trial, a --ledger or --firms-dir other than the
  repository's (TrialInputsNotRepository).
- Bar stores are read-only to pfb: nothing writes into a store, and only the gateway opens one. No paid data
  API call, no broker code and no credentials in the repository; keys stay in a git-ignored .env.
- The ledger is append-only. Synthetic runs (the demo included) append records but add nothing to N; reset a
  working copy with `git checkout ledger/registry.jsonl`. `pfb registry genesis` starts a ledger at N 0; an
  optional [upstream] root in config/paths.toml (an earlier research checkout with an append-only trial ledger,
  ledger/trial_registrations.jsonl and ledger/trial_amendments.jsonl) counts that ledger's trials into N, and
  `pfb registry sync` imports its new lines; without it, nothing is imported.

