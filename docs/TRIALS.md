# Counted trials: the binding rules

Written 2026-11-16. Extends ADR-001 sections 13, A1, A2, A7, A10 and A14; where the two
differ, this file holds. The ledger schema string stays "pfb.registry/1". Money in integer cents.

## 1. What counts in N

- C1 A configuration is the tuple hashed by config_id: hypothesis sha256, strategy sha256, params, root, window,
  session, cost model id and config sha256, fees sha256, instruments sha256, engine version, data-file hashes,
  code sha256 and, new here, calendars sha256.
- C2 One trial begin record names exactly one configuration. N rises at begin, by 1 when the config_id is new to
  its hypothesis and by 0 on an exact replay. A crash after begin still counts and leaves the run open.
- C3 Neighbourhood runs (S5) are trials: each neighbour is its own begin record, inside the declared grid and
  the budget. Before anything is begun the runner plans the whole set; if the budget cannot cover the
  configurations not yet counted, or any planned configuration would be refused for a look-ahead check that saw
  no signal, the whole request is refused and nothing is begun. The neighbours are begun and run first and the
  centre last, so the centre's N and DSR grid include its own neighbourhood.
- C4 Cost stress, the random-entry baseline, breakdowns, walk-forward folds, the sizing table and the prop matrix
  are outputs of the one configuration. The hypothesis fixes them ([suite], [secondary]) before any trial; no
  command-line switch changes them for a trial. They add nothing to N. No function takes one of them as an
  input to choosing, adding or ordering configurations; selection walk-forward reads counted runs only.
- C5 Kinds: engine_check (allowlisted, real data, SPEC defaults, N unchanged), synthetic (no data, N unchanged),
  trial (real data, counted). A strategy sha256 pinned by a registered hypothesis is refused as an engine_check,
  and a hypothesis whose strategy sha256 equals that of an earlier engine_check begin record or of an allowlist
  entry is refused at registration (replay checks the ledger half of both).
- C6 Order of a trial: lint, load, the parameters resolved against the SPEC grid (a value outside it is
  refused here, before begin) and look-ahead on synthetic bars; resolve hashes; checkpoint if the ledger has
  lines not at HEAD; begin (N); checkpoint; engine plus suite under the permit, one bar read; result record;
  artifacts; secondary outputs from the artifacts; checkpoint at the end of the invocation.

## 2. Hypothesis file, schema "pfb.hypothesis/2" (hypotheses/<id>.toml)

Exactly these keys (missing or extra refused), plus two optional top-level keys: `lookahead_seed_reason`
(below) and `lookahead_seed` (int >= 0; default 0): the seed of a trial's step-1 look-ahead check. A trial given another seed is refused
(SeedMismatch) before step 1 and again at registry begin; the switch `--seed` stays for other kinds.
One program-wide look-ahead seed, 0 (pfb.registry.hypfile.PROGRAM_LOOKAHEAD_SEED): `pfb
check` and every default use it. A lookahead_seed other than 0 needs the optional top-level key
`lookahead_seed_reason` (a non-empty string; schema 2 and 3), else the file is refused
(LookaheadSeedUnexplained). The hypothesis record stores both (tier 3) and replay checks them and a
trial's seed against them (SeedMismatch); records written before this key existed store no seed and are not checked.
`pfb check --seed k` with k != 0 is marked "not the program seed" and stands for no trial's check.
Schema 1 files are refused; none was ever registered.

    schema = "pfb.hypothesis/2"
    id = "h_example"                  # [A-Za-z0-9][A-Za-z0-9_.-]{0,63}
    purpose = "research"              # or "test": a plumbing check, never read for selection; budget 1
    statement = "..."
    strategy = "strategies/x.py"      # repo-relative, committed at HEAD
    strategy_sha256 = "<64 hex>"
    roots = ["MNQ"]                   # one or more; each trial runs one root
    bar_minutes = 1                   # must equal the SPEC's
    session = "globex"                # must equal the SPEC's
    budget = 3                        # distinct configurations, cumulative, neighbours included
    no_signal_reason = ""             # non-empty: why step 1 of the look-ahead check may see no signal
    [[window]]                        # one or more; a trial's first..last equals one of them exactly
    first = 2017-03-06
    last = 2020-09-18
    [grid]                            # every SPEC parameter; numeric lists strictly increasing
    buffer_ticks = [2, 4, 6]
    [cost]
    model = "book"                    # book, fixed:<k> with k > 0, or stress:<spec>:<k>; zero slippage refused
    fees = "topstep"
    event_costs = "off"               # or "on": event-window costs per bar from the pinned release calendar
    event_stance = "..."              # required while off: why event minutes do not matter here (may be empty under on)
    [power]
    trades_per_year = 220.0
    trade_sd_cents = 21000            # per-trade net standard deviation at SPEC contracts
    trade_skew = 0.0                  # per-trade net skewness (required; negative for small-win, large-loss)
    trade_kurtosis = 3.0              # per-trade net raw kurtosis (required; 3 = normal, never below 1)
    sd_source = "stated"              # or "run:<run_id>" of a synthetic or engine_check run in the ledger
    target_edge_cents = 3000          # the net edge per trade the hypothesis claims
    [suite]
    cost_stress_ticks = [1, 2]
    baseline_draws = 200              # >= 200
    baseline_seed = 7
    walk_forward_folds = 5            # >= 2
    neighbourhood = true
    sizing_plan = "topstep/combine_xfa_standard_50k"   # firm/plan whose cap and drawdown S7 uses, or ""
    [secondary]
    prop_plans = ["topstep/combine_xfa_standard_50k"]  # or ["*"] for the matrix, or [] for none
    prop_mc = 20000
    prop_seed = 20261116

Registration (`pfb hypothesis register <file>`) refuses: a file or strategy not committed at HEAD; a strategy
hash mismatch; roots, bar_minutes or session unlike the SPEC's (roots may add to the SPEC root); a grid that
is not exactly the SPEC's parameter names, or that holds a value outside the SPEC parameter's own grid; an
sd_source "run:<run_id>" whose run is not an ok synthetic or engine_check run with realized power figures, or
whose realized per-trade sd is above the stated trade_sd_cents, or whose realized trades per year are below
the stated trades_per_year, or (synthetic source) whose strategy sha256 is not the hypothesis's (replay checks
all of these); budget above roots x windows x grid points; a window that touches a
closed window for any root (WindowTable.request_refusal) or for which the catalog names no data file; two windows that share a day
(WindowsOverlap; every window applies to every root; inclusive, so d and d+1 may touch); purpose "test" with
a budget other than 1 (PurposeTestBudget) (replay applies both rules to the hypothesis record);
a purpose "test" hypothesis while another TEST hypothesis is registered and not spent, spent meaning its
budget counted in N (OpenTestHypothesis; a crash after begin spends; replay applies it to
records written after the rule existed); a cost
model with zero slippage; an empty event_stance under event_costs "off"; a failed power check (section 5, at
N + budget); rule C5; an upstream ledger out of sync (only when one is configured). Registration adds nothing to N.

Event costs: event_costs "on" needs a book-based cost model (book, or stress over book);
registration identifies the model as <model>+events (its config sha256 covers the release calendar's
sha256, the 30-minute window and the sibling pairs) and refuses a declared window (a book: a member
window) that the pinned release calendar does not cover (RunRefused EventCoverageError). The hypothesis
record keeps cost.model the base spec and cost.model_id <model>+events; replay refuses a record whose
event_costs and model_id disagree ("on" exactly when the id ends in "+events"). No switch sets
event_costs: a trial, each of its neighbourhood runs and each book member run at the hypothesis's value.
Coverage is checked by the runner and by the engine; pfb.registry's own functions check the id and
event_windows only, and every entry point goes through the runner. A micro root is charged by the union of
its own release instants and its full-size sibling's (SIL with SI, MNG with NG). The mini-sized QM and QG are in the union as well (QM with CL, QG with NG; their own release lists lie wholly
inside their full-size sibling's), so a "+events" config sha256 covers those lists too.

## 3. Records

- Hypothesis record: the common keys plus hypothesis_id, hypothesis_sha256, hypothesis_path, purpose (stored
  "TEST" for purpose test, else "research"), statement, strategy_path, strategy_sha256, roots, bar_minutes,
  session, windows (list of [first, last]), grid, budget, no_signal_reason, cost (model, fees, model_id,
  config_sha256, fees_sha256, event_costs, event_stance), power (the four inputs, years, n_used,
  min_edge_cents, min_edge_cents_alpha), suite, secondary, closed_windows_sha256.
- Begin records gain a third key tier (the earlier two stay valid as a prefix only, as for code_sha256):
  calendars_sha256 (sha256 of the canonical list of [file name, sha256] over config/calendars/*.json, in
  config_id), lookahead_signal ("seen" or "none"), power (trial: n_used, min_edge_cents, target_edge_cents;
  other kinds null).
- Result records gain begin_sha256, the sha256 of the run's begin line; replay verifies it (older result
  records without it stay valid as a prefix only).
- hypothesis records gain tier 3 (lookahead_seed, lookahead_seed_reason); result records
  gain tier 3, artifact_sha256: each name of the record's artifacts, exactly, to the sha256 of the bytes
  the runner writes under it after the record ({} when there are none). The runner serializes the
  artifacts before the result record and writes exactly those bytes. `pfb registry verify` hashes every
  run folder present against it (exit 2 on a mismatch); `pfb prop` refuses a source folder that does not
  match (ArtifactMismatch). Older records carry no hashes: their folders are unchecked, never matching.

## 4. Refusals at a trial's begin (each also a replay check where the ledger alone can decide it)

Unregistered hypothesis; hypothesis file changed; strategy changed; root not in roots; window not one of the
declared windows; session, cost model, fees or their hashes unlike the hypothesis record; event_windows unlike
the hypothesis's cost.event_costs, or a cost_model_id that ends in "+events" under "off" or does not under "on"
(UndeclaredConfiguration; replay: also ChainBroken for an "on" begin of a kind other than trial); under "on", a
window the pinned release calendar does not cover (refused by the runner before begin, and by the engine
before any bar is read); params outside the grid; budget exceeded; a dirty tree (nothing may differ from HEAD, the ledger included); a ledger not tracked
at HEAD; a configured upstream ledger out of sync; lookahead_signal "none" with an empty no_signal_reason; a failed power check
recomputed at n_used = n_before + budget - configurations already counted; no data file.
finish takes the begin Receipt and is refused unless its record_sha256 is the begin line's (FinishNotPermitted);
finish no longer needs upstream in sync, begin and registration still do.
The engine's RunPermit names event_costs (default "off"); run_strategy refuses a permit whose event_costs
differs from its own argument. An engine error after begin (LockedExitThroughSessionEnd among them) leaves
the configuration counted with a failed result record whose error is the exception's class name.
Checkpoint: `pfb registry checkpoint` commits ledger/registry.jsonl alone ("ledger checkpoint: N <n>, records
<r>", plus the lines of the environment variable PFB_COMMIT_TRAILER when set). The trial runner calls it as in
C6. No other kind commits anything.

## 5. Power check (S3), one pure function used by registration, begin, replay and the suite

Inputs: trades per year m, per-trade sd s (cents), years Y, N_used. Constants: alpha 0.05 one-sided, power 0.80.
Y is the inclusive calendar days / 365.25 of ONE window, never a sum (a trial runs one window):
at registration every declared window must pass on its own and the hypothesis record stores the figures of
the shortest window (the largest minimum edge); at begin and in replay Y is the trial's own window; for a
finished run it is the run's own days / 252.

    T = max(2, round(252 Y));  n = m Y;  b = pfb.metrics.stats.bracket(N_used)
    z = norm_ppf(0.95) + norm_ppf(0.80);  lam = m / 252                  # trades per day
    g3 = trade_skew / sqrt(lam);  g4 = 3 + trade_kurtosis / lam          # daily moments, compound Poisson
    K(x) = sqrt(max(1, 1 - g3 x + ((g4 - 1) / 4) x^2))                   # sd factor of the Sharpe estimator
    s0 = (b + z) / sqrt(T - 1);  k = K(s0);  t_dsr = b + z k             # in units of daily Sharpe x sqrt(T - 1)
    sa = z / sqrt(T - 1);  ka = K(sa);  t_alpha = z ka                   # alpha only: the same with b = 0
    c = t / sqrt(n (T - 1) / T);  edge = c s / sqrt(1 - c^2) if c < 1 else undetectable
    min_edge_cents = ceil(edge at t_dsr);  min_edge_cents_alpha = ceil(edge at t_alpha)

The daily variance is the compound-Poisson one (trades arrive at random), the DSR bar uses the floor variance
1 / (T - 1), so the figure is a lower bound: a wider spread of Sharpe across the grid raises the real bar.
The stated trade skew and kurtosis enter the Sharpe estimator's variance (finding: with normal moments
assumed, trades of skew -2 had 0.74 power, not 0.80; with them stated, 0.81). k never falls below 1; a positive
skew can put the bar slightly under the normal case (2591 against 2601 cents in the worked example), where the
realized power was 0.88. The power test is one-sided: realized power must not fall under 0.80 - 3 SE. A
trial is refused when target_edge_cents < min_edge_cents or the edge is undetectable. Every result also
reports the realized figure (the run's own trade count, sd and days) and warns when the realized minimum
exceeds the stated one.
Alpha rule 2 (2026-11-17). Until this date t_alpha was z k with the gate's k,
taken at the deflated Sharpe s0 (alpha rule 1), so min_edge_cents_alpha rose with N_used and overstated the alpha-only
minimum by 2% to 7% in the go/no-go cases. From this date the alpha figure takes its factor at the alpha
test's own Sharpe sa: it is the gate formula with nothing deflated, equal to min_edge_cents at N_used = 1 and
the same at every N_used. Checked by simulation (PSR(0) >= 0.95, synthetic trades): realized power 0.80 to
0.85 at the new figure, against up to 0.95 at the old one in a one-year skew -2 design. min_edge_cents,
t_dsr and every refusal are unchanged to the cent (tests pin 12 go/no-go checks). The alpha figure is written in one place, the hypothesis record's power object: replay
accepts a record whose figure is the rule 1 one only while no earlier record needs rule 2 (the old rule is
valid as a prefix), so earlier records still verify.
Result contracts recompute the alpha twin, so an older trial's contract now shows the rule 2 figure; a
result record's realized_min_edge_cents_alpha stays as its run wrote it. Not chosen: the full fixed point
S sqrt(T - 1) = z K(S), which is 0% to 4% higher, further from 0.80 in the same simulation and would put the
alpha figure above the gate figure at N_used = 1.

## 6. Prop matrix as a secondary output (T4)

After the result record and the artifacts, the runner computes the plans in [secondary] with the declared mc and
seed and writes runs/<run_id>/prop/. `pfb prop` on a trial run is refused unless plan, mc and seed are the
declared ones, or with a contracts per mini other than the root's own. A run's kind comes from its begin
record (a folder with no begin record is refused); on a trial the root is the begin record's, and a
summary.json naming another root is refused.
On a trial, a --ledger other than the registry's own ledger or a --firms-dir other than the repository's
config/firms is refused (TrialInputsNotRepository). Nothing in pfb reads a prop figure to choose
a configuration.

Suite part errors: a part that cannot be computed (SuiteError, BaselineError, FoldError,
PowerInputError) is null with a warning in the result and its ledger record; the run stays ok. A failed
cross-check of the base run's own trades, days or nets (SuiteIntegrityError) still fails the run.

## 7. Result contract keys ("pfb.result/1"; the schema file is docs/result.schema.json)

schema, run_id, kind, status, hypothesis_id, purpose, strategy, root, window, params, n, n_added, net_cents,
trades, win_rate, profit_factor, expectancy_cents, sharpe, sortino, max_drawdown_cents, psr, dsr,
minbtl_years, cost_stress ({"1": {net_cents, sharpe}, "2": ...}), baseline (draws, mode, net_percentile,
sharpe_percentile, p_net, p_sharpe), power (min_edge_cents, min_edge_cents_alpha, target_edge_cents,
realized_min_edge_cents, realized_edge_cents, passed), walk_forward (folds, sharpe_spread, positive_folds),
neighbourhood (verdict, ratio, run_ids), prop (list of firm, plan, p_pass, p_pass_se, ev_per_purchase_cents,
ev_se_cents), warnings, artifacts, artifact_sha256 (the result record's map, or null). A part that was turned off or does
not apply is null, never absent.

Baseline headline: the four baseline figures come from ONE mode, named in `mode`:
of "uniform", "tod" (keeps each trade's CT minute) and "cost_matched" (uniform's own placements, same seed and
draws, each charged the commission and slippage the engine charged the real trade it replaces), the one with
the largest p_net, ties to "cost_matched" then "tod". The largest of the p-values is conservative as long as
one of them is valid; "cost_matched" alone is not always the harder bar (a strategy that fills at expensive
minutes is flattered by it), which is why it is not reported by itself. Older records have two modes and
the larger p_net of tod and uniform. All modes stay in suite.json. When every draw of the reported mode equals
the strategy (draw sd 0), the contract warns that the percentile is uninformative.
power.passed is the realized test (realized edge >= realized minimum), not the gate; a trial always passed the
gate. A realized minimum that cannot be computed or is undetectable is warned about, never silent.

Baseline and the price-limit rule: the random baseline holds a
PRICE_LIMIT exit as it holds a SIGNAL or FLAT_BY exit (it fills at the open of its exit bar), and its eligible
entry bars exclude every decision bar the engine's price-limit rule refuses (a trade date without a settlement
reference, a close at or beyond a stop level), read from the run's own price-limit arrays. On a root where the
rule never acts the draws are byte-identical to before. A drawn hold is not cut short where the kernel would
force a PRICE_LIMIT exit inside it.

PBO, CPCV and family tests of a hypothesis (library functions; not yet
a result key): they take the whole declared grid with ok results or refuse, and run at fixed module conventions
echoed in their output (PBO: Sharpe, 16 blocks, drop_first, worst, min; CPCV: 6 groups, 2 test groups, purge 1
day, embargo 5 days; family tests: stationary bootstrap, block ceil(T^(1/3)), 1,000 replications, seed 20261117,
alpha 0.05); no caller argument changes them. Each run's daily file is checked against its result record and a
mismatch is refused (GridFileMismatch): by the sha256 in the record's artifact_sha256 where the record carries
one, else, for an older record, by the sum of net cents and the number of days; the runs checked
the older way are listed in the output as sum_checked_run_ids.

## 7a. Engine version in force: pfb-engine/0.3.1

Since 0.3.0 the engine applies the price-limit and locked-market rules ported from upstream's
price-limit rules (the sourced limit table, the settlement proxy over the run's own bars of the previous
trade date, exact arithmetic): a close at or beyond a stop level (the limit less the 2% buffer) queues a
PRICE_LIMIT exit and refuses an entry, and a trade date without a prior settlement reference refuses entries.
The buffer and the switch are engine arguments a trial cannot reach; every config id changed with the version.
At 0.3.1 an exit that meets a market locked
against it (a pending SIGNAL or PRICE_LIMIT exit, a touched STOP, a FLAT_BY) waits and fills at the open of the
first bar that trades through, with its own reason; a session-end exit on a bar locked against the position
refuses the run by name (locked_exit_through_session_end), as does a lock still holding at the session end.
A PRICE_LIMIT exit fill pays the cost model's event-window slippage of its CT minute and side (max half spread
plus the bucket's depth, as upstream's event fill cost) whether event_costs is "off" or "on"; a model
without an event table (fixed:<k>, stress over fixed) charges its own slippage and the result warns
"price-limit exit charged the fixed slippage: this cost model has no event table". The cost-stress levels
re-price that fill from the stressed event table. The standing warning "price-limit and locked-market rules
are not modelled" is carried only when a trade date lies outside the sourced limit table or a position was open
on a date without a prior settlement reference; a run whose only such dates had entries refused and no
position open carries "price-limit rule: entries refused on Q of N trade date(s) without a prior settlement
reference in the run's bars". A root with dynamic circuit breakers only has no lock limit and carries its own
narrower warning. Prop outputs built from day paths keep the standing warning (a day path has no provenance).

## 8. Book hypotheses, schema "pfb.hypothesis/3"

The test statistic is the daily net of a POOLED book of K member configurations, pooled into one account.
Keys exactly: schema, id, purpose, statement, budget, no_signal_reason, cost, power, suite, secondary, book,
optional lookahead_seed. [book]: account ("<firm>/<plan file stem>"), cap ("plan"), statistic
("daily_net"), [[book.member]]: strategy, strategy_sha256, root, contracts, params (one value per SPEC
parameter, inside its SPEC grid), window = {first, last}, trades_per_year, trade_sd_cents (at contracts).
Counting rule: (1) K >= 2, budget = K, purpose research, no two members share (strategy sha256, root,
params, window), contracts equal the SPEC's; (2) each member is its own trial, N + 1 when its config_id is
new; a begin that is not exactly one member is refused (NotABookMember), and a member already counted under
another config_id is refused (MemberAlreadyCounted); (3) the pooled statistic adds no record and nothing to
N and is refused (BookIncomplete) until every member has an ok result; it is read at the live N; (4) the
file sha256 is pinned: a changed file is a new hypothesis in which every member counts again; (5) the power
check is the book's: Y = days in at least one member window / 365.25; derived m = sum(m_i Y_i) / Y;
[power] may state a lower trades_per_year or a higher trade_sd_cents than the derived ones, never the
reverse (PowerFlattered, exact fractions); registration at N + K, each begin at n_before + K - members
counted, replay of both; (6) suite.neighbourhood false, sd_source "stated"; (7) members are not assumed
independent: two members are tied when their windows share a day AND they share a root family (a micro and
its full-size sibling are one family: config/instruments.toml `exposure`, recorded per member as
root_family) or a strategy path or sha256; a group (connected set of tied members) moves together; across
groups the correlation is the stated [power] member_correlation (0 to 1, required) with a non-empty
member_correlation_reason. With a_i = sqrt(m_i Y_i) s_i and A_g the sum over group g, the derived
s = sqrt((sum_g A_g^2 + rho sum_{g != h} A_g A_h) / sum_i m_i Y_i); every square root is rounded UP to a
multiple of 1e-6 in integer arithmetic, and the stated trade_sd_cents must be at least that figure.
Registration, every begin and replay use the same derived figure; replay refuses a record whose figures
do not follow from its members, root families, groups and rho.
Every schema-2 member check applies per member (committed at HEAD, hash, SPEC, closed window, data file,
rule C5; every member strategy is pinned against engine_check). The hypothesis record gains the key "book"
(null for schema 2; tier 2, never falls): account, cap, statistic, members (each with its SPEC session and
bar_minutes and root_family), member_correlation, member_correlation_reason, derived {union_days,
trades_per_year, trade_sd_cents, groups}; strategy_path, strategy_sha256,
bar_minutes, session and grid are null, roots and windows the members' own in member order.
`pfb hypothesis register <file> --dry-run` makes every check, appends nothing and prints the refusals.
Limit of the derived figures: the derived m and s come from each member's STATED
trades_per_year and trade_sd_cents, and no code yet compares them with what the K runs realize. A member
stating many trades at a small SD therefore lowers the book's registered minimum edge as far as its author
likes (the pooled edge is diluted by the same factor, so the bar for the real members does not move, but the
registered figure reads easier than it is). The derived figures are only as good as the members' stated ones.
Clause 8, binding on whatever code first computes the pooled statistic (none exists:
`pfb.registry.bookreg.book_runs` has no caller): it recomputes the book's minimum edge from each member's
REALIZED trades and per-trade SD (the K results' suite power inputs) with the formula of clauses 5 and 7; where
the realized minimum edge is above the registered one, the pooled result is marked underpowered and cannot
carry a positive verdict. It reads the K runs' daily files only after checking each against its result record.
