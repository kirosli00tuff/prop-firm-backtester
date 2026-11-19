# Prop Monte Carlo and the rule-engine interface (binding)

Interface types: src/pfb/prop/types.py (changed only through review). Rule semantics:
docs/PROP_RULES_SCHEMA.md section 8.

## 1. Module map

- src/pfb/prop/config.py, rules.py (scalar reference), kernel.py (Numba, pinned to rules.py), fees.py.
- src/pfb/prop/days.py (DayPaths from a run), paths.py (generators), mc.py (estimators).
- src/pfb/prop/pool.py. src/pfb/cli/prop_cmd.py, matrix.py: the CLI.

## 2. Rule-engine entry points (the rule engine provides, the estimators call)

```python
load_plan(path: Path) -> PlanRules                               # config.py; validates per the schema
simulate_eval(plan, days: DayPaths, seq: np.ndarray, *, soft_daily_loss="limit") -> EvalOutcome    # rules.py
simulate_funded(plan, days: DayPaths, seq: np.ndarray, *, soft_daily_loss="limit") -> FundedOutcome  # rules.py
simulate_eval_many(plan, days, seqs: np.ndarray[n_paths, horizon], *, soft_daily_loss="limit") -> EvalBatch
simulate_funded_many(plan, days, seqs, max_payouts=None, *, soft_daily_loss="limit") -> FundedBatch  # kernel
attempt_fee_cents(plan, outcome_days: int, calendar_days: int, *, first: bool) -> int  # fees.py
funded_fee_cents(plan, days_alive: int, calendar_days: int) -> int                     # fees.py
```

`soft_daily_loss` is "limit" (a soft daily loss books exactly -DLL, as upstream) or "minute_low" (the triggering
minute's low, the pessimistic sensitivity); run_mc takes the same keyword and passes it to the
engine. `seq` holds day indices into `DayPaths`; -1 pads past the horizon and ends the phase as `timeout`. The batch
results hold the same fields as the scalar outcomes as numpy arrays; a test pins every field of the batch to the
scalar version on at least 2,000 random sequences per plan and on every config in config/firms/.

## 3. Path generators

- `block_bootstrap(n_days, horizon, n_paths, mean_block, rng)`: stationary bootstrap (Politis-Romano): start at a
  uniform day; with prob 1/mean_block jump to a new uniform day, otherwise take the next day (wrapping at the
  end). Default mean_block 5 (a parameter, recorded).
- `random_start(n_days, horizon, n_paths, rng)`: historical order from a uniform start day; the sequence pads
  with -1 at the end of history (those paths end as `timeout` if still open, counted and reported).
- Cost stress: `stress(days, k_ticks, tick_cents)` returns a new DayPaths with each day's equity lowered by
  k_ticks * tick_cents * sides traded so far that day (applied at each fill minute when fill minutes are known;
  else the whole day cost at the first minute, conservative).
- RNG: numpy PCG64 from the seed; the seed, generator, mean_block, horizon and n_paths go into every output.

## 4. Outputs, per plan file

Attempt = one evaluation from a fresh account along one sequence, horizon H_e (default 252 trading days, and
the plan's time_limit converted at 30 calendar = 21 trading days when shorter). Funded life = the same
sequence continued from the day after the pass, horizon H_f (default 252). n = number of paths.

| output | estimator | standard error |
|---|---|---|
| p_pass | passes / n | sqrt(p(1-p)/n) |
| days_to_pass median, p90 | over passing paths | bootstrap over paths (200 resamples) or order-statistic interval |
| cost_per_attempt | mean attempt fee (purchase price, monthly fee per started period; no reset) | sample sd / sqrt(n) |
| cost_to_funded | cost_per_attempt / p_pass + activation | delta method |
| cost_to_funded_with_resets | first price + (1/p_pass - 1) * retry fee + activation; retry fee = reset if any, else price (one-time); for monthly plans the mean months used per attempt | delta method |
| funded_ev | mean over funded lives of trader payouts (after split) minus funded fees | sample sd / sqrt(n_pass) |
| p_breach_before_first_payout | over funded lives | binomial |
| ev_per_purchase | per attempt: payout_after_split * 1[pass] - attempt fee - activation * 1[pass] - funded fees * 1[pass], averaged over all n paths | sample sd / sqrt(n) |

Rules added after the first review:
- Retry fee: after END_TIME_LIMIT a retry is a new purchase; after a breach it is the reset (if known, else a
  new purchase is NOT assumed: cost_to_funded_with_resets is null with a warning naming fees.reset_cents);
  a reset is modelled with a full access window (warning). An unknown fee nulls only the outputs that need it.
- Platform fee: pro rata to calendar days (platform_monthly_cents x calendar_days // 30, integer cents) in both
  phases, not per started period.
- random_start: starts are drawn only where H_e + H_f days of history remain; when the history is too short,
  padded paths are counted (eval and funded apart) and the affected outputs carry lower_bound true.
- Size-limit ends (END_SIZE_LIMIT) are counted apart from breaches and warned, with the share of days above the
  eval limit and above the first funded tier.
- SEs are Monte Carlo errors given the day pool; they say nothing about the finite history (stated in output).

Every output carries n, the seed, the warnings (timeouts, unknown rules refused, the standing price-limit
warning), and the plan's sha256: the file's hash for a plan as loaded; for a sensitivity plan
(config.with_values) a hash over the file's hash and the overrides, with plan_file_sha256 and plan_overrides
beside it. soft_daily_loss names the booking used, and a plan with a soft daily loss gets a
warning naming it. Runs on synthetic paths are kind synthetic; the demo on the
reference strategy's engine-check run is kind engine_check; nothing counts as a trial.

## 5. `pfb prop --json` (the result contract)

`pfb prop <run_id> --firm f --plan p --size s [--mc n --seed k ...] --json` (one plan) and
`pfb prop matrix <run_id> ... --json` (every plan file) print one object and nothing else on stdout: the
saved run's result contract "pfb.result/1" (docs/result.schema.json, the same object as
runs/<run_id>/result.json and pfb.app.get_run) with two keys replaced by this request:

- `prop`: one row per plan, {firm, plan (the file stem <plan>_<size>), status ("ok" or "skipped"), reason
  (why skipped, else null), p_pass and p_pass_se, ev_per_purchase_cents and ev_se_cents (Monte Carlo
  means and SEs, floats in cents), n_paths, seed, soft_daily_loss ("limit" or "minute_low"); the figures
  and settings are null on a skipped row};
- `artifacts`: the run's artifacts plus the files this request wrote.

Every other key (n, the metrics, the suite parts, warnings) is the saved run's; n is N after that run's
begin, not the live N. result.json is not rewritten, and nothing counts toward N. The full M2 outputs of
section 4 (days to pass, cost to funded, funded EV, breach before payout, warnings, plan sha256 and the
whole MCResult under "mc") are in the files named in `artifacts`: runs/<run_id>/prop/
<firm>_<plan>_<size>_<seed>.json (suffix _stress<k>, _minute_low or _pool-<sha8> where they apply) for
one plan, runs/<run_id>/prop_matrix.json and .md for the matrix. A trial run accepts only what its
hypothesis's [secondary] declares (TRIALS.md section 6; TrialNotDeclared otherwise). A refusal prints
{"schema": "pfb.result/1", "status": "refused", "error": {"code", "reason"}} and exits 2. Without --json
the command prints a short human summary of the same rows. The MCP tool run_prop_mc and
pfb.app.run_prop_mc return the same object.
