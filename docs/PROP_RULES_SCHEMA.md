# Prop rule-config schema, pfb-rules-1 (binding)

One TOML file per firm, plan and size: `config/firms/<firm>/<plan>_<size>.toml` (lower case, `_` in plan ids,
size like `50k`). A plan with two payout paths (Topstep XFA standard and consistency) gets one file per path:
`config/firms/topstep/combine_xfa_standard_50k.toml`, `.../combine_xfa_consistency_50k.toml`.
No rule number lives in code. Units: money in integer cents (`_cents`), fractions in integer basis points (`_bps`,
4000 = 40%), days as integers, dates as ISO `YYYY-MM-DD`. A float never decides a rule boundary.

## 1. Leaf values

Every rule leaf is an inline table. Exactly one of these forms:

```toml
profit_target_cents = { v = 300000, src = "S01" }                # a value with its source id
daily_loss_cents    = { v = "none", src = "S02" }                # the firm has no such rule (only where allowed)
consistency_bps     = { unknown = "page does not say; A2 reads 50, third party 40", src = "S03" }
```

- `v`: int, bool, string enum, or a list (`caps_cents`, `scaling`). `"none"` only where the field list says
  `| none`.
- `src`: a key of `[sources]`. Required for `v` and `"none"`; optional for `unknown`.
- `unknown`: the reason, non-empty. The loader keeps it as Unknown; the engine raises `RuleUnknown(<dotted
  path>)` the moment a run needs that value (an evaluation-only run does not need `[funded]` or `[payout]`).
- Optional on any leaf: `note` (string), `alt` (list of alternative readings, the stricter one first; used for
  sensitivity runs, and read as the value only under `assume = true`, below), `kind` = `"exact" | "conservative" | "inferred"` (default exact; conservative = the
  stricter reading chosen when the page is unclear, the note says why).
- `assume = true` (only on an `unknown` leaf with an `alt`): the run reads alt[0] in place of the
  unknown value, as an assumption every prop result names ("rule unknown, assumed: <path> = <alt0> ...").
  Refused (PlanConfigError) without an alt, on a `v` leaf, on funded.funded_reset_cents and
  fees.rebill_adds_reset_credit (never assumed available or free), and with alt[0] = "none" on a daily loss
  limit (never assumed absent). Without it an unknown raises RuleUnknown as before.
- Any field the schema lists but the file omits is an error (no silent default). Unknown extra keys are an error.

## 2. Sources

```toml
[sources.S01]
url = "https://help.example.com/..."          # a firm's own page, or
record = "<record path>"                # or a research record path (+ rule id or JSON path)
looked_up = "2026-10-21"                       # date the value was read on that page (for a record: the record's lookup date)
title = "short title"
quote = "at most about 15 words, verbatim"     # optional for records
```

## 3. Top level

```toml
schema = "pfb-rules-1"
firm = "topstep"              # topstep | lucid | bulenox | mffu | tradeify
plan = "combine_xfa_standard" # plan id, same as the file name stem without the size
size = "50k"
label = "Topstep 50K Trading Combine, Standard path, XFA standard payout path"
status = "sourced"            # sourced (every v from a page or record) | provisional (some v not re-checked)
effective_from = { v = "2026-10-21", src = "S01" }   # date the rules took effect, or the earliest lookup that shows them
version = { n = 1, rules_read = "2026-11-16", read_in = "<where the rules were read>" }
```

`version` (required): `n` is an int >= 1, `rules_read` the ISO date the rules were read (before
effective_from is refused), `read_in` the record that holds the reading. `effective_from` is the firm's own
in-force date of these rules and is what chooses the version charged on a trade date (section 8); the date
the rules were read is `version.rules_read`, never a substitute for it.

Older versions of a plan live at `config/firms/<firm>/versions/<effective_from>/<plan>_<size>.toml` (the same
schema, a lower `version.n`, an earlier effective_from; `discover_plans` and the catalogue never see them as
plans). The plan file itself is always the newest version; a plan file older than one of its dated versions
is refused (PlanConfigError).

## 4. `[eval]` (evaluation phase)

| field | type | meaning |
|---|---|---|
| start_balance_cents | int | starting balance |
| profit_target_cents | int | profit over start needed |
| target_check | "eod" \| "intraday" | eod: the close balance is tested after each day; intraday: equity touching the target counts at that minute (the day still ends) |
| drawdown_cents | int | maximum loss distance |
| drawdown_type | "eod_trailing" \| "intraday_trailing" \| "static" | see section 8 |
| breach_check | "intraday" \| "eod" | intraday: every minute's eq_low is tested against the floor; eod: only the close |
| breach_at_equal | bool | true: equity equal to the floor is a breach |
| drawdown_lock_rel_cents | int \| none | the floor never rises above start + this; none = it keeps trailing |
| daily_loss_cents | int \| none | loss from the day's opening balance |
| daily_loss_kind | "soft" \| "hard" \| "none" | soft: the day stops (flattened), no breach, booked at exactly -daily_loss (default) or at the minute's low (sensitivity, section 8); hard: breach |
| consistency_type | "none" \| "best_day_frac_of_profit" \| "best_day_frac_of_target" | best day against total profit or against the target |
| consistency_bps | int \| none | the fraction |
| consistency_strict | bool | true: best day must be < frac; false: <= frac |
| min_trading_days | int | days with at least one trade (a day path with sides > 0) |
| min_winning_days | int | 0 when no such rule |
| winning_day_cents | int \| none | threshold for a winning day (net >= this) |
| max_contracts_minis | int | max open contracts in mini equivalents |
| time_limit_calendar_days | int \| none | hard access window per purchase (Bulenox 30) |
| flat_by | string \| none | flag the engine already knows (e.g. "session_close"); informational, the run enforced it |
| news_rule | string \| none | flag ("none", "tier1_no_open_5min"); informational, the engine does not model it unless the run did |

Pass: at the close of a day (or the minute for intraday target_check) when profit >= target AND consistency holds
AND min_trading_days and min_winning_days are met (counting this day). If the target is met but consistency or
day counts are not, the attempt continues.

## 5. `[funded]` (funded phase: XFA, Master, funded sim)

Same drawdown fields as `[eval]` (start_balance_cents, drawdown_cents, drawdown_type, breach_check,
breach_at_equal, drawdown_lock_rel_cents, daily_loss_cents, daily_loss_kind), plus:

| field | type | meaning |
|---|---|---|
| floor_after_first_payout_rel_cents | int \| none | after the first payout the floor is set to start + this and stops trailing; none = unchanged |
| scaling | list of [below_rel_cents \| "none", max_minis] | ascending tiers by balance over start; last has "none" |
| scaling_boundary | "lower" \| "upper" | a balance exactly on a boundary takes the lower or upper tier |
| payout_count_limit | int \| none | the funded account ends (no further value) after this many payouts |
| inactivity_days | int \| none | informational |
| funded_reset_cents | int \| none | price to reopen a breached funded account (Topstep Back2Funded, Tradeify Funded Reset); none = no such option. Not modelled: a result of a plan holding a value or an unknown names it and is computed as if the account is never bought back |

## 6. `[payout]` (one payout path per file)

| field | type | meaning |
|---|---|---|
| path | string | label, e.g. "standard", "consistency" |
| win_day_cents | int \| none | a winning day nets >= this (or > if win_day_strict) |
| win_day_strict | bool | |
| win_days_required | int | winning days in the window |
| traded_days_required | int | traded days in the window |
| net_positive_since_last | bool | window net > 0 required |
| net_min_first_cents | int \| none | window net needed for the first payout |
| net_min_later_cents | int \| none | window net needed for later payouts |
| min_balance_rel_cents | int \| none | balance over start needed to request |
| buffer_rel_cents | int \| none | balance after the payout must stay >= start + this |
| frac_of_profit_bps | int | at most this share of (balance - start) per request (10000 = all) |
| min_request_cents | int | smallest request |
| caps_cents | list of int \| none | cap of the k-th payout (1-based); the last value repeats; none = no fixed cap |
| window_cap_bps | int \| none | a payout is also at most this share of the window net (Tradeify Select Daily: 20000 = 2x) |
| window_cap_from_payout | int | first payout index (1-based) the window cap applies to (1 when window_cap_bps is none) |
| consistency_bps | int \| none | best day vs profit in scope |
| consistency_scope | "window" \| "cumulative" | since last payout, or since funded start |
| consistency_strict | bool | |
| request_weekday | int \| none | ISO weekday (1 = Monday) on which a request may be made; none = any day (the weekday of the sequence clock, section 8) |
| window_calendar_days | int \| none | a request needs at least this many calendar days of the sequence clock since the window start: the first traded day of the funded life for the first payout, the close of the previous payout after it (MFFU Pro 14); none = no calendar window |
| dead_days_after_payout | int | trading days after a payout that count toward nothing |
| payout_day_counts | bool | the payout day counts toward the next window |
| split_bps | int | trader share |
| split_first_full_cents | int \| none | the first this-many cents of payouts are paid at 100% (Bulenox), then split_bps |

## 7. `[fees]`

| field | type | meaning |
|---|---|---|
| billing | "monthly" \| "one_time" | |
| eval_price_cents | int | price per purchase or per billing period |
| billing_period_calendar_days | int \| none | monthly plans |
| reset_cents | int \| none | none = a failed attempt needs a new purchase |
| rebill_adds_reset_credit | bool | |
| reset_pushes_rebill | bool | |
| activation_cents | int | one time at the pass |
| funded_monthly_cents | int | 0 when none |
| platform_monthly_cents | int | API, platform or data fee a bot needs (0 when none) |

## 8. Semantics the engine implements (binding)

Day input: one trading day = a sequence of minutes with eq_low, eq_high, eq_close in cents relative to the
day's opening balance (equity_minute). The balance at the close = opening balance + last eq_close.

- Floor. static: start - drawdown. eod_trailing: after each close, peak = max(peak, close balance), floor =
  min(peak - drawdown, start + lock) (no cap when lock is none). intraday_trailing: minute by minute, peak =
  max(peak, opening balance + eq_high); floor = min(peak - drawdown, start + lock); within one minute the high
  is applied before the low (conservative ordering). After floor_after_first_payout the floor is fixed.
- Breach. breach_check intraday: any minute with opening balance + eq_low < floor (<= when breach_at_equal) is
  a breach at that minute; eod: only the close balance is tested. The floor used for an intraday test under
  eod_trailing is the floor set at the previous close.
- Daily loss. Tested on eq_low each minute from the day's opening balance; soft: the day's P&L becomes exactly
  -daily_loss and the rest of the day is ignored; hard: breach. A breach test on the same minute comes first.
  Sensitivity switch (not a plan field): `soft_daily_loss = "minute_low"` books the
  triggering minute's eq_low instead, never better than -daily_loss (the fill at the worst price of that
  minute). The default "limit" is upstream's reading and is optimistic when a minute's low is far below the limit;
  every MC output names the mode used and warns when the plan has a soft daily loss.
- Size. A day whose max_contracts exceeds the phase limit (eval max_contracts_minis; funded: the scaling tier
  for the previous close's balance before any payout taken at that close, as upstream) ends the phase as END_SIZE_LIMIT.
- Size input. A day's max_contracts is its largest minute exposure in mini equivalents: per
  minute the larger of |position| at the close and the contracts of any trade entered or exited in that minute
  (the largest such trade, not the sum: a reversal exits and enters in one minute), so a round trip inside
  one minute shows its size. A pool sums the runs' exposures per minute (a conservative bound, like its low).
- Days. A traded day has sides > 0. Winning day: day net >= winning_day_cents.
- Sequence clock (N3 and window_calendar_days). The funded phase reads calendar time from the trade
  dates: the first day of a sequence is its trade date, and each later day advances the clock by that day's
  own gap from the previous trading day of the history (1 for the history's first day; trade dates must
  increase strictly). In historical order (random_start) the clock is the trade date itself; under the block
  bootstrap a block keeps its real spacing and a jump advances by the new day's gap, so weekdays and calendar
  days stay in step with the trading days drawn (the weekday gate read each sampled day's own
  weekday: a random 1-in-5 gate out of step with dead days). request_weekday reads the clock's weekday;
  window_calendar_days counts clock days. A gap in the history (a closed window, an outage) advances the
  clock by its real length; the inactivity rules that would close an account over such a gap are not
  modelled.
- Window net. net_positive_since_last, net_min_*, the window consistency base and best day, and
  the window cap use EVERY day since the last payout, the processing day (payout_day_counts false) and the
  dead days included: the account's net since the last payout, which is what the firms' rules name. Only the
  traded-day and winning-day counts skip those days. Upstream left their P&L out, which paid a
  payout when the true net since the last one was negative (review example: Tradeify Select Flex, a second
  payout of 625 USD with a net of -250 USD since the first). The development-time parity test against upstream therefore
  compared lives in full only up to a second payout.
- Payout policy (simulation choice, not a firm rule): request the largest amount allowed at the close of the
  first eligible day; the window resets after it.
- Sensitivity plans. `config.with_values` (a plan with overridden leaves) keeps the file's hash
  as `file_sha256` and sets `sha256` to sha256 over the file's hash and the overrides in order (canonical
  JSON), so no output carries the file's hash with different rules; MC outputs record plan_sha256,
  plan_file_sha256 and plan_overrides.

Choices where the reading is a judgement (the one that does not flatter p_pass or EV):
- window_calendar_days after the first payout runs from the previous payout's close, not on a fixed 14-day
  grid anchored at the first trade (MFFU states a 14-day cycle counted from the first trade); the grid can only
  be earlier.
- The window clock starts at the first traded day (sides > 0) of the funded life, not at its first day.
- Window net with the skipped and dead days follows the rule text (the net since the last payout) rather than
  a one-sided rule; it removes the case where upstream paid on a negative true net. It can also add a positive
  processing day to the window net, so it is not uniformly stricter; no firm reading excludes those days.
- Fees and outcomes are defined in docs/PROP_MC.md.
- Not modelled: funded.funded_reset_cents and fees.rebill_adds_reset_credit. Every prop
  result whose plan holds one (a value other than none/false, or an unknown) names it in its warnings
  ("rule not modelled: ..."), with how the figure was computed without it.
- Rules by date. A dated run (`run_dated`, the scalar reference engine) charges each day under
  the version in force on its trade date: the last version whose effective_from is on or before it (integer
  yyyymmdd compare). A day before the earliest version is charged under it and counted in a warning. On a
  version change every rule of the new version applies from that day; the floor carried from the prior
  close stays until the new version's trailing rule moves it. The Monte Carlo resamples day paths, which
  carry no dates, so it charges every day under the newest version, labels the record
  `rules_in_force = "newest_undated"` and warns; a dated run labels it "by_day" and lists each version that
  charged a day with its days. The numba kernel has no per-day version.
