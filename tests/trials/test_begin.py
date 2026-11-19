"""TRIALS.md sections 3 and 4: refusals at a trial's begin, the replay checks, finish bound to the receipt."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from registry_env import (  # pyright: ignore[reportMissingImports]
    HYPOTHESIS_REL,
    PE_AMENDMENT,
    PE_BASE_LINES,
    STRATEGY_REL,
    Env,
    git,
    pe_line,
)
from trial_support import kinds  # pyright: ignore[reportMissingImports]

from pfb.registry.api import Receipt, begin, finish, read_state, register_hypothesis, sync
from pfb.registry.chain import canonical_bytes, read_ledger, sha256_hex
from pfb.registry.errors import (
    ChainBroken,
    DirtyTree,
    EngineCheckStrategy,
    FinishNotPermitted,
    HypothesisChanged,
    HypothesisNotRegistered,
    LedgerNotTracked,
    NoSignalUnexplained,
    PowerInsufficient,
    StrategyChanged,
    SyncRequired,
    UndeclaredConfiguration,
)
from pfb.registry.records import KEYS, config_id, replay

ENGINE_CHECKS_REL = "config/engine_checks.toml"


# ------------------------------------------------------------------------------------- the begin record ----


def test_a_trial_begin_record_carries_the_third_tier(trial_env: Env) -> None:
    receipt = trial_env.begin_trial()
    record = receipt.record
    assert set(record) == KEYS["trial"]
    assert record["calendars_sha256"] == "c" * 64 and record["lookahead_signal"] == "seen"
    hyp = read_state(trial_env.config).hypotheses["H1"]
    assert record["power"] == {
        "n_used": 302 + 2 - 0,
        "min_edge_cents": hyp["power"]["min_edge_cents"],
        "target_edge_cents": 1000,
    }
    assert record["configurations"] == [config_id(record)]


def test_the_power_n_used_falls_by_the_configurations_already_counted(trial_env: Env) -> None:
    trial_env.begin_trial(params={"a": 1})
    second = trial_env.begin_trial(params={"a": 2})
    assert second.record["n_before"] == 303 and second.record["power"]["n_used"] == 303 + 2 - 1


def test_other_kinds_carry_a_null_power(genesis_env: Env) -> None:
    record = begin(genesis_env.request("synthetic"), config=genesis_env.config).record
    assert record["power"] is None and record["lookahead_signal"] == "seen"


def test_a_begin_without_calendars_sha256_is_refused(genesis_env: Env) -> None:
    from pfb.registry.errors import BadRequest

    with pytest.raises(BadRequest):
        begin(genesis_env.request("synthetic", calendars_sha256=""), config=genesis_env.config)


# ----------------------------------------------------------------------------- refusals at begin (s4) ----


def test_an_unregistered_hypothesis_is_refused(genesis_env: Env) -> None:
    genesis_env.write(HYPOTHESIS_REL, genesis_env.hypothesis_text())
    genesis_env.commit()
    with pytest.raises(HypothesisNotRegistered):
        genesis_env.begin_trial()


def test_a_changed_hypothesis_file_is_refused(trial_env: Env) -> None:
    trial_env.write(HYPOTHESIS_REL, trial_env.hypothesis_text(statement='"edited"'))
    trial_env.commit()
    with pytest.raises(HypothesisChanged):
        trial_env.begin_trial()


def test_a_changed_strategy_is_refused(trial_env: Env) -> None:
    trial_env.write(STRATEGY_REL, "x = 3\n")
    trial_env.commit()
    with pytest.raises(StrategyChanged):
        trial_env.begin_trial(strategy_sha256=trial_env.strategy_sha)


@pytest.mark.parametrize(
    "override",
    [
        {"root": "MES"},
        {"first": "2025-04-02"},
        {"last": "2025-04-29"},
        {"session": "rth"},
        {"cost_model": "fixed:1"},
        {"cost_model_id": "fixed:1"},
        {"cost_config_sha256": "9" * 64},
        {"fees": "lucid"},
        {"fees_sha256": "9" * 64},
        {"params": {"a": 4}},
        {"params": {"a": 1, "b": 2}},
    ],
    ids=[
        "root", "sub-window-first", "sub-window-last", "session", "cost-model", "cost-model-id",
        "cost-config-sha", "fees", "fees-sha", "param-value", "param-extra",
    ],
)  # fmt: skip
def test_a_configuration_unlike_the_hypothesis_record_is_refused(
    trial_env: Env, override: dict[str, Any]
) -> None:
    with pytest.raises(UndeclaredConfiguration):
        trial_env.begin_trial(**override)
    assert kinds(trial_env) == ["genesis", "hypothesis"]


def test_a_dirty_tree_is_refused(trial_env: Env) -> None:
    trial_env.write("notes.txt", "x\n")
    with pytest.raises(DirtyTree):
        begin(trial_env.request("trial"), config=trial_env.config)


def test_ledger_lines_not_at_head_are_refused(trial_env: Env) -> None:
    begin(trial_env.request("synthetic"), config=trial_env.config)
    with pytest.raises(DirtyTree):
        begin(trial_env.request("trial"), config=trial_env.config)


def test_a_ledger_not_tracked_at_head_is_refused(genesis_env: Env) -> None:
    # Arrange: H1 registered, then the ledger is removed from HEAD (kept on disk)
    genesis_env.write(HYPOTHESIS_REL, genesis_env.hypothesis_text())
    genesis_env.commit()
    register_hypothesis(genesis_env.repo / HYPOTHESIS_REL, genesis_env.config, genesis_env.facts())
    git(genesis_env.repo, "rm", "-q", "--cached", "--", "ledger/registry.jsonl")
    genesis_env.write(".gitignore", "runs/\nledger/\n")
    git(genesis_env.repo, "commit", "-q", "-am", "untrack the ledger")
    # Act / Assert
    with pytest.raises(LedgerNotTracked):
        begin(genesis_env.request("trial"), config=genesis_env.config)


def test_a_ledger_outside_the_repository_takes_no_trial(trial_env: Env) -> None:
    outside = trial_env.repo.parent / "elsewhere" / "registry.jsonl"
    outside.parent.mkdir()
    outside.write_bytes(trial_env.config.registry_path.read_bytes())
    config = replace(trial_env.config, registry_path=outside)
    with pytest.raises(LedgerNotTracked):
        begin(trial_env.request("trial"), config=config)


def test_a_begin_with_pe_out_of_sync_is_refused(trial_env: Env) -> None:
    trial_env.write_pe([*PE_BASE_LINES, pe_line("r002-T2", "T2", ["T2-a"], 302, 303)], [PE_AMENDMENT])
    with pytest.raises(SyncRequired):
        trial_env.begin_trial()


def test_no_signal_without_a_no_signal_reason_is_refused(trial_env: Env) -> None:
    with pytest.raises(NoSignalUnexplained):
        trial_env.begin_trial(lookahead_signal="none")


def test_no_signal_with_a_no_signal_reason_is_accepted(genesis_env: Env) -> None:
    genesis_env.write(HYPOTHESIS_REL, genesis_env.hypothesis_text(no_signal_reason='"signals are rare"'))
    genesis_env.commit()
    register_hypothesis(genesis_env.repo / HYPOTHESIS_REL, genesis_env.config, genesis_env.facts())
    assert genesis_env.begin_trial(lookahead_signal="none").record["lookahead_signal"] == "none"


def _target_at_the_edge(env: Env) -> int:
    """A target equal to the minimum detectable edge at registration (N + budget = 304)."""
    from datetime import date

    from pfb.analysis.power import min_edge, years_of

    years = years_of([(date(2025, 4, 1), date(2025, 4, 30))])
    return int(min_edge(trades_per_year=5000.0, trade_sd_cents=1000, years=years, n_trials=304,
                        trade_skew=0.0, trade_kurtosis=3.0).min_edge_cents or 0)  # fmt: skip


def test_a_power_check_failing_at_begin_is_refused(genesis_env: Env) -> None:
    # Arrange: registered at its edge, then upstream's N jumps by 5000 (sync), so n_used at begin is 5475
    target = _target_at_the_edge(genesis_env)
    genesis_env.write(HYPOTHESIS_REL, genesis_env.hypothesis_text(power__target_edge_cents=str(target)))
    genesis_env.commit()
    register_hypothesis(genesis_env.repo / HYPOTHESIS_REL, genesis_env.config, genesis_env.facts())
    ids = [f"T2-{i}" for i in range(5000)]
    genesis_env.write_pe([*PE_BASE_LINES, pe_line("r002-T2", "T2", ids, 302, 5302)], [PE_AMENDMENT])
    sync(genesis_env.config)
    # Act / Assert
    with pytest.raises(PowerInsufficient):
        genesis_env.begin_trial()
    assert kinds(genesis_env) == ["genesis", "hypothesis", "sync"]


def test_a_trial_without_data_files_is_refused(trial_env: Env) -> None:
    with pytest.raises(ChainBroken):
        trial_env.begin_trial(data_files=())


def test_an_engine_check_of_a_pinned_strategy_is_refused(trial_env: Env) -> None:
    trial_env.write(ENGINE_CHECKS_REL, trial_env.engine_checks_text())
    trial_env.commit()
    with pytest.raises(EngineCheckStrategy):
        begin(trial_env.request("engine_check"), config=trial_env.config)


# ----------------------------------------------------------------------------------- replay of tiers ----


def _rechain(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out, previous = [], "0" * 64
    for record in records:
        rec = {**record, "prev_sha256": previous}
        out.append(rec)
        previous = sha256_hex(canonical_bytes(rec))
    return out


def _records(env: Env) -> list[dict[str, Any]]:
    return [dict(line.record) for line in read_ledger(env.config.registry_path)]


def test_replay_refuses_a_trial_without_the_third_tier(trial_env: Env) -> None:
    trial_env.begin_trial()
    records = _records(trial_env)
    old = {k: v for k, v in records[-1].items() if k not in ("calendars_sha256", "lookahead_signal", "power")}
    old["configurations"] = [config_id(old)]
    with pytest.raises(ChainBroken):
        replay(_rechain([*records[:-1], old]))


def test_replay_refuses_altered_power_figures(trial_env: Env) -> None:
    trial_env.begin_trial()
    records = _records(trial_env)
    records[-1] = {**records[-1], "power": {**records[-1]["power"], "min_edge_cents": 1}}
    with pytest.raises(ChainBroken):
        replay(_rechain(records))


def test_replay_refuses_an_altered_hypothesis_power(trial_env: Env) -> None:
    records = _records(trial_env)
    records[1] = {**records[1], "power": {**records[1]["power"], "n_used": 999}}
    with pytest.raises(ChainBroken):
        replay(_rechain(records))


def test_replay_refuses_a_result_whose_begin_sha256_is_not_the_begin_line(genesis_env: Env) -> None:
    receipt = begin(genesis_env.request("synthetic"), config=genesis_env.config)
    finish(receipt, "ok", {}, [], config=genesis_env.config)
    records = _records(genesis_env)
    records[-1] = {**records[-1], "begin_sha256": "f" * 64}
    with pytest.raises(FinishNotPermitted):
        replay(_rechain(records))


def test_a_result_without_begin_sha256_after_one_with_it_is_refused(genesis_env: Env) -> None:
    for _ in range(2):
        receipt = begin(genesis_env.request("synthetic"), config=genesis_env.config)
        finish(receipt, "ok", {}, [], config=genesis_env.config)
    records = _records(genesis_env)
    records[-1] = {k: v for k, v in records[-1].items() if k != "begin_sha256"}
    with pytest.raises(ChainBroken):
        replay(_rechain(records))
    records[2] = {k: v for k, v in records[2].items() if k not in ("begin_sha256", "artifact_sha256")}
    records[-1] = _records(genesis_env)[-1]
    chained = _rechain(records)  # the second begin line's sha256 changes with its prev_sha256
    chained[-1] = {**chained[-1], "begin_sha256": sha256_hex(canonical_bytes(chained[-2]))}
    assert replay(chained).records == 5  # an older result tier is valid as a prefix


# --------------------------------------------------------------------------------- finish and receipts ----


def test_finish_with_a_foreign_receipt_is_refused(genesis_env: Env) -> None:
    mine = begin(genesis_env.request("synthetic"), config=genesis_env.config)
    other = begin(genesis_env.request("synthetic"), config=genesis_env.config)
    forged = Receipt(mine.run_id, "synthetic", other.record_sha256, 302, 302, mine.record)
    with pytest.raises(FinishNotPermitted):
        finish(forged, "ok", {}, [], config=genesis_env.config)
    assert kinds(genesis_env) == ["genesis", "synthetic", "synthetic"]


def test_finish_records_the_begin_sha256(genesis_env: Env) -> None:
    receipt = begin(genesis_env.request("synthetic"), config=genesis_env.config)
    line = finish(receipt, "ok", {}, [], config=genesis_env.config)
    assert line.record["begin_sha256"] == receipt.record_sha256 and set(line.record) == KEYS["result"]


def test_a_trial_result_after_a_pe_change_is_accepted_and_the_next_begin_refused(trial_env: Env) -> None:
    receipt = trial_env.begin_trial()
    trial_env.write_pe([*PE_BASE_LINES, pe_line("r002-T2", "T2", ["T2-a"], 302, 303)], [PE_AMENDMENT])
    assert finish(receipt, "ok", {}, [], config=trial_env.config).record["n_after"] == 303
    with pytest.raises(SyncRequired):
        trial_env.begin_trial(params={"a": 2})
