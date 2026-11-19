""": the hypothesis record's alpha-only figure under alpha rule 2, and replay of a
record written under rule 1 (valid as a prefix only). Synthetic ledgers only."""

from __future__ import annotations

from typing import Any

import pytest
from registry_env import Env  # pyright: ignore[reportMissingImports]

from pfb.analysis.power import ALPHA_RULE, LEGACY_ALPHA_RULE
from pfb.registry.api import register_hypothesis
from pfb.registry.chain import read_ledger
from pfb.registry.errors import ChainBroken
from pfb.registry.powercheck import hypothesis_power, power_at, shortest_window
from pfb.registry.records import replay

R1_REL, R2_REL = "hypotheses/r1.toml", "hypotheses/r2.toml"


def _register(env: Env, rel: str, hyp_id: str) -> None:
    env.write(rel, env.hypothesis_text(hyp_id=hyp_id))
    env.commit()
    register_hypothesis(env.repo / rel, env.config, env.facts())


def _records(env: Env) -> list[dict[str, Any]]:
    return [dict(line.record) for line in read_ledger(env.config.registry_path)]


def _as_written_under(rule: int, record: dict[str, Any]) -> dict[str, Any]:
    n_used = record["n_before"] + record["budget"]
    return {**record, "power": hypothesis_power(record["power"], record["windows"], n_used, alpha_rule=rule)}


def test_registration_writes_the_alpha_figure_of_rule_2(genesis_env: Env) -> None:
    # Arrange / Act
    _register(genesis_env, R1_REL, "R1")
    record = _records(genesis_env)[-1]
    power = record["power"]
    window = [shortest_window(record["windows"])]
    # Assert: the gate formula with nothing deflated (N = 1), not the rule 1 figure at N
    assert power["min_edge_cents_alpha"] == power_at(power, window, 1).min_edge_cents
    assert power == _as_written_under(ALPHA_RULE, record)["power"]
    legacy = _as_written_under(LEGACY_ALPHA_RULE, record)["power"]
    assert legacy["min_edge_cents_alpha"] > power["min_edge_cents_alpha"]
    assert {k: v for k, v in legacy.items() if k != "min_edge_cents_alpha"} == {
        k: v for k, v in power.items() if k != "min_edge_cents_alpha"
    }


def test_replay_accepts_a_record_written_under_rule_1_before_any_rule_2_record(genesis_env: Env) -> None:
    # Arrange: the ledger as and D would have written it
    _register(genesis_env, R1_REL, "R1")
    records = _records(genesis_env)
    old = _as_written_under(LEGACY_ALPHA_RULE, records[-1])
    assert old != records[-1]
    # Act / Assert: no ChainBroken
    replay([*records[:-1], old])
    replay(records)


def test_replay_refuses_a_rule_1_record_after_a_rule_2_record(genesis_env: Env) -> None:
    # Arrange: R1 under rule 2, then R2 forged back to the rule 1 figure
    _register(genesis_env, R1_REL, "R1")
    _register(genesis_env, R2_REL, "R2")
    records = _records(genesis_env)
    forged = _as_written_under(LEGACY_ALPHA_RULE, records[-1])
    # Act / Assert: the older rule is valid as a prefix only
    with pytest.raises(ChainBroken, match="power figures differ"):
        replay([*records[:-1], forged])
    # ...while two rule 1 records in a row, then a rule 2 one, replay
    first_old = _as_written_under(LEGACY_ALPHA_RULE, records[-2])
    replay([*records[:-2], first_old, forged])
    replay([*records[:-2], first_old, records[-1]])


@pytest.mark.parametrize("key", ["min_edge_cents_alpha", "min_edge_cents"])
def test_replay_refuses_a_figure_neither_rule_gives(genesis_env: Env, key: str) -> None:
    _register(genesis_env, R1_REL, "R1")
    records = _records(genesis_env)
    for base in (records[-1], _as_written_under(LEGACY_ALPHA_RULE, records[-1])):
        forged = {**base, "power": {**base["power"], key: base["power"][key] + 1}}
        with pytest.raises(ChainBroken, match="power figures differ"):
            replay([*records[:-1], forged])
