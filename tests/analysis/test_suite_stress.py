"""S1 cost stress: each level equals a separate run made with that stress cost model (synthetic bars)."""

from __future__ import annotations

import dataclasses

import pytest
from suite_planting import HASH_RANDOM, REFERENCE

import pfb.analysis.stress as stress_module
from pfb.analysis.spec import SuiteIntegrityError, SuiteSpec
from pfb.engine.run import run_strategy

STRESS_ONLY = SuiteSpec(cost_stress_ticks=(1, 3), baseline_draws=0, walk_forward_folds=0)


@pytest.mark.parametrize("strategy", [REFERENCE, HASH_RANDOM])
def test_each_level_equals_a_separate_run_with_that_cost_model(strategy: object) -> None:
    # Arrange, Act
    base = run_strategy(strategy, source="synthetic", cost_model="fixed:1", seed=4, suite=STRESS_ONLY)  # type: ignore[arg-type]
    levels = base.summary["suite"]["cost_stress"]  # type: ignore[index]
    # Assert
    assert list(levels) == ["1", "3"]
    for k in (1, 3):
        alone = run_strategy(strategy, source="synthetic", cost_model=f"stress:fixed:1:{k}", seed=4)  # type: ignore[arg-type]
        level = levels[str(k)]
        assert level["cost_model"] == f"stress:fixed:1:{k}"
        assert level["net_cents"] == alone.summary["net_cents"]
        assert level["sharpe_annual"] == alone.summary["sharpe_annual"]
        assert level["trades"] == alone.summary["n_trades"] == base.trades.height
    assert levels["3"]["net_cents"] < levels["1"]["net_cents"] < base.summary["net_cents"]


def test_a_level_with_a_different_trade_count_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real = stress_module.rerun_kernel

    def fewer(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        out = real(*args, **kwargs)  # type: ignore[arg-type]
        return dataclasses.replace(out, trades=out.trades.head(out.trades.height - 1))

    monkeypatch.setattr(stress_module, "rerun_kernel", fewer)
    with pytest.raises(SuiteIntegrityError, match="trades"):
        stress_module.cost_stress(_context(), (1, 3))
    # An integrity cross-check, not a part error: in a run it still fails the run.
    with pytest.raises(SuiteIntegrityError, match="trades"):
        run_strategy(REFERENCE, source="synthetic", cost_model="fixed:1", seed=4, suite=STRESS_ONLY)


def _context():  # type: ignore[no-untyped-def]
    """The RunContext run_strategy hands the suite (captured from one synthetic run)."""
    import pfb.analysis.inrun as inrun

    seen = []
    real = inrun.compute_suite

    def spy(ctx, spec):  # type: ignore[no-untyped-def]
        seen.append(ctx)
        return real(ctx, spec)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(inrun, "compute_suite", spy)
        run_strategy(REFERENCE, source="synthetic", cost_model="fixed:1", seed=4, suite=SuiteSpec.off())
    return seen[0]


def test_no_ticks_gives_none() -> None:
    result = run_strategy(REFERENCE, source="synthetic", cost_model="fixed:1", seed=4, suite=SuiteSpec.off())
    assert result.summary["suite"]["cost_stress"] is None  # type: ignore[index]


def test_a_level_reruns_only_the_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """No generate call and no look-ahead check after the base simulation (the cost-stress part only)."""
    import pfb.engine.run as run_module

    calls: list[str] = []
    real_check = run_module.lookahead_check

    def counting(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        calls.append("check")
        return real_check(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(run_module, "lookahead_check", counting)
    run_strategy(REFERENCE, source="synthetic", cost_model="fixed:1", seed=4, suite=STRESS_ONLY)
    assert calls == ["check"]  # the base run's only


def test_the_replay_stays_out_of_the_summary_and_the_result() -> None:
    import json

    from pfb.engine.run import RunResult

    result = run_strategy(REFERENCE, source="synthetic", cost_model="fixed:1", seed=4, suite=STRESS_ONLY)
    json.dumps(result.summary, allow_nan=False)
    assert '"replay"' not in json.dumps(result.summary) and "replay" not in result.provenance
    assert {f.name for f in dataclasses.fields(RunResult)} == {
        "trades",
        "equity_minute",
        "daily",
        "summary",
        "counters",
        "provenance",
        "spec",
        "params",
        "extras",
    }
