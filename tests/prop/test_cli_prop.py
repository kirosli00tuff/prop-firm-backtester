"""`pfb prop` and `pfb prop matrix` (-X2) on synthetic run directories written under tmp_path.

No market data: the runs are seeded coin-flip days; the plans are the known-answer fixtures. Nothing here
registers a trial; the ledger used is a temporary file holding only the keys read here. Since
`--json` prints the run's result contract with prop = the request's rows; the compact per-plan record is read
from its artifact under runs/<run_id>/prop/ and the matrix from prop_matrix.json.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from pfb.cli.main import app
from pfb.contract import validate
from pfb.prop.matrix import OUTPUTS, MatrixRow, matrix_document, matrix_markdown

FIXTURES = Path(__file__).resolve().parent / "fixtures"
NS_MINUTE = 60_000_000_000
RUN_ID = "synthetic-20261116T000000000000Z-0000abcd"
N_PATHS = "300"
runner = CliRunner()


def _write_run(runs_dir: Path, run_id: str, n_days: int = 80, seed: int = 7, position: int = 1) -> Path:
    """Two minutes per day: enter at the first (position open), exit at the second; day net +/-500 USD."""
    rng = np.random.default_rng(seed)
    nets = np.where(rng.random(n_days) < 0.5, 50_000, -50_000).astype(np.int64)
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=int(i)) for i in range(n_days)]
    ts = np.arange(2 * n_days, dtype=np.int64) * NS_MINUTE
    open_eq = np.concatenate(([0], np.cumsum(nets)[:-1]))
    close_eq = np.repeat(np.cumsum(nets), 2)
    close_eq[0::2] = open_eq + nets // 2
    low = np.minimum(close_eq, np.repeat(open_eq, 2)) - 1_000
    high = np.maximum(close_eq, np.repeat(open_eq, 2)) + 1_000
    pos = np.tile([position, 0], n_days).astype(np.int64)
    run = runs_dir / run_id
    run.mkdir(parents=True)
    d = {"trade_date": pl.Date}
    pl.DataFrame(
        {
            "ts_event": ts,
            "trade_date": [x for x in dates for _ in (0, 1)],
            "position": pos,
            "eq_close": close_eq,
            "eq_low": low,
            "eq_high": high,
        },
        schema_overrides=d,
    ).write_parquet(run / "equity_minute.parquet")
    pl.DataFrame({"trade_date": dates, "eq_close": np.cumsum(nets)}, schema_overrides=d).write_parquet(
        run / "daily.parquet"
    )
    pl.DataFrame({"contracts": np.full(n_days, abs(position) or 1, dtype=np.int64), "entry_ts": ts[0::2],
                  "exit_ts": ts[1::2], "entry_bar": np.arange(0, 2 * n_days, 2, dtype=np.int64),
                  "exit_bar": np.arange(1, 2 * n_days, 2, dtype=np.int64), "trade_date": dates},
                 schema_overrides=d).write_parquet(run / "trades.parquet")  # fmt: skip
    return run


def _firms(tmp: Path, names: tuple[str, ...] = ("ka_static_50k", "ka_eod_trailing_50k")) -> Path:
    firms = tmp / "firms"
    (firms / "topstep").mkdir(parents=True)
    for name in names:
        shutil.copy(FIXTURES / f"{name}.toml", firms / "topstep" / f"{name}.toml")
    return firms


def _ledger(tmp: Path, run_ids: tuple[str, ...] = (RUN_ID,), root: str | None = "NQ") -> Path:
    """Begin and ok result records of synthetic runs (only the keys read here)."""
    ledger = tmp / "registry.jsonl"
    rows = []
    for run_id in run_ids:
        begin = {"kind": "synthetic", "run_id": run_id, **({"root": root} if root else {})}
        rows += [begin, {"kind": "result", "run_id": run_id, "status": "ok"}]
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return ledger


def _invoke(args: list[str], tmp: Path, firms: Path, *, ledger: Path | None = None):
    base = ["prop", *args, "--runs-dir", str(tmp / "runs"), "--firms-dir", str(firms),
            "--ledger", str(ledger or _ledger(tmp)), "--json"]  # fmt: skip
    return runner.invoke(app, base)


def _saved(tmp: Path, name: str, run_id: str = RUN_ID) -> dict:
    return json.loads((tmp / "runs" / run_id / "prop" / name).read_text())


def _error(res) -> dict:  # noqa: ANN001
    out = json.loads(res.stdout)
    validate(out)
    assert out["status"] == "refused"
    return out["error"]


def _single(plan: str = "ka_static") -> list[str]:
    return [RUN_ID, "--firm", "topstep", "--plan", plan, "--size", "50k", "--mc", N_PATHS, "--seed", "11",
            "--root", "NQ"]  # fmt: skip


def test_single_plan_json_shape_and_artifact(tmp_path: Path) -> None:
    _write_run(tmp_path / "runs", RUN_ID)
    firms = _firms(tmp_path)
    res = _invoke(_single(), tmp_path, firms)
    assert res.exit_code == 0, res.output
    contract = json.loads(res.stdout)
    validate(contract)
    [row] = contract["prop"]
    assert (row["firm"], row["plan"], row["seed"], row["n_paths"]) == ("topstep", "ka_static_50k", 11, 300)
    out = _saved(tmp_path, "topstep_ka_static_50k_11.json")
    assert row["p_pass"] == out["outputs"]["p_pass"]["value"] and contract["run_id"] == RUN_ID
    assert out["status"] == "ok" and out["kind"] == "synthetic" and out["trial"] is False
    assert out["seed"] == 11 and out["n_paths"] == 300 and out["run_id"] == RUN_ID
    assert len(out["plan_sha256"]) == 64 and out["firm"] == "topstep" and out["plan"] == "ka_static"
    assert set(out["outputs"]) == set(OUTPUTS)
    for name in OUTPUTS:
        assert set(out["outputs"][name]) >= {"value", "se", "n"}
    p = out["outputs"]["p_pass"]
    assert 0.0 < p["value"] < 1.0 and p["se"] > 0 and p["n"] == 300
    assert any("price-limit" in w for w in out["warnings"])
    artifact = tmp_path / "runs" / RUN_ID / "prop" / "topstep_ka_static_50k_11.json"
    assert artifact.is_file()
    saved = json.loads(artifact.read_text())
    assert saved["mc"]["schema"] == "pfb.prop_mc/1" and saved["mc"]["trial"] is False
    assert any(a.endswith("/prop/topstep_ka_static_50k_11.json") for a in contract["artifacts"])


def test_same_seed_same_numbers_and_stress_lowers_pass(tmp_path: Path) -> None:
    _write_run(tmp_path / "runs", RUN_ID)
    firms = _firms(tmp_path)
    assert _invoke(_single(), tmp_path, firms).exit_code == 0
    a = _saved(tmp_path, "topstep_ka_static_50k_11.json")
    assert _invoke(_single(), tmp_path, firms).exit_code == 0
    b = _saved(tmp_path, "topstep_ka_static_50k_11.json")
    assert a["outputs"] == b["outputs"]
    s = _invoke([*_single(), "--stress-ticks", "40"], tmp_path, firms)
    assert s.exit_code == 0, s.output
    stressed = _saved(tmp_path, "topstep_ka_static_50k_11_stress40.json")
    assert stressed["stress_ticks"] == 40 and stressed["tick_cents"] == 500
    assert stressed["outputs"]["p_pass"]["value"] < a["outputs"]["p_pass"]["value"]
    assert json.loads(s.stdout)["prop"][0]["p_pass"] == stressed["outputs"]["p_pass"]["value"]


def test_matrix_over_two_fixture_configs(tmp_path: Path) -> None:
    _write_run(tmp_path / "runs", RUN_ID)
    firms = _firms(tmp_path)
    res = _invoke(["matrix", RUN_ID, "--mc", N_PATHS, "--seed", "3", "--root", "NQ"], tmp_path, firms)
    assert res.exit_code == 0, res.output
    out = json.loads(res.stdout)
    validate(out)
    assert [r["plan"] for r in out["prop"]] == ["ka_eod_trailing_50k", "ka_static_50k"]
    doc = json.loads((tmp_path / "runs" / RUN_ID / "prop_matrix.json").read_text())
    assert doc["n_plans"] == 2 and doc["n_ok"] == 2 and doc["kind"] == "synthetic"
    assert doc["schema"] == "pfb.prop_matrix/1" and len(doc["rows"]) == 2
    assert all(r["record"]["seed"] == 3 for r in doc["rows"])
    md = (tmp_path / "runs" / RUN_ID / "prop_matrix.md").read_text()
    assert "| topstep | ka_static | 50k |" in md and "| topstep | ka_eod_trailing | 50k |" in md
    # the payout policy behind funded_ev and the scope of the SEs, in the .md and the JSON
    assert "largest request at the first eligible close" in doc["payout_policy"]
    assert f"Payout policy: {doc['payout_policy']}." in md
    assert "Monte Carlo error given the day pool" in doc["se_scope"] and doc["se_scope"] in md
    record = doc["rows"][0]["record"]
    assert set(record["counts"]) == {"eval_padded", "funded_padded", "eval_size_limit", "funded_size_limit",
                                     "funded_open_no_payout"}  # fmt: skip


def _unknown_target(firms: Path) -> None:
    """Rewrite topstep/ka_static_50k.toml with an unknown profit target (file name = plan)."""
    path = firms / "topstep" / "ka_static_50k.toml"
    old = 'profit_target_cents = { v = 100000, src = "S01" }'
    text = path.read_text()
    assert old in text
    path.write_text(
        text.replace(old, 'profit_target_cents = { unknown = "test: not published", src = "S01" }')
    )


def test_unknown_value_refused_with_field(tmp_path: Path) -> None:
    _write_run(tmp_path / "runs", RUN_ID)
    firms = _firms(tmp_path)
    _unknown_target(firms)
    res = _invoke(_single("ka_static"), tmp_path, firms)
    assert res.exit_code == 2
    error = _error(res)
    assert error["code"] == "RuleUnknown" and "(field eval.profit_target_cents)" in error["reason"]
    m = _invoke(["matrix", RUN_ID, "--mc", "100", "--root", "NQ"], tmp_path, firms)
    assert m.exit_code == 0, m.output
    assert {r["plan"]: r["status"] for r in json.loads(m.stdout)["prop"]} == {
        "ka_eod_trailing_50k": "ok", "ka_static_50k": "skipped"}  # fmt: skip
    matrix = json.loads((tmp_path / "runs" / RUN_ID / "prop_matrix.json").read_text())
    rows = {r["plan"]: r for r in matrix["rows"]}
    assert (
        rows["ka_static"]["status"] == "skipped" and rows["ka_static"]["field"] == "eval.profit_target_cents"
    )
    assert rows["ka_eod_trailing"]["status"] == "ok"
    md = (tmp_path / "runs" / RUN_ID / "prop_matrix.md").read_text()
    assert "Skipped plans" in md and "- topstep/ka_static_50k: rule value unknown" in md


TRIAL_ID = "trial-20261116T000000000000Z-0000beef"


def _trial_ledger(tmp: Path, plans: list[str], mc: int = 300, seed: int = 11) -> Path:
    """A ledger fragment with the trial's begin record and its hypothesis record (only the keys read here)."""
    ledger = tmp / "ledger.jsonl"
    hyp = {"kind": "hypothesis", "hypothesis_id": "H9",
           "secondary": {"prop_plans": plans, "prop_mc": mc, "prop_seed": seed}}  # fmt: skip
    begin = {"kind": "trial", "run_id": TRIAL_ID, "hypothesis_id": "H9", "root": "NQ"}
    ledger.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in (hyp, begin)))
    return ledger


def test_a_trial_run_without_its_records_in_the_ledger_is_refused(tmp_path: Path) -> None:
    _write_run(tmp_path / "runs", TRIAL_ID)
    firms = _firms(tmp_path)
    for args in ([TRIAL_ID, "--firm", "topstep", "--plan", "ka_static", "--size", "50k", "--root", "NQ"],
                 ["matrix", TRIAL_ID, "--root", "NQ"]):  # fmt: skip
        res = _invoke(args, tmp_path, firms)
        assert res.exit_code == 2
        assert _error(res)["code"] == "RunNotFound"  # the contract needs its begin record first
    assert not (tmp_path / "runs" / TRIAL_ID / "prop").exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--plan", "ka_eod_trailing", "--mc", "300", "--seed", "11"],  # a plan not declared
        ["--plan", "ka_static", "--mc", "301", "--seed", "11"],  # another mc
        ["--plan", "ka_static", "--mc", "300", "--seed", "12"],  # another seed
        ["--plan", "ka_static", "--mc", "300", "--seed", "11", "--stress-ticks", "1"],  # a stress
        ["--plan", "ka_static", "--mc", "300", "--seed", "11", "--block", "7"],  # another block
    ],
    ids=["plan", "mc", "seed", "stress", "block"],
)
def test_prop_on_a_trial_is_refused_unless_plan_mc_and_seed_are_declared(
    tmp_path: Path, extra: list[str]
) -> None:
    # Arrange: TRIALS.md section 6
    _write_run(tmp_path / "runs", TRIAL_ID)
    firms = _firms(tmp_path)
    ledger = _trial_ledger(tmp_path, ["topstep/ka_static_50k"])
    # Act
    res = _invoke([TRIAL_ID, "--firm", "topstep", "--size", "50k", *extra], tmp_path, firms, ledger=ledger)
    # Assert
    assert res.exit_code == 2, res.stdout
    assert _error(res)["code"] == "TrialNotDeclared"
    assert not (tmp_path / "runs" / TRIAL_ID / "prop").exists()


def test_the_matrix_on_a_trial_needs_star_declared(tmp_path: Path) -> None:
    _write_run(tmp_path / "runs", TRIAL_ID)
    firms = _firms(tmp_path)
    ledger = _trial_ledger(tmp_path, ["topstep/ka_static_50k"])
    res = _invoke(["matrix", TRIAL_ID, "--mc", "300", "--seed", "11"], tmp_path, firms, ledger=ledger)
    assert res.exit_code == 2 and _error(res)["code"] == "TrialNotDeclared"


def test_the_declared_plan_on_a_trial_passes_the_declaration_check(tmp_path: Path) -> None:
    from pfb.cli.prop_cmd import Request, require_declared, resolve_source

    _write_run(tmp_path / "runs", TRIAL_ID)
    ledger = _trial_ledger(tmp_path, ["topstep/ka_static_50k"])
    source = resolve_source(TRIAL_ID, tmp_path / "runs", ledger, None)
    args = {"mc": 300, "seed": 11, "block": 5, "generator": "block", "stress_ticks": 0}
    req = Request(False, TRIAL_ID, [], "topstep", "ka_static", "50k", args, None, None, tmp_path / "runs",
                  tmp_path / "firms", ledger)  # fmt: skip
    require_declared(source, req)  # no refusal
    assert source.declared == {"prop_plans": ["topstep/ka_static_50k"], "prop_mc": 300, "prop_seed": 11}


def test_root_and_kind_from_ledger(tmp_path: Path) -> None:
    _write_run(tmp_path / "runs", RUN_ID)
    firms = _firms(tmp_path)
    no_root = [a for a in _single() if a not in ("--root", "NQ")]
    res = _invoke(no_root, tmp_path, firms, ledger=_ledger(tmp_path, root=None))
    assert res.exit_code == 2 and "pass --root" in _error(res)["reason"]
    ledger = _ledger(tmp_path)
    res = _invoke(no_root, tmp_path, firms, ledger=ledger)
    assert res.exit_code == 0, res.output
    ledger.write_text(json.dumps({"kind": "engine_check", "run_id": RUN_ID, "root": "NQ"}) + "\n")
    res = _invoke(no_root, tmp_path, firms, ledger=ledger)
    assert res.exit_code == 2 and "differs" in _error(res)["reason"]


@pytest.mark.parametrize(
    ("args", "needle"),
    [
        (["matrix"], "usage"),
        (["matrix", RUN_ID, "--firm", "topstep"], "drop --firm"),
        ([RUN_ID, "extra", "--firm", "topstep", "--plan", "ka_static", "--size", "50k"], "unexpected"),
        ([RUN_ID, "--firm", "topstep"], "needs --firm"),
        ([RUN_ID, "--firm", "topstep", "--plan", "nope", "--size", "50k", "--root", "NQ"], "no plan file"),
        (["engine_check-x", "--firm", "topstep", "--plan", "ka_static", "--size", "50k"], "no run directory"),
        (["bogus-x", "--firm", "topstep", "--plan", "ka_static", "--size", "50k"], "run kind"),
    ],
)
def test_usage_refusals(tmp_path: Path, args: list[str], needle: str) -> None:
    _write_run(tmp_path / "runs", RUN_ID)
    res = _invoke(args, tmp_path, _firms(tmp_path))
    assert res.exit_code == 2
    assert needle in _error(res)["reason"]


def test_human_output_and_stderr_refusal(tmp_path: Path) -> None:
    _write_run(tmp_path / "runs", RUN_ID)
    firms = _firms(tmp_path)
    base = ["prop", *_single(), "--runs-dir", str(tmp_path / "runs"), "--firms-dir", str(firms),
            "--ledger", str(_ledger(tmp_path))]  # fmt: skip
    res = runner.invoke(app, base)
    assert res.exit_code == 0 and "p_pass:" in res.stdout and "not a trial" in res.stdout
    res = runner.invoke(app, [*base[:2], "--firm", "topstep"])
    assert res.exit_code == 2 and "refused:" in res.stderr


def test_pool_runs_when_pool_module_present(tmp_path: Path) -> None:
    pytest.importorskip("pfb.prop.pool")
    other = "synthetic-20261116T000000000001Z-0000abce"
    _write_run(tmp_path / "runs", RUN_ID)
    _write_run(tmp_path / "runs", other, seed=8)
    ledger = _ledger(tmp_path, (RUN_ID, other))
    res = _invoke([*_single(), "--pool", other], tmp_path, _firms(tmp_path), ledger=ledger)
    assert res.exit_code == 0, res.output
    [name] = [p.name for p in (tmp_path / "runs" / RUN_ID / "prop").glob("*_pool-*.json")]
    out = _saved(tmp_path, name)
    assert out["pool"] == [other] and out["days_source"] == [RUN_ID, other]
    assert any(a.endswith(f"/prop/{name}") for a in json.loads(res.stdout)["artifacts"])


def test_matrix_markdown_formats_money_in_usd() -> None:
    est = {"value": 12345.0, "se": 100.0, "n": 10}
    record = {"outputs": {n: ({"value": 0.5, "se": 0.01, "n": 10} if n.startswith("p_") else est)
                          for n in OUTPUTS}, "warnings": ["w1", "3 of 10 attempts timed out"]}  # fmt: skip
    rows = [
        MatrixRow("f", "p", "50k", "x.toml", "ok", None, None, 0.1, record),
        MatrixRow("f", "q", "50k", "y.toml", "skipped", "caps", None, 0.0, None),
    ]
    md = matrix_markdown(matrix_document(rows, {"run_id": "r"}))
    assert "| f | p | 50k | 0.5000 (0.0100) | 12345 (100.0) |" in md
    assert "123 (1)" in md and "- f/q_50k: caps" in md and "- w1" in md and "timed out" not in md


def test_over_cap_run_refused_and_skipped(tmp_path: Path) -> None:
    pytest.importorskip("pfb.prop.pool")
    _write_run(tmp_path / "runs", RUN_ID, position=101)  # the fixtures cap at 100 minis
    firms = _firms(tmp_path)
    res = _invoke(_single(), tmp_path, firms)
    assert res.exit_code == 2
    error = _error(res)
    assert error["code"] == "PoolOverLimit" and "cap of 100" in error["reason"]
    m = json.loads(_invoke(["matrix", RUN_ID, "--mc", "50", "--root", "NQ"], tmp_path, firms).stdout)
    assert all(r["status"] == "skipped" and "contract caps" in r["reason"] for r in m["prop"])


def test_matrix_markdown_marks_lower_bounds() -> None:
    est = {"value": 12345.0, "se": 100.0, "n": 10, "lower_bound": True}
    record = {"outputs": {n: est for n in OUTPUTS}, "warnings": []}
    row = MatrixRow("f", "p", "50k", "x.toml", "ok", None, None, 0.1, record)
    md = matrix_markdown(matrix_document([row], {"run_id": "r"}))
    assert "| 123 (1) lb |" in md and "lb marks a lower bound" in md


def test_first_funded_tier_warning() -> None:
    from pfb.prop.config import load_plan
    from pfb.prop.days import from_day_nets
    from pfb.prop.matrix import first_tier_warning

    repo = Path(__file__).resolve().parents[2]
    plan = load_plan(repo / "config" / "firms" / "topstep" / "combine_xfa_standard_50k.toml")
    assert plan.value("funded.scaling")[0][1] == 2
    days = from_day_nets([10_000] * 4, max_contracts=[1, 2, 3, 3])
    (warning,) = first_tier_warning(days, plan)
    assert "2 of 4 pool days (50.0%) exceed the first funded scaling tier (2 minis)" in warning
    assert first_tier_warning(from_day_nets([10_000] * 4, max_contracts=[2] * 4), plan) == []
