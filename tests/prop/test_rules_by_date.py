"""Rules in force by date. Every plan file carries version and effective_from, every prop result
names the rules that produced it, and a dated day path charges each day under the version in force on it."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path

import numpy as np
import pytest
from _synth import FIRMS_DIR, closes, seq_all

from pfb.prop.config import PlanConfigError, check_versions, load_plan, load_versions, version_paths
from pfb.prop.matrix import McSettings, PlanRefused, evaluate_plan, prop_row
from pfb.prop.rules import RULES_BY_DAY, RULES_NEWEST, run_dated, simulate_eval, simulate_funded
from pfb.prop.types import END_PASS

FIXTURE = Path(__file__).parent / "fixtures" / "ka_static_50k.toml"
PLAN_FILES = sorted(FIRMS_DIR.glob("*/*.toml"))
READ_DATES = {"2026-11-16", "2026-11-17"}  # refresh; the live-rule note of 2026-11-17

# sha256 of each plan file's parsed TOML document (canonical JSON) BEFORE added the version key: the
# files must load unchanged in value. Both pin sets were recomputed once after a wording-only edit of note
# and title strings (the public scrub): every non-text leaf was compared equal beforehand.
VALUE_HASHES = {
    "bulenox/qualification_option1_50k.toml": "b6bb5ce0443b8300cc493a5a2f10ad43fec356d45c6120a3c3959331cf9c437e",  # noqa: E501
    "bulenox/qualification_option2_50k.toml": "ad4840b89375b64a694d0e43990437b1b499bab71e5bbcec721c689135e2638e",  # noqa: E501
    "lucid/lucidflex_50k.toml": "3171dff2e9a6819c3ddf626bda9543dce356d8e0f676d059daafbb868cbee048",
    "lucid/lucidpro_50k.toml": "6b36d0cee4a062fdb30f3dcd2af4785c3a141e237909c4f0f90ff012f9d01c0a",
    # moved the four MFFU pins on purpose: each file gained source M_RESET_E3 (the resets article
    # read 2026-11-17); rapid_eod also P_EOD and two values the page states (funded.start_balance_cents 0,
    # funded.inactivity_days 7; unknown before, alt 0 and 7). No other value changed.
    "mffu/builder_50k.toml": "ef6dafa986838dcdaa903c9f30477570e3ff11cb4e5d7ae66b584a1870717dc9",
    "mffu/pro_50k.toml": "5da24f541d589fb51f14b5acd2387807b99103ff36a8b28597e5de8efa47c1c3",
    "mffu/rapid_50k.toml": "01033aec3bc4cc717b3b1fb01c937fed127644dc0d47eb20dd98692c7097bdee",
    "mffu/rapid_eod_50k.toml": "cf125af3f52d1df3dfab09bf40916ba854b893cf6bd97f5e1b56ef8ebb85d366",
    "topstep/combine_xfa_consistency_50k.toml": "dbbac445c00b4172d0e34fc491c8c59180a604961237d7ec24a56dd769c17c48",  # noqa: E501
    "topstep/combine_xfa_standard_50k.toml": "8c02b0de1fb194758a85e065feb94604ad18e4563fb5879ca4f95427b6ee8253",  # noqa: E501
    "tradeify/growth_50k.toml": "7415d4298b505c1c10d263fcb9e3a6686144ce70fa25cc7e8f047c8e624c652a",
    "tradeify/select_daily_50k.toml": "b569f3d1c4f98e588da8301dde38587d6785032805e155729e10d0abf8e89c80",
    "tradeify/select_flex_50k.toml": "bd9578e352867df3dc5b52b2fd765b1d753216dfcaf77f748e1c365a29c015cf",
}
# A later change moved every pin on purpose: each file gained funded.funded_reset_cents (Topstep with its
# source R106) and the LucidFlex funded DLL became an assumed unknown (alt [120000], the value it held). The
# pins above still hold once those additions are stripped (_strip_unknowns): no existing rule value changed.
# These are the full documents (without the version key) after that change.
VALUE_HASHES_WITH_UNKNOWNS = {
    "bulenox/qualification_option1_50k.toml": "f46ac96f7c12db6f5028ee712747b04f1c240dfad735df8eaddd190b559c95ef",  # noqa: E501
    "bulenox/qualification_option2_50k.toml": "de601df3a65940f5a098504044e420e8342f2e839f93be726c16ed7cb4543e97",  # noqa: E501
    "lucid/lucidflex_50k.toml": "dbf987b4cfed659dd850d953e00272d6115f369797b01dad3f7ec0d05d18f4f6",
    "lucid/lucidpro_50k.toml": "cc9bc17ba3c46258beb8ddd6ce8131503fa402b7825a7a25127bdd23eeb89e5c",
    "mffu/builder_50k.toml": "4f2adfb0c1c2899922a90d54ac45960e68fceddd51686cc46439c1a304c4776f",
    "mffu/pro_50k.toml": "febf8abd6db6f0cef4e419c485cd76abb08c3522e425ec2a6fffd540d07b4416",
    "mffu/rapid_50k.toml": "8a79605c8db721f70a6d2d1eead1d90f1ae67efd2fabdfa68b8f1152dd3e1039",
    "mffu/rapid_eod_50k.toml": "24d4fd70e67ff2eb2ccacd5e31ea353a21e438ee141c5e2278bc9b0d4d5555bb",
    "topstep/combine_xfa_consistency_50k.toml": "7ab083b8945d334d38d8327911f9307505b22752c642b8535a99ad09dbd9d084",  # noqa: E501
    "topstep/combine_xfa_standard_50k.toml": "dda96f906cbbb71840c7efb01687f40399cc1e47bb718a0ef4820a96233b29b3",  # noqa: E501
    "tradeify/growth_50k.toml": "473c5ef28f618e25154dd688f38fad28e7f89fa8dc0a95c09b975ae03d6ce27c",
    "tradeify/select_daily_50k.toml": "74f178a5c05cb445a7552293e3c17c36628083303ef7a714b97c943906ddb41f",
    "tradeify/select_flex_50k.toml": "cca2b3d0600ca4b1b5a940fe42102a980a525d14d9694f49fd15ac32e99eb223",
}
# The LucidFlex funded DLL leaf as it stood before (a value, not an unknown).
FLEX_DLL_BEFORE_UNKNOWNS = {
    "v": 120000, "src": "S08", "kind": "conservative",
    "note": "config row for the DLL below the initial trail: $1,200; the Flex rule above the trail is not "
    "published (LucidPro uses 60% of peak EOD profit, larger); fixed 1,200 kept throughout",
}  # fmt: skip


def _key(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"


def _canonical_hash(doc: dict) -> str:
    return hashlib.sha256(
        json.dumps(doc, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()


def _strip_unknowns(path: Path, doc: dict) -> dict:
    """The document without what that change added: funded.funded_reset_cents, Topstep's source R106, and the
    LucidFlex funded DLL leaf put back as it was (its assumed alt equals the old value: asserted here)."""
    doc["funded"].pop("funded_reset_cents")
    if path.parent.name == "topstep":
        doc["sources"].pop("R106")
    if path.name == "lucidflex_50k.toml":
        leaf = doc["funded"]["daily_loss_cents"]
        assert leaf["assume"] is True and leaf["alt"][0] == FLEX_DLL_BEFORE_UNKNOWNS["v"]
        doc["funded"]["daily_loss_cents"] = dict(FLEX_DLL_BEFORE_UNKNOWNS)
    return doc


# ------------------------------------------------------------------- the files ----
def test_thirteen_plan_files() -> None:
    assert sorted(_key(p) for p in PLAN_FILES) == sorted(VALUE_HASHES)


@pytest.mark.parametrize("path", PLAN_FILES, ids=_key)
def test_plan_file_loads_unchanged_in_value(path: Path) -> None:
    doc = tomllib.loads(path.read_text())
    doc.pop("version")
    assert _canonical_hash(doc) == VALUE_HASHES_WITH_UNKNOWNS[_key(path)]
    assert _canonical_hash(_strip_unknowns(path, doc)) == VALUE_HASHES[_key(path)]  # no value changed


@pytest.mark.parametrize("path", PLAN_FILES, ids=_key)
def test_plan_file_carries_version_and_effective_from(path: Path) -> None:
    plan = load_plan(path)
    assert plan.version >= 1 and plan.rules_read in READ_DATES and plan.read_in
    assert plan.effective_from <= plan.rules_read
    label = plan.rules_label()
    assert label["plan_file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest() == plan.sha256
    assert (label["firm"], label["plan"], label["plan_version"]) == (plan.firm, plan.plan, plan.version)


@pytest.mark.parametrize("path", PLAN_FILES, ids=_key)
def test_newest_version_is_the_plan_file(path: Path) -> None:
    versions = load_versions(path)
    assert versions.newest.path == str(path)
    assert versions.newest.version == len(versions.versions)


def test_lucidflex_has_the_dated_50_percent_version() -> None:
    versions = load_versions(FIRMS_DIR / "lucid" / "lucidflex_50k.toml")
    got = [
        (v.version, v.effective_from, v.value("eval.consistency_bps"), v.status) for v in versions.versions
    ]
    assert got == [(1, "2026-08-31", 5000, "provisional"), (2, "2026-10-19", 4000, "sourced")]
    assert (versions.in_force(20261018), versions.in_force(20261019), versions.in_force(20261117)) == (
        0,
        1,
        1,
    )
    assert versions.in_force(20240102) == 0 and versions.before_earliest(20240102)
    assert not versions.before_earliest(20260831)
    assert versions.earliest.is_unknown("fees.eval_price_cents")  # the old price is not recorded


def test_only_lucidflex_has_older_versions() -> None:
    older = {_key(p) for p in PLAN_FILES if version_paths(p)}
    assert older == {"lucid/lucidflex_50k.toml"}


# -------------------------------------------------------------------------- version validation ----
def _text(n: int = 1, read: str = "2026-11-16", eff: str = "2026-11-16") -> str:
    text = FIXTURE.read_text()
    line = f'version = {{ n = {n}, rules_read = "{read}", read_in = "test" }}'
    text = re.sub(r"^version = .*$", line, text, flags=re.M)
    return re.sub(
        r"^effective_from = .*$", f'effective_from = {{ v = "{eff}", src = "S01" }}', text, flags=re.M
    )


def _problems(tmp_path: Path, text: str) -> str:
    path = tmp_path / FIXTURE.name
    path.write_text(text)
    with pytest.raises(PlanConfigError) as info:
        load_plan(path)
    return " | ".join(info.value.problems)


def test_version_is_required_and_checked(tmp_path: Path) -> None:
    no_version = re.sub(r"^version = .*\n", "", FIXTURE.read_text(), flags=re.M)
    assert "missing top-level keys ['version']" in _problems(tmp_path, no_version)
    assert "version.n 0 must be an int >= 1" in _problems(tmp_path, _text(n=0))
    assert "version.rules_read 2026-11-15 is before effective_from" in _problems(
        tmp_path, _text(read="2026-11-15")
    )
    bad_key = _text().replace('read_in = "test"', 'read_in = "test", by = "x"')
    assert "version: unknown keys ['by']" in _problems(tmp_path, bad_key)


def _versions_dir(tmp_path: Path, rows: list[tuple[int, str, int]]) -> Path:
    """Plan files of a synthetic plan, one per (n, effective_from, consistency bps); the last is newest."""
    newest = tmp_path / "topstep" / FIXTURE.name
    for i, (n, eff, bps) in enumerate(rows):
        text = _text(n=n, read="2026-11-16", eff=eff)
        text = text.replace(
            'consistency_type = { v = "none", src = "S01" }',
            'consistency_type = { v = "best_day_frac_of_profit", src = "S01" }',
            1,
        )
        text = text.replace('consistency_bps = { v = "none", src = "S01" }',
                            f'consistency_bps = {{ v = {bps}, src = "S01" }}', 1)  # fmt: skip
        path = newest if i == len(rows) - 1 else tmp_path / "topstep" / "versions" / eff / FIXTURE.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return newest


def test_check_versions_refuses_bad_sets(tmp_path: Path) -> None:
    newest = _versions_dir(tmp_path, [(1, "2024-01-02", 5000), (2, "2024-01-05", 4000)])
    old = load_plan(next(iter(version_paths(newest))))
    new = load_plan(newest)
    with pytest.raises(PlanConfigError, match="the plan file is not the newest"):
        check_versions([old, new], str(old.path))
    swapped = _versions_dir(tmp_path / "b", [(2, "2024-01-02", 5000), (1, "2024-01-05", 4000)])
    with pytest.raises(PlanConfigError, match="does not ascend"):
        load_versions(swapped)


# --------------------------------------------------------------- a dated day path ----
# Day nets (weekdays from 2024-01-02): +450, +300, +250 USD reach the 1,000 target on 2024-01-04 with a best
# day of 45%: a pass under the 50% version, a fail under the 40% version. Then flat days.
NETS = [45_000, 30_000, 25_000, 0, 0, 0, 0, 0]


@pytest.mark.parametrize(("v2_from", "passes"), [("2024-01-05", True), ("2024-01-04", False)])
def test_boundary_day_decides_pass_or_fail(tmp_path: Path, v2_from: str, passes: bool) -> None:
    # Arrange: version 1 (50%) from 2024-01-02, version 2 (40%) from v2_from
    newest = _versions_dir(tmp_path, [(1, "2024-01-02", 5000), (2, v2_from, 4000)])
    versions = load_versions(newest)
    days = closes(NETS)
    # Act
    run = run_dated(versions, days)
    # Assert
    assert (run.eval.end == END_PASS) is passes
    assert run.rules_in_force == RULES_BY_DAY and run.days_before_earliest == 0 and run.warnings == ()
    if passes:
        assert run.eval.days == 3 and run.day_version[:3] == (0, 0, 0)
    else:
        assert run.day_version[:3] == (0, 0, 1)
    assert sum(lab["days_charged"] for lab in run.labels) == len(run.day_version)
    for lab in run.labels:
        plan = versions.versions[lab["plan_version"] - 1]
        assert lab["plan_file_sha256"] == hashlib.sha256(Path(plan.path).read_bytes()).hexdigest()
        assert lab["effective_from"] == plan.effective_from and lab["firm"] == "topstep"


def test_days_before_the_earliest_version_are_counted(tmp_path: Path) -> None:
    newest = _versions_dir(tmp_path, [(1, "2024-01-03", 5000), (2, "2024-01-05", 4000)])
    run = run_dated(load_versions(newest), closes(NETS))
    assert run.eval.end == END_PASS and run.days_before_earliest == 1
    assert run.day_version[0] == 0  # 2024-01-02 is charged under the earliest version
    assert len(run.warnings) == 1 and run.warnings[0].startswith("rules: 1 of ")
    assert "predate the earliest rules version" in run.warnings[0]


def test_single_version_dated_path_matches_the_reference(tmp_path: Path) -> None:
    newest = _versions_dir(tmp_path, [(1, "2024-01-02", 5000)])
    versions = load_versions(newest)
    days = closes(NETS)
    run = run_dated(versions, days)
    assert run.eval == simulate_eval(versions.newest, days, seq_all(days))
    rest = np.arange(run.eval.days, days.n_days, dtype=np.int64)
    assert run.funded == simulate_funded(versions.newest, days, rest)
    assert [lab["plan_version"] for lab in run.labels] == [1]


# ----------------------------------------------- an undated path uses the newest version ----
def test_resampled_record_names_the_newest_version_and_says_so(tmp_path: Path) -> None:
    newest = _versions_dir(tmp_path, [(1, "2024-01-02", 5000), (2, "2024-01-05", 4000)])
    days = closes(NETS)
    settings = McSettings(n_paths=40, seed=1, kind="synthetic")
    record, _ = evaluate_plan(newest, days, settings, repo_root=tmp_path, cap_check=lambda _d, _p: [])
    assert (record["plan_version"], record["effective_from"], record["rules_read"]) == (
        2,
        "2024-01-05",
        "2026-11-16",
    )
    assert record["plan_file_sha256"] == hashlib.sha256(newest.read_bytes()).hexdigest()
    assert record["rules_in_force"] == RULES_NEWEST
    [warning] = [w for w in record["warnings"] if w.startswith("rules: resampled day paths are undated")]
    assert "version 2, effective_from 2024-01-05" in warning and "3 of 8 source days predate it" in warning
    row = prop_row("topstep", "ka_static_50k", "ok", None, record)
    assert (row["plan_version"], row["effective_from"], row["rules_in_force"]) == (
        2,
        "2024-01-05",
        RULES_NEWEST,
    )
    assert row["plan_file_sha256"] == record["plan_file_sha256"]


def test_skipped_row_has_null_labels() -> None:
    row = prop_row("topstep", "x_50k", "skipped", "why", None)
    assert all(
        row[k] is None for k in ("plan_version", "effective_from", "plan_file_sha256", "rules_in_force")
    )


def test_a_plan_file_older_than_a_dated_version_is_refused(tmp_path: Path) -> None:
    # Arrange: the plan file says 2024-01-02 but versions/ holds a newer 2024-01-05 version
    newest = _versions_dir(tmp_path, [(1, "2024-01-05", 4000), (2, "2024-01-02", 5000)])
    # Act / Assert: the undated run would not use the newest version, so the plan is refused
    with pytest.raises(PlanRefused, match="not the newest version"):
        evaluate_plan(newest, closes(NETS), McSettings(n_paths=10, seed=1, kind="synthetic"),
                      repo_root=tmp_path, cap_check=lambda _d, _p: [])  # fmt: skip
