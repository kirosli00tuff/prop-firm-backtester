"""The exact-name path rule and config/paths.toml loading."""

import os
from pathlib import Path

import pytest
from gateway_synth import SynthStores, hist_name, research_name, step2_name

from pfb.data.errors import PathRefused, PathsConfigError
from pfb.data.paths import (
    ERA_HIST,
    ERA_MES_CONFIRMATION,
    ERA_RESEARCH,
    ERA_STEP2,
    check_path,
    load_paths,
)

MES_CONFIRMATION = "ohlcv-1m_MES_v_0_2019-01-02_2022-01-31_confirmation.parquet"


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def test_each_era_pattern_is_accepted(stores: SynthStores) -> None:
    p = stores.paths
    cases = [
        (p.research / "NQ" / research_name("NQ"), ("NQ", ERA_RESEARCH, None)),
        (p.research / "MES" / MES_CONFIRMATION, ("MES", ERA_MES_CONFIRMATION, None)),
        (p.step2 / "NQ" / step2_name("NQ"), ("NQ", ERA_STEP2, None)),
        (p.hist / "plan_a" / "NQ" / hist_name("NQ", "plan_a"), ("NQ", ERA_HIST, "plan_a")),
        (p.hist / "plan_b" / "UB" / hist_name("UB", "plan_b"), ("UB", ERA_HIST, "plan_b")),
    ]

    for path, expected in cases:
        found = check_path(_touch(path), p)
        assert (found.root, found.era, found.plan) == expected


@pytest.mark.parametrize(
    "relative",
    [
        "research/ES/" + research_name("NQ"),  # root differs from the parent directory
        "research/NQ/superseded/" + research_name("NQ"),
        "research/NQ/" + research_name("NQ") + ".outright-pattern-bug",
        "research/NQ/" + research_name("NQ").replace("research.parquet", "research.csv"),
        "step2/NQ/ohlcv-1m_NQ_v_0_2019-01-03_2022-01-31_step2.parquet",
        "hist/plan_a/NQ/" + hist_name("NQ", "plan_b"),  # plan dir and name disagree
        "hist/plan_c/ES/" + hist_name("ES", "plan_a"),
        "hist/plan_a/NQ/ohlcv-1m_NQ_v_0_2010-13-45_2018-12-31_plan_a.parquet",
        "research/sealed/NQ/" + research_name("NQ"),
        "research/NQ/derived/" + research_name("NQ"),
        "research/NQ/" + step2_name("NQ"),
    ],
)
def test_bad_names_and_forbidden_parts_are_refused(stores: SynthStores, relative: str) -> None:
    path = _touch(stores.base / relative)

    with pytest.raises(PathRefused):
        check_path(path, stores.paths)


def test_path_outside_the_roots_is_refused(stores: SynthStores) -> None:
    outside = _touch(stores.base / "elsewhere" / "NQ" / research_name("NQ"))

    with pytest.raises(PathRefused, match="not under any store root"):
        check_path(outside, stores.paths)


def test_symlink_escaping_a_root_is_refused(stores: SynthStores) -> None:
    target = _touch(stores.base / "elsewhere" / "NQ" / research_name("NQ"))
    link = stores.paths.research / "NQ" / research_name("NQ")
    link.parent.mkdir(parents=True)
    link.symlink_to(target)

    with pytest.raises(PathRefused):
        check_path(link, stores.paths)


def test_symlink_into_a_sealed_directory_is_refused(stores: SynthStores) -> None:
    target = _touch(stores.base / "sealed" / "NQ" / research_name("NQ"))
    link = stores.paths.research / "NQ" / research_name("NQ")
    link.parent.mkdir(parents=True)
    link.symlink_to(target)

    with pytest.raises(PathRefused, match="sealed"):
        check_path(link, stores.paths)


def test_relative_path_is_refused(stores: SynthStores) -> None:
    with pytest.raises(PathRefused):
        check_path(Path("NQ") / research_name("NQ"), stores.paths)


def test_load_paths_reads_all_keys(tmp_path: Path) -> None:
    toml = tmp_path / "paths.toml"
    toml.write_text(
        '[stores]\nresearch = "/a"\nstep2 = "/b"\nhist = "/c"\n'
        '[upstream]\nroot = "/pe"\n[costs]\nbook_table = "/pe/costs.json"\n',
        encoding="utf-8",
    )

    paths = load_paths(toml)

    assert (paths.research, paths.step2, paths.hist) == (Path("/a"), Path("/b"), Path("/c"))
    assert paths.upstream_root == Path("/pe")


def test_load_paths_refuses_missing_key_and_missing_file(tmp_path: Path) -> None:
    toml = tmp_path / "paths.toml"
    toml.write_text('[stores]\nresearch = "/a"\n', encoding="utf-8")

    with pytest.raises(PathsConfigError):
        load_paths(toml)
    with pytest.raises(PathsConfigError):
        load_paths(tmp_path / "absent.toml")


def test_load_paths_turns_undecodable_bytes_into_paths_config_error(tmp_path: Path) -> None:
    toml = tmp_path / "paths.toml"
    toml.write_bytes(b'[stores]\nresearch = "/a\xff\xfe"\n')

    with pytest.raises(PathsConfigError, match="UTF-8"):
        load_paths(toml)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-000 file")
def test_load_paths_turns_an_unreadable_file_into_paths_config_error(tmp_path: Path) -> None:
    toml = tmp_path / "paths.toml"
    toml.write_text('[stores]\nresearch = "/a"\n', encoding="utf-8")
    toml.chmod(0)
    try:
        with pytest.raises(PathsConfigError, match="cannot be read"):
            load_paths(toml)
    finally:
        toml.chmod(0o600)


def test_example_paths_file_has_placeholders_only() -> None:
    example = Path(__file__).resolve().parents[2] / "config" / "paths.example.toml"

    paths = load_paths(example)

    assert all(str(p).startswith("/path/to/") for p in paths.store_roots().values())
