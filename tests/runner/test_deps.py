"""The runner's path mapping: data files enter records as '<store>/<relative path>', never a home path."""

from __future__ import annotations

from pathlib import Path

import pytest

from pfb.runner.deps import provenance_pairs, store_relative


def test_store_relative_names_the_store_and_the_path_inside_it(tmp_path: Path) -> None:
    roots = {"research": tmp_path / "processed", "hist": tmp_path / "hist"}
    path = tmp_path / "hist" / "plan_a" / "f.parquet"
    assert store_relative(str(path), roots) == "hist/plan_a/f.parquet"


def test_a_path_under_no_store_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        store_relative(str(tmp_path / "elsewhere.parquet"), {"research": tmp_path / "processed"})


def test_provenance_pairs_read_the_engine_files_list_in_order(tmp_path: Path) -> None:
    roots = {"research": tmp_path}
    provenance = {
        "files": [
            {"path": str(tmp_path / "b.parquet"), "sha256": "b" * 64, "rows": 3},
            {"path": str(tmp_path / "a.parquet"), "sha256": "a" * 64, "rows": 4},
        ]
    }
    assert provenance_pairs(provenance, roots) == (
        ("research/b.parquet", "b" * 64),
        ("research/a.parquet", "a" * 64),
    )
