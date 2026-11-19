"""The closed-window table: pin, loading, request refusal and open ranges."""

import hashlib
from datetime import date
from pathlib import Path

import pytest
from gateway_synth import WINDOW_TOML, SynthStores

from pfb.data.errors import ClosedWindowRefused, WindowTableMismatch
from pfb.data.windows import CLOSED_WINDOWS_SHA256, DEFAULT_WINDOWS_TOML, INT64_MAX, load_windows

REAL_WINDOW_IDS = {
    "demo_holdout_tail",
    "demo_embargo_tail",
    "demo_holdout_2022",
    "demo_embargo_2022_02",
}


def test_pinned_hash_equals_the_committed_table() -> None:
    actual = hashlib.sha256(DEFAULT_WINDOWS_TOML.read_bytes()).hexdigest()

    assert actual == CLOSED_WINDOWS_SHA256


def test_real_table_loads_with_four_windows() -> None:
    table = load_windows()

    assert len(table.windows) == 4
    assert {w.id for w in table.windows} == REAL_WINDOW_IDS
    assert table.sha256 == CLOSED_WINDOWS_SHA256


def test_real_table_leaves_the_mnq_week_before_the_embargo_open() -> None:
    table = load_windows()

    assert table.request_refusal("MNQ", date(2026, 5, 4), date(2026, 5, 8)) is None
    assert table.request_refusal("MNQ", date(2026, 5, 4), date(2026, 5, 11)) is not None


def test_table_with_other_bytes_is_refused(tmp_path: Path) -> None:
    edited = tmp_path / "closed_windows.toml"
    edited.write_bytes(DEFAULT_WINDOWS_TOML.read_bytes() + b"\n")

    with pytest.raises(WindowTableMismatch):
        load_windows(edited)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('kind = "embargo"', 'kind = "maybe closed"'),
        ('last_trade_date = "2021-02-19"', 'last_trade_date = "2021-02-10"'),
        ('utc_start = "2021-02-12T22:01:00Z"', 'utc_start = "2021-02-12T22:01:00"'),
        ('schema = "pfb.closed_windows/1"', 'schema = "other/1"'),
        ('id = "syn_test"', 'id = "syn_embargo"'),
    ],
)
def test_malformed_table_is_refused(tmp_path: Path, old: str, new: str) -> None:
    table = tmp_path / "bad.toml"
    table.write_text(WINDOW_TOML.replace(old, new, 1), encoding="utf-8")
    sha = hashlib.sha256(table.read_bytes()).hexdigest()

    with pytest.raises(WindowTableMismatch):
        load_windows(table, sha)


def test_open_ended_window_end_is_past_int64(stores: SynthStores) -> None:
    holdout = next(w for w in stores.windows.windows if w.id == "syn_holdout")

    assert holdout.utc_end_ns > INT64_MAX
    assert holdout.last_trade_date == date(9999, 12, 31)


def test_request_refusal_names_the_window(stores: SynthStores) -> None:
    refusal = stores.windows.request_refusal("NQ", date(2021, 2, 16), date(2021, 2, 17))

    assert isinstance(refusal, ClosedWindowRefused)
    assert refusal.window_id == "syn_embargo"


def test_request_ending_the_day_before_a_window_is_open(stores: SynthStores) -> None:
    assert stores.windows.request_refusal("NQ", date(2021, 2, 8), date(2021, 2, 14)) is None


def test_product_scoped_window_applies_only_to_its_roots(stores: SynthStores) -> None:
    assert [w.id for w in stores.windows.closed_windows_for("ZL")] == [
        "syn_holdout",
        "syn_embargo",
        "syn_test",
    ]
    assert stores.windows.request_refusal("NQ", date(2021, 1, 11), date(2021, 1, 15)) is None


def test_open_ranges_cut_the_closed_labels(stores: SynthStores) -> None:
    spans = stores.windows.open_ranges("NQ", date(2021, 2, 8), date(2021, 3, 5))

    assert spans == ((date(2021, 2, 8), date(2021, 2, 14)), (date(2021, 2, 20), date(2021, 2, 28)))


def test_open_ranges_empty_when_fully_closed(stores: SynthStores) -> None:
    assert stores.windows.open_ranges("ZM", date(2021, 2, 1), date(2021, 2, 5)) == ()
