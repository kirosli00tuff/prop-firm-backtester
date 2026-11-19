"""The [cost] table's event_costs and event_stance (pfb.registry.hypfile._cost)."""

from __future__ import annotations

from typing import Any

import pytest

from pfb.registry import hypfile
from pfb.registry.errors import HypothesisInvalid
from pfb.registry.inputs import EVENT_COSTS_NOT_WIRED


def _raw(model: str, event_costs: Any, stance: str) -> dict[str, Any]:
    return {"cost": {"model": model, "fees": "topstep", "event_costs": event_costs, "event_stance": stance}}


def test_off_with_a_stance_is_as_before() -> None:
    decl = hypfile._cost(_raw("book", "off", "no release minutes traded"), "h.toml")
    assert (decl.model, decl.fees, decl.event_costs, decl.event_stance) == (
        "book", "topstep", "off", "no release minutes traded"
    )  # fmt: skip


@pytest.mark.parametrize("stance", ["", "   "])
def test_off_without_a_stance_is_refused(stance: str) -> None:
    with pytest.raises(HypothesisInvalid, match="event_stance"):
        hypfile._cost(_raw("book", "off", stance), "h.toml")


@pytest.mark.parametrize("value", ["maybe", "ON", True, 1])
def test_other_values_are_refused(value: Any) -> None:
    with pytest.raises(HypothesisInvalid, match='"off" or "on"'):
        hypfile._cost(_raw("book", value, "x"), "h.toml")


def test_on_is_accepted_now_the_switch_is_set() -> None:
    # part 2: the engine charges "on" per bar and the runner and registry carry it
    assert hypfile.EVENT_COSTS_ENGINE_WIRED is True
    decl = hypfile._cost(_raw("book", "on", ""), "h.toml")
    assert (decl.model, decl.event_costs, decl.event_stance) == ("book", "on", "")


def test_on_is_refused_while_the_switch_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hypfile, "EVENT_COSTS_ENGINE_WIRED", False)
    with pytest.raises(HypothesisInvalid) as info:
        hypfile._cost(_raw("book", "on", ""), "h.toml")
    assert EVENT_COSTS_NOT_WIRED in str(info.value)


@pytest.mark.parametrize("model", ["fixed:1", "stress:fixed:1:1"])
def test_on_needs_a_book_based_model(model: str) -> None:
    with pytest.raises(HypothesisInvalid, match="book-based"):
        hypfile._cost(_raw(model, "on", ""), "h.toml")


@pytest.mark.parametrize("model", ["book", "stress:book:1", "stress:stress:book:1:0.5"])
def test_on_is_accepted_once_wired_and_the_stance_may_be_empty(
    model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hypfile, "EVENT_COSTS_ENGINE_WIRED", True)
    decl = hypfile._cost(_raw(model, "on", ""), "h.toml")
    assert (decl.model, decl.event_costs, decl.event_stance) == (model, "on", "")
