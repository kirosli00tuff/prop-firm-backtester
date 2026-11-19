"""The pfb MCP server in-process: ten thin tools over pfb.app, refusals as JSON.

Every tool is checked against the pfb.app function of the same name: the same parameters and defaults, the
arguments forwarded unchanged and the app's dict returned unchanged. Nothing here reads the real ledger
except the read-only equality checks at the end, which skip while pfb.app is still a stub.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import anyio
import pytest
from mcp import Client
from mcp_types import CallToolResult

import pfb.app as app
from pfb.contract import SCHEMA
from pfb.mcp.server import TOOL_NAMES, build_server

EXPECTED_TOOLS = (
    "list_instruments",
    "list_firms",
    "describe_strategy_template",
    "validate_strategy",
    "register_hypothesis",
    "run_backtest",
    "run_prop_mc",
    "get_run",
    "compare_runs",
    "trial_count",
)

# One call per tool: the MCP arguments and the call pfb.app must receive.
CALLS: dict[str, tuple[dict[str, Any], tuple[Any, ...], dict[str, Any]]] = {
    "list_instruments": ({}, (), {}),
    "list_firms": ({}, (), {}),
    "describe_strategy_template": ({}, (), {}),
    "validate_strategy": ({"path": "strategies/x.py", "seed": 3}, ("strategies/x.py",), {"seed": 3}),
    "register_hypothesis": ({"path": "hypotheses/h.toml"}, ("hypotheses/h.toml",), {}),
    "run_backtest": (
        {
            "strategy": "strategies/x.py",
            "kind": "synthetic",
            "root": "MNQ",
            "window": ["2020-01-02", "2020-03-31"],
            "params": {"buffer_ticks": 4},
            "seed": 3,
            "stress": False,
            "neighbourhood": True,
        },
        ("strategies/x.py",),
        {
            "kind": "synthetic",
            "hypothesis": None,
            "check_id": None,
            "root": "MNQ",
            "window": ["2020-01-02", "2020-03-31"],
            "params": {"buffer_ticks": 4},
            "seed": 3,
            "fees": "topstep",
            "cost_model": "book",
            "firm": None,
            "plan": None,
            "mc": 0,
            "stress": False,
            "baseline": True,
            "walk_forward": True,
            "neighbourhood": True,
        },
    ),
    "run_prop_mc": (
        {
            "run_id": "r1",
            "firm": "topstep",
            "plan": "p",
            "mc": 100,
            "seed": 7,
            "pool": ["r2"],
            "block": 5,
            "soft_daily_loss": "minute_low",
            "contracts_per_mini": 10,
            "copies": 3,
        },  # fmt: skip
        ("r1",),
        {
            "firm": "topstep",
            "plan": "p",
            "mc": 100,
            "seed": 7,
            "pool": ("r2",),
            "block": 5,
            "stress_ticks": 0,
            "generator": "block",
            "soft_daily_loss": "minute_low",
            "root": None,
            "contracts_per_mini": 10,
            "copies": 3,
        },
    ),
    "get_run": ({"run_id": "r1"}, ("r1",), {}),
    "compare_runs": ({"run_ids": ["r1", "r2"]}, (["r1", "r2"],), {}),
    "trial_count": ({}, (), {}),
}


def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    async def go() -> CallToolResult:
        async with Client(build_server()) as client:
            return await client.call_tool(name, arguments)

    return anyio.run(go)


def list_tool_schemas() -> dict[str, dict[str, Any]]:
    async def go() -> dict[str, dict[str, Any]]:
        async with Client(build_server()) as client:
            listed = await client.list_tools()
            return {tool.name: tool.input_schema for tool in listed.tools}

    return anyio.run(go)


def recorder(
    log: list[tuple[str, tuple[Any, ...], dict[str, Any]]], name: str
) -> Callable[..., dict[str, Any]]:
    def fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
        log.append((name, args, kwargs))
        return {"tool": name, "nested": {"n": 1000, "values": [1, 2.5, None]}}

    return fake


def test_server_lists_exactly_the_ten_app_tools() -> None:
    # Act
    schemas = list_tool_schemas()
    # Assert
    assert set(schemas) == set(EXPECTED_TOOLS)
    assert set(TOOL_NAMES) == set(EXPECTED_TOOLS)


@pytest.mark.parametrize("name", EXPECTED_TOOLS)
def test_tool_parameters_and_defaults_match_the_app_function(name: str) -> None:
    # Arrange
    signature = inspect.signature(getattr(app, name))
    # Act
    schema = list_tool_schemas()[name]
    # Assert
    properties = schema.get("properties", {})
    assert list(properties) == list(signature.parameters)
    required = {p.name for p in signature.parameters.values() if p.default is inspect.Parameter.empty}
    assert set(schema.get("required", [])) == required
    for param in signature.parameters.values():
        if param.default is inspect.Parameter.empty:
            continue
        expected = list(param.default) if isinstance(param.default, tuple) else param.default
        assert properties[param.name].get("default") == expected, param.name


@pytest.mark.parametrize("name", EXPECTED_TOOLS)
def test_tool_forwards_its_arguments_and_returns_the_app_result_unchanged(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    log: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    fake = recorder(log, name)
    monkeypatch.setattr(app, name, fake)
    arguments, args, kwargs = CALLS[name]
    # Act
    result = call_tool(name, arguments)
    # Assert
    assert not result.is_error, result.content
    assert result.structured_content == {"tool": name, "nested": {"n": 1000, "values": [1, 2.5, None]}}
    assert log == [(name, args, kwargs)]


@pytest.mark.parametrize("name", EXPECTED_TOOLS)
def test_a_refusal_becomes_the_refusal_json(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    def refuse(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise app.Refused("budget_exceeded", "the hypothesis budget is spent")

    monkeypatch.setattr(app, name, refuse)
    # Act
    result = call_tool(name, CALLS[name][0])
    # Assert
    assert not result.is_error
    assert result.structured_content == {
        "schema": SCHEMA,
        "status": "refused",
        "error": {"code": "budget_exceeded", "reason": "the hypothesis budget is spent"},
    }


def test_an_unexpected_error_is_a_tool_error_and_the_server_keeps_serving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    def boom() -> dict[str, Any]:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(app, "trial_count", boom)
    monkeypatch.setattr(app, "list_firms", lambda: {"firms": []})

    async def go() -> tuple[CallToolResult, CallToolResult]:
        async with Client(build_server()) as client:
            first = await client.call_tool("trial_count", {})
            second = await client.call_tool("list_firms", {})
            return first, second

    # Act
    first, second = anyio.run(go)
    # Assert
    assert first.is_error
    assert second.structured_content == {"firms": []}


def test_missing_required_argument_is_a_tool_error_without_calling_the_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    log: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(app, "get_run", recorder(log, "get_run"))
    # Act
    result = call_tool("get_run", {})
    # Assert
    assert result.is_error
    assert log == []


def app_result_or_skip(name: str) -> dict[str, Any]:
    try:
        return getattr(app, name)()
    except NotImplementedError:
        pytest.skip(f"pfb.app.{name} is not implemented yet")
    except app.Refused as exc:
        return {"schema": SCHEMA, "status": "refused", "error": {"code": exc.code, "reason": exc.reason}}


@pytest.mark.parametrize(
    "name", ["list_instruments", "list_firms", "describe_strategy_template", "trial_count"]
)
def test_read_only_tool_equals_the_real_app_call(name: str) -> None:
    # Arrange
    expected = app_result_or_skip(name)
    # Act
    result = call_tool(name, {})
    # Assert
    assert not result.is_error, result.content
    assert result.structured_content == expected
