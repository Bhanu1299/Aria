"""tests/test_tool_registry.py — ToolRegistry unit tests."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from tool import ToolDescriptor, ToolRegistry


def _make_tool(name: str, available: bool = True, result: str = "ok") -> ToolDescriptor:
    return ToolDescriptor(
        name=name,
        description=f"Tool {name}",
        input_schema={"type": "object", "properties": {}},
        execute=lambda p: result,
        availability=(lambda: available) if not available else None,
    )


def test_register_and_get():
    registry = ToolRegistry()
    tool = _make_tool("search")
    registry.register(tool)
    assert registry.get("search") is tool


def test_get_missing_returns_none():
    registry = ToolRegistry()
    assert registry.get("nonexistent") is None


def test_all_available_includes_available_tools():
    registry = ToolRegistry()
    registry.register(_make_tool("a", available=True))
    available = registry.all_available()
    assert any(t.name == "a" for t in available)


def test_all_available_excludes_unavailable_tools():
    registry = ToolRegistry()
    registry.register(_make_tool("unavail", available=False))
    available = registry.all_available()
    assert not any(t.name == "unavail" for t in available)


def test_availability_none_means_always_available():
    tool = ToolDescriptor(
        name="always",
        description="always on",
        input_schema={},
        execute=lambda p: "ok",
        availability=None,
    )
    assert tool.is_available()


def test_execute_calls_function():
    tool = _make_tool("calc", result="42")
    assert tool.execute({}) == "42"


def test_to_llm_dict_format():
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    tool = ToolDescriptor(
        name="search",
        description="search the web",
        input_schema=schema,
        execute=lambda p: "",
    )
    d = tool.to_llm_dict()
    assert d["name"] == "search"
    assert d["description"] == "search the web"
    assert d["input_schema"] == schema


def test_overwrite_logs_warning(caplog):
    import logging
    registry = ToolRegistry()
    registry.register(_make_tool("dupe"))
    with caplog.at_level(logging.WARNING, logger="tool"):
        registry.register(_make_tool("dupe"))
    assert "already registered" in caplog.text
