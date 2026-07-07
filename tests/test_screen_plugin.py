"""Tests for plugins/screen — screen tools registered on the agent tool loop."""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool import ToolRegistry
from plugins.screen import ScreenPlugin


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    ScreenPlugin().register(reg)
    return reg


def test_registers_screen_look_and_screen_point_tools():
    reg = _registry()
    names = {t.name for t in reg.all_available()}
    assert "screen_look" in names
    assert "screen_point" in names


def test_screen_look_calls_screen_qa_answer():
    reg = _registry()
    with patch("screen_qa.answer", return_value="A code editor with a traceback.") as mock_answer:
        result = reg.get("screen_look").execute({"question": "what's on my screen"})
    assert result == "A code editor with a traceback."
    mock_answer.assert_called_once_with("what's on my screen")


def test_screen_point_calls_explain_visual():
    reg = _registry()
    with patch("screen_qa.explain_visual", return_value="The save button is top right.") as mock_ev:
        result = reg.get("screen_point").execute({"question": "show me where the save button is"})
    assert result == "The save button is top right."
    mock_ev.assert_called_once_with("show me where the save button is")


def test_tools_never_raise_on_backend_explosion():
    reg = _registry()
    with patch("screen_qa.answer", side_effect=RuntimeError("boom")):
        result = reg.get("screen_look").execute({"question": "x"})
    assert isinstance(result, str) and result


def test_tool_schemas_are_valid_for_llm():
    reg = _registry()
    for name in ("screen_look", "screen_point"):
        d = reg.get(name).to_llm_dict()
        assert d["input_schema"]["type"] == "object"
        assert "question" in d["input_schema"]["properties"]
        assert d["description"]
