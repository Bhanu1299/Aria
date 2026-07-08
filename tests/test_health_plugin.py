"""Tests for plugins/health — Aria reports on her own reliability."""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aria.core.tool import ToolRegistry
from aria.plugins.health import HealthPlugin


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    HealthPlugin().register(reg)
    return reg


def test_registers_self_report_tool():
    assert "self_report" in {t.name for t in _registry().all_available()}


def test_self_report_returns_spoken_summary():
    reg = _registry()
    with patch("aria.observability.flight_recorder.spoken_report", return_value="I handled 10 commands.") as mock_rep:
        result = reg.get("self_report").execute({"days": 7})
    assert result == "I handled 10 commands."
    mock_rep.assert_called_once_with(days=7)


def test_self_report_defaults_to_seven_days_and_never_raises():
    reg = _registry()
    with patch("aria.observability.flight_recorder.spoken_report", side_effect=RuntimeError("boom")):
        result = reg.get("self_report").execute({})
    assert isinstance(result, str) and result
