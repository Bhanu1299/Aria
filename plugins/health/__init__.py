"""plugins/health/__init__.py — HealthPlugin: Aria reports on her own reliability.

Tool:
  self_report — spoken summary of recent command volume, failure rate, and
                the most failure-prone tool, from the local flight recorder.
"""
from __future__ import annotations

import logging

import plugin as _plugin_base
from tool import ToolDescriptor, ToolRegistry

logger = logging.getLogger(__name__)


class HealthPlugin(_plugin_base.PluginBase):

    def register(self, registry: ToolRegistry) -> None:
        registry.register(self._self_report_tool())

    def _self_report_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            try:
                import flight_recorder
                days = int(params.get("days", 7) or 7)
                days = max(1, min(days, 90))
                return flight_recorder.spoken_report(days=days)
            except Exception as exc:
                logger.error("self_report failed: %s", exc)
                return "I couldn't read my flight log right now."

        return ToolDescriptor(
            name="self_report",
            description=(
                "Report on Aria's own recent reliability: how many commands were handled, "
                "failure rate, and the most failure-prone tool. Use when the user asks "
                "'how have you been performing', 'how reliable have you been', "
                "'what's been failing', or 'health report'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Lookback window in days (default 7)",
                    },
                },
                "required": [],
            },
            execute=execute,
        )
