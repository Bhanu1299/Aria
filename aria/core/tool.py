"""tool.py — ToolDescriptor and ToolRegistry for Aria's agent layer."""
from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)


class ToolDescriptor:
    """A single capability Aria can invoke."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict,
        execute: Callable[[dict], str],
        availability: Callable[[], bool] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self._execute = execute
        self.availability = availability

    def is_available(self) -> bool:
        if self.availability is None:
            return True
        try:
            return bool(self.availability())
        except Exception:
            return False

    def execute(self, params: dict) -> str:
        return self._execute(params)

    def to_llm_dict(self) -> dict:
        """Convert to Anthropic-style tool dict for llm_client.complete()."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    """Holds all registered tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDescriptor] = {}

    def register(self, tool: ToolDescriptor) -> None:
        if tool.name in self._tools:
            logger.warning("Tool %r already registered — overwriting", tool.name)
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDescriptor | None:
        return self._tools.get(name)

    def all_available(self) -> list[ToolDescriptor]:
        return [t for t in self._tools.values() if t.is_available()]
