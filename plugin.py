"""plugin.py — PluginBase ABC for all Aria plugins."""
from __future__ import annotations

from abc import ABC, abstractmethod

from tool import ToolRegistry


class PluginBase(ABC):
    @abstractmethod
    def register(self, registry: ToolRegistry) -> None:
        """Register this plugin's tools into registry. Called once at startup."""
