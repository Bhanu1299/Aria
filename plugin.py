"""plugin.py — Aria's plugin contract: PluginBase + PluginContext.

A plugin is a folder under plugins/ whose __init__.py defines a
PluginBase subclass. That is the whole contract — plugins.discover()
finds it at startup, builds it via from_context(), and calls register().
No core edits, no registration lists.

Lifecycle (see plugins/__init__.py:discover and main.py):
  phase 1 — plugins with requires_agent=False are built and registered;
  the Agent is then created over the registry;
  phase 2 — plugins with requires_agent=True are built and receive the
  live Agent through ctx.agent (e.g. cron jobs that run prompts).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from tool import ToolRegistry


class PluginContext:
    """Shared services handed to every plugin at startup.

    Any field may be None (tests, partial startup) — plugins must
    degrade gracefully. `agent` is populated only for phase-2 plugins
    (requires_agent=True).
    """

    def __init__(
        self,
        browser=None,
        speaker=None,
        voice_capture=None,
        transcriber=None,
        menubar=None,
        keyterms_prompt: str = "",
        agent=None,
    ) -> None:
        self.browser = browser
        self.speaker = speaker
        self.voice_capture = voice_capture
        self.transcriber = transcriber
        self.menubar = menubar
        self.keyterms_prompt = keyterms_prompt
        self.agent = agent


class PluginBase(ABC):
    # True → the plugin needs the live Agent (ctx.agent) and is built in
    # phase 2, after the Agent exists over the registry.
    requires_agent: bool = False

    @classmethod
    def from_context(cls, ctx: PluginContext) -> "PluginBase":
        """Build the plugin from shared services. Default: no-arg constructor.

        Override when the plugin needs services from ctx.
        """
        return cls()

    @abstractmethod
    def register(self, registry: ToolRegistry) -> None:
        """Register this plugin's tools into registry. Called once at startup."""
