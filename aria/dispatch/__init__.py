"""dispatch — the single entry point for every command.

Rules (patterns.json) send on-device commands straight to a tool with zero LLM
tokens; everything else goes to the agent, whose LLM calls go through GateLLM.
See docs/superpowers/specs/2026-10-18-dispatcher-gateway-design.md.
"""
from __future__ import annotations

from aria.dispatch.dispatcher import DispatchResult, Dispatcher
from aria.dispatch.rules import RuleSet

__all__ = ["Dispatcher", "DispatchResult", "RuleSet"]
