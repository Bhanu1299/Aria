"""agent.py — Tool-loop agent for Aria."""
from __future__ import annotations

import json
import logging
import os
from datetime import date

from llm import llm_client
from tool import ToolRegistry

logger = logging.getLogger(__name__)

_AGENT_MAX_TOOL_CALLS = int(os.getenv("AGENT_MAX_TOOL_CALLS", "10"))
_IDENTITY_PATH = os.path.join(os.path.dirname(__file__), "identity.json")


def _load_name() -> str:
    try:
        with open(_IDENTITY_PATH) as f:
            return json.load(f).get("name", "the user")
    except Exception:
        return "the user"


def _build_system_prompt(memory_context: str = "") -> str:
    base = (
        "You are Aria, a personal voice assistant running on macOS.\n"
        "The user speaks to you — your response will be read aloud.\n"
        "Keep answers short and conversational. No markdown.\n"
        "Use tools when needed. You may chain multiple tools.\n"
        f"Today is {date.today().isoformat()}. User: {_load_name()}."
    )
    if memory_context:
        return f"{base}\n\n{memory_context}"
    return base


def _get_memory_context(query: str) -> str:
    try:
        from plugins.memory.context_injector import build
        return build(query)
    except Exception:
        return ""


class Agent:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self._history: list[dict] = []   # cross-turn session history

    def run(self, text: str) -> str:
        """Run the tool loop for a user utterance. Returns final spoken response. Never raises."""
        try:
            return self._run(text)
        except Exception as exc:
            logger.error("agent.run unhandled: %s", exc)
            return "Something went wrong. Please try again."

    def _run(self, text: str) -> str:
        from compact import should_compact_messages, compact_messages
        if should_compact_messages(self._history):
            self._history = compact_messages(self._history)

        tools = self.registry.all_available()
        tool_dicts = [t.to_llm_dict() for t in tools]
        memory_ctx = _get_memory_context(text)
        system = _build_system_prompt(memory_ctx)
        messages: list[dict] = self._history + [{"role": "user", "content": text}]
        tool_calls_used = 0

        while tool_calls_used < _AGENT_MAX_TOOL_CALLS:
            resp = llm_client.complete(
                messages=messages,
                tools=tool_dicts if tool_dicts else None,
                tier="smart",
                max_tokens=2048,
                system=system,
            )

            if resp.stop_reason == "end_turn":
                answer = resp.text.strip() or "I didn't understand that. Could you rephrase?"
                self._history += [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": answer},
                ]
                return answer

            if resp.stop_reason == "error":
                return resp.text

            if resp.stop_reason != "tool_use" or not resp.tool_calls:
                answer = resp.text.strip() or "I didn't understand that. Could you rephrase?"
                self._history += [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": answer},
                ]
                return answer

            # Build assistant turn
            assistant_content: list[dict] = []
            if resp.text:
                assistant_content.append({"type": "text", "text": resp.text})
            for tc in resp.tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.input,
                })

            # Execute each tool call
            tool_results: list[dict] = []
            for tc in resp.tool_calls:
                tool_calls_used += 1
                descriptor = self.registry.get(tc.name)
                if descriptor is None:
                    result = f"Unknown tool: {tc.name}"
                    logger.warning("Unknown tool called: %r", tc.name)
                else:
                    try:
                        result = descriptor.execute(tc.input)
                    except Exception as exc:
                        result = f"I ran into an issue with {tc.name}: {exc}"
                        logger.error("Tool %r failed: %s", tc.name, exc)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result,
                })

            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

        # Exceeded max tool calls — return what we have
        partial = resp.text.strip() if resp.text else ""
        suffix = "I hit my step limit on that one."
        return f"{partial} {suffix}".strip() if partial else suffix
