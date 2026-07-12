"""agent.py — Tool-loop agent for Aria."""
from __future__ import annotations

from aria.core import paths as _aria_paths

import json
import logging
import os
from datetime import date

from aria.llm import llm_client
from aria.core.tool import ToolRegistry
from aria.core.tasklist import TaskList, make_task_tool
from aria.core.subagent import make_subagent_tool

logger = logging.getLogger(__name__)

_AGENT_MAX_TOOL_CALLS = int(os.getenv("AGENT_MAX_TOOL_CALLS", "25"))
_TOOL_RESULT_MAX_CHARS = int(os.getenv("AGENT_TOOL_RESULT_MAX_CHARS", "8000"))
_IDENTITY_PATH = str(_aria_paths.IDENTITY_JSON)


def truncate_tool_result(text: str, limit: int = _TOOL_RESULT_MAX_CHARS) -> str:
    """Middle-truncate oversized tool output so one tool can't blow the context."""
    if len(text) <= limit:
        return text
    head = text[: (limit * 2) // 3]
    tail = text[-(limit // 3):]
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n... [truncated {omitted} characters] ...\n{tail}"


def _load_name() -> str:
    try:
        with open(_IDENTITY_PATH) as f:
            return json.load(f).get("name", "the user")
    except Exception:
        return "the user"


def _build_system_prompt(memory_context: str = "") -> str:
    base = (
        f"You are Aria, {_load_name()}'s personal voice assistant on their Mac. "
        f"Today is {date.today().isoformat()}.\n"
        "\n"
        "# Voice\n"
        "Everything you say is read aloud by text-to-speech. Speak like a sharp, warm "
        "assistant: plain sentences, no markdown, no bullet points, no URLs unless asked. "
        "Default to 1-3 sentences; go longer only when the user asked for detail. "
        "Never read out raw data dumps — summarize what matters.\n"
        "\n"
        "# Using tools\n"
        "Act, don't narrate. Pick the single best tool; chain multiple tools when a task "
        "needs it (search then summarize, look at the screen then message someone). "
        "Trust tool results over your own assumptions about the user's machine. "
        "If the request is about their screen, windows, or something they're looking at, "
        "use the screen tools — never guess at what's visible. "
        "For anything after your knowledge cutoff or about current events, prices, or "
        "weather, use web_search instead of answering from memory.\n"
        "\n"
        "# Multi-step tasks\n"
        "When a request needs several steps, decide the sequence before the first tool "
        "call, then work one step at a time. After each tool result, check it actually "
        "gave you what that step needed — if not, adjust before moving on rather than "
        "building on a bad result. Before telling the user something is done, make sure "
        "what you have completes the whole request, not just the last step of it.\n"
        "\n"
        "# When things fail\n"
        "If a tool fails or returns nothing useful, try one sensible alternative before "
        "giving up. When you do give up, say plainly what went wrong and what the user "
        "can do — never fail silently and never pretend it worked. If you're uncertain "
        "about a fact, say so briefly instead of guessing confidently.\n"
        "\n"
        "# Task tracking\n"
        "For any request needing three or more steps, call update_tasks first with "
        "the full step list, keep exactly one step in_progress, and mark steps "
        "completed the moment they're done. Never tell the user a job is finished "
        "while a task is still unfinished — finish it or say plainly what's left "
        "and why. For one big self-contained chunk (deep research, a long web flow), "
        "delegate to run_subtask with a standalone prompt and use its summary.\n"
        "\n"
        "# Judgment\n"
        "If a request is ambiguous in a way that changes the action (which contact, which "
        "app, what time), ask one short clarifying question instead of guessing. "
        "For destructive or outward-facing actions — sending messages or emails, deleting "
        "things, submitting forms — state what you're about to do and confirm first. "
        "Casual conversation needs no tools: just talk. "
        "Remember details the user shares; they expect you to know them next time."
    )
    if memory_context:
        return f"{base}\n\n{memory_context}"
    return base


def _get_memory_context(query: str) -> str:
    try:
        from aria.plugins.memory.context_injector import build
        return build(query)
    except Exception:
        return ""


class Agent:
    def __init__(self, registry: ToolRegistry, enable_subagent: bool = True) -> None:
        self.registry = registry
        self._history: list[dict] = []   # cross-turn session history
        self.last_run_tools: list[tuple] = []  # [(tool_name, succeeded)] for flight_recorder
        self.max_tool_calls = _AGENT_MAX_TOOL_CALLS
        self.tasks = TaskList()
        registry.register(make_task_tool(self.tasks))
        if enable_subagent:
            registry.register(make_subagent_tool(registry))

    def run(self, text: str) -> str:
        """Run the tool loop for a user utterance. Returns final spoken response. Never raises."""
        try:
            return self._run(text)
        except Exception as exc:
            logger.error("agent.run unhandled: %s", exc)
            return "Something went wrong. Please try again."

    def _run(self, text: str) -> str:
        from aria.core.compact import should_compact_messages, compact_messages
        self.last_run_tools = []
        if self.tasks.is_complete():
            self.tasks.clear()  # drop a finished list so it can't gate the next request
        if should_compact_messages(self._history):
            self._history = compact_messages(self._history)

        tools = self.registry.all_available()
        tool_dicts = [t.to_llm_dict() for t in tools]
        memory_ctx = _get_memory_context(text)
        system = _build_system_prompt(memory_ctx)
        messages: list[dict] = self._history + [{"role": "user", "content": text}]
        tool_calls_used = 0
        tasks_touched = False   # update_tasks used this run
        gate_fired = False      # completion gate nudges at most once per run

        while tool_calls_used < self.max_tool_calls:
            resp = llm_client.complete(
                messages=messages,
                tools=tool_dicts if tool_dicts else None,
                tier="smart",
                max_tokens=2048,
                system=system,
            )

            if resp.stop_reason == "end_turn":
                # Completion gate (Claude Code pattern): don't accept "done"
                # while tasks written this run are still unfinished.
                if tasks_touched and not gate_fired and not self.tasks.is_complete():
                    gate_fired = True
                    unfinished = "\n".join(
                        f"- {t['content']}" for t in self.tasks.unfinished()
                    )
                    messages.append(
                        {"role": "assistant", "content": resp.text or "Done."}
                    )
                    messages.append({
                        "role": "user",
                        "content": (
                            "<system-reminder>You ended your turn, but these tasks "
                            f"are not completed:\n{unfinished}\n"
                            "Finish them now using tools, or if they truly can't be "
                            "done, update the task list and tell the user plainly "
                            "what's unfinished and why. Never claim success for "
                            "unfinished work.</system-reminder>"
                        ),
                    })
                    continue
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
                    self.last_run_tools.append((tc.name, False))
                else:
                    try:
                        result = descriptor.execute(tc.input)
                        self.last_run_tools.append((tc.name, True))
                        if tc.name == "update_tasks":
                            tasks_touched = True
                    except Exception as exc:
                        result = f"I ran into an issue with {tc.name}: {exc}"
                        logger.error("Tool %r failed: %s", tc.name, exc)
                        self.last_run_tools.append((tc.name, False))

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": truncate_tool_result(str(result)),
                })

            remaining = self.max_tool_calls - tool_calls_used
            if 0 < remaining <= 2:
                tool_results[-1]["content"] += (
                    f"\n\n<system-reminder>Only {remaining} tool call"
                    f"{'s' if remaining != 1 else ''} left in your step budget — "
                    "wrap up: finish the most important remaining work and give "
                    "your final answer.</system-reminder>"
                )

            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

        # Exceeded max tool calls — return what we have
        partial = resp.text.strip() if resp.text else ""
        suffix = "I hit my step limit on that one."
        return f"{partial} {suffix}".strip() if partial else suffix
