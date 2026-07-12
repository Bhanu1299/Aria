"""subagent.py — run_subtask tool: dispatch a focused sub-agent.

Extracted from Claude Code's Task/AgentTool pattern: a big self-contained
chunk of work runs in a fresh agent with its own message history, so the
main conversation only pays for the final summary instead of every
intermediate tool result. The sub-agent sees the same tools except
run_subtask itself (no recursion) and update_tasks (it gets its own).
"""
from __future__ import annotations

import logging

from aria.core.tool import ToolDescriptor, ToolRegistry

logger = logging.getLogger(__name__)

_AGENT_OWNED_TOOLS = ("run_subtask", "update_tasks")

_DESCRIPTION = (
    "Delegate one big self-contained chunk of work to a focused sub-agent "
    "and get back only its final summary. Use for deep research, long web "
    "flows, or anything that would take many tool calls whose intermediate "
    "results you don't need. Give it a complete, standalone prompt: the goal, "
    "all context it needs, and exactly what to report back. It cannot ask "
    "you questions and cannot see this conversation."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            "description": "Standalone task prompt: goal, context, and what to report back",
        },
    },
    "required": ["prompt"],
}


def make_subagent_tool(registry: ToolRegistry) -> ToolDescriptor:
    """Build the run_subtask ToolDescriptor over the main agent's registry."""

    def execute(params: dict) -> str:
        from aria.core.agent import Agent  # here to avoid circular import

        prompt = str(params.get("prompt", "")).strip()
        if not prompt:
            return "run_subtask needs a prompt describing the subtask."
        try:
            sub_registry = ToolRegistry()
            for tool in registry.all_available():
                if tool.name in _AGENT_OWNED_TOOLS:
                    continue
                sub_registry.register(tool)
            sub_agent = Agent(sub_registry, enable_subagent=False)
            result = sub_agent.run(prompt)  # never raises
            logger.info("run_subtask finished (%d chars)", len(result))
            return result
        except Exception as exc:
            logger.error("run_subtask failed: %s", exc)
            return f"The subtask failed to run: {exc}"

    return ToolDescriptor(
        name="run_subtask",
        description=_DESCRIPTION,
        input_schema=_SCHEMA,
        execute=execute,
    )
