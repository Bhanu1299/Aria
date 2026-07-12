"""tasklist.py — session task list + update_tasks tool.

Extracted from Claude Code's TodoWrite pattern: the model writes the full
task list each time (content + pending/in_progress/completed status), the
tool result echoes the rendered list back so the model always sees current
state, and the agent loop uses unfinished() to gate premature "done" claims.
"""
from __future__ import annotations

import logging

from aria.core.tool import ToolDescriptor

logger = logging.getLogger(__name__)

VALID_STATUSES = ("pending", "in_progress", "completed")


class TaskList:
    """Mutable task list owned by one Agent session."""

    def __init__(self) -> None:
        self.tasks: list[dict] = []

    def set_tasks(self, tasks: list) -> None:
        """Replace the whole list. Raises ValueError on malformed input."""
        cleaned: list[dict] = []
        if not isinstance(tasks, list):
            raise ValueError("tasks must be a list")
        for i, item in enumerate(tasks):
            if not isinstance(item, dict):
                raise ValueError(f"task {i + 1} must be an object")
            content = str(item.get("content", "")).strip()
            status = str(item.get("status", "")).strip()
            if not content:
                raise ValueError(f"task {i + 1} has empty content")
            if status not in VALID_STATUSES:
                raise ValueError(
                    f"task {i + 1} has invalid status {status!r} "
                    f"(must be one of {', '.join(VALID_STATUSES)})"
                )
            cleaned.append({"content": content, "status": status})
        self.tasks = cleaned

    def clear(self) -> None:
        self.tasks = []

    def unfinished(self) -> list[dict]:
        return [t for t in self.tasks if t["status"] != "completed"]

    def is_complete(self) -> bool:
        return not self.unfinished()

    def render(self) -> str:
        if not self.tasks:
            return "(task list is empty)"
        return "\n".join(
            f"{i + 1}. [{t['status']}] {t['content']}"
            for i, t in enumerate(self.tasks)
        )


_DESCRIPTION = (
    "Write your task list for the current request. Use this whenever a request "
    "needs 3 or more distinct steps: call it first with all steps as pending, "
    "mark exactly one step in_progress while you work on it, mark it completed "
    "immediately after finishing, and rewrite the list if the plan changes. "
    "Always send the FULL list — it replaces the previous one. Skip this tool "
    "for single-step or conversational requests."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Short description of the step"},
                    "status": {"type": "string", "enum": list(VALID_STATUSES)},
                },
                "required": ["content", "status"],
            },
        },
    },
    "required": ["tasks"],
}


def make_task_tool(tasklist: TaskList) -> ToolDescriptor:
    """Build the update_tasks ToolDescriptor bound to one TaskList."""

    def execute(params: dict) -> str:
        try:
            tasklist.set_tasks(params.get("tasks", []))
        except ValueError as exc:
            return f"Invalid task list, nothing changed: {exc}"
        return f"Task list updated:\n{tasklist.render()}"

    return ToolDescriptor(
        name="update_tasks",
        description=_DESCRIPTION,
        input_schema=_SCHEMA,
        execute=execute,
    )
