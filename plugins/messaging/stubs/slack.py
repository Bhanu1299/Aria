"""plugins/messaging/stubs/slack.py — Slack stub tool descriptors."""
from __future__ import annotations

from tool import ToolDescriptor

_MSG = (
    "Slack isn't configured yet. "
    "To enable it, add SLACK_BOT_TOKEN to your .env and restart Aria."
)


def read_tool() -> ToolDescriptor:
    return ToolDescriptor(
        name="slack_read",
        description="Read recent Slack messages from a channel or user.",
        input_schema={"type": "object", "properties": {
            "contact": {"type": "string", "description": "Channel or username"},
            "limit": {"type": "integer", "description": "Number of messages"},
        }, "required": ["contact"]},
        execute=lambda p: _MSG,
    )


def send_tool() -> ToolDescriptor:
    return ToolDescriptor(
        name="slack_send",
        description="Send a Slack message to a channel or user.",
        input_schema={"type": "object", "properties": {
            "contact": {"type": "string", "description": "Channel or username"},
            "message": {"type": "string", "description": "Message text"},
        }, "required": ["contact", "message"]},
        execute=lambda p: _MSG,
    )
