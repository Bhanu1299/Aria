"""plugins/messaging/stubs/discord.py — Discord stub tool descriptors."""
from __future__ import annotations

from aria.core.tool import ToolDescriptor

_MSG = (
    "Discord isn't configured yet. "
    "To enable it, add DISCORD_BOT_TOKEN to your .env and restart Aria."
)


def read_tool() -> ToolDescriptor:
    return ToolDescriptor(
        name="discord_read",
        description="Read recent Discord messages from a channel or user.",
        input_schema={"type": "object", "properties": {
            "contact": {"type": "string", "description": "Channel or user name"},
            "limit": {"type": "integer", "description": "Number of messages"},
        }, "required": ["contact"]},
        execute=lambda p: _MSG,
    )


def send_tool() -> ToolDescriptor:
    return ToolDescriptor(
        name="discord_send",
        description="Send a Discord message to a channel or user.",
        input_schema={"type": "object", "properties": {
            "contact": {"type": "string", "description": "Channel or user name"},
            "message": {"type": "string", "description": "Message text"},
        }, "required": ["contact", "message"]},
        execute=lambda p: _MSG,
    )
