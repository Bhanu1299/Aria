"""plugins/messaging/stubs/telegram.py — Telegram stub tool descriptors."""
from __future__ import annotations

from aria.core.tool import ToolDescriptor

_MSG = (
    "Telegram isn't configured yet. "
    "To enable it, add TELEGRAM_BOT_TOKEN to your .env and restart Aria."
)


def read_tool() -> ToolDescriptor:
    return ToolDescriptor(
        name="telegram_read",
        description="Read recent Telegram messages from a contact.",
        input_schema={"type": "object", "properties": {
            "contact": {"type": "string", "description": "Contact name"},
            "limit": {"type": "integer", "description": "Number of messages"},
        }, "required": ["contact"]},
        execute=lambda p: _MSG,
    )


def send_tool() -> ToolDescriptor:
    return ToolDescriptor(
        name="telegram_send",
        description="Send a Telegram message to a contact.",
        input_schema={"type": "object", "properties": {
            "contact": {"type": "string", "description": "Contact name"},
            "message": {"type": "string", "description": "Message text"},
        }, "required": ["contact", "message"]},
        execute=lambda p: _MSG,
    )
