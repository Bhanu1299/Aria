"""plugins/messaging/__init__.py — MessagingPlugin: iMessage, WhatsApp, and stubs."""
from __future__ import annotations

import logging

import aria.core.plugin as _plugin_base
from aria.core.tool import ToolDescriptor, ToolRegistry

logger = logging.getLogger(__name__)


def _str_prop(desc: str) -> dict:
    return {"type": "string", "description": desc}


def _int_prop(desc: str) -> dict:
    return {"type": "integer", "description": desc}


class MessagingPlugin(_plugin_base.PluginBase):

    def register(self, registry: ToolRegistry) -> None:
        from aria.plugins.messaging.imessage import iMessageClient
        from aria.plugins.messaging.whatsapp import WhatsAppClient
        from aria.plugins.messaging.stubs import telegram, discord, slack

        im = iMessageClient()
        wa = WhatsAppClient()

        registry.register(self._imessage_read(im))
        registry.register(self._imessage_send(im))
        registry.register(self._imessage_contacts(im))
        registry.register(self._whatsapp_read(wa))
        registry.register(self._whatsapp_send(wa))
        registry.register(self._whatsapp_contacts(wa))
        registry.register(telegram.read_tool())
        registry.register(telegram.send_tool())
        registry.register(discord.read_tool())
        registry.register(discord.send_tool())
        registry.register(slack.read_tool())
        registry.register(slack.send_tool())

    # ------------------------------------------------------------------
    # iMessage tools
    # ------------------------------------------------------------------

    def _imessage_read(self, client) -> ToolDescriptor:
        def execute(params: dict) -> str:
            result = client.read(params["contact"], int(params.get("limit", 5)))
            if isinstance(result, str):
                return result
            lines = []
            for m in result:
                lines.append(f"{m['timestamp']} {m['sender']}: {m['text']}")
            return "\n".join(lines) if lines else "No messages found."

        return ToolDescriptor(
            name="imessage_read",
            description=(
                "Read recent iMessages from a contact. "
                "Use for: 'read my last message from [contact]', 'what did [contact] say'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "contact": _str_prop("Contact name, phone number, or email"),
                    "limit": _int_prop("Number of messages to read (default 5, max 20)"),
                },
                "required": ["contact"],
            },
            execute=execute,
        )

    def _imessage_send(self, client) -> ToolDescriptor:
        def execute(params: dict) -> str:
            return client.send(params["contact"], params["message"],
                               params.get("service", "auto"))

        return ToolDescriptor(
            name="imessage_send",
            description=(
                "Send an iMessage or SMS to a contact. "
                "Use for: 'text [contact] saying ...', 'send message to [contact]', "
                "'reply to [contact] on iMessage'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "contact": _str_prop("Contact name, phone number, or email"),
                    "message": _str_prop("Message text to send"),
                    "service": _str_prop("iMessage | SMS | auto (default: auto)"),
                },
                "required": ["contact", "message"],
            },
            execute=execute,
        )

    def _imessage_contacts(self, client) -> ToolDescriptor:
        def execute(params: dict) -> str:
            contacts = client.get_contacts()
            if not contacts:
                return "No recent iMessage contacts found."
            return "\n".join(c["display_name"] for c in contacts[:20])

        return ToolDescriptor(
            name="imessage_contacts",
            description="List recent iMessage contacts.",
            input_schema={"type": "object", "properties": {}},
            execute=execute,
        )

    # ------------------------------------------------------------------
    # WhatsApp tools
    # ------------------------------------------------------------------

    def _whatsapp_read(self, client) -> ToolDescriptor:
        def execute(params: dict) -> str:
            result = client.read(params["contact"], int(params.get("limit", 5)))
            if isinstance(result, str):
                return result
            lines = []
            for m in result:
                lines.append(f"{m['timestamp']} {m['sender']}: {m['text']}")
            return "\n".join(lines) if lines else "No messages found."

        return ToolDescriptor(
            name="whatsapp_read",
            description=(
                "Read recent WhatsApp messages from a contact. "
                "Use for: 'what did [contact] say on WhatsApp', 'read WhatsApp from [contact]'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "contact": _str_prop("Contact name or phone number"),
                    "limit": _int_prop("Number of messages (default 5, max 20)"),
                },
                "required": ["contact"],
            },
            execute=execute,
        )

    def _whatsapp_send(self, client) -> ToolDescriptor:
        def execute(params: dict) -> str:
            return client.send(params["contact"], params["message"])

        return ToolDescriptor(
            name="whatsapp_send",
            description=(
                "Send a WhatsApp message to a contact. "
                "Use for: 'reply to [contact] on WhatsApp saying ...', 'WhatsApp [contact] that ...'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "contact": _str_prop("Contact name or phone number"),
                    "message": _str_prop("Message text to send"),
                },
                "required": ["contact", "message"],
            },
            execute=execute,
        )

    def _whatsapp_contacts(self, client) -> ToolDescriptor:
        def execute(params: dict) -> str:
            result = client.get_contacts()
            if isinstance(result, str):
                return result
            return "\n".join(c["display_name"] for c in result[:20])

        return ToolDescriptor(
            name="whatsapp_contacts",
            description="List recent WhatsApp contacts.",
            input_schema={"type": "object", "properties": {}},
            execute=execute,
        )
