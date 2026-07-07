"""tests/test_messaging_stubs.py — Stub tool descriptors return correct setup strings."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_telegram_read_stub():
    from plugins.messaging.stubs.telegram import read_tool
    tool = read_tool()
    assert tool.name == "telegram_read"
    result = tool.execute({"contact": "Alice"})
    assert "Telegram" in result
    assert "TELEGRAM_BOT_TOKEN" in result


def test_telegram_send_stub():
    from plugins.messaging.stubs.telegram import send_tool
    tool = send_tool()
    assert tool.name == "telegram_send"
    result = tool.execute({"contact": "Alice", "message": "hi"})
    assert "Telegram" in result


def test_discord_read_stub():
    from plugins.messaging.stubs.discord import read_tool
    tool = read_tool()
    assert tool.name == "discord_read"
    result = tool.execute({"contact": "general"})
    assert "Discord" in result
    assert "DISCORD_BOT_TOKEN" in result


def test_discord_send_stub():
    from plugins.messaging.stubs.discord import send_tool
    tool = send_tool()
    result = tool.execute({"contact": "general", "message": "hello"})
    assert "Discord" in result


def test_slack_read_stub():
    from plugins.messaging.stubs.slack import read_tool
    tool = read_tool()
    assert tool.name == "slack_read"
    result = tool.execute({"contact": "general"})
    assert "Slack" in result
    assert "SLACK_BOT_TOKEN" in result


def test_slack_send_stub():
    from plugins.messaging.stubs.slack import send_tool
    tool = send_tool()
    result = tool.execute({"contact": "general", "message": "hello"})
    assert "Slack" in result


def test_stubs_all_have_required_fields():
    from plugins.messaging.stubs import telegram, discord, slack
    for mod in (telegram, discord, slack):
        for factory in (mod.read_tool, mod.send_tool):
            tool = factory()
            assert tool.name
            assert tool.description
            assert isinstance(tool.input_schema, dict)
