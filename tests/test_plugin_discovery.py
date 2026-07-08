"""Tests for the plugin auto-discovery system (plugins.discover)."""
from __future__ import annotations

import plugins
from plugin import PluginBase, PluginContext


def test_discover_finds_all_builtin_plugins():
    names = {cls.__name__ for cls in plugins.discover()}
    assert {
        "CorePlugin", "MemoryPlugin", "MessagingPlugin", "MediaPlugin",
        "ScreenPlugin", "HealthPlugin", "ProductivityPlugin",
    } <= names


def test_discover_orders_agent_dependent_plugins_last():
    classes = plugins.discover()
    flags = [bool(c.requires_agent) for c in classes]
    # once we hit the first requires_agent plugin, all the rest require it too
    assert flags == sorted(flags)
    assert any(flags), "ProductivityPlugin should require the agent"


def test_discover_returns_only_pluginbase_subclasses():
    for cls in plugins.discover():
        assert issubclass(cls, PluginBase)
        assert cls is not PluginBase


def test_core_plugin_from_context_receives_services():
    from plugins.core import CorePlugin
    sentinel = object()
    ctx = PluginContext(browser=sentinel, keyterms_prompt="hint")
    p = CorePlugin.from_context(ctx)
    assert p._browser is sentinel
    assert p._keyterms_prompt == "hint"


def test_productivity_plugin_from_context_receives_agent():
    from plugins.productivity import ProductivityPlugin
    agent = object()
    ctx = PluginContext(agent=agent)
    p = ProductivityPlugin.from_context(ctx)
    assert p._agent is agent
    assert ProductivityPlugin.requires_agent is True


def test_default_from_context_builds_noarg_plugins():
    from plugins.health import HealthPlugin
    p = HealthPlugin.from_context(PluginContext())
    assert isinstance(p, HealthPlugin)


def test_discover_never_raises_on_scan_failure(monkeypatch):
    monkeypatch.setattr(plugins, "__path__", ["/nonexistent/nowhere"])
    assert plugins.discover() == []
