from __future__ import annotations

import httpx

from aria.web import browser_profile


def _fake(monkeypatch, targets):
    calls = []
    monkeypatch.setattr(browser_profile.httpx, "get",
                        lambda url, timeout: httpx.Response(200, json=targets, request=httpx.Request("GET", url)))
    monkeypatch.setattr(browser_profile.httpx, "put", lambda url, timeout: calls.append(url))
    return calls


def test_windowless_chrome_gets_a_blank_tab(monkeypatch):
    calls = _fake(monkeypatch, [])
    assert browser_profile.ensure_window(9333) is True
    assert calls == ["http://127.0.0.1:9333/json/new?about:blank"]


def test_chrome_with_a_tab_is_left_alone(monkeypatch):
    calls = _fake(monkeypatch, [{"type": "page", "url": "https://boards.greenhouse.io/x"}])
    assert browser_profile.ensure_window(9333) is False and calls == []


def test_unreachable_chrome_does_not_raise(monkeypatch):
    def boom(url, timeout):
        raise httpx.ConnectError("down")
    monkeypatch.setattr(browser_profile.httpx, "get", boom)
    assert browser_profile.ensure_window(9333) is False
