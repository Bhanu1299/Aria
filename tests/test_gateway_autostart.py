"""GateLLM on-demand start / idle stop (aria/llm/autostart.py)."""
from __future__ import annotations

import httpx
import pytest

import aria.core.config as config
from aria.llm import autostart
from aria.llm.base import GatewayUnavailable
from aria.llm.providers import gateway

URL = "http://127.0.0.1:8000"
OK_BODY = {"choices": [{"message": {"role": "assistant", "content": "hi"}}], "usage": {}}


class _FakeProc:
    def __init__(self, *args, **kwargs):
        self.pid = 4242
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.terminated = True


@pytest.fixture
def gw(monkeypatch, tmp_path):
    """Autostart enabled with a fake GateLLM install, no real processes."""
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "uvicorn").write_text("")
    monkeypatch.setattr(config, "GATEWAY_AUTOSTART", True)
    monkeypatch.setattr(config, "GATEWAY_DIR", str(tmp_path))
    monkeypatch.setattr(autostart, "_PIDFILE", tmp_path / "gatellm.pid")
    monkeypatch.setattr(autostart, "_LOGFILE", tmp_path / "gatellm.log")
    monkeypatch.setattr(autostart, "_start_watchdog", lambda: None)
    monkeypatch.setattr(autostart, "_proc", None)
    monkeypatch.setattr(autostart, "_pid", None)
    monkeypatch.setattr(autostart, "_active", 0)
    state = {"up": False, "spawned": []}

    def _popen(*args, **kwargs):
        state["up"] = True
        proc = _FakeProc()
        state["spawned"].append(proc)
        return proc

    monkeypatch.setattr(autostart.subprocess, "Popen", _popen)
    monkeypatch.setattr(autostart, "_healthy", lambda url: state["up"])
    return state


def test_unavailable_when_disabled_or_remote(gw, monkeypatch):
    assert autostart.available(URL)
    assert not autostart.available("https://gateway.example.com")
    monkeypatch.setattr(config, "GATEWAY_AUTOSTART", False)
    assert not autostart.available(URL)
    assert autostart.ensure_running(URL) is False
    assert gw["spawned"] == []


def test_unavailable_without_gatellm_install(gw, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GATEWAY_DIR", str(tmp_path / "missing"))
    assert autostart.ensure_running(URL) is False


def test_starts_once_and_writes_pidfile(gw):
    assert autostart.ensure_running(URL) is True
    assert autostart.ensure_running(URL) is True  # already healthy → no second spawn
    assert len(gw["spawned"]) == 1
    assert autostart._PIDFILE.read_text() == "4242"


def test_crash_during_startup_reports_false(gw, monkeypatch):
    def _dies(*args, **kwargs):
        proc = _FakeProc()
        proc.returncode = 1
        return proc

    monkeypatch.setattr(autostart.subprocess, "Popen", _dies)
    assert autostart.ensure_running(URL) is False
    assert not autostart._PIDFILE.exists()


def test_stop_only_touches_owned_gateway(gw):
    autostart.stop()  # nothing owned → no-op
    autostart.ensure_running(URL)
    proc = gw["spawned"][0]
    autostart.stop()
    assert proc.terminated
    assert autostart._pid is None
    assert not autostart._PIDFILE.exists()


def test_post_starts_gateway_and_retries(gw):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if not gw["up"]:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json=OK_BODY)

    client = httpx.Client(base_url=URL, transport=httpx.MockTransport(handler))
    data, _ = gateway._post(client, {"model": "small", "messages": []})
    assert data["choices"][0]["message"]["content"] == "hi"
    assert len(calls) == 2 and len(gw["spawned"]) == 1
    assert autostart._active == 0


def test_post_raises_unavailable_when_autostart_off(gw, monkeypatch):
    monkeypatch.setattr(config, "GATEWAY_AUTOSTART", False)

    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    client = httpx.Client(base_url=URL, transport=httpx.MockTransport(handler))
    with pytest.raises(GatewayUnavailable):
        gateway._post(client, {"model": "small", "messages": []})


def test_adopts_orphan_from_pidfile(gw, monkeypatch):
    autostart._PIDFILE.write_text("777")
    monkeypatch.setattr(autostart, "_alive", lambda pid: pid == 777)
    autostart._adopt_orphan()
    assert autostart._pid == 777

    killed = []
    monkeypatch.setattr(autostart.os, "kill", lambda pid, sig: killed.append(pid))
    autostart.stop()
    assert killed == [777]
    assert not autostart._PIDFILE.exists()


def test_stale_pidfile_is_removed(gw, monkeypatch):
    autostart._PIDFILE.write_text("777")
    monkeypatch.setattr(autostart, "_alive", lambda pid: False)
    autostart._adopt_orphan()
    assert autostart._pid is None
    assert not autostart._PIDFILE.exists()
