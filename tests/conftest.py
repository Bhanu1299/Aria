from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

# Add project root to path so tests can import db, etc.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(autouse=True)
def _no_live_http(monkeypatch):
    """No test reaches the real network (GateLLM at 127.0.0.1:8000 included).

    Blocks httpx's real transports only — tests that inject httpx.MockTransport
    keep working. An unmocked LLM call fails fast as "gateway unreachable".
    """
    def _refuse(self, request):
        raise httpx.ConnectError(f"live HTTP blocked in tests: {request.url}", request=request)

    async def _refuse_async(self, request):
        raise httpx.ConnectError(f"live HTTP blocked in tests: {request.url}", request=request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _refuse)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _refuse_async)
    # ...and never launches the real GateLLM process on an unreachable gateway.
    monkeypatch.setattr("aria.core.config.GATEWAY_AUTOSTART", False)
    # Google's client library talks over httplib2, not httpx: with real tokens on disk an
    # unmocked Sheets/Gmail call would reach the user's real account.
    import httplib2

    def _refuse_google(self, uri, *args, **kwargs):
        raise RuntimeError(f"live Google API call blocked in tests: {uri}")

    monkeypatch.setattr(httplib2.Http, "request", _refuse_google)


@pytest.fixture(autouse=True)
def _run_log_in_tmp(tmp_path, monkeypatch):
    """Pipeline run events never land in the real ~/.aria/run_log.jsonl."""
    monkeypatch.setattr("aria.observability.run_log.LOG_PATH", tmp_path / "run_log.jsonl")
