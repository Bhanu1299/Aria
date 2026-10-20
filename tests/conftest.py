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
