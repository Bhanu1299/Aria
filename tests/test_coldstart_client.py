"""tests/test_coldstart_client.py — the Coldstart HTTP client: mapping, errors, rate limit, pattern cache."""
from __future__ import annotations

import json

import httpx
import pytest

from aria.integrations import coldstart_client as cs


def _client(handler, **kw):
    seen = []

    def wrap(request):
        seen.append(json.loads(request.content) if request.content else None)
        return handler(request)
    http = httpx.Client(transport=httpx.MockTransport(wrap))
    kw.setdefault("min_interval", 0)
    return cs.ColdstartClient(base_url="http://coldstart.test", client=http, **kw), seen


def _reply(data, code=200):
    return lambda request: httpx.Response(code, json=data)


@pytest.mark.parametrize("status,label", [("verified", "verified"), ("pattern_guess", "pattern"),
                                          ("not_found", "not found")])
def test_status_is_mapped_to_the_sheet_label(status, label):
    email = "" if status == "not_found" else "pat.lee@acme.com"
    c, seen = _client(_reply({"email": email, "status": status, "confidence": 0.9, "pattern": "{first}.{last}"}))
    lk = c.find("Pat Lee", "Acme", "acme.com")
    assert lk.ok and lk.confidence == label and lk.email == email and lk.score == 0.9
    assert seen == [{"name": "Pat Lee", "company": "Acme", "domain": "acme.com"}]


def test_posts_to_find_on_the_configured_url():
    urls = []
    c, _ = _client(lambda r: (urls.append(str(r.url)), httpx.Response(200, json={"status": "not_found"}))[1])
    c.find("Pat Lee", "Acme", "acme.com")
    assert urls == ["http://coldstart.test/find"]


@pytest.mark.parametrize("handler,why", [
    (_reply({"error": "boom"}, 500), "HTTP 500"),
    (lambda r: httpx.Response(200, content=b"not json"), "invalid JSON"),
    (_reply({"status": "maybe"}), "unknown status"),
    (_reply({"status": "verified", "email": ""}), "no address"),
    (_reply(["x"]), "isn't a JSON object"),
])
def test_failures_come_back_as_errors_never_raise(handler, why):
    c, _ = _client(handler)
    lk = c.find("Pat Lee", "Acme", "acme.com")
    assert not lk.ok and why in lk.error


def test_unreachable_service_is_an_error():
    def down(request):
        raise httpx.ConnectError("refused", request=request)
    c, _ = _client(down)
    lk = c.find("Pat Lee", "Acme", "acme.com")
    assert not lk.ok and "unreachable" in lk.error


def test_calls_are_spaced_by_min_interval():
    now, slept = [100.0], []
    c, _ = _client(_reply({"status": "not_found"}), min_interval=2.0,
                   clock=lambda: now[0], sleep=lambda s: (slept.append(s), now.__setitem__(0, now[0] + s)))
    c.find("Pat Lee", "Acme", "acme.com")
    now[0] += 0.5
    c.find("Ana Diaz", "Beta", "beta.com")
    assert slept == [1.5] and c.calls == 2


def test_pattern_is_cached_per_domain_and_reused_without_a_call():
    c, seen = _client(_reply({"email": "pat.lee@acme.com", "status": "verified", "pattern": "{first}.{last}"}))
    c.find("Pat Lee", "Acme", "acme.com")
    lk = c.find("Ana Diaz", "Acme", "acme.com")
    assert len(seen) == 1
    assert (lk.email, lk.confidence, lk.source) == ("ana.diaz@acme.com", "pattern", "Coldstart pattern cache (acme.com)")
    c.find("Ana Diaz", "Beta", "beta.com")          # another domain still calls
    assert len(seen) == 2


def test_cache_skipped_when_the_name_cannot_fill_the_pattern():
    c, seen = _client(_reply({"status": "not_found"}))
    c.remember("acme.com", "first.last")
    c.find("Cher", "Acme", "acme.com")
    assert len(seen) == 1


@pytest.mark.parametrize("pattern,email", [
    ("{first}.{last}", "pat.lee@acme.com"), ("first.last", "pat.lee@acme.com"), ("flast", "plee@acme.com"),
    ("{f}{last}@acme.com", "plee@acme.com"), ("first", "pat@acme.com"), ("{first_initial}.{last}", "p.lee@acme.com"),
    ("garbage!", ""), ("", ""),
])
def test_render(pattern, email):
    assert cs.render(pattern, "Pat O. Lee", "acme.com") == email


def test_ensure_running_starts_a_local_coldstart_that_is_down(monkeypatch, tmp_path):
    (tmp_path / "venv" / "bin").mkdir(parents=True)
    (tmp_path / "venv" / "bin" / "python").write_text("")
    monkeypatch.setattr(cs.config, "COLDSTART_DIR", str(tmp_path))
    monkeypatch.setattr(cs.config, "COLDSTART_AUTOSTART", True)
    started = []
    monkeypatch.setattr(cs, "_start", lambda python, port: started.append((str(python), port)))
    up = [False]
    c = cs.ColdstartClient(base_url="http://127.0.0.1:8765", sleep=lambda s: up.__setitem__(0, True))
    c.health = lambda: "" if up[0] else "refused"
    assert c.ensure_running() == ""
    assert started == [(str(tmp_path / "venv" / "bin" / "python"), 8765)]


def test_ensure_running_never_starts_a_remote_coldstart(monkeypatch):
    monkeypatch.setattr(cs, "_start", lambda *a: pytest.fail("started a remote Coldstart"))
    c = cs.ColdstartClient(base_url="http://coldstart.example.com")
    c.health = lambda: "refused"
    assert "isn't running" in c.ensure_running()


def test_ensure_running_leaves_a_running_coldstart_alone(monkeypatch):
    monkeypatch.setattr(cs, "_start", lambda *a: pytest.fail("started a second Coldstart"))
    c, _ = _client(_reply({"ok": True}))
    assert c.ensure_running() == ""
