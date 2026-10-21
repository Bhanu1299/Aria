"""Dashboard Live run / LLM costs / Settings tabs, the run log, LLM call labels. Synthetic data only."""
from __future__ import annotations

import http.client
import json
import os
import stat
import threading
from datetime import datetime
from pathlib import Path

import httpx
import pytest

import aria.core.config as config
from aria.dashboard import costs, settings, tabs
from aria.dashboard.__main__ import serve
from aria.llm import context
from aria.observability import run_log

CONFIG_YAML = """# test config
providers:
  openai:
    type: openai_compat
    api_key_env: OPENAI_API_KEY
  groq:
    type: groq
    api_key_env: GROQ_API_KEY
models:
  - id: gpt-mini
    provider: openai
    input_per_mtok: 0.1
    output_per_mtok: 0.5
  - id: gpt-big
    provider: openai
    input_per_mtok: 2.0
    output_per_mtok: 10.0
  - id: groq/oss
    provider: groq
    input_per_mtok: 0.15
    output_per_mtok: 0.6
# Tier comments must survive edits.
tiers:
  small: [groq/oss, gpt-mini]   # cheapest first
  frontier: [gpt-big, groq/oss]
reliability:
  attempt_timeout_s: 30
"""
KEY = "fake-" + "k" * 20 + "A1B2"   # built at runtime: not key-shaped in source
KEEP = "placeholder-" + "g" * 12
OLD = "placeholder-" + "o" * 12


@pytest.fixture
def gw(tmp_path, monkeypatch):
    d = tmp_path / "GateLLM"
    d.mkdir()
    (d / "config.yaml").write_text(CONFIG_YAML)
    (d / "notes.txt").write_text("untouched")
    monkeypatch.setattr(config, "GATEWAY_DIR", str(d))
    return d


def _tree(root: Path) -> dict:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


# ── settings: the only writer ───────────────────────────────────────────────

def test_mask_never_shows_more_than_ends():
    assert settings.mask(KEY) == "fak...A1B2"
    assert settings.mask("short") == "set (short key)" and settings.mask("") == ""


def test_set_key_creates_env_600_and_returns_masked_only(gw, caplog):
    caplog.set_level("DEBUG")
    out = settings.set_key("openai", KEY)
    env = gw / ".env"
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    assert env.read_text() == f"OPENAI_API_KEY={KEY}\n"
    assert out == {"provider": "openai", "masked": "fak...A1B2", "has_key": True}
    assert KEY not in caplog.text
    p = {x["name"]: x for x in settings.providers()}
    assert p["openai"]["masked"] == "fak...A1B2" and p["groq"]["has_key"] is False
    assert KEY not in json.dumps(settings.providers())


def test_replace_and_remove_keep_other_lines(gw):
    (gw / ".env").write_text("# my keys\nGROQ_API_KEY=" + KEEP + "\nOPENAI_API_KEY=" + OLD + "\nGATEWAY_CONFIG=config.yaml\n")
    settings.set_key("openai", KEY)
    text = (gw / ".env").read_text()
    assert f"OPENAI_API_KEY={KEY}" in text and OLD not in text
    assert "# my keys" in text and f"GROQ_API_KEY={KEEP}" in text and "GATEWAY_CONFIG" in text
    settings.set_key("openai", None)
    text = (gw / ".env").read_text()
    assert "OPENAI_API_KEY" not in text and f"GROQ_API_KEY={KEEP}" in text
    assert stat.S_IMODE((gw / ".env").stat().st_mode) == 0o600


@pytest.mark.parametrize("bad", ["has space key 123", "quote'key12345", "short", "x" * 600, "line\nbreak12345"])
def test_bad_keys_rejected(gw, bad):
    with pytest.raises(settings.SettingsError):
        settings.set_key("openai", bad)
    assert not (gw / ".env").exists()


def test_unknown_provider_rejected(gw):
    with pytest.raises(settings.SettingsError, match="unknown provider"):
        settings.set_key("google", KEY)


def test_set_tier_edits_one_line_and_keeps_comments(gw):
    assert settings.set_tier("small", ["gpt-mini", "groq/oss"]) == ["gpt-mini", "groq/oss"]
    text = (gw / "config.yaml").read_text()
    assert "  small: [gpt-mini, groq/oss]   # cheapest first" in text
    assert "# Tier comments must survive edits." in text and "frontier: [gpt-big, groq/oss]" in text
    assert settings.tiers()["tiers"]["small"] == ["gpt-mini", "groq/oss"]


@pytest.mark.parametrize("tier,models,msg", [("small", [], "at least one"), ("small", ["nope"], "Unknown model"),
                                             ("small", ["gpt-mini", "gpt-mini"], "twice"), ("huge", ["gpt-mini"], "unknown tier")])
def test_set_tier_validation(gw, tier, models, msg):
    before = (gw / "config.yaml").read_text()
    with pytest.raises(settings.SettingsError, match=msg):
        settings.set_tier(tier, models)
    assert (gw / "config.yaml").read_text() == before


def test_writes_stay_inside_env_and_config(gw):
    before = _tree(gw)
    settings.set_key("groq", KEEP)
    settings.set_tier("frontier", ["groq/oss"])
    after = _tree(gw)
    changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    assert changed == {".env", "config.yaml"}


def test_reload_when_gateway_down_is_a_message(gw):
    assert "isn't running" in settings.reload_gateway()["message"]


def test_provider_test_uses_cheapest_model_and_scrubs_keys(gw, monkeypatch):
    settings.set_key("openai", KEY)
    monkeypatch.setattr("aria.llm.autostart.ensure_running", lambda url: True)
    sent = {}

    def fake_post(url, **kw):
        sent.update(url=url, **kw)
        return httpx.Response(401, json={"error": {"message": f"bad key {KEY}"}})

    monkeypatch.setattr(settings.httpx, "post", fake_post)
    out = settings.test_provider("openai")
    assert out["model"] == "gpt-mini" and out["ok"] is False
    assert KEY not in out["message"] and "fak...A1B2" in out["message"]
    assert sent["headers"]["X-Gateway-Fallback"] == "off" and sent["json"]["max_tokens"] == 16

    monkeypatch.setattr(settings.httpx, "post", lambda url, **kw: httpx.Response(
        200, json={"choices": []}, headers={"x-gateway-cost-usd": "0.000010"}))
    assert settings.test_provider("openai")["ok"] is True


def test_settings_tab_never_contains_keys(gw):
    settings.set_key("openai", KEY)
    html = tabs.settings_tab(settings.providers(), settings.tiers())
    assert KEY not in html and "fak...A1B2" in html and "groq/oss" in html


# ── server: localhost + token + origin ──────────────────────────────────────

@pytest.fixture
def server():
    calls = []
    srv = serve(0, build=lambda: "<p>page</p>", token="tok123",
                runs=lambda: {"runs": [{"id": "r1"}]},
                act=lambda path, body: (calls.append((path, body)) or (200, {"ok": True})))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, calls
    srv.shutdown()
    srv.server_close()


def _req(srv, method, path, body=None, headers=None):
    host, port = srv.server_address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    data = json.dumps(body).encode() if body is not None else None
    h = {"Content-Type": "application/json", **(headers or {})}
    conn.request(method, path, body=data, headers=h)
    r = conn.getresponse()
    out = (r.status, r.read().decode())
    conn.close()
    return out


def test_post_needs_token(server):
    srv, calls = server
    assert _req(srv, "POST", "/api/settings/key", {"provider": "openai"})[0] == 403
    assert _req(srv, "POST", "/api/settings/key", {"provider": "openai"}, {"X-Dashboard-Token": "nope"})[0] == 403
    assert calls == []
    assert _req(srv, "POST", "/api/settings/key", {"provider": "openai"}, {"X-Dashboard-Token": "tok123"})[0] == 200
    assert calls == [("/api/settings/key", {"provider": "openai"})]


def test_foreign_origin_and_host_are_refused(server):
    srv, calls = server
    tok = {"X-Dashboard-Token": "tok123"}
    assert _req(srv, "POST", "/api/settings/test", {"provider": "x"}, {**tok, "Origin": "http://evil.test"})[0] == 403
    assert _req(srv, "GET", "/api/runs", headers={"Host": "evil.test:80"})[0] == 403
    assert _req(srv, "POST", "/api/settings/test", {"provider": "x"}, {**tok, "Host": "attacker.test"})[0] == 403
    assert calls == []


def test_runs_endpoint(server):
    srv, _ = server
    assert json.loads(_req(srv, "GET", "/api/runs")[1]) == {"runs": [{"id": "r1"}]}


# ── run log ─────────────────────────────────────────────────────────────────

def test_run_log_rebuilds_runs_newest_first():
    a = run_log.start("apply_job")
    a.job("Acme", "SWE", 3)
    with a.step("read sheet") as s:
        s.detail = "12 rows"
    with pytest.raises(ValueError):
        with a.step("fill form"):
            raise ValueError("form changed")
    a.finish("error", "fill failed")
    b = run_log.start("submit")
    runs = run_log.read_runs()
    assert [r["id"] for r in runs] == [b.id, a.id] and runs[0]["status"] == "running"
    steps = {s["step"]: s for s in runs[1]["steps"]}
    assert steps["read sheet"]["detail"] == "12 rows"
    assert steps["fill form"]["status"] == "error" and "form changed" in steps["fill form"]["detail"]
    assert steps["draft"]["status"] == "not in this run"
    assert [s["step"] for s in runs[1]["steps"]] == run_log.STEPS


def test_run_log_keeps_last_n():
    for _ in range(12):
        run_log.start("apply_job").finish("ok")
    assert len(run_log.read_runs(10)) == 10


def test_live_tab_embeds_json_safely():
    html = tabs.live_tab([{"id": "r", "summary": "</script><b>x</b>"}])
    assert "</script><b>" not in html


# ── LLM call labels ─────────────────────────────────────────────────────────

def test_context_labels_and_headers():
    assert context.headers() == {}
    with context.job("Acme | SWE | row 3", "run-1"):
        with context.scope(task="form_short_answer"):
            with context.scope(escalated_from="small: output failed validation"):
                h = context.headers()
        assert context.current() == {"job": "Acme | SWE | row 3", "run": "run-1"}
    assert h == {"X-Gateway-Job": "Acme | SWE | row 3", "X-Gateway-Run": "run-1",
                 "X-Gateway-Task": "form_short_answer", "X-Gateway-Escalated-From": "small: output failed validation"}
    with context.scope(task="naïve—task"):
        context.headers()["X-Gateway-Task"].encode("latin-1")  # header-safe
    assert context.current() == {}


def test_gateway_requests_carry_labels():
    from aria.llm.providers import gateway
    seen = []

    def handler(request):
        seen.append(dict(request.headers))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}})

    client = httpx.Client(base_url="http://gw.test", transport=httpx.MockTransport(handler))
    chat = gateway.GatewayChatClient("http://gw.test", client=client)
    with context.job("J", "R"):
        chat.chat.completions.create(model="small", messages=[], task="screen_qa")
    assert seen[0]["x-gateway-task"] == "screen_qa" and seen[0]["x-gateway-job"] == "J"


def test_escalation_is_labelled(monkeypatch):
    from aria.llm import llm_client
    labels = []

    class Chain:
        def complete(self, *a, **k):
            from aria.llm.base import LLMResponse
            labels.append(context.current())
            return LLMResponse(text="bad", tool_calls=[], stop_reason="end_turn", provider_used="t", model_used="t")

    monkeypatch.setattr(llm_client, "_chains", {t: Chain() for t in ("small", "frontier", "smart")})
    llm_client.complete_checked("form_short_answer", [{"role": "user", "content": "q"}], validate=lambda t: None)
    assert labels[0] == {"task": "form_short_answer"}
    assert labels[1] == {"task": "form_short_answer", "escalated_from": "small: output failed validation"}


# ── costs ───────────────────────────────────────────────────────────────────

NOW = datetime(2026, 10, 21, 12, 0)


def _call(ts_days_ago, cost, task="form_short_answer", job="Acme | SWE | row 3", **kw):
    return {"ts": NOW.timestamp() - ts_days_ago * 86400, "status": "ok", "task": task, "job": job, "tier": "small",
            "model": "groq/oss", "tokens_in": 100, "tokens_out": 10, "cost_usd": cost, "attempts": [], **kw}


def test_costs_summary():
    calls = [
        _call(0.1, 0.01),
        _call(1, 0.02, task="form_free_text", model="gpt-big", tier="frontier",
              escalated_from="small: output failed validation"),
        _call(3, 0.04, job="Hooli | PE | row 9", fallbacks=1,
              attempts=[{"model": "groq/oss", "outcome": "rate_limited"}, {"model": "gpt-mini", "outcome": "ok"}]),
        _call(0.2, 0.005, job="", task="agent"),
        {"ts": NOW.timestamp() - 3600, "status": "error", "task": "agent", "attempts": [{"model": "a", "outcome": "auth"}]},
    ]
    first = min(c["ts"] for c in calls)
    flight = [
        {"kind": "job", "ts": first - 86400 * 5, "date": "2026-10-13", "company": "Old Co", "role": "Dev", "row": 2,
         "stage": "filled", "usage": {"cost_usd": 0.1, "calls": 4, "tokens_in": 9, "tokens_out": 1}},
        {"ts": first + 10, "usage": {"cost_usd": 9.0}},       # after the call log began: not double counted
    ]
    s = costs.summarize(calls, flight, NOW, applied_jobs=2)
    assert s["spend"]["today"] == pytest.approx(0.015)
    assert s["spend"]["week"] == pytest.approx(0.035)        # week starts Mon Oct 19
    assert s["spend"]["all"] == pytest.approx(0.175)
    assert s["spend"]["errors"] == 1
    jobs = {r["job"]: r for r in s["jobs"]}
    assert jobs["Acme | SWE | row 3"]["total"] == pytest.approx(0.03) and len(jobs["Acme | SWE | row 3"]["calls"]) == 2
    assert jobs["Old Co | Dev | row 2"]["steps"][0]["cost_usd"] == 0.1
    t = {r["task"]: r for r in s["tasks"]}
    assert t["form_short_answer"]["count"] == 2 and t["form_free_text"]["models"]["gpt-big"] == 1
    kinds = [x["kind"] for x in s["escalations"]]
    assert "Aria: small → frontier" in kinds and "GateLLM fallback" in kinds
    fb = next(x for x in s["escalations"] if x["kind"] == "GateLLM fallback" and x["model"] == "groq/oss")
    assert "rate_limited" in fb["why"]
    assert s["priciest"][0]["job"] == "Old Co | Dev | row 2" and s["cheapest"][0]["job"] == "Acme | SWE | row 3"
    assert s["per_applied"] == pytest.approx((0.03 + 0.04 + 0.1) / 2)
    html = tabs.costs_tab(s, "file")
    assert "Per task" in html and "form_free_text" in html and "rate_limited" in html


def test_fetch_calls_falls_back_to_file(tmp_path):
    log = tmp_path / "calls.jsonl"
    log.write_text(json.dumps({"ts": 1, "status": "ok"}) + "\nnot json\n")
    out = costs.fetch_calls("http://127.0.0.1:9", log)   # endpoint unreachable in tests
    assert out == {"calls": [{"ts": 1, "status": "ok"}], "via": "file"}
    assert costs.fetch_calls("http://127.0.0.1:9", tmp_path / "none.jsonl")["via"] == "none"


def test_costs_tab_no_data():
    assert "No data" in tabs.costs_tab(None, "none", "boom")


# ── Live run: open application tabs ─────────────────────────────────────────

def test_runs_json_includes_live_apps_without_closed(monkeypatch, tmp_path):
    from aria.dashboard.__main__ import runs_json
    from aria.features.applying import tabs as apply_tabs
    from aria.observability import run_log
    monkeypatch.setattr(apply_tabs, "REGISTRY_PATH", tmp_path / "apply_tabs.json")
    monkeypatch.setattr(run_log, "LOG_PATH", tmp_path / "run_log.jsonl")
    a = apply_tabs.add({"row": 15, "company": "Scale AI", "role": "SWE New Grad", "url": "https://x.example/a"})
    apply_tabs.update(a["id"], state=apply_tabs.FILLED, needs_me=["Pronouns", "Work authorization"],
                      filled={"secret": {"value": "do not ship"}})
    b = apply_tabs.add({"row": 16, "company": "SingleStore", "role": "SWE", "url": "https://x.example/b"})
    apply_tabs.mark_closed(b["id"])
    out = runs_json()
    assert out["runs"] == []
    assert out["apps"] == [{"company": "Scale AI", "role": "SWE New Grad", "row": 15, "state": "filled",
                            "needs_me": ["Pronouns", "Work authorization"], "note": ""}]


def test_live_tab_renders_apps_table():
    html = tabs.live_tab([], apps=[{"company": "Ookla</script>", "role": "Data Engineer", "row": 14,
                                    "state": "open", "needs_me": [], "note": "stopped mid-fill"}])
    assert 'id="apptabs"' in html and 'id="apps-data"' in html
    assert "Ookla<\\/script>" in html and "Ookla</script>" not in html
    assert "Open application tabs" in tabs.LIVE_JS and "Fields needing you" in tabs.LIVE_JS
