"""Pipeline dashboard (aria/dashboard): matching, stages, sections, sources, server, read-only.

Every fixture here is synthetic (example companies, .test addresses)."""
from __future__ import annotations

import http.client
import re
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from aria.dashboard import model as m
from aria.dashboard import render, sources
from aria.dashboard.__main__ import serve
from aria.dashboard.sources import Source

NOW = datetime(2026, 10, 21, 12, 0)
ME = "seeker@example.test"
ARIA = Path(__file__).parent.parent / "aria"


def ts(days_ago: float) -> float:
    return (NOW - timedelta(days=days_ago)).timestamp()


def sheet_row(n, company, role, **kw):
    return {"_row": n, "Company": company, "Role": role, "Status": kw.pop("Status", ""), "Apply link": "",
            "Post link": "", "Who to contact": "", "Profile alignment /5": "3", **kw}


def thread(tid, to, subject, sent_days, reply_days=None, bounce_days=None):
    msgs = [{"id": f"{tid}a", "ts": ts(sent_days), "from": ME, "to": [to], "subject": subject,
             "is_reply": False, "from_me": True, "bounce": False}]
    if reply_days is not None:
        msgs.append({"id": f"{tid}b", "ts": ts(reply_days), "from": to, "to": [ME], "subject": "Re: " + subject,
                     "is_reply": True, "from_me": False, "bounce": False})
    if bounce_days is not None:
        msgs.append({"id": f"{tid}c", "ts": ts(bounce_days), "from": "mailer-daemon@googlemail.test", "to": [ME],
                     "subject": "Delivery Status Notification (Failure)", "is_reply": False, "from_me": False,
                     "bounce": True})
    return {"id": tid, "messages": msgs}


@pytest.fixture
def srcs(tmp_path):
    review = tmp_path / "acme_review.md"
    review.write_text("# Review\n\n## Needs you (2)\n\n| Field | Value | Required | Note |\n|---|---|---|---|\n"
                      "| Current location ✱ |  | yes | still empty |\n| Work authorization |  | yes | still empty |\n\n"
                      "## From your answer bank (1)\n\n| Field | Value |\n|---|---|\n| Name | A |\n")
    sheet = {
        "current": [
            sheet_row(2, "Acme Robotics", "Software Engineer I", Status="In review",
                      **{"Apply link": "https://boards.example.test/acme/jobs/1?src=muse",
                         "Post link": "https://jobs.example.test/acme/1"}),
            sheet_row(3, "Globex", "Data Engineer", Status="Not gonna apply"),
            sheet_row(4, "Initech", "Backend Engineer", Status="Needs human"),
        ],
        "applied": [
            sheet_row(2, "Umbrella Labs", "ML Engineer", Status="Applied", Source="HN Who is hiring Oct 20",
                      **{"Who to contact": "Pat Lee (Recruiter)", "Channel": "Email"}),
            sheet_row(3, "Hooli", "Platform Engineer", Status="Rejected (Oct 13)",
                      Outcome="Looking for more experience"),
            sheet_row(4, "Vandelay", "Full-Stack Engineer", Status="Applied"),
            sheet_row(5, "Stark Industries", "AI Engineer", Status="Applied"),
        ],
    }
    runs = [
        {"id": 1, "sheet_row": 2, "company": "Acme Robotics", "role": "Software Engineer I",
         "url": "https://boards.example.test/acme/jobs/1", "ats": "greenhouse", "stage": "filled",
         "needs_me": 2, "cost_usd": 0.02, "review_path": str(review), "note": "", "created_at": "2026-10-20 10:00:00"},
        {"id": 2, "sheet_row": 9, "company": "Nobody Inc", "role": "Ghost Role", "url": "", "ats": "", "stage": "filled",
         "needs_me": 0, "cost_usd": 0.01, "review_path": "", "note": "", "created_at": "2026-10-20 11:00:00"},
    ]
    gmail = {"me": ME, "drafts": [
        {"id": "d1", "thread_id": "td1", "to": ["pat.lee@umbrellalabs.test"], "subject": "ML Engineer — Seeker",
         "ts": ts(3)},
        {"id": "d2", "thread_id": "td2", "to": ["x@stranger.test"], "subject": "Hello", "ts": ts(0.5)},
    ], "threads": [
        thread("t1", "hr.person@vandelay.test", "Full-Stack Engineer — Seeker", sent_days=10),            # follow-up due
        thread("t2", "cto@stark.test", "AI Engineer — Seeker", sent_days=4, reply_days=1),                # replied
        thread("t3", "jo@hooli.test", "Platform Engineer — Seeker", sent_days=9, bounce_days=9),          # bounced
        thread("t4", "z@vandelay.test", "Quantum Physicist — Seeker", sent_days=2),                       # role mismatch
        thread("t5", "friend@gmail.com", "hi", sent_days=1),                                               # not outreach
    ]}
    coldstart = {"contacts": [{"domain": "umbrellalabs.test", "name": "Pat Lee", "title": "Recruiter",
                               "email": "pat.lee@umbrellalabs.test", "status": "catch_all", "confidence": 0.6,
                               "source": "pattern", "checked_at": "2026-10-19"}], "domains": {}}
    return {"sheet": Source(True, sheet), "aria_db": Source(True, {"runs": runs, "applications": []}),
            "gmail": Source(True, gmail), "coldstart": Source(True, coldstart),
            "reviews": Source(True, {str(review): sources.parse_review_needs_me(review.read_text())})}


def by_company(mdl):
    return {j.company: j for j in mdl["jobs"]}


# ── matching + stages ───────────────────────────────────────────────────────

def test_stage_is_furthest_evidence(srcs):
    jobs = by_company(m.build(srcs, NOW))
    assert jobs["Acme Robotics"].stage_label == "Form filled (in review)"
    assert jobs["Globex"].stage_label == "Skipped"
    assert jobs["Initech"].stage_label == "Applying"
    assert jobs["Umbrella Labs"].stage_label == "Draft ready"      # Coldstart email + Gmail draft
    assert jobs["Vandelay"].stage_label == "Sent"
    assert jobs["Stark Industries"].stage_label == "Replied"
    assert jobs["Hooli"].stage_label == "Rejected"


def test_each_stage_records_its_source(srcs):
    umbrella = by_company(m.build(srcs, NOW))["Umbrella Labs"]
    seen = {(m.STAGES[e.stage], e.source.split(" (")[0]) for e in umbrella.evidence}
    assert ("Found by Muse", "Muse sheet") in seen
    assert ("Contact found", "Coldstart contact cache") in seen
    assert ("Draft ready", "Gmail draft") in seen
    assert umbrella.email == "pat.lee@umbrellalabs.test" and umbrella.email_confidence == 0.6


def test_runs_match_by_apply_link_ignoring_query(srcs):
    acme = by_company(m.build(srcs, NOW))["Acme Robotics"]
    assert len(acme.runs) == 1 and acme.cost == pytest.approx(0.02)
    assert acme.link == "https://jobs.example.test/acme/1"


def test_company_and_role_match_when_no_link():
    s = {"sheet": Source(True, {"current": [sheet_row(2, "Acme", "SWE")], "applied": []}),
         "aria_db": Source(True, {"runs": [{"id": 1, "company": "ACME", "role": "swe", "url": "", "stage": "submitted",
                                            "cost_usd": 0, "created_at": "2026-10-20 09:00:00"}], "applications": []})}
    job = m.build(s, NOW)["jobs"][0]
    assert job.stage_label == "Applied" and job.applied_on


def test_unmatched_records_are_kept(srcs):
    mdl = m.build(srcs, NOW)
    what = {(u["source"], u["what"].split(" — ")[0]) for u in mdl["unmatched"]}
    assert ("aria.db application_runs", "Nobody Inc") in what
    assert ("Gmail sent", "z@vandelay.test") in what          # subject names a role Vandelay doesn't have
    assert ("Gmail draft", "x@stranger.test") in what
    assert not any("friend@gmail.com" in u["what"] for u in mdl["unmatched"])   # freemail is not outreach


def test_domain_company_matching():
    assert m.domain_matches_company("acme.test", "Acme Inc")
    assert m.domain_matches_company("umbrellalabs.test", "Umbrella Labs")
    assert not m.domain_matches_company("ac.test", "Acme")
    assert m.norm_link("https://www.Boards.example.test/x/1/?a=b#c") == "boards.example.test/x/1"


# ── sections ────────────────────────────────────────────────────────────────

def test_funnel_counts_and_drops(srcs):
    rows = {r["stage"]: r for r in m.funnel(m.build(srcs, NOW)["jobs"])}
    assert rows["Found by Muse"]["reached"] == 7 and rows["Found by Muse"]["drop"] is None
    assert rows["Skipped / Applying"]["skipped"] == 1
    assert rows["Replied"]["reached"] == 2 and rows["Replied"]["at"] == 1     # Stark replied, Hooli is past it
    assert rows["Sent"]["drop"] == rows["Draft ready"]["reached"] - rows["Sent"]["reached"]


def test_needs_me(srcs):
    n = m.needs_me(m.build(srcs, NOW))
    assert [(i["job"].company, i["fields"]) for i in n["fields"]] == [
        ("Acme Robotics", ["Current location", "Work authorization"])]
    assert [i["job"].company for i in n["needs_human"]] == ["Initech"]
    assert [d["id"] for d in n["stale_drafts"]] == ["d1"]                 # 3 days old; d2 is fresh
    assert [t["id"] for t in n["follow_ups"]] == ["t1"]                   # 10 days, no reply
    assert [t["id"] for t in n["bounces"]] == ["t3"]


def test_this_week_and_cost(srcs):
    mdl = m.build(srcs, NOW)
    w = m.this_week(mdl)
    assert w["start"] == datetime(2026, 10, 19)
    assert w["found"] == 1 and w["sent"] == 1 and w["replies"] == 1      # t4 sent Oct 19; t2 (Oct 17) replied Oct 20
    assert w["reply_rate_all"] == pytest.approx(1 / 4)
    c = m.cost(mdl)
    assert c["total"] == pytest.approx(0.03) and c["applied"] == 3        # Umbrella, Vandelay, Stark (sheet Status)


def test_page_renders_everything(srcs):
    html = render.page(m.build(srcs, NOW))
    for text in ("Funnel", "Needs me", "All jobs", "This week", "Cost", "Acme Robotics", "follow-up due",
                 "records matched no job", 'id="jobs"'):
        assert text in html
    assert f"authuser={ME.replace('@', '%40')}#all/t2" in html


def test_page_survives_every_source_down():
    down = {k: Source(False, None, "boom") for k in ("sheet", "gmail", "aria_db", "coldstart", "reviews")}
    html = render.page(m.build(down, NOW))
    assert html.count("No data") >= 4 and "Muse sheet: down" in html


def test_failing_section_does_not_break_page(srcs, monkeypatch):
    monkeypatch.setattr(m, "this_week", lambda mdl: 1 / 0)
    html = render.page(m.build(srcs, NOW))
    assert "this section failed (ZeroDivisionError" in html and "Acme Robotics" in html


def test_html_is_escaped(srcs):
    srcs["sheet"].data["current"].append(sheet_row(9, "<script>x</script>", "Role"))
    assert "<script>x</script>" not in render.page(m.build(srcs, NOW))


# ── sources ─────────────────────────────────────────────────────────────────

def test_ttl_cache_keeps_successes_only():
    clock = [0.0]
    cache = sources.TTLCache(ttl=300, clock=lambda: clock[0])
    calls = []
    ok = lambda: calls.append(1) or Source(True, len(calls))
    assert cache.get("k", ok).data == 1 and cache.get("k", ok).data == 1
    clock[0] = 301
    assert cache.get("k", ok).data == 2
    bad = lambda: calls.append(1) or Source(False, None, "down")
    assert not cache.get("b", bad).ok and not cache.get("b", bad).ok and len(calls) == 4


def test_guard_turns_errors_into_down_source():
    s = sources._guard("x", lambda: 1 / 0)
    assert not s.ok and "ZeroDivisionError" in s.error


def _make_db(path: Path, sql: str) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(sql)
    conn.commit()
    conn.close()


def test_aria_db_is_read_only(tmp_path):
    path = tmp_path / "aria.db"
    _make_db(path, "CREATE TABLE application_runs (id INTEGER, stage TEXT, created_at TEXT);"
                   "INSERT INTO application_runs VALUES (1, 'filled', '2026-10-20');")
    before = path.read_bytes()
    data = sources.fetch_aria_db(path)
    assert data == {"runs": [{"id": 1, "stage": "filled", "created_at": "2026-10-20"}], "applications": []}
    assert path.read_bytes() == before
    conn = sources._ro_connect(path)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO application_runs VALUES (2, 'x', 'y')")
    conn.close()


def test_missing_db_is_a_down_source(tmp_path):
    s = sources._guard("aria.db", lambda: sources.fetch_aria_db(tmp_path / "nope.db"))
    assert not s.ok and not (tmp_path / "nope.db").exists()


def test_coldstart_contacts_read_only(tmp_path):
    path = tmp_path / "coldstart.db"
    _make_db(path, "CREATE TABLE contacts (domain TEXT, name TEXT, title TEXT, email TEXT, status TEXT, "
                   "confidence REAL, source TEXT, checked_at TEXT);"
                   "CREATE TABLE domains (domain TEXT, provider TEXT, catch_all INTEGER, inferred_pattern TEXT);"
                   "INSERT INTO contacts VALUES ('acme.test','A B','CTO','a.b@acme.test','valid',0.9,'site','x');"
                   "INSERT INTO contacts VALUES ('acme.test','C D','','',NULL,0,'',NULL);"
                   "INSERT INTO domains VALUES ('acme.test','google',1,'first.last');")
    data = sources.fetch_coldstart(path)
    assert [c["email"] for c in data["contacts"]] == ["a.b@acme.test"]
    assert data["domains"]["acme.test"]["provider"] == "google"


def test_review_parser():
    text = "## Needs you (1)\n\n| Field | Value |\n|---|---|\n| Salary ✱ | | \n\n## Drafted\n| x | y |\n"
    assert sources.parse_review_needs_me(text) == ["Salary"]
    assert sources.parse_review_needs_me("# nothing") == []


class _Req:
    def __init__(self, i):
        self.i = i


class _FakeSvc:
    """Batch API stand-in: item 1 is rate-limited on the first attempt only (or always)."""

    def __init__(self, always_fail=False):
        self.attempts, self.always_fail = {}, always_fail

    def new_batch_http_request(self):
        svc = self

        class Batch:
            def __init__(self):
                self.items = []

            def add(self, req, callback):
                self.items.append((req, callback))

            def execute(self):
                for req, cb in self.items:
                    n = svc.attempts[req.i] = svc.attempts.get(req.i, 0) + 1
                    if req.i == 1 and (svc.always_fail or n == 1):
                        cb(None, None, RuntimeError("rateLimitExceeded"))
                    else:
                        cb(None, {"id": req.i}, None)
        return Batch()


def test_batch_retries_rate_limited_items():
    out = sources._batch_get(_FakeSvc(), [_Req(i) for i in range(3)], sleep=lambda s: None)
    assert out == [{"id": 0}, {"id": 1}, {"id": 2}]


def test_batch_never_returns_partial_data():
    with pytest.raises(RuntimeError, match="1 of 3 Gmail reads failed"):
        sources._batch_get(_FakeSvc(always_fail=True), [_Req(i) for i in range(3)], sleep=lambda s: None)


def test_parse_thread_flags_bounce_and_reply():
    raw = {"id": "t", "messages": [
        {"id": "1", "internalDate": "1000", "payload": {"headers": [
            {"name": "From", "value": f"Me <{ME}>"}, {"name": "To", "value": "Jo <jo@acme.test>"},
            {"name": "Subject", "value": "Hi"}]}},
        {"id": "2", "internalDate": "2000", "payload": {"headers": [
            {"name": "From", "value": "Mail Delivery Subsystem <mailer-daemon@googlemail.com>"},
            {"name": "Subject", "value": "Delivery Status Notification (Failure)"}]}}]}
    t = sources._parse_thread(raw, ME)
    assert t["messages"][0]["from_me"] and t["messages"][0]["to"] == ["jo@acme.test"]
    assert t["messages"][1]["bounce"]


def test_fetch_sheet_reads_both_tabs(monkeypatch):
    import aria.core.config as config
    monkeypatch.setattr(config, "MUSE_SHEET_ID", "sheet-id")

    class Values:
        def batchGet(self, spreadsheetId, ranges):
            assert spreadsheetId == "sheet-id" and len(ranges) == 2
            return self

        def execute(self):
            return {"valueRanges": [{"values": [["Company", "Role", "Outreach", "Outreach"],
                                                ["Acme", "SWE", "a", "b"], [], ["", ""]]},
                                    {"values": [["Company", "Role", "Status"], ["Hooli", "PE", "Applied"]]}]}

    class Svc:
        def spreadsheets(self):
            return self

        def values(self):
            return Values()

    data = sources.fetch_sheet(Svc())
    assert data["current"] == [{"Company": "Acme", "Role": "SWE", "Outreach": "a", "Outreach_2": "b", "_row": 2}]
    assert data["applied"][0]["Status"] == "Applied"


# ── server ──────────────────────────────────────────────────────────────────

def test_server_binds_localhost_and_serves():
    server = serve(0, build=lambda: "<p>pipeline</p>")
    host, port = server.server_address
    assert host == "127.0.0.1"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        for path, code, body in (("/", 200, "<p>pipeline</p>"), ("/health", 200, "ok"), ("/x", 404, "not found")):
            conn = http.client.HTTPConnection(host, port, timeout=5)
            conn.request("GET", path)
            resp = conn.getresponse()
            assert (resp.status, resp.read().decode()) == (code, body)
            conn.close()
    finally:
        server.shutdown()
        server.server_close()


# ── guard: the dashboard never writes ───────────────────────────────────────

def test_dashboard_has_no_write_calls():
    pattern = re.compile(r"batchUpdate|\.update\(|\.append\(spreadsheetId|\.send\(|\.create\(|\.modify\(|"
                         r"\.trash\(|\.delete\(|INSERT |UPDATE |DELETE |\.commit\(|record_feedback|find_emails|"
                         r"get_company|set_status|write_text|open\([^)]*['\"][wa]")
    offenders = [f"{p.name}: {mm.group(0)}" for p in sorted((ARIA / "dashboard").glob("*.py"))
                 for mm in pattern.finditer(p.read_text())]
    assert not offenders, offenders


def test_needs_me_hides_machine_field_names_and_shows_draft_subject(srcs, tmp_path):
    review = next(iter(srcs["reviews"].data))
    srcs["reviews"].data[review] = ["Current location", "cards[abc-123][field0]", "cards[abc-123][field1]"]
    srcs["gmail"].data["drafts"].append({"id": "d3", "thread_id": "td3", "to": [], "subject": "Intro for a role",
                                         "ts": ts(5)})
    html = render.page(m.build(srcs, NOW))
    assert "Current location, 2 unlabeled fields" in html and "cards[abc" not in html
    assert "no recipient" in html and "Intro for a role" in html


@pytest.mark.parametrize("cell,name", [
    ("LinkedIn: Pat Lee (Recruiter, Acme) — linkedin.com/in/patlee", "Pat Lee"),
    ("Jo Park, Eng Manager", "Jo Park"),
    ("Email draft ready — Sam Roe", ""),
    ("Via Breezy ATS", ""),
    ("jo@acme.test", ""),
    ("", ""),
])
def test_contact_name(cell, name):
    assert m.contact_name(cell) == name
