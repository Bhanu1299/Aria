"""Job prep §6–7: follow-ups, employer replies, reposts and the do-not-apply list."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

import aria.llm as llm
from aria.features.jobprep import blocklist, followups, replies, reposts


# ---------------------------------------------------------------------------
# do-not-apply
# ---------------------------------------------------------------------------

def test_company_match_ignores_case_punctuation_and_suffixes():
    blocklist.add("Acme, Inc.", "ghosted me")
    assert blocklist.hit({"company": "ACME"})["reason"] == "ghosted me"
    assert blocklist.hit({"company": "acme inc"}) is not None
    assert blocklist.hit({"company": "Acme Rockets"}) is None


def test_domain_matches_host_and_subdomains():
    e = blocklist.add("globex.com")
    assert e["scope"] == "domain"
    assert blocklist.hit({"company": "Other", "url": "https://jobs.globex.com/123"}) is not None
    assert blocklist.hit({"company": "Other", "url": "https://globex.com.evil.io/x"}) is None
    assert blocklist.hit({"company": "Other", "url": "https://notglobex.com/x"}) is None


def test_remove_and_list():
    blocklist.add("Initech")
    assert [x["label"] for x in blocklist.entries()] == ["Initech"]
    assert blocklist.remove("initech, llc")
    assert blocklist.entries() == [] and not blocklist.remove("Initech")


def test_empty_name_is_refused():
    with pytest.raises(ValueError):
        blocklist.add("  ")


def test_batch_selection_skips_listed_companies(monkeypatch):
    from aria.features.applying import session, tabs
    monkeypatch.setattr(tabs, "is_open_for", lambda r: False)
    blocklist.add("Globex")
    rows = [{"company": "Globex", "role": "SWE", "status": "", "alignment": "5", "row": 2},
            {"company": "Acme", "role": "SWE", "status": "", "alignment": "4", "row": 3}]
    held: list = []
    assert [r["company"] for r in session.select_jobs(rows, 5, held)] == ["Acme"]
    assert [r["company"] for r, _e in held] == ["Globex"]
    assert "Globex" in session._blocked_line(held)


def test_apply_job_asks_once_for_a_listed_company(monkeypatch):
    from aria.core.tool import ToolRegistry
    from aria.plugins import apply as plugin
    blocklist.add("Globex", "bad interview")
    job = {"company": "Globex", "role": "SWE", "row": 4, "url": "https://x", "fields": {}, "job_id": "J4"}
    monkeypatch.setattr(plugin.sheets, "get_job", lambda row: job)
    ran = []
    import aria.features.applicator as applicator
    monkeypatch.setattr(applicator, "run_application", lambda j, **k: ran.append(j) or "filled")
    reg = ToolRegistry()
    plugin.ApplyPlugin(None, None, None, "").register(reg)
    tool = reg.get("apply_job")
    first = tool.execute({"row": 4})
    assert "do-not-apply" in first and "still apply" in first and not ran
    assert tool.execute({"row": 4, "still_apply": True}) == "filled" and ran


# ---------------------------------------------------------------------------
# reposts
# ---------------------------------------------------------------------------

def test_same_role_new_url_on_another_day_is_a_repost():
    now = datetime(2026, 10, 26, 9)
    reposts.record({"company": "Acme Inc", "role": "Senior Backend Engineer", "url": "https://acme.com/jobs/1"},
                   now - timedelta(days=20))
    assert reposts.is_repost("Acme", "Backend Engineer (Remote)", now) is None
    reposts.record({"company": "ACME", "role": "Senior Backend Engineer - Remote", "url": "https://acme.com/jobs/2?x=1"},
                   now)
    got = reposts.is_repost("acme", "Senior Backend Engineer", now)
    assert got and got["count"] == 2 and len(got["dates"]) == 2
    assert [r["company"] for r in reposts.reposts(now)] == ["Acme Inc"]


def test_same_url_seen_again_is_not_a_repost():
    reposts.record({"company": "Acme", "role": "SWE", "url": "https://acme.com/1"}, datetime(2026, 10, 1))
    reposts.record({"company": "Acme", "role": "SWE", "url": "https://www.acme.com/1/"}, datetime(2026, 10, 20))
    assert reposts.is_repost("Acme", "SWE", datetime(2026, 10, 21)) is None


def test_two_urls_on_one_day_or_outside_90_days_are_not_reposts():
    reposts.record({"company": "Acme", "role": "SWE", "url": "https://a.com/1"}, datetime(2026, 10, 1, 9))
    reposts.record({"company": "Acme", "role": "SWE", "url": "https://a.com/2"}, datetime(2026, 10, 1, 15))
    assert reposts.is_repost("Acme", "SWE", datetime(2026, 10, 2)) is None
    reposts.record({"company": "Initech", "role": "SWE", "url": "https://i.com/1"}, datetime(2026, 5, 1))
    reposts.record({"company": "Initech", "role": "SWE", "url": "https://i.com/2"}, datetime(2026, 10, 20))
    assert reposts.is_repost("Initech", "SWE", datetime(2026, 10, 21)) is None


# ---------------------------------------------------------------------------
# follow-ups
# ---------------------------------------------------------------------------

def test_business_days_skip_weekends():
    assert followups.add_business_days(date(2026, 10, 23), 7) == date(2026, 11, 3)       # Fri → Tue


def test_cadence_is_7_business_days_then_7_more_max_2():
    job = {"job_id": "J0042", "company": "Acme", "role": "SWE"}
    followups.on_applied(job, date(2026, 10, 1))
    followups.on_applied(job, date(2026, 10, 5))                      # idempotent
    assert followups.due(date(2026, 10, 9)) == []
    due = followups.due(date(2026, 10, 12))
    assert [(r["job_id"], r["n"], r["next_due"]) for r in due] == [("J0042", 1, "2026-10-12")]
    followups.mark_done("J0042", 1, today=date(2026, 10, 12))
    assert followups.due(date(2026, 10, 12)) == []
    assert [(r["n"], r["next_due"]) for r in followups.due(date(2026, 10, 21))] == [(2, "2026-10-21")]
    followups.mark_done("J0042", 2, today=date(2026, 10, 21))
    assert followups.due(date(2027, 1, 1)) == [] and followups.upcoming(today=date(2026, 10, 21)) == []


class FakeGmail:
    def __init__(self, ok=True):
        self.ok, self.drafts, self.sent = ok, [], []

    def save_draft(self, to, subject, body, cc="", reply_to_id="", attachments=()):
        self.drafts.append({"to": to, "subject": subject, "body": body, "reply_to_id": reply_to_id,
                            "attachments": list(attachments)})
        return {"ok": self.ok, "link": "https://mail/d1", "message": "Draft saved" if self.ok else "not connected"}

    def send_message(self, *a, **k):                                   # must never be called
        self.sent.append(a)


def test_draft_due_writes_drafts_to_the_contact_never_sends():
    followups.on_applied({"job_id": "J1", "company": "Acme", "role": "SWE"}, date(2026, 10, 1))
    g = FakeGmail()
    out = followups.draft_due(g, contact_fn=lambda r: {"name": "Dana Lee", "email": "dana@acme.com"},
                              thread_fn=lambda r: None, today=date(2026, 10, 12))
    assert "drafted 1 follow-up" in out and "Nothing was sent" in out
    d = g.drafts[0]
    assert d["to"] == "dana@acme.com" and d["body"].startswith("Hi Dana,") and "SWE" in d["subject"]
    assert g.sent == []
    assert followups.due(date(2026, 10, 12)) == []


def test_draft_due_replies_in_the_application_thread_without_a_contact():
    followups.on_applied({"job_id": "J2", "company": "Globex", "role": "Data Engineer"}, date(2026, 10, 1))
    g = FakeGmail()
    followups.draft_due(g, contact_fn=lambda r: None, thread_fn=lambda r: "msg-77", today=date(2026, 10, 12))
    assert g.drafts[0]["reply_to_id"] == "msg-77" and g.drafts[0]["to"] == ""


def test_draft_due_attaches_the_tailored_resume(tmp_path):
    from aria.features.resume import prepare
    folder = prepare.JOBS_DIR / "J3-initech"
    folder.mkdir(parents=True)
    (folder / "resume.pdf").write_bytes(b"%PDF-1.4")
    followups.on_applied({"job_id": "J3", "company": "Initech", "role": "SWE"}, date(2026, 10, 1))
    g = FakeGmail()
    followups.draft_due(g, contact_fn=lambda r: {"name": "", "email": "hr@initech.com"}, today=date(2026, 10, 12))
    assert g.drafts[0]["attachments"] == [str(folder / "resume.pdf")]
    assert g.drafts[0]["body"].startswith("Hello,")


def test_draft_due_reports_what_it_couldnt_do():
    followups.on_applied({"job_id": "J4", "company": "Hooli", "role": "SWE"}, date(2026, 10, 1))
    out = followups.draft_due(FakeGmail(), contact_fn=lambda r: None, thread_fn=lambda r: None,
                              today=date(2026, 10, 12))
    assert "Couldn't draft 1" in out and "Hooli" in out
    assert len(followups.due(date(2026, 10, 12))) == 1                   # still due


def test_nothing_due_says_when_the_next_one_is():
    followups.on_applied({"job_id": "J5", "company": "Acme", "role": "SWE"}, date(2026, 10, 20))
    assert "next is Acme on 2026-10-29" in followups.draft_due(FakeGmail(), today=date(2026, 10, 21))


def test_marking_applied_schedules_the_first_follow_up(monkeypatch):
    from aria.features import applicator
    applicator.schedule_followup({"job_id": "J9", "company": "Acme", "role": "SWE"})
    assert followups.upcoming(today=date.today())[0]["job_id"] == "J9"


# ---------------------------------------------------------------------------
# replies
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("subject,snippet,kind", [
    ("Your application to Acme", "Unfortunately we have decided to move forward with other candidates.", "rejected"),
    ("Interview invitation — Acme", "We'd love to schedule a call. Please share your availability.", "interview"),
    ("Offer letter", "We are pleased to extend an offer of employment.", "offer"),
    ("Thank you for applying to Acme", "We have received your application.", "noise"),
    ("Thank you for your interest — next steps", "Unfortunately the position has been filled.", "rejected"),
])
def test_rules_classify(subject, snippet, kind):
    assert replies.classify(subject, snippet, "talent@acme.com", use_llm=False) == (kind, "rules")


def test_llm_only_when_rules_cant_tell(monkeypatch):
    calls = []

    def complete_checked(task, messages, validate, system=None, max_tokens=0):
        calls.append((task, messages[0]["content"], system))
        return SimpleNamespace(text="Interview", stop_reason="end_turn"), validate("Interview")

    monkeypatch.setattr(llm.llm_client, "complete_checked", complete_checked)
    assert replies.classify("Quick question", "Are you free Thursday to chat with Sam?", "sam@acme.com") == \
        ("interview", "llm")
    assert calls[0][0] == "reply_classify" and "<<<DATA" in calls[0][1]
    replies.classify("Unfortunately", "", "x@acme.com")
    assert len(calls) == 1


JOBS = [{"job_id": "J1", "company": "Acme", "role": "SWE", "row": 5},
        {"job_id": "J2", "company": "Globex Corporation", "role": "Data Engineer", "row": 9}]


def test_scan_matches_jobs_and_stores_suggestions_once():
    msgs = [{"id": "m1", "from": "Acme Talent <talent@acme.com>", "subject": "Interview request",
             "snippet": "Can we schedule a call?"},
            {"id": "m2", "from": "no-reply@greenhouse.io", "subject": "Globex Corporation: application received",
             "snippet": "Thank you for applying"},
            {"id": "m3", "from": "friend@gmail.com", "subject": "lunch?", "snippet": ""}]
    new = replies.scan(msgs, JOBS, use_llm=False)
    assert [(s["company"], s["kind"], s["suggested_status"]) for s in new] == \
        [("Acme", "interview", "Interview"), ("Globex Corporation", "noise", "")]
    assert replies.scan(msgs, JOBS, use_llm=False) == []                 # each message once
    assert [p["company"] for p in replies.pending()] == ["Acme"]         # noise never asks


class FakeSheet:
    def __init__(self):
        self.calls = []

    def set_cell(self, row, column, value, expect=None, company=None, role=None, job_id=None):
        self.calls.append((row, column, value, company, role, job_id))
        return row


def test_nothing_is_written_until_approve():
    replies.scan([{"id": "m9", "from": "hr@acme.com", "subject": "Regret to inform", "snippet": ""}], JOBS,
                 use_llm=False)
    sheet = FakeSheet()
    sid = replies.pending()[0]["id"]
    assert sheet.calls == []
    assert "Rejected" in replies.approve(sid, sheet)
    assert sheet.calls == [(5, "Status", "Rejected", "Acme", "SWE", "J1")]
    assert replies.pending() == []
    with pytest.raises(ValueError):
        replies.approve(sid, sheet)


def test_dismiss_writes_nothing():
    replies.scan([{"id": "m10", "from": "hr@acme.com", "subject": "Offer letter", "snippet": ""}], JOBS, use_llm=False)
    replies.dismiss(replies.pending()[0]["id"])
    assert replies.pending() == []


# ---------------------------------------------------------------------------
# voice routes + dashboard
# ---------------------------------------------------------------------------

def test_voice_routes():
    from aria.dispatch import parse
    from aria.dispatch.rules import RuleSet
    rs = RuleSet.load()
    assert rs.match(parse.normalize("draft my follow-ups")).tool == "draft_followups"
    got = rs.match(parse.normalize("never apply to Globex again"))
    assert (got.tool, got.params) == ("do_not_apply", {"action": "add", "company": "globex"})
    assert rs.match(parse.normalize("never apply to them")) is None


def test_dashboard_cards_and_actions():
    from aria.dashboard import jobprep
    blocklist.add("Globex", "ghosted")
    replies.scan([{"id": "m1", "from": "hr@acme.com", "subject": "Interview", "snippet": ""}], JOBS, use_llm=False)
    followups.on_applied({"job_id": "J1", "company": "Acme", "role": "SWE"}, date.today() - timedelta(days=30))
    html = jobprep.render({"jobs": [], "now": datetime.now()})
    assert "Application follow-ups due" in html and "Draft my follow-ups" in html
    assert "Employer replies" in html and 'data-op="approve_reply"' in html and "Set Status → Interview" in html
    assert "Do-not-apply list" in html and "Globex" in html
    assert jobprep.act({"op": "dna_add", "text": "hooli.com"})["message"].startswith("Added hooli.com")
    assert jobprep.act({"op": "dna_remove", "text": "Globex"})["message"] == "Removed."
    with pytest.raises(jobprep.JobPrepError):
        jobprep.act({"op": "nope"})
    with pytest.raises(jobprep.JobPrepError):
        jobprep.act({"op": "approve_reply", "id": "x"})


def test_dashboard_post_route_needs_the_token():
    from aria.dashboard import __main__ as dash
    status, out = dash.settings_action("/api/jobprep", {"op": "dna_add", "text": "Initech"})
    assert status == 200 and "Initech" in out["message"]
    status, out = dash.settings_action("/api/jobprep", {"op": "dna_add", "text": ""})
    assert status == 400


def test_unfortunately_with_an_interview_is_an_interview():
    assert replies.classify("Re: interview", "Unfortunately we need to reschedule your interview to Friday.",
                            "hr@acme.com", use_llm=False) == ("interview", "rules")
