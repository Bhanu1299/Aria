"""tests/test_outreach_feedback.py — step 4: replies/bounces in the jobs Gmail → Coldstart record_feedback."""
from __future__ import annotations

import sqlite3
from datetime import date

from aria.features.outreach import feedback, pipeline
from aria.integrations.sheets import JobSheet
from tests.test_sheets import HEADERS, FakeSheetsService

ME = "me@jobs.example"


def _msg(i, frm, to=(), subject="ML Engineer — Me", reply=False, bounce=False):
    return {"id": f"m{i}", "ts": 1_760_000_000 + i * 3600, "from": frm, "to": list(to), "subject": subject,
            "is_reply": reply, "from_me": frm == ME, "bounce": bounce}


def _gmail():
    return {"me": ME, "threads": [
        {"id": "t1", "messages": [_msg(1, ME, ["pat@acme.com"]), _msg(2, "pat@acme.com", [ME], "Re: x", reply=True)]},
        {"id": "t2", "messages": [_msg(1, ME, ["ana@globex.com"]),
                                  _msg(2, "mailer-daemon@googlemail.com", [ME], "Delivery Status Notification", bounce=True)]},
        {"id": "t3", "messages": [_msg(1, ME, ["sam@initech.com"])]},                        # no answer yet
        {"id": "t4", "messages": [_msg(1, ME, ["friend@gmail.com"]), _msg(2, "friend@gmail.com", [ME], reply=True)]},
    ]}


def _db(tmp_path, feedback_rows=(), contacts=()):
    path = tmp_path / "coldstart.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE feedback (email TEXT PRIMARY KEY, domain TEXT, pattern TEXT, outcome TEXT, name TEXT)")
    conn.execute("CREATE TABLE contacts (id INTEGER PRIMARY KEY, domain TEXT, name TEXT, email TEXT)")
    conn.executemany("INSERT INTO feedback (email, outcome) VALUES (?,?)", feedback_rows)
    conn.executemany("INSERT INTO contacts (name, email) VALUES (?,?)", contacts)
    conn.commit()
    conn.close()
    return path


def test_outcomes_from_outreach_threads_only():
    assert feedback.outcomes(_gmail()) == {"pat@acme.com": "replied", "ana@globex.com": "bounced"}


def test_sync_records_each_new_outcome_once_with_the_contact_name(tmp_path):
    calls = []
    db = _db(tmp_path, feedback_rows=[("ana@globex.com", "bounced")], contacts=[("Pat Lee", "pat@acme.com")])
    out = feedback.sync(_gmail(), record_fn=lambda e, o, name=None: calls.append((e, o, name)), coldstart_db=db)
    assert calls == [("pat@acme.com", "replied", "Pat Lee")]          # ana's bounce is already in Coldstart
    assert out == {"replied": ["pat@acme.com"], "bounced": [], "error": ""}
    assert feedback.spoken(out) == "Told Coldstart about 1 reply."


def test_changed_outcome_is_recorded_again(tmp_path):
    calls = []
    db = _db(tmp_path, feedback_rows=[("pat@acme.com", "bounced")])
    feedback.sync(_gmail(), record_fn=lambda e, o, name=None: calls.append((e, o)), coldstart_db=db)
    assert ("pat@acme.com", "replied") in calls


def test_sync_never_raises(tmp_path):
    def boom(*a, **k):
        raise ValueError("invalid email")
    out = feedback.sync(_gmail(), record_fn=boom, coldstart_db=_db(tmp_path))
    assert out["replied"] == [] and out["bounced"] == [] and out["error"] == ""
    assert feedback.sync({"me": ME, "threads": [{"id": "x", "messages": [{}]}]}, record_fn=boom)["error"]  # malformed
    assert feedback.sync({}, record_fn=boom) == {"replied": [], "bounced": [], "error": ""}


def test_run_outreach_syncs_feedback_first(monkeypatch):
    seen = []
    monkeypatch.setattr(pipeline.feedback, "sync",
                        lambda data: seen.append(data) or {"replied": ["pat@acme.com"], "bounced": [], "error": ""})
    sheet = JobSheet("s", "Sheet1", service=FakeSheetsService([HEADERS]), today=lambda: date(2026, 10, 20))
    out = pipeline.run_outreach(sheet=sheet, search_fn=lambda q: [], gmail_data=_gmail(), gmail=object())
    assert seen and seen[0]["me"] == ME
    assert out.endswith("Told Coldstart about 1 reply.")
