"""Dashboard Today / My answers / Applications / Data health + review images. Synthetic data only."""
from __future__ import annotations

import http.client
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from aria.dashboard import answers, applications, health, today
from aria.dashboard import model as m
from aria.dashboard.__main__ import serve
from aria.dashboard.sources import Source, parse_review
from aria.features import answer_bank as ab
from aria.features import applicator

NOW = datetime(2026, 10, 21, 12, 0)
ME = "seeker@example.test"


def ts(days):
    return (NOW - timedelta(days=days)).timestamp()


def msg(frm, days, subject="Hi", to=None, bounce=False, reply=False):
    return {"id": f"{frm}{days}", "ts": ts(days), "from": frm, "to": to or [ME], "subject": subject,
            "is_reply": reply, "from_me": frm == ME, "bounce": bounce}


def thread(tid, to, subject, *more):
    return {"id": tid, "messages": [msg(ME, 20, subject, to=[to])] + list(more)}


def row(n, company, role, **kw):
    return {"_row": n, "Company": company, "Role": role, "Status": kw.pop("Status", "Applied"),
            "Apply link": "", "Post link": "", "Who to contact": "", "Profile alignment /5": kw.pop("fit", "3"), **kw}


@pytest.fixture
def mdl():
    sheet = {"current": [], "applied": [
        row(2, "Acme", "SWE I", fit="4", **{"Who to contact": "Pat Lee (Recruiter, Acme)", "Channel": "Email"}),
        row(3, "Acme", "ML Engineer", fit="3"),
        row(4, "Hooli", "Data Engineer", fit="5"),
        row(5, "Initech", "Backend", fit="2", Status="Interview scheduled"),
        row(6, "Globex", "AI Engineer", fit="4", Channel="LinkedIn note"),
        row(7, "Umbrella", "Platform", Status="Rejected (Oct 13)"),
    ]}
    gmail = {"me": ME, "drafts": [], "threads": [
        thread("t1", "pat@acme.test", "SWE I — Seeker", msg("pat@acme.test", 2, "Re: SWE I", reply=True)),
        thread("t2", "jo@hooli.test", "Data Engineer — Seeker",
               msg("mailer-daemon@mail.test", 19, "Delivery Status Notification (Failure)", bounce=True)),
    ]}
    cs = {"contacts": [{"domain": "hooli.test", "name": "Jo Park", "title": "Eng Manager", "email": "jo.park@hooli.test",
                        "status": "valid", "confidence": 0.95, "source": "site", "checked_at": "x"}], "domains": {}}
    srcs = {"sheet": Source(True, sheet), "gmail": Source(True, gmail), "aria_db": Source(True, {"runs": [], "applications": []}),
            "coldstart": Source(True, cs), "reviews": Source(True, {})}
    return m.build(srcs, NOW)


def test_today(mdl):
    d = today.build(mdl)
    assert [t["to"] for t in d["replies"]] == ["pat@acme.test"]
    assert [b["to"] for b in d["bounces"]] == ["jo@hooli.test"] and "jo.park@hooli.test" in d["bounces"][0]["fix"]
    assert [i["job"].company for i in d["interviews"] if i.get("job")] == ["Initech"]
    assert [j.company for j in d["linkedin"]] == ["Globex"]
    assert "1 reply to answer" in today.summary_line(d) and "1 bounce" in today.summary_line(d)
    assert "no Deadline column" in today.render(mdl)


def test_applications_group_by_company(mdl):
    comps = {c["company"]: c for c in applications.build(mdl)}
    acme = {r["job"].role: r for r in comps["Acme"]["roles"]}
    assert acme["SWE I"]["action"] == "Answer their reply" and acme["SWE I"]["bucket"] == "replied"
    assert acme["ML Engineer"]["action"] == "Outreach already sent for SWE I"
    assert comps["Hooli"]["roles"][0]["action"] == "Fix the bounced email"
    assert comps["Umbrella"]["roles"][0]["bucket"] == "closed"
    assert comps["Globex"]["roles"][0]["action"] == "Send the LinkedIn note"
    html = applications.render(mdl)
    assert 'data-f="no_contact"' in html and "Reply received" in html


def test_timeline_order(mdl):
    acme = next(j for j in mdl["jobs"] if j.role == "SWE I")
    whats = [w for _, w, _ in applications.timeline(acme)]
    assert whats[0].startswith("Found by Muse") and "Reply received" in whats and any(w.startswith("Sent to") for w in whats)


def test_health_suggests_jobs(mdl):
    job = health.suggest_job({"what": "x@hooli.test — hello"}, mdl["jobs"])
    assert job is not None and job.company == "Hooli"
    assert health.suggest_job({"what": "x@zzz.test — hi"}, mdl["jobs"]) is None
    assert "Records that match no job" in health.render(mdl)


# ── My answers ──────────────────────────────────────────────────────────────

REVIEW = """# Review: SWE I at Acme

- **Form:** Greenhouse, https://boards.example.test/acme/1
- **Sheet row:** 2
- **Screenshot:** /tmp/acme.png

## Needs you (4)

| Field | Value | Required | Note |
|---|---|---|---|
| How did you hear about us? |  | yes | not answerable from your profile |
| cards[abc][field0] |  | yes | still empty |
| Resume/CV ✱ |  | yes | file upload |
| Phone |  | yes | no answer in your bank yet |

## Guessed by the LLM: check these (1)

| Field | Value | Required | Note |
|---|---|---|---|
| Years of Python | 3 | yes |  |

## Drafted by the LLM: check the wording (1)

### Why Acme?

> I like robots.
> A lot.
"""


@pytest.fixture
def reviews(tmp_path):
    p = tmp_path / "acme.md"
    p.write_text(REVIEW)
    parsed = parse_review(REVIEW)
    parsed.update(path=str(p), mtime=1.0, unlabeled=[{"name": "cards[abc][field0]", "tag": "textarea",
                                                      "nearby": "Tell us about a project", "crop": str(tmp_path / "acme_field0.png")}])
    return {str(p): parsed}


def test_parse_review_sections(reviews):
    r = next(iter(reviews.values()))
    assert (r["role"], r["company"], r["row"]) == ("SWE I", "Acme", 2)
    assert r["sections"]["llm_guess"][0] == {"field": "Years of Python", "value": "3", "required": True, "note": ""}
    assert r["sections"]["llm_draft"][0]["value"] == "I like robots.\nA lot."


def test_new_questions_dedupe_skip_and_unlabeled(reviews, monkeypatch):
    monkeypatch.setattr(ab, "_identity_answers", lambda: {})
    bank = ab.load()
    qs, unlabeled, skipped = answers.new_questions(bank, reviews)
    assert [q["question"] for q in qs] == ["How did you hear about us?", "Phone"]
    assert qs[1]["key"] == "phone" and qs[0]["jobs"][0]["label"] == "Acme — SWE I (row 2)"
    assert skipped == 1 and unlabeled[0]["capture"]["nearby"] == "Tell us about a project"
    ab.set_learned("How did you hear about us?", "LinkedIn")
    assert [q["question"] for q in answers.new_questions(ab.load(), reviews)[0]] == ["Phone"]


def test_save_question_routes_to_key_learned_or_job(monkeypatch):
    monkeypatch.setattr(ab, "_identity_answers", lambda: {})
    answers.act({"op": "save_question", "question": "Phone", "value": "555-0100", "scope": "all"})
    assert json.loads(ab.BANK_PATH.read_text())["answers"]["phone"] == "555-0100"
    answers.act({"op": "save_question", "question": "How did you hear about us?", "value": "LinkedIn", "scope": "all"})
    assert ab.load()["learned"]["how did you hear about us"] == "LinkedIn"
    answers.act({"op": "save_question", "question": "Why Acme?", "value": "Robots", "scope": "job", "job": "acme | swe i"})
    assert ab.lookup("Why Acme?", job={"company": "Acme", "role": "SWE I"}).value == "Robots"
    with pytest.raises(answers.AnswerError):
        answers.act({"op": "save_question", "question": "x", "value": " ", "scope": "all"})
    with pytest.raises(answers.AnswerError):
        answers.act({"op": "drop_tables"})


def test_answers_tab_renders_groups_sources_and_history(reviews, monkeypatch):
    monkeypatch.setattr(ab, "_identity_answers", lambda: {"email": "seeker@example.test"})
    ab.set_answer("school", "State U")
    html = answers.render(Source(True, reviews))
    for text in ("New questions", "Unlabeled fields", "Tell us about a project", "/review-img?f=acme_field0.png",
                 "Identity", "Contact", "Education", "Links", "Work preferences", "Salary rule", "Learned answers",
                 "LLM guesses in your reviews", "History", "identity.json", "State U", "always ask me"):
        assert text in html, text


# ── unlabeled-field capture ─────────────────────────────────────────────────

def test_is_unlabeled():
    assert applicator.is_unlabeled("cards[1a][field0]") and applicator.is_unlabeled("question_12345")
    assert not applicator.is_unlabeled("Current location")


def test_capture_unlabeled_writes_sidecar(tmp_path, monkeypatch):
    from aria.features.ats import FillResult
    monkeypatch.setattr(applicator, "REVIEWS_DIR", tmp_path)

    class El:
        def evaluate(self, js):
            return {"tag": "textarea", "type": "", "placeholder": "", "aria": "", "nearby": "Tell us more"}

        def evaluate_handle(self, js):
            return type("H", (), {"as_element": lambda s: self})()

        def screenshot(self, path):
            Path(path).write_bytes(b"png")

    class Page:
        def query_selector(self, sel):
            return El() if "cards" in sel else None

    import aria.web.agent_browser as agent_browser
    monkeypatch.setattr(agent_browser, "run", lambda fn, tab=None: fn(Page()))
    result = FillResult(ats="generic", review=[{"name": "cards[x][field0]", "label": "cards[x][field0]", "kind": "",
                                                "value": "", "source": "needs_me", "required": True, "note": ""},
                                               {"name": "city", "label": "City", "kind": "text", "value": "",
                                                "source": "needs_me", "required": True, "note": ""}])
    applicator._capture_unlabeled(result, tmp_path / "r.fields.json", "r")
    [entry] = json.loads((tmp_path / "r.fields.json").read_text())
    assert entry["nearby"] == "Tell us more" and Path(entry["crop"]).read_bytes() == b"png"


# ── server: answers endpoint + review images ────────────────────────────────

@pytest.fixture
def srv(tmp_path, monkeypatch):
    monkeypatch.setattr(applicator, "REVIEWS_DIR", tmp_path)
    (tmp_path / "shot.png").write_bytes(b"\x89PNG")
    (tmp_path / "notes.md").write_text("secret")
    calls = []
    s = serve(0, build=lambda: "ok", token="tok", act=lambda p, b: (calls.append(p) or (200, {"ok": True})))
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield s, calls
    s.shutdown()
    s.server_close()


def _get(s, path, method="GET", body=None, headers=None):
    conn = http.client.HTTPConnection(*s.server_address, timeout=5)
    conn.request(method, path, body=json.dumps(body).encode() if body is not None else None,
                 headers={"Content-Type": "application/json", **(headers or {})})
    r = conn.getresponse()
    out = (r.status, r.read())
    conn.close()
    return out


def test_review_images_only_png_in_reviews_dir(srv):
    s, _ = srv
    assert _get(s, "/review-img?f=shot.png") == (200, b"\x89PNG")
    for bad in ("notes.md", "../shot.png", "%2e%2e%2fshot.png", "missing.png", ""):
        assert _get(s, f"/review-img?f={bad}")[0] == 404, bad


def test_answers_post_needs_token(srv):
    s, calls = srv
    assert _get(s, "/api/answers", "POST", {"op": "undo"})[0] == 403 and calls == []
    assert _get(s, "/api/answers", "POST", {"op": "undo"}, {"X-Dashboard-Token": "tok"})[0] == 200
    assert calls == ["/api/answers"]
