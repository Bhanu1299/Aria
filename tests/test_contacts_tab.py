"""tests/test_contacts_tab.py — the Contacts tab: header-keyed sheet, pending/dedupe, waves,
stop-on-reply, and the end-to-end run with Coldstart, the LLM and Gmail mocked."""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from aria.features.outreach import contacts_tab as ct
from aria.features.outreach import writer
from aria.integrations import coldstart_client as cs
from aria.integrations.header_sheet import ROW, HeaderSheet
from aria.integrations.sheets import SheetError, col_letter

# The real Contacts header (its order differs from the spec's — code must not care).
HEADERS = ["Company", "Tracked role", "Person", "Title", "Email", "Source", "Confidence", "LinkedIn URL",
           "Job URL", "Domain", "Email pattern", "Find status", "Draft status", "Gmail draft link", "Wave",
           "Updated", "Notes", "Company problem", "Signal", "Signal source", "Angle", "Added by"]

# Muse's research, filled on every row unless a test says otherwise.
PROBLEM = "Teams can't tell which LLM answers are wrong until customers complain."
ANGLE = "Built an eval harness for a voice agent; the hard part was grading open-ended answers."
BODY = ("SUBJECT: catching wrong LLM answers\n\n"
        "You're trying to catch wrong answers before customers complain, which is mostly a grading problem.\n\n"
        "I built an eval harness for my voice agent. Grading open-ended answers was the hard part, since "
        "there's rarely one right string.\n\n"
        "I keep running into this while building Aria, so I'd like to work on it full time.")
RESUME = "/r/resume.pdf"


def r(company, person, **kw):
    kw.setdefault("problem", PROBLEM)
    kw.setdefault("angle", ANGLE)
    cells = {"Company": company, "Person": person}
    cells.update({MAP[k]: v for k, v in kw.items()})
    out = [""] * len(HEADERS)
    for k, v in cells.items():
        out[HEADERS.index(k)] = v
    while out and out[-1] == "":
        out.pop()
    return out


MAP = {"role": "Tracked role", "title": "Title", "email": "Email", "confidence": "Confidence",
       "linkedin": "LinkedIn URL", "job_url": "Job URL", "domain": "Domain", "pattern": "Email pattern",
       "find": "Find status", "draft": "Draft status", "wave": "Wave", "notes": "Notes", "source": "Source",
       "link": "Gmail draft link", "problem": "Company problem", "signal": "Signal", "angle": "Angle"}


class LiveSheet:
    """spreadsheets().values() over an in-memory grid; batchUpdate edits the grid."""

    def __init__(self, values):
        self.grid = [list(v) for v in values]
        self.writes = []

    def spreadsheets(self):
        return self

    def values(self):
        return self

    def get(self, spreadsheetId, range):
        return SimpleNamespace(execute=lambda: {"values": [list(v) for v in self.grid]})

    def batchUpdate(self, spreadsheetId, body):
        for d in body["data"]:
            cell = d["range"].split("!")[1]
            letters = "".join(c for c in cell if c.isalpha())
            n = int("".join(c for c in cell if c.isdigit()))
            col = next(i for i in range(200) if col_letter(i) == letters)
            while len(self.grid) < n:
                self.grid.append([])
            line = self.grid[n - 1]
            line += [""] * (col + 1 - len(line))
            line[col] = d["values"][0][0]
            self.writes.append((n, HEADERS[col] if col < len(HEADERS) else letters, line[col]))
        return SimpleNamespace(execute=lambda: {})

    def cell(self, n, header):
        line = self.grid[n - 1]
        i = self.grid[0].index(header)
        return line[i] if i < len(line) else ""


def sheet_of(*rows, headers=HEADERS):
    svc = LiveSheet([headers, *rows])
    return HeaderSheet("sheet-id", "Contacts", service=svc, writable=ct.WRITABLE), svc


def coldstart(answers: dict):
    """answers: name → JSON reply (or an int HTTP status)."""
    calls = []

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"ok": True})
        body = json.loads(request.content)
        calls.append(body)
        a = answers[body["name"]]
        return httpx.Response(a, json={}) if isinstance(a, int) else httpx.Response(200, json=a)
    client = cs.ColdstartClient(base_url="http://cs.test", min_interval=0,
                                client=httpx.Client(transport=httpx.MockTransport(handler)))
    return client, calls


class FakeGmail:
    account = "jobs"

    def __init__(self, ok=True):
        self.ok, self.drafts = ok, []

    def save_draft(self, to, subject, body, cc="", reply_to_id="", attachments=()):
        self.drafts.append({"to": to, "subject": subject, "body": body, "attachments": list(attachments)})
        if not self.ok:
            return {"ok": False, "message": "Gmail is rate-limited."}
        n = len(self.drafts)
        return {"ok": True, "message": "Draft saved", "draft_id": f"r{n}", "message_id": f"m{n}",
                "link": f"https://mail.google.com/mail/u/candidate@example.com/#drafts?compose=m{n}"}

    def send_message(self, *a, **k):
        raise AssertionError("Aria never sends")


@pytest.fixture(autouse=True)
def env(monkeypatch):
    bank = {"answers": {"first_name": "Bhanu", "last_name": "Teja", "linkedin": "https://linkedin.com/in/me"}}
    monkeypatch.setattr(writer.answer_bank, "load", lambda: bank)
    monkeypatch.setattr(writer, "_identity", lambda: {"summary": "Builds voice agents."})
    monkeypatch.setattr(writer, "_resume", lambda: 'Voice agent with a tool-calling loop (Aria), 2025.')
    monkeypatch.setattr(writer, "_history", lambda: [{"label": "LLM Gateway", "key": "llm gateway",
                                                      "text": "Routes prompts between small and frontier models."}])
    monkeypatch.setattr(ct.google_auth, "account_email", lambda account: "candidate@example.com")
    monkeypatch.setattr(ct.config, "OUTREACH_FROM", "candidate@example.com")
    monkeypatch.setattr(ct.config, "CONTACTS_PER_COMPANY", 3)


@pytest.fixture
def llm(monkeypatch):
    calls = []

    def complete_checked(task, messages, validate, system=None, max_tokens=1024):
        calls.append({"task": task, "content": messages[0]["content"]})
        return SimpleNamespace(text=BODY, stop_reason="end_turn"), validate(BODY)
    monkeypatch.setattr(writer.llm_client, "complete_checked", complete_checked)
    return calls


def run(sheet, client, gmail=None, **kw):
    lines = []
    kw.setdefault("resume", RESUME)
    out = ct.run(sheet=sheet, client=client, gmail=gmail or FakeGmail(), step_fn=lines.append, **kw)
    return out, lines


# ---------------------------------------------------------------------------
# header sheet
# ---------------------------------------------------------------------------

def test_records_and_writes_follow_headers_not_positions():
    shuffled = ["Notes", "Person", "Company", "Email", "Find status", "Updated"]
    svc = LiveSheet([shuffled, ["", "Pat Lee", "Acme"]])
    sheet = HeaderSheet("id", "Contacts", service=svc)
    rec = sheet.records()[0]
    assert (rec["Company"], rec["Person"], rec["Email"], rec[ROW]) == ("Acme", "Pat Lee", "", 2)
    sheet.update(2, {"email": "pat@acme.com", "FIND STATUS": "done"})      # case-insensitive
    assert svc.grid[1] == ["", "Pat Lee", "Acme", "pat@acme.com", "done"]


def test_update_refuses_unknown_and_user_columns():
    sheet, _ = sheet_of(r("Acme", "Pat Lee"))
    with pytest.raises(SheetError, match="no 'Phone' column"):
        sheet.update(2, {"Phone": "1"})
    with pytest.raises(SheetError, match="doesn't write the 'Person'"):
        sheet.update(2, {"Person": "Someone"})


def test_update_refinds_a_row_that_moved():
    sheet, svc = sheet_of(r("Acme", "Pat Lee"))
    svc.grid.insert(1, r("Beta", "Ana Diaz"))                  # a row inserted above since the read
    assert sheet.update(2, {"Email": "pat@acme.com"}, match={"Company": "Acme", "Person": "Pat Lee"}) == 3
    assert svc.cell(3, "Email") == "pat@acme.com" and svc.cell(2, "Email") == ""


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def _recs(*rows):
    return HeaderSheet("id", "Contacts", service=LiveSheet([HEADERS, *rows])).records()


def test_pending_rules():
    recs = _recs(
        r("Acme", "Pat Lee"),                                   # 2 pending
        r("Acme", "—"),                                         # no person
        r("Acme", "(hiring contact)"),                          # placeholder
        r("Beta", "Ana Diaz", email="ana@beta.com"),            # has email
        r("Gamma", "Sam Roe", find="done"),                     # done
        r("Delta", "Kim Ito", find="pending"),                  # 7 pending
        r("Eps", "Lu Wen", find="error"),                       # errors don't auto-retry
        r("", "No Company"),
    )
    todo, _ = ct.pending(recs)
    assert [x[ROW] for x in todo] == [2, 7]


def test_dedupe_by_linkedin_then_company_and_person():
    recs = _recs(
        r("Acme", "Pat Lee", linkedin="https://www.linkedin.com/in/patlee/"),
        r("Acme Inc", "Patrick Lee", linkedin="https://linkedin.com/in/patlee?trk=x"),    # same profile
        r("Beta", "Ana Diaz"),
        r("Beta (YC W24)", "ana diaz"),                                                   # same company+person
        r("Gamma", "Sam Roe", email="sam@gamma.com", find="done"),
        r("Gamma", "Sam Roe"),                                                            # already looked up
    )
    todo, skipped = ct.pending(recs)
    assert [x[ROW] for x in todo] == [2, 4]
    assert sorted(x[ROW] for x, _ in skipped) == [3, 5, 7]


def test_domain_resolution_order():
    recs = _recs(
        r("Acme", "Pat Lee", domain="https://www.Acme.com/"),
        r("Beta", "Ana Diaz", job_url="https://careers.beta.io/jobs/1"),
        r("Gamma", "Sam Roe", job_url="https://jobs.lever.co/gamma/1"),
        r("Gamma", "Kim Ito", email="kim@gamma.ai", find="done"),
        r("Delta", "Lu Wen", job_url="https://boards.greenhouse.io/delta/1"),
    )
    assert ct.resolve_domain(recs[0], recs) == ("acme.com", "Domain column")
    assert ct.resolve_domain(recs[1], recs) == ("beta.io", "Job URL")
    assert ct.resolve_domain(recs[2], recs) == ("gamma.ai", "email on row 5")    # job board isn't the company
    search = lambda q: [{"url": "https://delta.dev/"}]
    assert ct.resolve_domain(recs[4], recs, search) == ("delta.dev", "web search")
    assert ct.resolve_domain(recs[4], recs)[0] == ""


def test_waves_cap_and_stagger():
    recs = _recs(*[r("Acme", n, email=f"{n.split()[0].lower()}@acme.com", confidence=c, draft=d)
                   for n, c, d in [("Pat Lee", "verified", "sent"), ("Ana Diaz", "pattern", ""),
                                   ("Sam Roe", "verified", "none"), ("Kim Ito", "verified", ""),
                                   ("Lu Wen", "not found", "")]])
    recs[0]["Wave"] = "1"
    queue, waves = ct.draft_queue(recs, cap=3)
    assert waves == {3: 2, 4: 3}                 # cap 3: Pat (wave 1) + two more; Kim is left out
    assert [x[ROW] for x in queue] == [3]        # wave 1 sent → wave 2 drafted; wave 3 waits


def test_next_wave_waits_until_the_earlier_one_is_sent():
    recs = _recs(r("Acme", "Pat Lee", email="pat@acme.com", confidence="verified", draft="drafted", wave="1"),
                 r("Acme", "Ana Diaz", email="ana@acme.com", confidence="pattern"))
    queue, waves = ct.draft_queue(recs)
    assert queue == [] and waves == {3: 2}


def test_stop_on_first_reply():
    recs = _recs(r("Acme", "Pat Lee", email="pat@acme.com", confidence="verified", draft="replied"),
                 r("Acme Inc", "Ana Diaz", email="ana@acme.com", confidence="verified", draft="drafted"),
                 r("Acme", "Sam Roe", email="sam@acme.com", confidence="pattern", draft="none"),
                 r("Beta", "Kim Ito", email="kim@beta.com", confidence="verified"))
    ups = ct.stop_updates(recs)
    assert [(x[ROW], f["Draft status"]) for x, f in ups] == [(3, "no reply"), (4, "no reply")]
    assert "Pat Lee replied" in ups[0][1]["Notes"]
    queue, _ = ct.draft_queue(recs)
    assert [x[ROW] for x in queue] == [5]        # only the other company


# ---------------------------------------------------------------------------
# the run (thin slice + rules)
# ---------------------------------------------------------------------------

def test_thin_slice_one_contact_end_to_end(llm):
    sheet, svc = sheet_of(
        r("Acme", "Pat Lee", role="Founding Engineer", title="CTO", job_url="https://acme.com/careers/founding"),
        r("Acme", "Ana Diaz", role="Founding Engineer", title="Eng Manager"),
    )
    client, calls = coldstart({"Pat Lee": {"email": "pat.lee@acme.com", "status": "verified",
                                           "confidence": 0.97, "pattern": "{first}.{last}"}})
    gmail = FakeGmail()
    out, lines = run(sheet, client, gmail, limit=1)

    assert calls == [{"name": "Pat Lee", "company": "Acme", "domain": "acme.com"}]
    for header, want in [("Domain", "acme.com"), ("Email", "pat.lee@acme.com"), ("Source", "Coldstart"),
                         ("Confidence", "verified"), ("Email pattern", "{first}.{last}"), ("Find status", "done"),
                         ("Draft status", "drafted"), ("Wave", "1"),
                         ("Gmail draft link", "https://mail.google.com/mail/u/candidate@example.com/#drafts?compose=m1")]:
        assert svc.cell(2, header) == want, header
    assert svc.cell(2, "Updated")
    assert svc.cell(3, "Find status") == "" and svc.cell(3, "Email") == ""    # limit=1 left row 3 alone
    d = gmail.drafts[0]
    assert d["to"] == "pat.lee@acme.com" and d["subject"] == "catching wrong LLM answers"
    assert d["body"].startswith("Hi Pat,\n\nYou're trying") and d["attachments"] == [RESUME]
    assert "I applied for the Founding Engineer role. Resume attached." in d["body"]
    assert d["body"].endswith("Thanks,\nBhanu\nlinkedin.com/in/me")
    assert llm[0]["task"] == "outreach_email"
    assert f"Company problem: {PROBLEM}" in llm[0]["content"] and f"Angle (the only experience you may claim): {ANGLE}" in llm[0]["content"]
    assert [l.split(":")[0] for l in lines if l[:1].isdigit()] == ["1. read", "2. stop on reply", "3. pending",
                                                                  "4. find row 2", "5. draft"]
    assert out == "Contacts: 1 verified, 0 pattern, 0 not found; 1 draft in the jobs Gmail. Nothing was sent."


def test_pattern_reused_for_the_second_contact_at_a_company(llm):
    sheet, svc = sheet_of(r("Acme", "Pat Lee", domain="acme.com"), r("Acme", "Ana Diaz"))
    client, calls = coldstart({"Pat Lee": {"email": "pat.lee@acme.com", "status": "pattern_guess",
                                           "confidence": 0.6, "pattern": "first.last"}})
    run(sheet, client, draft=False)
    assert len(calls) == 1
    assert (svc.cell(3, "Email"), svc.cell(3, "Confidence")) == ("ana.diaz@acme.com", "pattern")
    assert svc.cell(3, "Source").startswith("Coldstart pattern cache")


def test_sheet_pattern_seeds_the_cache(llm):
    sheet, svc = sheet_of(r("Acme", "Pat Lee", email="pat@acme.com", find="done", pattern="{first}", domain="acme.com"),
                          r("Acme", "Ana Diaz"))
    client, calls = coldstart({})
    run(sheet, client, draft=False)
    assert calls == [] and svc.cell(3, "Email") == "ana@acme.com"


def test_not_found_is_done_with_a_note_and_no_draft(llm):
    sheet, svc = sheet_of(r("Acme", "Pat Lee", domain="acme.com"))
    client, _ = coldstart({"Pat Lee": {"email": None, "status": "not_found", "confidence": 0}})
    gmail = FakeGmail()
    out, _ = run(sheet, client, gmail)
    assert (svc.cell(2, "Find status"), svc.cell(2, "Confidence"), svc.cell(2, "Email")) == ("done", "not found", "not found")
    assert "no address" in svc.cell(2, "Notes") and gmail.drafts == [] and llm == []
    client2, calls2 = coldstart({})
    run(sheet, client2, gmail)                      # no retry loop
    assert calls2 == []


def test_coldstart_error_marks_error_and_moves_on(llm):
    sheet, svc = sheet_of(r("Acme", "Pat Lee", domain="acme.com"), r("Beta", "Ana Diaz", domain="beta.com"))
    client, _ = coldstart({"Pat Lee": 503, "Ana Diaz": {"email": "ana@beta.com", "status": "verified"}})
    out, _ = run(sheet, client)
    assert svc.cell(2, "Find status") == "error" and "HTTP 503" in svc.cell(2, "Notes")
    assert svc.cell(3, "Find status") == "done" and svc.cell(3, "Draft status") == "drafted"
    assert "1 lookup error" in out


def test_no_domain_is_an_error_without_calling_coldstart(llm):
    sheet, svc = sheet_of(r("Acme", "Pat Lee", job_url="https://jobs.lever.co/acme/1"))
    client, calls = coldstart({})
    run(sheet, client)
    assert calls == [] and svc.cell(2, "Find status") == "error" and "no domain" in svc.cell(2, "Notes")


def test_no_drafts_when_the_jobs_gmail_is_another_account(llm, monkeypatch):
    monkeypatch.setattr(ct.google_auth, "account_email", lambda account: "someone.else@gmail.com")
    sheet, svc = sheet_of(r("Acme", "Pat Lee", email="pat@acme.com", confidence="verified", find="done"))
    gmail = FakeGmail()
    run(sheet, coldstart({})[0], gmail)
    assert gmail.drafts == [] and svc.cell(2, "Draft status") == ""


def test_no_drafts_without_a_sender_address(llm, monkeypatch):
    monkeypatch.setattr(ct.config, "OUTREACH_FROM", "")
    sheet, svc = sheet_of(r("Acme", "Pat Lee", email="pat@acme.com", confidence="verified", find="done"))
    gmail = FakeGmail()
    _, lines = run(sheet, coldstart({})[0], gmail)
    assert gmail.drafts == [] and any("OUTREACH_FROM" in l for l in lines)


def test_coldstart_down_writes_nothing_and_leaves_rows_pending(llm):
    sheet, svc = sheet_of(r("Acme", "Pat Lee", domain="acme.com"))
    client, calls = coldstart({})
    client.ensure_running = lambda: "Coldstart isn't running at http://cs.test (refused)"
    out, _ = run(sheet, client)
    assert calls == [] and svc.writes == [] and "isn't running" in out
    assert ct.pending(sheet.records())[0]


def test_real_coldstart_find_reply_is_understood(llm):
    """The shape Coldstart's POST /find actually sends (extra keys, its own source names)."""
    sheet, svc = sheet_of(r("Acme", "Pat Lee", domain="acme.com"))
    client, _ = coldstart({"Pat Lee": {"email": "plee@acme.com", "status": "pattern_guess", "confidence": 0.41,
                                       "pattern": "{f}{last}", "source": "pattern", "smtp_status": "catch_all"}})
    run(sheet, client, draft=False)
    assert (svc.cell(2, "Email"), svc.cell(2, "Confidence"), svc.cell(2, "Email pattern"), svc.cell(2, "Source")) == \
        ("plee@acme.com", "pattern", "{f}{last}", "Coldstart (pattern)")


def test_gmail_failure_leaves_the_row_undrafted_with_a_note(llm):
    sheet, svc = sheet_of(r("Acme", "Pat Lee", email="pat@acme.com", confidence="verified", find="done"))
    run(sheet, coldstart({})[0], FakeGmail(ok=False))
    assert svc.cell(2, "Draft status") == "" and "rate-limited" in svc.cell(2, "Notes")


def test_user_statuses_are_never_written(llm):
    sheet, svc = sheet_of(r("Acme", "Pat Lee", email="pat@acme.com", confidence="verified", draft="approved", wave="1"),
                          r("Acme", "Ana Diaz", email="ana@acme.com", confidence="verified", draft="sent", wave="2"))
    run(sheet, coldstart({})[0])
    assert [w for w in svc.writes if w[1] == "Draft status"] == []
    with pytest.raises(SheetError):
        ct._Writer(sheet, lambda m: None)(sheet.records()[0], {"Draft status": "sent"})


def test_reply_closes_the_company_in_the_sheet(llm):
    sheet, svc = sheet_of(r("Acme", "Pat Lee", email="pat@acme.com", confidence="verified", draft="replied"),
                          r("Acme", "Ana Diaz", email="ana@acme.com", confidence="verified", draft="none"),
                          r("Acme", "Sam Roe"))
    client, calls = coldstart({"Sam Roe": {"email": "sam@acme.com", "status": "verified"}})
    gmail = FakeGmail()
    run(sheet, client, gmail)
    assert svc.cell(3, "Draft status") == "no reply" and svc.cell(4, "Draft status") == "no reply"
    assert gmail.drafts == []


def test_background_run_reports_when_done(llm):
    import threading
    sheet, svc = sheet_of(r("Acme", "Pat Lee", domain="acme.com"))
    client, _ = coldstart({"Pat Lee": {"email": "pat@acme.com", "status": "verified"}})
    done = threading.Event()
    got = []
    assert ct.start_background(done_fn=lambda s: (got.append(s), done.set()), sheet=sheet, client=client,
                               gmail=FakeGmail(), resume=RESUME)
    assert done.wait(5) and "1 draft" in got[0]


def test_a_person_listed_twice_gets_one_wave_at_the_company_address():
    recs = _recs(r("Superset (YC P26)", "Kiet Ho", email="kiet@gmail.com", confidence="verified"),
                 r("Superset", "Kiet Ho", email="kiet@superset.sh", confidence="pattern"),
                 r("Superset", "Ana Diaz", email="ana@superset.sh", confidence="pattern"))
    queue, waves = ct.draft_queue(recs)
    assert waves == {3: 1, 4: 2} and [x[ROW] for x in queue] == [3]


def test_one_with_nothing_pending_drafts_the_next_ready_row(llm):
    sheet, svc = sheet_of(r("Acme", "Pat Lee", email="pat@acme.com", confidence="verified", find="done"),
                          r("Beta", "Ana Diaz", email="ana@beta.com", confidence="verified", find="done"))
    gmail = FakeGmail()
    run(sheet, coldstart({})[0], gmail, limit=1)
    assert [d["to"] for d in gmail.drafts] == ["pat@acme.com"]
    assert svc.cell(2, "Draft status") == "drafted" and svc.cell(3, "Draft status") == ""


def test_writer_reads_the_real_identity_and_resume_helpers(monkeypatch):
    """The env fixture mocks writer._identity; this calls the real one (its import had gone stale)."""
    monkeypatch.undo()
    assert isinstance(writer._identity(), dict) and isinstance(writer._resume(), str)


@pytest.mark.parametrize("cells", [{"problem": ""}, {"angle": "unknown"}, {"problem": "Unknown", "angle": ""}])
def test_no_problem_or_angle_means_needs_research_and_no_draft(llm, cells):
    sheet, svc = sheet_of(r("Acme", "Pat Lee", email="pat@acme.com", confidence="verified", find="done", **cells))
    gmail = FakeGmail()
    out, _ = run(sheet, coldstart({})[0], gmail)
    assert gmail.drafts == [] and llm == [] and svc.cell(2, "Draft status") == ""
    assert svc.cell(2, "Notes").endswith("needs research") and "1 needs research" in out
    run(sheet, coldstart({})[0], gmail)                        # the note isn't repeated
    assert svc.cell(2, "Notes").count("needs research") == 1


def test_generic_inbox_is_drafted_to_the_team(llm):
    sheet, svc = sheet_of(r("Acme (YC W24)", "—", role="AI Engineer", email="team@acme.com", confidence="verified",
                            find="done"),
                          r("Acme (YC W24)", "—", role="AI Engineer", email="hello@acme.com", confidence="verified",
                            find="done"))                       # one inbox per company
    gmail = FakeGmail()
    run(sheet, coldstart({})[0], gmail)
    [d] = gmail.drafts
    assert d["to"] == "team@acme.com" and d["body"].startswith("Hi Acme team,")
    assert "Could you pass this to whoever is hiring for the AI Engineer role?" in d["body"]
    assert "a generic company inbox" in llm[0]["content"]


def test_pattern_address_gets_a_pattern_guess_note(llm):
    sheet, svc = sheet_of(r("Acme", "Pat Lee", email="pat@acme.com", confidence="pattern", find="done"))
    run(sheet, coldstart({})[0])
    assert svc.cell(2, "Draft status") == "drafted" and svc.cell(2, "Notes").endswith("pattern guess")


def test_a_row_with_a_draft_link_is_never_drafted_again(llm):
    sheet, _ = sheet_of(r("Acme", "Pat Lee", email="pat@acme.com", confidence="verified", draft="none",
                          link="https://mail.google.com/mail/u/x/#drafts?compose=m9"))
    gmail = FakeGmail()
    run(sheet, coldstart({})[0], gmail)
    assert gmail.drafts == []


def test_no_resume_pdf_means_no_drafts(llm):
    sheet, svc = sheet_of(r("Acme", "Pat Lee", email="pat@acme.com", confidence="verified", find="done"))
    gmail = FakeGmail()
    _, lines = run(sheet, coldstart({})[0], gmail, resume="")
    assert gmail.drafts == [] and any("no resume PDF" in l for l in lines)


def test_brief_comes_from_the_research_columns():
    rec = _recs(r("Aurelian (YC)", "Max Keenan", title="CEO / co-founder", role="Software Engineer (New Grad)",
                  signal="MACC 911 automated its non-emergency line"))[0]
    b = ct.brief_of(rec)
    assert (b.company, b.role, b.problem, b.angle, b.signal, b.person, b.kind) == (
        "Aurelian (YC)", "Software Engineer (New Grad)", PROBLEM, ANGLE, "MACC 911 automated its non-emergency line",
        "Max Keenan", writer.ENGINEER)


def test_role_override_is_noted_and_the_row_is_still_drafted(llm, monkeypatch):
    monkeypatch.setattr(writer, "plan", lambda brief: writer.Plan(
        {"label": "Eval harness", "key": "eval harness", "text": "Eval harness for a voice agent; grading open-ended answers."},
        use_angle=False, override=True, why="analogy"))
    sheet, svc = sheet_of(r("Acme", "Pat Lee", email="pat@acme.com", confidence="pattern", find="done"))
    run(sheet, coldstart({})[0])
    assert svc.cell(2, "Draft status") == "drafted"
    notes = svc.cell(2, "Notes").splitlines()
    assert [n.split(": ", 1)[1] for n in notes] == ["pattern guess", "angle: role override"]
