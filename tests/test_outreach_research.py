"""Aria's own research for applied jobs Muse left blank (features/outreach/research.py)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from aria.features.outreach import contacts as finder
from aria.features.outreach import research
from aria.integrations.header_sheet import HeaderSheet
from aria.integrations.sheets import SheetError

COMPANY_HEADERS = ["Job ID", "Company", "Role", "Job URL", "Domain", "Email pattern", "Company problem", "Signal",
                   "Signal source", "Angle", "Blocker", "Outreach status", "Notes"]
CONTACT_HEADERS = ["Job ID", "Company", "Person", "Title", "LinkedIn URL", "Email", "Source", "Confidence",
                   "Find status", "Wave", "Draft status", "Gmail draft link", "Updated", "Added by", "Notes"]
HOUR = 3600.0
POSTING = ("Glean builds a work AI assistant that searches every company app and answers questions with "
           "permissions respected. You will build retrieval pipelines and ranking for enterprise search.")
SITE = ("Glean is the Work AI platform. Glean connects to over 100 enterprise apps so employees find answers. "
        "Glean raised a Series F at a 7.2 billion valuation in 2026 to expand its agents.")


class Grid:
    """spreadsheets().values(): get + append over an in-memory grid."""

    def __init__(self, values):
        self.grid = [list(v) for v in values]

    def spreadsheets(self):
        return self

    def values(self):
        return self

    def get(self, spreadsheetId, range):
        return SimpleNamespace(execute=lambda: {"values": [list(v) for v in self.grid]})

    def append(self, spreadsheetId, range, valueInputOption, insertDataOption, body):
        assert valueInputOption == "RAW"
        self.grid += [list(v) for v in body["values"]]
        n = len(self.grid)
        return SimpleNamespace(execute=lambda: {"updates": {"updatedRange": f"'Tab'!A{n}:O{n}"}})

    def rows(self):
        return [dict(zip(self.grid[0], r + [""] * (len(self.grid[0]) - len(r)))) for r in self.grid[1:]]


@pytest.fixture
def jobs_dir(tmp_path, monkeypatch):
    from aria.features.resume import prepare
    monkeypatch.setattr(prepare, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(research, "SEEN_PATH", tmp_path / "seen.json")
    d = tmp_path / "J0221-glean"
    d.mkdir()
    (d / "posting.txt").write_text(POSTING)
    return tmp_path


def job(jid="J0221", company="Glean", status="Applied"):
    return {"job_id": jid, "company": company, "role": "Software Engineer, New Grad", "status": status,
            "post_link": "https://app.joinhandshake.com/jobs/1", "apply_link": "", "url": "https://app.joinhandshake.com/jobs/1"}


# ---------------------------------------------------------------------------
# which jobs
# ---------------------------------------------------------------------------

def test_a_gap_waits_for_muse_then_is_researched(jobs_dir):
    seen = {}
    assert research.gaps([], [job()], now=1000.0, seen=seen) == []          # first seen: Muse goes first
    assert seen == {"J0221": 1000.0}
    assert research.gaps([], [job()], now=1000.0 + 23 * HOUR, seen=seen) == []
    assert [j["job_id"] for j in research.gaps([], [job()], now=1000.0 + 25 * HOUR, seen=seen)] == ["J0221"]


def test_jobs_muse_has_unapplied_or_unprepared_are_never_gaps(jobs_dir):
    seen = {"J0221": 0.0, "J0300": 0.0}
    muse_row = [{"Job ID": "j0221", "Company": "Glean"}]
    assert research.gaps(muse_row, [job()], now=HOUR * 99, seen=seen) == []
    assert "J0221" not in seen and "J0300" not in seen                      # filled / gone: forgotten
    assert research.gaps([], [job(status="new")], now=HOUR * 99, seen={"J0221": 0.0}) == []
    assert research.gaps([], [job(jid="J0300")], now=HOUR * 99, seen={"J0300": 0.0}) == []   # no job folder


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, why", [
    ("Applicants must be a U.S. citizen.", "U.S. citizenship required"),
    ("U.S. citizenship is required for this role.", "U.S. citizenship required"),
    ("An active Secret clearance is a plus; security clearance required.", "security clearance"),
    ("We are unable to sponsor visas for this position.", "posting says no visa sponsorship"),
    ("We do not offer visa sponsorship.", "posting says no visa sponsorship"),
    ("We sponsor visas and love new grads.", ""),
])
def test_blockers_come_from_the_posting(text, why):
    assert research.blocker(text) == why


SRCS = [research.Source("the job posting", POSTING), research.Source("https://glean.com/about", SITE)]


def test_a_problem_the_sources_dont_support_writes_nothing():
    assert research.grounded({"problem": "Builds rockets for lunar mining crews"}, SRCS) is None
    assert research.grounded({"problem": "unknown"}, SRCS) is None


def test_a_supported_answer_is_kept():
    d = research.grounded({"problem": "Glean helps employees find answers across enterprise apps with AI search.",
                           "signal": "Glean raised a Series F at a 7.2 billion valuation in 2026",
                           "signal_source": "https://glean.com/about",
                           "fact": "Glean connects to enterprise apps so employees find answers"}, SRCS)
    assert d["signal_source"] == "https://glean.com/about" and d["signal"].startswith("Glean raised")
    assert d["fact"].startswith("Glean connects")


@pytest.mark.parametrize("signal, url", [
    ("Glean raised a Series F at a 7.2 billion valuation in 2026", "https://elsewhere.com/x"),   # not a source
    ("Glean raised a Series G at a 9 billion valuation in 2026", "https://glean.com/about"),     # number not there
    ("Glean opened an office", "https://glean.com/about"),                                       # too little overlap
])
def test_an_unsupported_signal_becomes_unknown(signal, url):
    d = research.grounded({"problem": "Glean helps employees find answers across enterprise apps.",
                           "signal": signal, "signal_source": url, "fact": "built a moon base in 1969"}, SRCS)
    assert d["signal"] == "unknown" and d["signal_source"] == "unknown" and d["fact"] == ""


# ---------------------------------------------------------------------------
# research + writing
# ---------------------------------------------------------------------------

def _stub(monkeypatch, answer, people=()):
    calls = []

    def complete_checked(task, messages, validate, system=None, max_tokens=1024):
        calls.append({"task": task, "content": messages[0]["content"], "system": system})
        text = json.dumps(answer)
        return SimpleNamespace(text=text, stop_reason="end_turn"), validate(text)
    monkeypatch.setattr(research.llm_client, "complete_checked", complete_checked)
    monkeypatch.setattr(research.finder, "company_domain", lambda j, s=None: "glean.com")
    monkeypatch.setattr(research, "sources", lambda posting, domain, company, search_fn, client: SRCS)
    monkeypatch.setattr(research.finder, "find_people", lambda *a, **k: list(people))
    monkeypatch.setattr(research.writer, "role_work",
                        lambda b: {"label": "LLM Gateway", "key": "llm gateway", "text": "Routes prompts by difficulty."})
    return calls


GOOD = {"problem": "Glean helps employees find answers across enterprise apps with AI search.",
        "signal": "unknown", "signal_source": "unknown", "fact": "Glean connects to enterprise apps so employees find answers"}


def test_research_writes_one_companies_row_and_its_people(jobs_dir, monkeypatch):
    pat = finder.Person("Pat Lee", "Engineering Manager", "search", "https://www.linkedin.com/in/patlee")
    calls = _stub(monkeypatch, GOOD, people=[pat])
    co, pe = Grid([COMPANY_HEADERS]), Grid([CONTACT_HEADERS])
    out = research.fill_gaps(HeaderSheet("id", "Companies", service=co, writable=research.COMPANY_COLUMNS),
                             HeaderSheet("id", "Contacts", service=pe, writable=research.CONTACT_COLUMNS),
                             jobs_fn=lambda: [job()], now=0.0)
    assert out == []                                    # first sight: Muse's turn
    out = research.fill_gaps(HeaderSheet("id", "Companies", service=co, writable=research.COMPANY_COLUMNS),
                             HeaderSheet("id", "Contacts", service=pe, writable=research.CONTACT_COLUMNS),
                             jobs_fn=lambda: [job()], now=25 * HOUR)
    assert out == ["Glean (J0221): 1 contact(s)"]
    assert calls[0]["task"] == "outreach_research" and "<<<DATA" in calls[0]["content"]
    assert research.untrusted.RULE in calls[0]["system"]
    row = co.rows()[0]
    assert row["Job ID"] == "J0221" and row["Company problem"] == GOOD["problem"]
    assert row["Angle"] == "LLM Gateway: Glean connects to enterprise apps so employees find answers"
    assert row["Outreach status"] == "active" and "researched by Aria" in row["Notes"]
    assert row["Domain"] == "glean.com" and row["Email pattern"] == ""
    person = pe.rows()[0]
    assert person["Person"] == "Pat Lee" and person["Added by"] == "Aria" and person["Find status"] == "pending"
    assert person["LinkedIn URL"] == "https://www.linkedin.com/in/patlee" and person["Wave"] == "1"
    # the next run: the Job ID is on Companies now, nothing is added twice
    again = research.fill_gaps(HeaderSheet("id", "Companies", service=co, writable=research.COMPANY_COLUMNS),
                               HeaderSheet("id", "Contacts", service=pe, writable=research.CONTACT_COLUMNS),
                               jobs_fn=lambda: [job()], now=99 * HOUR)
    assert again == [] and len(co.rows()) == 1 and len(pe.rows()) == 1


def test_nobody_found_is_a_team_inbox_row_and_a_blocker_holds_it(jobs_dir, monkeypatch):
    (jobs_dir / "J0221-glean" / "posting.txt").write_text(POSTING + " We are unable to sponsor visas.")
    _stub(monkeypatch, GOOD)
    co, pe = Grid([COMPANY_HEADERS]), Grid([CONTACT_HEADERS])
    research._save_seen({"J0221": 0.0})
    out = research.fill_gaps(HeaderSheet("id", "Companies", service=co), HeaderSheet("id", "Contacts", service=pe),
                             jobs_fn=lambda: [job()], now=25 * HOUR)
    assert out == ["Glean (J0221): team inbox contact(s), held for review — posting says no visa sponsorship"]
    assert co.rows()[0]["Outreach status"] == "review"
    assert pe.rows()[0]["Person"] == "—"


def test_unsupported_research_writes_nothing(jobs_dir, monkeypatch):
    _stub(monkeypatch, {"problem": "Builds rockets for lunar mining crews", "signal": "unknown",
                        "signal_source": "unknown", "fact": "unknown"})
    co, pe = Grid([COMPANY_HEADERS]), Grid([CONTACT_HEADERS])
    research._save_seen({"J0221": 0.0})
    lines = []
    out = research.fill_gaps(HeaderSheet("id", "Companies", service=co), HeaderSheet("id", "Contacts", service=pe),
                             jobs_fn=lambda: [job()], say=lines.append, now=25 * HOUR)
    assert out == [] and co.rows() == [] and pe.rows() == []
    assert any("left for Muse" in l for l in lines)


def test_append_refuses_columns_the_tab_lacks_or_aria_may_not_write():
    sheet = HeaderSheet("id", "Contacts", service=Grid([CONTACT_HEADERS]), writable=("Person",))
    with pytest.raises(SheetError):
        sheet.append({"Nope": "x"})
    with pytest.raises(SheetError):
        sheet.append({"Email": "x@y.io"})
    assert sheet.append({"Person": "Pat"}) == 2


def test_a_problem_only_the_posting_supports_writes_nothing():
    # "retrieval pipelines and ranking" is the posting's job text, not the company's own words
    assert research.grounded({"problem": "Builds retrieval pipelines and ranking for search."},
                             [research.Source("the job posting", POSTING)]) is None


def test_only_the_posting_to_go_on_writes_nothing(jobs_dir, monkeypatch):
    calls = _stub(monkeypatch, GOOD)
    monkeypatch.setattr(research, "sources", lambda *a: [research.Source("the job posting", POSTING)])
    assert research.research(job()) is None and calls == []


def test_the_site_is_guessed_from_the_name_and_kept_only_when_its_title_names_it():
    import httpx

    def handler(req):
        pages = {"glean.com": "<title>Some Other Glean Co</title>", "glean.ai": "<title>Glean | Work AI</title>"}
        page = {"acmeinc.com": "<title>Parked domain</title>"}.get(req.url.host) or pages.get(req.url.host)
        return httpx.Response(200, text=page or "", headers={"content-type": "text/html"}) if page else httpx.Response(404)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert research.guess_domain("Glean", client) == "glean.com"        # title names it ("someotherglean co")
    assert research.guess_domain("Acme Inc.", client) == ""            # parked: the title doesn't name Acme
    assert research.guess_domain("AB", client) == ""


def test_a_js_site_still_gives_its_description_and_parked_or_unnamed_pages_are_refused():
    import httpx
    html = ('<html><head><meta name="description" content="Palantir builds AI platforms for defense and industry.">'
            '<meta property="og:site_name" content="Palantir"></head><body><div id="root"></div></body></html>')
    assert research.meta_text(html) == "Palantir builds AI platforms for defense and industry.. Palantir"

    def handler(req):
        pages = {"virtusa.com": '<title>Making experiences better</title><div class="copyrightCol"><p>©<span>2026</span> <span>Virtusa Corp. All rights reserved.</span></p></div>',
                 "namo.com": "<title>namo.com</title><p>This domain may be for sale.</p>"}
        page = pages.get(req.url.host)
        return httpx.Response(200, text=page, headers={"content-type": "text/html"}) if page else httpx.Response(404)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert research.guess_domain("Virtusa Corporation", client) == "virtusa.com"
    assert research.guess_domain("NAMO Technologies LLC", client) == ""


def test_the_angle_without_a_company_fact_is_the_first_point_of_the_work(jobs_dir, monkeypatch):
    _stub(monkeypatch, dict(GOOD, fact="unknown"))
    monkeypatch.setattr(research.writer, "role_work", lambda b: {
        "label": "Software Engineer Intern at Appetit", "key": "appetit",
        "text": "• Designed the loyalty tables with ACID transactions\n• Shipped loyalty features"})
    r = research.research(job())
    assert r.angle == "Appetit: Designed the loyalty tables with ACID transactions"


def test_a_fact_in_the_postings_own_words_is_dropped():
    d = research.grounded({"problem": "Glean helps employees find answers across enterprise apps.",
                           "fact": "Glean builds retrieval pipelines and ranking for enterprise search"}, SRCS)
    assert d["fact"] == ""
