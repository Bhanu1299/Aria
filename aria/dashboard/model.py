"""dashboard/model.py — pure: records from every source → jobs with a stage + evidence.

Matching: apply link (normalised), else company + role. Gmail threads and Coldstart
contacts attach by company domain (an address in the job's sheet cells, or a domain
whose name matches the company); among several jobs at one company the subject's role
words decide. Anything that matches no job is kept in `unmatched`, never dropped.

A job's stage is the furthest one with evidence; each piece of evidence records its source.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlparse

STAGES = ["Found by Muse", "Skipped / Applying", "Form filled (in review)", "Applied", "Contact found",
          "Email found", "Draft ready", "Sent", "Replied", "Interview / Rejected / Offer"]
FOUND, APPLYING, FILLED, APPLIED, CONTACT, EMAIL, DRAFT, SENT, REPLIED, OUTCOME = range(len(STAGES))

DRAFT_STALE_DAYS = 2
FOLLOW_UP_DAYS = 7
NEEDS_ME_WINDOW_DAYS = 60   # older drafts / unanswered sends aren't actionable any more

_EMAIL = re.compile(r"[A-Za-z0-9._%+'-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_FREEMAIL = {"gmail.com", "googlemail.com", "yahoo.com", "outlook.com", "hotmail.com", "live.com",
             "icloud.com", "me.com", "aol.com", "proton.me", "protonmail.com", "ymail.com"}
_AUTOMATED = re.compile(r"amazonses\.com$|user-subscription\.com$|[@.]bounces?\.|mailer-daemon|"
                        r"^no-?reply@|^notifications?@|[@.]inbound\.", re.I)
_REPLY_SUBJECT = re.compile(r"(?i)^\s*(re|fwd?|fw)\s*:")
_SUFFIXES = re.compile(r"\b(inc|llc|ltd|corp|corporation|co|company|technologies|technology|labs|group|"
                       r"holdings|the)\b\.?", re.I)
_NO_CONTACT = re.compile(r"(?i)^(via |no |none|n/?a|-|tbd|apply|ats|email draft|draft|gmail draft|inferred)")
_CONTACT_PREFIX = re.compile(r"(?i)^(linkedin|email|contact|recruiter)\s*:\s*")
# Written by aria/features/outreach: "LinkedIn (Aria): Name (Title) — profile\nNote: …"
_ARIA_NOTE = re.compile(r"(?is)^linkedin \(aria\):\s*(?P<who>.*?)\s*\nnote:\s*(?P<note>.+)$")
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep",
                                       "oct", "nov", "dec"], start=1)}


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------

def norm_link(url: str) -> str:
    url = (url or "").strip()
    if not re.match(r"(?i)^https?://", url):
        return ""
    u = urlparse(url)
    host = (u.hostname or "").lower()
    host = host[4:] if host.startswith("www.") else host
    return f"{host}{u.path.rstrip('/')}".lower()


def norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def company_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _SUFFIXES.sub(" ", (name or "").lower()))


def domain_matches_company(domain: str, company: str) -> bool:
    slug = company_slug(company)
    label = re.sub(r"[^a-z0-9]", "", (domain or "").lower().split(".")[0])
    if len(slug) < 2 or len(label) < 2:
        return False
    return slug == label or (len(label) >= 4 and (slug.startswith(label) or label.startswith(slug)))


def is_company_address(addr: str) -> bool:
    dom = addr.rsplit("@", 1)[-1].lower()
    return bool(dom) and dom not in _FREEMAIL and not dom.endswith(".edu") and not _AUTOMATED.search(addr)


def emails_in(*texts: str) -> list[str]:
    seen = []
    for t in texts:
        for e in _EMAIL.findall(t or ""):
            e = e.lower().strip(".")
            if e not in seen:
                seen.append(e)
    return seen


def contact_name(cell: str) -> str:
    """'LinkedIn: Pat Lee (Recruiter) — linkedin.com/…' → 'Pat Lee'; '' for notes that name nobody."""
    text = _CONTACT_PREFIX.sub("", (cell or "").strip())
    if not text or _NO_CONTACT.match(text) or _EMAIL.fullmatch(text):
        return ""
    name = re.split(r"\s+[—–|]\s+|\(|,|\s+-\s+", text)[0].strip()
    name = _EMAIL.sub("", name).strip()
    return name[:60] if re.search(r"[A-Za-z]", name) and len(name.split()) <= 6 else ""


def parse_found_date(text: str, today: datetime) -> Optional[datetime]:
    """'HN Who is hiring Oct 20' / '2026-10-20' → date; None when there isn't one."""
    m = re.search(r"\b(20\d\d)-(\d\d)-(\d\d)\b", text or "")
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? (\d{1,2})\b", (text or "").lower())
    if m:
        d = datetime(today.year, _MONTHS[m.group(1)], int(m.group(2)))
        return d.replace(year=today.year - 1) if d > today + timedelta(days=1) else d
    return None


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    stage: int
    source: str
    detail: str = ""


@dataclass
class Job:
    company: str
    role: str
    link: str = ""            # posting (post link, else apply link)
    apply_key: str = ""
    fit: str = ""
    status: str = ""
    outcome: str = ""
    contact: str = ""
    contact_title: str = ""
    channels: set = field(default_factory=set)       # "email" / "linkedin"
    email: str = ""
    email_confidence: Optional[float] = None
    email_status: str = ""
    linkedin_note: str = ""   # a note Aria wrote, ready to send by hand
    linkedin_to: str = ""
    found_date: Optional[datetime] = None
    deadline: Optional[datetime] = None
    last_update: Optional[datetime] = None
    cost: float = 0.0
    sheet_refs: list = field(default_factory=list)
    domains: set = field(default_factory=set)
    runs: list = field(default_factory=list)
    threads: list = field(default_factory=list)
    drafts: list = field(default_factory=list)
    contacts: list = field(default_factory=list)
    applied_on: list = field(default_factory=list)   # dates from aria.db (submitted runs, applications)
    evidence: list = field(default_factory=list)
    stage: int = FOUND
    stage_label: str = STAGES[FOUND]

    @property
    def key(self) -> tuple:
        return (norm_text(self.company), norm_text(self.role))

    def add(self, stage: int, source: str, detail: str = "") -> None:
        self.evidence.append(Evidence(stage, source, detail))

    def touch(self, when: Optional[datetime]) -> None:
        if when and (self.last_update is None or when > self.last_update):
            self.last_update = when


def _ts(value) -> Optional[datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value)
    try:
        return datetime.fromisoformat(str(value).replace("Z", ""))
    except ValueError:
        return None


class _Index:
    def __init__(self) -> None:
        self.jobs: list[Job] = []
        self.by_link: dict[str, Job] = {}
        self.by_key: dict[tuple, Job] = {}

    def find(self, link: str = "", company: str = "", role: str = "") -> Optional[Job]:
        k = norm_link(link)
        if k and k in self.by_link:
            return self.by_link[k]
        if company and role:
            return self.by_key.get((norm_text(company), norm_text(role)))
        return None

    def add(self, job: Job) -> Job:
        self.jobs.append(job)
        self.register(job)
        return job

    def register(self, job: Job) -> None:
        if job.apply_key:
            self.by_link.setdefault(job.apply_key, job)
        if job.company and job.role:
            self.by_key.setdefault(job.key, job)

    def at_domain(self, domain: str) -> list[Job]:
        return [j for j in self.jobs if domain in j.domains or domain_matches_company(domain, j.company)]


def _sheet_job(idx: _Index, row: dict, tab: str, today: datetime) -> Job:
    company, role = row.get("Company", ""), row.get("Role", "")
    apply_link, post_link = row.get("Apply link", ""), row.get("Post link", "")
    job = idx.find(apply_link, company, role) or idx.find(post_link)
    if job is None:
        job = idx.add(Job(company=company, role=role))
    job.apply_key = job.apply_key or norm_link(apply_link)
    job.link = job.link or (post_link if norm_link(post_link) else "") or (apply_link if norm_link(apply_link) else "")
    job.fit = job.fit or row.get("Profile alignment /5", "")
    status = row.get("Status", "")
    if tab == "applied" or not job.status:
        job.status = status or job.status
    job.outcome = row.get("Outcome", "") or job.outcome
    ref = f"{tab} row {row.get('_row')}"
    job.sheet_refs.append(ref)
    job.add(FOUND, f"Muse sheet ({ref})")

    who = row.get("Who to contact", "")
    text_cells = [who, row.get("Outreach", ""), row.get("Outreach_2", ""), row.get("Channel", "")]
    for e in emails_in(*text_cells):
        if is_company_address(e):
            job.domains.add(e.split("@")[1])
            if not job.email:
                job.email = e
            job.add(EMAIL, f"Muse sheet ({ref})", e)
    person = contact_name(who)
    if person:
        job.contact = job.contact or person
        title = re.search(r"\(([^)]{2,80})\)", _CONTACT_PREFIX.sub("", who))
        if title and not job.contact_title:
            job.contact_title = title.group(1).split(",")[0].strip()
        job.add(CONTACT, f"Muse sheet ({ref}) Who to contact", who[:80])
    text_all = " ".join(text_cells).lower()
    if "linkedin" in text_all:
        job.channels.add("linkedin")
    if "email" in text_all or job.email:
        job.channels.add("email")
    note = _ARIA_NOTE.match(row.get("Outreach_2", "").strip())
    if note:
        job.linkedin_note, job.linkedin_to = note.group("note").strip(), note.group("who").strip()
    if re.search(r"(?i)draft ready", " ".join(text_cells)):
        job.add(DRAFT, f"Muse sheet ({ref})", "says a draft is ready")

    s = status.lower()
    if s in ("not gonna apply", "skipped", "not applying"):
        job.add(APPLYING, f"Muse sheet ({ref}) Status", status)
    elif s in ("in review", "needs human"):
        job.add(FILLED if s == "in review" else APPLYING, f"Muse sheet ({ref}) Status", status)
    elif s.startswith("applied"):
        job.add(APPLIED, f"Muse sheet ({ref}) Status", status)
    outcome_text = f"{status} {row.get('Outcome', '')}".lower()
    if re.search(r"reject|interview|offer", outcome_text) or (tab == "applied" and row.get("Outcome")):
        job.add(OUTCOME, f"Muse sheet ({ref}) Status/Outcome", (row.get("Outcome") or status)[:80])
    dl_col = next((k for k in row if "deadline" in k.lower() and row.get(k)), "")
    if dl_col and not job.deadline:
        job.deadline = parse_found_date(row[dl_col], today)
    found = parse_found_date(row.get("Source", "") + " " + row.get("Updates", ""), today)
    if found and (job.found_date is None or found < job.found_date):
        job.found_date = found
    job.touch(found)
    idx.register(job)
    return job


def build(sources: dict, now: Optional[datetime] = None) -> dict:
    """sources: {name: Source} from sources.load_all(). Missing/failed sources are skipped."""
    now = now or datetime.now()
    idx = _Index()
    unmatched: list[dict] = []

    def data(name: str):
        src = sources.get(name)
        return src.data if src is not None and src.ok else None

    sheet = data("sheet") or {}
    for tab in ("current", "applied"):
        for row in sheet.get(tab, []):
            if row.get("Company") or row.get("Role"):
                _sheet_job(idx, row, tab, now)

    db = data("aria_db") or {}
    reviews = data("reviews") or {}
    for run in db.get("runs", []):
        job = idx.find(run.get("url", ""), run.get("company", ""), run.get("role", ""))
        if job is None:
            unmatched.append({"source": "aria.db application_runs", "what": f"{run.get('company')} — {run.get('role')}",
                              "detail": f"{run.get('stage')} {run.get('created_at')}"})
            continue
        job.runs.append(run)
        job.cost += float(run.get("cost_usd") or 0)
        when = _ts(run.get("created_at"))
        job.touch(when)
        st = run.get("stage", "")
        if st == "submitted" and when:
            job.applied_on.append(when)
        src = f"aria.db application_runs #{run.get('id')}"
        job.add({"filled": FILLED, "submitted": APPLIED}.get(st, APPLYING), src, st)
    for app in db.get("applications", []):
        job = idx.find(app.get("url", ""), app.get("company", ""), app.get("role", ""))
        if job is None:
            unmatched.append({"source": "aria.db applications", "what": f"{app.get('company')} — {app.get('role')}",
                              "detail": f"{app.get('platform')} {app.get('applied_at')}"})
            continue
        job.add(APPLIED, f"aria.db applications #{app.get('id')}", app.get("status", ""))
        when = _ts(app.get("applied_at"))
        if when:
            job.applied_on.append(when)
        job.touch(when)

    cs = data("coldstart") or {}
    for c in cs.get("contacts", []):
        jobs = idx.at_domain(c["domain"])
        if not jobs:
            continue
        for job in jobs:
            job.contacts.append(c)
            job.domains.add(c["domain"])
            job.add(CONTACT, "Coldstart contact cache", c.get("name") or c["email"])
            job.add(EMAIL, "Coldstart contact cache", f"{c['email']} ({c.get('status')}, {c.get('confidence')})")
            if not job.contact:
                job.contact = c.get("name") or ""
                job.contact_title = c.get("title") or job.contact_title
            if job.email_confidence is None or (c.get("confidence") or 0) > job.email_confidence:
                job.email, job.email_confidence, job.email_status = c["email"], c.get("confidence"), c.get("status") or ""

    gmail = data("gmail") or {}
    me = gmail.get("me", "")
    threads_out = []
    for t in gmail.get("threads", []):
        info = _thread_info(t, me)
        if info is None:
            continue
        threads_out.append(info)
        job = _pick_job(idx, info["domain"], info["subject"])
        if job is None:
            unmatched.append({"source": "Gmail sent", "what": f"{info['to']} — {info['subject'][:70]}",
                              "detail": info["sent"].strftime("%Y-%m-%d"), "thread_id": info["id"]})
            continue
        info["job"] = job
        job.threads.append(info)
        job.channels.add("email")
        job.domains.add(info["domain"])
        if not job.email:
            job.email = info["to"]
        job.add(EMAIL, "Gmail sent (address used)", info["to"])
        job.add(SENT, "Gmail sent", f"{info['subject'][:60]} — {info['sent']:%Y-%m-%d}")
        job.touch(info["sent"])
        if info["replied"]:
            job.add(REPLIED, "Gmail reply", f"{info['replied']:%Y-%m-%d}")
            job.touch(info["replied"])
        if info["bounced"]:
            job.add(EMAIL, "Gmail bounce", f"{info['to']} bounced {info['bounced']:%Y-%m-%d}")
            job.touch(info["bounced"])

    drafts_out = []
    for d in gmail.get("drafts", []):
        to = next((a for a in d.get("to", []) if is_company_address(a)), "")
        info = {"id": d.get("id"), "thread_id": d.get("thread_id"), "to": to, "subject": d.get("subject", ""),
                "created": datetime.fromtimestamp(d.get("ts") or 0)}
        drafts_out.append(info)
        job = _pick_job(idx, to.split("@")[1], info["subject"]) if to else None
        if job is None:
            unmatched.append({"source": "Gmail draft", "what": f"{to or '(no recipient)'} — {info['subject'][:70]}",
                              "detail": f"{info['created']:%Y-%m-%d}", "thread_id": info["thread_id"]})
            continue
        info["job"] = job
        job.drafts.append(info)
        job.add(DRAFT, "Gmail draft", f"{info['subject'][:60]} — {info['created']:%Y-%m-%d}")
        job.touch(info["created"])

    for job in idx.jobs:
        _settle(job)

    has_deadline_col = any("deadline" in k.lower() for tab in ("current", "applied")
                           for row in sheet.get(tab, [])[:1] for k in row)
    return {"jobs": idx.jobs, "unmatched": unmatched, "threads": threads_out, "drafts": drafts_out,
            "reviews": reviews, "now": now, "sources": sources, "has_deadline_col": has_deadline_col,
            "coldstart": cs}


def _thread_info(t: dict, me: str) -> Optional[dict]:
    """Outreach thread: its first message is mine, not a reply, to a company address."""
    msgs = t.get("messages", [])
    if not msgs or not msgs[0]["from_me"] or msgs[0]["is_reply"] or _REPLY_SUBJECT.match(msgs[0]["subject"]):
        return None
    to = next((a for a in msgs[0]["to"] if a != me and is_company_address(a)), "")
    if not to:
        return None
    replied = next((datetime.fromtimestamp(m["ts"]) for m in msgs[1:]
                    if not m["from_me"] and not m["bounce"] and m["from"]
                    and m["from"].split("@")[-1] == to.split("@")[1]), None)
    bounced = next((datetime.fromtimestamp(m["ts"]) for m in msgs[1:] if m["bounce"]), None)
    last = msgs[-1]
    awaiting_me = bool(replied) and not last["from_me"] and not last["bounce"]
    return {"id": t["id"], "to": to, "domain": to.split("@")[1], "subject": msgs[0]["subject"],
            "sent": datetime.fromtimestamp(msgs[0]["ts"]), "replied": replied, "bounced": bounced,
            "awaiting_me": awaiting_me, "last": datetime.fromtimestamp(last["ts"]),
            "reply_subjects": [m["subject"] for m in msgs[1:] if not m["from_me"] and not m["bounce"]]}


ROLE_MATCH = 0.6


def _role_score(subject_role: str, role: str) -> float:
    want = set(norm_text(role).split())
    return len(want & set(norm_text(subject_role).split())) / len(want) if want else 0.0


def _pick_job(idx: _Index, domain: str, subject: str) -> Optional[Job]:
    """Job at this domain's company. A subject that names a role ("<Role> — Name") must
    match the job's role; otherwise a lone job at the company takes it."""
    jobs = idx.at_domain(domain)
    if not jobs:
        return None
    parts = re.split(r"\s+[—–]\s+", subject or "", maxsplit=1)
    if len(parts) == 2:
        best = max(jobs, key=lambda j: _role_score(parts[0], j.role))
        return best if _role_score(parts[0], best.role) >= ROLE_MATCH else None
    if len(jobs) == 1:
        return jobs[0]
    return max(jobs, key=lambda j: _role_score(subject, j.role))


def _settle(job: Job) -> None:
    stage = max((e.stage for e in job.evidence), default=FOUND)
    job.stage = stage
    if stage == APPLYING:
        skipped = job.status.lower() in ("not gonna apply", "skipped", "not applying") or \
            any(r.get("stage") == "skipped" for r in job.runs)
        job.stage_label = "Skipped" if skipped else "Applying"
    elif stage == OUTCOME:
        text = f"{job.status} {job.outcome}".lower()
        job.stage_label = ("Offer" if "offer" in text else "Interview" if "interview" in text else "Rejected")
    else:
        job.stage_label = STAGES[stage]


# ---------------------------------------------------------------------------
# page sections
# ---------------------------------------------------------------------------

def funnel(jobs: list[Job]) -> list[dict]:
    """Per stage: jobs that reached it (stage ≥ i), jobs currently at it, drop from the previous."""
    rows, prev = [], None
    for i, name in enumerate(STAGES):
        reached = sum(1 for j in jobs if j.stage >= i)
        at = [j for j in jobs if j.stage == i]
        row = {"stage": name, "reached": reached, "at": len(at), "drop": (prev - reached) if prev is not None else None,
               "sources": sorted({e.source.split(" (")[0].split(" #")[0] for j in jobs for e in j.evidence if e.stage == i})}
        if i == APPLYING:
            row["skipped"] = sum(1 for j in at if j.stage_label == "Skipped")
        rows.append(row)
        prev = reached
    return rows


def needs_me(model: dict) -> dict:
    now = model["now"]
    out = {"fields": [], "needs_human": [], "stale_drafts": [], "follow_ups": [], "bounces": []}
    for job in model["jobs"]:
        if job.runs:
            last = job.runs[-1]
            if last.get("stage") == "filled" and last.get("needs_me"):
                review = model["reviews"].get(last.get("review_path", "")) or {}
                fields = [r["field"] for r in review.get("sections", {}).get("needs_me", [])]
                out["fields"].append({"job": job, "count": int(last["needs_me"]), "fields": fields,
                                      "review": last.get("review_path", "")})
        if job.status.lower() == "needs human":
            out["needs_human"].append({"job": job, "where": ", ".join(job.sheet_refs)})
    window = timedelta(days=NEEDS_ME_WINDOW_DAYS)
    for d in model["drafts"]:
        if timedelta(days=DRAFT_STALE_DAYS) < now - d["created"] <= window:
            out["stale_drafts"].append(d)
    for t in model["threads"]:
        if t["bounced"]:
            if now - t["bounced"] <= window:
                out["bounces"].append(t)
        elif not t["replied"] and timedelta(days=FOLLOW_UP_DAYS) < now - t["sent"] <= window:
            out["follow_ups"].append(t)
    out["follow_ups"].sort(key=lambda t: t["sent"])
    return out


def week_start(now: datetime) -> datetime:
    return (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)


def this_week(model: dict) -> dict:
    start = week_start(model["now"])
    jobs = model["jobs"]
    sent = [t for t in model["threads"] if t["sent"] >= start]
    replies = [t for t in model["threads"] if t["replied"] and t["replied"] >= start]
    all_sent = model["threads"]
    return {
        "start": start,
        "found": sum(1 for j in jobs if j.found_date and j.found_date >= start),
        "found_undated": sum(1 for j in jobs if not j.found_date),
        "applied": sum(1 for j in jobs if any(d >= start for d in j.applied_on)),
        "sent": len(sent),
        "replies": len(replies),
        "reply_rate_week": (sum(1 for t in sent if t["replied"]) / len(sent)) if sent else None,
        "reply_rate_all": (sum(1 for t in all_sent if t["replied"]) / len(all_sent)) if all_sent else None,
        "sent_all": len(all_sent),
    }


def cost(model: dict) -> dict:
    db = model["sources"].get("aria_db")
    runs = (db.data or {}).get("runs", []) if db is not None and db.ok else []
    total = sum(float(r.get("cost_usd") or 0) for r in runs)
    # Jobs with direct evidence of an application (not merely a later outreach stage).
    applied = [j for j in model["jobs"] if any(e.stage == APPLIED for e in j.evidence)]
    with_runs = [j for j in applied if j.runs]
    return {"total": total, "runs": len(runs), "applied": len(applied),
            "per_applied": (total / len(applied)) if applied else None,
            "per_applied_by_aria": (sum(j.cost for j in with_runs) / len(with_runs)) if with_runs else None,
            "applied_by_aria": len(with_runs)}


# ---------------------------------------------------------------------------
# per-job facts the Applications / Today tabs share
# ---------------------------------------------------------------------------

CLOSED_WORDS = re.compile(r"(?i)reject|offer|not gonna apply|skipped|withdrawn|closed")


def applied_date(job: Job) -> Optional[datetime]:
    return min(job.applied_on) if job.applied_on else None


def has_applied(job: Job) -> bool:
    return any(e.stage == APPLIED for e in job.evidence)


def is_closed(job: Job) -> bool:
    return job.stage_label in ("Rejected", "Offer", "Skipped") or bool(CLOSED_WORDS.search(job.status))


def response(job: Job) -> str:
    if any(t["replied"] for t in job.threads):
        return "replied " + max(t["replied"] for t in job.threads if t["replied"]).strftime("%Y-%m-%d")
    if any(t["bounced"] for t in job.threads):
        return "bounced"
    if job.stage == OUTCOME:
        return job.stage_label.lower()
    return ""


def next_action(job: Job, now: datetime, outreach_elsewhere: str = "") -> tuple:
    """(action, bucket) — bucket: needs_action | waiting | no_contact | replied | closed | todo."""
    if is_closed(job):
        return "", "closed"
    if any(t["awaiting_me"] for t in job.threads):
        return "Answer their reply", "replied"
    if any(t["bounced"] for t in job.threads) and not any(t["replied"] for t in job.threads):
        return "Fix the bounced email", "needs_action"
    if any(t["replied"] for t in job.threads):
        return "", "replied"
    if job.threads:
        last = max(t["sent"] for t in job.threads)
        if now - last > timedelta(days=FOLLOW_UP_DAYS):
            return "Follow up", "needs_action"
        return "Wait for a reply", "waiting"
    if job.status.lower() == "needs human":
        return "Finish the application", "needs_action"
    if job.stage_label == "Form filled (in review)":
        return "Review and submit", "needs_action"
    if not has_applied(job) and job.stage < APPLIED:
        return "Apply", "todo"
    if job.drafts:
        return "Send the draft", "needs_action"
    if outreach_elsewhere:
        return f"Outreach already sent for {outreach_elsewhere}", "waiting"
    if job.linkedin_note or ("linkedin" in job.channels and not job.email):   # a LinkedIn note is planned
        return "Send the LinkedIn note", "needs_action"
    if not job.contact and not job.email:
        return "Find a contact", "no_contact"
    if job.email:
        return "Draft outreach", "needs_action"
    return "Find their email", "needs_action"
