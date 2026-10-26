"""features/outreach/writer.py — the outreach text. The LLM writes the lines; code does the rest.

    email(brief, to) → {"to", "subject", "body"} | None          task outreach_email
    linkedin_note(target) → str | None                         task outreach_linkedin_note

The email follows the user's spec (v2). Muse researches each company into the Contacts
tab; Aria never looks anything up. A Brief carries what the email may talk about:

    Company problem   what they're solving        REQUIRED
    Angle             what the candidate built that overlaps with it, and what was hard   REQUIRED
    Signal            something recent (a launch, a post, a hire)   optional
    Person type       generic inbox | hiring (recruiter / manager) | engineer

The email draws on ONE piece of the candidate's work (plan → Plan), from the answer
bank's Experience and Projects lists. The role's pick (role_work) is code: voice → Aria,
ML / data → AppsTek, AI / LLM / agents → LLM Gateway, frontend / full-stack / product →
Draw.AI, anything else (backend, general SWE) → Appetit. Muse's Angle usually names a
project ("Aria: …"); the user's rule when the two differ:

    Muse wins only when its Angle ties the project to a CONCRETE fact about the company
    (CarMax runs voice agents → Aria). An analogy ("plan-act-verify is like flight
    software") → the role wins, Muse's Angle is left out of the email, and the Contacts
    row's Notes get "angle: role override". Rows are never held for this.

Concrete vs analogy is judged by the small tier (task outreach_angle_judge); a "concrete"
verdict counts only when the fact it quotes is in Muse's research (Problem / Signal /
Signal source). The LLM writing the email sees only the chosen entry, never the resume.

missing(brief) names a required input that is blank or "unknown": no email then (the
caller writes "needs research"). The LLM writes the subject and three lines — they
understand the problem, the Angle, why it pulls the candidate in. Code adds the greeting
("Hi {first}," / "Hi {company} team,"), the fixed "I applied… Resume attached." ask, the
sign-off, and drops output that fails a check:

    60-100 words (greeting to ask), three lines, no greeting / sign-off / placeholder of
    its own, subject under 45 chars without "application", no banned word, no number
    unless it's in the company's own name / problem / signal (a "911", never a metric),
    no 6-word run copied from the problem or signal (no restated marketing copy), line 1
    shares 2+ words with the problem, the signal or the Angle's company side (its words
    that aren't the project's) and line 2 with the Angle (it can't go to another company).
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

from aria.features import answer_bank
from aria.llm import llm_client

logger = logging.getLogger(__name__)

NOTE_LIMIT = 300                 # LinkedIn's connection-note cap
EMAIL_WORDS = (60, 100)          # greeting to the ask, signature not counted
SUBJECT_CHARS = 50
GENERIC, HIRING, ENGINEER = "generic inbox", "hiring", "engineer"
_PROFILE_KEYS = ("preferred_name", "first_name", "last_name", "school", "degree", "grad_date",
                 "years_experience", "current_company", "location", "linkedin", "github")
_IDENTITY_KEYS = ("summary", "skills", "experience", "education", "target_roles", "projects")
RESUME_CHARS = 5000             # resume text the writer sees (the facts it may use)
_PLACEHOLDER = re.compile(r"\[[^\]]*\]|\{[^}]*\}|<[^>]+>|\bTODO\b|lorem ipsum", re.I)
_GREETING = re.compile(r"(?i)^\s*(hi|hello|hey|dear)\b")
_SIGNOFF = re.compile(r"(?im)^\s*(best|thanks|thank you|regards|cheers|sincerely)[,!.]?\s*$")
_BANNED = re.compile(
    r"(?i)hope this (?:email |message )?finds you|passion|excit|reaching out|strong fit|leverag"
    r"|innovat|revolutioni[sz]|great addition|amazing|impressive|impressed")
# Subjects that read like a pitch (user, 2026-10-26: "feels like I'm selling something").
_SALESY_SUBJECT = re.compile(
    r"(?i)\b(?:solution|opportunit|proposal|partner|help(?:ing)? you|value|boost|streamlin|unlock|transform|"
    r"grow(?:th)?|scal(?:e|ing) your|improv\w* your|reduc\w+|optimi[sz]\w*|roi|synerg|offer|demo|"
    r"collaborat|introduc)|^\s*(?:re|fwd?)\s*:|[:!]")


def subject_problem(subject: str) -> str:
    """Why `subject` reads like a sales email, or "" when it's fine."""
    m = _SALESY_SUBJECT.search(subject or "")
    if m:
        return f"salesy ({m.group(0).strip()!r})"
    words_ = [w for w in re.findall(r"[A-Za-z][A-Za-z']*", subject or "") if len(w) > 3]
    if len(words_) >= 3 and all(w[0].isupper() for w in words_):
        return "title case"
    return ""


_HIRING_TITLE = re.compile(r"(?i)recruit|talent|hiring|people|\bhr\b|human resources|manager|head of|director|vp\b|"
                           r"president|chief|\bc[eitfo]o\b")
_NUMBER = re.compile(r"\d[\d,.]*")
_WORD = re.compile(r"[a-z0-9][a-z0-9'-]{3,}")
_STOP = {"that", "with", "from", "they", "their", "them", "this", "these", "into", "about", "while",
         "which", "where", "when", "have", "help", "helps", "more", "than", "over", "every", "each",
         "built", "build", "builds", "company", "platform", "teams", "using", "used", "what", "your"}

_EMAIL_SYSTEM = (
    "You write a short cold email from an engineer who just applied for a role, to someone at "
    "that company. Write like one engineer to another: short sentences, contractions fine, "
    "curious not flattering, specific not impressive. Output exactly:\n"
    "SUBJECT: <under 50 characters, lowercase. It reads like a note from a curious person, not a "
    "pitch: warm and a little intriguing — a genuine question about their work, a small shared-"
    "interest hook, or a hint of the candidate's own work that connects to theirs (e.g. "
    "\"curious how you triage 911 calls\", \"i built a tiny cousin of your planner\", \"a fellow "
    "agent tinkerer says hi\"). Never a problem statement or value pitch (\"reducing X\", "
    "\"scaling your Y\"), never sales words (solution, opportunity, proposal, partner, help you, "
    "value), no \"re:\", no colon, no exclamation mark, never the word application.>\n\n"
    "<line 1>\n\n<line 2>\n\n<line 3>\n\n"
    "Line 1 (one sentence): show you understand what they're solving, in your own words. Use the Signal if it "
    "fits. Never restate their marketing copy. If the recipient is a recruiter or manager, talk "
    "about the team's problem, not the company mission.\n"
    "Line 2 (one or two sentences): the Angle (or, when there's none, the candidate's work given below) — what the "
    "candidate built and the specific part that overlaps with their problem. Concrete and "
    "honest: what it does, what was hard.\n"
    "Line 3 (one sentence): one genuine sentence on why this problem pulls the candidate in, tied to the "
    "candidate's work given below, not passion.\n\n"
    "Lines 1-3 together are 30-60 words; count them, and cut rather than go over. The greeting, \"I applied for the role. Resume "
    "attached.\", the ask and the signature are added for you: never write them. Never claim "
    "experience beyond the Angle and the work given. Never bring up other projects. No numbers "
    "or metrics. Never: hope this finds you well, "
    "passionate, excited, reaching out, strong fit, leverage, innovative, revolutionize, great "
    "addition, amazing, impressive. If it could go to another company by changing the name, "
    "rewrite it."
)
_NOTE_SYSTEM = (
    "You write a LinkedIn connection note from a job candidate to someone at a company. Plain "
    "text, one or two sentences, at most 230 characters. Name the role, one concrete reason the "
    "candidate fits from the profile only, and a light ask. No greeting (it's added for you), no "
    "sign-off, no hashtags, no emoji, no placeholders, never invent facts."
)


@dataclass
class Brief:
    """One email's inputs, from a Contacts row (Muse's research columns)."""
    company: str
    role: str
    problem: str
    angle: str
    signal: str = ""
    person: str = ""            # the recipient's name; "" for a generic inbox
    title: str = ""
    person_type: str = ""       # GENERIC | HIRING | ENGINEER ("" → from person/title)
    signal_source: str = ""

    @property
    def kind(self) -> str:
        return person_type(self.person, self.title, self.person_type)


def person_type(person: str, title: str = "", given: str = "") -> str:
    """The sheet's Person type when set, else: no name → generic inbox; a founder →
    engineer; a recruiter / manager / head-of / executive title → hiring; anyone else → engineer."""
    g = (given or "").strip().lower()
    if g:
        return GENERIC if "generic" in g or "inbox" in g else HIRING if _HIRING_TITLE.search(g) or "big" in g else ENGINEER
    if not (person or "").strip() or person.strip() in ("—", "–", "-"):
        return GENERIC
    if re.search(r"(?i)founder", title or ""):     # a startup founder: engineer to engineer
        return ENGINEER
    return HIRING if _HIRING_TITLE.search(title or "") else ENGINEER


def _blank(value: str) -> bool:
    return (value or "").strip().lower() in ("", "unknown", "n/a", "na", "?", "—", "-", "tbd")


def missing(brief: Brief) -> list:
    """The required inputs that are blank or "unknown"."""
    return [name for name, v in (("Company problem", brief.problem), ("Angle", brief.angle)) if _blank(v)]


def _identity() -> dict:
    from aria.features.ats.planner import _identity as identity
    return identity()


def _resume() -> str:
    """The resume's text (the same one form answers are written from), trimmed. '' if none."""
    try:
        from aria.features.ats import planner, resume_text
        return resume_text.text(planner.resume_path())[:RESUME_CHARS]
    except Exception as exc:
        logger.info("outreach: no resume text: %s", exc)
        return ""


def sender_name(bank: Optional[dict] = None) -> str:
    a = (bank or answer_bank.load())["answers"]
    first = a.get("preferred_name") or a.get("first_name") or ""
    return " ".join(x for x in (first, a.get("last_name") or "") if x).strip()


def signature(bank: Optional[dict] = None) -> str:
    """'Thanks,' / first name / 'linkedin.com/in/x | github.com/y' (whichever the bank has)."""
    a = (bank or answer_bank.load())["answers"]
    me = first_name(a.get("preferred_name") or a.get("first_name") or "")
    links = " | ".join(re.sub(r"^https?://(www\.)?", "", str(a[k])).rstrip("/")
                       for k in ("linkedin", "github") if a.get(k) and not str(a[k]).startswith("<"))
    return "\n".join(x for x in ("Thanks,", me if me != "there" else "", links) if x)


def outreach_profile(target) -> str:
    """Facts the LinkedIn note may use: today's date, education/experience basics, identity
    summary, the resume, fit notes."""
    bank = answer_bank.load()
    lines = [f"today: {time.strftime('%Y-%m-%d')}"]
    lines += [f"{k}: {bank['answers'][k]}" for k in _PROFILE_KEYS
              if bank["answers"].get(k) not in (None, "") and not str(bank["answers"][k]).startswith("<")]
    ident = _identity()
    for k in _IDENTITY_KEYS:
        if ident.get(k):
            lines.append(f"{k}: {ident[k] if isinstance(ident[k], str) else json.dumps(ident[k])}")
    resume = _resume()
    if resume:
        lines.append(f"resume:\n{resume}")
    for j in target.jobs:
        fit = next((v for k, v in (j.get("fields") or {}).items() if k.lower().startswith("why it's a fit")), "")
        if fit:
            lines.append(f"fit notes for {j['role']}:\n{fit[:1500]}")
    return "\n".join(lines)


def roles_text(target) -> str:
    roles = list(dict.fromkeys(j["role"] for j in target.jobs if j.get("role")))
    if len(roles) <= 1:
        return roles[0] if roles else "a role"
    return ", ".join(roles[:-1]) + " and " + roles[-1]


def _brief(target) -> str:
    p = target.contact
    return (f"Company: {target.company}\nRole(s): {roles_text(target)}\n"
            f"Recipient: {p.name}, {p.title or 'works there'}\n\nCandidate profile:\n{outreach_profile(target)}")


def first_name(name: str) -> str:
    return (name or "").split()[0] if (name or "").strip() else "there"


def bare_company(company: str) -> str:
    """'Brainbase Labs (YC W24)' → 'Brainbase Labs'."""
    return re.sub(r"\s*\(.*?\)\s*", " ", company or "").strip()


def _clean(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", (text or "").strip().strip('"').strip())


def words(text: str) -> set:
    """Content words (4+ letters, common filler dropped), lowercased, plurals folded
    ("dealerships" matches "dealership")."""
    return {re.sub(r"(?<=[^s])s$", "", w) for w in _WORD.findall((text or "").lower()) if w not in _STOP}


def _numbers(text: str) -> set:
    return {n.replace(",", "").rstrip(".,") for n in _NUMBER.findall(text or "")}


def _runs(text: str, n: int = 6) -> set:
    w = re.findall(r"[a-z0-9']+", (text or "").lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def greeting(brief: Brief) -> str:
    return f"Hi {bare_company(brief.company)} team," if brief.kind == GENERIC else f"Hi {first_name(brief.person)},"


def role_phrase(role: str) -> str:
    """'Software Engineer (New Grad)' → 'Software Engineer (New Grad) role'; '… role' stays as is."""
    role = (role or "").strip() or "open"
    return role if re.search(r"(?i)\b(role|position)$", role) else f"{role} role"


def ask(brief: Brief) -> str:
    role = role_phrase(brief.role)
    if brief.kind == GENERIC:
        return f"I applied for the {role}. Resume attached. Could you pass this to whoever is hiring for the {role}?"
    return (f"I applied for the {role}. Resume attached. Would you be open to 15 minutes, "
            "or could you point me to the right person on the team?")


# role → which of the candidate's entries fits it (first rule that matches wins)
_VOICE = re.compile(r"(?i)\bvoice|speech|audio|conversational|call.?cent|phone|dispatch|\b911\b")
_WORK_RULES = (
    ("role", re.compile(r"(?i)machine learning|\bml\b|\bdata\b|analytics|data scien"), ("appstek",)),
    ("role", re.compile(r"(?i)\bllm|\bai\b|\bagent|genai|gen ai|\bnlp\b|inference"), ("llm gateway",)),
    ("role", re.compile(r"(?i)front.?end|full.?stack|product engineer|\bui\b|design engineer|react|mobile"),
     ("draw.ai", "appetit")),
    ("problem", re.compile(r"(?i)\bllm|\bagents?\b|\bmodels?\b|\bai\b"), ("llm gateway",)),
    ("any", None, ("appetit",)),
)


def _history() -> list:
    """[{label, text, key}] for every job and project in the answer bank (resume until edited)."""
    out = []
    for e in answer_bank.get_list("experience"):
        if e.get("company"):
            out.append({"label": f"{e.get('title') or 'Work'} at {e['company']}", "key": e["company"].lower(),
                        "text": e.get("description") or ""})
    for e in answer_bank.get_list("projects"):
        if e.get("name"):
            out.append({"label": e["name"], "key": e["name"].lower(), "text": e.get("description") or ""})
    return out


def _named(entry: dict, text: str) -> bool:
    """The entry's short name ('Appetit', 'LLM Gateway', 'Draw.AI') appears in text."""
    short = re.split(r"\s+[-–—(]|\s+corp\b|\s+inc\b", entry["key"])[0].strip()
    return bool(short) and re.search(r"(?i)(?<![a-z])" + re.escape(short) + r"(?![a-z])", text or "") is not None


@dataclass
class Plan:
    """What the email is about: one work entry, and whether Muse's Angle is used for it."""
    work: Optional[dict]
    use_angle: bool = True
    override: bool = False      # Muse named another project by analogy only: the role's pick won
    why: str = ""


def angle_work(brief: Brief) -> Optional[dict]:
    """The job or project Muse's Angle names, if any."""
    return next((e for e in _history() if _named(e, brief.angle)), None)


def role_work(brief: Brief) -> Optional[dict]:
    """The job or project that fits the role (and, for voice, the company's problem)."""
    history = _history()
    if not history:
        return None
    rules = ((("any", None, ("aria",)),) if _VOICE.search(f"{brief.role} {brief.problem}") else ()) + _WORK_RULES
    for where, pattern, keys in rules:
        text = {"role": brief.role, "problem": brief.problem}.get(where, "")
        if pattern is None or pattern.search(text):
            for k in keys:
                hit = next((e for e in history if k in e["key"]), None)
                if hit:
                    return hit
    return history[0]


_JUDGE_SYSTEM = (
    "You check one line of job-outreach research. The Angle links one of the candidate's "
    "projects to a company. Decide: does it rest on a CONCRETE fact about this company — "
    "something the company actually does, ships, runs, announced or credits (e.g. \"CarMax "
    "runs voice agents for inbound calls\") — or only on an ANALOGY (\"plan-act-verify is like "
    "flight software\", \"the same pattern applied to travel\")? A generic description of "
    "the industry is an analogy. Reply with JSON only: "
    '{"concrete": true, "fact": "<the company fact, quoted from the research>"} or '
    '{"concrete": false}.'
)


def _judge_validator(research: str):
    def check(text: str):
        m = re.search(r"\{.*\}", text or "", re.S)
        try:
            data = json.loads(m.group(0)) if m else None
        except ValueError:
            return None
        if not isinstance(data, dict) or not isinstance(data.get("concrete"), bool):
            return None
        if not data["concrete"]:
            return (False, "")
        fact = str(data.get("fact") or "").strip()
        claimed, have = words(fact), words(research)
        if len(claimed) < 2 or sum(1 for w in claimed if w in have) / len(claimed) < 0.6:
            return None                             # the "fact" isn't in Muse's research
        return (True, fact)
    return check


def angle_is_concrete(brief: Brief) -> tuple:
    """(True, the fact) when the Angle rests on a concrete company fact, else (False, '').
    A judge that fails twice counts as an analogy (the role's pick is the safe one)."""
    research = f"{brief.problem} {brief.signal} {brief.signal_source}"
    text = (f"Company: {bare_company(brief.company)}\nRole: {brief.role}\n\nResearch:\n"
            f"Problem: {brief.problem}\nSignal: {brief.signal}\nSignal source: {brief.signal_source}\n\n"
            f"Angle: {brief.angle}")
    try:
        _resp, out = llm_client.complete_checked(
            "outreach_angle_judge", [{"role": "user", "content": text}],
            validate=_judge_validator(research), system=_JUDGE_SYSTEM, max_tokens=200)
    except Exception as exc:
        logger.error("outreach: angle judge for %s failed: %s", brief.company, exc)
        return False, ""
    return out or (False, "")


def plan(brief: Brief) -> Plan:
    """Which work the email is about (the user's rule; see the module docstring)."""
    named, fits = angle_work(brief), role_work(brief)
    if named is None:
        return Plan(fits, why="the Angle names no project; the role's pick")
    if fits is None or named["key"] == fits["key"]:
        return Plan(named, why="Muse's pick fits the role")
    concrete, fact = angle_is_concrete(brief)
    if concrete:
        return Plan(named, why=f"Muse's pick, tied to a company fact: {fact}")
    return Plan(fits, use_angle=False, override=True,
                why=f"Muse's {named['label']} rests on an analogy; the role fits {fits['label']}")


def work_for(brief: Brief) -> Optional[dict]:
    """The one job or project the email talks about."""
    return plan(brief).work


def _input(brief: Brief, p: Plan) -> str:
    who = "a generic company inbox" if brief.kind == GENERIC else \
        f"{brief.person}, {brief.title or 'works there'}" + (" (recruiter / manager)" if brief.kind == HIRING else "")
    text = f"Company: {bare_company(brief.company)}\nRole: {brief.role}\nRecipient: {who}\n\nCompany problem: {brief.problem}\n"
    if p.use_angle:
        text += f"Angle (the only experience you may claim): {brief.angle}\n"
    if not _blank(brief.signal):
        text += f"Signal: {brief.signal}\n"
    if p.work:
        text += f"\nThe candidate's work for this role (the only project to talk about): {p.work['label']}\n{p.work['text']}"
        if not p.use_angle:
            text += "\n(There's no Angle for this one: line 2 is this work and what was hard in it.)"
    return text


def _validator(brief: Brief, p: Plan):
    """(subject, [line1, line2, line3]) when the output passes every check, else None."""
    built = brief.angle if p.use_angle or not p.work else f"{p.work['label']} {p.work['text']}"
    theirs = f"{brief.company} {brief.problem} {brief.signal}"
    grounded = words(f"{brief.problem} {'' if _blank(brief.signal) else brief.signal}")
    if p.use_angle:                                 # Muse's tie to the company, minus the project itself
        grounded |= words(brief.angle) - (words(f"{p.work['label']} {p.work['text']}") if p.work else set())
    allowed = _numbers(theirs)
    copied = _runs(f"{brief.problem} {brief.signal}")
    fixed = f"{greeting(brief)} {ask(brief)}"

    def no(why: str):
        logger.warning("outreach: %s email rejected — %s", brief.company, why)
        check.why = why
        return None

    def check(text: str):
        parts = [p.strip() for p in _clean(text).split("\n\n") if p.strip()]
        if len(parts) != 4:                         # some models break the lines with single newlines
            parts = [p.strip() for p in _clean(text).split("\n") if p.strip()]
        m = re.match(r"(?i)^subject:\s*(.+)$", parts[0]) if parts else None
        if not m or len(parts) != 4:
            return no(f"shape: {len(parts)} blocks, subject line {'found' if m else 'missing'}")
        subject, lines = m.group(1).strip().strip('"'), parts[1:]
        prose = " ".join(lines)
        if not subject or len(subject) >= SUBJECT_CHARS or "application" in subject.lower():
            return no(f"subject: {subject!r}")
        if subject_problem(subject):
            return no(f"subject {subject!r} is {subject_problem(subject)}")
        if any(_GREETING.match(l) or _SIGNOFF.search(l) for l in lines) or _PLACEHOLDER.search(prose + subject):
            return no("greeting / sign-off / placeholder in the body")
        if _BANNED.search(prose + " " + subject) or re.search(r"(?i)\bresume attached|\bI applied\b", prose):
            return no("banned phrase")
        bad = [n for n in _numbers(prose + " " + subject) if n not in allowed]
        if bad:
            return no(f"numbers not in their problem/signal: {bad}")
        if _runs(prose) & copied:
            return no("restates their own copy")
        if len(words(lines[0]) & grounded) < 2:
            return no("line 1 not tied to the company's problem/signal")
        if len(words(lines[1]) & words(built)) < 2:
            return no("line 2 not tied to the candidate's work")
        n = len(f"{fixed} {prose}".split())
        if not EMAIL_WORDS[0] <= n <= EMAIL_WORDS[1]:
            return no(f"{n} words (want {EMAIL_WORDS[0]}–{EMAIL_WORDS[1]})")
        return subject, lines
    check.why = ""
    return check


def email(brief: Brief, to: str, p: Optional[Plan] = None) -> Optional[dict]:
    """{to, subject, body} for one Contacts row. None when a required input is missing or
    the LLM output failed the checks (nothing is drafted then). p: plan(brief), if the
    caller already has it."""
    if missing(brief):
        logger.info("outreach: %s needs research (%s)", brief.company, ", ".join(missing(brief)))
        return None
    p = p or plan(brief)
    check = _validator(brief, p)
    messages = [{"role": "user", "content": _input(brief, p)}]
    try:
        resp, out = llm_client.complete_checked(
            "outreach_email", messages, validate=check, system=_EMAIL_SYSTEM, max_tokens=500)
        if out is None and check.why and (resp.text or "").strip():
            # One rewrite with the reason it was refused (models miscount words).
            messages += [{"role": "assistant", "content": resp.text},
                         {"role": "user", "content": f"That was refused: {check.why}. Rewrite it in "
                                                     "the same format, fixing only that."}]
            resp = llm_client.complete(messages, task="outreach_email", max_tokens=500,
                                       system=_EMAIL_SYSTEM)
            out = check(resp.text or "") if resp.stop_reason != "error" else None
    except Exception as exc:
        logger.error("outreach: email for %s failed: %s", brief.company, exc)
        return None
    if not out:
        return None
    subject, lines = out
    body = "\n\n".join([greeting(brief), *lines, ask(brief), signature()])
    return {"to": to, "subject": subject, "body": body}


def _note_validator(prefix: str):
    def check(text: str) -> Optional[str]:
        text = re.sub(r"\s+", " ", _clean(text))
        if not text or _PLACEHOLDER.search(text) or _GREETING.match(text) or "#" in text:
            return None
        note = prefix + text
        return note if len(note) <= NOTE_LIMIT else None
    return check


def linkedin_note(target) -> Optional[str]:
    prefix = f"Hi {first_name(target.contact.name)}, "
    try:
        _resp, note = llm_client.complete_checked(
            "outreach_linkedin_note", [{"role": "user", "content": _brief(target)}],
            validate=_note_validator(prefix), system=_NOTE_SYSTEM, max_tokens=200)
    except Exception as exc:
        logger.error("outreach: LinkedIn note for %s failed: %s", target.company, exc)
        return None
    return note or None
