"""features/ats/planner.py — what goes in every form field. Code decides; the LLM only writes.

Shared by the Greenhouse adapter and the generic DOM form engine (form_engine.py).

classify(field) → category, by rules in this order:
  upload     file inputs (resume / cover letter)
  consent    agreements, attestations, privacy / terms checkboxes — ticked by the user
             when they approve the submit, never by Aria
  fact       the answer bank knows the question (name, email, links, education, location)
  sensitive  salary, sponsorship, work authorization, demographics, disability, veteran,
             criminal history — stored answers only, never guessed
  choice     fixed options the bank doesn't cover
  open_text  any free-text question the bank doesn't cover

plan(form, job, profile) → [Planned], each with a source:
  bank       from the answer bank (facts, stored sensitive answers, choices it covers)
  llm_draft  open_text written by the frontier LLM from the resume + job (char limit kept)
  reused     an earlier LLM answer to the same generic question (generated_answers table)
  llm_guess  a background choice question ("Which archetype fits you?") picked from its
             options by the LLM using the resume; UNKNOWN → needs_me
  needs_me   UNKNOWN: a fact or sensitive answer missing from the bank — the ONLY thing
             that goes to the user
  consent    an agreement for the user to tick when approving the submit
  failed     the LLM couldn't write it (gateway down / error)
  skipped    optional, nothing to say

HARD RULE: open_text never becomes needs_me.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Optional

import aria.features.answer_bank as answer_bank
import aria.state.db as db
from aria.core import paths as _aria_paths
from aria.features.ats import answers as llm_answers
from aria.features.ats import resume_text

logger = logging.getLogger(__name__)

SOURCES = ("bank", "llm_draft", "reused", "llm_guess", "needs_me", "consent", "failed", "skipped")
FILLED_SOURCES = ("bank", "llm_draft", "reused", "llm_guess")

_DRAFT_LABEL = re.compile(
    r"\bwhy\b|\bdescribe\b|\btell us\b|\bwhat (?:excites|interests|draws|motivates)\b|\bcover letter\b", re.I
)
_QUESTIONISH = re.compile(
    r"\?\s*\*?\s*$|^\s*(?:what|why|how|which|describe|tell|explain|share|list|give|walk|briefly|provide|insert|"
    r"please (?:describe|explain|share|tell|provide|list|briefly)|in (?:a|one|two|three|\d+) )\b",
    re.I,
)
# Optional catch-alls: an LLM paragraph there is noise.
_OPTIONAL_CATCHALL = re.compile(
    r"additional information|anything else|other (?:comments|information)|\bcomments?\b|\bnotes?\b|\bmessage\b", re.I
)
# Agreements, attestations and consents are the user's to give — never filled for them.
_ATTESTATION = re.compile(
    r"\bagree|\backnowledg|\barbitrat|\bconsent|\bcertif|\battest|\bpolicy\b|\bterms\b|"
    r"\bprivacy\b|\bplease (?:read|review)\b|\bsignature\b|\bsign\b",
    re.I,
)
# Never guessed: answered from the bank or left for the user.
_SENSITIVE = re.compile(
    r"salary|compensation|\bpay\b|\bwage|sponsor|\bvisa\b|h-?1b|authori[sz]ed|work authori|citizen|"
    r"immigration|gender|\bsex\b|pronoun|\brace\b|ethnic|hispanic|latin[oax]|veteran|military|disabilit|"
    r"sexual orientation|lgbt|transgender|criminal|convict|felony|background check|date of birth|\bage\b|"
    r"\bssn\b|social security|security clearance",
    re.I,
)
# Your preferences (offices, locations, deadlines) are facts only you know: never written by the LLM.
_PREFERENCE = re.compile(r"\bprefer(?:red|ence|ences|s)?\b|\bdeadlines?\b", re.I)
# Labels that ask for a personal fact even in a text box: the LLM may say UNKNOWN for these
# (and only these) — every other open question gets written from the resume.
_FACT_SHAPED = re.compile(r"\bname\b|\bdate\b|\bhigh school\b|\baddress\b|\bphone\b|\bnumber\b|\bhow many\b|"
                          r"\bwhen\b|\bwhere\b|\byear\b|\bid\b|\bcity\b", re.I)
_COVER_LETTER = re.compile(r"cover\s*letter", re.I)
_RESUME = re.compile(r"\bresume\b|\bcv\b|résumé", re.I)
# A company-specific question: its LLM answer is never reused for another company.
_COMPANY_SPECIFIC = re.compile(
    r"\bwhy\b|\bthis (?:role|position|job|team|company|opportunity)\b|\bour\b|\bus\b|\bhere\b|\byou(?:r)? (?:company|team)\b",
    re.I,
)


@dataclass
class FormField:
    name: str                 # DOM id (Greenhouse) or the engine's data-aria-fid
    label: str
    kind: str                 # text | textarea | file | select | multiselect | location | checkbox | radio | date
    required: bool = False
    options: list = field(default_factory=list)
    section: str = "questions"  # questions | eeoc | demographic | dom
    max_chars: int = 0        # the field's maxlength, 0 = none


@dataclass
class Planned:
    field: FormField
    value: str = ""
    source: str = "skipped"
    note: str = ""
    category: str = ""


@dataclass
class ParsedForm:
    fields: list
    title: str = ""
    company: str = ""
    description: str = ""
    salary_range: Optional[tuple] = None


# ---------------------------------------------------------------------------
# Profile (the facts the LLM may use)
# ---------------------------------------------------------------------------

def _identity() -> dict:
    try:
        return json.loads(_aria_paths.IDENTITY_JSON.read_text())
    except (OSError, ValueError):
        return {}


def resume_path() -> Optional[str]:
    """resume_path from the answer bank, else identity.json — only if the file exists."""
    bank = answer_bank.load()
    candidates = [bank["answers"].get("resume_path"), _identity().get("resume_path")]
    for c in candidates:
        if c and Path(str(c)).expanduser().is_file():
            return str(Path(str(c)).expanduser())
    return None


_PROFILE_SKIP = {"eeo", "salary_rule", "resume_path", "sponsorship", "work_authorization"}


def build_profile(job: dict, resume: Optional[str] = None) -> str:
    """Facts the LLM may use: the bank's plain answers, the resume's text, identity, the sheet row.
    Sensitive answers (EEO, sponsorship, work authorization, salary) are never included."""
    bank = answer_bank.load()
    lines = []
    for k, v in bank["answers"].items():
        if k in _PROFILE_SKIP or v in (None, "") or isinstance(v, dict):
            continue
        if not str(v).startswith("<"):
            lines.append(f"{k}: {v}")
    ident = _identity()
    for k in ("summary", "skills", "experience", "education", "target_roles"):
        if ident.get(k):
            lines.append(f"{k}: {json.dumps(ident[k]) if not isinstance(ident[k], str) else ident[k]}")
    fit = next((v for k, v in (job.get("fields") or {}).items() if k.lower().startswith("why it's a fit")), "")
    if fit:
        lines.append(f"notes on this role and the candidate's relevant work:\n{fit}")
    cv = resume_text.text(resume if resume is not None else resume_path())
    if cv:
        lines.append(f"\nRESUME:\n{cv}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def is_question(label: str) -> bool:
    return bool(_QUESTIONISH.search((label or "").strip()))


def is_sensitive(f: FormField) -> bool:
    return f.section in ("eeoc", "demographic") or bool(_SENSITIVE.search(f.label or ""))


_CONSENT_OPTION = re.compile(r"\b(?:i )?(?:agree|acknowledge|consent|accept|certify|attest)\b", re.I)
_TODAY_LABEL = re.compile(r"^(?:today'?s )?date$|^(?:signature|current|signed) date$|^date signed$", re.I)
_RESIDENCE = re.compile(r"\bcountry\b|(?:location|country|place) of residence|\bresid(?:e|ence)\b", re.I)


def is_consent(f: FormField) -> bool:
    """Agreement wording in the label, or options that are agreements ("I agree") — not a
    language list that happens to contain "Sign Language"."""
    return bool(_ATTESTATION.search(f.label or "")) or any(_CONSENT_OPTION.search(o) for o in f.options)


def open_text_kind(f: FormField) -> bool:
    """Free text the LLM writes: every textarea, and one-line inputs asked as a question."""
    return f.kind == "textarea" or (f.kind == "text" and is_question(f.label))


# ---------------------------------------------------------------------------
# Generated answers, reused on the same generic question
# ---------------------------------------------------------------------------

def reusable(label: str, company: str) -> bool:
    if _COMPANY_SPECIFIC.search(label or ""):
        return False
    return not (company and company.lower() in (label or "").lower())


def saved_answer(label: str, max_chars: int = 0) -> Optional[str]:
    conn = None
    try:
        conn = db.get_connection()
        row = conn.execute("SELECT answer FROM generated_answers WHERE question = ?",
                           (answer_bank.normalize(label),)).fetchone()
    except Exception as exc:
        logger.warning("planner: generated_answers read failed: %s", exc)
        return None
    finally:
        if conn is not None:
            conn.close()
    if not row:
        return None
    answer = row["answer"]
    return answer if not max_chars or len(answer) <= max_chars else None


def save_answer(label: str, answer: str, company: str = "", max_chars: int = 0) -> None:
    q = answer_bank.normalize(label)
    if not q or not (answer or "").strip():
        return
    conn = None
    try:
        conn = db.get_connection()
        conn.execute(
            "INSERT INTO generated_answers (question, answer, company, max_chars, created_at) "
            "VALUES (?, ?, ?, ?, datetime('now')) ON CONFLICT(question) DO UPDATE SET "
            "answer=excluded.answer, company=excluded.company, max_chars=excluded.max_chars, "
            "created_at=excluded.created_at",
            (q, answer.strip(), company or "", int(max_chars or 0)))
        conn.commit()
    except Exception as exc:
        logger.warning("planner: generated_answers write failed: %s", exc)
    finally:
        if conn is not None:
            conn.close()


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------

COVER_LETTER_DIR = Path.home() / ".aria" / "cover_letters"


def upload_path(p: Planned) -> str:
    """The file to attach: the resume path as planned, or a drafted cover letter written
    to ~/.aria/cover_letters/<field>.txt."""
    if p.source == "bank" or Path(p.value).expanduser().is_file():
        return p.value
    COVER_LETTER_DIR.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^\w-]+", "_", p.field.name)[:60] or "cover_letter"
    out = COVER_LETTER_DIR / f"cover_letter_{name}.txt"
    out.write_text(p.value)
    return str(out)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

# Greenhouse's education block: field ids say what each one is; labels ("Start date
# year") would match the job's start date instead.
_EDU_FIELD = re.compile(r"^(school|degree|discipline|start-month|start-year|end-month|end-year)--\d+$")
_EDU_KEY = {"school": "school", "degree": "degree", "discipline": "major", "start-month": "edu_start_month",
            "start-year": "edu_start_year", "end-month": "grad_month", "end-year": "grad_year"}
# Remote-search dropdowns: the first 100 options are all that load until you type.
_SEARCHABLE = {"school", "discipline"}


def _why_unfilled(key: str, f: FormField, job_ctx: dict) -> str:
    """The note for a field the bank knows but couldn't fill."""
    plain = answer_bank.lookup_key(key, job=job_ctx, label=f.label)
    if plain and f.options:
        return f"your answer “{plain.value}” isn't one of the options"
    return f"no {answer_bank.key_title(key).lower()} in your bank yet — add it in My answers"


def _unknown(p: Planned, note: str) -> None:
    p.source = "needs_me" if p.field.required else "skipped"
    p.note = note if p.field.required else ""


def _bank(f: FormField, job_ctx: dict, llm_available: bool):
    """(answer, matched key) from the bank for this field."""
    edu = _EDU_FIELD.match(f.name)
    key = _EDU_KEY[edu.group(1)] if edu else None
    if key:      # the education block: known by field name, whatever its label says
        searchable = edu.group(1) in _SEARCHABLE
        found = answer_bank.lookup_key(key, options=None if searchable else (f.options or None),
                                       job=job_ctx, label=f.label)
    else:
        # An open question ("What's the best project you've built?") is never a bank fact by
        # an LLM's say-so — only the bank's own aliases / patterns can claim it.
        # ...and the LLM key pick only gets short field names ("Current employer"), never
        # textareas or sentences ("Provide any further context on preference" isn't your name).
        llm = (llm_available and f.kind != "textarea" and not is_question(f.label)
               and len(answer_bank.normalize(f.label).split()) <= 4)
        found = answer_bank.lookup(f.label, options=f.options or None, job=job_ctx, llm=llm)
        if found:
            return found, key
        if f.options and _RESIDENCE.search(f.label or ""):     # "Location of Residence": US / EEA / Other
            found = answer_bank.lookup_key("country", options=f.options, job=job_ctx, label=f.label)
            if found:
                return found, "country"
        return None, answer_bank.matched_key(f.label, llm=llm)
    if found:
        return found, key
    return None, key


def plan(
    form: ParsedForm,
    job: dict,
    profile: str,
    guess_fn: Callable = llm_answers.guess_short,
    draft_fn: Callable = llm_answers.draft_long,
    resume: Optional[str] = None,
    progress_fn: Optional[Callable[[str], None]] = None,
    llm_available: bool = True,
    reuse: bool = True,
) -> list:
    job_ctx = {**job, "salary_range": form.salary_range or job.get("salary_range"),
               "description": form.description or job.get("description", "")}
    company = form.company or job.get("company", "")
    plans: list[Planned] = []
    to_guess: list[Planned] = []
    to_draft: list[Planned] = []

    for f in form.fields:
        p = Planned(f)
        plans.append(p)

        if f.kind == "file":
            p.category = "upload"
            if f.name == "resume" or _RESUME.search(f.label) and not _COVER_LETTER.search(f.label):
                if resume:
                    p.value, p.source = resume, "bank"
                else:
                    _unknown(p, "no resume_path set")
            elif _COVER_LETTER.search(f.label) and f.required:
                to_draft.append(p)
            elif f.required:
                _unknown(p, "file upload")
            continue

        if is_consent(f):
            p.category = "consent"
            p.source = "consent" if f.required else "skipped"
            p.note = "agreement — tick it yourself when you approve the submit" if f.required else ""
            continue

        if _TODAY_LABEL.match(answer_bank.normalize(f.label)) and not f.options:   # a signature block's date
            p.category, p.value, p.source = "fact", date.today().strftime("%m/%d/%Y"), "bank"
            p.note = "today's date"
            continue

        if answer_bank.always_ask(f.label):
            p.category = "fact"
            _unknown(p, "you asked to always answer this yourself")
            continue

        found, key = _bank(f, job_ctx, llm_available)
        if found and f.kind == "textarea" and answer_bank.normalize(found.value) in ("yes", "no"):
            found, key = None, None            # a bare Yes/No never answers an essay box
        if found:
            p.category = "sensitive" if is_sensitive(f) else ("choice" if f.options else "fact")
            p.value, p.source = found.value, "bank"
            continue
        if key:
            p.category = "sensitive" if is_sensitive(f) else "fact"
            _unknown(p, _why_unfilled(key, f, job_ctx))
            continue
        if is_sensitive(f):
            p.category = "sensitive"
            _unknown(p, "sensitive — answer it once in My answers; never guessed")
            continue

        if _PREFERENCE.search(f.label) and not f.options:
            p.category = "fact"
            _unknown(p, "your preference — answer it once and Aria remembers")
            continue

        if f.options or f.kind in ("select", "multiselect", "radio", "checkbox"):
            p.category = "choice"
            if is_question(f.label) or f.required:
                to_guess.append(p)
            continue

        if open_text_kind(f):
            p.category = "open_text"
            # Optional one-liners ("What address will you work from?") are facts or logistics,
            # not essays; optional textareas only when they ask about you or the job.
            if f.required or (f.kind == "textarea" and _DRAFT_LABEL.search(f.label)
                              and not _OPTIONAL_CATCHALL.search(f.label)):
                to_draft.append(p)
            continue

        p.category = "fact"
        _unknown(p, "not in your answer bank — add it in My answers")

    if not llm_available:
        for p in to_guess:
            p.source, p.note = ("failed" if p.field.required else "skipped"), "LLM gateway was down"
        for p in to_draft:
            p.source, p.note = ("failed" if p.field.required or p.category == "open_text" else "skipped"), \
                "LLM gateway was down"
        to_guess, to_draft = [], []

    if to_guess:
        guesses = guess_fn([{"name": p.field.name, "label": p.field.label,
                             "options": p.field.options or (["Yes", "No"] if p.field.kind == "checkbox" else [])}
                            for p in to_guess], profile)
        for p in to_guess:
            if p.field.name in guesses:
                p.value, p.source = guesses[p.field.name], "llm_guess"
            else:
                _unknown(p, "your resume doesn't say — answer it once and Aria remembers")

    job_text = f"{form.title or job.get('role', '')} at {company}\n\n{job_ctx['description']}"
    for p in to_draft:
        f = p.field
        p.category = p.category or "open_text"
        label = f"Cover letter for {form.title or job.get('role', 'this role')} at {company}" \
            if f.kind == "file" else f.label
        if reuse and f.kind != "file" and reusable(f.label, company):
            again = saved_answer(f.label, f.max_chars)
            if again:
                p.value, p.source, p.note = again, "reused", "same answer as an earlier application"
                continue
        if progress_fn:
            progress_fn(f"Drafting an answer for: {label[:60]}")
        short = f.kind == "text"
        may_be_fact = f.kind != "file" and (not is_question(f.label) or bool(_FACT_SHAPED.search(f.label)))
        text = ""
        for _attempt in range(2):
            text = draft_fn(label, job_text, profile, max_chars=f.max_chars, short=short, allow_unknown=may_be_fact)
            if text and not (text == llm_answers.UNKNOWN and not may_be_fact):
                break
            text = ""
        if text == llm_answers.UNKNOWN:      # a personal fact in a text box, not something to write
            p.category = "fact"
            _unknown(p, "a personal fact your resume doesn't have — answer it once and Aria remembers")
        elif text:
            p.value, p.source = text, "llm_draft"
            if f.kind != "file":
                save_answer(f.label, text, company, f.max_chars)
        else:
            p.source, p.note = "failed", ("the LLM wouldn't write it from your resume" if llm_available
                                          else "the LLM gateway was down")
    return plans
