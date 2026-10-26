"""features/resume/cover_letter.py — a one-page cover letter PDF from the tailored resume + JD.

write(resume, jd_text, job, path) → bool
    One strong-tier call (task cover_letter_write) returns the body as JSON
    {"paragraphs": [3–4 strings]}, validated in code: 150–380 words, no greeting/sign-off,
    every fact checked against the resume (answer_check: numbers, employers, schools, projects
    must be on it — offending sentences are dropped). Rendered with the resume's fonts and
    header (render.render_letter). False when the LLM failed or nothing usable was left.

from_text(resume, text, path) → bool
    An already-written letter (the planner's form_free_text draft) rendered the same way.

Never raises.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date

from aria.features.ats import untrusted
from aria.features.resume import model, render

logger = logging.getLogger(__name__)

MIN_WORDS, MAX_WORDS = 150, 380

_SYSTEM = (
    "You write a candidate's cover letter body for one job, in first person, plain and specific. "
    "Use ONLY facts from the candidate's resume (employers, projects, numbers, skills) and the job "
    "description. Never invent an employer, project, number, credential or claim. 3 or 4 short "
    "paragraphs, 180-320 words in total: why this role, the two most relevant pieces of work from the "
    "resume tied to what the job needs, a one-sentence close. No greeting, no sign-off, no markdown. "
    'Reply with JSON only: {"paragraphs": ["...", "..."]}. ' + untrusted.RULE
)
_GREETING = re.compile(r"^\s*(?:dear|hello|hi|to whom)\b.*$|^\s*(?:sincerely|best regards|regards|thank you),?\s*$",
                       re.I | re.M)


def _paragraphs(raw: str):
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    got = json.loads(m.group(0)).get("paragraphs")
    if not isinstance(got, list):
        return None
    paras = [" ".join(_GREETING.sub("", str(p)).split()) for p in got]
    paras = [p for p in paras if p]
    words = sum(len(p.split()) for p in paras)
    return paras if 2 <= len(paras) <= 5 and MIN_WORDS <= words <= MAX_WORDS else None


def _checked(paras: list, resume: dict) -> list:
    """Each paragraph with sentences that state facts the resume doesn't have removed."""
    from aria.features.ats import answer_check
    out = []
    for p in paras:
        fixed = answer_check.strip_violations(p, resume)
        if fixed:
            out.append(fixed)
    return out


def write(resume: dict, jd_text: str, job: dict, path) -> bool:
    from aria.llm import llm_client
    block, _flags = untrusted.guard((jd_text or "")[:6000], "Job description")
    user = (f"Job: {job.get('role') or '?'} at {job.get('company') or '?'}\n\n{block}\n\n"
            f"Candidate resume:\n{model.plain_text(resume)}")
    try:
        _resp, paras = llm_client.complete_checked("cover_letter_write", [{"role": "user", "content": user}],
                                                   validate=_paragraphs, system=_SYSTEM, max_tokens=900)
    except Exception as exc:
        logger.warning("cover_letter: LLM failed: %s", exc)
        return False
    if not paras:
        return False
    paras = _checked(paras, resume)
    if sum(len(p.split()) for p in paras) < MIN_WORDS // 2:
        return False
    return _render(resume, paras, path)


def from_text(resume: dict, text: str, path) -> bool:
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", _GREETING.sub("", text or "")) if p.strip()]
    return bool(paras) and _render(resume, paras, path)


def _render(resume: dict, paras: list, path) -> bool:
    try:
        return render.render_letter(resume, paras, path, date_line=date.today().strftime("%B %-d, %Y"))
    except Exception as exc:
        logger.warning("cover_letter: render failed: %s", exc)
        return False
