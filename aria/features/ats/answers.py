"""features/ats/answers.py — LLM help for form questions the answer bank can't answer.

Both go through GateLLM (aria.llm.llm_client), tiers from aria/llm/tasks.py:
  guess_short(questions, profile)          task form_short_answer (small), one batched call,
                                           validated in code (JSON, answers within options,
                                           length) and escalated once to frontier on failure.
                                           Returns {name: answer} only for answers the
                                           profile supports.
  draft_long(label, job_text, profile, max_chars, short)
                                           task form_free_text (frontier), one call per question;
                                           the field's character limit is enforced in code.

Callers mark the results "llm_guess" / "llm_draft" so the user reviews them.
Never raises — failures return {} / "".
"""
from __future__ import annotations

import json
import logging
import re
from typing import Callable, Optional

from aria.features.answer_bank import choose_option
from aria.features.ats import untrusted
from aria.llm import llm_client

logger = logging.getLogger(__name__)

UNKNOWN = "UNKNOWN"

_GUESS_SYSTEM = (
    "You fill in short job-application questions for a candidate, using ONLY the candidate "
    "profile you are given. Rules:\n"
    "- If a question lists options, the answer must be exactly one of those options, copied verbatim.\n"
    f"- If the profile does not clearly support an answer, answer {UNKNOWN}. Never guess personal "
    "facts (dates, numbers, names, legal or visa status, criminal history, demographics).\n"
    "- Return a single JSON object mapping each question id to its answer. No prose, no markdown.\n"
    f"- {untrusted.RULE}"
)

_DRAFT_SYSTEM = (
    "You write a job candidate's answer to one free-text application question. First person, "
    "plain text, specific and concrete, no greeting or sign-off, no markdown. Use ONLY facts from the "
    "candidate profile and resume (projects, employers, numbers, skills) and the job description — "
    "never invent a project, employer, number, credential or opinion the resume doesn't support. "
    "Name a real project or result from the resume when the question allows it. Answer exactly "
    "what was asked. " + untrusted.RULE
)
_UNKNOWN_RULE = (f"If this asks for a personal fact the profile doesn't state (a name, date, place, number), "
                 f"reply with exactly {UNKNOWN}.")
_ALWAYS_ANSWER = ("This is an open question about the candidate: always write an answer grounded in the "
                  "resume — never refuse, never say you lack information. If it asks for something not on the "
                  "resume, pick something the resume only implies (how a listed project was built, a hard "
                  "trade-off it forced, what it taught) and tell that — still no new facts.")
_REFUSAL = re.compile(r"^\s*(?:i (?:can(?:no|')t|cannot|am unable to|don't have)|unfortunately|unknown\b|n/?a\b|"
                      r"(?:the|your) (?:profile|resume) (?:does not|doesn't))", re.I)


def _parse_json_object(text: str) -> dict:
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(raw[start:end + 1])
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _validator(questions: list[dict]) -> Callable[[str], Optional[dict]]:
    """Valid = a JSON object; every non-UNKNOWN answer to an options question is one of
    its options. Returns {name: answer} with unknowns dropped, or None (→ escalate)."""
    def validate(text: str) -> Optional[dict]:
        data = _parse_json_object(text)
        if not data:
            return None
        out: dict[str, str] = {}
        for q in questions:
            ans = str(data.get(q["name"], "")).strip()
            if not ans or ans.upper() == UNKNOWN:
                continue
            if q.get("options"):
                chosen = choose_option(ans, q["options"])
                if chosen is None:
                    return None
                ans = chosen
            elif len(ans) > 255:
                return None
            out[q["name"]] = ans
        return out
    return validate


def guess_short(questions: list[dict], profile: str) -> dict[str, str]:
    """questions: [{"name", "label", "options": [...]}] → {name: answer} (unknowns omitted)."""
    if not questions:
        return {}
    listing = [
        {"id": q["name"], "question": q["label"], **({"options": q["options"]} if q.get("options") else {})}
        for q in questions
    ]
    questions_block, _flags = untrusted.guard(json.dumps(listing, indent=1), "Questions from the form")
    user = f"Candidate profile:\n{profile}\n\n{questions_block}"
    try:
        _resp, answers = llm_client.complete_checked(
            "form_short_answer",
            [{"role": "user", "content": user}],
            validate=_validator(questions),
            system=_GUESS_SYSTEM,
            max_tokens=600,
        )
    except Exception as exc:
        logger.error("guess_short failed: %s", exc)
        return {}
    return answers or {}


def fit_limit(text: str, max_chars: int) -> str:
    """text cut to max_chars at the last sentence end (else the last word) inside it."""
    text = re.sub(r"\s+\n", "\n", (text or "").strip())
    if not max_chars or len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "), cut.rfind(".\n"))
    if end >= max_chars // 3:
        return cut[:end + 1].strip()
    if cut.endswith((".", "!", "?")):
        return cut.strip()
    return cut[:cut.rfind(" ")].rstrip(",;:").strip() if " " in cut else cut


def _length_rule(max_chars: int, short: bool) -> str:
    if short:
        limit = min(max_chars or 250, 250)
        return f"Length: one or two sentences, at most {limit} characters."
    if max_chars and max_chars < 900:
        return f"Length: at most {int(max_chars * 0.9)} characters (hard limit {max_chars})."
    return "Length: 80-150 words."


def draft_long(label: str, job_text: str, profile: str, max_chars: int = 0, short: bool = False,
               allow_unknown: bool = False) -> str:
    """allow_unknown: the label may be a personal fact in a text box ("High School Name") —
    the model may answer UNKNOWN. Otherwise it must write an answer from the resume."""
    rule = _UNKNOWN_RULE if allow_unknown else _ALWAYS_ANSWER
    question, _qflags = untrusted.guard(label, "Question from the form")
    job_block, _jflags = untrusted.guard(job_text[:6000], "Job")
    user = (
        f"{question}\n{_length_rule(max_chars, short)}\n{rule}\n\n{job_block}\n\n"
        f"Candidate profile and resume:\n{profile}"
    )
    try:
        resp = llm_client.complete(
            messages=[{"role": "user", "content": user}],
            task="form_free_text",
            system=_DRAFT_SYSTEM,
            max_tokens=200 if short else 500,
        )
    except Exception as exc:
        logger.error("draft_long(%r) failed: %s", label, exc)
        return ""
    if getattr(resp, "stop_reason", "") == "error":
        return ""
    text = re.sub(r"^\s*(?:answer|response)\s*:\s*", "", (resp.text or "").strip(), flags=re.I).strip().strip('"')
    if text.upper().strip(". ") == UNKNOWN or _REFUSAL.match(text):
        return UNKNOWN
    return fit_limit(text, max_chars or (250 if short else 0))
