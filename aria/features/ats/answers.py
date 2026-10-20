"""features/ats/answers.py — LLM help for form questions the answer bank can't answer.

Both go through GateLLM (aria.llm.llm_client), tiers from aria/llm/tasks.py:
  guess_short(questions, profile)          task form_short_answer (small), one batched call,
                                           validated in code (JSON, answers within options,
                                           length) and escalated once to frontier on failure.
                                           Returns {name: answer} only for answers the
                                           profile supports.
  draft_long(label, job_text, profile)     task form_free_text (frontier), one call per question.

Callers mark the results "llm_guess" / "llm_draft" so the user reviews them.
Never raises — failures return {} / "".
"""
from __future__ import annotations

import json
import logging
import re
from typing import Callable, Optional

from aria.features.answer_bank import choose_option
from aria.llm import llm_client

logger = logging.getLogger(__name__)

UNKNOWN = "UNKNOWN"

_GUESS_SYSTEM = (
    "You fill in short job-application questions for a candidate, using ONLY the candidate "
    "profile you are given. Rules:\n"
    "- If a question lists options, the answer must be exactly one of those options, copied verbatim.\n"
    f"- If the profile does not clearly support an answer, answer {UNKNOWN}. Never guess personal "
    "facts (dates, numbers, names, legal or visa status, criminal history, demographics).\n"
    "- Return a single JSON object mapping each question id to its answer. No prose, no markdown."
)

_DRAFT_SYSTEM = (
    "You write a job candidate's answer to one free-text application question. First person, "
    "plain text, 80-150 words, specific and direct, no greeting or sign-off. Use ONLY facts from "
    "the candidate profile and the job description — never invent projects, employers, numbers "
    "or credentials. If the profile is thin, stay concrete about what it does say."
)


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
    user = f"Candidate profile:\n{profile}\n\nQuestions:\n{json.dumps(listing, indent=1)}"
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


def draft_long(label: str, job_text: str, profile: str) -> str:
    user = (
        f"Question: {label}\n\nJob description:\n{job_text[:6000]}\n\n"
        f"Candidate profile:\n{profile}"
    )
    try:
        resp = llm_client.complete(
            messages=[{"role": "user", "content": user}],
            task="form_free_text",
            system=_DRAFT_SYSTEM,
            max_tokens=450,
        )
    except Exception as exc:
        logger.error("draft_long(%r) failed: %s", label, exc)
        return ""
    if getattr(resp, "stop_reason", "") == "error":
        return ""
    return (resp.text or "").strip()
