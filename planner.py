"""
planner.py — Aria agentic planner.

Detects multi-step goals, generates a plan, confirms with user, and executes
each step with context passing, status narration, and retry logic.

Public API:
  is_multi_step(command: str) -> bool
  run(goal, speaker, voice_capture, transcriber, handle_intent_fn) -> str | None
"""
from __future__ import annotations

import json
import logging
import re

from groq import Groq

import config
import memory
from plan_context import PlanContext

logger = logging.getLogger(__name__)

_CLIENT: Groq | None = None

_KNOWN_INTENT_TYPES = frozenset({
    "browser_task", "knowledge", "web_search", "jobs",
    "apply", "app_control", "navigate", "media",
})

_MAX_STEPS = 5
_MIN_STEPS = 2
_MAX_RETRIES = 3

_YES_WORDS = frozenset({"yes", "yeah", "yep", "sure", "go ahead", "do it",
                         "proceed", "ok", "okay", "go", "sounds good"})
_STOP_WORDS = frozenset({"stop", "cancel", "abort", "no", "nope", "don't",
                          "forget it", "never mind"})

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

_CONJUNCTION_RE = re.compile(
    r"\b("
    r"and\s+then"
    r"|after\s+that"
    r"|and\s+also"
    r"|and\s+add"
    r"|and\s+book"
    r"|and\s+send"
    r"|and\s+message"
    r"|then\s+(?:also\s+)?(?:add|book|send|open|search|find|check|play|message|buy|order)"
    r")\b",
    re.IGNORECASE,
)

_ACTION_VERB_RE = re.compile(
    r"\b(search|find|book|add|send|open|check|play|message|buy|order|"
    r"look\s+up|navigate|go\s+to|apply|compare|calculate|remind|schedule|get)\b",
    re.IGNORECASE,
)


def is_multi_step(command: str) -> bool:
    """
    Two-layer multi-step detection.
    Layer 1: regex — conjunction present AND ≥2 distinct action verbs found.
    Layer 2: conjunction present but only 1 verb — ask Groq cheaply.
    Returns False on any error (safe fallback to single-intent routing).
    """
    if not _CONJUNCTION_RE.search(command):
        return False
    verbs = {m.group(1).lower().replace(" ", "_")
             for m in _ACTION_VERB_RE.finditer(command)}
    if len(verbs) >= 2:
        return True
    return _groq_is_multi(command)


def _groq_is_multi(command: str) -> bool:
    """One cheap Groq call. Returns False on any error."""
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system",
                 "content": "Answer with exactly one word: 'single' or 'multi'."},
                {"role": "user",
                 "content": f"Single task or multiple sequential tasks?\n\n{command}"},
            ],
            temperature=0.0,
            max_tokens=5,
        )
        return "multi" in resp.choices[0].message.content.strip().lower()
    except Exception as exc:
        logger.error("_groq_is_multi failed: %s", exc)
        return False


def _validate_steps(steps) -> list[dict] | None:
    """Return steps if valid, None if any constraint fails."""
    if not isinstance(steps, list):
        return None
    if not (_MIN_STEPS <= len(steps) <= _MAX_STEPS):
        logger.warning("Plan has %d steps (need %d-%d) — rejecting",
                       len(steps), _MIN_STEPS, _MAX_STEPS)
        return None
    required_fields = ("id", "description", "intent_type", "params", "result_key", "depends_on")
    for step in steps:
        if step.get("intent_type") not in _KNOWN_INTENT_TYPES:
            logger.warning("Unknown intent_type %r — rejecting plan", step.get("intent_type"))
            return None
        for field in required_fields:
            if field not in step:
                logger.warning("Step missing field %r — rejecting plan", field)
                return None
    return steps


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set in .env")
        _CLIENT = Groq(api_key=config.GROQ_API_KEY)
    return _CLIENT


# ---------------------------------------------------------------------------
# Plan generation
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = (
    "You are a task decomposition assistant for Aria, a Mac voice agent.\n"
    "Break the user's goal into 2–5 sequential steps.\n\n"
    "Each step MUST use one of these intent_type values exactly:\n"
    "browser_task, knowledge, web_search, jobs, apply, app_control, navigate, media\n\n"
    "Return ONLY a JSON array. No markdown. No explanation.\n\n"
    "Example:\n"
    '[\n'
    '  {"id": 1, "description": "Search Kayak for cheapest flight NYC April 25",\n'
    '   "intent_type": "browser_task",\n'
    '   "params": {"browser_goal": "find cheapest flight to NYC on April 25 on Kayak"},\n'
    '   "result_key": "flight_result", "depends_on": []},\n'
    '  {"id": 2, "description": "Book the flight found in step 1",\n'
    '   "intent_type": "browser_task",\n'
    '   "params": {"browser_goal": "book the flight: {{flight_result}}"},\n'
    '   "result_key": "booking_result", "depends_on": [1]}\n'
    ']\n\n'
    "Rules:\n"
    "- 2 steps minimum, 5 maximum\n"
    "- intent_type must be exactly one of the allowed values\n"
    "- result_key must be a unique snake_case identifier\n"
    "- depends_on lists step IDs whose results this step needs\n"
    "- params key depends on intent: browser_task→browser_goal, "
    "web_search/knowledge/jobs/media→query, navigate→site_name, app_control→query"
)

_REVISE_SYSTEM = (
    "You are revising an Aria task plan based on user feedback.\n"
    "Return the complete revised plan as a JSON array with the same format.\n"
    "Only change the steps the user asked to change. Keep all other steps identical.\n"
    "Allowed intent_type values: "
    "browser_task, knowledge, web_search, jobs, apply, app_control, navigate, media\n"
    "Return ONLY a JSON array. No markdown. No explanation."
)


def _strip_fences(raw: str) -> str:
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    return re.sub(r"\s*```$", "", raw.strip())


def generate_plan(goal: str) -> list[dict] | None:
    """
    Call Groq to decompose goal into steps.
    Returns validated steps or None on any failure.
    """
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _PLAN_SYSTEM},
                {"role": "user", "content": goal},
            ],
            temperature=0.1,
            max_tokens=700,
        )
        raw = _strip_fences(resp.choices[0].message.content)
        return _validate_steps(json.loads(raw))
    except Exception as exc:
        logger.error("generate_plan failed: %s", exc)
        return None


def revise_plan(steps: list[dict], feedback: str) -> list[dict] | None:
    """
    Revise plan based on user voice feedback. One revision allowed before execution.
    Returns revised steps or None on failure (caller keeps original).
    """
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _REVISE_SYSTEM},
                {"role": "user", "content": (
                    f"Current plan:\n{json.dumps(steps, indent=2)}\n\n"
                    f"User feedback: {feedback}"
                )},
            ],
            temperature=0.1,
            max_tokens=700,
        )
        raw = _strip_fences(resp.choices[0].message.content)
        return _validate_steps(json.loads(raw))
    except Exception as exc:
        logger.error("revise_plan failed: %s", exc)
        return None
