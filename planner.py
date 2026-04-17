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


# ---------------------------------------------------------------------------
# Context injection
# ---------------------------------------------------------------------------

_INJECT_SYSTEM = (
    "You are updating a task step's params by incorporating results from previous steps.\n"
    "Return ONLY the updated params dict as a JSON object. No markdown. No explanation.\n"
    "Replace any {{result_key}} placeholders and enrich params with relevant previous results."
)

_FAILURE_PHRASES = (
    "got stuck", "couldn't complete", "ran into an error",
    "error while", "i ran out", "couldn't find", "i got stuck",
    "couldn't get", "i couldn't", "browser got stuck",
)


def _is_failure(result: str | None) -> bool:
    """Return True if result indicates a failure or is empty."""
    if not result:
        return True
    lower = result.lower()
    return any(p in lower for p in _FAILURE_PHRASES)


def _substitute_placeholders(params: dict, results: dict[str, str]) -> dict:
    """
    Pure string substitution: replace {{result_key}} with actual values.
    No Groq call. Safe to test without mocking.
    """
    if not results:
        return params
    params_str = json.dumps(params)
    for key, value in results.items():
        params_str = params_str.replace(f"{{{{{key}}}}}", value)
    try:
        return json.loads(params_str)
    except Exception:
        return params


def _inject_context(step: dict, results: dict[str, str]) -> dict:
    """
    Inject context from previous steps into this step's params.
    Phase 1: simple {{placeholder}} substitution (no Groq).
    Phase 2: Groq call to further enrich params with relevant context.
    Falls back gracefully on any error.
    """
    if not results:
        return step
    substituted_params = _substitute_placeholders(step["params"], results)
    context_summary = "\n".join(f"{k}: {v}" for k, v in results.items())
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _INJECT_SYSTEM},
                {"role": "user", "content": (
                    f"Step description: {step['description']}\n"
                    f"Current params: {json.dumps(substituted_params)}\n"
                    f"Previous step results:\n{context_summary}\n\n"
                    "Update params to incorporate the most relevant previous results."
                )},
            ],
            temperature=0.0,
            max_tokens=250,
        )
        raw = _strip_fences(resp.choices[0].message.content)
        enriched = json.loads(raw)
        if isinstance(enriched, dict):
            return {**step, "params": enriched}
    except Exception as exc:
        logger.warning("_inject_context Groq call failed: %s — using substituted params", exc)
    return {**step, "params": substituted_params}


def _step_to_intent(step: dict) -> dict:
    """Convert a plan step dict into an Aria routing intent dict."""
    params = step.get("params", {})
    return {
        "type": step["intent_type"],
        "query": params.get("query") or step["description"],
        "url": params.get("url", ""),
        "instructions": "",
        "app_name": params.get("app_name", ""),
        "contact": "",
        "site_name": params.get("site_name", ""),
        "browser_goal": params.get("browser_goal", step["description"]),
    }


# ---------------------------------------------------------------------------
# Retry — alternative approach generation
# ---------------------------------------------------------------------------

_RETRY_SYSTEM = (
    "You are regenerating a failed Aria task step with a meaningfully different approach.\n"
    "Return ONLY a single JSON step object (not an array).\n"
    "Allowed intent_type values: "
    "browser_task, knowledge, web_search, jobs, apply, app_control, navigate, media\n"
    "Try a different site, different phrasing, or different method — not a blind repeat."
)


def _generate_retry_step(step: dict, failure_reason: str, attempt: int) -> dict | None:
    """
    Generate a new variant of step with a different approach.
    Preserves id, result_key, depends_on from the original step.
    Returns None on failure (caller retries with original params).
    """
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _RETRY_SYSTEM},
                {"role": "user", "content": (
                    f"Failed step (attempt {attempt}):\n{json.dumps(step, indent=2)}\n\n"
                    f"Failure reason: {failure_reason}\n\n"
                    "Generate a different approach for the same goal."
                )},
            ],
            temperature=0.3,
            max_tokens=250,
        )
        raw = _strip_fences(resp.choices[0].message.content)
        revised = json.loads(raw)
        if not isinstance(revised, dict):
            return None
        if revised.get("intent_type") not in _KNOWN_INTENT_TYPES:
            return None
        return {
            **revised,
            "id": step["id"],
            "result_key": step["result_key"],
            "depends_on": step.get("depends_on", []),
        }
    except Exception as exc:
        logger.error("_generate_retry_step failed: %s", exc)
        return None
