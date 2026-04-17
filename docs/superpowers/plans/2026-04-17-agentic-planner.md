# Agentic Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a planner layer to Aria that detects multi-step voice goals, generates a sequential plan, confirms it with the user, and executes each step with context passing, status narration, and retry logic.

**Architecture:** Two new files (`plan_context.py`, `planner.py`), minor additions to `memory.py` and `main.py`. Planner intercepts multi-step commands before routing and drives `_handle_intent()` per step. All existing handlers untouched. Circular import avoided by passing `_handle_intent` as a callable.

**Tech Stack:** Python 3.11+, Groq (`llama-3.3-70b-versatile` for plan generation, `llama-3.1-8b-instant` for detection/injection), `dataclasses`, `unittest.mock` for tests.

---

## File Map

| File | Change |
|---|---|
| `plan_context.py` | CREATE — PlanContext dataclass |
| `planner.py` | CREATE — detection, generation, context injection, execution |
| `memory.py` | MODIFY — add `store_last_plan()` / `get_last_plan()` |
| `main.py` | MODIFY — planner check before routing |
| `tests/test_planner.py` | CREATE — unit tests for pure functions |

---

## Task 1: `plan_context.py` and `memory.py` additions

**Files:**
- Create: `plan_context.py`
- Modify: `memory.py`
- Test: `tests/test_planner.py` (partial — memory round-trip)

- [ ] **Step 1: Write the failing test for PlanContext and memory round-trip**

```python
# tests/test_planner.py
import json
import pytest
from plan_context import PlanContext


def test_plan_context_defaults():
    ctx = PlanContext(
        goal="find flights and book",
        steps=[{"id": 1, "description": "find", "intent_type": "browser_task",
                "params": {}, "result_key": "r1", "depends_on": []}],
    )
    assert ctx.results == {}
    assert ctx.current_step == 0
    assert ctx.retry_count == 0


def test_plan_context_to_dict():
    ctx = PlanContext(
        goal="test",
        steps=[],
        results={"r1": "some result"},
        current_step=2,
        retry_count=1,
    )
    d = ctx.to_dict()
    assert d["goal"] == "test"
    assert d["results"] == {"r1": "some result"}
    assert d["current_step"] == 2


def test_memory_store_and_get_last_plan(tmp_path, monkeypatch):
    import db, memory
    # Point db at a temp file so test doesn't pollute real aria.db
    monkeypatch.setattr(db, "_DB_PATH", str(tmp_path / "test.db"))
    db._conn = None  # force reconnect to new path
    db._init_db()

    ctx = PlanContext(goal="g", steps=[], results={"k": "v"}, current_step=1, retry_count=0)
    memory.store_last_plan(ctx.to_dict())
    loaded = memory.get_last_plan()
    assert loaded is not None
    assert loaded["goal"] == "g"
    assert loaded["results"] == {"k": "v"}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd /Users/bhanuteja/Documents/trae_projects/Aria
python3 -m pytest tests/test_planner.py::test_plan_context_defaults -v
```

Expected: `ERROR` — `plan_context` module not found.

- [ ] **Step 3: Create `plan_context.py`**

```python
# plan_context.py
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class PlanContext:
    goal: str
    steps: list[dict]
    results: dict[str, str] = field(default_factory=dict)
    current_step: int = 0
    retry_count: int = 0

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "steps": self.steps,
            "results": self.results,
            "current_step": self.current_step,
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlanContext":
        return cls(
            goal=d["goal"],
            steps=d["steps"],
            results=d.get("results", {}),
            current_step=d.get("current_step", 0),
            retry_count=d.get("retry_count", 0),
        )
```

- [ ] **Step 4: Add `store_last_plan` and `get_last_plan` to `memory.py`**

Add at the end of `memory.py` (after `store_cached_jobs`):

```python
# ---------------------------------------------------------------------------
# Last plan (24-hour TTL — for restart recovery)
# ---------------------------------------------------------------------------

def store_last_plan(plan_dict: dict) -> None:
    """Persist the current plan context dict. 24-hour TTL."""
    with _lock:
        session["last_plan"] = plan_dict
    _save("last_plan", plan_dict, expires_hours=24)


def get_last_plan() -> dict | None:
    """Return the last stored plan dict, or None if not set / expired."""
    with _lock:
        return session.get("last_plan")
```

- [ ] **Step 5: Run all three tests**

```bash
python3 -m pytest tests/test_planner.py -v
```

Expected: all 3 pass. (The `db` monkeypatch test may need `db._init_db` to exist — if it fails with AttributeError, replace `db._init_db()` with a direct `db.get_connection()` call to trigger init.)

- [ ] **Step 6: Commit**

```bash
git add plan_context.py memory.py tests/test_planner.py
git commit -m "feat: add PlanContext dataclass and memory store_last_plan/get_last_plan"
```

---

## Task 2: `planner.py` — Detection Layer

**Files:**
- Create: `planner.py` (detection section only)
- Test: `tests/test_planner.py` (append)

- [ ] **Step 1: Append detection tests to `tests/test_planner.py`**

```python
# append to tests/test_planner.py
from unittest.mock import patch
import planner


def test_is_multi_step_conjunction_two_verbs():
    assert planner.is_multi_step("find the cheapest flight and then book it") is True


def test_is_multi_step_and_add():
    assert planner.is_multi_step("search for flights and add the best one to my calendar") is True


def test_is_multi_step_single_action():
    assert planner.is_multi_step("what is the weather today") is False


def test_is_multi_step_no_conjunction():
    assert planner.is_multi_step("search for jobs in New York") is False


def test_is_multi_step_borderline_calls_groq():
    # One conjunction, one verb — should call _groq_is_multi
    with patch("planner._groq_is_multi", return_value=True) as mock:
        result = planner.is_multi_step("find flights and depart tomorrow")
        mock.assert_called_once()
        assert result is True


def test_validate_steps_valid():
    steps = [
        {"id": 1, "description": "d1", "intent_type": "browser_task",
         "params": {"browser_goal": "g"}, "result_key": "r1", "depends_on": []},
        {"id": 2, "description": "d2", "intent_type": "knowledge",
         "params": {"query": "q"}, "result_key": "r2", "depends_on": [1]},
    ]
    assert planner._validate_steps(steps) == steps


def test_validate_steps_too_few():
    steps = [{"id": 1, "description": "d", "intent_type": "browser_task",
              "params": {}, "result_key": "r1", "depends_on": []}]
    assert planner._validate_steps(steps) is None


def test_validate_steps_too_many():
    steps = [{"id": i, "description": "d", "intent_type": "browser_task",
              "params": {}, "result_key": f"r{i}", "depends_on": []}
             for i in range(1, 7)]  # 6 steps
    assert planner._validate_steps(steps) is None


def test_validate_steps_unknown_intent():
    steps = [
        {"id": 1, "description": "d1", "intent_type": "browser_task",
         "params": {}, "result_key": "r1", "depends_on": []},
        {"id": 2, "description": "d2", "intent_type": "INVALID_TYPE",
         "params": {}, "result_key": "r2", "depends_on": []},
    ]
    assert planner._validate_steps(steps) is None


def test_validate_steps_missing_field():
    steps = [
        {"id": 1, "description": "d1", "intent_type": "browser_task",
         "params": {}, "depends_on": []},  # missing result_key
        {"id": 2, "description": "d2", "intent_type": "knowledge",
         "params": {}, "result_key": "r2", "depends_on": []},
    ]
    assert planner._validate_steps(steps) is None
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
python3 -m pytest tests/test_planner.py -k "multi_step or validate" -v
```

Expected: `ERROR` — `planner` module not found.

- [ ] **Step 3: Create `planner.py` with detection layer**

```python
# planner.py
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
        logger.warning("Plan has %d steps (need %d–%d) — rejecting",
                       len(steps), _MIN_STEPS, _MAX_STEPS)
        return None
    required_fields = ("id", "description", "intent_type", "params", "result_key")
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
```

- [ ] **Step 4: Run detection tests**

```bash
python3 -m pytest tests/test_planner.py -k "multi_step or validate" -v
```

Expected: all 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add planner.py tests/test_planner.py
git commit -m "feat: add planner detection layer (is_multi_step, _validate_steps)"
```

---

## Task 3: `planner.py` — Generation Layer

**Files:**
- Modify: `planner.py` (append generation section)
- Test: `tests/test_planner.py` (append — mocked Groq)

- [ ] **Step 1: Append generation tests**

```python
# append to tests/test_planner.py
import json
from unittest.mock import MagicMock, patch


def _make_groq_response(content: str):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


_VALID_STEPS_JSON = json.dumps([
    {"id": 1, "description": "Search Kayak for flights",
     "intent_type": "browser_task",
     "params": {"browser_goal": "find cheap flights NYC"},
     "result_key": "flight_result", "depends_on": []},
    {"id": 2, "description": "Book the flight",
     "intent_type": "browser_task",
     "params": {"browser_goal": "book flight: {{flight_result}}"},
     "result_key": "booking_result", "depends_on": [1]},
])


def test_generate_plan_valid(monkeypatch):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_groq_response(_VALID_STEPS_JSON)
    with patch("planner._get_client", return_value=mock_client):
        result = planner.generate_plan("find flights and book cheapest")
    assert result is not None
    assert len(result) == 2
    assert result[0]["intent_type"] == "browser_task"


def test_generate_plan_returns_none_on_invalid_json(monkeypatch):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_groq_response("not json at all")
    with patch("planner._get_client", return_value=mock_client):
        result = planner.generate_plan("whatever")
    assert result is None


def test_generate_plan_returns_none_on_bad_intent(monkeypatch):
    bad = json.dumps([
        {"id": 1, "description": "d", "intent_type": "BOGUS",
         "params": {}, "result_key": "r1", "depends_on": []},
        {"id": 2, "description": "d", "intent_type": "browser_task",
         "params": {}, "result_key": "r2", "depends_on": []},
    ])
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_groq_response(bad)
    with patch("planner._get_client", return_value=mock_client):
        result = planner.generate_plan("whatever")
    assert result is None


def test_revise_plan_valid(monkeypatch):
    original = json.loads(_VALID_STEPS_JSON)
    revised_json = json.dumps([
        {"id": 1, "description": "Search Google Flights",
         "intent_type": "browser_task",
         "params": {"browser_goal": "find cheap flights NYC on Google Flights"},
         "result_key": "flight_result", "depends_on": []},
        {"id": 2, "description": "Book the flight",
         "intent_type": "browser_task",
         "params": {"browser_goal": "book flight: {{flight_result}}"},
         "result_key": "booking_result", "depends_on": [1]},
    ])
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_groq_response(revised_json)
    with patch("planner._get_client", return_value=mock_client):
        result = planner.revise_plan(original, "use Google Flights instead of Kayak")
    assert result is not None
    assert "Google Flights" in result[0]["description"]
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
python3 -m pytest tests/test_planner.py -k "generate or revise" -v
```

Expected: `AttributeError` — `planner` has no attribute `generate_plan`.

- [ ] **Step 3: Append generation layer to `planner.py`**

Add after `_get_client()`:

```python
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
```

- [ ] **Step 4: Run generation tests**

```bash
python3 -m pytest tests/test_planner.py -k "generate or revise" -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add planner.py tests/test_planner.py
git commit -m "feat: add planner generation layer (generate_plan, revise_plan)"
```

---

## Task 4: `planner.py` — Context & Retry Layer

**Files:**
- Modify: `planner.py` (append)
- Test: `tests/test_planner.py` (append)

- [ ] **Step 1: Append context + retry tests**

```python
# append to tests/test_planner.py


def test_substitute_placeholders_replaces_key():
    params = {"browser_goal": "book flight: {{flight_result}}"}
    results = {"flight_result": "Delta $320 8am Apr 25"}
    out = planner._substitute_placeholders(params, results)
    assert out["browser_goal"] == "book flight: Delta $320 8am Apr 25"


def test_substitute_placeholders_no_match():
    params = {"browser_goal": "find hotels in NYC"}
    results = {"flight_result": "Delta $320"}
    out = planner._substitute_placeholders(params, results)
    assert out == params


def test_substitute_placeholders_empty_results():
    params = {"query": "weather NYC"}
    out = planner._substitute_placeholders(params, {})
    assert out == params


def test_step_to_intent_browser_task():
    step = {
        "id": 1, "description": "search Kayak",
        "intent_type": "browser_task",
        "params": {"browser_goal": "find flights NYC"},
        "result_key": "r1", "depends_on": [],
    }
    intent = planner._step_to_intent(step)
    assert intent["type"] == "browser_task"
    assert intent["browser_goal"] == "find flights NYC"
    assert intent["url"] == ""
    assert intent["app_name"] == ""


def test_step_to_intent_knowledge():
    step = {
        "id": 1, "description": "what is the capital of France",
        "intent_type": "knowledge",
        "params": {"query": "capital of France"},
        "result_key": "r1", "depends_on": [],
    }
    intent = planner._step_to_intent(step)
    assert intent["type"] == "knowledge"
    assert intent["query"] == "capital of France"


def test_is_failure_empty():
    assert planner._is_failure(None) is True
    assert planner._is_failure("") is True


def test_is_failure_stuck_phrase():
    assert planner._is_failure("I got stuck and couldn't complete that.") is True
    assert planner._is_failure("I ran into an error while researching.") is True


def test_is_failure_success():
    assert planner._is_failure("The cheapest flight is Delta at $320.") is False
    assert planner._is_failure("Done. Added to your calendar.") is False
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
python3 -m pytest tests/test_planner.py -k "substitute or step_to_intent or is_failure" -v
```

Expected: `AttributeError` — functions not yet defined.

- [ ] **Step 3: Append context + retry layer to `planner.py`**

```python
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
    Preserves id and result_key from the original step.
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
```

- [ ] **Step 4: Run context + retry tests**

```bash
python3 -m pytest tests/test_planner.py -k "substitute or step_to_intent or is_failure" -v
```

Expected: all 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add planner.py tests/test_planner.py
git commit -m "feat: add planner context injection and retry layer"
```

---

## Task 5: `planner.py` — Execution Layer + `run()`

**Files:**
- Modify: `planner.py` (append execution + entry point)
- No new tests (execution loop depends on voice I/O — covered by manual test in Task 6)

- [ ] **Step 1: Append execution helpers + `execute_plan` + `run` to `planner.py`**

```python
# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def _speak_plan(steps: list[dict], speaker) -> None:
    """Read the plan aloud."""
    parts = [f"Step {s['id']}: {s['description']}" for s in steps]
    speaker.say("Here's my plan: " + ". ".join(parts) + ". Should I go ahead?")


def _listen_reply(voice_capture, transcriber, max_seconds: int = 8) -> str:
    """Record and transcribe a short user reply. Returns '' on error."""
    wav = voice_capture.record_once(max_seconds=max_seconds)
    if wav is None:
        return ""
    try:
        return transcriber.transcribe(wav).strip()
    except Exception as exc:
        logger.error("_listen_reply transcription failed: %s", exc)
        return ""


def _save_ctx(ctx: PlanContext) -> None:
    """Persist plan context for restart recovery. Swallows errors."""
    try:
        memory.store_last_plan(ctx.to_dict())
    except Exception as exc:
        logger.warning("_save_ctx failed: %s", exc)


# ---------------------------------------------------------------------------
# Execution loop
# ---------------------------------------------------------------------------

def execute_plan(
    ctx: PlanContext,
    speaker,
    voice_capture,
    transcriber,
    handle_intent_fn,
) -> str:
    """
    Execute all steps in ctx sequentially.
    - Injects context from previous steps into each step's params.
    - Narrates status between steps.
    - Retries up to _MAX_RETRIES times with different approaches on failure.
    - After all retries exhausted: asks user to skip / stop / give alternative.
    Returns final spoken summary string.
    """
    from router import route as _route

    completed: list[str] = []

    for i, step in enumerate(ctx.steps):
        ctx.current_step = step["id"]
        ctx.retry_count = 0
        current_step = _inject_context(step, ctx.results)
        result: str | None = None
        failure_reason = ""
        succeeded = False

        for attempt in range(1, _MAX_RETRIES + 1):
            ctx.retry_count = attempt
            _save_ctx(ctx)
            try:
                intent = _step_to_intent(current_step)
                result = handle_intent_fn(intent, current_step["description"])
                if _is_failure(result):
                    failure_reason = result or "empty result"
                    logger.warning("Step %d attempt %d failed: %s",
                                   step["id"], attempt, failure_reason)
                    if attempt < _MAX_RETRIES:
                        retry_step = _generate_retry_step(current_step, failure_reason, attempt)
                        if retry_step:
                            current_step = _inject_context(retry_step, ctx.results)
                    continue
                succeeded = True
                break
            except Exception as exc:
                failure_reason = str(exc)
                logger.error("Step %d attempt %d exception: %s", step["id"], attempt, exc)
                if attempt < _MAX_RETRIES:
                    retry_step = _generate_retry_step(current_step, failure_reason, attempt)
                    if retry_step:
                        current_step = _inject_context(retry_step, ctx.results)

        if not succeeded:
            speaker.say(
                f"I tried {_MAX_RETRIES} approaches and got stuck on step {step['id']}: "
                f"{step['description']}. "
                "Say skip to move on, stop to abort, or give me a different instruction."
            )
            reply = _listen_reply(voice_capture, transcriber, max_seconds=8).lower()

            if any(w in reply for w in ("stop", "abort")):
                done_str = (", ".join(completed)) if completed else "nothing"
                return f"Stopped. Completed so far: {done_str}."

            if any(w in reply for w in ("skip", "next", "move on", "forget it")):
                speaker.say(f"Skipping step {step['id']}.")
                continue

            if reply:
                speaker.say("Got it, trying that instead.")
                try:
                    alt_intent = _route(reply)
                    result = handle_intent_fn(alt_intent, reply)
                    if result and not _is_failure(result):
                        succeeded = True
                except Exception as exc:
                    logger.error("Alternative instruction failed: %s", exc)

            if not succeeded:
                done_str = (", ".join(completed)) if completed else "nothing"
                return (
                    f"I got stuck on step {step['id']} and couldn't recover. "
                    f"Completed: {done_str}."
                )

        ctx.results[step["result_key"]] = result or ""
        completed.append(step["description"])
        _save_ctx(ctx)

        if i < len(ctx.steps) - 1:
            next_step = ctx.steps[i + 1]
            speaker.say(
                f"Done. Moving to step {next_step['id']}: {next_step['description']}."
            )

    all_results = " ".join(v for v in ctx.results.values() if v)
    return f"All done. {all_results}" if all_results else "All steps completed."


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    goal: str,
    speaker,
    voice_capture,
    transcriber,
    handle_intent_fn,
) -> str | None:
    """
    Full planner flow: generate → confirm (with optional one revision) → execute.

    Returns:
      str  — final spoken result to say to the user
      None — plan generation failed or user declined; caller should fall back
             to single-intent routing
    """
    try:
        steps = generate_plan(goal)
        if steps is None:
            logger.warning("Plan generation failed for %r — caller should fall back", goal)
            return None

        _speak_plan(steps, speaker)
        reply = _listen_reply(voice_capture, transcriber)

        if any(w in reply.lower() for w in _STOP_WORDS):
            return "Got it, I won't proceed with that."

        if not any(w in reply.lower() for w in _YES_WORDS):
            # Treat as revision request (allowed once)
            revised = revise_plan(steps, reply)
            if revised:
                steps = revised
            _speak_plan(steps, speaker)
            reply2 = _listen_reply(voice_capture, transcriber)
            if not any(w in reply2.lower() for w in _YES_WORDS):
                return "Got it, let me know when you're ready."

        ctx = PlanContext(goal=goal, steps=steps)
        return execute_plan(ctx, speaker, voice_capture, transcriber, handle_intent_fn)

    except Exception as exc:
        logger.error("planner.run failed: %s", exc)
        return None
```

- [ ] **Step 2: Syntax check**

```bash
python3 -c "import ast; ast.parse(open('planner.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add planner.py
git commit -m "feat: add planner execution loop and run() entry point"
```

---

## Task 6: `main.py` Integration

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add `import planner` to top-level imports in `main.py`**

In `main.py`, after the `import agent_browser` / `import computer_use` lines:

```python
import planner
```

- [ ] **Step 2: Add multi-step check in `handle_command()` before `route()` call**

In `main.py`, find this block inside `handle_command()`:

```python
        intent = route(transcript)
        print(f"[Aria] Intent: type={intent['type']!r}  query={intent['query']!r}")
        answer = _handle_intent(intent, transcript)
```

Replace with:

```python
        if planner.is_multi_step(transcript):
            print(f"[Aria] Multi-step detected: {transcript!r}")
            answer = planner.run(
                goal=transcript,
                speaker=speaker,
                voice_capture=voice_capture,
                transcriber=transcriber_instance,
                handle_intent_fn=_handle_intent,
            )
            if answer is None:
                # Plan generation failed — fall back to single-intent
                logger.warning("Planner returned None — falling back to single-intent routing")
                intent = route(transcript)
                print(f"[Aria] Intent (fallback): type={intent['type']!r}  query={intent['query']!r}")
                answer = _handle_intent(intent, transcript)
        else:
            intent = route(transcript)
            print(f"[Aria] Intent: type={intent['type']!r}  query={intent['query']!r}")
            answer = _handle_intent(intent, transcript)
```

- [ ] **Step 3: Syntax check `main.py`**

```bash
python3 -c "import ast; ast.parse(open('main.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Run full test suite**

```bash
python3 -m pytest tests/test_planner.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Manual smoke test — single-intent command (regression check)**

Start Aria, say: *"What's the weather in New York?"*

Expected: routes normally via `web_search`, speaks weather result. No planner involved. Confirm in logs: no "Multi-step detected" line.

- [ ] **Step 6: Manual smoke test — multi-step command**

Say: *"Search for SWE jobs in New York and then open LinkedIn"*

Expected:
1. Aria says "Here's my plan: Step 1: search jobs. Step 2: open LinkedIn. Should I go ahead?"
2. You say "go ahead"
3. Aria executes step 1, narrates "Done. Moving to step 2..."
4. Aria executes step 2
5. Aria says "All done."

- [ ] **Step 7: Commit**

```bash
git add main.py
git commit -m "feat: wire planner into main.py handle_command with single-intent fallback"
```

---

## Self-Review

**Spec coverage check:**
- ✅ `is_multi_step` two-layer detection
- ✅ Plan generation (Groq, 2–5 steps, known intent types)
- ✅ Plan confirmation (speak plan, listen for yes/revision/stop)
- ✅ One revision before execution
- ✅ Context passing via `_inject_context` + `_substitute_placeholders`
- ✅ Ambiguity rule: handled inside `research_loop`/`_handle_intent` (the `confirm` checkpoint) — the planner auto-advances on non-ambiguous results, and the existing confirm mechanism handles ambiguous ones
- ✅ Status narration between steps
- ✅ Retry up to 3 times with Groq-generated alternatives
- ✅ After 3 failures: voice checkpoint (skip/stop/alternative)
- ✅ No rollback on stop — speak completed steps, return
- ✅ Context injection failure → falls back to substituted params
- ✅ Plan generation failure → falls back to single-intent routing
- ✅ Unknown intent_type → plan rejected, fallback
- ✅ `memory.store_last_plan` / `get_last_plan` for restart recovery
- ✅ `plan_context.to_dict` / `from_dict` for persistence

**Placeholder scan:** None found.

**Type consistency:**
- `PlanContext.to_dict()` → `dict` ✅ used in `_save_ctx` → `memory.store_last_plan(dict)` ✅
- `_step_to_intent(step: dict) -> dict` ✅ called in `execute_plan` ✅
- `planner.run(...) -> str | None` ✅ handled in `main.py` with None fallback ✅
- `execute_plan(..., handle_intent_fn)` ✅ called with `_handle_intent` from `main.py` ✅
