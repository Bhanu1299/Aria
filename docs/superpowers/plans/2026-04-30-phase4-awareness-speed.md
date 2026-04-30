# Phase 4 (cont.): Awareness & Speed Upgrades — Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to implement task-by-task.

**Goal:** Extend Aria with memory consolidation, macOS notifications, STT accuracy improvements, follow-up suggestions, and faster transcription — all ported from `claude-code-main/src/services/`.

**Architecture:** Six self-contained tasks (Tasks 9–14 of Phase 4). Tier 1 builds on Phase 4's live memory system. Tier 2 adds awareness and accuracy. Tier 3 cuts hotkey-to-response latency. Each task produces working, testable code before moving to the next.

**Tech Stack:** Python 3.9+, Groq (llama-3.3-70b-versatile), faster-whisper, osascript, numpy — all already in requirements. No new dependencies.

**Python 3.9 rule:** Always add `from __future__ import annotations` at the top of every file touched.

---

## TIER 1 — Memory Maturity

---

### Task 1: Session Notes Compaction (`compact.py`)

**What:** When accumulated session notes exceed 3000 chars, a Groq call compresses them to ~500 chars and `memory.py` writes the compact version back. Keeps memory lean without losing history.

**Files:**
- Create: `compact.py`
- Modify: `memory.py` — call `compact.compress()` inside `store_session_notes()` when threshold crossed
- Test: `tests/test_compact.py`

**Reference:** `claude-code-main/src/services/compact/compactConversation.ts`

**Steps:**
- [ ] Write tests: `test_compress_returns_shorter_string`, `test_compress_graceful_on_groq_failure`, `test_store_session_notes_compacts_when_over_threshold`, `test_store_session_notes_does_not_compact_when_under_threshold`
- [ ] Run tests — confirm they fail
- [ ] Implement `compact.py`:

```python
from __future__ import annotations

import logging
from groq import Groq
import config

logger = logging.getLogger(__name__)

_NOTES_MAX_CHARS = 3000
_CLIENT: Groq | None = None

_SYSTEM_PROMPT = (
    "You are a concise notes compressor for a voice agent called Aria. "
    "Given a running session log, compress it to the most essential facts "
    "in 3-7 bullet points. Keep key data: names, URLs, prices, job titles. "
    "Return ONLY bullet points starting with '- '."
)

def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set")
        _CLIENT = Groq(api_key=config.GROQ_API_KEY)
    return _CLIENT

def needs_compaction(notes: str) -> bool:
    return len(notes) > _NOTES_MAX_CHARS

def compress(notes: str) -> str:
    """Call Groq to compress notes. Returns original on any failure."""
    if not notes.strip():
        return notes
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Compress these session notes:\n\n{notes}"},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        result = response.choices[0].message.content or ""
        result = result.strip()
        logger.debug("compact.compress: %d → %d chars", len(notes), len(result))
        return result if result else notes
    except Exception as exc:
        logger.warning("compact.compress failed: %s", exc)
        return notes
```

- [ ] Modify `memory.py` `store_session_notes()` — add compaction check after building `combined`:

```python
def store_session_notes(notes: str) -> None:
    """Append bullet-point notes for this turn to the running session log. Never expires."""
    import compact  # local import to avoid circular deps at module level
    with _lock:
        existing = session.get(_SESSION_NOTES_KEY, "")
        combined = (existing + "\n\n" + notes).strip() if existing else notes
        if compact.needs_compaction(combined):
            combined = compact.compress(combined)
        session[_SESSION_NOTES_KEY] = combined
    _save(_SESSION_NOTES_KEY, combined, expires_hours=None)
```

- [ ] Run tests — confirm they pass
- [ ] `git commit -m "feat(phase5): session notes compaction"`

---

### Task 2: AutoDream — Background Memory Consolidation (`auto_dream.py`)

**What:** After every 5 commands, fires a background Groq call that reads `session_notes` + `learned_facts` from `identity.json`, consolidates them into cleaner organized versions, and writes both back. Deduplicates facts, trims noise. Counter persisted in SQLite via `memory.py`.

**Files:**
- Create: `auto_dream.py`
- Modify: `memory.py` — add `increment_command_count() -> int`, `reset_command_count()`
- Modify: `main.py` — call `auto_dream.maybe_consolidate_async(transcript, answer)` after `memory_extractor.extract_async()`
- Test: `tests/test_auto_dream.py`

**Reference:** `claude-code-main/src/services/autoDream/autoDream.ts`

**Steps:**
- [ ] Write tests: `test_consolidate_rewrites_session_notes`, `test_consolidate_deduplicates_facts`, `test_maybe_consolidate_fires_at_interval`, `test_maybe_consolidate_does_not_fire_before_interval`, `test_consolidate_graceful_on_groq_failure`
- [ ] Run tests — confirm they fail
- [ ] Add to `memory.py`:

```python
_COMMAND_COUNT_KEY = "command_count"

def increment_command_count() -> int:
    """Increment and return the persistent command counter."""
    with _lock:
        count = session.get(_COMMAND_COUNT_KEY, 0) + 1
        session[_COMMAND_COUNT_KEY] = count
    _save(_COMMAND_COUNT_KEY, count, expires_hours=None)
    return count

def reset_command_count() -> None:
    with _lock:
        session[_COMMAND_COUNT_KEY] = 0
    _save(_COMMAND_COUNT_KEY, 0, expires_hours=None)
```

- [ ] Implement `auto_dream.py`:

```python
from __future__ import annotations

import json
import logging
import os
import threading

from groq import Groq
import config
import memory

logger = logging.getLogger(__name__)

_CONSOLIDATE_EVERY = 5
_IDENTITY_PATH = os.path.join(os.path.dirname(__file__), "identity.json")
_CLIENT: Groq | None = None

_SYSTEM_PROMPT = (
    "You are a memory consolidation assistant for a voice agent called Aria. "
    "Given session notes and known user facts, return a JSON object with two keys: "
    "'session_notes' (3-5 bullet points of the most important recent activity) and "
    "'learned_facts' (deduplicated array of durable user facts, max 50, newest kept). "
    "Return ONLY valid JSON. No markdown."
)

def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set")
        _CLIENT = Groq(api_key=config.GROQ_API_KEY)
    return _CLIENT

def _load_identity() -> dict:
    try:
        with open(_IDENTITY_PATH) as f:
            return json.load(f)
    except Exception:
        return {}

def _save_identity(identity: dict) -> None:
    try:
        with open(_IDENTITY_PATH, "w") as f:
            json.dump(identity, f, indent=2)
    except Exception as exc:
        logger.warning("auto_dream: failed to save identity: %s", exc)

def consolidate() -> None:
    """Synchronous consolidation. Reads notes + facts, rewrites both. Never raises."""
    try:
        notes = memory.get_session_notes()
        identity = _load_identity()
        facts = identity.get("learned_facts", [])

        if not notes and not facts:
            return

        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Session notes:\n{notes}\n\n"
                    f"Known facts:\n{json.dumps(facts)}"
                )},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        raw = (response.choices[0].message.content or "").strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(raw)

        new_notes = result.get("session_notes", "")
        new_facts = result.get("learned_facts", facts)

        if isinstance(new_notes, list):
            new_notes = "\n".join(f"- {n}" if not n.startswith("-") else n for n in new_notes)

        if new_notes:
            memory.clear_session_notes()
            memory.store_session_notes(new_notes)

        if isinstance(new_facts, list):
            identity["learned_facts"] = new_facts[:50]
            _save_identity(identity)

        memory.reset_command_count()
        logger.debug("auto_dream.consolidate: complete")

    except Exception as exc:
        logger.warning("auto_dream.consolidate failed: %s", exc)

def maybe_consolidate_async(transcript: str, answer: str) -> None:
    """Increment counter; if interval reached, consolidate in a daemon thread."""
    count = memory.increment_command_count()
    if count % _CONSOLIDATE_EVERY == 0:
        t = threading.Thread(target=consolidate, daemon=True, name="auto-dream")
        t.start()
```

- [ ] Modify `main.py` — add import and wire after `memory_extractor.extract_async()`:

```python
import auto_dream
# in handle_command(), after memory_extractor.extract_async(transcript, answer):
auto_dream.maybe_consolidate_async(transcript, answer)
```

- [ ] Run tests — confirm they pass
- [ ] `git commit -m "feat(phase5): auto-dream background memory consolidation"`

---

## TIER 2 — Awareness & Accuracy

---

### Task 3: Notifier (`notifier.py`)

**What:** After `research_loop()` returns, if the task took >15 seconds, send a macOS notification via `osascript` with the task summary. Aria already speaks the result; the notification catches you when you've switched windows.

**Files:**
- Create: `notifier.py`
- Modify: `main.py` — wrap `research_loop()` call in `_handle_intent()` with timing + notify
- Test: `tests/test_notifier.py`

**Reference:** `claude-code-main/src/services/notifier.ts`

**Steps:**
- [ ] Write tests: `test_send_notification_calls_osascript`, `test_send_notification_handles_osascript_failure`, `test_notify_after_long_task`, `test_no_notify_after_short_task`
- [ ] Run tests — confirm they fail
- [ ] Implement `notifier.py`:

```python
from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)

_MIN_TASK_SECONDS = 15.0

def send_notification(title: str, body: str) -> None:
    """Send a macOS notification via osascript. Never raises."""
    try:
        script = (
            f'display notification "{body[:200]}" '
            f'with title "{title[:100]}" '
            f'sound name "Glass"'
        )
        subprocess.run(
            ["osascript", "-e", script],
            timeout=5,
            capture_output=True,
        )
        logger.debug("notifier: sent '%s'", title)
    except Exception as exc:
        logger.warning("notifier.send_notification failed: %s", exc)

def notify_if_slow(elapsed: float, goal: str, summary: str) -> None:
    """Send notification only if task took longer than threshold."""
    if elapsed >= _MIN_TASK_SECONDS:
        short_summary = summary[:120] + "..." if len(summary) > 120 else summary
        send_notification(title=f"Aria: {goal[:60]}", body=short_summary)
```

- [ ] Modify `main.py` — wrap `research_loop()` call in `_handle_intent()`:

```python
import time
import notifier

# replace the existing research_loop call block:
try:
    _t0 = time.monotonic()
    answer = computer_use.research_loop(
        goal=goal,
        max_steps=80,
        confirm_fn=_confirm,
        input_fn=_get_input,
        progress_fn=_on_progress,
    )
    notifier.notify_if_slow(time.monotonic() - _t0, goal, answer)
except Exception as exc:
    logger.error("research_loop failed: %s", exc)
    answer = "I ran into an error. Try again with more detail."
```

- [ ] Run tests — confirm they pass
- [ ] `git commit -m "feat(phase5): macOS notification after long browser tasks"`

---

### Task 4: Voice Keyterms — STT Accuracy (`voice_keyterms.py`)

**What:** Build a vocab hint list (static domain terms + dynamic terms from `identity.json`) and pass it as `initial_prompt` to Whisper's `transcribe()`. Whisper uses the prompt as context to bias spelling toward these terms. No model change, no new dependency.

**Files:**
- Create: `voice_keyterms.py`
- Modify: `transcriber.py` — accept `initial_prompt: str` in `transcribe()` and pass to model
- Test: `tests/test_voice_keyterms.py`

**Reference:** `claude-code-main/src/services/voiceKeyterms.ts`

**Steps:**
- [ ] Write tests: `test_build_prompt_includes_static_terms`, `test_build_prompt_includes_identity_skills`, `test_build_prompt_handles_missing_identity`, `test_transcriber_passes_initial_prompt_to_model`
- [ ] Run tests — confirm they fail
- [ ] Implement `voice_keyterms.py`:

```python
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

_IDENTITY_PATH = os.path.join(os.path.dirname(__file__), "identity.json")

_STATIC_TERMS = [
    "LinkedIn", "Indeed", "Playwright", "Groq", "Whisper",
    "Python", "FastAPI", "PostgreSQL", "Docker", "GitHub",
    "LangChain", "RAG", "Pinecone", "FAISS", "Anthropic",
    "browser task", "job search", "apply", "resume",
    "software engineer", "full stack", "GenAI",
]

def _load_identity() -> dict:
    try:
        with open(_IDENTITY_PATH) as f:
            return json.load(f)
    except Exception:
        return {}

def build_prompt() -> str:
    """
    Return a short initial_prompt string for Whisper containing domain vocab.
    Whisper treats this as preceding context, biasing toward these spellings.
    """
    identity = _load_identity()
    dynamic: list[str] = []

    skills = identity.get("skills", [])
    dynamic.extend(skills[:10])

    target_roles = identity.get("target_roles", [])
    dynamic.extend(target_roles[:5])

    name = identity.get("name", "")
    if name:
        dynamic.append(name)

    all_terms = _STATIC_TERMS + [t for t in dynamic if t not in _STATIC_TERMS]
    return ", ".join(all_terms[:40]) + "."
```

- [ ] Modify `transcriber.py` — add `initial_prompt` param and wire it through:

```python
# In Transcriber.transcribe():
def transcribe(self, audio_path: str, initial_prompt: str = "") -> str:
    """Submit audio_path for transcription. Blocks until the result arrives."""
    result_q: queue.Queue[str] = queue.Queue()
    self._request_q.put((audio_path, initial_prompt, result_q))
    try:
        return result_q.get(timeout=60)
    except queue.Empty:
        logger.error("Transcription timed out for: %s", audio_path)
        return ""

# In _worker_main() loop, unpack the extra field:
audio_path, initial_prompt, result_q = item

# Pass to model.transcribe():
segments, _info = self._model.transcribe(
    audio_path,
    beam_size=5,
    language="en",
    initial_prompt=initial_prompt or None,
)
```

- [ ] Modify `main.py` — build keyterms prompt once at startup and pass to transcribe calls:

```python
import voice_keyterms

# near the top of _process_release(), replace:
#   question = transcriber_instance.transcribe(wav_path)
# with:
_KEYTERMS_PROMPT = voice_keyterms.build_prompt()
question = transcriber_instance.transcribe(wav_path, initial_prompt=_KEYTERMS_PROMPT)
```

Note: `_KEYTERMS_PROMPT` should be built once after startup (call `voice_keyterms.build_prompt()` once in `main()` and store as a module-level variable, or just call it inline — it reads a file so it's fast).

- [ ] Run tests — confirm they pass
- [ ] `git commit -m "feat(phase5): domain vocab hints for STT accuracy"`

---

### Task 5: Prompt Suggestions (`prompt_suggester.py`)

**What:** After `speaker.say(answer)` in `handle_command()`, a background Groq call generates one follow-up suggestion based on what Aria just did. Spoken as a soft aside: *"Also — want me to apply to any of those?"* Only activates for `browser_task`, `jobs`, and `web_search` intents. Skips if the answer is an error or fewer than 20 words.

**Files:**
- Create: `prompt_suggester.py`
- Modify: `main.py` — call `prompt_suggester.suggest_async(intent_type, answer, speaker)` after `speaker.say(answer)`
- Test: `tests/test_prompt_suggester.py`

**Reference:** `claude-code-main/src/services/PromptSuggestion/promptSuggestion.ts`

**Steps:**
- [ ] Write tests: `test_suggest_returns_string_for_browser_task`, `test_suggest_returns_empty_for_short_answer`, `test_suggest_returns_empty_for_excluded_intent`, `test_suggest_async_does_not_block`, `test_suggest_graceful_on_groq_failure`
- [ ] Run tests — confirm they fail
- [ ] Implement `prompt_suggester.py`:

```python
from __future__ import annotations

import logging
import threading

from groq import Groq
import config

logger = logging.getLogger(__name__)

_TRIGGER_INTENTS = frozenset({"browser_task", "jobs", "web_search"})
_MIN_ANSWER_WORDS = 20
_CLIENT: Groq | None = None

_SYSTEM_PROMPT = (
    "You are Aria, a sharp voice assistant. "
    "Given what you just did for the user, suggest ONE short follow-up action "
    "they might want next. Return a single spoken sentence starting with 'Also —'. "
    "Keep it under 15 words. If no obvious follow-up exists, return empty string."
)

def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set")
        _CLIENT = Groq(api_key=config.GROQ_API_KEY)
    return _CLIENT

def suggest(intent_type: str, answer: str) -> str:
    """Return a follow-up suggestion string, or '' if none applies. Never raises."""
    if intent_type not in _TRIGGER_INTENTS:
        return ""
    if len(answer.split()) < _MIN_ANSWER_WORDS:
        return ""
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"You just told the user: {answer}"},
            ],
            temperature=0.4,
            max_tokens=60,
        )
        result = (response.choices[0].message.content or "").strip()
        return result if result.lower().startswith("also") else ""
    except Exception as exc:
        logger.warning("prompt_suggester.suggest failed: %s", exc)
        return ""

def _worker(intent_type: str, answer: str, speaker) -> None:
    suggestion = suggest(intent_type, answer)
    if suggestion:
        try:
            speaker.say(suggestion)
        except Exception as exc:
            logger.warning("prompt_suggester: speaker.say failed: %s", exc)

def suggest_async(intent_type: str, answer: str, speaker) -> None:
    """Fire-and-forget: speak a follow-up suggestion in a daemon thread."""
    t = threading.Thread(
        target=_worker,
        args=(intent_type, answer, speaker),
        daemon=True,
        name="prompt-suggester",
    )
    t.start()
```

- [ ] Modify `main.py` — add import and wire after `speaker.say(answer)` in `handle_command()`. The `intent_type` comes from the routed intent. Pass it through from `_handle_intent()` or detect it from the answer context:

```python
import prompt_suggester

# In handle_command(), after speaker.say(answer):
# intent_type is available inside _handle_intent but not in handle_command.
# Simplest fix: return intent_type alongside answer from _handle_intent,
# OR call suggest_async with the last known intent stored in a closure.
# Use a module-level _last_intent_type variable updated in _handle_intent():

_last_intent_type: str = ""

# In _handle_intent(), at the top of each intent branch, set:
#   global _last_intent_type; _last_intent_type = intent_type

# In handle_command(), after speaker.say(answer):
prompt_suggester.suggest_async(_last_intent_type, answer, speaker)
```

- [ ] Run tests — confirm they pass
- [ ] `git commit -m "feat(phase5): follow-up suggestions after browser and job tasks"`

---

## TIER 3 — Real-time Feel

---

### Task 6: Numpy Transcription — Faster STT (`transcriber.py` upgrade)

**What:** Add a `transcribe_numpy(audio: np.ndarray)` method to `Transcriber` that accepts a numpy array directly, bypassing the `sf.write` + disk I/O roundtrip. In `_process_release()`, use `voice_capture.get_audio_array()` + `transcribe_numpy()` instead of writing to disk. Cuts hotkey-to-response latency by ~300–500ms.

Note: faster-whisper's `WhisperModel.transcribe()` natively accepts `np.ndarray` (float32, normalized to [-1, 1]).

**Files:**
- Modify: `voice_capture.py` — add `get_audio_array() -> np.ndarray | None`
- Modify: `transcriber.py` — add `transcribe_numpy(audio: np.ndarray, initial_prompt: str = "") -> str`
- Modify: `main.py` — use numpy path in `_process_release()`
- Test: `tests/test_transcriber_numpy.py`

**Steps:**
- [ ] Write tests: `test_transcribe_numpy_returns_string`, `test_transcribe_numpy_accepts_float32_array`, `test_get_audio_array_returns_normalized_float32`, `test_process_release_uses_numpy_path`
- [ ] Run tests — confirm they fail
- [ ] Add `get_audio_array()` to `VoiceCapture` in `voice_capture.py`:

```python
def get_audio_array(self) -> np.ndarray | None:
    """
    Return recorded audio as a float32 numpy array normalized to [-1, 1].
    Returns None if no audio has been recorded.
    faster-whisper accepts this directly — no disk I/O needed.
    """
    if not self._chunks:
        return None
    audio = np.concatenate(self._chunks, axis=0)
    # Convert int16 → float32, normalize to [-1.0, 1.0]
    return audio.astype(np.float32) / 32768.0
```

- [ ] Add `transcribe_numpy()` to `Transcriber` in `transcriber.py`:

```python
import numpy as np

# New public method:
def transcribe_numpy(self, audio: np.ndarray, initial_prompt: str = "") -> str:
    """Submit a numpy float32 audio array for transcription. Blocks until result."""
    result_q: queue.Queue[str] = queue.Queue()
    self._request_q.put((audio, initial_prompt, result_q))
    try:
        return result_q.get(timeout=60)
    except queue.Empty:
        logger.error("Transcription (numpy) timed out")
        return ""

# In _worker_main(), update the handler to detect numpy vs path:
audio_or_path, initial_prompt, result_q = item
try:
    segments, _info = self._model.transcribe(
        audio_or_path,    # faster-whisper accepts str or np.ndarray
        beam_size=5,
        language="en",
        initial_prompt=initial_prompt or None,
    )
    text = "".join(seg.text for seg in segments).strip()
except Exception as exc:
    logger.error("Transcription error: %s", exc)
    text = ""
result_q.put(text)
```

- [ ] Modify `main.py` `_process_release()` — use numpy path:

```python
# Replace:
#   wav_path = voice_capture.stop_recording()
#   question = transcriber_instance.transcribe(wav_path, initial_prompt=_KEYTERMS_PROMPT)
# With:
audio_array = voice_capture.get_audio_array()
voice_capture.stop_recording()   # still saves WAV for any callers that need the file
if audio_array is not None:
    question = transcriber_instance.transcribe_numpy(audio_array, initial_prompt=_KEYTERMS_PROMPT)
else:
    wav_path = voice_capture.stop_recording()
    question = transcriber_instance.transcribe(wav_path, initial_prompt=_KEYTERMS_PROMPT)
```

Note: `stop_recording()` is still called to clean up the stream and save the WAV (needed by `record_once` callers). `get_audio_array()` is called first while the chunks are still in memory.

Actually the order must be: call `get_audio_array()` BEFORE `stop_recording()` because `stop_recording()` clears `_chunks`. Update to:

```python
audio_array = voice_capture.get_audio_array()   # read chunks before stop clears them
voice_capture.stop_recording()                   # clears chunks, closes stream
question = transcriber_instance.transcribe_numpy(audio_array, initial_prompt=_KEYTERMS_PROMPT)
if not question:
    # fallback to file path
    question = transcriber_instance.transcribe(voice_capture.OUTPUT_PATH, initial_prompt=_KEYTERMS_PROMPT)
```

- [ ] Run tests — confirm they pass
- [ ] Run full suite: `python3 -m pytest tests/ --ignore=tests/test_jobs.py --ignore=tests/test_voice_capture.py --ignore=tests/test_transcriber.py --ignore=tests/test_computer_use_dom.py --ignore=tests/test_dom_snapshot.py -q`
- [ ] `git commit -m "feat(phase5): numpy transcription path — eliminate disk I/O latency"`

---

## Execution Order

```
Task 1 → Task 2    (Tier 1: memory)
Task 3 → Task 4 → Task 5   (Tier 2: awareness + accuracy)
Task 6             (Tier 3: real-time)
```

## Before Starting

- Branch: `git checkout -b feature/phase5-awareness-speed`
- Confirm clean baseline: `python3 -m pytest tests/ --ignore=tests/test_jobs.py --ignore=tests/test_voice_capture.py --ignore=tests/test_transcriber.py --ignore=tests/test_computer_use_dom.py --ignore=tests/test_dom_snapshot.py -q`
- Expected: 117 passed

## Definition of Done (Phase 5)

- Notes never grow unbounded — compact fires at 3000 chars
- AutoDream consolidates memory every 5 commands
- macOS notification appears after any browser task >15s
- Whisper correctly spells LinkedIn, Groq, Python engineer, etc.
- Aria suggests a follow-up after job searches and browser tasks
- Hotkey-to-response latency reduced by ~300–500ms via numpy transcription path
