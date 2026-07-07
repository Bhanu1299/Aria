"""
flight_recorder.py — every command Aria handles, recorded locally.

This is the engine of the weekly reliability loop: use Aria normally, then
review what actually failed and fix the top offenders.

record(transcript, answer, duration, tools, error)  — append one JSONL entry
read_recent(days)                                   — parsed entries, corrupt lines skipped
spoken_report(days)                                 — voice-friendly self-report
is_failure(answer, error)                           — heuristic failure classifier

Log lives at ~/.aria/flight_log.jsonl — local only, never uploaded.
Rotates when it exceeds ~5MB (keeps the newest half).
Never raises.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import Counter

logger = logging.getLogger(__name__)

_LOG_PATH = os.path.join(os.path.expanduser("~"), ".aria", "flight_log.jsonl")
_MAX_BYTES = 5 * 1024 * 1024
_LOCK = threading.Lock()

# Spoken fallbacks that mean "Aria failed the user" — keep in sync with the
# error strings used across agent.py, screen_qa.py, plugins/*.
_FAILURE_MARKERS = (
    "something went wrong",
    "i didn't understand",
    "i couldn't",
    "i can't help with that",
    "i ran into an issue",
    "unavailable",
    "please try again",
    "i'm not sure how to",
    "couldn't find a good result",
)


def is_failure(answer: str, error: str | None) -> bool:
    """Heuristic: did this command fail the user?"""
    if error:
        return True
    if not (answer or "").strip():
        return True
    low = answer.lower()
    return any(marker in low for marker in _FAILURE_MARKERS)


def record(transcript: str, answer: str, duration: float,
           tools: list | None = None, error: str | None = None) -> None:
    """Append one command entry. Never raises."""
    try:
        entry = {
            "ts": time.time(),
            "transcript": (transcript or "")[:500],
            "answer": (answer or "")[:500],
            "duration": round(float(duration), 2),
            "tools": [list(t) for t in (tools or [])],
            "error": (error or None) and str(error)[:300],
            "failed": is_failure(answer, error),
        }
        with _LOCK:
            os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
            _rotate_if_needed()
            with open(_LOG_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.debug("flight_recorder.record failed: %s", exc)


def _rotate_if_needed() -> None:
    try:
        if os.path.exists(_LOG_PATH) and os.path.getsize(_LOG_PATH) > _MAX_BYTES:
            with open(_LOG_PATH) as f:
                lines = f.readlines()
            with open(_LOG_PATH, "w") as f:
                f.writelines(lines[len(lines) // 2:])
    except Exception:
        pass


def read_recent(days: int = 7) -> list:
    """Entries from the last `days` days, oldest first. Never raises."""
    cutoff = time.time() - days * 86400
    entries = []
    try:
        with open(_LOG_PATH) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if e.get("ts", 0) >= cutoff:
                    entries.append(e)
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.debug("flight_recorder.read_recent failed: %s", exc)
    return entries


def spoken_report(days: int = 7) -> str:
    """Voice-friendly self-report for 'how have you been performing'."""
    entries = read_recent(days)
    if not entries:
        return f"I have no recorded commands in the last {days} days."
    total = len(entries)
    failures = [e for e in entries if e.get("failed")]
    pct = round(100 * (total - len(failures)) / total)
    report = (
        f"In the last {days} days I handled {total} commands "
        f"with {len(failures)} failures, about {pct} percent success."
    )
    tool_fails: Counter = Counter()
    for e in failures:
        for name, ok in e.get("tools") or []:
            if not ok:
                tool_fails[name] += 1
    if tool_fails:
        top, n = tool_fails.most_common(1)[0]
        report += f" The most failure-prone tool was {top} with {n} failures."
    slow = [e for e in entries if e.get("duration", 0) > 15]
    if slow:
        report += f" {len(slow)} commands took over fifteen seconds."
    return report
