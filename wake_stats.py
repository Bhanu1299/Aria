"""
wake_stats.py — wake word observability log.

Answers "is the wake word even working?" with data instead of guessing:

  heartbeat(backend)       — engine-alive marker, throttled to one/minute
  log_event(kind, backend, score=None) — detection / near_miss / no_speech / restart
  read_recent(days)        — parsed entries, corrupt lines skipped
  summary(days)            — dict used by `daily_check.py wake`

Log lives at ~/.aria/wake_log.jsonl — local only. Rotates at ~5MB
(keeps the newest half). Never raises; the wake listener must survive
anything this module does.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

_LOG_PATH = os.path.join(os.path.expanduser("~"), ".aria", "wake_log.jsonl")
_MAX_BYTES = 5 * 1024 * 1024
_HEARTBEAT_INTERVAL = 60.0

_LOCK = threading.Lock()
_last_heartbeat = 0.0

KINDS = ("heartbeat", "detection", "near_miss", "no_speech", "restart")


def _append(entry: dict) -> None:
    with _LOCK:
        os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
        _rotate_if_needed()
        with open(_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")


def _rotate_if_needed() -> None:
    try:
        if os.path.exists(_LOG_PATH) and os.path.getsize(_LOG_PATH) > _MAX_BYTES:
            with open(_LOG_PATH) as f:
                lines = f.readlines()
            with open(_LOG_PATH, "w") as f:
                f.writelines(lines[len(lines) // 2:])
    except Exception:
        pass


def log_event(kind: str, backend: str, score: Optional[float] = None) -> None:
    """Append one wake event. Never raises."""
    try:
        entry = {"ts": time.time(), "kind": kind, "backend": backend}
        if score is not None:
            entry["score"] = round(float(score), 3)
        _append(entry)
    except Exception as exc:
        logger.debug("wake_stats.log_event failed: %s", exc)


def heartbeat(backend: str) -> None:
    """Engine-alive marker; throttled internally, cheap to call every loop."""
    global _last_heartbeat
    try:
        now = time.time()
        if now - _last_heartbeat < _HEARTBEAT_INTERVAL:
            return
        _last_heartbeat = now
        _append({"ts": now, "kind": "heartbeat", "backend": backend})
    except Exception as exc:
        logger.debug("wake_stats.heartbeat failed: %s", exc)


def read_recent(days: float = 7) -> list:
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
        logger.debug("wake_stats.read_recent failed: %s", exc)
    return entries


def summary(days: float = 7) -> dict:
    """
    Aggregate for the wake report:
      alive_secs_ago  — seconds since last heartbeat/any event, or None (never seen)
      backend         — backend named by the most recent entry
      detections, near_misses, no_speech, restarts — counts
      near_miss_scores — sorted list
      suggested_threshold — float or None: set just below the median near-miss
                            score when misses outnumber detections
    """
    entries = read_recent(days)
    out = {
        "alive_secs_ago": None, "backend": None,
        "detections": 0, "near_misses": 0, "no_speech": 0, "restarts": 0,
        "near_miss_scores": [], "suggested_threshold": None,
        "total_entries": len(entries),
    }
    if not entries:
        return out

    last = entries[-1]
    out["alive_secs_ago"] = max(0.0, time.time() - last.get("ts", 0))
    out["backend"] = last.get("backend")

    for e in entries:
        kind = e.get("kind")
        if kind == "detection":
            out["detections"] += 1
        elif kind == "near_miss":
            out["near_misses"] += 1
            if isinstance(e.get("score"), (int, float)):
                out["near_miss_scores"].append(float(e["score"]))
        elif kind == "no_speech":
            out["no_speech"] += 1
        elif kind == "restart":
            out["restarts"] += 1

    out["near_miss_scores"].sort()
    if out["near_miss_scores"] and out["near_misses"] > out["detections"]:
        median = statistics.median(out["near_miss_scores"])
        out["suggested_threshold"] = max(0.5, round(median - 0.02, 2))
    return out
