"""llm/meter.py — process-wide tally of every LLM call that went through GateLLM.

The dispatcher snapshots it around each command, so the flight log's cost covers
everything the command triggered: agent turns and LLM calls made inside tools
(vision, classifiers, computer use).
"""
from __future__ import annotations

import threading

_LOCK = threading.Lock()
_TOTALS = {"cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0, "calls": 0}


def add(meta: dict) -> None:
    with _LOCK:
        _TOTALS["cost_usd"] += float(meta.get("cost_usd", 0.0))
        _TOTALS["tokens_in"] += int(meta.get("tokens_in", 0))
        _TOTALS["tokens_out"] += int(meta.get("tokens_out", 0))
        _TOTALS["calls"] += 1


def snapshot() -> dict:
    with _LOCK:
        return dict(_TOTALS)


def since(before: dict) -> dict:
    now = snapshot()
    return {
        "cost_usd": round(now["cost_usd"] - before["cost_usd"], 6),
        "tokens_in": now["tokens_in"] - before["tokens_in"],
        "tokens_out": now["tokens_out"] - before["tokens_out"],
        "calls": now["calls"] - before["calls"],
    }
