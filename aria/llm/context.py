"""llm/context.py — labels attached to every GateLLM call, for its per-call log.

  task            which TASK_TIERS entry made the call         (per thread)
  escalated_from  why this call is a retry on a bigger tier     (per thread)
  job / run       the pipeline job and run in progress          (process-wide: one
                  pipeline runs at a time, and its LLM calls may come from the
                  browser worker thread, not the thread that started the run)

Sent as X-Gateway-Task / -Escalated-From / -Job / -Run headers. Labels only: no prompt text.
"""
from __future__ import annotations

import contextlib
import threading
from typing import Iterator, Optional

_local = threading.local()
_global_lock = threading.Lock()
_global: dict[str, str] = {}

_HEADERS = {"task": "X-Gateway-Task", "escalated_from": "X-Gateway-Escalated-From",
            "job": "X-Gateway-Job", "run": "X-Gateway-Run"}


def _thread_labels() -> dict:
    if not hasattr(_local, "labels"):
        _local.labels = {}
    return _local.labels


@contextlib.contextmanager
def scope(task: Optional[str] = None, escalated_from: Optional[str] = None) -> Iterator[None]:
    """Label LLM calls made on this thread inside the block (inner scopes win)."""
    labels = _thread_labels()
    saved = dict(labels)
    if task:
        labels["task"] = task
    if escalated_from:
        labels["escalated_from"] = escalated_from
    try:
        yield
    finally:
        labels.clear()
        labels.update(saved)


@contextlib.contextmanager
def job(job_label: str, run_id: str) -> Iterator[None]:
    """Label every LLM call, on any thread, as part of this pipeline job + run."""
    with _global_lock:
        saved = dict(_global)
        _global.update({"job": job_label, "run": run_id})
    try:
        yield
    finally:
        with _global_lock:
            _global.clear()
            _global.update(saved)


def current() -> dict:
    with _global_lock:
        out = dict(_global)
    out.update(_thread_labels())
    return out


def headers() -> dict:
    """Header values must be latin-1 for HTTP: anything else is replaced."""
    return {_HEADERS[k]: str(v).encode("latin-1", "replace").decode("latin-1")[:200]
            for k, v in current().items() if v}
