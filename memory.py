"""
memory.py — Aria session state (Phase 3A)

Module-level dict that persists for the lifetime of the process.
Enables follow-up commands ("apply to the second one") to reference
results from earlier in the same session.
"""

from __future__ import annotations

session: dict = {}


def store_jobs(results: list[dict]) -> None:
    """Overwrite last_jobs with the given results list."""
    session["last_jobs"] = results


def get_job_by_index(n: int) -> dict | None:
    """Return the 1-based nth job from last_jobs, or None if out of range."""
    jobs = session.get("last_jobs", [])
    return jobs[n - 1] if 0 < n <= len(jobs) else None
