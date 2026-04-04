"""
skills/apply_status — Aria Phase 3D: Read application history from tracker.py.

Handles: "what jobs have I applied to", "application history", "show my applications"
"""

from __future__ import annotations

TRIGGERS = [
    "what jobs have i applied to",
    "application history",
    "show my applications",
    "jobs i've applied to",
    "jobs ive applied to",
    "applied to any",
    "what have i applied for",
]


def handle(command: str) -> str:
    """Return a spoken summary of the most recent job applications."""
    import tracker

    apps = tracker.get_applications()
    if not apps:
        return "You haven't applied to any jobs through Aria yet."

    recent = apps[:5]
    total = len(apps)
    parts = []
    for app in recent:
        date = app.get("applied_at", "")[:10]  # "2026-04-04"
        parts.append(
            f"{app['role']} at {app['company']}"
            + (f" on {date}" if date else "")
        )

    summary = f"You've applied to {total} job{'s' if total != 1 else ''} through Aria. "
    summary += "Most recent: " + "; ".join(parts) + "."
    return summary
