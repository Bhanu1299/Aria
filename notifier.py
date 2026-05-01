from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)

_MIN_TASK_SECONDS = 15.0


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def send_notification(title: str, body: str) -> None:
    """Send a macOS notification via osascript. Never raises."""
    try:
        script = (
            f'display notification "{_esc(body[:200])}" '
            f'with title "{_esc(title[:100])}" '
            f'sound name "Glass"'
        )
        result = subprocess.run(
            ["osascript", "-e", script],
            timeout=5,
            capture_output=True,
        )
        if result.returncode != 0:
            logger.debug("notifier: osascript exit %d: %s", result.returncode, result.stderr.decode())
        else:
            logger.debug("notifier: sent '%s'", title)
    except Exception as exc:
        logger.warning("notifier.send_notification failed: %s", exc)


def notify_if_slow(elapsed: float, goal: str, summary: str) -> None:
    """Send notification only if task took longer than threshold."""
    if elapsed >= _MIN_TASK_SECONDS:
        short_summary = summary[:120] + "..." if len(summary) > 120 else summary
        send_notification(title=f"Aria: {goal[:60]}", body=short_summary)
