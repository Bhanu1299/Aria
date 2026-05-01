"""
tests/test_notifier.py — Unit tests for notifier.py

Tests:
  test_send_notification_calls_osascript        — subprocess.run called with osascript + title + body
  test_send_notification_handles_osascript_failure — subprocess.run raises → function does not raise
  test_notify_after_long_task                   — elapsed > 15.0 → send_notification called
  test_no_notify_after_short_task               — elapsed <= 15.0 → send_notification NOT called
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# test_send_notification_calls_osascript
# ---------------------------------------------------------------------------

def test_send_notification_calls_osascript():
    """send_notification() must call subprocess.run with osascript, title, and body."""
    import notifier

    with patch("notifier.subprocess.run") as mock_run:
        notifier.send_notification(title="Test Title", body="Test body text")

    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    cmd = args[0]
    assert cmd[0] == "osascript", f"Expected 'osascript', got {cmd[0]!r}"
    assert cmd[1] == "-e", f"Expected '-e' flag, got {cmd[1]!r}"
    script = cmd[2]
    assert "Test Title" in script, f"Title not found in script: {script!r}"
    assert "Test body text" in script, f"Body not found in script: {script!r}"


# ---------------------------------------------------------------------------
# test_send_notification_handles_osascript_failure
# ---------------------------------------------------------------------------

def test_send_notification_handles_osascript_failure():
    """send_notification() must not raise even if subprocess.run raises."""
    import notifier

    with patch("notifier.subprocess.run", side_effect=OSError("osascript not found")):
        # Must not raise
        notifier.send_notification(title="Aria", body="something happened")


# ---------------------------------------------------------------------------
# test_notify_after_long_task
# ---------------------------------------------------------------------------

def test_notify_after_long_task():
    """notify_if_slow() must call send_notification when elapsed > 15.0 seconds."""
    import notifier

    with patch("notifier.send_notification") as mock_notify:
        notifier.notify_if_slow(
            elapsed=20.5,
            goal="find flights to NYC",
            summary="Found 3 cheap flights under $200.",
        )

    mock_notify.assert_called_once()
    args, kwargs = mock_notify.call_args
    # Called as keyword args: title=..., body=...
    title = kwargs.get("title") or args[0]
    body = kwargs.get("body") or args[1]
    assert "find flights to NYC" in title, f"Goal not in title: {title!r}"
    assert "Found 3 cheap flights" in body, f"Summary not in body: {body!r}"


# ---------------------------------------------------------------------------
# test_no_notify_after_short_task
# ---------------------------------------------------------------------------

def test_no_notify_after_short_task():
    """notify_if_slow() must NOT call send_notification when elapsed <= 15.0 seconds."""
    import notifier

    with patch("notifier.send_notification") as mock_notify:
        notifier.notify_if_slow(
            elapsed=14.9,
            goal="quick search",
            summary="Done quickly.",
        )

    mock_notify.assert_not_called()

    # Also test exactly at threshold (15.0 should trigger)
    with patch("notifier.send_notification") as mock_notify_exact:
        notifier.notify_if_slow(
            elapsed=15.0,
            goal="exact threshold",
            summary="Exactly 15 seconds.",
        )

    mock_notify_exact.assert_called_once()


# ---------------------------------------------------------------------------
# test_send_notification_escapes_quotes
# ---------------------------------------------------------------------------

def test_send_notification_escapes_quotes() -> None:
    """send_notification() must escape double-quotes so AppleScript syntax is not broken."""
    import notifier

    with patch("notifier.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        notifier.send_notification(title='He said "hello"', body='Price: $10 "good deal"')
        call_args = mock_run.call_args
        script = call_args[0][0][2]  # third element of the args list is the script string
        assert '\\"' in script  # escaped quotes present
        assert 'He said "hello"' not in script  # raw unescaped form not present
