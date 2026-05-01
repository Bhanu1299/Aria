"""
tests/test_auto_dream.py — Unit tests for auto_dream.py (Task 10: AutoDream)

Tests:
  test_consolidate_rewrites_session_notes     — mock Groq, verify clear + store called
  test_consolidate_deduplicates_facts         — mock Groq returning merged facts, verify identity.json updated
  test_maybe_consolidate_fires_at_interval    — at count=5, consolidate() is called
  test_maybe_consolidate_does_not_fire_before_interval — at count=4, consolidate() not called
  test_consolidate_graceful_on_groq_failure   — when Groq raises, function returns without raising
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_groq_response(content: str):
    """Build a minimal mock that looks like a groq ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_temp_identity(facts: list | None = None) -> str:
    """Write a temporary identity.json with given learned_facts and return path."""
    identity = {"name": "Test User", "learned_facts": facts or []}
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(identity, tmp)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# test_consolidate_rewrites_session_notes
# ---------------------------------------------------------------------------

def test_consolidate_rewrites_session_notes():
    """consolidate() should call clear_session_notes() then store_session_notes() with new notes."""
    new_notes = "- User searched for Python jobs\n- User asked about salaries"
    groq_payload = json.dumps({
        "session_notes": new_notes,
        "learned_facts": ["User is a Python developer"],
    })
    fake_resp = _make_fake_groq_response(groq_payload)
    tmp_path = _make_temp_identity()

    try:
        with patch("auto_dream._get_client") as mock_get_client, \
             patch("auto_dream._IDENTITY_PATH", tmp_path), \
             patch("auto_dream.memory.get_session_notes", return_value="old notes"), \
             patch("auto_dream.memory.clear_session_notes") as mock_clear, \
             patch("auto_dream.memory.store_session_notes") as mock_store, \
             patch("auto_dream.memory.reset_command_count"):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = fake_resp
            mock_get_client.return_value = mock_client

            import auto_dream
            auto_dream.consolidate()

        mock_clear.assert_called_once()
        mock_store.assert_called_once_with(new_notes)
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# test_consolidate_deduplicates_facts
# ---------------------------------------------------------------------------

def test_consolidate_deduplicates_facts():
    """consolidate() should write the deduplicated facts list returned by Groq to identity.json."""
    original_facts = ["User likes Python", "User likes Python", "User is a developer"]
    deduped_facts = ["User likes Python", "User is a developer"]

    groq_payload = json.dumps({
        "session_notes": "- Consolidated",
        "learned_facts": deduped_facts,
    })
    fake_resp = _make_fake_groq_response(groq_payload)
    tmp_path = _make_temp_identity(facts=original_facts)

    try:
        with patch("auto_dream._get_client") as mock_get_client, \
             patch("auto_dream._IDENTITY_PATH", tmp_path), \
             patch("auto_dream.memory.get_session_notes", return_value="some notes"), \
             patch("auto_dream.memory.clear_session_notes"), \
             patch("auto_dream.memory.store_session_notes"), \
             patch("auto_dream.memory.reset_command_count"):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = fake_resp
            mock_get_client.return_value = mock_client

            import auto_dream
            auto_dream.consolidate()

        with open(tmp_path) as f:
            saved = json.load(f)

        assert saved["learned_facts"] == deduped_facts, (
            f"Expected {deduped_facts}, got {saved['learned_facts']}"
        )
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# test_maybe_consolidate_fires_at_interval
# ---------------------------------------------------------------------------

def test_maybe_consolidate_fires_at_interval():
    """maybe_consolidate_async() should trigger consolidate() when count reaches 5."""
    done_event = threading.Event()

    def fake_consolidate():
        done_event.set()

    with patch("auto_dream.memory.increment_command_count", return_value=5), \
         patch("auto_dream.consolidate", side_effect=fake_consolidate):
        import auto_dream
        auto_dream.maybe_consolidate_async("hello", "world")

    # Give the daemon thread a moment to fire
    fired = done_event.wait(timeout=2.0)
    assert fired, "consolidate() was not called when count == 5"


# ---------------------------------------------------------------------------
# test_maybe_consolidate_does_not_fire_before_interval
# ---------------------------------------------------------------------------

def test_maybe_consolidate_does_not_fire_before_interval():
    """maybe_consolidate_async() should NOT trigger consolidate() when count is 4."""
    called = []

    def fake_consolidate():
        called.append(True)

    with patch("auto_dream.memory.increment_command_count", return_value=4), \
         patch("auto_dream.consolidate", side_effect=fake_consolidate):
        import auto_dream
        auto_dream.maybe_consolidate_async("hello", "world")

    # Wait briefly to ensure no background thread fires
    time.sleep(0.15)
    assert not called, f"consolidate() should not have been called at count=4, but was called {len(called)} time(s)"


# ---------------------------------------------------------------------------
# test_consolidate_graceful_on_groq_failure
# ---------------------------------------------------------------------------

def test_consolidate_graceful_on_groq_failure():
    """consolidate() must not raise even when Groq throws an exception."""
    with patch("auto_dream._get_client") as mock_get_client, \
         patch("auto_dream.memory.get_session_notes", return_value="some notes"):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("Groq is down")
        mock_get_client.return_value = mock_client

        import auto_dream
        # Should not raise
        try:
            auto_dream.consolidate()
        except Exception as exc:
            raise AssertionError(
                f"consolidate() raised {type(exc).__name__}: {exc} — expected graceful handling"
            ) from exc
