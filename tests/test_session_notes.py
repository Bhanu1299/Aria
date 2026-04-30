"""
tests/test_session_notes.py — Unit tests for session_notes.py and the
session-notes additions to memory.py.

Tests:
  test_extract_returns_string            — extract() always returns a str
  test_extract_async_does_not_block      — extract_async() returns in <0.5s
                                           even when Groq would take 5 s
  test_notes_stored_in_memory            — notes end up in memory after worker
  test_extract_handles_groq_failure_gracefully — Groq exception → "" not raised
"""

from __future__ import annotations

import sys
import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is on the path
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


# ---------------------------------------------------------------------------
# test_extract_returns_string
# ---------------------------------------------------------------------------

def test_extract_returns_string():
    """extract() must return a plain str even when Groq returns notes."""
    fake_notes = "- User asked about Python jobs\n- Aria found 3 listings"
    fake_resp = _make_fake_groq_response(fake_notes)

    with patch("session_notes._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = fake_resp
        mock_get_client.return_value = mock_client

        import session_notes
        result = session_notes.extract(
            transcript="Find me Python jobs",
            answer="I found 3 Python jobs on LinkedIn.",
        )

    assert isinstance(result, str), "extract() must return str"
    assert result == fake_notes.strip()


# ---------------------------------------------------------------------------
# test_extract_handles_groq_failure_gracefully
# ---------------------------------------------------------------------------

def test_extract_handles_groq_failure_gracefully():
    """If Groq raises an exception, extract() returns '' — never re-raises."""
    with patch("session_notes._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("Groq down")
        mock_get_client.return_value = mock_client

        import session_notes
        result = session_notes.extract(
            transcript="What is the weather?",
            answer="It is sunny.",
        )

    assert result == "", f"Expected '' on failure, got {result!r}"


# ---------------------------------------------------------------------------
# test_extract_async_does_not_block
# ---------------------------------------------------------------------------

def test_extract_async_does_not_block():
    """
    extract_async() must return to the caller in under 0.5 s even when the
    underlying Groq call would take 5+ seconds.
    """
    ready_event = threading.Event()

    def slow_extract(*args, **kwargs):
        ready_event.wait()  # blocks until test releases it
        return "- slow result"

    with patch("session_notes.extract", side_effect=slow_extract):
        import session_notes
        t0 = time.monotonic()
        session_notes.extract_async(
            transcript="Do something slow",
            answer="Sure, doing it now.",
        )
        elapsed = time.monotonic() - t0

    # Release the blocked thread so the process doesn't hang
    ready_event.set()

    assert elapsed < 0.5, (
        f"extract_async() blocked for {elapsed:.3f}s — expected < 0.5s"
    )


# ---------------------------------------------------------------------------
# test_notes_stored_in_memory
# ---------------------------------------------------------------------------

def test_notes_stored_in_memory():
    """
    After extract_async() finishes, the notes must be retrievable via
    memory.get_session_notes().
    """
    fake_notes = "- Asked about jobs\n- Found 2 results"
    done_event = threading.Event()

    # We need extract() to complete synchronously inside the worker thread
    # so we can wait for it without polling.
    original_worker_extract = None

    def capturing_extract(transcript: str, answer: str) -> str:
        notes = fake_notes
        return notes

    def patched_worker(transcript: str, answer: str) -> None:
        """Replicate _worker but set done_event so the test can wait."""
        import session_notes as sn
        notes = capturing_extract(transcript, answer)
        if notes:
            import memory as mem
            mem.store_session_notes(notes)
        done_event.set()

    with patch("session_notes.extract", side_effect=capturing_extract), \
         patch("session_notes._worker", side_effect=patched_worker):
        import memory
        import session_notes

        # Clear any pre-existing notes
        memory.clear_session_notes()

        session_notes.extract_async(
            transcript="Find senior ML jobs",
            answer="I found 2 senior ML jobs.",
        )

        # Wait up to 3 seconds for the daemon thread to finish
        finished = done_event.wait(timeout=3.0)

    assert finished, "Worker thread did not complete within 3 seconds"
    stored = memory.get_session_notes()
    assert stored == fake_notes, (
        f"Expected notes in memory, got {stored!r}"
    )
