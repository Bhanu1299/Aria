"""
tests/test_memory_extractor.py — Unit tests for memory_extractor.py

Tests:
  test_extract_finds_name_from_transcript   — extract() returns facts from Groq response
  test_no_duplicate_facts                  — same fact is not stored twice
  test_facts_capped_at_50                  — 51st fact displaces the oldest
  test_extract_async_does_not_block        — extract_async() returns in < 0.1s even with slow Groq
  test_graceful_on_groq_failure            — Groq exception → [] not raised
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
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


def _make_temp_identity(facts: list[str] | None = None) -> str:
    """Write a temporary identity.json with given learned_facts and return path."""
    identity = {"name": "Test User", "learned_facts": facts or []}
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    )
    json.dump(identity, tmp)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# test_extract_finds_name_from_transcript
# ---------------------------------------------------------------------------

def test_extract_finds_name_from_transcript():
    """extract() should return new facts parsed from Groq's JSON array response."""
    expected_facts = ["User's name is Alice"]
    fake_resp = _make_fake_groq_response(json.dumps(expected_facts))

    tmp_path = _make_temp_identity()
    try:
        with patch("memory_extractor._get_client") as mock_get_client, \
             patch("memory_extractor._IDENTITY_PATH", tmp_path):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = fake_resp
            mock_get_client.return_value = mock_client

            import memory_extractor
            result = memory_extractor.extract(
                transcript="Hi, I'm Alice.",
                answer="Nice to meet you, Alice!",
            )

        assert result == expected_facts, f"Expected {expected_facts}, got {result!r}"
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# test_no_duplicate_facts
# ---------------------------------------------------------------------------

def test_no_duplicate_facts():
    """Calling extract_async twice with the same fact should only store it once."""
    fact = "User prefers dark mode"
    fake_resp = _make_fake_groq_response(json.dumps([fact]))

    tmp_path = _make_temp_identity()
    done_events = [threading.Event(), threading.Event()]
    call_count = [0]

    try:
        with patch("memory_extractor._get_client") as mock_get_client, \
             patch("memory_extractor._IDENTITY_PATH", tmp_path):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = fake_resp
            mock_get_client.return_value = mock_client

            import memory_extractor

            original_save = memory_extractor._save_identity

            def tracking_save(identity: dict) -> None:
                original_save(identity)
                idx = call_count[0]
                if idx < len(done_events):
                    done_events[idx].set()
                call_count[0] += 1

            with patch("memory_extractor._save_identity", side_effect=tracking_save):
                memory_extractor.extract_async(
                    transcript="I love dark mode.",
                    answer="Dark mode enabled.",
                )
                done_events[0].wait(timeout=5.0)

                memory_extractor.extract_async(
                    transcript="I love dark mode.",
                    answer="Already using dark mode.",
                )
                done_events[1].wait(timeout=5.0)

        with open(tmp_path) as f:
            saved = json.load(f)

        facts = saved.get("learned_facts", [])
        count = sum(1 for f in facts if f.lower() == fact.lower())
        assert count == 1, f"Expected fact once, found {count} times: {facts}"
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# test_facts_capped_at_50
# ---------------------------------------------------------------------------

def test_facts_capped_at_50():
    """When identity already has 50 facts, a new extraction drops the oldest."""
    existing_facts = [f"Fact number {i}" for i in range(50)]
    new_fact = "User drinks coffee every morning"

    fake_resp = _make_fake_groq_response(json.dumps([new_fact]))
    tmp_path = _make_temp_identity(facts=existing_facts)

    done_event = threading.Event()

    try:
        with patch("memory_extractor._get_client") as mock_get_client, \
             patch("memory_extractor._IDENTITY_PATH", tmp_path):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = fake_resp
            mock_get_client.return_value = mock_client

            import memory_extractor

            original_save = memory_extractor._save_identity

            def capturing_save(identity: dict) -> None:
                original_save(identity)
                done_event.set()

            with patch("memory_extractor._save_identity", side_effect=capturing_save):
                memory_extractor.extract_async(
                    transcript="I have coffee every morning.",
                    answer="Got it!",
                )
                done_event.wait(timeout=5.0)

        with open(tmp_path) as f:
            saved = json.load(f)

        facts = saved.get("learned_facts", [])
        assert len(facts) <= 50, f"Expected ≤50 facts, got {len(facts)}"
        assert new_fact in facts, f"New fact not found in {facts}"
        assert "Fact number 0" not in facts, "Oldest fact should have been displaced"
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# test_extract_async_does_not_block
# ---------------------------------------------------------------------------

def test_extract_async_does_not_block():
    """
    extract_async() must return to the caller in under 0.1s even when the
    underlying Groq call takes 2 seconds.
    """
    ready_event = threading.Event()

    def slow_extract(*args, **kwargs):
        ready_event.wait()  # blocks until test releases it
        return ["User is patient"]

    tmp_path = _make_temp_identity()
    try:
        with patch("memory_extractor.extract", side_effect=slow_extract), \
             patch("memory_extractor._IDENTITY_PATH", tmp_path):
            import memory_extractor
            t0 = time.monotonic()
            memory_extractor.extract_async(
                transcript="Do something slow",
                answer="Sure, doing it now.",
            )
            elapsed = time.monotonic() - t0

        # Release the blocked thread so the process doesn't hang
        ready_event.set()
    finally:
        # Wait briefly for the thread to finish before removing the temp file
        time.sleep(0.1)
        os.unlink(tmp_path)

    assert elapsed < 0.1, (
        f"extract_async() blocked for {elapsed:.3f}s — expected < 0.1s"
    )


# ---------------------------------------------------------------------------
# test_graceful_on_groq_failure
# ---------------------------------------------------------------------------

def test_graceful_on_groq_failure():
    """If Groq raises an exception, extract() returns [] — never re-raises."""
    with patch("memory_extractor._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("Groq down")
        mock_get_client.return_value = mock_client

        import memory_extractor
        result = memory_extractor.extract(
            transcript="What's my name?",
            answer="You are Bhanu.",
        )

    assert result == [], f"Expected [] on failure, got {result!r}"
