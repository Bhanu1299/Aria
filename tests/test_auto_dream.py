"""
tests/test_auto_dream.py — Unit tests for auto_dream.py (Task 10: AutoDream)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.llm.base import LLMResponse


def _make_llm_response(text: str) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], stop_reason="end_turn",
                       provider_used="groq", model_used="llama-3.3-70b-versatile")


def _make_temp_identity(facts: list | None = None) -> str:
    identity = {"name": "Test User", "learned_facts": facts or []}
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(identity, tmp)
    tmp.close()
    return tmp.name


def test_consolidate_rewrites_session_notes():
    """consolidate() should call clear_session_notes() then store_session_notes() with new notes."""
    new_notes = "- User searched for Python jobs\n- User asked about salaries"
    groq_payload = json.dumps({
        "session_notes": new_notes,
        "learned_facts": ["User is a Python developer"],
    })
    tmp_path = _make_temp_identity()

    try:
        with patch("aria.llm.llm_client.complete") as mock_complete, \
             patch("aria.state.auto_dream._IDENTITY_PATH", tmp_path), \
             patch("aria.state.auto_dream.memory.get_session_notes", return_value="old notes"), \
             patch("aria.state.auto_dream.memory.clear_session_notes") as mock_clear, \
             patch("aria.state.auto_dream.memory.store_session_notes") as mock_store, \
             patch("aria.state.auto_dream.memory.reset_command_count"):
            mock_complete.return_value = _make_llm_response(groq_payload)

            import aria.state.auto_dream as auto_dream
            auto_dream.consolidate()

        mock_clear.assert_called_once()
        mock_store.assert_called_once_with(new_notes)
    finally:
        os.unlink(tmp_path)


def test_consolidate_deduplicates_facts():
    """consolidate() should write the deduplicated facts list returned by LLM to identity.json."""
    original_facts = ["User likes Python", "User likes Python", "User is a developer"]
    deduped_facts = ["User likes Python", "User is a developer"]

    groq_payload = json.dumps({
        "session_notes": "- Consolidated",
        "learned_facts": deduped_facts,
    })
    tmp_path = _make_temp_identity(facts=original_facts)

    try:
        with patch("aria.llm.llm_client.complete") as mock_complete, \
             patch("aria.state.auto_dream._IDENTITY_PATH", tmp_path), \
             patch("aria.state.auto_dream.memory.get_session_notes", return_value="some notes"), \
             patch("aria.state.auto_dream.memory.clear_session_notes"), \
             patch("aria.state.auto_dream.memory.store_session_notes"), \
             patch("aria.state.auto_dream.memory.reset_command_count"):
            mock_complete.return_value = _make_llm_response(groq_payload)

            import aria.state.auto_dream as auto_dream
            auto_dream.consolidate()

        with open(tmp_path) as f:
            saved = json.load(f)

        assert saved["learned_facts"] == deduped_facts
    finally:
        os.unlink(tmp_path)


def test_maybe_consolidate_fires_at_interval():
    """maybe_consolidate_async() should trigger consolidate() when count reaches 5."""
    done_event = threading.Event()

    def fake_consolidate():
        done_event.set()

    with patch("aria.state.auto_dream.memory.increment_command_count", return_value=5), \
         patch("aria.state.auto_dream.consolidate", side_effect=fake_consolidate):
        import aria.state.auto_dream as auto_dream
        auto_dream.maybe_consolidate_async("hello", "world")

    fired = done_event.wait(timeout=2.0)
    assert fired, "consolidate() was not called when count == 5"


def test_maybe_consolidate_does_not_fire_before_interval():
    """maybe_consolidate_async() should NOT trigger consolidate() when count is 4."""
    called = []

    def fake_consolidate():
        called.append(True)

    with patch("aria.state.auto_dream.memory.increment_command_count", return_value=4), \
         patch("aria.state.auto_dream.consolidate", side_effect=fake_consolidate):
        import aria.state.auto_dream as auto_dream
        auto_dream.maybe_consolidate_async("hello", "world")

    time.sleep(0.15)
    assert not called


def test_consolidate_graceful_on_groq_failure():
    """consolidate() must not raise even when LLM throws an exception."""
    with patch("aria.llm.llm_client.complete", side_effect=RuntimeError("LLM is down")), \
         patch("aria.state.auto_dream.memory.get_session_notes", return_value="some notes"):
        import aria.state.auto_dream as auto_dream
        try:
            auto_dream.consolidate()
        except Exception as exc:
            raise AssertionError(
                f"consolidate() raised {type(exc).__name__}: {exc}"
            ) from exc
