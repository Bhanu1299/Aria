"""
tests/test_memory_extractor.py — Unit tests for memory_extractor.py
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.base import LLMResponse


def _make_llm_response(text: str) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], stop_reason="end_turn",
                       provider_used="groq", model_used="llama-3.3-70b-versatile")


def _make_temp_identity(facts: list[str] | None = None) -> str:
    identity = {"name": "Test User", "learned_facts": facts or []}
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(identity, tmp)
    tmp.close()
    return tmp.name


def test_extract_finds_name_from_transcript():
    """extract() should return new facts parsed from LLM's JSON array response."""
    expected_facts = ["User's name is Alice"]
    tmp_path = _make_temp_identity()
    try:
        with patch("llm.llm_client.complete") as mock_complete, \
             patch("memory_extractor._IDENTITY_PATH", tmp_path):
            mock_complete.return_value = _make_llm_response(json.dumps(expected_facts))

            import memory_extractor
            result = memory_extractor.extract(
                transcript="Hi, I'm Alice.",
                answer="Nice to meet you, Alice!",
            )

        assert result == expected_facts, f"Expected {expected_facts}, got {result!r}"
    finally:
        os.unlink(tmp_path)


def test_no_duplicate_facts():
    """Calling extract_async twice with the same fact should only store it once."""
    fact = "User prefers dark mode"
    tmp_path = _make_temp_identity()
    done_events = [threading.Event(), threading.Event()]
    call_count = [0]

    try:
        with patch("llm.llm_client.complete") as mock_complete, \
             patch("memory_extractor._IDENTITY_PATH", tmp_path):
            mock_complete.return_value = _make_llm_response(json.dumps([fact]))

            import memory_extractor

            original_save = memory_extractor._save_identity

            def tracking_save(identity: dict) -> None:
                original_save(identity)
                idx = call_count[0]
                if idx < len(done_events):
                    done_events[idx].set()
                call_count[0] += 1

            with patch("memory_extractor._save_identity", side_effect=tracking_save):
                memory_extractor.extract_async("I love dark mode.", "Dark mode enabled.")
                done_events[0].wait(timeout=5.0)
                memory_extractor.extract_async("I love dark mode.", "Already using dark mode.")
                done_events[1].wait(timeout=5.0)

        with open(tmp_path) as f:
            saved = json.load(f)

        facts = saved.get("learned_facts", [])
        count = sum(1 for f in facts if f.lower() == fact.lower())
        assert count == 1, f"Expected fact once, found {count} times: {facts}"
    finally:
        os.unlink(tmp_path)


def test_facts_capped_at_50():
    """When identity already has 50 facts, a new extraction drops the oldest."""
    existing_facts = [f"Fact number {i}" for i in range(50)]
    new_fact = "User drinks coffee every morning"
    tmp_path = _make_temp_identity(facts=existing_facts)
    done_event = threading.Event()

    try:
        with patch("llm.llm_client.complete") as mock_complete, \
             patch("memory_extractor._IDENTITY_PATH", tmp_path):
            mock_complete.return_value = _make_llm_response(json.dumps([new_fact]))

            import memory_extractor

            original_save = memory_extractor._save_identity

            def capturing_save(identity: dict) -> None:
                original_save(identity)
                done_event.set()

            with patch("memory_extractor._save_identity", side_effect=capturing_save):
                memory_extractor.extract_async("I have coffee every morning.", "Got it!")
                done_event.wait(timeout=5.0)

        with open(tmp_path) as f:
            saved = json.load(f)

        facts = saved.get("learned_facts", [])
        assert len(facts) <= 50, f"Expected ≤50 facts, got {len(facts)}"
        assert new_fact in facts, f"New fact not found in {facts}"
        assert "Fact number 0" not in facts, "Oldest fact should have been displaced"
    finally:
        os.unlink(tmp_path)


def test_extract_async_does_not_block():
    """extract_async() must return in under 0.1s even when the LLM call is slow."""
    ready_event = threading.Event()

    def slow_extract(*args, **kwargs):
        ready_event.wait()
        return ["User is patient"]

    tmp_path = _make_temp_identity()
    try:
        with patch("memory_extractor.extract", side_effect=slow_extract), \
             patch("memory_extractor._IDENTITY_PATH", tmp_path):
            import memory_extractor
            t0 = time.monotonic()
            memory_extractor.extract_async("Do something slow", "Sure, doing it now.")
            elapsed = time.monotonic() - t0

        ready_event.set()
    finally:
        time.sleep(0.1)
        os.unlink(tmp_path)

    assert elapsed < 0.1, f"extract_async() blocked for {elapsed:.3f}s"


def test_graceful_on_groq_failure():
    """If LLM raises an exception, extract() returns [] — never re-raises."""
    with patch("llm.llm_client.complete", side_effect=RuntimeError("LLM down")):
        import memory_extractor
        result = memory_extractor.extract(
            transcript="What's my name?",
            answer="You are Bhanu.",
        )

    assert result == [], f"Expected [] on failure, got {result!r}"
