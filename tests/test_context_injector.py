"""tests/test_context_injector.py — context_injector unit tests."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _fake_store(facts):
    store = MagicMock()
    store.search.return_value = facts
    store.increment_recall = MagicMock()
    return store


def test_returns_formatted_string_with_facts():
    facts = [
        {"id": "1", "text": "User likes Python", "score": 0.9, "metadata": {}},
        {"id": "2", "text": "User is job hunting", "score": 0.8, "metadata": {}},
    ]
    with patch("plugins.memory.vector_store.ChromaStore.get", return_value=_fake_store(facts)):
        from plugins.memory import context_injector
        result = context_injector.build("Python jobs")
    assert "User likes Python" in result
    assert "User is job hunting" in result
    assert "Relevant things I know" in result


def test_returns_empty_string_when_no_facts():
    with patch("plugins.memory.vector_store.ChromaStore.get", return_value=_fake_store([])):
        from plugins.memory import context_injector
        result = context_injector.build("anything")
    assert result == ""


def test_increments_recall_for_retrieved_facts():
    facts = [{"id": "id1", "text": "User prefers dark mode", "score": 0.7, "metadata": {}}]
    store = _fake_store(facts)
    with patch("plugins.memory.vector_store.ChromaStore.get", return_value=store):
        from plugins.memory import context_injector
        context_injector.build("dark mode")
    store.increment_recall.assert_called_once_with("id1")


def test_returns_empty_string_on_error():
    with patch("plugins.memory.vector_store.ChromaStore.get", side_effect=Exception("boom")):
        from plugins.memory import context_injector
        result = context_injector.build("anything")
    assert result == ""
