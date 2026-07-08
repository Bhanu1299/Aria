"""tests/test_auto_dream_promote.py — auto_dream fact promotion tests."""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_fact(fact_id, text, recall=5, unique_sessions=3, age_seconds=3600):
    ts = time.time() - age_seconds
    sessions = [f"s{i}" for i in range(unique_sessions)]
    return {
        "id": fact_id,
        "text": text,
        "metadata": {
            "recall_count": recall,
            "session_ids": json.dumps(sessions),
            "timestamp": ts,
            "source": "conversation",
        },
    }


def test_promote_writes_top_facts_to_identity():
    facts = [
        _make_fact("1", "User is a Python developer", recall=10, unique_sessions=5),
        _make_fact("2", "User targets ML engineer roles", recall=7, unique_sessions=3),
        _make_fact("3", "User prefers dark mode", recall=5, unique_sessions=2),
    ]
    store = MagicMock()
    store.get_all.return_value = facts

    with tempfile.TemporaryDirectory() as tmpdir:
        identity = {}
        with patch("aria.plugins.memory.vector_store.ChromaStore.get", return_value=store):
            import aria.state.auto_dream as auto_dream
            auto_dream._promote_top_facts(identity)

    assert "promoted_facts" in identity
    assert len(identity["promoted_facts"]) == 3
    texts = [p["fact"] for p in identity["promoted_facts"]]
    assert "User is a Python developer" in texts


def test_promote_skips_low_recall_facts():
    facts = [
        _make_fact("1", "ephemeral thing", recall=1, unique_sessions=1),
        _make_fact("2", "also ephemeral", recall=2, unique_sessions=1),
    ]
    store = MagicMock()
    store.get_all.return_value = facts

    identity = {}
    with patch("aria.plugins.memory.vector_store.ChromaStore.get", return_value=store):
        import aria.state.auto_dream as auto_dream
        auto_dream._promote_top_facts(identity)

    assert identity.get("promoted_facts", []) == []


def test_promote_caps_at_5():
    facts = [
        _make_fact(str(i), f"fact {i}", recall=10, unique_sessions=5)
        for i in range(10)
    ]
    store = MagicMock()
    store.get_all.return_value = facts

    identity = {}
    with patch("aria.plugins.memory.vector_store.ChromaStore.get", return_value=store):
        import aria.state.auto_dream as auto_dream
        auto_dream._promote_top_facts(identity)

    assert len(identity.get("promoted_facts", [])) <= 5


def test_promote_does_nothing_when_no_facts():
    store = MagicMock()
    store.get_all.return_value = []

    identity = {}
    with patch("aria.plugins.memory.vector_store.ChromaStore.get", return_value=store):
        import aria.state.auto_dream as auto_dream
        auto_dream._promote_top_facts(identity)

    assert "promoted_facts" not in identity


def test_promote_never_raises():
    store = MagicMock()
    store.get_all.side_effect = Exception("db error")

    identity = {}
    with patch("aria.plugins.memory.vector_store.ChromaStore.get", return_value=store):
        import aria.state.auto_dream as auto_dream
        auto_dream._promote_top_facts(identity)  # should not raise
