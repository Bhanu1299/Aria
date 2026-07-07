"""tests/test_vector_store.py — ChromaStore unit tests."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_chroma_mocks():
    """Return (mock_client, mock_collection) with sane defaults."""
    collection = MagicMock()
    collection.count.return_value = 1
    collection.query.return_value = {
        "ids": [["id1"]],
        "documents": [["User likes Python"]],
        "metadatas": [[{"source": "conversation", "recall_count": 2,
                        "timestamp": 1000.0, "session_ids": "[]"}]],
        "distances": [[0.2]],  # similarity = 1 - 0.2 = 0.8
    }
    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    return client, collection


def _make_embedder():
    emb = MagicMock()
    emb.embed.return_value = [0.1] * 384
    return emb


def _make_store():
    """Create a ChromaStore with all external deps mocked."""
    from plugins.memory.vector_store import ChromaStore
    ChromaStore._instance = None
    client, collection = _make_chroma_mocks()
    embedder = _make_embedder()
    with patch("chromadb.PersistentClient", return_value=client), \
         patch("plugins.memory.embedder.Embedder.get", return_value=embedder), \
         patch("pathlib.Path.mkdir"):
        store = ChromaStore()
        store._client = client
        store._collection = collection
        store._embedder = embedder
    return store, collection


def test_upsert_calls_collection():
    store, col = _make_store()
    store.upsert("id1", "User likes Python", {"source": "conversation"})
    assert col.upsert.called


def test_search_returns_facts_above_threshold():
    store, col = _make_store()
    results = store.search("Python")
    assert len(results) == 1
    assert results[0]["text"] == "User likes Python"
    assert results[0]["score"] >= 0.6


def test_search_filters_below_threshold():
    store, col = _make_store()
    # distance = 0.5 → sim = 0.5 < 0.6 threshold
    col.query.return_value = {
        "ids": [["id1"]],
        "documents": [["low score fact"]],
        "metadatas": [[{}]],
        "distances": [[0.5]],
    }
    results = store.search("query", min_score=0.6)
    assert results == []


def test_search_returns_empty_when_no_facts():
    store, col = _make_store()
    col.count.return_value = 0
    results = store.search("query")
    assert results == []


def test_increment_recall_updates_metadata():
    store, col = _make_store()
    col.get.return_value = {
        "ids": ["id1"],
        "documents": ["fact"],
        "metadatas": [{"recall_count": 2}],
        "embeddings": [[0.1] * 384],
    }
    store.increment_recall("id1")
    update_call = col.update.call_args
    assert update_call[1]["metadatas"][0]["recall_count"] == 3


def test_count_returns_integer():
    store, col = _make_store()
    col.count.return_value = 5
    assert store.count() == 5


def test_upsert_never_raises_on_error():
    store, col = _make_store()
    col.upsert.side_effect = Exception("disk full")
    store.upsert("id1", "fact", {})  # should not raise
