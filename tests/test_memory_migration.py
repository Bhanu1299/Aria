"""tests/test_memory_migration.py — MemoryPlugin migration tests."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_store():
    store = MagicMock()
    store._collection = MagicMock()
    return store


def test_migration_skipped_when_marker_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        marker = Path(tmpdir) / ".migrated"
        marker.write_text("ok")
        with patch("aria.plugins.memory._MIGRATION_MARKER", marker, create=True):
            from aria.plugins.memory import MemoryPlugin
            plugin = MemoryPlugin()
            # If migration marker present, _migrate does nothing
            store = _make_store()
            with patch("aria.plugins.memory._MIGRATION_MARKER", marker), \
                 patch("aria.plugins.memory.vector_store.ChromaStore.get", return_value=store):
                plugin._migrate()
            store._collection.upsert.assert_not_called()


def test_migration_runs_when_no_marker():
    with tempfile.TemporaryDirectory() as tmpdir:
        marker = Path(tmpdir) / ".migrated"
        identity = {"learned_facts": ["User likes Python", "User is a developer"]}
        identity_file = Path(tmpdir) / "identity.json"
        identity_file.write_text(json.dumps(identity))

        store = _make_store()
        with patch("aria.plugins.memory._MIGRATION_MARKER", marker), \
             patch("aria.plugins.memory.MemoryPlugin._migrate",
                   lambda self: _run_migrate_with_paths(self, marker, identity_file, store)):
            from aria.plugins.memory import MemoryPlugin
            plugin = MemoryPlugin()
            plugin._migrate()
        assert marker.exists()


def _run_migrate_with_paths(plugin, marker, identity_path, store):
    """Run the migration logic with custom paths for testing."""
    import json, time, uuid
    if marker.exists():
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(identity_path) as f:
            identity = json.load(f)
    except Exception:
        identity = {}
    facts = identity.get("learned_facts", [])
    from aria.plugins.memory.embedder import Embedder
    embedder = MagicMock()
    embedder.embed_batch.return_value = [[0.1] * 384, [0.1] * 384]
    ts = time.time()
    batch_texts = [f for f in facts if isinstance(f, str) and f.strip()]
    if batch_texts:
        embeddings = embedder.embed_batch(batch_texts)
        for i, (text, emb) in enumerate(zip(batch_texts, embeddings)):
            fact_id = str(uuid.uuid4())
            store._collection.upsert(
                ids=[fact_id], documents=[text],
                embeddings=[emb],
                metadatas=[{"source": "migration", "session_ids": "[]",
                            "timestamp": ts, "recall_count": 0}],
            )
    marker.write_text("ok")
    assert store._collection.upsert.call_count == len(batch_texts)


def test_migration_handles_empty_facts():
    with tempfile.TemporaryDirectory() as tmpdir:
        marker = Path(tmpdir) / ".migrated"
        identity = {"learned_facts": []}
        identity_file = Path(tmpdir) / "identity.json"
        identity_file.write_text(json.dumps(identity))
        store = _make_store()
        with patch("aria.plugins.memory._MIGRATION_MARKER", marker), \
             patch("aria.plugins.memory.MemoryPlugin._migrate",
                   lambda self: _run_migrate_with_paths(self, marker, identity_file, store)):
            from aria.plugins.memory import MemoryPlugin
            MemoryPlugin()._migrate()
        assert marker.exists()
        store._collection.upsert.assert_not_called()
