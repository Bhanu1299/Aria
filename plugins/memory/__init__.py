"""plugins/memory/__init__.py — MemoryPlugin: one-time SQLite→ChromaDB migration."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

import plugin as _plugin_base
from tool import ToolRegistry

logger = logging.getLogger(__name__)

_MIGRATION_MARKER = Path.home() / ".aria" / "memory" / ".migrated"


class MemoryPlugin(_plugin_base.PluginBase):
    """Registers no tools — wires up ChromaDB migration at startup."""

    def register(self, registry: ToolRegistry) -> None:
        t = threading.Thread(target=self._migrate, daemon=True, name="memory-migration")
        t.start()

    def _migrate(self) -> None:
        """One-time migration: embed all identity.json facts into ChromaDB."""
        if _MIGRATION_MARKER.exists():
            return
        try:
            _MIGRATION_MARKER.parent.mkdir(parents=True, exist_ok=True)
            identity_path = Path(__file__).parent.parent.parent / "identity.json"
            try:
                with open(identity_path) as f:
                    identity = json.load(f)
            except Exception:
                identity = {}

            facts = identity.get("learned_facts", [])
            if not isinstance(facts, list):
                facts = []

            if facts:
                import memory as _mem
                from plugins.memory.vector_store import ChromaStore
                store = ChromaStore.get()
                ts = time.time()
                batch = []
                ids = []
                for fact_text in facts:
                    if not isinstance(fact_text, str) or not fact_text.strip():
                        continue
                    fact_id = str(uuid.uuid4())
                    ids.append(fact_id)
                    batch.append(fact_text)
                    # Also save to SQLite
                    _mem.store_fact(fact_id, fact_text, {
                        "source": "migration",
                        "session_ids": [],
                        "timestamp": ts,
                        "recall_count": 0,
                    })
                    if len(batch) >= 50:
                        _flush_batch(store, ids, batch, ts)
                        ids, batch = [], []
                if batch:
                    _flush_batch(store, ids, batch, ts)
                logger.info("memory-migration: migrated %d facts to ChromaDB", len(facts))

            _MIGRATION_MARKER.write_text("ok")
        except Exception as exc:
            logger.warning("memory-migration failed: %s", exc)


def _flush_batch(store, ids: list, texts: list, ts: float) -> None:
    from plugins.memory.embedder import Embedder
    embedder = Embedder.get()
    embeddings = embedder.embed_batch(texts)
    for i, (fact_id, text, emb) in enumerate(zip(ids, texts, embeddings)):
        store._collection.upsert(
            ids=[fact_id],
            documents=[text],
            embeddings=[emb],
            metadatas=[{"source": "migration", "session_ids": "[]",
                        "timestamp": ts, "recall_count": 0}],
        )
