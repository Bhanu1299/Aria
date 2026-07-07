"""plugins/memory/vector_store.py — ChromaDB vector store for Aria facts."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_CHROMA_DIR = str(Path.home() / ".aria" / "memory" / "vectors")
_COLLECTION = "aria_facts"


class ChromaStore:
    _instance: "ChromaStore | None" = None

    @classmethod
    def get(cls) -> "ChromaStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        import chromadb
        from plugins.memory.embedder import Embedder
        Path(_CHROMA_DIR).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=_CHROMA_DIR)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = Embedder.get()

    def upsert(self, fact_id: str, text: str, metadata: dict) -> None:
        """Embed text and upsert into ChromaDB. Never raises."""
        try:
            # ChromaDB metadata values must be scalar — serialize lists
            safe_meta = {}
            for k, v in metadata.items():
                safe_meta[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
            embedding = self._embedder.embed(text)
            self._collection.upsert(
                ids=[fact_id],
                documents=[text],
                embeddings=[embedding],
                metadatas=[safe_meta],
            )
        except Exception as exc:
            logger.warning("ChromaStore.upsert(%r) failed: %s", fact_id, exc)

    def search(self, query: str, k: int = 5, min_score: float = 0.6) -> list:
        """
        Return up to k facts with cosine similarity >= min_score.
        Each result is a dict: {id, text, score, metadata}.
        """
        try:
            count = self._collection.count()
            if count == 0:
                return []
            embedding = self._embedder.embed(query)
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(k, count),
                include=["documents", "metadatas", "distances"],
            )
            facts = []
            for i, doc in enumerate(results["documents"][0]):
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                # Convert to similarity: sim = 1 - distance
                distance = results["distances"][0][i]
                score = 1.0 - distance
                if score >= min_score:
                    meta = results["metadatas"][0][i] if results["metadatas"] else {}
                    facts.append({
                        "id": results["ids"][0][i],
                        "text": doc,
                        "score": round(score, 4),
                        "metadata": meta,
                    })
            return facts
        except Exception as exc:
            logger.warning("ChromaStore.search failed: %s", exc)
            return []

    def fetch(self, fact_id: str) -> dict | None:
        """Return a single fact by ID, or None if not found."""
        try:
            result = self._collection.get(
                ids=[fact_id],
                include=["documents", "metadatas"],
            )
            if not result["ids"]:
                return None
            return {
                "id": result["ids"][0],
                "text": result["documents"][0],
                "metadata": result["metadatas"][0],
            }
        except Exception as exc:
            logger.warning("ChromaStore.fetch(%r) failed: %s", fact_id, exc)
            return None

    def delete(self, fact_id: str) -> None:
        try:
            self._collection.delete(ids=[fact_id])
        except Exception as exc:
            logger.warning("ChromaStore.delete(%r) failed: %s", fact_id, exc)

    def increment_recall(self, fact_id: str) -> None:
        """Increment recall_count metadata field for a fact."""
        try:
            result = self._collection.get(
                ids=[fact_id],
                include=["documents", "metadatas", "embeddings"],
            )
            if not result["ids"]:
                return
            meta = result["metadatas"][0].copy()
            meta["recall_count"] = int(meta.get("recall_count", 0)) + 1
            self._collection.update(
                ids=[fact_id],
                metadatas=[meta],
            )
        except Exception as exc:
            logger.warning("ChromaStore.increment_recall(%r) failed: %s", fact_id, exc)

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception:
            return 0

    def get_all(self) -> list:
        """Return all facts as list of {id, text, metadata} dicts."""
        try:
            count = self._collection.count()
            if count == 0:
                return []
            result = self._collection.get(include=["documents", "metadatas"])
            facts = []
            for i, doc in enumerate(result["documents"]):
                facts.append({
                    "id": result["ids"][i],
                    "text": doc,
                    "metadata": result["metadatas"][i],
                })
            return facts
        except Exception as exc:
            logger.warning("ChromaStore.get_all failed: %s", exc)
            return []
