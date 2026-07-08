"""plugins/memory/embedder.py — Singleton sentence-transformer embedder."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class Embedder:
    _instance: "Embedder | None" = None

    @classmethod
    def get(cls) -> "Embedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        logger.info("embedder: loading all-MiniLM-L6-v2")
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, text: str) -> list:
        return self._model.encode(text).tolist()

    def embed_batch(self, texts: list) -> list:
        return self._model.encode(texts).tolist()
