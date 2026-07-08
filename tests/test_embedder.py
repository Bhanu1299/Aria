"""tests/test_embedder.py — Embedder unit tests."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_mock_model():
    import numpy as np
    m = MagicMock()
    m.encode.side_effect = lambda texts: (
        np.array([[0.1] * 384] * len(texts)) if isinstance(texts, list)
        else np.array([0.1] * 384)
    )
    return m


def test_embed_returns_list_of_floats():
    with patch("sentence_transformers.SentenceTransformer", return_value=_make_mock_model()):
        from aria.plugins.memory.embedder import Embedder
        Embedder._instance = None
        emb = Embedder.get()
        result = emb.embed("hello world")
    assert isinstance(result, list)
    assert len(result) == 384
    assert all(isinstance(v, float) for v in result)


def test_embed_batch_returns_list_of_lists():
    with patch("sentence_transformers.SentenceTransformer", return_value=_make_mock_model()):
        from aria.plugins.memory.embedder import Embedder
        Embedder._instance = None
        emb = Embedder.get()
        result = emb.embed_batch(["hello", "world"])
    assert isinstance(result, list)
    assert len(result) == 2
    assert len(result[0]) == 384


def test_singleton_returns_same_instance():
    with patch("sentence_transformers.SentenceTransformer", return_value=_make_mock_model()):
        from aria.plugins.memory.embedder import Embedder
        Embedder._instance = None
        a = Embedder.get()
        b = Embedder.get()
    assert a is b
