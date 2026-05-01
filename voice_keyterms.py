from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

_IDENTITY_PATH = os.path.join(os.path.dirname(__file__), "identity.json")

_STATIC_TERMS = [
    "LinkedIn", "Indeed", "Playwright", "Groq", "Whisper",
    "Python", "FastAPI", "PostgreSQL", "Docker", "GitHub",
    "LangChain", "RAG", "Pinecone", "FAISS", "Anthropic",
    "browser task", "job search", "apply", "resume",
    "software engineer", "full stack", "GenAI",
]


def _load_identity() -> dict:
    try:
        with open(_IDENTITY_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def build_prompt() -> str:
    """
    Return a short initial_prompt string for Whisper containing domain vocab.
    Whisper treats this as preceding context, biasing toward these spellings.
    """
    identity = _load_identity()
    dynamic: list[str] = []

    skills = identity.get("skills", [])
    dynamic.extend(skills[:10])

    target_roles = identity.get("target_roles", [])
    dynamic.extend(target_roles[:5])

    name = identity.get("name", "")
    if name:
        dynamic.append(name)

    all_terms = _STATIC_TERMS + [t for t in dynamic if t not in _STATIC_TERMS]
    return ", ".join(all_terms[:40]) + "."
