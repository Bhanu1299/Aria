"""llm/models.py — single source of truth for Groq model IDs.

Groq retires models without much notice (llama-3.3-70b-versatile,
llama-3.1-8b-instant and llama-4-scout all disappeared by 2026-10). When that
happens, change the default here or set the env var — nothing else.
Live list: groq.Groq().models.list()
"""
from __future__ import annotations

import os

GROQ_TEXT = os.getenv("ARIA_GROQ_TEXT_MODEL", "openai/gpt-oss-120b")
GROQ_FAST = os.getenv("ARIA_GROQ_FAST_MODEL", "openai/gpt-oss-20b")
GROQ_VISION = os.getenv("ARIA_GROQ_VISION_MODEL", "qwen/qwen3.8-27b")


def groq_params(model: str) -> dict:
    """Extra chat.completions kwargs a model needs.

    gpt-oss reasons before answering and the reasoning counts against
    max_tokens — at Aria's small budgets (60–300) it returns EMPTY content
    unless reasoning effort is turned down.
    """
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low"}
    return {}
