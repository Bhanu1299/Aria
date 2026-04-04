"""
summarizer.py — Aria response synthesizer (Phase 2B-1)

Two public functions:
  summarize(page_text, query, instructions) → spoken answer from web page text
  answer_knowledge(query)                   → spoken answer direct from LLM
"""

from __future__ import annotations

import json
import logging
import os

from groq import Groq

import config

logger = logging.getLogger(__name__)

_CLIENT: Groq | None = None
_MAX_PAGE_CHARS = 6000

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_WEB_SYSTEM_PROMPT = (
    "You are Aria, a voice assistant. "
    "Based on the web page content below, answer the user's question in 2 to 4 spoken sentences. "
    "No markdown, no bullet points, no lists, no headers — only plain prose. "
    "No preamble like 'Based on the page' or 'According to'. "
    "If the page does not contain the answer, say clearly: "
    "'I couldn't find a clear answer for that.'"
)

def _load_identity() -> dict:
    path = os.path.join(os.path.dirname(__file__), "identity.json")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _build_identity_context(identity: dict) -> str:
    """Build a compact natural-language identity description from identity.json."""
    parts = []
    if identity.get("name"):
        parts.append(f"The user's name is {identity['name']}.")
    if identity.get("email"):
        parts.append(f"Their email is {identity['email']}.")
    if identity.get("location"):
        parts.append(f"They are based in {identity['location']}.")
    if identity.get("linkedin"):
        parts.append(f"LinkedIn: {identity['linkedin']}.")
    if identity.get("github"):
        parts.append(f"GitHub: {identity['github']}.")
    skills = identity.get("skills", [])
    if skills:
        top = skills[:6]
        parts.append(f"Their top skills include: {', '.join(top)}.")
    education = identity.get("education", "")
    if education:
        parts.append(f"Education: {education}.")
    return " ".join(parts)


_IDENTITY = _load_identity()
_IDENTITY_CONTEXT = _build_identity_context(_IDENTITY)

_KNOWLEDGE_SYSTEM_PROMPT = (
    "You are Aria, a voice assistant on Mac. "
    + (_IDENTITY_CONTEXT + " " if _IDENTITY_CONTEXT else "")
    + "Answer concisely in 2 to 4 sentences maximum. "
    "No markdown, no bullet points, no lists — plain spoken sentences only. "
    "The response will be read aloud. "
    "If the question asks about personal details you don't have "
    "(item locations, the user's schedule, personal files, physical surroundings), "
    "say you don't have that information. "
    "Use the user information above to answer personal questions like "
    "'what's my name', 'what's my email', 'where am I based', 'what are my skills'. "
    "If asked what you can do or what your capabilities are, list only these: "
    "answer general knowledge questions, search the web, open apps and websites, "
    "control media playback (Apple Music and YouTube), control Mac settings via voice, "
    "give a morning briefing (weather, calendar, email, news), "
    "search for jobs on LinkedIn and Indeed, help apply to jobs, "
    "track submitted applications, check system info (battery, wifi, disk), "
    "take notes, set reminders, and run calculations. "
    "Do not claim capabilities beyond this list."
)

# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set in .env")
        _CLIENT = Groq(api_key=config.GROQ_API_KEY)
    return _CLIENT

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize(page_text: str, query: str, instructions: str = "") -> str:
    """
    Summarize page_text to answer query.

    Args:
        page_text:    Raw text extracted from the web page.
        query:        The user's original question / search terms.
        instructions: Specific extraction guidance from the router.

    Returns:
        A spoken-word answer string. Never raises.
    """
    if not page_text.strip():
        return "I couldn't load the page to find an answer."

    client = _get_client()

    user_content = (
        f"The user asked: {query}\n"
        f"What to extract: {instructions}\n\n"
        f"Page content:\n{page_text[:_MAX_PAGE_CHARS]}"
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _WEB_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        answer = response.choices[0].message.content.strip()
        logger.debug("Summarizer answer: %s", answer)
        return answer

    except Exception as exc:
        logger.error("Summarizer failed: %s", exc)
        return "I had trouble processing the answer, please try again."


def answer_knowledge(query: str) -> str:
    """
    Answer a knowledge query directly from the LLM — no browser, no fetching.

    Args:
        query: The user's question.

    Returns:
        A spoken-word answer string. Never raises.
    """
    client = _get_client()

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _KNOWLEDGE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.4,
            max_tokens=300,
        )
        answer = response.choices[0].message.content.strip()
        logger.debug("Knowledge answer: %s", answer)
        return answer

    except Exception as exc:
        logger.error("Knowledge handler failed: %s", exc)
        return "I had trouble answering that, please try again."
