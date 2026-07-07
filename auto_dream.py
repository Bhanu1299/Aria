"""
auto_dream.py — Aria Phase 4: Background Memory Consolidation (Task 10)

After every 5 commands, fires a background Groq call that reads session_notes
and learned_facts from identity.json, consolidates them into cleaner organized
versions, and writes both back. Deduplicates facts, trims noise.
Counter is persisted in SQLite via memory.py.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time

from llm import llm_client
import memory

logger = logging.getLogger(__name__)

_CONSOLIDATE_EVERY = 5
_IDENTITY_PATH = os.path.join(os.path.dirname(__file__), "identity.json")

_SYSTEM_PROMPT = (
    "You are a memory consolidation assistant for a voice agent called Aria. "
    "Given session notes and known user facts, return a JSON object with two keys: "
    "'session_notes' (3-5 bullet points of the most important recent activity) and "
    "'learned_facts' (deduplicated array of durable user facts, max 50, newest kept). "
    "Return ONLY valid JSON. No markdown."
)


def _load_identity() -> dict:
    try:
        with open(_IDENTITY_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_identity(identity: dict) -> None:
    try:
        dir_ = os.path.dirname(_IDENTITY_PATH)
        with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False, suffix=".tmp") as f:
            json.dump(identity, f, indent=2)
            tmp = f.name
        os.replace(tmp, _IDENTITY_PATH)
    except Exception as exc:
        logger.warning("auto_dream: failed to save identity: %s", exc)


def consolidate() -> None:
    """Synchronous consolidation. Reads notes + facts, rewrites both. Never raises."""
    try:
        notes = memory.get_session_notes()
        identity = _load_identity()
        facts = identity.get("learned_facts", [])

        if not notes and not facts:
            return

        resp = llm_client.complete(
            messages=[
                {"role": "user", "content": (
                    f"Session notes:\n{notes}\n\n"
                    f"Known facts:\n{json.dumps(facts)}"
                )},
            ],
            tier="fast",
            system=_SYSTEM_PROMPT,
            max_tokens=600,
        )
        raw = resp.text.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(raw)

        new_notes = result.get("session_notes", "")
        new_facts = result.get("learned_facts", facts)

        if isinstance(new_notes, list):
            new_notes = "\n".join(f"- {n}" if not n.startswith("-") else n for n in new_notes)

        if new_notes:
            memory.clear_session_notes()
            memory.store_session_notes(new_notes)

        if isinstance(new_facts, list) and identity:
            identity["learned_facts"] = new_facts[:50]
            _save_identity(identity)

        memory.reset_command_count()
        _promote_top_facts(identity)
        logger.debug("auto_dream.consolidate: complete")

    except Exception as exc:
        logger.warning("auto_dream.consolidate failed: %s", exc)


def _promote_top_facts(identity: dict) -> None:
    """
    Score all ChromaDB facts: recall_count*0.4 + recency*0.3 + session_spread*0.3.
    Top 5 qualifying facts (recall_count>=3, unique_sessions>=2) are promoted to
    identity.json["promoted_facts"] as permanent memory.
    """
    try:
        from plugins.memory.vector_store import ChromaStore
        store = ChromaStore.get()
        all_facts = store.get_all()
        if not all_facts:
            return

        now = time.time()
        scored = []
        for f in all_facts:
            meta = f.get("metadata", {})
            recall = int(meta.get("recall_count", 0))
            sessions_raw = meta.get("session_ids", "[]")
            try:
                sessions = json.loads(sessions_raw) if isinstance(sessions_raw, str) else sessions_raw
            except Exception:
                sessions = []
            unique_sessions = len(sessions) if isinstance(sessions, list) else 0

            if recall < 3 or unique_sessions < 2:
                continue

            ts = float(meta.get("timestamp", 0))
            age_days = (now - ts) / 86400
            recency = max(0.0, 1.0 - age_days / 30)
            session_spread = min(1.0, unique_sessions / 5)
            score = recall * 0.4 + recency * 0.3 + session_spread * 0.3
            scored.append((score, f["text"]))

        if not scored:
            return

        scored.sort(key=lambda x: x[0], reverse=True)
        today = __import__("datetime").date.today().isoformat()
        promoted = [{"fact": text, "promoted_at": today} for _, text in scored[:5]]

        identity["promoted_facts"] = promoted
        _save_identity(identity)
        logger.debug("auto_dream: promoted %d facts to identity.json", len(promoted))

    except Exception as exc:
        logger.warning("auto_dream._promote_top_facts failed: %s", exc)


def maybe_consolidate_async(transcript: str, answer: str) -> None:
    """Increment counter; if interval reached, consolidate in a daemon thread."""
    count = memory.increment_command_count()
    if count % _CONSOLIDATE_EVERY == 0:
        t = threading.Thread(target=consolidate, daemon=True, name="auto-dream")
        t.start()
