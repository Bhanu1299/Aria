"""
skills/skill_loader.py — Aria Phase 3D: Skill auto-discovery and matching.

Scans the skills/ directory on startup. Each skill folder must contain:
  __init__.py  with TRIGGERS: list[str] and handle(command: str) -> str

Skills are matched before the Groq classifier in router.py — matched skills
execute with zero API latency.

Public API:
  load_skills()                          call once at startup
  match_skill(transcript: str)           returns handle fn or None
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_SKILLS_DIR: Path = Path(__file__).parent
_registry: dict[str, Callable] = {}


def load_skills() -> None:
    """
    Auto-discover and register all skills in the skills/ directory.
    Skips __pycache__ and any directory without an __init__.py.
    A skill that fails to import is logged and skipped — it does not
    prevent other skills from loading.
    """
    global _registry
    _registry = {}

    for skill_dir in _SKILLS_DIR.iterdir():
        if not skill_dir.is_dir():
            continue
        if skill_dir.name.startswith("_"):
            continue
        init_file = skill_dir / "__init__.py"
        if not init_file.exists():
            continue

        module_name = f"skills.{skill_dir.name}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, init_file)
            if spec is None or spec.loader is None:
                logger.error("Failed to create spec for skill %s", skill_dir.name)
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            triggers: list[str] = getattr(module, "TRIGGERS", [])
            handle_fn: Optional[Callable] = getattr(module, "handle", None)

            if not callable(handle_fn):
                logger.warning("Skill %s has no callable handle() — skipping", skill_dir.name)
                continue
            if not triggers:
                logger.warning("Skill %s has no TRIGGERS — skipping", skill_dir.name)
                continue

            for trigger in triggers:
                _registry[trigger.lower()] = handle_fn

            logger.info(
                "Loaded skill: %s  triggers: %s",
                skill_dir.name, triggers,
            )
        except Exception as exc:
            logger.error("Failed to load skill %s: %s", skill_dir.name, exc)


def match_skill(transcript: str) -> Optional[Callable]:
    """
    Return the handle function of the first skill whose trigger phrase
    appears anywhere in the transcript (case-insensitive). Returns None
    if no skill matches.
    """
    lower = transcript.lower()
    for trigger, fn in _registry.items():
        if trigger in lower:
            return fn
    return None
