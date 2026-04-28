"""
scene_executor.py — Aria Phase 3D Session 2: Scene / macro execution engine.

Loads scenes.json, matches transcript against trigger phrases, and runs all
actions in a scene. open_app and open_url actions run concurrently so
"Daddy's Home" opens everything at once instead of sequentially.

Public API:
  load_scenes(path)            load scenes from JSON (called once at startup)
  match_scene(transcript)      return matching scene dict or None
  run_scene(scene)             execute all actions; concurrent open_app/open_url
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_scenes: list[dict] = []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_scenes(path: str = "scenes.json") -> None:
    """Load scenes from JSON file. Safe to call multiple times (re-loads)."""
    global _scenes
    scene_path = Path(path)
    if not scene_path.is_absolute():
        scene_path = Path(__file__).parent / path
    try:
        data = json.loads(scene_path.read_text())
        _scenes = data.get("scenes", [])
        logger.debug("Loaded %d scenes from %s", len(_scenes), scene_path)
    except FileNotFoundError:
        logger.warning("scenes.json not found at %s — no scenes loaded", scene_path)
        _scenes = []
    except Exception as exc:
        logger.error("Failed to load scenes.json: %s", exc)
        _scenes = []


def match_scene(transcript: str) -> Optional[dict]:
    """
    Check if transcript matches any scene trigger phrase.
    Returns the first matching scene dict, or None.
    Matching is case-insensitive substring check.
    """
    lower = transcript.lower().strip()
    for scene in _scenes:
        for trigger in scene.get("triggers", []):
            if trigger.lower() in lower:
                logger.debug("Scene matched: %r (trigger: %r)", scene["name"], trigger)
                return scene
    return None


def run_scene(scene: dict) -> None:
    """
    Execute all actions in a scene.

    open_app and open_url actions at the start of the scene are submitted
    to a thread pool and run concurrently. All other action types are
    executed sequentially in declaration order after the concurrent batch.
    """
    actions = scene.get("actions", [])
    scene_name = scene.get("name", "unnamed")
    logger.debug("Running scene: %r (%d actions)", scene_name, len(actions))

    # Split into concurrent batch (open_app / open_url) and sequential remainder
    concurrent_actions = [a for a in actions if a.get("type") in ("open_app", "open_url")]
    sequential_actions = [a for a in actions if a.get("type") not in ("open_app", "open_url")]

    # Launch all open_app / open_url actions in parallel
    if concurrent_actions:
        with ThreadPoolExecutor(max_workers=len(concurrent_actions)) as pool:
            futures = [pool.submit(_execute_action, a, scene_name) for a in concurrent_actions]
            for f in as_completed(futures):
                exc = f.exception()
                if exc:
                    logger.error("Scene %r concurrent action failed: %s", scene_name, exc)

    # Run everything else sequentially
    for action in sequential_actions:
        try:
            _execute_action(action, scene_name)
        except Exception as exc:
            logger.error("Scene %r sequential action failed: %s", scene_name, exc)


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

def _execute_action(action: dict, scene_name: str) -> None:
    action_type = action.get("type", "")

    if action_type == "open_app":
        _open_app(action.get("app", ""))

    elif action_type == "open_url":
        _open_url(action.get("url", ""))

    elif action_type == "briefing":
        _run_briefing()

    elif action_type == "job_alerts":
        _run_job_alerts()

    elif action_type == "speak":
        _speak(action.get("text", ""))

    elif action_type == "pause_music":
        _pause_music()

    elif action_type == "lock_mac":
        _lock_mac()

    elif action_type == "play_hype_music":
        _play_hype_music(action.get("duration_seconds", 10))

    else:
        logger.warning("Scene %r: unknown action type %r", scene_name, action_type)


def _open_app(app_name: str) -> None:
    if not app_name:
        return
    try:
        subprocess.run(["open", "-a", app_name], check=False, timeout=10)
    except Exception as exc:
        logger.error("open_app(%r) failed: %s", app_name, exc)


def _open_url(url: str) -> None:
    if not url:
        return
    try:
        subprocess.run(["open", url], check=False, timeout=10)
    except Exception as exc:
        logger.error("open_url(%r) failed: %s", url, exc)


def _run_briefing() -> None:
    try:
        import briefing as _briefing
        import speaker as _speaker_mod
        text = _briefing.build_briefing()
        _speaker_mod.speak(text)
    except Exception as exc:
        logger.error("scene briefing action failed: %s", exc)


def _run_job_alerts() -> None:
    try:
        import jobs as _jobs
        import memory as _memory
        import speaker as _speaker_mod
        last_query = _memory.get_last_search() or "software engineer"
        results = _jobs.search_jobs(last_query)
        _memory.store_jobs(results)
        text = _jobs.format_spoken_results(results)
        _speaker_mod.speak(text)
    except Exception as exc:
        logger.error("scene job_alerts action failed: %s", exc)


def _speak(text: str) -> None:
    if not text:
        return
    try:
        import speaker as _speaker_mod
        _speaker_mod.speak(text)
    except Exception as exc:
        logger.error("scene speak action failed: %s", exc)


def _pause_music() -> None:
    script = 'tell application "Music" to pause'
    try:
        subprocess.run(["osascript", "-e", script], check=False, timeout=5)
    except Exception as exc:
        logger.error("pause_music failed: %s", exc)


def _lock_mac() -> None:
    try:
        subprocess.run(["pmset", "sleepnow"], check=False, timeout=5)
    except Exception as exc:
        logger.error("lock_mac failed: %s", exc)


def _play_hype_music(duration_seconds: int = 10) -> None:
    """Search for a hype track via yt-dlp and stream it with ffplay for N seconds."""
    try:
        import shutil
        if not shutil.which("yt-dlp") or not shutil.which("ffplay"):
            logger.warning("play_hype_music: yt-dlp or ffplay not installed — skipping")
            return

        # Get stream URL for a hype track
        result = subprocess.run(
            ["yt-dlp", "--get-url", "--format", "bestaudio",
             "ytsearch1:hype music get pumped"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        stream_url = result.stdout.strip()
        if not stream_url:
            logger.warning("play_hype_music: yt-dlp returned no URL")
            return

        # Stream for N seconds then kill
        proc = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", stream_url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        def _stop_after():
            import time
            time.sleep(duration_seconds)
            proc.terminate()

        threading.Thread(target=_stop_after, daemon=True).start()

    except Exception as exc:
        logger.error("play_hype_music failed: %s", exc)


# Load scenes on module import
load_scenes()
