"""
media.py — Aria Phase 2D: Media Playback

Handles:
  1. Music app control via AppleScript (configurable MUSIC_APP)
  2. YouTube audio streaming via yt-dlp + ffplay (background, no screen)
  3. YouTube video via browser (open URL)

Public API:
  handle_media_command(transcript: str) -> str
    Groq sub-classifies the transcript, routes to the appropriate handler,
    returns a spoken-word response string.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess

from dotenv import load_dotenv
from groq import Groq

import config

load_dotenv()

MUSIC_APP = os.getenv("MUSIC_APP", "Music")

logger = logging.getLogger(__name__)

_CLIENT: Groq | None = None


def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set in .env")
        _CLIENT = Groq(api_key=config.GROQ_API_KEY)
    return _CLIENT


# ---------------------------------------------------------------------------
# Dependency check — runs at import time
# ---------------------------------------------------------------------------

def check_dependencies() -> None:
    missing = []
    if not shutil.which("yt-dlp"):
        missing.append("yt-dlp  →  brew install yt-dlp")
    if not shutil.which("ffplay"):
        missing.append("ffplay  →  brew install ffmpeg")
    for m in missing:
        print(f"[media] Missing dependency: {m}")


check_dependencies()


# ---------------------------------------------------------------------------
# AppleScript runner
# ---------------------------------------------------------------------------

def _run_applescript(script: str) -> tuple[bool, str]:
    """Run an AppleScript string. Returns (success, output). Never raises."""
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        err = result.stderr.strip() or result.stdout.strip()
        logger.debug("[media] osascript error (rc=%d): %s", result.returncode, err)
        return False, err
    except subprocess.TimeoutExpired:
        logger.debug("[media] osascript timed out")
        return False, "timeout"
    except Exception as exc:
        logger.debug("[media] osascript exception: %s", exc)
        return False, str(exc)


# ---------------------------------------------------------------------------
# Music app control
# ---------------------------------------------------------------------------

def music_now_playing() -> str:
    script = f'''tell application "{MUSIC_APP}"
    if player state is playing then
        set t to name of current track
        set a to artist of current track
        return t & " by " & a
    else
        return ""
    end if
end tell'''
    ok, out = _run_applescript(script)
    if ok and out.strip():
        return f"Now playing {out}."
    return "Nothing is playing right now."


def music_play(query: str) -> str:
    if not query.strip():
        ok, _ = _run_applescript(f'tell application "{MUSIC_APP}" to play')
        return music_now_playing() if ok else f"Couldn't resume {MUSIC_APP}."

    safe_query = query.replace('"', '\\"')

    # Layer 1 — search local library (works for downloaded/added tracks)
    script = f'''tell application "{MUSIC_APP}"
    activate
    set results to (search (library playlist 1) for "{safe_query}")
    if (count of results) > 0 then
        play item 1 of results
        return "ok"
    else
        return "not_found"
    end if
end tell'''
    try:
        ok, out = _run_applescript(script)
        if ok and out == "ok":
            return f"Playing {query}."
    except Exception as exc:
        logger.debug("[media] Library search failed: %s", exc)

    # Layer 2 — song not in local library, stream via YouTube audio (background, no screen)
    logger.debug("[media] Not in local library — YouTube audio fallback: %r", query)
    return play_youtube_audio(query)


def music_pause() -> str:
    ok, _ = _run_applescript(f'tell application "{MUSIC_APP}" to pause')
    return "Paused." if ok else f"Couldn't pause {MUSIC_APP}."


def music_resume() -> str:
    ok, _ = _run_applescript(f'tell application "{MUSIC_APP}" to play')
    return music_now_playing() if ok else f"Couldn't resume {MUSIC_APP}."


def music_skip() -> str:
    ok, _ = _run_applescript(f'tell application "{MUSIC_APP}" to next track')
    if ok:
        return music_now_playing() or "Skipped."
    return f"Couldn't skip in {MUSIC_APP}."


# ---------------------------------------------------------------------------
# YouTube playback
# ---------------------------------------------------------------------------

def play_youtube_audio(query: str) -> str:
    """Stream audio-only in background via yt-dlp + ffplay. No download, no screen."""
    if not shutil.which("yt-dlp"):
        return "yt-dlp is not installed. Run: brew install yt-dlp"
    if not shutil.which("ffplay"):
        return "ffplay is not installed. Run: brew install ffmpeg"

    try:
        # Get stream URL
        url_result = subprocess.run(
            ["yt-dlp", f"ytsearch1:{query}", "--get-url", "-f", "bestaudio"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        url = url_result.stdout.strip().split("\n")[0]
        if not url:
            return "Couldn't find that on YouTube."

        # Get title for spoken response
        title_result = subprocess.run(
            ["yt-dlp", f"ytsearch1:{query}", "--get-title"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        title = title_result.stdout.strip().split("\n")[0] or query

        # Stream silently in background
        subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return f"Playing {title} from YouTube."

    except subprocess.TimeoutExpired:
        return "YouTube took too long to respond."
    except Exception as exc:
        logger.error("[media] play_youtube_audio error: %s", exc)
        return "Something went wrong playing YouTube audio."


def play_youtube_video(query: str) -> str:
    """Get video ID via yt-dlp, open in default browser."""
    if not shutil.which("yt-dlp"):
        return "yt-dlp is not installed. Run: brew install yt-dlp"

    try:
        id_result = subprocess.run(
            ["yt-dlp", f"ytsearch1:{query}", "--get-id"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        video_id = id_result.stdout.strip().split("\n")[0]
        if not video_id:
            return "Couldn't find that on YouTube."

        url = f"https://www.youtube.com/watch?v={video_id}"
        subprocess.Popen(["open", url])
        return f"Opening YouTube for {query}."

    except subprocess.TimeoutExpired:
        return "YouTube took too long to respond."
    except Exception as exc:
        logger.error("[media] play_youtube_video error: %s", exc)
        return "Something went wrong opening the video."


def stop_youtube() -> None:
    """Kill any running ffplay processes."""
    try:
        subprocess.run(["pkill", "-f", "ffplay"], capture_output=True)
    except Exception as exc:
        logger.debug("[media] stop_youtube error: %s", exc)


# ---------------------------------------------------------------------------
# Groq sub-classifier
# ---------------------------------------------------------------------------

_MEDIA_CLASSIFY_SYSTEM = f"""You are a media command parser for a Mac voice assistant.
The configured music app is "{MUSIC_APP}".
Parse the voice command and return ONLY a JSON object — no markdown, no explanation.

Schema:
{{
  "action": "play_music" | "play_youtube_audio" | "play_youtube_video" | "pause" | "resume" | "skip" | "now_playing" | "stop",
  "query": "<search string or empty string>",
  "platform": "music_app" | "youtube" | "auto"
}}

Rules:
- "play X" with no platform → action=play_music, platform=auto
- "play X on YouTube" / "play X on youtube" → action=play_youtube_audio, platform=youtube
- "watch X" / "show me X" / "video" anywhere in command → action=play_youtube_video, platform=youtube
- "pause" / "stop music" / "stop {MUSIC_APP}" → action=pause, platform=music_app
- "resume" / "unpause" / "continue" → action=resume, platform=music_app
- "skip" / "next" / "next track" / "next song" → action=skip, platform=music_app
- "what's playing" / "now playing" / "what song is this" → action=now_playing, platform=auto
- "stop" with no music/app context → action=stop, platform=auto (covers YouTube + music app)
- Non-music YouTube content (tutorials, podcasts, reviews, how-to) → platform=youtube, action=play_youtube_audio
"""


def _sub_classify(transcript: str) -> dict:
    """Groq call → structured media intent."""
    client = _get_client()
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": _MEDIA_CLASSIFY_SYSTEM},
            {"role": "user", "content": transcript.strip()},
        ],
        temperature=0.1,
        max_tokens=100,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw.strip())
    parsed = json.loads(raw)
    if "action" not in parsed:
        raise ValueError(f"No action in media classifier response: {parsed}")
    # Normalise query
    parsed["query"] = str(parsed.get("query") or "").strip()
    return parsed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def handle_media_command(transcript: str) -> str:
    """
    Sub-classify transcript via Groq, route to music or YouTube handler.
    Returns a spoken-word response string. Never raises.
    """
    try:
        intent = _sub_classify(transcript)
    except Exception as exc:
        logger.error("[media] Sub-classify failed: %s", exc)
        return "I didn't understand that media command, please try again."

    action = intent.get("action", "")
    query = intent.get("query", "")
    platform = intent.get("platform", "auto")

    # Fallback: if LLM returned empty query for a play action, extract from transcript
    if not query and action in ("play_music", "play_youtube_audio", "play_youtube_video"):
        # Strip leading "play"/"watch"/"listen to" verb and trailing platform hints
        extracted = re.sub(
            r"(?i)^(play|watch|listen\s+to|put\s+on)\s+",
            "",
            transcript.strip(),
        )
        extracted = re.sub(
            r"(?i)\s+(on\s+)?(youtube|apple\s+music|spotify|music|tidal)\s*\.?$",
            "",
            extracted,
        ).strip()
        if extracted:
            query = extracted
            logger.debug("[media] query extracted from transcript: %r", query)

    logger.debug("[media] action=%r query=%r platform=%r", action, query, platform)
    print(f"[media] action={action!r} query={query!r} platform={platform!r}")

    try:
        if action == "play_music" or (action == "play_youtube_audio" and platform == "music_app"):
            return music_play(query)
        elif action == "play_youtube_audio":
            return play_youtube_audio(query)
        elif action == "play_youtube_video":
            return play_youtube_video(query)
        elif action == "pause":
            return music_pause()
        elif action == "resume":
            return music_resume()
        elif action == "skip":
            return music_skip()
        elif action == "now_playing":
            return music_now_playing()
        elif action == "stop":
            stop_youtube()
            _run_applescript(f'tell application "{MUSIC_APP}" to pause')
            return "Stopped."
        else:
            return f"I don't know how to do {action!r} yet."
    except Exception as exc:
        logger.error("[media] Handler error for action=%r: %s", action, exc)
        return "Something went wrong with the media command."
