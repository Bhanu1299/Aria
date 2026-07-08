#!/usr/bin/env python3
"""
Aria — Background Mac Voice Agent

Pipeline:
  hotkey / wake word → record → Whisper transcribe →
  tool-loop Agent (plugins: core, memory, messaging, media, screen,
  health, productivity; llm/ failover Claude↔Groq) → speaker.say() →
  conversation.hold() follow-up window → async extractors
  (session notes, memory, auto_dream, suggester).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MKL / OpenMP guard — must be set before ANY library that loads OpenMP
# (ctranslate2, numpy, sounddevice, playwright can all trigger duplicate-lib
#  detection on macOS Python 3.9 which calls abort() if not suppressed).
# ---------------------------------------------------------------------------
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
import re as _re
import signal
import sys
import time
import threading
import warnings
from typing import Optional

# ---------------------------------------------------------------------------
# Log suppression — silence all noisy third-party libraries before they load.
# Only keep: startup lines, [Aria] lines, genuine ERROR/WARNING output.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="[%(name)s] %(levelname)s: %(message)s",
)
# Silence specific noisy loggers
for _noisy_logger in [
    "httpx", "httpcore",
    "faster_whisper", "ctranslate2",
    "browser", "agent_browser",
    "jobs", "briefing", "memory",
    "skills.skill_loader",
    "menubar",
    "urllib3",
    "playwright",
    "asyncio",
    "scene_executor",
]:
    logging.getLogger(_noisy_logger).setLevel(logging.ERROR)

# Suppress urllib3 / OpenSSL warning
warnings.filterwarnings("ignore", ".*OpenSSL.*")
warnings.filterwarnings("ignore", ".*NotOpenSSLWarning.*")

logger = logging.getLogger(__name__)

import config
from transcriber import Transcriber
from voice_capture import VoiceCapture
from speaker import Speaker, ThinkingAck
from browser import BrowserExecutor
from menubar import AriaMenuBar
from hotkey import HotkeyListener
from wake_word import WakeWordListener
import memory
import agent_browser
import session_notes
import memory_extractor
import auto_dream
import away_summary
import voice_keyterms
import prompt_suggester
import prevent_sleep
import tips
import conversation
import listening_indicator
from sleep_guard import SleepGuard
from tool import ToolRegistry
from agent import Agent
from plugin import PluginContext
import plugins as plugin_packs
import flight_recorder

# Build domain vocab hint prompt once at module load — passed to every transcribe() call
_KEYTERMS_PROMPT = voice_keyterms.build_prompt()
# vision is imported lazily inside _vision_fallback() to keep it off the
# startup critical path — Playwright + ctranslate2 + vision all loading at
# once on Python 3.9 macOS can trigger OpenMP duplicate-lib abort()

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_processing = threading.Event()   # set = currently processing, clear = idle
_recording_active = threading.Event()  # set = start_recording() was called
sleep_guard = SleepGuard()         # keeps Mac awake during long tasks

menubar: AriaMenuBar = None
voice_capture: VoiceCapture = None
transcriber_instance: Transcriber = None
speaker: Speaker = None
browser: BrowserExecutor = None
hotkey_listener: HotkeyListener = None
_agent: Agent = None


# ---------------------------------------------------------------------------
# Shutdown handler
# ---------------------------------------------------------------------------
def _shutdown(signum, frame):
    print("\nAria stopping...")
    if hotkey_listener is not None:
        try:
            hotkey_listener.stop()
        except Exception as e:
            print(f"Error stopping hotkey listener: {e}")
    if browser is not None:
        try:
            browser.stop()
        except Exception as e:
            print(f"Error stopping browser: {e}")
    try:
        agent_browser.close()
    except Exception as e:
        print(f"Error closing agent browser: {e}")
    print("Aria stopped.")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Hotkey callbacks
# ---------------------------------------------------------------------------
def on_press():
    # Always stop ongoing speech first — hotkey acts as interrupt
    if speaker is not None:
        speaker.stop()

    if _processing.is_set():
        return
    # Mark processing NOW so wake word can't fire while we're recording
    _processing.set()
    menubar.set_state("LISTENING")
    listening_indicator.show("Listening...")
    voice_capture.start_recording(auto_stop=True, on_auto_stop=on_release)
    _recording_active.set()  # set AFTER start_recording() returns — prevents on_release racing in


def _run_agent_turn(transcript: str) -> str:
    """
    One agent turn: speaks a brief acknowledgment if the run is slow, and
    records the outcome to the flight recorder. Never raises (Agent.run
    catches internally).
    """
    _t0 = time.time()
    ack = ThinkingAck(speaker)
    ack.start()
    try:
        answer = _agent.run(transcript)
    finally:
        ack.cancel()
    flight_recorder.record(
        transcript, answer, time.time() - _t0,
        tools=getattr(_agent, "last_run_tools", []),
    )
    return answer


def _run_followup_turn(text: str) -> str:
    """Agent turn for conversation-mode follow-ups, with async extractors."""
    answer = _run_agent_turn(text)
    if answer:
        session_notes.extract_async(text, answer)
        memory_extractor.extract_async(text, answer)
        away_summary.update_last_active(answer[:120])
    return answer


def handle_command(transcript: str) -> None:
    """
    Full command pipeline. Called by hotkey path and wake word path.
    transcript is always a real transcribed string — WakeWordListener handles
    its own recording + transcription before calling here.

    Never raises. Guards against concurrent execution via _processing event.
    """
    if _processing.is_set():
        return   # already handling a command — ignore concurrent trigger
    _processing.set()

    try:
        sleep_guard.acquire()
        prevent_sleep.start()
        # Check for away gap before processing — speak recap if 30+ min idle
        away_summary.check_and_speak(speaker)
        # Validate — reject silence / noise
        _cleaned = _re.sub(r'[\s\.\,\!\?\-\[\]]+', '', transcript)
        _SINGLE_WORD_COMMANDS = {
            "mute", "unmute", "pause", "stop", "skip", "next",
            "resume", "play", "screenshot",
        }
        _is_single_word = transcript.lower().strip() in _SINGLE_WORD_COMMANDS
        if not _cleaned or (len(transcript.split()) < 2 and not _is_single_word):
            speaker.say("I didn't catch that, please try again.")
            menubar.set_state("IDLE")
            return

        print(f"[Aria] Transcribed: {transcript!r}")
        menubar.set_state("THINKING")

        # Pre-check: ordinal job follow-ups ("tell me more about the second job")
        _t0 = time.time()
        followup = _check_jobs_followup(transcript)
        if followup is not None:
            print(f"[Aria] Jobs follow-up: {followup!r}")
            answer = followup
            flight_recorder.record(transcript, answer, time.time() - _t0)
        else:
            print(f"[Aria] Agent running: {transcript!r}")
            answer = _run_agent_turn(transcript)

        print(f"[Aria] Answer: {answer[:80]!r}")
        if answer:
            speaker.say(answer)
            session_notes.extract_async(transcript, answer)
            memory_extractor.extract_async(transcript, answer)
            auto_dream.maybe_consolidate_async(transcript, answer)
            away_summary.update_last_active(answer[:120])
            # Jarvis-style follow-ups: mic re-opens for a short window after
            # the answer. Runs BEFORE suggester/tips so their async speech
            # can't leak into the open mic.
            if conversation.enabled():
                conversation.hold(
                    speaker=speaker,
                    transcriber=transcriber_instance,
                    menubar=menubar,
                    keyterms_prompt=_KEYTERMS_PROMPT,
                    run_turn=_run_followup_turn,
                )
            prompt_suggester.suggest_async("", answer, speaker)
            _count = memory.increment_command_count()
            tips.maybe_speak_tip(_count, speaker)

        menubar.set_state("DONE")
        time.sleep(1)
        menubar.set_state("IDLE")

    except Exception as e:
        print(f"[Aria] Error during processing: {e}")
        flight_recorder.record(transcript, "", 0.0, error=str(e))
        try:
            speaker.say("Something went wrong, please try again.")
        except Exception as say_err:
            print(f"[Aria] Error speaking error message: {say_err}")
        menubar.set_state("IDLE")
    finally:
        _processing.clear()
        sleep_guard.release()
        prevent_sleep.stop()


def _process_release():
    """
    Hotkey release handler — transcribes then calls handle_command().
    Note: _processing is already set by on_press(). handle_command() will
    detect it is set and skip its own set() — but will still run the pipeline
    and clear it at the end. We clear it here only on early error.
    """
    listening_indicator.hide()
    try:
        # Call get_audio_array() BEFORE stop_recording() (stop clears _chunks)
        audio_array = voice_capture.get_audio_array()
        wav_path = voice_capture.stop_recording()
        if audio_array is not None:
            question = transcriber_instance.transcribe_numpy(audio_array, initial_prompt=_KEYTERMS_PROMPT)
            if not question:
                # fallback to file path if numpy path returns empty
                question = transcriber_instance.transcribe(wav_path, initial_prompt=_KEYTERMS_PROMPT)
        else:
            question = transcriber_instance.transcribe(wav_path, initial_prompt=_KEYTERMS_PROMPT)
        # handle_command checks _processing.is_set() before setting it.
        # Since on_press already set it, temporarily clear so handle_command
        # can proceed (it will re-set immediately).
        _processing.clear()
        handle_command(question)
    except Exception as e:
        print(f"[Aria] Transcription error: {e}")
        try:
            speaker.say("Something went wrong, please try again.")
        except Exception:
            pass
        menubar.set_state("IDLE")
        _processing.clear()


_ORDINAL_MAP = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4, "fifth": 5, "5th": 5,
}
_JOB_FOLLOWUP_RE = _re.compile(
    r"\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)\b",
    _re.IGNORECASE,
)
_APPLY_RE = _re.compile(
    r"\b(open|go\s+to|visit|navigate)\b",
    _re.IGNORECASE,
)
# Apply commands must pass through to the agent — never intercept them here.
_IS_APPLY_RE = _re.compile(r"\b(apply|apply\s+to|apply\s+for|submit)\b", _re.IGNORECASE)


def _check_jobs_followup(question: str) -> Optional[str]:
    """
    Intercept ordinal job-reference questions before routing.

    Matches any question with an ordinal ("first", "second", etc.) while
    jobs are saved in session memory. Uses vision to screenshot the job page
    and answer the actual question, or opens the URL for "open"/"visit".

    Apply commands are deliberately excluded — they fall through to the agent.
    """
    # Only intercept if we have jobs saved from a prior search
    if not memory.get_persistent("last_jobs"):
        return None
    # Let apply/submit commands fall through to the agent
    if _IS_APPLY_RE.search(question):
        return None
    m = _JOB_FOLLOWUP_RE.search(question)
    if not m:
        return None
    n = _ORDINAL_MAP.get(m.group(1).lower())
    if n is None:
        return None
    job = memory.get_job_by_index(n)
    if job is None:
        return (
            f"I don't have a {m.group(1)} job saved from this session. "
            "Try searching for jobs first."
        )

    url = job.get("url", "").strip()

    # "Apply for the third job" / "open the first listing" → open in browser
    if url and _APPLY_RE.search(question):
        import webbrowser
        webbrowser.open(url)
        return f"Opening the job listing for {job['title']} at {job['company']}."

    # Detail questions ("tell me more", "what's the salary") → vision screenshot
    if url:
        try:
            return _vision_fallback(url, question)
        except Exception as exc:
            print(f"[Aria] Vision fallback failed for job URL: {exc}")

    # Fallback: return what we have cached.
    detail = f"{job['title']} at {job['company']}"
    if job.get("location"):
        detail += f", located in {job['location']}"
    if job.get("posted"):
        detail += f", posted {job['posted']}"
    if job.get("platform"):
        detail += f". Listed on {job['platform']}."
    else:
        detail += "."
    return detail


def _vision_fallback(url: str, query: str) -> str:
    """Lazy-import vision and call read_screen. Isolated so startup never loads vision."""
    import vision as _vision  # noqa: PLC0415 — intentional lazy import
    return _vision.read_screen(url, query)


def on_release():
    # Guard: ignore spurious releases that have no matching on_press
    if not _recording_active.is_set():
        return
    _recording_active.clear()
    threading.Thread(target=_process_release, daemon=True).start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    global menubar, voice_capture, transcriber_instance, speaker, browser, hotkey_listener

    print("Aria starting up (Phase 2E)...")
    print(
        "[Aria] App control tip: For full app control, grant accessibility access:\n"
        "       System Settings → Privacy & Security → Accessibility → add Terminal"
    )

    # 1. Check permissions
    if not config.check_permissions():
        print("Error: Required permissions not granted.")
        sys.exit(1)

    # 2. Signal handlers
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # 3. Preload Whisper (loads inside Transcriber's own worker thread)
    print("Loading Whisper model (first run may download ~140 MB)...")
    transcriber_instance = Transcriber()  # blocks until model is ready
    voice_capture = VoiceCapture()
    speaker = Speaker()

    # 5. Start headless browser
    browser = BrowserExecutor()
    browser.start()

    # 6. Menu bar
    menubar = AriaMenuBar()

    # 6b. Discover and load plugins (two phases around Agent creation).
    # Drop a PluginBase subclass into plugins/<name>/ and it loads here —
    # no core edits. A failing plugin is skipped, never fatal.
    global _agent
    _registry = ToolRegistry()
    ctx = PluginContext(
        browser=browser,
        speaker=speaker,
        voice_capture=voice_capture,
        transcriber=transcriber_instance,
        menubar=menubar,
        keyterms_prompt=_KEYTERMS_PROMPT,
    )
    plugin_classes = plugin_packs.discover()

    def _load_plugins(classes) -> None:
        for cls in classes:
            try:
                cls.from_context(ctx).register(_registry)
                print(f"[Aria] Plugin loaded: {cls.__name__}")
            except Exception as exc:
                print(f"[Aria] Plugin {cls.__name__} failed to load — skipped: {exc}")

    _load_plugins([c for c in plugin_classes if not c.requires_agent])
    _agent = Agent(_registry)
    ctx.agent = _agent  # phase 2: plugins that run prompts through the Agent
    _load_plugins([c for c in plugin_classes if c.requires_agent])

    # 6c. Away summary — speak a greeting based on prior session notes
    away_summary.speak_greeting(speaker)

    # 7. Hotkey listener
    hotkey_listener = HotkeyListener(on_press_cb=on_press, on_release_cb=on_release)
    hotkey_listener.start()

    # 8. Wake word listener (always-on; gracefully disabled if openwakeword missing)
    wake_word_listener = WakeWordListener(
        handle_command_fn=handle_command,
        processing_event=_processing,
        transcriber=transcriber_instance,
        menubar=menubar,
    )
    wake_word_listener.start()

    print("Aria ready. Hold ⌥ Space to ask a question.")

    # 8. Run rumps main loop — blocks until quit
    menubar.run()


_LOGIN_URLS = {
    "gmail": "https://mail.google.com",
    "google": "https://accounts.google.com",
    "linkedin": "https://www.linkedin.com",
}


def _handle_login() -> None:
    """Handle --login <service> CLI argument. Opens a visible browser for manual login."""
    if len(sys.argv) < 3:
        print("Usage: python main.py --login <service>")
        print(f"Available services: {', '.join(sorted(_LOGIN_URLS))}")
        sys.exit(1)

    service = sys.argv[2].lower()
    url = _LOGIN_URLS.get(service)
    if url is None:
        print(f"Unknown service: {service!r}")
        print(f"Available services: {', '.join(sorted(_LOGIN_URLS))}")
        sys.exit(1)

    print("Run this once. After logging in, Aria will reuse your session automatically.")
    from browser_profile import login_session
    login_session(url)
    sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--login":
        _handle_login()
    else:
        main()
