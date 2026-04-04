#!/usr/bin/env python3
"""
Aria — Background Mac Voice Agent
Phase 2B-3: Vision fallback for auth-walled / JS-heavy pages

Pipeline:
  transcribe → classify intent →
    knowledge  : Groq direct answer → speaker
    web_search : Google first → real URL → fetch → summarize → speaker
                 (fallback: vision.read_screen if all fetches return garbage)
    web_direct : site URL → fetch → summarize → speaker
                 (fallback: vision.read_screen if fetch returns garbage)
    navigate   : open URL in default browser → speak "Opening [site]"
    app        : open macOS app via `open -a` → speak result
    media      : YouTube search → fetch → summarize → speaker
                 (fallback: vision.read_screen if fetch returns garbage)
"""

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
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(levelname)s: %(message)s",
)

import config
from transcriber import Transcriber
from voice_capture import VoiceCapture
from speaker import Speaker
from browser import BrowserExecutor, goto as browser_goto
from router import route
from summarizer import summarize, answer_knowledge
from menubar import AriaMenuBar
from hotkey import HotkeyListener
from app_launcher import open_app
import mac_controller
import media
import briefing
import jobs
import memory
import tracker
import agent_browser
# vision is imported lazily inside _vision_fallback() to keep it off the
# startup critical path — Playwright + ctranslate2 + vision all loading at
# once on Python 3.9 macOS can trigger OpenMP duplicate-lib abort()

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_processing = threading.Event()   # set = currently processing, clear = idle
_recording_active = threading.Event()  # set = start_recording() was called

menubar: AriaMenuBar = None
voice_capture: VoiceCapture = None
transcriber_instance: Transcriber = None
speaker: Speaker = None
browser: BrowserExecutor = None
hotkey_listener: HotkeyListener = None


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
    _processing.set()
    menubar.set_state("LISTENING")
    voice_capture.start_recording()
    _recording_active.set()  # set AFTER start_recording() returns — prevents on_release racing in


def _process_release():
    """Full pipeline — runs on a dedicated thread, never blocks the hotkey thread."""
    try:
        wav_path = voice_capture.stop_recording()
        question = transcriber_instance.transcribe(wav_path)

        # Treat empty or noise-only transcripts as silence
        # Whisper sometimes outputs ". . .", "...", "[BLANK_AUDIO]" etc. for silence
        import re as _re
        _cleaned = _re.sub(r'[\s\.\,\!\?\-\[\]]+', '', question)
        _SINGLE_WORD_COMMANDS = {
            "mute", "unmute", "pause", "stop", "skip", "next",
            "resume", "play", "screenshot",
        }
        _is_single_word = question.lower().strip() in _SINGLE_WORD_COMMANDS
        if not _cleaned or (len(question.split()) < 2 and not _is_single_word):
            speaker.say("I didn't catch that, please try again.")
            menubar.set_state("IDLE")
            _processing.clear()
            return

        print(f"[Aria] Transcribed: {question!r}")
        menubar.set_state("THINKING")

        # Pre-check: ordinal job follow-ups ("tell me more about the second job")
        followup = _check_jobs_followup(question)
        if followup is not None:
            print(f"[Aria] Jobs follow-up: {followup!r}")
            answer = followup
        else:
            # Step 1 — classify and route
            intent = route(question)
            print(f"[Aria] Intent: type={intent['type']!r}  query={intent['query']!r}")
            answer = _handle_intent(intent, question)

        print(f"[Aria] Answer: {answer!r}")
        speaker.say(answer)

        menubar.set_state("DONE")
        time.sleep(1)
        menubar.set_state("IDLE")
        _processing.clear()

    except Exception as e:
        print(f"[Aria] Error during processing: {e}")
        try:
            speaker.say("Something went wrong, please try again.")
        except Exception as say_err:
            print(f"[Aria] Error speaking error message: {say_err}")
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
# Apply commands must pass through to the router — never intercept them here.
_IS_APPLY_RE = _re.compile(r"\b(apply|apply\s+to|apply\s+for|submit)\b", _re.IGNORECASE)


def _check_jobs_followup(question: str) -> Optional[str]:
    """
    Intercept ordinal job-reference questions before routing.

    Matches any question with an ordinal ("first", "second", etc.) while
    jobs are saved in session memory. Uses vision to screenshot the job page
    and answer the actual question, or opens the URL for "open"/"visit".

    Apply commands are deliberately excluded — they route through the intent
    system so the full applicator flow can run.
    """
    # Only intercept if we have jobs saved from a prior search
    if not memory.get_persistent("last_jobs"):
        return None
    # Let apply/submit commands fall through to router → "apply" intent handler
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


def _get_capability_response() -> str:
    return (
        "I can answer general knowledge questions, search the web, open websites and apps, "
        "control media playback on Apple Music and YouTube, adjust Mac system settings by voice, "
        "give you a morning briefing with weather, calendar, email, and news, "
        "search for jobs on LinkedIn and Indeed, help you apply to jobs and track your applications, "
        "check your system info like battery and wifi, take notes, set reminders, "
        "and run calculations. Just tell me what you need."
    )


def _handle_intent(intent: dict, original_question: str) -> str:
    """Dispatch to the correct handler based on intent type."""
    intent_type = intent["type"]

    # --- Knowledge: answer directly from LLM, no browser ---
    if intent_type == "knowledge":
        return answer_knowledge(intent["query"])

    # --- Web search: Google first → extract real URLs → fetch best → summarize ---
    # Vision fallback if all fetched pages are garbage / auth-walled.
    if intent_type == "web_search":
        print(f"[Aria] Google search: {intent['url']}")
        links = browser.extract_links(intent["url"])
        print(f"[Aria] Found {len(links)} result URLs")

        page_text = None
        used_url = ""
        for link in links[:3]:
            text = browser.fetch(link)
            if text is not None:
                page_text = text
                used_url = link
                break

        if page_text is not None:
            print(f"[Aria] Fetched content from: {used_url}")
            return summarize(
                page_text=page_text,
                query=intent["query"],
                instructions=intent.get("instructions", ""),
            )

        # All pages were garbage/auth-walled — vision fallback on the first result
        fallback_url = links[0] if links else intent["url"]
        print(f"[Aria] All fetches garbage — vision fallback: {fallback_url}")
        return _vision_fallback(fallback_url, original_question)

    # --- Web direct: site URL → fetch → summarize ---
    # Vision fallback if headless fetch returns garbage (auth-walled, JS-heavy, etc.)
    if intent_type == "web_direct":
        print(f"[Aria] Direct site fetch: {intent['url']}")
        page_text = browser.fetch(intent["url"])
        if page_text is None:
            print(f"[Aria] Headless failed — trying authenticated profile: {intent['url']}")
            from browser import fetch_authenticated
            page_text = fetch_authenticated(intent["url"], intent["query"])
        if page_text is not None:
            return summarize(
                page_text=page_text,
                query=intent["query"],
                instructions=intent.get("instructions", ""),
            )
        print(f"[Aria] Auth fetch also failed — vision fallback: {intent['url']}")
        return _vision_fallback(intent["url"], original_question)

    # --- Navigate: open URL in default browser ---
    if intent_type == "navigate":
        url = intent.get("url", "")
        site_name = intent.get("site_name") or url
        if not url:
            return "I'm not sure which site to open. Could you try again?"
        print(f"[Aria] Navigate: {url}")
        try:
            browser_goto(url)
            return f"Opening {site_name}."
        except Exception as exc:
            print(f"[Aria] Navigate failed: {exc}")
            return f"I couldn't open {site_name}."

    # --- App: launch macOS application ---
    if intent_type == "app":
        app_name = intent.get("app_name") or intent.get("query", "")
        contact = intent.get("contact") or None
        print(f"[Aria] App launch: app={app_name!r} contact={contact!r}")
        result = open_app(app_name, contact)
        # If app not found and no contact, try treating it as a folder name
        if "couldn't find" in result and not contact:
            folder_result = mac_controller._finder_open(app_name)
            if "couldn't find" not in folder_result:
                return folder_result
        return result

    # --- Media: music app control + YouTube playback via media.py ---
    if intent_type == "media":
        print(f"[Aria] Media command: {original_question!r}")
        return media.handle_media_command(original_question)

    # --- App control: native Mac app control via AppleScript ---
    if intent_type == "app_control":
        print(f"[Aria] App control: {original_question!r}")
        return mac_controller.handle_app_command(original_question)

    # --- Briefing: morning summary with weather, calendar, email, news ---
    if intent_type == "briefing":
        print("[Aria] Building briefing...")
        speaker.say("Getting your briefing, one moment.")
        return briefing.build_briefing()

    # --- Jobs: LinkedIn + Indeed search via Google → vision pipeline ---
    if intent_type == "jobs":
        print(f"[Aria] Job search: {intent['query']!r}")
        speaker.say("Searching for jobs, one moment.")
        results = jobs.search_jobs(intent["query"])
        memory.store_jobs(results)
        memory.store_last_search(intent["query"])
        return jobs.format_spoken_results(results)

    # --- Apply: fill job application in visible browser → voice confirm → submit ---
    if intent_type == "apply":
        m = _JOB_FOLLOWUP_RE.search(original_question)
        if not m:
            return (
                "I'm not sure which job you want to apply to. "
                "Try saying 'apply to the first job'."
            )
        n = _ORDINAL_MAP.get(m.group(1).lower())
        job = memory.get_job_by_index(n) if n else None
        if job is None:
            return (
                f"I don't have a {m.group(1)} job saved from this session. "
                "Try searching for jobs first."
            )
        if not job.get("url"):
            return (
                f"I don't have a URL for the {m.group(1)} job "
                f"at {job.get('company', 'that company')}."
            )
        print(f"[Aria] Apply: {job['title']} @ {job['company']} → {job['url']}")
        speaker.say(
            f"Starting application for {job['title']} at {job['company']}. "
            "Opening browser now."
        )
        import applicator
        result = applicator.run_application(
            job=job,
            voice_capture=voice_capture,
            transcriber=transcriber_instance,
            speaker=speaker,
        )
        if result == "submitted":
            return f"Application submitted for {job['title']} at {job['company']}."
        elif result == "not submitted":
            return (
                "Application not submitted. "
                "The form is filled in your browser if you want to review it."
            )
        else:
            return (
                f"I had trouble completing the application for {job['title']} "
                f"at {job['company']}. The browser is open so you can finish manually."
            )

    # --- Capability: hardcoded response, no LLM call ---
    if intent_type == "capability":
        return _get_capability_response()

    # Fallback
    return answer_knowledge(original_question)


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

    # 7. Hotkey listener
    hotkey_listener = HotkeyListener(on_press_cb=on_press, on_release_cb=on_release)
    hotkey_listener.start()

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
