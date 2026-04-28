"""
applicator.py — Phase 3C: Voice-driven job application for Aria

Uses DOM-based automation for known sites (LinkedIn) and falls back
to the coordinate-based computer_use loop for unknown sites.

Public API:
  run_application(job, voice_capture, transcriber, speaker)
    → "submitted" | "not submitted" | "failed"
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import agent_browser
import computer_use
import dom_browser
import linkedin_applicator
import tracker

logger = logging.getLogger(__name__)


def _load_identity() -> dict:
    """Read identity.json from the same directory as this file."""
    path = Path(__file__).parent / "identity.json"
    return json.loads(path.read_text())


def _capture_voice_confirmation(voice_capture, transcriber) -> str:
    """
    Record ~3 seconds of audio and transcribe.
    speaker.say() must complete BEFORE this is called — it blocks until
    macOS `say` exits (speaker.py:43), so recording starts after the prompt.
    Returns transcription string (empty on error).
    """
    try:
        voice_capture.start_recording()
        time.sleep(3)
        wav_path = voice_capture.stop_recording()
        return transcriber.transcribe(wav_path)
    except Exception as exc:
        logger.error("Voice confirmation capture failed: %s", exc)
        return ""


def _voice_ask(field_name: str, speaker, voice_capture, transcriber) -> str:
    """Ask the user a question via voice and return their transcribed answer."""
    speaker.say(f"I need your {field_name}. Please say it now.")
    return _capture_voice_confirmation(voice_capture, transcriber)


def _is_linkedin_url(url: str) -> bool:
    """Check if URL is a LinkedIn job page."""
    return "linkedin.com" in url.lower()


def run_application(job: dict, voice_capture, transcriber, speaker) -> str:
    """
    Run the full job application flow for the given job.

    1. Navigate the persistent visible browser to the job URL.
    2. For LinkedIn: use DOM-based linkedin_applicator.
       For other sites: use coordinate-based computer_use loop.
    3. When the loop returns "confirm", ask for voice confirmation.
    4. Submit on "yes", leave the browser open on "no".
    5. Log every submission to tracker.py (SQLite).

    Returns:
        "submitted"     — form submitted, row logged in tracker
        "not submitted" — user declined at confirmation step
        "failed"        — navigation error, stuck, or no URL
    """
    url = job.get("url", "").strip()
    if not url:
        logger.error("run_application: job has no URL — %r", job.get("title"))
        return "failed"

    identity = _load_identity()

    # Navigate the already-open visible browser to the job URL
    try:
        agent_browser.navigate(url, settle_secs=5.0)
    except Exception as exc:
        logger.error("Navigation to job URL failed: %s", exc)
        return "failed"

    context_data = {
        "name":        identity.get("name", ""),
        "email":       identity.get("email", ""),
        "phone":       identity.get("phone", ""),
        "location":    identity.get("location", ""),
        "linkedin":    identity.get("linkedin", ""),
        "github":      identity.get("github", ""),
        "summary":     identity.get("summary", ""),
        "education":   identity.get("education", ""),
        "resume_path": identity.get("resume_path", ""),
        "experience":  identity.get("experience", []),
        "skills":      identity.get("skills", []),
    }

    # ----------------------------------------------------------------
    # Route: LinkedIn → DOM, others → vision loop
    # ----------------------------------------------------------------
    if _is_linkedin_url(url):
        voice_ask_fn = lambda field: _voice_ask(field, speaker, voice_capture, transcriber)
        result, data = linkedin_applicator.run_linkedin_application(
            job, context_data, voice_ask_fn,
        )
        logger.info("linkedin_applicator returned: %r (data=%r)", result, data)

        # Handle LinkedIn-specific failures
        if result == "stuck" and isinstance(data, dict):
            reason = data.get("reason", "")
            if reason == "not_logged_in":
                speaker.say("You are not logged in to LinkedIn. Please log in and try again.")
                return "failed"

    else:
        # Vision-based fallback with voice-ask loop for unknown sites
        goal = (
            f"Fill out the job application form for: "
            f"{job.get('title', 'this position')} at {job.get('company', 'this company')}. "
            "Fill every required field using the applicant profile in Context. "
            "Do NOT click Submit or Apply — return 'confirm' when the form is complete."
        )
        max_steps = 30
        current_step = 1
        voice_asks = 0
        max_voice_asks = 5

        while True:
            result, data = computer_use.run_loop(
                goal, context_data, max_steps=max_steps, start_step=current_step,
            )
            logger.info("computer_use.run_loop returned: %r (data=%r)", result, data)

            if result == "needs_input" and voice_asks < max_voice_asks and data:
                voice_asks += 1
                field_name = data.get("field", "this field")
                current_step = data.get("step", current_step) + 1

                answer = _voice_ask(field_name, speaker, voice_capture, transcriber)
                logger.info("Voice answer for %r: %r", field_name, answer)

                if answer.strip():
                    try:
                        computer_use.execute({"action": "type", "text": answer.strip()})
                        time.sleep(0.5)
                        computer_use.execute({"action": "key", "key": "Tab"})
                    except Exception as exc:
                        logger.warning("Failed to type voice answer: %s", exc)
                    context_data[field_name] = answer.strip()
                continue

            break

    # ----------------------------------------------------------------
    # Confirm + Submit (shared by both paths)
    # ----------------------------------------------------------------
    if result == "confirm":
        # Form is filled — ask user for voice confirmation before submitting
        speaker.say(
            f"The application for {job.get('title', 'this job')} "
            f"at {job.get('company', 'this company')} looks complete. "
            "Say yes to submit, or no to cancel."
        )
        response = _capture_voice_confirmation(voice_capture, transcriber)
        logger.info("Voice confirmation response: %r", response)

        _CONFIRM_WORDS = {"yes", "submit", "go ahead", "do it", "yes submit",
                          "send it", "yep", "yeah", "sure", "confirm"}
        if any(w in response.lower() for w in _CONFIRM_WORDS):
            # Click the submit button
            submitted_via_dom = dom_browser.click_by_text("Submit application")
            if not submitted_via_dom:
                # Fallback: vision-based submit click
                b64 = computer_use.take_screenshot()
                if b64 is not None:
                    submit_action = computer_use.decide(
                        b64=b64,
                        goal="Click the final Submit Application or Apply Now button.",
                        context_data={},
                        step=1,
                        max_steps=1,
                    )
                    if submit_action.get("action") == "click":
                        computer_use.execute(submit_action)

            time.sleep(2)  # let the submit request complete

            # Verify success
            success = (
                dom_browser.page_has_text("application was sent")
                or dom_browser.page_has_text("applied")
                or dom_browser.page_has_text("successfully submitted")
            )
            if success:
                logger.info("Application confirmed submitted successfully")
            else:
                logger.warning("Submit may have failed — confirmation text not found")

            tracker.log_application(
                company=job.get("company", ""),
                role=job.get("title", ""),
                platform=job.get("platform", ""),
                url=url,
                status="applied",
            )
            return "submitted"

        else:
            # User said no — browser stays open so they can review the filled form
            return "not submitted"

    # stuck or max_steps — browser stays open so user can finish manually
    return "failed"
