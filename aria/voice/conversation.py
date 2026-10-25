"""
conversation.py — Jarvis-style follow-up listening.

After Aria speaks an answer, the mic re-opens for a short window so the
user can follow up naturally ("what about tomorrow?") without pressing the
hotkey or saying the wake word again. The Agent keeps cross-turn history,
so follow-ups resolve pronouns and context automatically.

Flow per window:
  soft cue sound → VAD listen (give up after _PRE_SPEECH_TIMEOUT of
  silence) → transcribe → closing phrase or silence ends the conversation
  → otherwise run the turn and re-open the window.

The window opens after every answer (2026-10-25, user's call). Sound already
playing the moment the mic opens (a video, music — nobody starts talking within
_BACKGROUND_SECS of the cue) closes it unheard (2026-10-23: a video's dialogue was
being taken as follow-ups, turn after turn).

ARIA_CONVERSATION: 1 (default) = after every answer · question = only after an
answer that asked something · 0 = off.
Everything here is defensive: hold() never raises to the command loop.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

import aria.ui.listening_indicator as listening_indicator
from aria.voice import vad

SAMPLE_RATE = 16000
_CHUNK = 512                     # 32 ms @ 16 kHz per read

# VAD tuning — same int16 RMS scale as wake_word.py
_SPEECH_RMS = 400.0
_PRE_SPEECH_TIMEOUT = 4.0        # s of silence before deciding "no follow-up"
_SILENCE_END = 1.2               # s of trailing silence that ends an utterance
_MAX_UTTERANCE = 12.0            # s hard cap per follow-up
_MIN_SPEECH = 0.25               # s of actual speech required

_MAX_FOLLOWUPS = 6               # safety cap per conversation
_BACKGROUND_SECS = 0.25          # sound this soon after the cue = something already playing

_CUE_SOUND = "/System/Library/Sounds/Pop.aiff"
_CUE_SECS = "0.45"

# Phrases that end the conversation. Matched against the whole normalized
# transcript so "thanks for that summary" does NOT end it, but "thanks" does.
_END_PHRASES = {
    "no", "nope", "nothing", "nothing else", "no thanks", "no thank you",
    "stop", "cancel", "never mind", "nevermind",
    "that's all", "thats all", "that's it", "thats it", "that is all",
    "that's all for now", "thats all for now",
    "i'm good", "im good", "i'm done", "im done", "we're done", "were done",
    "done", "all good", "goodbye", "bye", "bye bye", "good night", "goodnight",
    "thanks", "thank you", "thanks aria", "thank you aria", "thanks a lot",
    "okay thanks", "ok thanks", "okay thank you", "ok thank you",
    "cool thanks", "great thanks", "perfect thanks", "awesome thanks",
}
_THANKS_RE = re.compile(r"\bthank", re.IGNORECASE)


def enabled() -> bool:
    """Conversation mode is on by default; ARIA_CONVERSATION=0 disables it."""
    return os.getenv("ARIA_CONVERSATION", "1").strip().lower() not in {
        "0", "false", "off", "no",
    }


def invites_reply(answer: Optional[str]) -> bool:
    """Should the mic re-open after this answer? After every answer (user 2026-10-25: "when a
    response ends make sure mic is on") — ARIA_CONVERSATION=question keeps it to answers that
    asked something. None (answer unknown) keeps the window."""
    if answer is None or os.getenv("ARIA_CONVERSATION", "1").strip().lower() != "question":
        return True
    return "?" in (answer or "").strip()[-160:]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z' ]+", " ", text.lower()).strip()


def is_end_phrase(text: str) -> bool:
    """True if the transcript is a conversation-closing phrase."""
    norm = re.sub(r"\s+", " ", _normalize(text))
    return norm in _END_PHRASES


def _play_cue() -> None:
    """Soft pop so the user knows the mic is open. Blocks until it has played: the mic opens
    only after it (2026-10-25: started in a thread, the pop reached the mic at RMS 450–2000
    ~0.26 s after the mic opened → "already playing" or a fake 0.1 s utterance → every
    follow-up window died). -t plays just the audible pop, not the file's 1.6 s tail."""
    if not os.path.exists(_CUE_SOUND):
        return
    try:
        subprocess.run(["afplay", "-t", _CUE_SECS, _CUE_SOUND], timeout=2,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def _listen_followup() -> Optional[np.ndarray]:
    """
    Open the mic and wait briefly for a follow-up utterance.

    Returns float32 audio normalized to [-1, 1] ready for
    Transcriber.transcribe_numpy(), or None if the user stayed silent
    (or the mic failed).
    """
    chunks: list[np.ndarray] = []
    speech_started = False
    speech_duration = 0.0
    silence_duration = 0.0
    pre_speech_elapsed = 0.0
    total_elapsed = 0.0
    chunk_secs = _CHUNK / SAMPLE_RATE
    # a speech model, not loudness, decides what is talking (fans, music, the cue aren't);
    # no model → the RMS gate
    detector = vad.detector()

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
        )
        stream.start()
    except Exception as exc:
        print(f"[Aria] Conversation mic unavailable: {exc}")
        return None

    try:
        while total_elapsed < _PRE_SPEECH_TIMEOUT + _MAX_UTTERANCE:
            try:
                data, _overflowed = stream.read(_CHUNK)
            except Exception as exc:
                print(f"[Aria] Conversation mic read error: {exc}")
                return None

            audio_np = np.asarray(data).reshape(-1)
            rms = float(np.sqrt(np.mean(audio_np.astype(np.float32) ** 2)))
            total_elapsed += chunk_secs
            speech = detector.is_speech(audio_np) if detector is not None else rms >= _SPEECH_RMS

            if speech and not speech_started and total_elapsed <= _BACKGROUND_SECS:
                print("[Aria] Someone was already talking when the mic opened — not a follow-up.")
                return None
            if speech:
                speech_started = True
                silence_duration = 0.0
                speech_duration += chunk_secs
                chunks.append(audio_np)
            elif speech_started:
                silence_duration += chunk_secs
                chunks.append(audio_np)
                if silence_duration >= _SILENCE_END:
                    break
                if speech_duration >= _MAX_UTTERANCE:
                    break
            else:
                pre_speech_elapsed += chunk_secs
                if pre_speech_elapsed >= _PRE_SPEECH_TIMEOUT:
                    return None
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

    if not speech_started or speech_duration < _MIN_SPEECH:
        return None

    audio = np.concatenate(chunks)
    return audio.astype(np.float32) / 32768.0


def hold(
    speaker,
    transcriber,
    menubar,
    keyterms_prompt: str,
    run_turn: Callable[[str], str],
    answer: Optional[str] = None,
) -> None:
    """
    Keep the conversation open after an answer: listen for follow-ups and
    run each through run_turn() until the user goes silent, says a closing
    phrase, or the follow-up cap is hit. Never raises.

    run_turn(text) -> spoken answer; it owns agent execution and logging.
    answer = what Aria just said; each window opens only if the last answer invites a reply.
    """
    try:
        for _ in range(_MAX_FOLLOWUPS):
            if not invites_reply(answer):
                return
            if menubar is not None:
                menubar.set_state("LISTENING")
            _play_cue()
            listening_indicator.show("Listening...")
            audio = _listen_followup()
            listening_indicator.hide()
            if audio is None:
                return  # silence — conversation over

            if menubar is not None:
                menubar.set_state("THINKING")
            try:
                text = transcriber.transcribe_numpy(
                    audio, initial_prompt=keyterms_prompt
                ).strip()
            except Exception as exc:
                print(f"[Aria] Follow-up transcription failed: {exc}")
                return

            if not text:
                return
            if is_end_phrase(text):
                if _THANKS_RE.search(text):
                    speaker.say("Anytime.")
                return

            print(f"[Aria] Follow-up: {text!r}")
            answer = run_turn(text)
            if not answer:
                return
            speaker.say(answer)
    except Exception as exc:
        print(f"[Aria] Conversation mode error: {exc}")
    finally:
        listening_indicator.hide()
        if menubar is not None:
            try:
                menubar.set_state("IDLE")
            except Exception:
                pass
