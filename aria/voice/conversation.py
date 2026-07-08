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

Disable with ARIA_CONVERSATION=0 in .env or the environment.
Everything here is defensive: hold() never raises to the command loop.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

import aria.ui.listening_indicator as listening_indicator

SAMPLE_RATE = 16000
_CHUNK = 512                     # 32 ms @ 16 kHz per read

# VAD tuning — same int16 RMS scale as wake_word.py
_SPEECH_RMS = 400.0
_PRE_SPEECH_TIMEOUT = 4.0        # s of silence before deciding "no follow-up"
_SILENCE_END = 1.2               # s of trailing silence that ends an utterance
_MAX_UTTERANCE = 12.0            # s hard cap per follow-up
_MIN_SPEECH = 0.25               # s of actual speech required

_MAX_FOLLOWUPS = 6               # safety cap per conversation

_CUE_SOUND = "/System/Library/Sounds/Pop.aiff"

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


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z' ]+", " ", text.lower()).strip()


def is_end_phrase(text: str) -> bool:
    """True if the transcript is a conversation-closing phrase."""
    norm = re.sub(r"\s+", " ", _normalize(text))
    return norm in _END_PHRASES


def _play_cue() -> None:
    """Soft pop so the user knows the mic is open. Non-blocking, best-effort."""
    if not os.path.exists(_CUE_SOUND):
        return
    threading.Thread(
        target=lambda: subprocess.run(
            ["afplay", _CUE_SOUND],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ),
        daemon=True,
    ).start()


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

            if rms >= _SPEECH_RMS:
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
) -> None:
    """
    Keep the conversation open after an answer: listen for follow-ups and
    run each through run_turn() until the user goes silent, says a closing
    phrase, or the follow-up cap is hit. Never raises.

    run_turn(text) -> spoken answer; it owns agent execution and logging.
    """
    try:
        for _ in range(_MAX_FOLLOWUPS):
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
