"""Siri-style listening and speaking (2026-10-25): a speech model instead of loudness, a second
chance after a near-miss, and text made sayable before `say` reads it."""
from __future__ import annotations

import numpy as np
import pytest

from aria.voice import speech_text, vad


@pytest.mark.parametrize("raw,heard", [
    ("There's no job with ID J0200 on your sheet. The closest match is J0220 — Full Stack at **Acme**.",
     "There's no job with ID 200 on your sheet. The closest match is job 220, Full Stack at Acme."),
    ("Found two:\n- Meijer, e.g. SWE 1\n- eBay w/ RippleMatch", "Found two: Meijer, for example, SWE 1. eBay with RippleMatch."),
    ("Apply at https://jobs.lever.co/acme/123 & tell me.", "Apply at jobs.lever.co and tell me."),
    ("See [the posting](https://x.com/a).", "See the posting."),
    ("", ""),
])
def test_text_is_made_sayable(raw, heard):
    assert speech_text.spoken(raw) == heard


def test_best_voice_prefers_premium_then_enhanced_then_samantha():
    listing = ("Albert              en_US    # Hello!\nSamantha (English (US)) en_US    # Hello!\n"
               "Ava (Enhanced)      en_US    # Hello!\nZoe (Premium)       en_US    # Hello!\n"
               "Amélie              fr_CA    # Bonjour\n")
    assert speech_text._pick(listing) == "Zoe (Premium)"
    assert speech_text._pick(listing.replace("Zoe (Premium)", "Zoe")) == "Ava (Enhanced)"
    assert speech_text._pick("Albert  en_US # x\nSamantha (English (US)) en_US    # Hello!\n") == "Samantha (English (US))"
    assert speech_text._pick("") is None


def test_say_args_use_the_voice_rate_and_spoken_text(monkeypatch):
    speech_text.best_voice.cache_clear()
    monkeypatch.setenv("ARIA_VOICE", "Zoe (Premium)")
    monkeypatch.setenv("ARIA_VOICE_RATE", "190")
    try:
        assert speech_text.say_args("Done — J0007.") == ["say", "-v", "Zoe (Premium)", "-r", "190", "Done, job 7."]
    finally:
        speech_text.best_voice.cache_clear()


needs_model = pytest.mark.skipif(not vad.MODEL_PATH.exists(), reason="silero_vad.onnx not downloaded")


@needs_model
def test_vad_ignores_noise_and_tones_that_the_loudness_gate_takes_for_speech():
    d = vad.detector()
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(16000) * 3000).astype(np.int16)              # RMS ~3000: "speech" to RMS ≥ 400
    tone = (np.sin(np.arange(16000) * 2 * np.pi * 440 / 16000) * 8000).astype(np.int16)
    for sound in (noise, tone):
        d.reset()
        assert not any(d.is_speech(sound[i:i + 512]) for i in range(0, 16000 - 512, 512))


def test_no_model_falls_back_to_the_loudness_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(vad, "MODEL_PATH", tmp_path / "missing.onnx")
    assert vad.detector() is None
    monkeypatch.setenv("ARIA_VAD", "0")
    assert vad.detector() is None


def test_model_ack_is_spoken_once_and_replaces_the_timed_one():
    import time
    from aria.voice.speaker import ThinkingAck

    class FakeSpeaker:
        def __init__(self):
            self.said = []

        def say(self, text):
            time.sleep(0.05)
            self.said.append(text)
    sp = FakeSpeaker()
    ack = ThinkingAck(sp, delay=0.2)
    ack.start()
    ack.say_now("On it.")
    ack.say_now("Still on it.")              # one ack per run
    time.sleep(0.4)                          # past the timer: the timed ack never comes
    ack.cancel()                             # waits for any ack still speaking
    assert sp.said == ["On it."]
