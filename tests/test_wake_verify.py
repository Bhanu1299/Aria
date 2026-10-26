"""tests/test_wake_verify.py — the wake model's hit is checked by Whisper before Aria wakes
(2026-10-23: a video kept waking it; real and false hits both scored 0.70–0.73)."""
from __future__ import annotations

import time

import numpy as np
import pytest

from aria.voice import wake_word
from aria.voice.wake_word import WakeWordListener, heard_wake_word


@pytest.mark.parametrize("text,ok", [
    ("Aria, let's apply to some jobs", True),
    ("Arya let's apply to some jobs", True),
    ("hey aria what time is it", True),
    ("so, Aria, open the dashboard", True),
    ("Area, set a timer", True),                      # a sound-alike up front: you said the name
    ("Good job, Michelle. I got really lucky", False),
    ("the parking area was full", False),             # a sound-alike mid-sentence: dialogue
    ("I'm just watching a video", False),
    ("", False),
])
def test_heard_wake_word(text, ok):
    assert heard_wake_word(text) is ok


class Stream:
    def __init__(self):
        self.reads = 0

    def read(self, n, exception_on_overflow=False):
        self.reads += 1
        time.sleep(0.005)
        return np.zeros(n, dtype=np.int16).tobytes()


class Transcriber:
    def __init__(self, text, delay=0.05, boom=False, no_speech=0.1):
        self.text, self.delay, self.boom, self.calls, self.no_speech = text, delay, boom, 0, no_speech

    def check_wake(self, audio, timeout=5.0):
        self.calls += 1
        time.sleep(self.delay)
        if self.boom:
            raise RuntimeError("whisper down")
        return self.text, self.no_speech


def _listener(transcriber):
    return WakeWordListener(lambda t: None, transcriber=transcriber)


def test_verify_keeps_reading_the_mic_while_whisper_checks():
    stream = Stream()
    heard, chunks = _listener(Transcriber("Aria, apply")).\
        _verify(stream, np.zeros(32000, dtype=np.int16), 512)
    assert heard is True and len(chunks) == stream.reads > 0       # nothing said meanwhile is lost


def test_verify_rejects_a_video():
    heard, _ = _listener(Transcriber("good job Michelle"))._verify(Stream(), np.zeros(32000, dtype=np.int16), 512)
    assert heard is False


@pytest.mark.parametrize("t", [Transcriber("x", boom=True), Transcriber("x", delay=0.5)])
def test_a_broken_or_slow_whisper_never_drops_a_real_wake(monkeypatch, t):
    monkeypatch.setattr(wake_word, "_VERIFY_MAX_SECS", 0.1)
    heard, _ = _listener(t)._verify(Stream(), np.zeros(32000, dtype=np.int16), 512)
    assert heard is True


def test_verify_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("ARIA_WAKE_VERIFY", "0")
    t = Transcriber("good job Michelle")
    assert _listener(t)._verify(Stream(), np.zeros(32000, dtype=np.int16), 512) == (True, [])
    assert t.calls == 0


def test_recording_starts_with_the_audio_read_during_the_check(monkeypatch, tmp_path):
    monkeypatch.setattr(wake_word, "_WAV_PATH", str(tmp_path / "w.wav"))
    monkeypatch.setattr(wake_word.vad, "detector", lambda: None)          # the loudness gate
    loud = (np.ones(512, dtype=np.int16) * 4000).tobytes()

    class Quiet(Stream):
        def read(self, n, exception_on_overflow=False):
            self.reads += 1
            return np.zeros(n, dtype=np.int16).tobytes()
    stream = Quiet()
    path = _listener(None)._record_until_silence(stream, 16000, 512, prefix=[loud] * 20)
    assert path is not None                               # the speech came only from the prefix


def test_a_primed_aria_over_noise_is_rejected():
    # the "Aria" prime makes Whisper write the name over room noise; its no_speech_prob gives it away
    heard, _ = _listener(Transcriber("Aria", no_speech=0.56))._verify(Stream(), np.zeros(32000, dtype=np.int16), 512)
    assert heard is False
    heard, _ = _listener(Transcriber("Aria", no_speech=0.40))._verify(Stream(), np.zeros(32000, dtype=np.int16), 512)
    assert heard is True


def test_check_hears_the_end_of_the_word():
    # 2026-10-25: the score crosses 0.7 before "Aria" is finished — checked at the crossing,
    # Whisper heard it in 13/37 replayed user clips, 0.5 s later in 33/37
    class Late(Stream):
        def read(self, n, exception_on_overflow=False):
            self.reads += 1
            return (np.ones(n, dtype=np.int16) * 7).tobytes()

    class Seen(Transcriber):
        def check_wake(self, audio, timeout=5.0):
            self.audio = audio
            return super().check_wake(audio, timeout)
    t = Seen("Aria")
    stream = Late()
    heard, chunks = _listener(t)._verify(stream, np.zeros(32000, dtype=np.int16), 512)
    settled = wake_word._VERIFY_SETTLE_CHUNKS * 512
    assert len(t.audio) == 32000
    assert np.all(t.audio[-settled:] > 0) and np.all(t.audio[:-settled] == 0)
    assert heard is True and len(chunks) >= wake_word._VERIFY_SETTLE_CHUNKS   # still handed to the recording


def test_a_rejected_check_is_logged_with_what_whisper_heard(tmp_path, monkeypatch):
    from aria.observability import wake_stats
    lines = []
    monkeypatch.setattr(wake_stats, "_append", lines.append)
    wake_stats.log_event("verify_reject", "custom_onnx", 0.72, heard=("I", 0.51))
    assert lines[0]["heard"] == "I" and lines[0]["no_speech"] == 0.51


def test_recording_ends_when_the_speech_model_says_you_stopped(monkeypatch, tmp_path):
    # a fan / TV hum stays loud after you stop talking: the speech model, not loudness, ends it
    monkeypatch.setattr(wake_word, "_WAV_PATH", str(tmp_path / "w.wav"))
    hum = (np.ones(512, dtype=np.int16) * 3000).tobytes()

    class Detector:
        def __init__(self):
            self.n = 0

        def is_speech(self, chunk):
            self.n += 1
            return self.n <= 20                                   # 0.64 s of you, then only the hum
    monkeypatch.setattr(wake_word.vad, "detector", Detector)

    class Hum(Stream):
        def read(self, n, exception_on_overflow=False):
            self.reads += 1
            return hum
    stream = Hum()
    path = _listener(None)._record_until_silence(stream, 16000, 512)
    assert path is not None and stream.reads < 20 + int(wake_word._VAD_SILENCE_SECS / 0.032) + 3


@pytest.mark.parametrize("text, no_speech, voiced, ok", [
    ("Aria.", 0.63, 20, True),       # live 2026-10-26: a real "Aria." was refused at no_speech 0.63
    ("Aria.", 0.63, 3, False),       # the same prime over a window with no voice in it
    ("Aria.", 0.63, None, False),    # no VAD: the old strict rule
    ("Aria", 0.40, None, True),
    ("Aria", 0.85, 30, False),       # Whisper sure it's not speech
    ("Thank you.", 0.10, 30, False),
])
def test_wake_accepted(text, no_speech, voiced, ok):
    assert wake_word.wake_accepted(text, no_speech, voiced) is ok


def test_verify_lets_a_voiced_borderline_aria_through(monkeypatch):
    monkeypatch.setattr(wake_word, "voiced_chunks", lambda w: 20)
    heard, _ = _listener(Transcriber("Aria.", no_speech=0.63))._verify(Stream(), np.zeros(32000, dtype=np.int16), 512)
    assert heard is True
    monkeypatch.setattr(wake_word, "voiced_chunks", lambda w: 2)
    heard, _ = _listener(Transcriber("Aria.", no_speech=0.63))._verify(Stream(), np.zeros(32000, dtype=np.int16), 512)
    assert heard is False


def test_silence_has_no_voice_in_it():
    if wake_word.vad.detector() is None:
        pytest.skip("Silero VAD model not installed")
    assert wake_word.voiced_chunks(np.zeros(32000, dtype=np.int16)) < wake_word._VAD_GATE_MIN


def test_fired_clips_are_kept_locally_and_rotated(monkeypatch, tmp_path):
    import os
    monkeypatch.setattr(wake_word, "_CLIPS_DIR", str(tmp_path))
    monkeypatch.setattr(wake_word, "_CLIPS_KEEP", 3)
    for i in range(5):
        wake_word.save_clip(np.zeros(32000, dtype=np.int16), f"k{i}", 0.7)
        time.sleep(1.01) if i == 0 else None
    assert len(os.listdir(tmp_path)) == 3
    monkeypatch.setenv("ARIA_WAKE_CLIPS", "0")
    wake_word.save_clip(np.zeros(32000, dtype=np.int16), "off", 0.7)
    assert not any("off" in n for n in os.listdir(tmp_path))


def test_a_vad_gate_event_logs_the_voiced_count(monkeypatch):
    from aria.observability import wake_stats
    lines = []
    monkeypatch.setattr(wake_stats, "_append", lines.append)
    wake_stats.log_event("vad_gate", "custom_onnx", 0.71, voiced=2)
    wake_stats.log_event("verify_reject", "custom_onnx", 0.72, heard=("", 1.0, 0))
    assert lines[0]["voiced"] == 2 and lines[1]["voiced"] == 0
