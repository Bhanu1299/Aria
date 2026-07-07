"""Tests for conversation.py (follow-up mode) and speaker.ThinkingAck."""
from __future__ import annotations

import time

import pytest

import conversation
from speaker import ThinkingAck


# ---------------------------------------------------------------------------
# is_end_phrase
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "thanks", "Thanks.", "thank you", "Thank you!", "thanks, Aria",
    "that's all", "That's it.", "no", "Nope", "nothing else",
    "stop", "never mind", "I'm good", "we're done", "bye", "Goodbye!",
    "okay thanks", "cool, thanks",
])
def test_end_phrases_close_conversation(text):
    assert conversation.is_end_phrase(text)


@pytest.mark.parametrize("text", [
    "thanks for that, now check my email",
    "what about tomorrow",
    "no wait, make it 7 pm",
    "stop the music",
    "that's all the jobs? search Indeed too",
    "open Safari",
    "",
])
def test_real_followups_do_not_close_conversation(text):
    assert not conversation.is_end_phrase(text)


# ---------------------------------------------------------------------------
# enabled toggle
# ---------------------------------------------------------------------------

def test_enabled_by_default(monkeypatch):
    monkeypatch.delenv("ARIA_CONVERSATION", raising=False)
    assert conversation.enabled()


@pytest.mark.parametrize("value", ["0", "false", "off", "no", "False", "OFF"])
def test_disabled_via_env(monkeypatch, value):
    monkeypatch.setenv("ARIA_CONVERSATION", value)
    assert not conversation.enabled()


def test_enabled_via_env(monkeypatch):
    monkeypatch.setenv("ARIA_CONVERSATION", "1")
    assert conversation.enabled()


# ---------------------------------------------------------------------------
# hold() control flow — mic and TTS stubbed out
# ---------------------------------------------------------------------------

class _FakeSpeaker:
    def __init__(self):
        self.spoken = []

    def say(self, text):
        self.spoken.append(text)


class _FakeTranscriber:
    def __init__(self, transcripts):
        self._transcripts = list(transcripts)

    def transcribe_numpy(self, audio, initial_prompt=""):
        return self._transcripts.pop(0) if self._transcripts else ""


class _FakeMenubar:
    def __init__(self):
        self.states = []

    def set_state(self, state):
        self.states.append(state)


def _fake_audio():
    import numpy as np
    return np.zeros(1600, dtype=np.float32)


def test_hold_runs_followups_until_end_phrase(monkeypatch):
    monkeypatch.setattr(conversation, "_listen_followup", lambda: _fake_audio())
    monkeypatch.setattr(conversation, "_play_cue", lambda: None)
    speaker = _FakeSpeaker()
    menubar = _FakeMenubar()
    transcriber = _FakeTranscriber(["what about tomorrow", "thanks"])
    turns = []

    def run_turn(text):
        turns.append(text)
        return f"answer to {text}"

    conversation.hold(speaker, transcriber, menubar, "", run_turn)

    assert turns == ["what about tomorrow"]
    assert "answer to what about tomorrow" in speaker.spoken
    assert "Anytime." in speaker.spoken  # thanks acknowledged
    assert menubar.states[-1] == "IDLE"


def test_hold_ends_on_silence(monkeypatch):
    monkeypatch.setattr(conversation, "_listen_followup", lambda: None)
    monkeypatch.setattr(conversation, "_play_cue", lambda: None)
    speaker = _FakeSpeaker()
    turns = []

    conversation.hold(speaker, _FakeTranscriber([]), _FakeMenubar(), "",
                      lambda t: turns.append(t) or "x")

    assert turns == []
    assert speaker.spoken == []


def test_hold_respects_followup_cap(monkeypatch):
    monkeypatch.setattr(conversation, "_listen_followup", lambda: _fake_audio())
    monkeypatch.setattr(conversation, "_play_cue", lambda: None)
    transcriber = _FakeTranscriber(["again"] * 50)
    turns = []

    conversation.hold(_FakeSpeaker(), transcriber, _FakeMenubar(), "",
                      lambda t: (turns.append(t), "ok")[1])

    assert len(turns) == conversation._MAX_FOLLOWUPS


def test_hold_never_raises(monkeypatch):
    def _boom():
        raise RuntimeError("mic exploded")
    monkeypatch.setattr(conversation, "_listen_followup", _boom)
    monkeypatch.setattr(conversation, "_play_cue", lambda: None)
    conversation.hold(_FakeSpeaker(), _FakeTranscriber([]), _FakeMenubar(), "",
                      lambda t: "x")  # must not raise


# ---------------------------------------------------------------------------
# ThinkingAck
# ---------------------------------------------------------------------------

def test_thinking_ack_fires_after_delay():
    speaker = _FakeSpeaker()
    ack = ThinkingAck(speaker, delay=0.05)
    ack.start()
    time.sleep(0.2)
    ack.cancel()
    assert len(speaker.spoken) == 1


def test_thinking_ack_cancelled_before_delay_stays_silent():
    speaker = _FakeSpeaker()
    ack = ThinkingAck(speaker, delay=5.0)
    ack.start()
    ack.cancel()
    time.sleep(0.1)
    assert speaker.spoken == []


def test_thinking_ack_cancel_blocks_until_ack_finishes():
    """cancel() must not let the real answer race an in-flight ack."""
    import threading

    class _SlowSpeaker:
        def __init__(self):
            self.events = []
            self._lock = threading.Lock()

        def say(self, text):
            with self._lock:
                self.events.append(("start", text))
            time.sleep(0.15)
            with self._lock:
                self.events.append(("end", text))

    speaker = _SlowSpeaker()
    ack = ThinkingAck(speaker, delay=0.01)
    ack.start()
    time.sleep(0.05)          # ack is now mid-speech
    ack.cancel()              # must block until the ack finishes
    speaker.say("real answer")

    kinds = [e[0] for e in speaker.events]
    assert kinds == ["start", "end", "start", "end"]
    assert speaker.events[2][1] == "real answer"
