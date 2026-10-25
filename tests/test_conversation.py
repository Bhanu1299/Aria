"""Tests for conversation.py (follow-up mode) and speaker.ThinkingAck."""
from __future__ import annotations

import time

import numpy as np
import pytest

import aria.voice.conversation as conversation
from aria.voice.speaker import ThinkingAck


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
        return f"answer to {text} — anything else?"       # a question keeps the window open

    conversation.hold(speaker, transcriber, menubar, "", run_turn)

    assert turns == ["what about tomorrow"]
    assert "answer to what about tomorrow — anything else?" in speaker.spoken
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
                      lambda t: (turns.append(t), "ok?")[1])

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


# ── 2026-10-23: a video's dialogue was taken as follow-ups ──────────────────

@pytest.mark.parametrize("answer", [
    "On it — I'm working through 10 jobs now, best fit first.",
    "Timer set for 10 minutes.",
    "Which one did you mean, the RTX or the eBay job?",
    None,
])
def test_window_opens_after_every_answer(monkeypatch, answer):
    # 2026-10-25, user: "when a response ends make sure mic is on to listen to my response"
    monkeypatch.delenv("ARIA_CONVERSATION", raising=False)
    assert conversation.invites_reply(answer) is True


@pytest.mark.parametrize("answer,opens", [
    ("On it — I'm working through 10 jobs now, best fit first.", False),
    ("Timer set for 10 minutes.", False),
    ("Which one did you mean, the RTX or the eBay job?", True),
    ("I found two. Want me to fill the first one? I can also skip it.", True),
    (None, True),
])
def test_question_mode_opens_only_after_a_question(monkeypatch, answer, opens):
    monkeypatch.setenv("ARIA_CONVERSATION", "question")
    assert conversation.invites_reply(answer) is opens


def test_question_mode_no_window_after_a_plain_answer(monkeypatch):
    monkeypatch.setenv("ARIA_CONVERSATION", "question")
    monkeypatch.setattr(conversation, "_listen_followup", lambda: pytest.fail("mic opened after a plain answer"))
    conversation.hold(_FakeSpeaker(), _FakeTranscriber([]), _FakeMenubar(), "", lambda t: "x",
                      answer="On it — working through 10 jobs.")


def test_followups_continue_after_plain_answers(monkeypatch):
    monkeypatch.delenv("ARIA_CONVERSATION", raising=False)
    monkeypatch.setattr(conversation, "_play_cue", lambda: None)
    monkeypatch.setattr(conversation, "_listen_followup", lambda: _fake_audio())
    turns = []
    conversation.hold(_FakeSpeaker(), _FakeTranscriber(["the RTX one", "and another", "thanks"]), _FakeMenubar(), "",
                      lambda t: turns.append(t) or "Filling the RTX one now.", answer="Which job did you mean?")
    assert turns == ["the RTX one", "and another"]


def test_cue_finishes_before_the_mic_opens(monkeypatch):
    # 2026-10-25: the pop played in a thread reached the open mic and killed every window
    order = []
    monkeypatch.setattr(conversation.os.path, "exists", lambda p: True)
    monkeypatch.setattr(conversation.subprocess, "run", lambda cmd, **kw: order.append(("cue", cmd)))
    monkeypatch.setattr(conversation, "_listen_followup", lambda: order.append(("mic",)) or None)
    conversation.hold(_FakeSpeaker(), _FakeTranscriber([]), _FakeMenubar(), "", lambda t: "x", answer="Done.")
    assert [o[0] for o in order] == ["cue", "mic"]
    assert "-t" in order[0][1]


def test_sound_already_playing_when_the_mic_opens_is_ignored(monkeypatch):
    class Stream:
        def __init__(self, **kw):
            pass

        def start(self):
            pass

        def read(self, n):
            return np.full((n, 1), 3000, dtype=np.int16), False      # loud from the very first chunk

        def stop(self):
            pass

        def close(self):
            pass
    monkeypatch.setattr(conversation.sd, "InputStream", Stream)
    monkeypatch.setattr(conversation.vad, "detector", lambda: None)          # the loudness gate
    assert conversation._listen_followup() is None
