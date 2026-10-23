"""tests/test_barge_in.py — Aria can be cut off mid-sentence, but never by its own name."""
from __future__ import annotations

import threading

from aria.voice import speaker as speaker_mod
from aria.voice.speaker import Speaker
from aria.voice.wake_word import WakeWordListener


class _Proc:
    def __init__(self):
        self.done = threading.Event()

    def poll(self):
        return 0 if self.done.is_set() else None

    def wait(self, timeout=None):
        self.done.wait(timeout)

    def terminate(self):
        self.done.set()

    def kill(self):
        self.done.set()


def _speaking(monkeypatch, text):
    proc = _Proc()
    monkeypatch.setattr(speaker_mod.subprocess, "Popen", lambda args: proc)
    sp = Speaker()
    t = threading.Thread(target=sp.say, args=(text,), daemon=True)
    t.start()
    for _ in range(200):
        if sp.is_speaking():
            break
        threading.Event().wait(0.005)
    return sp, proc, t


def test_interruptible_while_speaking_and_stop_ends_it(monkeypatch):
    sp, proc, t = _speaking(monkeypatch, "The next open job on your sheet is John Deere.")
    assert sp.is_speaking() and sp.interruptible()
    sp.stop()
    t.join(1)
    assert not sp.is_speaking() and not sp.interruptible()


def test_never_interruptible_while_saying_its_own_name(monkeypatch):
    sp, proc, t = _speaking(monkeypatch, "Your job dashboard is open in a tab in Aria's Chrome.")
    assert sp.is_speaking() and not sp.interruptible()
    sp.stop()
    t.join(1)


def test_listener_barges_in_only_with_both_hooks():
    stopped = []
    full = WakeWordListener(lambda t: None, interruptible=lambda: True, on_barge_in=lambda: stopped.append(1))
    assert full._can_barge_in()
    full._barge_in()
    assert stopped == [1]
    assert not WakeWordListener(lambda t: None)._can_barge_in()
    assert not WakeWordListener(lambda t: None, interruptible=lambda: False,
                                on_barge_in=lambda: None)._can_barge_in()


def test_hook_errors_never_raise():
    def boom():
        raise RuntimeError("x")
    w = WakeWordListener(lambda t: None, interruptible=boom, on_barge_in=boom)
    assert not w._can_barge_in()
    w._barge_in()
