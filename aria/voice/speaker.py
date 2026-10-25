"""
speaker.py — Aria TTS speaker

Wraps macOS `say` via subprocess.Popen so speech can be interrupted
mid-sentence by calling stop() (e.g. when the user presses the hotkey again,
or says the wake word while Aria talks — see interruptible()).
"""

from __future__ import annotations

import random
import re
import subprocess
import threading

from aria.voice import speech_text

_ACK_DELAY_SECS = 3.5
_ACK_LINES = ("One moment.", "Working on it.", "Just a second.")
# On speakers the mic hears Aria's own voice: while it says its own name, the wake word must
# not count as the user cutting in.
_OWN_NAME = re.compile(r"\baria\b", re.I)


class Speaker:
    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._says_name = False

    def say(self, text: str) -> None:
        """Speak text aloud. Stops any currently running speech first."""
        if not text or not text.strip():
            print("[SPEAKER] Nothing to say, skipping.")
            return

        self.stop()  # interrupt any in-progress speech

        text = text.strip()
        print(f"[SPEAKER] Speaking: {text[:60]}...")

        try:
            proc = subprocess.Popen(speech_text.say_args(text))
        except FileNotFoundError:
            print("[SPEAKER] Error: 'say' command not found — macOS only.")
            return
        except Exception as exc:
            print(f"[SPEAKER] Error starting say: {exc}")
            return

        with self._lock:
            self._proc = proc
            self._says_name = bool(_OWN_NAME.search(text))

        try:
            proc.wait()
        except Exception as exc:
            print(f"[SPEAKER] Error waiting for say: {exc}")
        finally:
            with self._lock:
                if self._proc is proc:
                    self._proc = None

    def is_speaking(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def interruptible(self) -> bool:
        """Speaking right now, and not saying "Aria" (its own voice could wake it)."""
        with self._lock:
            return self._proc is not None and self._proc.poll() is None and not self._says_name

    def stop(self) -> None:
        """Interrupt any currently running speech immediately."""
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()


class ThinkingAck:
    """
    Speaks a brief acknowledgment if work is still running after a delay,
    so long agent runs never leave the user in dead silence.

    Usage:
        ack = ThinkingAck(speaker); ack.start()
        try: answer = agent.run(...)
        finally: ack.cancel()

    cancel() blocks until any in-flight acknowledgment finishes speaking,
    which guarantees the real answer is never cut off by the ack.
    """

    def __init__(self, speaker: Speaker, delay: float = _ACK_DELAY_SECS) -> None:
        self._speaker = speaker
        self._done = False
        self._spoke = False
        self._lock = threading.Lock()
        self._timer = threading.Timer(delay, self._fire)
        self._timer.daemon = True

    def start(self) -> None:
        try:
            self._timer.start()
        except Exception as exc:
            print(f"[SPEAKER] ThinkingAck start failed: {exc}")

    def _fire(self) -> None:
        with self._lock:
            if self._done or self._spoke:
                return
            self._spoke = True
            try:
                self._speaker.say(random.choice(_ACK_LINES))
            except Exception as exc:
                print(f"[SPEAKER] ThinkingAck speak failed: {exc}")

    def say_now(self, text: str) -> None:
        """The agent's own short ack ("On it.") before a slow tool: spoken at once, in the
        background, instead of the timed one — once per run. cancel() still waits for it."""
        self._timer.cancel()

        def run() -> None:
            with self._lock:
                if self._done or self._spoke:
                    return
                self._spoke = True
                try:
                    self._speaker.say(text)
                except Exception as exc:
                    print(f"[SPEAKER] ack speak failed: {exc}")
        threading.Thread(target=run, daemon=True, name="aria-ack").start()

    def cancel(self) -> None:
        self._timer.cancel()
        with self._lock:
            self._done = True


# Backwards-compatible module-level function used by older code/tests
def speak(text: str) -> None:
    Speaker().say(text)
