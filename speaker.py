"""
speaker.py — Aria TTS speaker

Wraps macOS `say` via subprocess.Popen so speech can be interrupted
mid-sentence by calling stop() (e.g. when the user presses the hotkey again).
"""

from __future__ import annotations

import subprocess
import threading


class Speaker:
    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def say(self, text: str) -> None:
        """Speak text aloud. Stops any currently running speech first."""
        if not text or not text.strip():
            print("[SPEAKER] Nothing to say, skipping.")
            return

        self.stop()  # interrupt any in-progress speech

        text = text.strip()
        print(f"[SPEAKER] Speaking: {text[:60]}...")

        try:
            proc = subprocess.Popen(["say", text])
        except FileNotFoundError:
            print("[SPEAKER] Error: 'say' command not found — macOS only.")
            return
        except Exception as exc:
            print(f"[SPEAKER] Error starting say: {exc}")
            return

        with self._lock:
            self._proc = proc

        try:
            proc.wait()
        except Exception as exc:
            print(f"[SPEAKER] Error waiting for say: {exc}")
        finally:
            with self._lock:
                if self._proc is proc:
                    self._proc = None

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


# Backwards-compatible module-level function used by older code/tests
def speak(text: str) -> None:
    Speaker().say(text)
