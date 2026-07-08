"""
transcriber.py — Aria Whisper transcriber (thread-safe)

Transcriber owns a dedicated worker thread. The faster-whisper model is
loaded and called exclusively on that thread, which prevents the ctranslate2
segfault that occurs when the model is called from a different thread than
where it was initialized (known issue on macOS with Python < 3.11).

Usage:
    t = Transcriber()          # starts worker thread, loads model (blocks until ready)
    text = t.transcribe(path)  # thread-safe; blocks until result arrives
    t.stop()                   # shuts down worker thread
"""

from __future__ import annotations

import queue
import threading
import logging

import numpy as np

logger = logging.getLogger(__name__)

_STOP = object()  # sentinel to shut down the worker


class Transcriber:
    def __init__(self) -> None:
        self._request_q: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._init_error: Exception | None = None

        self._worker = threading.Thread(
            target=self._worker_main, daemon=True, name="aria-transcriber"
        )
        self._worker.start()

        # Block until the model is loaded (or fails)
        self._ready.wait(timeout=120)
        if self._init_error is not None:
            raise self._init_error

    # ------------------------------------------------------------------
    # Public API — safe to call from any thread
    # ------------------------------------------------------------------

    def transcribe(self, audio_path: str, initial_prompt: str = "") -> str:
        """Submit audio_path for transcription. Blocks until the result arrives."""
        result_q: queue.Queue[str] = queue.Queue()
        self._request_q.put((audio_path, initial_prompt, result_q))
        try:
            return result_q.get(timeout=60)
        except queue.Empty:
            logger.error("Transcription timed out for: %s", audio_path)
            return ""

    def transcribe_numpy(self, audio: np.ndarray, initial_prompt: str = "") -> str:
        """Submit a numpy float32 audio array for transcription. Blocks until result."""
        result_q: queue.Queue[str] = queue.Queue()
        self._request_q.put((audio, initial_prompt, result_q))
        try:
            return result_q.get(timeout=60)
        except queue.Empty:
            logger.error("Transcription (numpy) timed out")
            return ""

    def stop(self) -> None:
        """Shut down the worker thread."""
        self._request_q.put(_STOP)
        self._worker.join(timeout=10)

    # ------------------------------------------------------------------
    # Worker thread — model is loaded AND called here, never anywhere else
    # ------------------------------------------------------------------

    def _worker_main(self) -> None:
        try:
            from faster_whisper import WhisperModel

            print("[TRANSCRIBE] Loading Whisper base model...")
            self._model = WhisperModel("base", device="cpu", compute_type="int8")
            print("[TRANSCRIBE] Whisper model loaded.")
            self._ready.set()
        except Exception as exc:
            self._init_error = exc
            self._ready.set()
            return

        while True:
            item = self._request_q.get()

            if item is _STOP:
                break

            audio_or_path, initial_prompt, result_q = item
            is_numpy = isinstance(audio_or_path, np.ndarray)
            label = f"numpy array ({len(audio_or_path)} samples)" if is_numpy else str(audio_or_path)
            print(f"[TRANSCRIBE] Transcribing {label}...")
            try:
                segments, _info = self._model.transcribe(
                    audio_or_path,
                    beam_size=5,
                    language="en",
                    initial_prompt=initial_prompt or None,
                )
                text = "".join(seg.text for seg in segments).strip()
                print(f'[TRANSCRIBE] Result: "{text}"')
            except Exception as exc:
                logger.error("Transcription error: %s", exc)
                print(f"[TRANSCRIBE] Error: {exc}")
                text = ""

            result_q.put(text)
