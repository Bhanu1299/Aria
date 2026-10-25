"""voice/vad.py — is this 32 ms of audio someone speaking? (Silero VAD v5, ONNX, ~1 ms/chunk)

Loudness (RMS) can't tell a voice from a fan, a TV's music or Aria's own cue sound; a speech
model can (2026-10-25: the follow-up mic and the command recording both used RMS gates). Same
framing as the mic: 16 kHz, 512-sample chunks. The model keeps state between chunks — one
SpeechDetector per recording, reset() between them.

Model file: ~/.aria/models/silero_vad.onnx (github.com/snakers4/silero-vad, MIT). Missing or
broken → detector() returns None and callers keep their RMS gate. ARIA_VAD=0 turns it off.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = Path.home() / ".aria" / "models" / "silero_vad.onnx"
SAMPLE_RATE = 16000
CHUNK = 512
_CONTEXT = 64                  # v5 looks at the last 64 samples of the previous chunk too
START = 0.5                    # speech starts at this probability …
END = 0.35                     # … and counts as stopped only below this (hysteresis)

_session = None
_session_lock = threading.Lock()
_failed = False


def _load():
    global _session, _failed
    with _session_lock:
        if _session is None and not _failed:
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.inter_op_num_threads = 1
                opts.intra_op_num_threads = 1
                _session = ort.InferenceSession(str(MODEL_PATH), sess_options=opts)
            except Exception as exc:
                _failed = True
                logger.warning("VAD unavailable (%s) — using the loudness gate", exc)
        return _session


class SpeechDetector:
    def __init__(self, session) -> None:
        self._s = session
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT), dtype=np.float32)
        self.speaking = False

    def prob(self, chunk) -> float:
        """Speech probability of one 512-sample chunk (int16 or float32 in [-1, 1])."""
        a = np.asarray(chunk).reshape(-1)
        if a.dtype != np.float32:
            a = a.astype(np.float32) / 32768.0
        if a.size != CHUNK:
            a = np.pad(a, (0, max(0, CHUNK - a.size)))[:CHUNK]
        x = np.concatenate([self._context, a.reshape(1, -1)], axis=1)
        out, self._state = self._s.run(None, {"input": x, "state": self._state,
                                              "sr": np.array(SAMPLE_RATE, dtype=np.int64)})
        self._context = x[:, -_CONTEXT:]
        return float(out[0][0])

    def is_speech(self, chunk) -> bool:
        """With hysteresis: starts at START, keeps going until the probability drops below END."""
        p = self.prob(chunk)
        self.speaking = p >= (END if self.speaking else START)
        return self.speaking


def detector() -> Optional[SpeechDetector]:
    """A fresh detector, or None (no model / ARIA_VAD=0) — then use the RMS gate."""
    if os.getenv("ARIA_VAD", "1").strip().lower() in {"0", "false", "off", "no"}:
        return None
    if not MODEL_PATH.exists():
        return None
    s = _load()
    return SpeechDetector(s) if s is not None else None
