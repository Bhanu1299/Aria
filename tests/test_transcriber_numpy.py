"""
tests/test_transcriber_numpy.py — TDD tests for numpy transcription path (Task 14)

These tests must FAIL before implementation, then PASS after.
"""
from __future__ import annotations

import queue
import threading
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transcriber_with_mock_model(mock_model: MagicMock):
    """
    Build a Transcriber whose worker uses mock_model instead of loading
    a real WhisperModel.  We create a fake faster_whisper module so the
    worker thread's `from faster_whisper import WhisperModel` resolves to
    our mock without needing the real package installed.
    """
    import importlib

    # Create a fake faster_whisper module if not present
    fake_fw = types.ModuleType("faster_whisper")
    fake_fw.WhisperModel = MagicMock(return_value=mock_model)  # type: ignore[attr-defined]
    sys.modules.setdefault("faster_whisper", fake_fw)

    # If already cached (real or prior fake), override WhisperModel on it
    sys.modules["faster_whisper"].WhisperModel = MagicMock(return_value=mock_model)  # type: ignore[attr-defined]

    # Force-reload transcriber so it picks up the module-level numpy import etc.
    import transcriber as _transcriber_mod
    importlib.reload(_transcriber_mod)
    from transcriber import Transcriber

    t = Transcriber()
    return t


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestTranscriberNumpy(unittest.TestCase):

    def _build_mock_model(self, text: str = "hello world") -> MagicMock:
        """Return a mock WhisperModel whose transcribe() yields one segment."""
        seg = MagicMock()
        seg.text = text
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([seg]), MagicMock())
        return mock_model

    # ------------------------------------------------------------------
    # 1. transcribe_numpy returns a string
    # ------------------------------------------------------------------
    def test_transcribe_numpy_returns_string(self):
        mock_model = self._build_mock_model("test transcription")
        t = _make_transcriber_with_mock_model(mock_model)
        try:
            audio = np.zeros(1000, dtype=np.float32)
            result = t.transcribe_numpy(audio)
            self.assertIsInstance(result, str)
            self.assertEqual(result, "test transcription")
        finally:
            t.stop()

    # ------------------------------------------------------------------
    # 2. float32 array is passed directly to model.transcribe (not a path)
    # ------------------------------------------------------------------
    def test_transcribe_numpy_accepts_float32_array(self):
        mock_model = self._build_mock_model("numpy audio")
        t = _make_transcriber_with_mock_model(mock_model)
        try:
            audio = np.zeros(1600, dtype=np.float32)
            t.transcribe_numpy(audio)

            # The first positional arg to model.transcribe must be the ndarray,
            # not a string path
            call_args = mock_model.transcribe.call_args
            first_arg = call_args[0][0] if call_args[0] else call_args[1].get("audio")
            self.assertIsInstance(
                first_arg,
                np.ndarray,
                "model.transcribe should receive an ndarray, not a file path",
            )
        finally:
            t.stop()

    # ------------------------------------------------------------------
    # 3. get_audio_array returns float32 normalized to [-1, 1]
    # ------------------------------------------------------------------
    def test_get_audio_array_returns_normalized_float32(self):
        from voice_capture import VoiceCapture

        vc = VoiceCapture()
        # Inject some int16 chunks directly (bypasses sounddevice)
        chunk1 = np.array([0, 16384, -16384, 32767], dtype=np.int16).reshape(-1, 1)
        chunk2 = np.array([-32768, 100, -100, 0], dtype=np.int16).reshape(-1, 1)
        vc._chunks = [chunk1, chunk2]

        result = vc.get_audio_array()

        self.assertIsNotNone(result)
        self.assertEqual(result.dtype, np.float32)
        # All values must lie in [-1, 1]
        self.assertLessEqual(float(np.max(result)), 1.0)
        self.assertGreaterEqual(float(np.min(result)), -1.0)
        # int16 max (32767) / 32768.0 ≈ 0.9999…
        expected_max = 32767.0 / 32768.0
        self.assertAlmostEqual(float(np.max(result)), expected_max, places=4)
        # int16 min (-32768) / 32768.0 = -1.0
        self.assertAlmostEqual(float(np.min(result)), -1.0, places=4)
        assert result.ndim == 1

    # ------------------------------------------------------------------
    # 4. get_audio_array returns None when _chunks is empty
    # ------------------------------------------------------------------
    def test_get_audio_array_returns_none_when_no_chunks(self):
        from voice_capture import VoiceCapture

        vc = VoiceCapture()
        # _chunks starts as [] — don't add anything
        result = vc.get_audio_array()
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
