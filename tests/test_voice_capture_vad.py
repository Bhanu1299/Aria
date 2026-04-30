from __future__ import annotations

import os
import sys
import threading
import time
import types
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import voice_capture as _vc_module
from voice_capture import VoiceCapture, SAMPLE_RATE, _SILENCE_THRESHOLD, _SILENCE_WINDOW_SAMPLES


def _silent_chunk(frames: int = 512) -> np.ndarray:
    """Return a chunk of silence (all zeros) shaped (frames, 1)."""
    return np.zeros((frames, 1), dtype=np.int16)


def _loud_chunk(frames: int = 512) -> np.ndarray:
    """Return a chunk above silence threshold — RMS well above 983."""
    # Use value = 5000; RMS ≈ 5000 which is >> 983 threshold
    return np.full((frames, 1), 5000, dtype=np.int16)


def _inject_chunks(vc: VoiceCapture, chunks: list[np.ndarray]) -> None:
    """Feed audio chunks directly into VoiceCapture._audio_callback."""
    # Use a dummy CallbackFlags-like value (0 is falsy, so status check is skipped)
    for chunk in chunks:
        frames = chunk.shape[0]
        vc._audio_callback(chunk, frames, None, 0)


@contextmanager
def _mock_audio(mock_stream: MagicMock):
    """
    Replace voice_capture's sd and sf module-level references with mocks.
    This is robust against other tests stubbing sys.modules['sounddevice'].
    """
    real_sd = _vc_module.sd
    real_sf = _vc_module.sf

    fake_sd = types.ModuleType("sounddevice")
    fake_sd.InputStream = MagicMock(return_value=mock_stream)
    fake_sd.PortAudioError = Exception  # use base Exception as stand-in

    fake_sf = types.ModuleType("soundfile")
    mock_write = MagicMock()
    fake_sf.write = mock_write

    _vc_module.sd = fake_sd
    _vc_module.sf = fake_sf
    try:
        yield mock_write
    finally:
        _vc_module.sd = real_sd
        _vc_module.sf = real_sf


class TestVADAutoStop(unittest.TestCase):

    def setUp(self):
        self.mock_stream = MagicMock()
        self.mock_stream.start = MagicMock()
        self.mock_stream.stop = MagicMock()
        self.mock_stream.close = MagicMock()

    # ------------------------------------------------------------------
    # Test 1: auto-stop fires after 1.5s of silence
    # ------------------------------------------------------------------
    def test_auto_stop_fires_after_silence(self):
        called_event = threading.Event()
        callback_called = []

        def on_auto_stop():
            callback_called.append(True)
            called_event.set()

        with _mock_audio(self.mock_stream):
            vc = VoiceCapture()
            vc.start_recording(auto_stop=True, on_auto_stop=on_auto_stop)

            # Feed some loud chunks first (sound is present)
            _inject_chunks(vc, [_loud_chunk() for _ in range(10)])

            # Feed enough silent chunks to exceed _SILENCE_WINDOW_SAMPLES
            frames_per_chunk = 512
            silent_chunk_count = (_SILENCE_WINDOW_SAMPLES // frames_per_chunk) + 2
            _inject_chunks(vc, [_silent_chunk(frames_per_chunk) for _ in range(silent_chunk_count)])

            # Wait for the daemon thread to call on_auto_stop
            fired = called_event.wait(timeout=3.0)

        self.assertTrue(fired, "on_auto_stop was not called after injecting silence")
        self.assertEqual(len(callback_called), 1, "on_auto_stop should fire exactly once")

    # ------------------------------------------------------------------
    # Test 2: manual stop still works when auto_stop=True
    # ------------------------------------------------------------------
    def test_manual_stop_still_works_with_auto_stop_on(self):
        with _mock_audio(self.mock_stream) as mock_write:
            vc = VoiceCapture()
            vc.start_recording(auto_stop=True)

            # Inject only loud chunks — never reach silence threshold
            _inject_chunks(vc, [_loud_chunk() for _ in range(20)])

            # Manual stop should succeed
            wav_path = vc.stop_recording()

        self.assertEqual(wav_path, "/tmp/aria_recording.wav")
        mock_write.assert_called_once()

    # ------------------------------------------------------------------
    # Test 3: no auto-stop fires when audio is continuous sound
    # ------------------------------------------------------------------
    def test_no_auto_stop_when_speaking_continuously(self):
        callback_called = []

        def on_auto_stop():
            callback_called.append(True)

        with _mock_audio(self.mock_stream):
            vc = VoiceCapture()
            vc.start_recording(auto_stop=True, on_auto_stop=on_auto_stop)

            # Inject only loud chunks — never silence
            _inject_chunks(vc, [_loud_chunk() for _ in range(50)])

            # Brief pause to ensure no spurious trigger
            time.sleep(0.2)

            # Clean up
            try:
                vc.stop_recording()
            except Exception:
                pass

        self.assertEqual(len(callback_called), 0, "on_auto_stop should NOT fire during continuous speech")


if __name__ == "__main__":
    unittest.main()
