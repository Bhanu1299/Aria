from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf


SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
OUTPUT_PATH = "/tmp/aria_recording.wav"

# VAD constants
_SILENCE_THRESHOLD: float = 0.03 * 32767   # 3% of max int16 ≈ 983.0
_SILENCE_WINDOW_SAMPLES: int = int(1.5 * SAMPLE_RATE)  # 1.5s at 16kHz = 24000 samples


class VoiceCapture:
    def __init__(self):
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None

        # VAD state — initialised here so attrs always exist
        self._auto_stop: bool = False
        self._on_auto_stop: Optional[callable] = None
        self._silence_samples: int = 0
        self._auto_stop_triggered: bool = False

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            print(f"[VOICE] Stream status: {status}")
        self._chunks.append(indata.copy())

        if self._auto_stop and not self._auto_stop_triggered:
            rms = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2)))
            if rms < _SILENCE_THRESHOLD:
                self._silence_samples += frames
            else:
                self._silence_samples = 0  # reset on sound

            if self._silence_samples >= _SILENCE_WINDOW_SAMPLES:
                self._auto_stop_triggered = True
                t = threading.Thread(target=self._fire_auto_stop, daemon=True)
                t.start()

    def _fire_auto_stop(self) -> None:
        """Called from a daemon thread — stop recording and invoke callback."""
        try:
            self.stop_recording()
        except Exception:
            pass
        if self._on_auto_stop is not None:
            try:
                self._on_auto_stop()
            except Exception:
                pass

    def start_recording(
        self,
        auto_stop: bool = False,
        on_auto_stop: Optional[callable] = None,
    ) -> None:
        self._chunks = []

        # Init VAD state before starting the stream
        self._auto_stop = auto_stop
        self._on_auto_stop = on_auto_stop
        self._silence_samples = 0
        self._auto_stop_triggered = False

        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                callback=self._audio_callback,
            )
            self._stream.start()
        except sd.PortAudioError as e:
            print(f"[VOICE] ERROR: Microphone not available — {e}")
            raise

        print("[VOICE] Recording started")

    def stop_recording(self) -> str:
        if self._stream is None:
            raise RuntimeError(
                "[VOICE] stop_recording() called before start_recording()"
            )

        self._stream.stop()
        self._stream.close()
        self._stream = None

        if not self._chunks:
            raise RuntimeError("[VOICE] No audio data captured.")

        audio = np.concatenate(self._chunks, axis=0)

        try:
            sf.write(OUTPUT_PATH, audio, SAMPLE_RATE, subtype="PCM_16")
        except Exception as e:
            print(f"[VOICE] ERROR: Could not save WAV file — {e}")
            raise

        print(f"[VOICE] Recording stopped, saved to {OUTPUT_PATH}")
        self._chunks = []
        return OUTPUT_PATH

    def record_once(self, max_seconds: int = 5) -> Optional[str]:
        """Record for a fixed duration and return the WAV path. Returns None on error."""
        try:
            audio = sd.rec(
                int(max_seconds * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
            )
            sd.wait()
            path = "/tmp/aria_confirm.wav"
            sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
            return path
        except Exception as exc:
            print(f"[VOICE] record_once failed: {exc}")
            return None
