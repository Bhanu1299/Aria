from typing import Optional
import numpy as np
import sounddevice as sd
import soundfile as sf


SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
OUTPUT_PATH = "/tmp/aria_recording.wav"


class VoiceCapture:
    def __init__(self):
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None

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

    def start_recording(self) -> None:
        self._chunks = []

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
