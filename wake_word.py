"""
wake_word.py — Aria Phase 3D: Always-on wake word detection.

Two backends (auto-selected at startup):

  BACKEND A — Porcupine (preferred)
    Requires:
      ~/.aria/hey_aria.ppn  (custom model from picovoice.ai console)
      PORCUPINE_KEY         (free access key from picovoice.ai)
    Steps to set up:
      1. picovoice.ai → free account → get access key
      2. Console → Wake Word → "Hey Aria" → download macOS .ppn → save to ~/.aria/hey_aria.ppn
      3. Add PORCUPINE_KEY=<key> to .env (or export it in shell)
    Porcupine is ~1% CPU, purpose-built, reliable — same architecture as Siri.

  BACKEND B — openwakeword fallback
    Uses alexa model as phonetic proxy for "Aria" (best available match).
    Threshold 0.35 (lower than training threshold because the phrase doesn't match exactly).
    Will false-trigger occasionally; replace with Porcupine for reliability.
    Requires: openwakeword, pyaudio (in requirements.txt)

Siri-like behavior (both backends):
  1. Always-on mic → tiny model → wake word detected → ding
  2. VAD records until user stops talking
  3. Whisper transcribes → handle_command(transcript) called
  4. Returns to listening; 5s cooldown prevents re-triggering
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
import wave
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 16000
_CHUNK_SIZE = 512          # 32ms @ 16kHz — works for both backends
_COOLDOWN_SECS = 5.0

# openwakeword fallback
_OWW_MODEL = "alexa"       # best phonetic proxy for "Aria" (vowel-heavy, 3 syllables)
_OWW_THRESHOLD = 0.35      # intentionally low — proxy model, not exact phrase

# VAD
_VAD_SPEECH_RMS = 400
_VAD_SILENCE_SECS = 1.2
_VAD_MAX_SECS = 10.0
_PRE_SPEECH_TIMEOUT = 3.0

_WAV_PATH = "/tmp/aria_wake_recording.wav"
_DING_SOUND = "/System/Library/Sounds/Tink.aiff"

# Porcupine config
_PORCUPINE_MODEL_PATH = os.path.expanduser("~/.aria/hey_aria.ppn")
_PORCUPINE_KEY_ENV = "PORCUPINE_KEY"


class WakeWordListener:
    """
    Always-on wake word listener. Runs on a permanent daemon thread.

    Automatically selects the best available backend:
      - Porcupine if ~/.aria/hey_aria.ppn + PORCUPINE_KEY exist
      - openwakeword (alexa proxy, threshold 0.35) as fallback

    After wake word fires:
      1. Plays a ding (non-blocking)
      2. Records from mic using energy-based VAD
      3. Transcribes via transcriber.transcribe()
      4. Calls handle_command(transcript)
      5. Returns to listening after cooldown
    """

    def __init__(
        self,
        handle_command_fn: Callable[[str], None],
        processing_event: threading.Event | None = None,
        transcriber=None,
    ) -> None:
        self._handle_command = handle_command_fn
        self._processing = processing_event
        self._transcriber = transcriber
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="aria-wake-word",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Backend detection
    # ------------------------------------------------------------------

    def _can_use_porcupine(self) -> bool:
        """Returns True if Porcupine backend is fully configured."""
        if not os.path.isfile(_PORCUPINE_MODEL_PATH):
            return False
        key = os.environ.get(_PORCUPINE_KEY_ENV, "").strip()
        if not key:
            try:
                from dotenv import dotenv_values
                env_file = os.path.join(os.path.dirname(__file__), ".env")
                vals = dotenv_values(env_file)
                key = vals.get(_PORCUPINE_KEY_ENV, "").strip()
            except Exception:
                pass
        if not key:
            return False
        try:
            import pvporcupine  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Porcupine backend
    # ------------------------------------------------------------------

    def _run_porcupine(self) -> None:
        import pvporcupine
        import pyaudio as _pa

        # Load PORCUPINE_KEY from env or .env file
        key = os.environ.get(_PORCUPINE_KEY_ENV, "").strip()
        if not key:
            try:
                from dotenv import dotenv_values
                env_file = os.path.join(os.path.dirname(__file__), ".env")
                vals = dotenv_values(env_file)
                key = vals.get(_PORCUPINE_KEY_ENV, "").strip()
            except Exception:
                pass

        try:
            porcupine = pvporcupine.create(
                access_key=key,
                keyword_paths=[_PORCUPINE_MODEL_PATH],
            )
        except Exception as exc:
            print(f"[Aria] Porcupine init failed: {exc} — falling back to openwakeword")
            self._run_openwakeword()
            return

        pa = _pa.PyAudio()
        stream = None
        try:
            stream = pa.open(
                rate=porcupine.sample_rate,
                channels=1,
                format=_pa.paInt16,
                input=True,
                frames_per_buffer=porcupine.frame_length,
            )
            print(f"[Aria] Wake word active (Porcupine, 'Hey Aria').")
            last_triggered = 0.0

            while not self._stop_event.is_set():
                try:
                    pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
                except OSError as exc:
                    logger.warning("Porcupine mic read error: %s", exc)
                    time.sleep(0.1)
                    continue

                now = time.time()
                in_cooldown = (now - last_triggered) < _COOLDOWN_SECS
                is_processing = self._processing is not None and self._processing.is_set()
                if in_cooldown or is_processing:
                    continue

                pcm_np = np.frombuffer(pcm, dtype=np.int16)
                result = porcupine.process(pcm_np)
                if result < 0:
                    continue

                print("[Aria] Wake word detected: Hey Aria (Porcupine)")
                self._on_wake(stream, porcupine.sample_rate, porcupine.frame_length)
                last_triggered = time.time()  # cooldown starts AFTER full pipeline completes

        except Exception as exc:
            logger.error("Porcupine listener crashed: %s", exc)
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            try:
                porcupine.delete()
            except Exception:
                pass
            try:
                pa.terminate()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # openwakeword fallback backend
    # ------------------------------------------------------------------

    def _run_openwakeword(self) -> None:
        try:
            import openwakeword
            from openwakeword.model import Model
            import pyaudio as _pa
        except ImportError as exc:
            print(f"[Aria] Wake word disabled — openwakeword/pyaudio not installed: {exc}")
            return

        try:
            openwakeword.utils.download_models()
        except Exception:
            pass  # already cached

        try:
            oww = Model(wakeword_models=[_OWW_MODEL], inference_framework="onnx")
        except Exception as exc:
            print(f"[Aria] Wake word disabled — model load failed: {exc}")
            return

        pa = _pa.PyAudio()
        stream = None
        try:
            stream = pa.open(
                rate=_SAMPLE_RATE,
                channels=1,
                format=_pa.paInt16,
                input=True,
                frames_per_buffer=_CHUNK_SIZE,
            )
            print(
                f"[Aria] Wake word active (openwakeword proxy '{_OWW_MODEL}', "
                f"threshold={_OWW_THRESHOLD}). "
                "For reliable 'Hey Aria' detection, set up Porcupine — see wake_word.py header."
            )
            last_triggered = 0.0

            while not self._stop_event.is_set():
                try:
                    chunk = stream.read(_CHUNK_SIZE, exception_on_overflow=False)
                except OSError as exc:
                    logger.warning("Wake word mic read error: %s", exc)
                    time.sleep(0.1)
                    continue

                audio_np = np.frombuffer(chunk, dtype=np.int16)

                now = time.time()
                in_cooldown = (now - last_triggered) < _COOLDOWN_SECS
                is_processing = self._processing is not None and self._processing.is_set()
                if in_cooldown or is_processing:
                    continue

                oww.predict(audio_np)

                triggered = False
                for model_name, scores in oww.prediction_buffer.items():
                    if scores and scores[-1] >= _OWW_THRESHOLD:
                        triggered = True
                        print(f"[Aria] Wake word proxy triggered: {model_name} score={scores[-1]:.2f}")
                        break

                if not triggered:
                    continue

                # Reset scores to prevent double-trigger
                for model_name in oww.prediction_buffer:
                    oww.prediction_buffer[model_name] = []

                last_triggered = time.time()
                self._on_wake(stream, _SAMPLE_RATE, _CHUNK_SIZE)

        except Exception as exc:
            logger.error("openwakeword listener crashed: %s", exc)
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            try:
                pa.terminate()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Shared: post-wake recording + dispatch
    # ------------------------------------------------------------------

    def _on_wake(self, stream, sample_rate: int, chunk_size: int) -> None:
        """Called immediately after any backend detects the wake word."""
        self._play_ding()
        wav_path = self._record_until_silence(stream, sample_rate, chunk_size)
        if wav_path is None:
            return  # no speech after wake word — go back to listening

        if self._transcriber is None:
            logger.error("Wake word: no transcriber — cannot process command")
            return

        try:
            transcript = self._transcriber.transcribe(wav_path)
        except Exception as exc:
            logger.error("Wake word transcription failed: %s", exc)
            return

        if not transcript or not transcript.strip():
            return

        try:
            self._handle_command(transcript)
        except Exception as exc:
            logger.error("Wake word handle_command failed: %s", exc)

    def _play_ding(self) -> None:
        if os.path.exists(_DING_SOUND):
            threading.Thread(
                target=lambda: subprocess.run(
                    ["afplay", _DING_SOUND],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ),
                daemon=True,
            ).start()

    def _rms(self, audio_np: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio_np.astype(np.float32) ** 2)))

    def _record_until_silence(
        self, stream, sample_rate: int, chunk_size: int
    ) -> str | None:
        """VAD recording from the active mic stream. Returns WAV path or None."""
        chunks: list[bytes] = []
        speech_started = False
        speech_duration = 0.0
        silence_duration = 0.0
        pre_speech_elapsed = 0.0
        chunk_secs = chunk_size / sample_rate
        total_elapsed = 0.0

        while total_elapsed < _VAD_MAX_SECS:
            try:
                chunk = stream.read(chunk_size, exception_on_overflow=False)
            except OSError:
                break

            audio_np = np.frombuffer(chunk, dtype=np.int16)
            rms = self._rms(audio_np)
            total_elapsed += chunk_secs

            if rms >= _VAD_SPEECH_RMS:
                speech_started = True
                silence_duration = 0.0
                speech_duration += chunk_secs
                chunks.append(chunk)
            elif speech_started:
                silence_duration += chunk_secs
                chunks.append(chunk)
                if silence_duration >= _VAD_SILENCE_SECS:
                    break
            else:
                chunks.append(chunk)
                pre_speech_elapsed += chunk_secs
                if pre_speech_elapsed >= _PRE_SPEECH_TIMEOUT:
                    return None

        if not speech_started or speech_duration < 0.2:
            return None

        try:
            with wave.open(_WAV_PATH, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # int16 = 2 bytes
                wf.setframerate(sample_rate)
                wf.writeframes(b"".join(chunks))
            return _WAV_PATH
        except Exception as exc:
            logger.error("Wake word: failed to save WAV: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Entry point — selects backend
    # ------------------------------------------------------------------

    def _run(self) -> None:
        if self._can_use_porcupine():
            print("[Aria] Wake word backend: Porcupine (custom 'Hey Aria' model)")
            self._run_porcupine()
        else:
            print("[Aria] Wake word backend: openwakeword fallback (proxy model)")
            self._run_openwakeword()
