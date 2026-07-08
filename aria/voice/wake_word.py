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
    Uses custom-trained aria.onnx model (~/.aria/aria.onnx).
    Threshold 0.7 (tune in range 0.6–0.8 based on real-world false positive rate).
    Requires: openwakeword, pyaudio (in requirements.txt)

Siri-like behavior (both backends):
  1. Always-on mic → tiny model → wake word detected → ding
  2. VAD records until user stops talking
  3. Whisper transcribes → handle_command(transcript) called
  4. Returns to listening; 5s cooldown prevents re-triggering
"""

from __future__ import annotations

from aria.core import paths as _aria_paths

import logging
import os
import subprocess
import threading
import time
import wave
from typing import Callable

import numpy as np

import aria.ui.listening_indicator as listening_indicator
import aria.observability.wake_stats as wake_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 16000
_CHUNK_SIZE = 512          # 32ms @ 16kHz — works for both backends
_COOLDOWN_SECS = 5.0

# openwakeword fallback
_OWW_MODEL = "/Users/bhanuteja/.aria/aria.onnx"
_OWW_THRESHOLD = 0.7  # tune after real-world testing (range 0.6–0.8)

# VAD
_VAD_SPEECH_RMS = 400
_VAD_SILENCE_SECS = 1.2
_VAD_MAX_SECS = 10.0
_PRE_SPEECH_TIMEOUT = 3.0

_WAV_PATH = "/tmp/aria_wake_recording.wav"

# Activation chime — Glass is the most Siri-like and far more audible than
# Tink. Override with ARIA_WAKE_SOUND=/path/to/sound.aiff (or Tink etc.).
_DEFAULT_DING = "/System/Library/Sounds/Glass.aiff"
_FALLBACK_DING = "/System/Library/Sounds/Tink.aiff"
_DING_SOUND = os.environ.get("ARIA_WAKE_SOUND", "").strip() or _DEFAULT_DING
if not os.path.exists(_DING_SOUND):
    _DING_SOUND = _FALLBACK_DING
# Played when the wake word fired but no speech followed — so silence after
# the chime is never a mystery.
_NO_SPEECH_SOUND = "/System/Library/Sounds/Basso.aiff"

# Scores >= this (but below the trigger threshold) are logged as near-misses
# so the threshold can be tuned from real data via `daily_check.py wake`.
_NEAR_MISS_FLOOR = 0.40

# Restart backoff when a backend crashes or its mic stream dies.
_RESTART_BACKOFF_START = 5.0
_RESTART_BACKOFF_MAX = 60.0

# Porcupine config
_PORCUPINE_MODEL_PATH = os.path.expanduser("~/.aria/hey_aria.ppn")
_PORCUPINE_KEY_ENV = "PORCUPINE_KEY"

# Custom OWW model (set up by training pipeline)
_CUSTOM_MODEL_PATH = os.path.expanduser("~/.aria/aria.onnx")
_CUSTOM_THRESHOLD = 0.7   # starting point — tune after real-world testing
_CUSTOM_N_MFCC = 20
_CUSTOM_WINDOW_SECS = 2.0


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
        menubar=None,
    ) -> None:
        self._handle_command = handle_command_fn
        self._processing = processing_event
        self._transcriber = transcriber
        self._menubar = menubar
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._backend_name = "none"

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
                env_file = str(_aria_paths.ENV_FILE)
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

    def _can_use_custom_model(self) -> bool:
        """Returns True if a custom-trained ONNX model exists at ~/.aria/aria.onnx."""
        if not os.path.isfile(_CUSTOM_MODEL_PATH):
            return False
        try:
            import onnxruntime  # noqa: F401
            import librosa  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Porcupine backend
    # ------------------------------------------------------------------

    def _run_porcupine(self) -> bool:
        """Returns True if restartable (crash/stream death), False if disabled."""
        import pvporcupine
        import pyaudio as _pa

        # Load PORCUPINE_KEY from env or .env file
        key = os.environ.get(_PORCUPINE_KEY_ENV, "").strip()
        if not key:
            try:
                from dotenv import dotenv_values
                env_file = str(_aria_paths.ENV_FILE)
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
            return self._run_openwakeword()

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
            self._backend_name = "porcupine"
            last_triggered = 0.0

            while not self._stop_event.is_set():
                wake_stats.heartbeat("porcupine")
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
                wake_stats.log_event("detection", "porcupine")
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
        return True

    # ------------------------------------------------------------------
    # openwakeword fallback backend
    # ------------------------------------------------------------------

    def _run_openwakeword(self) -> bool:
        """Returns True if restartable (crash/stream death), False if disabled."""
        try:
            import openwakeword
            from openwakeword.model import Model
            import pyaudio as _pa
        except ImportError as exc:
            print(f"[Aria] Wake word disabled — openwakeword/pyaudio not installed: {exc}")
            return False

        try:
            openwakeword.utils.download_models()
        except Exception:
            pass  # already cached

        import os
        model_to_load = _OWW_MODEL
        if model_to_load != "alexa" and not os.path.exists(model_to_load):
            print(f"[Aria] Custom model not found at {model_to_load} — falling back to alexa proxy.")
            print("[Aria] Run training/record_samples.py + train_model.py + export_model.py to build your model.")
            model_to_load = "alexa"

        try:
            oww = Model(wakeword_models=[model_to_load], inference_framework="onnx")
        except Exception as exc:
            print(f"[Aria] Wake word disabled — model load failed: {exc}")
            return False

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
                f"[Aria] Wake word active (openwakeword '{model_to_load}', "
                f"threshold={_OWW_THRESHOLD})."
            )
            self._backend_name = "openwakeword"
            last_triggered = 0.0
            last_near_miss = 0.0

            while not self._stop_event.is_set():
                wake_stats.heartbeat("openwakeword")
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
                best_score = 0.0
                for model_name, scores in oww.prediction_buffer.items():
                    if scores:
                        best_score = max(best_score, float(scores[-1]))
                    if scores and scores[-1] >= _OWW_THRESHOLD:
                        triggered = True
                        print(f"[Aria] Wake word proxy triggered: {model_name} score={scores[-1]:.2f}")
                        break

                if not triggered:
                    # An almost-trigger: log once per 3s burst so the
                    # threshold can be tuned from real usage.
                    if best_score >= _NEAR_MISS_FLOOR and now - last_near_miss > 3.0:
                        last_near_miss = now
                        wake_stats.log_event("near_miss", "openwakeword", best_score)
                    continue

                # Reset scores to prevent double-trigger
                for model_name in oww.prediction_buffer:
                    oww.prediction_buffer[model_name] = []

                wake_stats.log_event("detection", "openwakeword", best_score)
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
        return True

    # ------------------------------------------------------------------
    # Custom ONNX backend
    # ------------------------------------------------------------------

    def _run_custom_onnx(self) -> bool:
        """Sliding-window MFCC inference using the custom-trained aria.onnx model.

        Returns True if restartable (crash/stream death), False if disabled.
        """
        try:
            import onnxruntime as ort
            import librosa
            import pyaudio as _pa
        except ImportError as exc:
            print(f"[Aria] Custom model disabled — missing dep: {exc}")
            return self._run_openwakeword()

        try:
            session = ort.InferenceSession(_CUSTOM_MODEL_PATH)
            input_name = session.get_inputs()[0].name
            label_name = session.get_outputs()[1].name
        except Exception as exc:
            print(f"[Aria] Custom model load failed: {exc} — falling back to openwakeword")
            return self._run_openwakeword()

        pa = _pa.PyAudio()
        stream = None
        window_samples = int(_SAMPLE_RATE * _CUSTOM_WINDOW_SECS)
        buffer = np.zeros(window_samples, dtype=np.int16)
        last_triggered = 0.0

        try:
            stream = pa.open(
                rate=_SAMPLE_RATE, channels=1,
                format=_pa.paInt16, input=True,
                frames_per_buffer=_CHUNK_SIZE,
            )
            print(f"[Aria] Wake word active (custom ONNX model, threshold={_CUSTOM_THRESHOLD}).")
            self._backend_name = "custom_onnx"
            last_near_miss = 0.0

            while not self._stop_event.is_set():
                wake_stats.heartbeat("custom_onnx")
                try:
                    chunk = stream.read(_CHUNK_SIZE, exception_on_overflow=False)
                except OSError as exc:
                    logger.warning("Custom model mic read error: %s", exc)
                    time.sleep(0.1)
                    continue

                # Slide buffer: drop oldest _CHUNK_SIZE samples, append new chunk
                new_samples = np.frombuffer(chunk, dtype=np.int16)
                buffer = np.roll(buffer, -_CHUNK_SIZE)
                buffer[-_CHUNK_SIZE:] = new_samples

                now = time.time()
                if (now - last_triggered) < _COOLDOWN_SECS:
                    continue
                if self._processing is not None and self._processing.is_set():
                    continue

                # Extract MFCC features and run inference
                try:
                    audio_f32 = buffer.astype(np.float32) / 32768.0
                    mfcc = librosa.feature.mfcc(
                        y=audio_f32, sr=_SAMPLE_RATE, n_mfcc=_CUSTOM_N_MFCC
                    )
                    feat = np.mean(mfcc, axis=1).astype(np.float32).reshape(1, -1)
                    probs = session.run([label_name], {input_name: feat})[0]
                    score = float(probs[0][1])
                except Exception as exc:
                    logger.debug("Custom model inference error: %s", exc)
                    continue

                if score >= _CUSTOM_THRESHOLD:
                    print(f"[Aria] Wake word detected (custom model, score={score:.2f})")
                    wake_stats.log_event("detection", "custom_onnx", score)
                    self._on_wake(stream, _SAMPLE_RATE, _CHUNK_SIZE)
                    last_triggered = time.time()
                elif score >= _NEAR_MISS_FLOOR and now - last_near_miss > 3.0:
                    last_near_miss = now
                    wake_stats.log_event("near_miss", "custom_onnx", score)

        except Exception as exc:
            logger.error("Custom ONNX listener crashed: %s", exc)
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
        return True

    # ------------------------------------------------------------------
    # Shared: post-wake recording + dispatch
    # ------------------------------------------------------------------

    def _set_menubar(self, state: str) -> None:
        if self._menubar is None:
            return
        try:
            self._menubar.set_state(state)
        except Exception:
            pass

    def _on_wake(self, stream, sample_rate: int, chunk_size: int) -> None:
        """Called immediately after any backend detects the wake word."""
        self._play_sound(_DING_SOUND)
        self._set_menubar("LISTENING")
        listening_indicator.show("Listening...")

        wav_path = self._record_until_silence(stream, sample_rate, chunk_size)
        listening_indicator.hide()

        if wav_path is None:
            # Wake word fired but no speech followed — make that audible so
            # the user knows Aria woke up and then gave up.
            wake_stats.log_event("no_speech", self._backend_name)
            self._play_sound(_NO_SPEECH_SOUND)
            self._set_menubar("IDLE")
            return

        if self._transcriber is None:
            logger.error("Wake word: no transcriber — cannot process command")
            self._set_menubar("IDLE")
            return

        try:
            transcript = self._transcriber.transcribe(wav_path)
        except Exception as exc:
            logger.error("Wake word transcription failed: %s", exc)
            self._set_menubar("IDLE")
            return

        if not transcript or not transcript.strip():
            wake_stats.log_event("no_speech", self._backend_name)
            self._play_sound(_NO_SPEECH_SOUND)
            self._set_menubar("IDLE")
            return

        try:
            self._handle_command(transcript)
        except Exception as exc:
            logger.error("Wake word handle_command failed: %s", exc)

    def _play_sound(self, path: str) -> None:
        if os.path.exists(path):
            threading.Thread(
                target=lambda: subprocess.run(
                    ["afplay", path],
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
        """Backend supervisor: restart on crash with backoff, never die silently."""
        backoff = _RESTART_BACKOFF_START
        while not self._stop_event.is_set():
            try:
                if self._can_use_porcupine():
                    print("[Aria] Wake word backend: Porcupine (custom 'Hey Aria' model)")
                    restartable = self._run_porcupine()
                elif self._can_use_custom_model():
                    print("[Aria] Wake word backend: custom ONNX model (~/.aria/aria.onnx)")
                    restartable = self._run_custom_onnx()
                else:
                    print("[Aria] Wake word backend: openwakeword fallback (proxy model)")
                    restartable = self._run_openwakeword()
            except Exception as exc:
                logger.error("Wake word backend crashed: %s", exc)
                restartable = True

            if self._stop_event.is_set():
                return
            if not restartable:
                print("[Aria] Wake word permanently disabled (missing deps or model). "
                      "Hotkey still works.")
                return

            print(f"[Aria] Wake word listener exited — restarting in {backoff:.0f}s")
            wake_stats.log_event("restart", self._backend_name)
            if self._stop_event.wait(backoff):
                return
            backoff = min(backoff * 2, _RESTART_BACKOFF_MAX)
