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
import re
import os
import subprocess
import threading
import time
import wave
from typing import Callable, Optional

import numpy as np

import aria.ui.listening_indicator as listening_indicator
from aria.voice.wake_features import N_FEATURES, wake_features
from aria.voice import vad
import aria.observability.wake_stats as wake_stats

logger = logging.getLogger(__name__)


def fires_on_silence(predict: Callable[[np.ndarray], list]) -> bool:
    """True if the custom model scores 2s of digital silence at or above the trigger
    threshold. predict(features) → the model's probability output ([{1: p, ...}])."""
    try:
        feat = wake_features(np.zeros(int(_SAMPLE_RATE * _CUSTOM_WINDOW_SECS), np.float32)).reshape(1, -1)
        return float(predict(feat)[0].get(1, 0.0)) >= _CUSTOM_THRESHOLD
    except Exception as exc:
        logger.warning("Wake model silence check failed: %s", exc)
        return False

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
# Cutting Aria off mid-sentence: on speakers the mic also hears Aria, so a barge-in needs a
# higher score held over several consecutive windows (32 ms apart) — a real "Aria" stays high
# for a few hundred ms, a stray sound-alike in Aria's own speech rarely does.
_BARGE_THRESHOLD = 0.85
_BARGE_HITS = 3
# Second check (2026-10-23: real and false wakes both scored 0.70–0.73, so no threshold can
# separate them): Whisper transcribes the 2 s window that fired and the wake counts only if it
# heard "Aria". Recording carries on while it checks, so the command's first words aren't lost.
# Measured on the training clips (2026-10-23): primed with "Aria", Whisper heard the name in
# 80/80 of the user's clips (no_speech_prob ≤ 0.42); unprimed only 43/60. The prime makes it
# write "Aria" over noise / room sound too, but those all had no_speech_prob ≥ 0.45 → rejected.
# ARIA_WAKE_VERIFY=0 turns the check off.
_VERIFY_MAX_SECS = 3.0
_VERIFY_SETTLE_CHUNKS = 16    # 0.51 s of audio read after the crossing, before the check
# Second chance (Apple's "Hey Siri" does the same): a near-miss or a rejected check means you may
# have just said it — for a few seconds a lower score is enough, so saying it again wakes Aria
# at once. The Whisper check still guards every wake.
_SECOND_CHANCE_FLOOR = 0.55
_SECOND_CHANCE_THRESHOLD = 0.6
_SECOND_CHANCE_SECS = 4.0
_SECOND_CHANCE_REST_SECS = 10.0   # after a window is used up, none opens for this long


class SecondChance:
    """The lowered threshold after a near-miss or a rejected check — one window at a time.
    A reject of a fire that only crossed the lowered threshold closes the window and rests:
    before 2026-10-26 every such reject reopened it, so room noise scoring 0.60–0.69 kept
    Aria sensitive (and Whisper busy) for hours — ~200 rejected fires in one live run."""

    def __init__(self) -> None:
        self.until = 0.0
        self.rest_until = 0.0

    def threshold(self, now: float, normal: float) -> float:
        return _SECOND_CHANCE_THRESHOLD if now < self.until else normal

    def _open(self, now: float) -> None:
        if now >= self.rest_until and now >= self.until:
            self.until = now + _SECOND_CHANCE_SECS

    def near_miss(self, now: float, score: float) -> None:
        if score >= _SECOND_CHANCE_FLOOR:
            self._open(now)

    def rejected(self, now: float, lowered: bool) -> None:
        if lowered:                                   # the second chance was spent on noise
            self.until, self.rest_until = 0.0, now + _SECOND_CHANCE_REST_SECS
        else:
            self._open(now)

    def woke(self) -> None:
        self.until = 0.0
_MAX_NO_SPEECH = 0.45
# Speech gate (2026-10-26 live: the model scored plain room sound 0.60–0.76 — ~350 fires in 6 h,
# most with Whisper hearing nothing — while a real "Aria." was refused at no_speech 0.63). Silero
# VAD runs on every chunk: a crossing counts only if it heard a voice in the window, and a
# window it heard a voice in may pass Whisper's check at no_speech up to _MAX_NO_SPEECH_VOICED
# (the strict 0.45 was only there to stop the "Aria" prime being written over silence).
_VAD_GATE_MIN = 9              # ≈ 0.3 s: the user's clips have ≥ 12, noise / quiet clips ≤ 8 (2026-10-26)
_MAX_NO_SPEECH_VOICED = 0.8


def voiced_chunks(window: np.ndarray) -> Optional[int]:
    """How many 32 ms chunks of this window Silero VAD calls speech (a fresh detector, ~60 ms
    for 2 s). None when VAD isn't available — then nothing is gated."""
    det = vad.detector()
    if det is None:
        return None
    a = np.asarray(window)
    return sum(det.is_speech(a[i:i + vad.CHUNK]) for i in range(0, len(a) - vad.CHUNK + 1, vad.CHUNK))


_CLIPS_DIR = os.path.expanduser("~/.aria/wake_clips")
_CLIPS_KEEP = 300


def save_clip(window: np.ndarray, kind: str, score: float) -> None:
    """The 2 s that fired, for tuning on the real room (local only, newest _CLIPS_KEEP kept).
    ARIA_WAKE_CLIPS=0 turns it off. Never raises."""
    if os.getenv("ARIA_WAKE_CLIPS", "1").strip().lower() in {"0", "false", "off", "no"}:
        return
    try:
        os.makedirs(_CLIPS_DIR, exist_ok=True)
        path = os.path.join(_CLIPS_DIR, f"{time.strftime('%Y%m%d-%H%M%S')}_{kind}_{score:.2f}.wav")
        with wave.open(path, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(_SAMPLE_RATE)
            f.writeframes(np.asarray(window, dtype=np.int16).tobytes())
        old = sorted(os.listdir(_CLIPS_DIR))
        for name in old[:-_CLIPS_KEEP]:
            os.remove(os.path.join(_CLIPS_DIR, name))
    except Exception as exc:
        logger.debug("wake clip not saved: %s", exc)


def wake_accepted(text: str, no_speech: float, voiced_chunks: Optional[int] = None) -> bool:
    """Whisper heard the name, and either it's sure there's speech or VAD heard a voice."""
    if not heard_wake_word(text):
        return False
    if no_speech < _MAX_NO_SPEECH:
        return True
    return voiced_chunks is not None and voiced_chunks >= _VAD_GATE_MIN and no_speech < _MAX_NO_SPEECH_VOICED
_WAKE_WORDS = re.compile(r"\b(?:aria|arya|ariya|ariah|aryah|aarya|arria|aria's|arya's)\b")
_WAKE_FIRST = re.compile(r"^(?:hey |hi |ok |okay |yo )?(?:area|maria|ariel|arian|aaria|orya|oria)\b")


def heard_wake_word(text: str) -> bool:
    """Did Whisper hear the wake word? The name anywhere; its sound-alikes only up front."""
    t = re.sub(r"[^a-z' ]+", " ", (text or "").lower())
    t = re.sub(r"\s+", " ", t).strip()
    return bool(_WAKE_WORDS.search(t) or _WAKE_FIRST.match(t))


def _verify_enabled() -> bool:
    return os.getenv("ARIA_WAKE_VERIFY", "1").strip().lower() not in {"0", "false", "off", "no"}
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
        interruptible: Callable[[], bool] | None = None,
        on_barge_in: Callable[[], None] | None = None,
    ) -> None:
        """interruptible() → True while Aria is speaking and may be cut off; on_barge_in()
        stops the speech. Both only apply to the custom ONNX backend while a command runs."""
        self._handle_command = handle_command_fn
        self._processing = processing_event
        self._interruptible = interruptible
        self._on_barge_in = on_barge_in
        self._transcriber = transcriber
        self._menubar = menubar
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._backend_name = "none"
        self._last_window: Optional[np.ndarray] = None      # the window _verify checked (for save_clip)

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

        # A model built for other features would fail every inference (silently deaf).
        width = session.get_inputs()[0].shape[-1]
        if width != N_FEATURES:
            print(f"[Aria] Custom wake model expects {width} features, not {N_FEATURES} "
                  "(retrain: training/prepare_data.py + train_model.py) — falling back to openwakeword")
            return self._run_openwakeword()
        # A model trained without negatives scores everything (even silence) as the wake
        # word and fires every cooldown. Refuse it instead of waking up on a quiet room.
        if fires_on_silence(lambda f: session.run([label_name], {input_name: f})[0]):
            print("[Aria] Custom wake model fires on silence (retrain with negatives: "
                  "training/prepare_data.py + train_model.py) — falling back to openwakeword")
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
            barge_hits = 0
            chance = SecondChance()

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
                    if not self._can_barge_in():
                        barge_hits = 0
                        continue
                    try:
                        feat = wake_features(buffer.astype(np.float32) / 32768.0).reshape(1, -1)
                        score = float(session.run([label_name], {input_name: feat})[0][0][1])
                    except Exception as exc:
                        logger.debug("Custom model inference error: %s", exc)
                        continue
                    barge_hits = barge_hits + 1 if score >= _BARGE_THRESHOLD else 0
                    if barge_hits >= _BARGE_HITS:
                        print(f"[Aria] Barge-in: wake word while speaking (score={score:.2f}) — stopping")
                        wake_stats.log_event("barge_in", "custom_onnx", score)
                        self._barge_in()
                        barge_hits, last_triggered = 0, time.time()
                    continue

                # Extract MFCC features and run inference
                try:
                    feat = wake_features(buffer.astype(np.float32) / 32768.0).reshape(1, -1)
                    probs = session.run([label_name], {input_name: feat})[0]
                    score = float(probs[0][1])
                except Exception as exc:
                    logger.debug("Custom model inference error: %s", exc)
                    continue

                threshold = chance.threshold(now, _CUSTOM_THRESHOLD)
                if score >= threshold:
                    voiced = voiced_chunks(buffer)
                    if voiced is not None and voiced < _VAD_GATE_MIN:     # no voice in it: room sound
                        wake_stats.log_event("vad_gate", "custom_onnx", score, voiced=voiced)
                        save_clip(buffer, "vadgate", score)
                        last_triggered = time.time() - _COOLDOWN_SECS + 1.0
                        continue
                    heard, prefix = self._verify(stream, buffer, _CHUNK_SIZE)
                    save_clip(self._last_window if self._last_window is not None else buffer,
                              "wake" if heard else "reject", score)
                    if not heard:
                        print(f"[Aria] Wake model fired (score={score:.2f}) but Whisper didn't hear "
                              "\"Aria\" — ignored.")
                        wake_stats.log_event("verify_reject", "custom_onnx", score,
                                             heard=getattr(self, "_last_reject", None))
                        last_triggered = time.time() - _COOLDOWN_SECS + 1.0     # listen again in 1 s
                        chance.rejected(time.time(), lowered=threshold < _CUSTOM_THRESHOLD)
                        continue
                    print(f"[Aria] Wake word detected (custom model, score={score:.2f})")
                    wake_stats.log_event("detection" if threshold == _CUSTOM_THRESHOLD else "detection_second_chance",
                                         "custom_onnx", score)
                    chance.woke()
                    if self._can_barge_in():          # e.g. the welcome-back line: stop it, listen
                        self._barge_in()
                    self._on_wake(stream, _SAMPLE_RATE, _CHUNK_SIZE, prefix)
                    last_triggered = time.time()
                elif score >= _NEAR_MISS_FLOOR and now - last_near_miss > 3.0:
                    last_near_miss = now
                    voiced = voiced_chunks(buffer) if score >= _SECOND_CHANCE_FLOOR else None
                    if voiced is None or voiced >= _VAD_GATE_MIN:     # room sound can't open a second chance
                        chance.near_miss(now, score)
                    if voiced is not None and voiced >= _VAD_GATE_MIN:
                        save_clip(buffer, "nearmiss", score)          # maybe a missed "Aria"
                    wake_stats.log_event("near_miss", "custom_onnx", score, voiced=voiced)

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

    def _verify(self, stream, window: np.ndarray, chunk_size: int) -> tuple:
        """(heard "Aria", chunks read meanwhile). Whisper runs on the window in a thread while
        the mic keeps being read, so nothing said after the wake word is dropped."""
        self._last_window = None
        if self._transcriber is None or not _verify_enabled():
            return True, []
        # The score crosses the threshold before "Aria" has been said in full: checked right at
        # the crossing, Whisper heard the name in 13/37 of the user's clips (replayed over room
        # noise, 2026-10-25), checked 0.5 s later in 33/37. So listen on before checking.
        chunks: list = []
        for _ in range(_VERIFY_SETTLE_CHUNKS):
            try:
                chunks.append(stream.read(chunk_size, exception_on_overflow=False))
            except OSError:
                break
        if chunks:
            tail = np.frombuffer(b"".join(chunks), dtype=np.int16)
            window = np.concatenate([window, tail])[-len(window):]
        out: dict = {}
        self._last_window = window
        audio = window.astype(np.float32) / 32768.0

        def run() -> None:
            try:
                out["text"], out["no_speech"] = self._transcriber.check_wake(audio)
            except Exception as exc:
                logger.warning("wake verify failed: %s", exc)
        t = threading.Thread(target=run, daemon=True, name="aria-wake-verify")
        t.start()
        deadline = time.time() + _VERIFY_MAX_SECS
        while t.is_alive() and time.time() < deadline:
            try:
                chunks.append(stream.read(chunk_size, exception_on_overflow=False))
            except OSError:
                break
        t.join(timeout=0.1)
        if "text" not in out:                     # Whisper errored or was too slow: don't drop a real wake
            return True, chunks
        voiced = voiced_chunks(window)
        heard = wake_accepted(out["text"], out.get("no_speech", 1.0), voiced)
        if not heard:
            logger.info("wake verify: rejected (no_speech=%.2f, voiced=%s)", out.get("no_speech", 1.0), voiced)
            self._last_reject = (out["text"], out.get("no_speech", 1.0), voiced)
        return heard, chunks

    def _can_barge_in(self) -> bool:
        if self._interruptible is None or self._on_barge_in is None:
            return False
        try:
            return bool(self._interruptible())
        except Exception:
            return False

    def _barge_in(self) -> None:
        try:
            self._on_barge_in()
        except Exception as exc:
            logger.warning("barge-in stop failed: %s", exc)

    def _set_menubar(self, state: str) -> None:
        if self._menubar is None:
            return
        try:
            self._menubar.set_state(state)
        except Exception:
            pass

    def _on_wake(self, stream, sample_rate: int, chunk_size: int, prefix: Optional[list] = None) -> None:
        """Called immediately after any backend detects the wake word. `prefix` = audio already
        read from the stream (while the wake word was being verified)."""
        self._play_sound(_DING_SOUND)
        self._set_menubar("LISTENING")
        listening_indicator.show("Listening...")

        wav_path = self._record_until_silence(stream, sample_rate, chunk_size, prefix)
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
        self, stream, sample_rate: int, chunk_size: int, prefix: Optional[list] = None
    ) -> str | None:
        """VAD recording from the active mic stream (after any `prefix` chunks already read).
        Returns WAV path or None."""
        pending = list(prefix or [])
        chunks: list[bytes] = []
        detector = vad.detector()          # speech, not loudness, decides when you've stopped
        speech_started = False
        speech_duration = 0.0
        silence_duration = 0.0
        pre_speech_elapsed = 0.0
        chunk_secs = chunk_size / sample_rate
        total_elapsed = 0.0

        while total_elapsed < _VAD_MAX_SECS:
            try:
                chunk = pending.pop(0) if pending else stream.read(chunk_size, exception_on_overflow=False)
            except OSError:
                break

            audio_np = np.frombuffer(chunk, dtype=np.int16)
            rms = self._rms(audio_np)
            total_elapsed += chunk_secs
            speech = detector.is_speech(audio_np) if detector is not None else rms >= _VAD_SPEECH_RMS

            if speech:
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
