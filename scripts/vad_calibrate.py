"""scripts/vad_calibrate.py — measure your mic/voice levels against Aria's
end-of-speech detectors, to tune cut-offs from evidence instead of guesses.

Run:  venv/bin/python scripts/vad_calibrate.py
Stay quiet for the first 2 seconds, then read the prompt naturally —
including the pauses — the way you'd talk to Aria.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHUNK = 512                      # 32 ms — same as conversation.py
CHUNK_SECS = CHUNK / SAMPLE_RATE
RECORD_SECS = 14.0
QUIET_SECS = 2.0

# (name, speech RMS threshold, trailing-silence seconds) — current settings
DETECTORS = [
    ("wake/hotkey  (voice_capture.py)", 0.03 * 32767, 1.5),
    ("follow-up    (conversation.py) ", 400.0, 1.2),
]

PROMPT = (
    'Say: "Can you check out the, um... cheapest rooms near the airport\n'
    '      for the 19th, and, uh, keep it under a hundred and twenty dollars."'
)


def _record() -> np.ndarray:
    frames = int(RECORD_SECS * SAMPLE_RATE)
    print("Recording starts in 3...", flush=True)
    time.sleep(1); print("2...", flush=True)
    time.sleep(1); print("1...", flush=True)
    time.sleep(1)
    print(f"\n● RECORDING — stay QUIET for {QUIET_SECS:.0f}s, then:\n{PROMPT}\n", flush=True)
    audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    print("■ Done.\n")
    return audio.reshape(-1)


def _chunk_rms(audio: np.ndarray) -> np.ndarray:
    n = len(audio) // CHUNK
    chunks = audio[: n * CHUNK].astype(np.float32).reshape(n, CHUNK)
    return np.sqrt(np.mean(chunks ** 2, axis=1))


def _simulate(rms: np.ndarray, threshold: float, silence_end: float, speech_end_idx: int) -> str:
    """Replay a detector; report when it would stop relative to your real end of speech."""
    started = False
    silence = 0.0
    for i, r in enumerate(rms):
        if r >= threshold:
            started = True
            silence = 0.0
        elif started:
            silence += CHUNK_SECS
            if silence >= silence_end:
                stop_at = (i + 1) * CHUNK_SECS - silence  # last loud chunk
                lost = (speech_end_idx + 1) * CHUNK_SECS - stop_at
                if lost > 0.3:
                    return f"CUTS YOU OFF at {stop_at:.1f}s — loses ~{lost:.1f}s of speech"
                return f"ok — stops {silence_end:.1f}s after you finish"
    if not started:
        return "NEVER HEARS YOU — threshold above your voice level"
    return "ok — never stopped early"


def main() -> int:
    try:
        audio = _record()
    except Exception as exc:
        print(f"Mic error: {exc}")
        return 1

    rms = _chunk_rms(audio)
    quiet_n = int((QUIET_SECS - 0.3) / CHUNK_SECS)
    ambient = float(np.percentile(rms[:quiet_n], 90))
    # "speech" = clearly above room noise; robust to either detector's threshold
    speech_mask = rms > max(ambient * 3.0, 150.0)
    speech_mask[:quiet_n] = False
    if not speech_mask.any():
        print("Didn't detect any speech — check the mic input in System Settings → Sound.")
        return 1
    idx = np.flatnonzero(speech_mask)
    first, last = int(idx[0]), int(idx[-1])
    speech_rms = rms[first:last + 1][speech_mask[first:last + 1]]

    # pauses inside the utterance
    pauses, run = [], 0
    for m in speech_mask[first:last + 1]:
        if m:
            if run:
                pauses.append(run * CHUNK_SECS)
            run = 0
        else:
            run += 1

    print(f"Room noise (90th pct RMS):     {ambient:7.0f}")
    print(f"Your speech RMS  quiet(10%):   {np.percentile(speech_rms, 10):7.0f}")
    print(f"                 median:       {np.median(speech_rms):7.0f}")
    print(f"                 loud(90%):    {np.percentile(speech_rms, 90):7.0f}")
    print(f"Speech span:                   {first * CHUNK_SECS:.1f}s → {(last + 1) * CHUNK_SECS:.1f}s")
    print(f"Longest pause mid-sentence:    {max(pauses, default=0.0):.2f}s")
    print()
    for name, thr, sil in DETECTORS:
        print(f"{name} [rms≥{thr:.0f}, {sil}s]: {_simulate(rms, thr, sil, last)}")

    out = "/tmp/aria_vad_calibration.npy"
    np.save(out, audio)
    print(f"\nRaw audio saved to {out} for re-analysis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
