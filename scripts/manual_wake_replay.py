"""scripts/manual_wake_replay.py — replay the live wake pipeline over recorded audio.

    venv/bin/python scripts/manual_wake_replay.py [--positives N] [--minutes M]

Positives: the user's own "Aria" clips (training/data/positive/recorded_*.wav), each placed
2 s into 6 s of room noise. Negatives: room / noise / quiet clips strung together. Every
32 ms chunk goes through the real ONNX model (aria.onnx), Silero VAD and, at a threshold
crossing, the real Whisper check (check_wake on the window + 0.5 s), exactly as the
listener does. Prints wake rate on positives and false wakes per minute on negatives for
the old policy and the VAD-gated one. Needs the mic-free models only (no audio device).
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aria.voice import wake_word                      # noqa: E402
from aria.voice.wake_features import wake_features         # noqa: E402

SR, CHUNK = 16000, 512
WIN = int(SR * wake_word._CUSTOM_WINDOW_SECS)


def read(path: str) -> np.ndarray:
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def run(stream: np.ndarray, session, inp, lab, transcriber, policy: str) -> list:
    """Accepted wake times (s) over one stream under `policy` ('old' | 'vad')."""
    buf = np.zeros(WIN, dtype=np.int16)
    out, cooldown_until, n = [], 0.0, len(stream) // CHUNK
    i = 0
    while i < n:
        chunk = stream[i * CHUNK:(i + 1) * CHUNK]
        buf = np.concatenate([buf[CHUNK:], chunk])
        t = (i + 1) * CHUNK / SR
        i += 1
        if t < cooldown_until:
            continue
        score = float(session.run([lab], {inp: wake_features(buf.astype(np.float32) / 32768.0).reshape(1, -1)})[0][0][1])
        if score < wake_word._CUSTOM_THRESHOLD:
            continue
        if policy == "vad" and (wake_word.voiced_chunks(buf) or 0) < wake_word._VAD_GATE_MIN:
            cooldown_until = t + 1.0
            continue
        settle = stream[i * CHUNK:(i + wake_word._VERIFY_SETTLE_CHUNKS) * CHUNK]
        i += wake_word._VERIFY_SETTLE_CHUNKS
        window = np.concatenate([buf, settle])[-WIN:]
        text, ns = transcriber.check_wake(window.astype(np.float32) / 32768.0)
        ok = wake_word.wake_accepted(text, ns, wake_word.voiced_chunks(window) if policy == "vad" else None)
        if ok:
            out.append(t)
            cooldown_until = i * CHUNK / SR + wake_word._COOLDOWN_SECS
        else:
            cooldown_until = i * CHUNK / SR + 1.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positives", type=int, default=60)
    ap.add_argument("--minutes", type=float, default=4.0)
    args = ap.parse_args()
    import onnxruntime as ort
    from aria.voice.transcriber import Transcriber
    session = ort.InferenceSession(wake_word._CUSTOM_MODEL_PATH)
    inp, lab = session.get_inputs()[0].name, session.get_outputs()[1].name
    tr = Transcriber()
    root = os.path.join(os.path.dirname(__file__), "..", "training", "data")
    pos = sorted(glob.glob(f"{root}/positive/recorded_*.wav"))
    room = [read(p) for p in sorted(glob.glob(f"{root}/negative/room_*.wav"))]
    neg = sorted(glob.glob(f"{root}/negative/room_*.wav") + glob.glob(f"{root}/negative/noise_*.wav")
                 + glob.glob(f"{root}/negative/quiet_*.wav"))
    rng = random.Random(7)
    picks = rng.sample(pos, min(args.positives, len(pos)))
    neg_stream = np.concatenate([read(p) for p in rng.sample(neg, len(neg))])[: int(args.minutes * 60 * SR)]
    for policy in ("old", "vad"):
        woke = 0
        for p in picks:
            bed = np.concatenate([room[rng.randrange(len(room))] for _ in range(3)]).astype(np.int32)
            clip = read(p).astype(np.int32)
            bed[2 * SR:2 * SR + len(clip)] += clip
            woke += bool(run(np.clip(bed, -32768, 32767).astype(np.int16), session, inp, lab, tr, policy))
        false = len(run(neg_stream, session, inp, lab, tr, policy))
        print(f"{policy:4s}  your 'Aria' clips: {woke}/{len(picks)} woke   "
              f"room/noise/quiet: {false} false wakes in {len(neg_stream) / SR / 60:.1f} min")
    tr.stop()


if __name__ == "__main__":
    main()
