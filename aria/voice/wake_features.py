"""wake_features.py — the custom wake model's input, shared by live detection
(aria/voice/wake_word.py) and training (training/train_model.py) so they can't drift.

The loudest 1 s of the window → mean and std of 20 MFCCs + mean |delta| → 60 floats.
Averaging the whole 2 s window (the old feature) let room noise around a ~0.5 s
"Aria" drown it out: the user's own recordings scored 0/50 once placed in real room
noise; with this feature 48/50 held-out recordings trigger and the room never does.
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
N_MFCC = 20
HOP = 256
SEGMENT_SECS = 1.0
N_FEATURES = 3 * N_MFCC


def wake_features(audio, sr: int = SAMPLE_RATE) -> np.ndarray:
    """audio: float32 in [-1, 1], about 2 s. Returns shape (N_FEATURES,) float32."""
    import librosa
    y = np.asarray(audio, dtype=np.float32)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP)
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0][: mfcc.shape[1]]
    span = max(1, min(mfcc.shape[1], int(SEGMENT_SECS * sr / HOP)))
    energy = np.convolve(rms, np.ones(span), mode="valid")
    start = int(np.argmax(energy)) if len(energy) else 0
    seg = mfcc[:, start:start + span]
    if seg.shape[1] >= 3:
        width = min(9, seg.shape[1] if seg.shape[1] % 2 else seg.shape[1] - 1)
        delta = librosa.feature.delta(seg, width=width)
    else:
        delta = np.zeros_like(seg)
    return np.concatenate([seg.mean(axis=1), seg.std(axis=1), np.abs(delta).mean(axis=1)]).astype(np.float32)
