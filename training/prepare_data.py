"""
Augment positive clips and generate synthetic negative samples.

Negatives are generated locally (no internet required):
  - Synthetic speech clips via macOS say with non-wake phrases
  - Gaussian + pink noise clips

Usage:
    python prepare_data.py

Skips negative generation if data/negative/ already has enough clips.
Prints clip count summary at end.
"""
from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

from common import (
    POSITIVE_DIR, NEGATIVE_DIR, AUGMENTED_DIR,
    SAMPLE_RATE, N_SAMPLES, CLIP_DURATION,
)

# Phrases that are NOT the wake word — used for speech negatives
NON_WAKE_PHRASES = [
    "Open the browser", "What time is it", "Play some music",
    "Send a message", "Set a timer for ten minutes",
    "Turn off the lights", "Call mom", "Navigate to downtown",
    "What is the weather today", "Remind me at noon",
    "Add milk to the shopping list", "Read my emails",
    "How do I get to the airport", "Translate this to Spanish",
    "Take a photo", "Book a restaurant", "What is the news",
    "Check my calendar", "Start the car", "Lock the door",
    "Hey computer", "Hello there", "Excuse me",
    "Good morning", "Good night", "Thank you very much",
    # Aria's own commands (said after the wake word, or on a call nearby)
    "Submit it", "I submitted it", "Skip this job", "Close all",
    "Start applications", "Pause", "Resume", "Stop", "Applications are done",
    # sound-alikes
    "Maria", "Area", "Ariel", "Sorry", "Harry", "Carrie", "Arya", "Ottawa",
    "Are you there", "Where are we", "Really", "Hurry up",
    # everyday speech
    "I think we should meet tomorrow", "Can you hear me okay", "Let me share my screen",
    "That sounds good to me", "One second please", "Yeah I agree with that",
]

# Karen and Daniel are held out so a model can be checked on voices it never heard
SAY_VOICES_NEGATIVE = [
    "Alex", "Samantha", "Fred", "Victoria", "Tom", "Moira", "Albert",
    "Junior", "Kathy", "Ralph", "Rishi", "Tessa", "Flo", "Eddy",
]



def _augment(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Gaussian noise (p=0.7), time stretch 0.9–1.1 (p=0.5), pitch ±2 semitones (p=0.5)."""
    out = samples.astype(np.float32)
    if random.random() < 0.5:
        out = librosa.effects.time_stretch(out, rate=random.uniform(0.9, 1.1))
    if random.random() < 0.5:
        out = librosa.effects.pitch_shift(out, sr=sample_rate, n_steps=random.uniform(-2, 2))
    if random.random() < 0.7:
        out = out + np.random.randn(len(out)).astype(np.float32) * random.uniform(0.001, 0.015)
    return out


def augment_positives(
    positive_dir: str = str(POSITIVE_DIR),
    aug_dir: str = str(AUGMENTED_DIR),
    n_augmentations: int = 2,
) -> int:
    """
    Apply random augmentations to each positive clip.
    Returns total number of augmented clips created.
    """
    pos = Path(positive_dir)
    aug = Path(aug_dir)
    aug.mkdir(parents=True, exist_ok=True)

    clips = list(pos.glob("*.wav"))
    if not clips:
        print(f"[warn] No WAV files found in {positive_dir}")
        return 0

    count = 0
    for wav_path in clips:
        try:
            data, sr = sf.read(str(wav_path), dtype="float32")
        except Exception as exc:
            print(f"[warn] Could not read {wav_path.name}: {exc}")
            continue

        for i in range(n_augmentations):
            aug_data = _augment(data, sr)
            # Pad or trim
            target = int(sr * CLIP_DURATION)
            if len(aug_data) < target:
                aug_data = np.pad(aug_data, (0, target - len(aug_data)))
            else:
                aug_data = aug_data[:target]

            out_path = aug / f"aug_{wav_path.stem}_{i:02d}.wav"
            sf.write(str(out_path), aug_data, sr, subtype="PCM_16")
            count += 1

    return count


def generate_negatives(
    neg_dir: str = str(NEGATIVE_DIR),
    n_clips: int = 900,
) -> int:
    """
    Generate synthetic negative samples (speech + noise).
    Skips if neg_dir already has >= n_clips files.
    Returns number of clips in neg_dir after this call.
    """
    neg = Path(neg_dir)
    neg.mkdir(parents=True, exist_ok=True)

    existing = list(neg.glob("*.wav"))
    if len(existing) >= n_clips:
        print(f"  [skip] {neg_dir} already has {len(existing)} clips — skipping generation")
        return len(existing)

    # Clips the loops below don't make (e.g. downloaded); each loop counts its own files once.
    count = sum(1 for f in existing if not f.name.startswith(("speech_", "quiet_", "noise_")))

    # 1. Speech negatives via macOS say
    for i, phrase in enumerate(NON_WAKE_PHRASES):
        for voice in SAY_VOICES_NEGATIVE:
            out_path = neg / f"speech_{voice.lower()}_{i:03d}.wav"
            if out_path.exists():
                count += 1
                continue
            result = subprocess.run(
                ["say", "-v", voice, "--data-format=LEI16@16000",
                 "-o", str(out_path), phrase],
                capture_output=True,
            )
            if result.returncode == 0:
                _pad_wav(out_path)
                count += 1

    # 2. Quiet-room negatives: digital silence and faint noise (a mic with nobody talking)
    for i in range(60):
        out_path = neg / f"quiet_{i:04d}.wav"
        if out_path.exists():
            count += 1
            continue
        level = 0 if i % 6 == 0 else random.uniform(5, 300)
        quiet = np.clip(np.random.randn(N_SAMPLES) * level, -32000, 32000).astype(np.int16)
        sf.write(str(out_path), quiet, SAMPLE_RATE, subtype="PCM_16")
        count += 1

    # 3. Noise negatives (gaussian + pink) to fill remaining quota
    n_noise = max(0, n_clips - count)
    for i in range(n_noise):
        out_path = neg / f"noise_{i:04d}.wav"
        if out_path.exists():
            count += 1
            continue
        if i % 2 == 0:
            noise = (np.random.randn(N_SAMPLES) * 3000).astype(np.int16)
        else:
            white = np.random.randn(N_SAMPLES)
            pink = np.cumsum(white) * 1000
            noise = np.clip(pink, -32000, 32000).astype(np.int16)
        sf.write(str(out_path), noise, SAMPLE_RATE, subtype="PCM_16")
        count += 1

    return count


def _pad_wav(path: Path) -> None:
    """Pad/trim WAV to CLIP_DURATION."""
    try:
        data, sr = sf.read(str(path), dtype="int16")
        target = int(sr * CLIP_DURATION)
        if len(data) < target:
            data = np.pad(data, (0, target - len(data)))
        else:
            data = data[:target]
        sf.write(str(path), data, sr, subtype="PCM_16")
    except Exception:
        pass


def _backgrounds(neg_dir: Path) -> list:
    """Room recordings (negative/room_*.wav) plus faint synthetic noise — what the live
    mic hears around the wake word. Training on zero-padded clips alone made the model
    miss every real "Aria" (live windows are never digital silence)."""
    rooms = [sf.read(str(p), dtype="int16")[0].astype(np.float32) for p in sorted(neg_dir.glob("room_*.wav"))]
    synth = [np.random.randn(N_SAMPLES).astype(np.float32) * random.uniform(10, 200) for _ in range(20)]
    return rooms + synth


def _trim(clip: np.ndarray, floor: float = 300.0) -> np.ndarray:
    loud = np.nonzero(np.abs(clip) > floor)[0]
    return clip[max(0, loud[0] - 1600): loud[-1] + 1600] if len(loud) else clip


def place_on_backgrounds(
    src_globs: list,
    out_dir: str,
    prefix: str,
    per_clip: int = 2,
    neg_dir: str = str(NEGATIVE_DIR),
    skip_prefix: str = "",
) -> int:
    """Each clip, trimmed, at a random offset over a random background, gain 0.3–1.2.
    src_globs are paths like "data/positive/*.wav". Files starting with skip_prefix are
    left out (held-out evaluation clips)."""
    bgs = _backgrounds(Path(neg_dir))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob(f"{prefix}_*.wav"):
        old.unlink()
    count = 0
    for pattern in src_globs:
        for wav_path in sorted(Path(pattern).parent.glob(Path(pattern).name)):
            if skip_prefix and wav_path.name.startswith(skip_prefix):
                continue
            try:
                word = _trim(sf.read(str(wav_path), dtype="int16")[0].astype(np.float32))[:N_SAMPLES]
            except Exception as exc:
                print(f"[warn] Could not read {wav_path.name}: {exc}")
                continue
            for i in range(per_clip):
                bg = random.choice(bgs)
                start = random.randrange(0, max(1, len(bg) - N_SAMPLES + 1))
                mix = bg[start:start + N_SAMPLES].copy()
                mix = np.pad(mix, (0, N_SAMPLES - len(mix)))
                off = random.randrange(0, N_SAMPLES - len(word) + 1)
                mix[off:off + len(word)] += word * random.uniform(0.3, 1.2)
                sf.write(str(out / f"{prefix}_{wav_path.stem}_{i:02d}.wav"),
                         np.clip(mix, -32000, 32000).astype(np.int16), SAMPLE_RATE, subtype="PCM_16")
                count += 1
    return count


def print_summary(
    positive_dir: str = str(POSITIVE_DIR),
    negative_dir: str = str(NEGATIVE_DIR),
    augmented_dir: str = str(AUGMENTED_DIR),
) -> None:
    n_pos = len(list(Path(positive_dir).glob("*.wav")))
    n_neg = len(list(Path(negative_dir).glob("*.wav")))
    n_aug = len(list(Path(augmented_dir).glob("*.wav")))
    total_pos = n_pos + n_aug
    print("\n── Clip count summary ──────────────────────────────")
    print(f"  Positive (TTS + recorded): {n_pos:>5}")
    print(f"  Augmented positives:       {n_aug:>5}")
    print(f"  Total positive:            {total_pos:>5}  (target: ~1100)")
    print(f"  Negative samples:          {n_neg:>5}")
    print("────────────────────────────────────────────────────")
    if total_pos < 800:
        print(f"  [warn] Only {total_pos} positive clips — run generate_tts.py first for best results")


if __name__ == "__main__":
    print("[1/3] Augmenting positive clips...")
    n_aug = augment_positives()
    print(f"      {n_aug} augmented clips → {AUGMENTED_DIR}")

    print("[2/3] Generating negative samples...")
    n_neg = generate_negatives()
    print(f"      {n_neg} negative clips → {NEGATIVE_DIR}")

    print("[3/3] Placing clips over room noise (what the live mic hears)...")
    n_bg = place_on_backgrounds([str(POSITIVE_DIR / "*.wav")], str(AUGMENTED_DIR), "bg", per_clip=4)
    n_bgn = place_on_backgrounds([str(NEGATIVE_DIR / "speech_*.wav")], str(NEGATIVE_DIR), "bg", per_clip=1)
    print(f"      {n_bg} positive + {n_bgn} negative clips over room noise")

    print_summary()
