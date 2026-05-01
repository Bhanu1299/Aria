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
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from audiomentations import (
    Compose, AddGaussianNoise, TimeStretch, PitchShift,
)

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
]

SAY_VOICES_NEGATIVE = [
    "Alex", "Samantha", "Fred", "Victoria", "Tom",
    "Karen", "Daniel", "Moira",
]

_AUGMENT = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.7),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
])


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
            aug_data = _AUGMENT(samples=data, sample_rate=sr)
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
    n_clips: int = 200,
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

    count = len(existing)

    # 1. Speech negatives via macOS say
    for i, phrase in enumerate(NON_WAKE_PHRASES):
        for voice in SAY_VOICES_NEGATIVE[:3]:  # 3 voices × 26 phrases = 78 speech clips
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

    # 2. Noise negatives (gaussian + pink) to fill remaining quota
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
    print("[1/2] Augmenting positive clips...")
    n_aug = augment_positives()
    print(f"      {n_aug} augmented clips → {AUGMENTED_DIR}")

    print("[2/2] Generating negative samples...")
    n_neg = generate_negatives()
    print(f"      {n_neg} negative clips → {NEGATIVE_DIR}")

    print_summary()
