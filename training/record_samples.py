# training/record_samples.py
"""
Interactive microphone recording for positive "Aria" samples.

Usage:
    python record_samples.py [--duration 2.0] [--target 50]

Controls:
    Enter   — start recording
    k       — keep clip
    d       — discard and re-record
    done    — finish session
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from common import POSITIVE_DIR, SAMPLE_RATE, CLIP_DURATION, N_SAMPLES


def _save_clip(
    audio: np.ndarray,
    out_path: str,
    sample_rate: int = SAMPLE_RATE,
    target_samples: int = N_SAMPLES,
) -> None:
    """Write audio to WAV, padding or trimming to target_samples."""
    flat = audio.flatten().astype(np.int16)
    if len(flat) < target_samples:
        flat = np.pad(flat, (0, target_samples - len(flat)))
    else:
        flat = flat[:target_samples]
    sf.write(out_path, flat, sample_rate, subtype="PCM_16")


def record_loop(
    out_dir: str = str(POSITIVE_DIR),
    target: int = 50,
    duration: float = CLIP_DURATION,
) -> int:
    """
    Interactive recording loop.
    Returns number of clips kept.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    existing = len(list(out.glob("recorded_*.wav")))
    count = existing
    n_samples = int(SAMPLE_RATE * duration)

    print(f"\n[Aria Training] Recording mode — target: {target} clips")
    print(f"  Duration per clip: {duration}s | Press Enter to record, k=keep, d=discard, done=quit\n")

    while count < target:
        cmd = input(f"  [{count}/{target}] Press Enter to record (or 'done' to stop): ").strip().lower()

        if cmd == "done":
            break

        print("  Recording... ", end="", flush=True)
        audio = sd.rec(n_samples, samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        sd.wait()
        print("done.")

        # Playback
        sd.play(audio, SAMPLE_RATE)
        sd.wait()

        action = input("  Keep? [k/d]: ").strip().lower()
        if action == "d":
            print("  Discarded.\n")
            continue

        out_path = out / f"recorded_{count:04d}.wav"
        _save_clip(audio, str(out_path))
        count += 1
        print(f"  Saved → {out_path.name}\n")

    print(f"\n[Done] {count} clips in {out_dir}")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=CLIP_DURATION,
                        help="Recording duration in seconds (default: 2.0)")
    parser.add_argument("--target", type=int, default=50,
                        help="Target number of clips (default: 50)")
    args = parser.parse_args()
    record_loop(duration=args.duration, target=args.target)
