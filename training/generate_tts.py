# training/generate_tts.py
"""
Generate synthetic "Aria" wake word clips using macOS say + Edge TTS.

Usage:
    python generate_tts.py

Output: data/positive/ — 16kHz mono WAV files
"""
import asyncio
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf

from common import POSITIVE_DIR, SAMPLE_RATE, CLIP_DURATION, WAKE_PHRASES

# ── macOS say voices to use (run `say -v ?` to list all) ──────────────────────
SAY_VOICES = [
    "Alex", "Allison", "Ava", "Daniel", "Fred",
    "Karen", "Moira", "Samantha", "Serena", "Susan",
    "Tom", "Veena", "Victoria", "Yuri", "Fiona",
    "Kate", "Tessa", "Rishi", "Nora", "Kyoko",
]

# ── Edge TTS voices (diverse accents) ────────────────────────────────────────
EDGE_VOICES = [
    "en-US-JennyNeural", "en-US-GuyNeural", "en-US-AriaNeural",
    "en-US-DavisNeural", "en-US-AmberNeural", "en-US-AnaNeural",
    "en-GB-SoniaNeural", "en-GB-RyanNeural", "en-GB-LibbyNeural",
    "en-AU-NatashaNeural", "en-AU-WilliamNeural",
    "en-CA-ClaraNeural", "en-CA-LiamNeural",
    "en-IN-NeerjaNeural", "en-IN-PrabhatNeural",
    "en-IE-EmilyNeural", "en-NZ-MitchellNeural",
    "en-SG-LunaNeural", "en-ZA-LeahNeural",
    "en-US-BrandonNeural", "en-US-ChristopherNeural",
    "en-US-EricNeural", "en-US-JacobNeural",
    "en-US-JennyMultilingualNeural", "en-US-MichelleNeural",
    "en-US-MonicaNeural", "en-US-RogerNeural",
    "en-US-SteffanNeural", "en-GB-MaisieNeural",
    "en-AU-AnnetteNeural",
]


def generate_say_clips(
    out_dir: str = str(POSITIVE_DIR),
    voices: List[str] = SAY_VOICES,
    phrases: List[str] = WAKE_PHRASES,
) -> int:
    """Generate clips using macOS say. Returns number of clips created."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    count = 0

    for voice in voices:
        for phrase in phrases:
            slug = phrase.lower().replace(" ", "_").replace("!", "")
            out_path = out / f"say_{voice.lower()}_{slug}_{count:04d}.wav"
            if out_path.exists():
                count += 1
                continue

            result = subprocess.run(
                [
                    "say",
                    "-v", voice,
                    "--data-format=LEI16@16000",
                    "-o", str(out_path),
                    phrase,
                ],
                capture_output=True,
            )
            if result.returncode != 0:
                print(f"  [warn] say failed for voice={voice}: {result.stderr.decode()}")
                continue

            # Pad or trim to CLIP_DURATION
            _normalise_wav(out_path)
            count += 1
            print(f"  [say] {out_path.name}")

    return count


async def _edge_clip(voice: str, phrase: str, out_path: Path) -> bool:
    """Generate one Edge TTS clip. Returns True on success."""
    try:
        import edge_tts
        tmp = out_path.with_suffix(".mp3")
        communicate = edge_tts.Communicate(phrase, voice)
        await communicate.save(str(tmp))

        # Convert MP3 → 16kHz mono WAV using ffmpeg (always available on macOS)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp),
             "-ar", "16000", "-ac", "1", str(out_path)],
            capture_output=True,
            check=True,
        )
        tmp.unlink(missing_ok=True)
        _normalise_wav(out_path)
        return True
    except Exception as exc:
        print(f"  [warn] edge_tts failed voice={voice}: {exc}")
        return False


async def generate_edge_clips(
    out_dir: str = str(POSITIVE_DIR),
    voices: List[str] = EDGE_VOICES,
    phrases: List[str] = WAKE_PHRASES,
) -> int:
    """Generate clips using Edge TTS (async). Returns number of clips created."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    count = 0

    tasks = []
    paths = []
    for i, voice in enumerate(voices):
        for phrase in phrases:
            slug = phrase.lower().replace(" ", "_").replace("!", "")
            out_path = out / f"edge_{voice.replace('-', '_').lower()}_{slug}_{count:04d}.wav"
            if out_path.exists():
                count += 1
                continue
            tasks.append(_edge_clip(voice, phrase, out_path))
            paths.append(out_path)
            count += 1

    results = await asyncio.gather(*tasks, return_exceptions=True)
    ok = sum(1 for r in results if r is True)
    print(f"  [edge_tts] {ok}/{len(tasks)} clips generated")
    return ok


def _normalise_wav(path: Path) -> None:
    """Pad short clips with silence or trim long clips to CLIP_DURATION."""
    try:
        data, sr = sf.read(str(path), dtype="int16")
        target = int(sr * CLIP_DURATION)
        if len(data) < target:
            data = np.pad(data, (0, target - len(data)))
        else:
            data = data[:target]
        sf.write(str(path), data, sr, subtype="PCM_16")
    except Exception as exc:
        print(f"  [warn] normalise failed for {path.name}: {exc}")


if __name__ == "__main__":
    print("[1/2] Generating macOS say clips...")
    n_say = generate_say_clips()
    print(f"      {n_say} clips written to {POSITIVE_DIR}")

    print("[2/2] Generating Edge TTS clips...")
    n_edge = asyncio.run(generate_edge_clips())
    print(f"      {n_edge} clips written to {POSITIVE_DIR}")

    total = len(list(POSITIVE_DIR.glob("*.wav")))
    print(f"\nDone. Total positive clips: {total}")
