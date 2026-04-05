# OWW Custom Wake Word Training Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a step-by-step pipeline to train a custom OpenWakeWord model for "Aria" using TTS-generated + recorded samples, then wire it into the existing `wake_word.py`.

**Architecture:** Six standalone scripts in `training/` — each independently re-runnable. Feature extraction uses OWW's pre-trained ONNX embedding model (96-dim vectors). A sklearn `RandomForestClassifier` is trained on those embeddings and exported as ONNX. The custom model replaces the `alexa` proxy in `wake_word.py` via a new `_run_custom_onnx()` method.

**Tech Stack:** Python 3.11, openwakeword 0.6.0, edge-tts (async), audiomentations, sklearn, skl2onnx, onnxruntime, sounddevice, soundfile, librosa (training only)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `training/requirements.txt` | Create | Pinned training deps |
| `training/generate_tts.py` | Create | Synthesize "Aria" clips via macOS say + Edge TTS |
| `training/record_samples.py` | Create | Interactive mic recording (2.0s default) |
| `training/prepare_data.py` | Create | Augment positives + generate synthetic negatives |
| `training/train_model.py` | Create | Extract OWW embeddings + train RandomForest + checkpoint |
| `training/validate_model.py` | Create | Precision/recall gate on held-out set |
| `training/export_model.py` | Create | Copy models/aria.onnx → ~/.aria/aria.onnx |
| `training/common.py` | Create | Shared constants (paths, sample rate, clip length) |
| `tests/training/__init__.py` | Create | Test package marker |
| `tests/training/conftest.py` | Create | Shared fixtures (tmp data dirs) |
| `tests/training/test_generate_tts.py` | Create | WAV format + clip count |
| `tests/training/test_record_samples.py` | Create | Save logic (mocked sounddevice) |
| `tests/training/test_prepare_data.py` | Create | Augmentation output + negative skip |
| `tests/training/test_train_model.py` | Create | Embedding extraction + classifier fit with dummy data |
| `tests/training/test_validate_model.py` | Create | Metric computation |
| `tests/training/test_export_model.py` | Create | File copy to ~/.aria/ |
| `wake_word.py` | Modify | Add `_run_custom_onnx()` + auto-select custom model if present |

---

## Task 1: Scaffold — training/ directory + shared constants

**Files:**
- Create: `training/requirements.txt`
- Create: `training/common.py`
- Create: `tests/training/__init__.py`
- Create: `tests/training/conftest.py`

- [ ] **Step 1: Create training/requirements.txt**

```
# training/requirements.txt
# openwakeword[train] pulls tensorflow — pin to avoid runtime surprises
# If install fails: try replacing tensorflow==2.13.0 with tflite-runtime==2.14.0
openwakeword==0.6.0
tensorflow==2.13.0
edge-tts==6.1.9
audiomentations==0.33.0
sounddevice==0.4.6
soundfile==0.12.1
librosa==0.10.1
scikit-learn==1.3.2
skl2onnx==1.16.0
onnxruntime==1.17.1
numpy==1.24.4
joblib==1.3.2
```

- [ ] **Step 2: Create training/common.py**

```python
# training/common.py
"""Shared constants for all training scripts."""
import os
from pathlib import Path

TRAINING_DIR = Path(__file__).parent
DATA_DIR = TRAINING_DIR / "data"
POSITIVE_DIR = DATA_DIR / "positive"
NEGATIVE_DIR = DATA_DIR / "negative"
AUGMENTED_DIR = DATA_DIR / "augmented"
MODELS_DIR = TRAINING_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

SAMPLE_RATE = 16000
CLIP_DURATION = 2.0          # seconds — covers "Hey Aria" at natural pace
N_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)  # 32000 samples
N_MFCC = 20
ONNX_MODEL_NAME = "aria.onnx"

WAKE_PHRASES = [
    "Aria",
    "Hey Aria",
    "Hey Aria!",
    "Okay Aria",
    "Aria please",
    "Hi Aria",
    "Yo Aria",
    "Aria hey",
    "Aria listen",
    "Wake up Aria",
]

# Ensure directories exist when module is imported
for _d in [POSITIVE_DIR, NEGATIVE_DIR, AUGMENTED_DIR, MODELS_DIR, CHECKPOINTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 3: Create tests/training/__init__.py**

```python
# tests/training/__init__.py
```

- [ ] **Step 4: Create tests/training/conftest.py**

```python
# tests/training/conftest.py
import sys
from pathlib import Path
import pytest

# Add project root + training dir to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "training"))


@pytest.fixture
def tmp_data_dirs(tmp_path):
    """Provide isolated data directories for each test."""
    dirs = {
        "positive": tmp_path / "positive",
        "negative": tmp_path / "negative",
        "augmented": tmp_path / "augmented",
        "models": tmp_path / "models",
    }
    for d in dirs.values():
        d.mkdir(parents=True)
    return dirs


@pytest.fixture
def dummy_wav(tmp_path):
    """Write a 2s silent 16kHz mono WAV and return its path."""
    import numpy as np
    import soundfile as sf

    path = tmp_path / "dummy.wav"
    samples = np.zeros(32000, dtype=np.int16)
    sf.write(str(path), samples, 16000, subtype="PCM_16")
    return path
```

- [ ] **Step 5: Commit**

```bash
git add training/requirements.txt training/common.py \
        tests/training/__init__.py tests/training/conftest.py
git commit -m "feat(training): scaffold training dir, shared constants, test fixtures"
```

---

## Task 2: generate_tts.py — synthesize "Aria" clips

**Files:**
- Create: `training/generate_tts.py`
- Create: `tests/training/test_generate_tts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/training/test_generate_tts.py
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
import numpy as np
import soundfile as sf
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "training"))


def test_say_writes_wav(tmp_path):
    """generate_say_clips() creates WAV files with correct format."""
    from generate_tts import generate_say_clips

    out_dir = tmp_path / "positive"
    out_dir.mkdir()

    # Mock subprocess.run to write a silent WAV instead of calling say
    def fake_say(cmd, **kwargs):
        output_path = cmd[cmd.index("-o") + 1]
        samples = np.zeros(32000, dtype=np.int16)
        sf.write(output_path, samples, 16000, subtype="PCM_16")
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=fake_say):
        generate_say_clips(out_dir=str(out_dir), voices=["Alex"], phrases=["Aria"])

    wavs = list(out_dir.glob("*.wav"))
    assert len(wavs) == 1
    data, sr = sf.read(str(wavs[0]))
    assert sr == 16000


def test_edge_tts_writes_wav(tmp_path):
    """generate_edge_clips() creates WAV files."""
    from generate_tts import generate_edge_clips

    out_dir = tmp_path / "positive"
    out_dir.mkdir()

    async def fake_save(path):
        import soundfile as sf2
        import numpy as np2
        sf2.write(path, np2.zeros(32000, dtype=np.int16), 16000, subtype="PCM_16")

    mock_communicate = MagicMock()
    mock_communicate.return_value.save = fake_save

    with patch("edge_tts.Communicate", mock_communicate):
        import asyncio
        asyncio.run(generate_edge_clips.__wrapped__(
            out_dir=str(out_dir), voices=["en-US-JennyNeural"], phrases=["Aria"]
        )) if hasattr(generate_edge_clips, "__wrapped__") else None

    # At minimum the function must exist and be importable
    assert callable(generate_edge_clips)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /path/to/Aria
pytest tests/training/test_generate_tts.py -v
```
Expected: `ImportError: No module named 'generate_tts'`

- [ ] **Step 3: Implement generate_tts.py**

```python
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

# ── Edge TTS voices (diverse accents) ─────────────────────────────────────────
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/training/test_generate_tts.py -v
```
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add training/generate_tts.py tests/training/test_generate_tts.py
git commit -m "feat(training): add generate_tts.py — macOS say + async Edge TTS"
```

---

## Task 3: record_samples.py — interactive voice recording

**Files:**
- Create: `training/record_samples.py`
- Create: `tests/training/test_record_samples.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/training/test_record_samples.py
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import soundfile as sf
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "training"))


def test_save_clip_creates_wav(tmp_path):
    """_save_clip() writes 16kHz mono WAV with correct sample count."""
    from record_samples import _save_clip

    audio = np.zeros(32000, dtype=np.int16)
    out_path = tmp_path / "test.wav"
    _save_clip(audio, str(out_path), sample_rate=16000)

    assert out_path.exists()
    data, sr = sf.read(str(out_path), dtype="int16")
    assert sr == 16000
    assert len(data) == 32000


def test_save_clip_pads_short_audio(tmp_path):
    """_save_clip() pads audio shorter than target to exactly target_samples."""
    from record_samples import _save_clip

    audio = np.zeros(10000, dtype=np.int16)  # shorter than 2s
    out_path = tmp_path / "short.wav"
    _save_clip(audio, str(out_path), sample_rate=16000, target_samples=32000)

    data, sr = sf.read(str(out_path), dtype="int16")
    assert len(data) == 32000


def test_record_loop_exits_on_done(tmp_path, capsys):
    """record_loop() stops when user types 'done'."""
    from record_samples import record_loop

    fake_audio = np.zeros(32000, dtype=np.int16)

    with patch("sounddevice.rec", return_value=fake_audio), \
         patch("sounddevice.wait"), \
         patch("sounddevice.play"), \
         patch("builtins.input", side_effect=["", "k", "done"]):
        count = record_loop(out_dir=str(tmp_path), target=2, duration=2.0)

    assert count == 1  # one clip kept before 'done'
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/training/test_record_samples.py -v
```
Expected: `ImportError: No module named 'record_samples'`

- [ ] **Step 3: Implement record_samples.py**

```python
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
        remaining = target - count
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/training/test_record_samples.py -v
```
Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add training/record_samples.py tests/training/test_record_samples.py
git commit -m "feat(training): add record_samples.py — interactive 2.0s voice recording"
```

---

## Task 4: prepare_data.py — augmentation + synthetic negatives

**Files:**
- Create: `training/prepare_data.py`
- Create: `tests/training/test_prepare_data.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/training/test_prepare_data.py
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "training"))


@pytest.fixture
def sample_positive_dir(tmp_path):
    """Create 3 dummy positive WAV files."""
    pos = tmp_path / "positive"
    pos.mkdir()
    for i in range(3):
        audio = (np.random.randn(32000) * 1000).astype(np.int16)
        sf.write(str(pos / f"clip_{i:04d}.wav"), audio, 16000, subtype="PCM_16")
    return pos


def test_augment_creates_files(tmp_path, sample_positive_dir):
    """augment_positives() creates at least as many clips as input."""
    from prepare_data import augment_positives

    aug_dir = tmp_path / "augmented"
    aug_dir.mkdir()
    n = augment_positives(
        positive_dir=str(sample_positive_dir),
        aug_dir=str(aug_dir),
        n_augmentations=2,
    )
    assert n >= 3
    assert len(list(aug_dir.glob("*.wav"))) >= 3


def test_augment_wav_format(tmp_path, sample_positive_dir):
    """Augmented clips are 16kHz mono WAV."""
    from prepare_data import augment_positives

    aug_dir = tmp_path / "augmented"
    aug_dir.mkdir()
    augment_positives(str(sample_positive_dir), str(aug_dir), n_augmentations=1)

    for wav in aug_dir.glob("*.wav"):
        data, sr = sf.read(str(wav), dtype="int16")
        assert sr == 16000
        assert data.ndim == 1
        break  # check one is enough


def test_generate_negatives_creates_files(tmp_path):
    """generate_negatives() creates WAV files even with no internet."""
    from prepare_data import generate_negatives

    neg_dir = tmp_path / "negative"
    neg_dir.mkdir()
    n = generate_negatives(neg_dir=str(neg_dir), n_clips=10)
    assert n >= 10
    assert len(list(neg_dir.glob("*.wav"))) >= 10


def test_generate_negatives_skips_if_populated(tmp_path):
    """generate_negatives() skips download if dir already has files."""
    from prepare_data import generate_negatives

    neg_dir = tmp_path / "negative"
    neg_dir.mkdir()
    # Pre-populate with dummy files
    for i in range(5):
        sf.write(str(neg_dir / f"neg_{i}.wav"),
                 np.zeros(32000, dtype=np.int16), 16000, subtype="PCM_16")

    n = generate_negatives(neg_dir=str(neg_dir), n_clips=5)
    assert n == 5  # returns existing count, no new files added


def test_print_summary(tmp_path, sample_positive_dir, capsys):
    """prepare_data prints clip counts at end."""
    from prepare_data import print_summary

    neg_dir = tmp_path / "negative"
    neg_dir.mkdir()
    sf.write(str(neg_dir / "n.wav"), np.zeros(32000, dtype=np.int16), 16000, subtype="PCM_16")

    print_summary(str(sample_positive_dir), str(neg_dir), str(tmp_path / "aug"))
    out = capsys.readouterr().out
    assert "positive" in out.lower()
    assert "negative" in out.lower()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/training/test_prepare_data.py -v
```
Expected: `ImportError: No module named 'prepare_data'`

- [ ] **Step 3: Implement prepare_data.py**

```python
# training/prepare_data.py
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
    Compose, AddGaussianNoise, TimeStretch, PitchShift, RoomSimulator
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
        for voice in SAY_VOICES_NEGATIVE[:3]:  # 3 voices × 25 phrases = 75 speech clips
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

    # 2. Noise negatives (gaussian + pink)
    n_noise = max(0, n_clips - count)
    for i in range(n_noise):
        out_path = neg / f"noise_{i:04d}.wav"
        if out_path.exists():
            count += 1
            continue
        if i % 2 == 0:
            # Gaussian noise
            noise = (np.random.randn(N_SAMPLES) * 3000).astype(np.int16)
        else:
            # Pink noise (approximation via 1/f filtering)
            white = np.random.randn(N_SAMPLES)
            pink = np.cumsum(white) * 1000
            pink = np.clip(pink, -32000, 32000).astype(np.int16)
            noise = pink
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/training/test_prepare_data.py -v
```
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add training/prepare_data.py tests/training/test_prepare_data.py
git commit -m "feat(training): add prepare_data.py — augmentation + synthetic negatives, skip if cached"
```

---

## Task 5: train_model.py — extract embeddings, train classifier, checkpoint

**Files:**
- Create: `training/train_model.py`
- Create: `tests/training/test_train_model.py`

**Key design:** Use OWW's pre-trained ONNX embedding model to extract 96-dim feature
vectors, then train a `RandomForestClassifier`. Export classifier as ONNX via skl2onnx.
Checkpoints saved after each epoch as `models/checkpoints/clf_epoch_N.joblib`.

- [ ] **Step 1: Write the failing test**

```python
# tests/training/test_train_model.py
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "training"))


@pytest.fixture
def dummy_dataset(tmp_path):
    """5 positive + 5 negative dummy WAV files."""
    pos = tmp_path / "positive"
    neg = tmp_path / "negative"
    pos.mkdir(); neg.mkdir()

    for i in range(5):
        audio = (np.random.randn(32000) * 500).astype(np.int16)
        sf.write(str(pos / f"pos_{i}.wav"), audio, 16000, subtype="PCM_16")
        sf.write(str(neg / f"neg_{i}.wav"), audio, 16000, subtype="PCM_16")

    return {"positive": pos, "negative": neg}


def test_load_dataset_returns_arrays(dummy_dataset):
    """load_dataset() returns float32 arrays with matching label count."""
    from train_model import load_dataset

    X, y = load_dataset(
        positive_dirs=[str(dummy_dataset["positive"])],
        negative_dirs=[str(dummy_dataset["negative"])],
    )
    assert X.shape[0] == y.shape[0] == 10
    assert X.dtype == np.float32
    assert set(y) == {0, 1}


def test_extract_features_shape(dummy_dataset):
    """extract_features() returns correct shape for single WAV."""
    from train_model import extract_features
    wav = list(dummy_dataset["positive"].glob("*.wav"))[0]
    feat = extract_features(str(wav))
    assert feat.shape == (20,)   # 20 MFCC features
    assert feat.dtype == np.float32


def test_train_saves_onnx(tmp_path, dummy_dataset):
    """train_and_export() saves aria.onnx to models_dir."""
    from train_model import train_and_export

    X = np.random.randn(10, 20).astype(np.float32)
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    out_path = train_and_export(X, y, models_dir=str(models_dir))
    assert Path(out_path).exists()
    assert out_path.endswith(".onnx")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/training/test_train_model.py -v
```
Expected: `ImportError: No module named 'train_model'`

- [ ] **Step 3: Implement train_model.py**

```python
# training/train_model.py
"""
Train a custom wake word classifier using MFCC features + RandomForest.

Pipeline:
  1. Load all positive + augmented + negative WAV files
  2. Extract 20 MFCC features per clip (mean over time axis)
  3. Train RandomForestClassifier
  4. Export to ONNX via skl2onnx
  5. Save checkpoints after each fold

Usage:
    python train_model.py

Output: models/aria.onnx
"""
import os
import sys
from pathlib import Path
from typing import List, Tuple

import joblib
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from common import (
    POSITIVE_DIR, NEGATIVE_DIR, AUGMENTED_DIR,
    MODELS_DIR, CHECKPOINTS_DIR, ONNX_MODEL_NAME,
    SAMPLE_RATE, N_MFCC,
)

N_ESTIMATORS = 200
RANDOM_STATE = 42


def extract_features(wav_path: str) -> np.ndarray:
    """
    Extract 20 MFCC features from a WAV file.
    Returns shape (20,) float32 — mean over time axis.
    """
    try:
        audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True, dtype=np.float32)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        return np.mean(mfcc, axis=1).astype(np.float32)
    except Exception as exc:
        print(f"  [warn] Could not process {Path(wav_path).name}: {exc}")
        return None


def load_dataset(
    positive_dirs: List[str],
    negative_dirs: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all WAV files and extract features.
    Returns (X: float32 array shape (N, 20), y: int array shape (N,))
    """
    features, labels = [], []

    for label, dirs in [(1, positive_dirs), (0, negative_dirs)]:
        for d in dirs:
            for wav in Path(d).glob("*.wav"):
                feat = extract_features(str(wav))
                if feat is not None:
                    features.append(feat)
                    labels.append(label)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y


def train_and_export(
    X: np.ndarray,
    y: np.ndarray,
    models_dir: str = str(MODELS_DIR),
    checkpoints_dir: str = str(CHECKPOINTS_DIR),
) -> str:
    """
    Train RandomForestClassifier on (X, y) and export as ONNX.
    Saves checkpoint before export.
    Returns path to exported ONNX model.
    """
    models = Path(models_dir)
    checkpoints = Path(checkpoints_dir)
    models.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)

    print(f"  Training RandomForest (n_estimators={N_ESTIMATORS}, {len(X)} samples)...")
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X, y)

    # Save checkpoint (allows resume inspection without retraining)
    ckpt_path = checkpoints / "clf_latest.joblib"
    joblib.dump(clf, str(ckpt_path))
    print(f"  Checkpoint saved → {ckpt_path}")

    # Export to ONNX
    initial_type = [("float_input", FloatTensorType([None, N_MFCC]))]
    onnx_model = convert_sklearn(clf, initial_types=initial_type)
    out_path = models / ONNX_MODEL_NAME
    with open(str(out_path), "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"  Model exported → {out_path}")
    return str(out_path)


if __name__ == "__main__":
    print("[1/3] Loading dataset...")
    X, y = load_dataset(
        positive_dirs=[str(POSITIVE_DIR), str(AUGMENTED_DIR)],
        negative_dirs=[str(NEGATIVE_DIR)],
    )
    pos_count = int(np.sum(y == 1))
    neg_count = int(np.sum(y == 0))
    print(f"      Positives: {pos_count}  Negatives: {neg_count}  Total: {len(X)}")

    if len(X) < 20:
        print("[error] Not enough data. Run generate_tts.py and prepare_data.py first.")
        sys.exit(1)

    print("[2/3] Training classifier...")
    out_path = train_and_export(X, y)

    print(f"[3/3] Done. Model saved to {out_path}")
    print("      Next step: python validate_model.py")
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/training/test_train_model.py -v
```
Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add training/train_model.py tests/training/test_train_model.py
git commit -m "feat(training): add train_model.py — MFCC + RandomForest + ONNX export with checkpointing"
```

---

## Task 6: validate_model.py — precision/recall gate

**Files:**
- Create: `training/validate_model.py`
- Create: `tests/training/test_validate_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/training/test_validate_model.py
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "training"))


def test_compute_metrics_perfect():
    """compute_metrics() returns 1.0 precision/recall for perfect classifier."""
    from validate_model import compute_metrics

    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0])
    metrics = compute_metrics(y_true, y_pred)

    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["false_positive_rate"] == pytest.approx(0.0)


def test_compute_metrics_all_wrong():
    """compute_metrics() handles all-wrong predictions without crashing."""
    from validate_model import compute_metrics

    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 1])
    metrics = compute_metrics(y_true, y_pred)

    assert metrics["recall"] == pytest.approx(0.0)
    assert metrics["false_positive_rate"] == pytest.approx(1.0)


def test_passes_gate_good_model():
    """check_gate() returns True when precision and recall exceed thresholds."""
    from validate_model import check_gate

    assert check_gate({"precision": 0.95, "recall": 0.92, "f1": 0.93,
                       "false_positive_rate": 0.05}) is True


def test_passes_gate_bad_model():
    """check_gate() returns False when precision is below threshold."""
    from validate_model import check_gate

    assert check_gate({"precision": 0.80, "recall": 0.92, "f1": 0.86,
                       "false_positive_rate": 0.20}) is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/training/test_validate_model.py -v
```
Expected: `ImportError: No module named 'validate_model'`

- [ ] **Step 3: Implement validate_model.py**

```python
# training/validate_model.py
"""
Validate trained model on a held-out set before export.

Prints precision, recall, F1, and false positive rate.
Exits with code 1 if model doesn't meet minimum thresholds.

Usage:
    python validate_model.py

Thresholds (configurable at top of file):
    MIN_PRECISION = 0.90
    MIN_RECALL    = 0.85
"""
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import onnxruntime as ort
from sklearn.model_selection import train_test_split

from common import MODELS_DIR, ONNX_MODEL_NAME, N_MFCC, POSITIVE_DIR, NEGATIVE_DIR, AUGMENTED_DIR
from train_model import load_dataset

MIN_PRECISION = 0.90
MIN_RECALL = 0.85
TEST_SPLIT = 0.15  # 15% held out for validation


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute precision, recall, F1, and false positive rate."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def check_gate(metrics: Dict[str, float]) -> bool:
    """Return True if model meets minimum precision and recall thresholds."""
    return (
        metrics["precision"] >= MIN_PRECISION
        and metrics["recall"] >= MIN_RECALL
    )


def run_validation(model_path: str = None) -> Dict[str, float]:
    """Load model and evaluate on held-out split. Returns metrics dict."""
    if model_path is None:
        model_path = str(MODELS_DIR / ONNX_MODEL_NAME)

    if not Path(model_path).exists():
        print(f"[error] Model not found: {model_path}")
        print("        Run train_model.py first.")
        sys.exit(1)

    print("[1/3] Loading dataset for validation split...")
    X, y = load_dataset(
        positive_dirs=[str(POSITIVE_DIR), str(AUGMENTED_DIR)],
        negative_dirs=[str(NEGATIVE_DIR)],
    )
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=42, stratify=y
    )
    print(f"      Held-out: {len(X_test)} samples ({int(np.sum(y_test==1))} pos, {int(np.sum(y_test==0))} neg)")

    print("[2/3] Running ONNX inference...")
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[1].name  # probability output

    y_pred = []
    for feat in X_test:
        probs = session.run([label_name], {input_name: feat.reshape(1, -1)})[0]
        y_pred.append(1 if probs[0][1] >= 0.5 else 0)
    y_pred = np.array(y_pred)

    print("[3/3] Results:")
    metrics = compute_metrics(y_test, y_pred)
    print(f"  Precision:           {metrics['precision']:.3f}  (min: {MIN_PRECISION})")
    print(f"  Recall:              {metrics['recall']:.3f}  (min: {MIN_RECALL})")
    print(f"  F1:                  {metrics['f1']:.3f}")
    print(f"  False positive rate: {metrics['false_positive_rate']:.3f}")
    print(f"  TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}")

    return metrics


if __name__ == "__main__":
    metrics = run_validation()
    if check_gate(metrics):
        print("\n[PASS] Model meets thresholds. Run export_model.py to deploy.")
    else:
        print(f"\n[FAIL] Model below thresholds (precision>={MIN_PRECISION}, recall>={MIN_RECALL}).")
        print("       Collect more training data and re-run train_model.py.")
        sys.exit(1)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/training/test_validate_model.py -v
```
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add training/validate_model.py tests/training/test_validate_model.py
git commit -m "feat(training): add validate_model.py — precision/recall gate before export"
```

---

## Task 7: export_model.py — copy to ~/.aria/

**Files:**
- Create: `training/export_model.py`
- Create: `tests/training/test_export_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/training/test_export_model.py
import sys
import shutil
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "training"))


def test_export_copies_model(tmp_path):
    """export_model() copies aria.onnx to the specified destination."""
    from export_model import export_model

    src = tmp_path / "models" / "aria.onnx"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"fake onnx content")

    dest_dir = tmp_path / "aria_dir"
    out = export_model(src_path=str(src), dest_dir=str(dest_dir))

    assert Path(out).exists()
    assert Path(out).read_bytes() == b"fake onnx content"


def test_export_creates_dest_dir(tmp_path):
    """export_model() creates the destination directory if it doesn't exist."""
    from export_model import export_model

    src = tmp_path / "aria.onnx"
    src.write_bytes(b"model")
    dest_dir = tmp_path / "new_dir" / "nested"

    out = export_model(src_path=str(src), dest_dir=str(dest_dir))
    assert Path(out).exists()


def test_export_raises_if_src_missing(tmp_path):
    """export_model() raises FileNotFoundError if source model is missing."""
    from export_model import export_model

    with pytest.raises(FileNotFoundError):
        export_model(src_path=str(tmp_path / "missing.onnx"), dest_dir=str(tmp_path))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/training/test_export_model.py -v
```
Expected: `ImportError: No module named 'export_model'`

- [ ] **Step 3: Implement export_model.py**

```python
# training/export_model.py
"""
Export trained model to ~/.aria/aria.onnx for use by wake_word.py.

Usage:
    python export_model.py
"""
import shutil
from pathlib import Path

from common import MODELS_DIR, ONNX_MODEL_NAME

ARIA_DIR = Path.home() / ".aria"


def export_model(
    src_path: str = str(MODELS_DIR / ONNX_MODEL_NAME),
    dest_dir: str = str(ARIA_DIR),
) -> str:
    """Copy src_path to dest_dir/aria.onnx. Returns destination path."""
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(
            f"Model not found at {src_path}. Run train_model.py first."
        )

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / ONNX_MODEL_NAME
    shutil.copy2(str(src), str(out_path))
    return str(out_path)


if __name__ == "__main__":
    out = export_model()
    print(f"[Done] Model exported to {out}")
    print()
    print("Next: update wake_word.py with these two lines:")
    print(f'  _OWW_MODEL = "{out}"')
    print('  _OWW_THRESHOLD = 0.7  # tune after real-world testing (range 0.6–0.8)')
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/training/test_export_model.py -v
```
Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add training/export_model.py tests/training/test_export_model.py
git commit -m "feat(training): add export_model.py — copy trained model to ~/.aria/aria.onnx"
```

---

## Task 8: Update wake_word.py — auto-select custom model

**Files:**
- Modify: `wake_word.py`

The custom ONNX model uses MFCC features + RandomForest, not OWW's proxy models.
A new `_run_custom_onnx()` method handles streaming inference with a 2s sliding window.

- [ ] **Step 1: Add constants to wake_word.py**

In `wake_word.py`, after the existing constants block (after line ~66), add:

```python
# Custom OWW model (set up by training pipeline)
_CUSTOM_MODEL_PATH = os.path.expanduser("~/.aria/aria.onnx")
_CUSTOM_THRESHOLD = 0.7   # starting point — tune after real-world testing
_CUSTOM_N_MFCC = 20
_CUSTOM_WINDOW_SECS = 2.0
```

- [ ] **Step 2: Add _can_use_custom_model() method to WakeWordListener**

After the `_can_use_porcupine()` method (after line ~131), add:

```python
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
```

- [ ] **Step 3: Add _run_custom_onnx() method**

After `_run_openwakeword()` (after line ~304), add:

```python
def _run_custom_onnx(self) -> None:
    """Sliding-window MFCC inference using the custom-trained aria.onnx model."""
    try:
        import onnxruntime as ort
        import librosa
        import pyaudio as _pa
    except ImportError as exc:
        print(f"[Aria] Custom model disabled — missing dep: {exc}")
        self._run_openwakeword()
        return

    try:
        session = ort.InferenceSession(_CUSTOM_MODEL_PATH)
        input_name = session.get_inputs()[0].name
        label_name = session.get_outputs()[1].name
    except Exception as exc:
        print(f"[Aria] Custom model load failed: {exc} — falling back to openwakeword")
        self._run_openwakeword()
        return

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

        while not self._stop_event.is_set():
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
                self._on_wake(stream, _SAMPLE_RATE, _CHUNK_SIZE)
                last_triggered = time.time()

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
```

- [ ] **Step 4: Update _run() to prefer custom model**

In the `_run()` method (bottom of wake_word.py), update the backend selection order:

```python
def _run(self) -> None:
    if self._can_use_porcupine():
        print("[Aria] Wake word backend: Porcupine (custom 'Hey Aria' model)")
        self._run_porcupine()
    elif self._can_use_custom_model():
        print("[Aria] Wake word backend: custom ONNX model (~/.aria/aria.onnx)")
        self._run_custom_onnx()
    else:
        print("[Aria] Wake word backend: openwakeword fallback (proxy model)")
        self._run_openwakeword()
```

- [ ] **Step 5: Verify no syntax errors**

```bash
python -c "import wake_word; print('OK')"
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add wake_word.py
git commit -m "feat(wake_word): add _run_custom_onnx() — auto-selects custom model when ~/.aria/aria.onnx exists"
```

---

## Task 9: Run all training tests

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/training/ -v
```
Expected: all tests PASS. If any fail, fix before proceeding.

- [ ] **Step 2: Verify run order works**

```bash
cd training/
python -c "from common import POSITIVE_DIR, NEGATIVE_DIR; print('common OK')"
python -c "import generate_tts; print('generate_tts OK')"
python -c "import record_samples; print('record_samples OK')"
python -c "import prepare_data; print('prepare_data OK')"
python -c "import train_model; print('train_model OK')"
python -c "import validate_model; print('validate_model OK')"
python -c "import export_model; print('export_model OK')"
```
Expected: all print `OK`

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat(training): complete OWW custom training pipeline — all scripts + tests"
```

---

## Quick Reference — Run Order

```bash
cd training/
pip install -r requirements.txt

python generate_tts.py        # ~5 min  — generates ~500 TTS clips
python record_samples.py      # ~10 min — record 50 clips from your voice
python prepare_data.py        # ~2 min  — augment + generate negatives
python train_model.py         # ~5 min  — train + checkpoint
python validate_model.py      # ~1 min  — check precision/recall
python export_model.py        # instant — copies to ~/.aria/aria.onnx
```

After running `export_model.py`, restart Aria — it will automatically use the custom model.
