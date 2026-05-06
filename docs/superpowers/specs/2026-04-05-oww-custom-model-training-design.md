# Custom OpenWakeWord Model Training Pipeline — Design Spec

**Date:** 2026-04-05
**Status:** Approved
**Wake phrase:** "Aria" (superset: "Hey Aria" triggers naturally)

---

## Problem

The current `wake_word.py` uses the `alexa` OWW model as a phonetic proxy for "Aria".
This causes false triggers and misses. A custom-trained OWW model for "Aria" will fix both.

---

## Approach

Step-by-step scripts in a `training/` folder. Each script is independently re-runnable.
Create everything now; train when ready.

---

## Folder Structure

```
training/
├── generate_tts.py      — synthesize "Aria" clips via macOS say + Edge TTS
├── record_samples.py    — interactive CLI: record your own voice samples
├── prepare_data.py      — augment positives + download/check negatives
├── train_model.py       — train OWW classification head with checkpointing
├── validate_model.py    — precision/recall on held-out set before export
├── export_model.py      — copy trained model to ~/.aria/aria.onnx
├── requirements.txt     — training-only deps (pinned versions)
├── data/
│   ├── positive/        — all "Aria" clips (TTS + recorded)
│   ├── negative/        — background/noise (OWW built-in set)
│   └── augmented/       — noise-mixed/speed-varied positives
└── models/              — intermediate model output + checkpoints
```

---

## Scripts

### generate_tts.py
- macOS `say`: ~20 voices × 10 phrase variations → ~200 WAV clips → `data/positive/`
- Edge TTS: ~30 voices × 10 variations → ~300 WAV clips → `data/positive/`
- Edge TTS uses `asyncio.run()` — must be wrapped in an async loop, not called synchronously
- Output format: 16kHz mono WAV (matches OWW input)
- Phrase variations: "Aria", "Hey Aria", "Aria!", "Okay Aria", etc.

### record_samples.py
- Interactive CLI loop
- Countdown → record 2.0s at 16kHz mono (configurable via `--duration`) → playback → user keeps or discards
- Default 2.0s (not 1.5s) — "Hey Aria" at natural pace hits 1.8–2.0s; 1.5s clips the end
- Target: ~50 clips from user's voice → `data/positive/`

### prepare_data.py
- Loads all `data/positive/` clips
- Augmentations: room noise overlay, mic noise, speed ±10%, volume ±20%
- Output: `data/augmented/` (~doubles dataset)
- Negative set: checks if `data/negative/` is already populated — skips download if so (re-runs are fast)
- If download needed: fetches OWW's built-in negative sample set with timeout + fallback message
- Prints clip count summary at end: positives / augmented / negatives — verify ~1100 positives hit

### train_model.py
- Loads: `data/positive/` + `data/augmented/` + `data/negative/`
- Uses OWW's pre-trained audio embedding model (no GPU required)
- Trains small classification head (~5 min on CPU)
- Checkpoints to `models/checkpoints/` — resumes from latest checkpoint if interrupted
- Saves final trained model to `models/aria.onnx`

### validate_model.py
- Runs trained `models/aria.onnx` against a held-out split (10% of positives + negatives)
- Prints precision, recall, F1, and false positive rate
- Gate: warn if precision < 0.90 or recall < 0.85 — don't export a bad model

### export_model.py
- Copies `models/aria.onnx` → `~/.aria/aria.onnx`
- Prints integration instructions for `wake_word.py`

---

## Integration (after training)

Two line changes in `wake_word.py`:

```python
_OWW_MODEL = os.path.expanduser("~/.aria/aria.onnx")  # was "alexa"
_OWW_THRESHOLD = 0.7  # starting point — tune after real-world testing (was 0.35)
```

No other changes to the main codebase. Threshold 0.7 is a starting point; expect to tune
it after a few real sessions (range likely 0.6–0.8 depending on environment).

---

## Training Data Summary

| Source              | Count     | Location          |
|---------------------|-----------|-------------------|
| macOS say TTS       | ~200 clips| data/positive/    |
| Edge TTS            | ~300 clips| data/positive/    |
| Your voice          | ~50 clips | data/positive/    |
| Augmented positives | ~550 clips| data/augmented/   |
| Negatives (OWW set) | built-in  | data/negative/    |
| **Total positives** | **~1100** |                   |

---

## Dependencies

```
# training/requirements.txt — pin versions to avoid mid-run surprises
# openwakeword[train] pulls tensorflow or tflite-runtime depending on version
# verify: pip install openwakeword==0.6.0 tensorflow==2.13.0
openwakeword==0.6.0
tensorflow==2.13.0          # or tflite-runtime — check OWW release notes for your version
edge-tts==6.1.9
audiomentations==0.33.0
sounddevice==0.4.6
soundfile==0.12.1
numpy==1.24.4
```

---

## Run Order

```bash
cd training/
pip install -r requirements.txt

python generate_tts.py        # ~5 min
python record_samples.py      # ~10 min (interactive), default 2.0s per clip
python prepare_data.py        # ~2 min (skips negative download if already cached)
python train_model.py         # ~5 min (checkpoints on interrupt)
python validate_model.py      # ~1 min (check precision/recall before export)
python export_model.py        # instant
```
