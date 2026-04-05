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
├── prepare_data.py      — augment positives (noise, speed, volume variation)
├── train_model.py       — train OWW classification head
├── export_model.py      — copy trained model to ~/.aria/aria.onnx
├── requirements.txt     — training-only deps (edge-tts, audiomentations, etc.)
├── data/
│   ├── positive/        — all "Aria" clips (TTS + recorded)
│   ├── negative/        — background/noise (OWW built-in set)
│   └── augmented/       — noise-mixed/speed-varied positives
└── models/              — intermediate model output
```

---

## Scripts

### generate_tts.py
- macOS `say`: ~20 voices × 10 phrase variations → ~200 WAV clips → `data/positive/`
- Edge TTS: ~30 voices × 10 variations → ~300 WAV clips → `data/positive/`
- Output format: 16kHz mono WAV (matches OWW input)
- Phrase variations: "Aria", "Hey Aria", "Aria!", "Okay Aria", etc.

### record_samples.py
- Interactive CLI loop
- Countdown → record 1.5s at 16kHz mono → playback → user keeps or discards
- Target: ~50 clips from user's voice → `data/positive/`

### prepare_data.py
- Loads all `data/positive/` clips
- Augmentations: room noise overlay, mic noise, speed ±10%, volume ±20%
- Output: `data/augmented/` (~doubles dataset)
- Downloads OWW's built-in negative sample set into `data/negative/`

### train_model.py
- Loads: `data/positive/` + `data/augmented/` + `data/negative/`
- Uses OWW's pre-trained audio embedding model (no GPU required)
- Trains small classification head (~5 min on CPU)
- Saves trained model to `models/aria.onnx`

### export_model.py
- Copies `models/aria.onnx` → `~/.aria/aria.onnx`
- Prints integration instructions for `wake_word.py`

---

## Integration (after training)

Two line changes in `wake_word.py`:

```python
_OWW_MODEL = os.path.expanduser("~/.aria/aria.onnx")  # was "alexa"
_OWW_THRESHOLD = 0.7  # raised from 0.35 — custom model is much more precise
```

No other changes to the main codebase.

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
# training/requirements.txt
edge-tts
audiomentations
openwakeword[train]
sounddevice
soundfile
numpy
```

---

## Run Order

```bash
cd training/
pip install -r requirements.txt

python generate_tts.py        # ~5 min
python record_samples.py      # ~10 min (interactive)
python prepare_data.py        # ~2 min
python train_model.py         # ~5 min
python export_model.py        # instant
```
