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
