# training/train_model.py
"""
Train a custom wake word classifier using MFCC features + RandomForest.

Pipeline:
  1. Load all positive + augmented + negative WAV files
  2. Extract 20 MFCC features per clip (mean over time axis)
  3. Train RandomForestClassifier
  4. Export to ONNX via skl2onnx
  5. Save checkpoint to models/checkpoints/clf_latest.joblib

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
    Returns None if file cannot be processed.
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

    # Save checkpoint
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
