"""
Validate trained model on a held-out set before export.

Prints precision, recall, F1, and false positive rate.
Exits with code 1 if model doesn't meet minimum thresholds.

Usage:
    python validate_model.py

Thresholds:
    MIN_PRECISION = 0.90
    MIN_RECALL    = 0.85
"""
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.model_selection import train_test_split

from common import MODELS_DIR, ONNX_MODEL_NAME, N_MFCC, POSITIVE_DIR, NEGATIVE_DIR, AUGMENTED_DIR
from train_model import load_dataset

MIN_PRECISION = 0.90
MIN_RECALL = 0.85
TEST_SPLIT = 0.15


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
    import onnxruntime as ort

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
