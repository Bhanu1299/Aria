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
