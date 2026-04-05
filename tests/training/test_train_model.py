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
