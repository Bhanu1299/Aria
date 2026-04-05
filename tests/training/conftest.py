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
