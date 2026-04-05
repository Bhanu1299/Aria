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
