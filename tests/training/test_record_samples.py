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
