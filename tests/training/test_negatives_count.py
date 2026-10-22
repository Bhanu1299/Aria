"""tests/training/test_negatives_count.py — generate_negatives fills its quota on a rerun.

No audiomentations needed (unlike test_prepare_data.py), so this always runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "training"))


def test_rerun_counts_existing_clips_once(tmp_path, monkeypatch):
    """#10: existing clips were counted twice, so a rerun generated too few negatives."""
    import prepare_data
    monkeypatch.setattr(prepare_data, "NON_WAKE_PHRASES", [])            # no `say` calls in the test
    neg = tmp_path / "negative"
    neg.mkdir()
    for i in range(30):                                                  # a partial earlier run
        sf.write(str(neg / f"quiet_{i:04d}.wav"), np.zeros(16000, dtype=np.int16), 16000, subtype="PCM_16")
    n = prepare_data.generate_negatives(neg_dir=str(neg), n_clips=100)
    assert len(list(neg.glob("*.wav"))) == 100 and n == 100
