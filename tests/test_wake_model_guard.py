from __future__ import annotations

import numpy as np

from aria.voice import wake_word
from aria.voice.wake_features import N_FEATURES, wake_features


def test_single_class_model_is_refused():
    # the old aria.onnx: trained on positives only, {1: 0.99999} for any input
    assert wake_word.fires_on_silence(lambda feat: [{1: 0.9999992}])


def test_model_that_rejects_silence_is_kept():
    assert not wake_word.fires_on_silence(lambda feat: [{0: 1.0, 1: 0.0}])


def test_broken_model_does_not_block_startup():
    def boom(feat):
        raise RuntimeError("bad model")
    assert not wake_word.fires_on_silence(boom)


def test_features_have_the_trained_width_for_any_window():
    for n in (32000, 16000, 2000):
        f = wake_features(np.random.randn(n).astype(np.float32) * 0.01)
        assert f.shape == (N_FEATURES,) and np.isfinite(f).all()


def test_features_look_at_the_word_not_the_padding():
    word = np.sin(np.linspace(0, 400 * np.pi, 8000)).astype(np.float32) * 0.3
    early, late = np.zeros(32000, np.float32), np.zeros(32000, np.float32)
    early[1000:9000] = word
    late[22000:30000] = word
    assert np.allclose(wake_features(early), wake_features(late), atol=1.0)
