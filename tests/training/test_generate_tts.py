# tests/training/test_generate_tts.py
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
import numpy as np
import soundfile as sf
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "training"))


def test_say_writes_wav(tmp_path):
    """generate_say_clips() creates WAV files with correct format."""
    from generate_tts import generate_say_clips

    out_dir = tmp_path / "positive"
    out_dir.mkdir()

    # Mock subprocess.run to write a silent WAV instead of calling say
    def fake_say(cmd, **kwargs):
        output_path = cmd[cmd.index("-o") + 1]
        samples = np.zeros(32000, dtype=np.int16)
        sf.write(output_path, samples, 16000, subtype="PCM_16")
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=fake_say):
        generate_say_clips(out_dir=str(out_dir), voices=["Alex"], phrases=["Aria"])

    wavs = list(out_dir.glob("*.wav"))
    assert len(wavs) == 1
    data, sr = sf.read(str(wavs[0]))
    assert sr == 16000


def test_edge_tts_writes_wav(tmp_path):
    """generate_edge_clips() creates WAV files via Edge TTS."""
    import asyncio
    from generate_tts import generate_edge_clips

    out_dir = tmp_path / "positive"
    out_dir.mkdir()

    async def fake_save(path):
        # Write a dummy MP3 file (edge_tts.save() creates an MP3)
        sf.write(path.replace(".mp3", ".wav"), np.zeros(32000, dtype=np.int16), 16000, subtype="PCM_16")

    mock_communicate = MagicMock()
    mock_communicate.return_value.save = fake_save

    def fake_ffmpeg(cmd, **kwargs):
        # ffmpeg command: ["ffmpeg", "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1", wav_path]
        # Just write the WAV directly since we already have audio
        wav_path = cmd[-1]
        sf.write(wav_path, np.zeros(32000, dtype=np.int16), 16000, subtype="PCM_16")
        return MagicMock(returncode=0)

    with patch("edge_tts.Communicate", mock_communicate):
        with patch("subprocess.run", side_effect=fake_ffmpeg):
            asyncio.run(generate_edge_clips(
                out_dir=str(out_dir), voices=["en-US-JennyNeural"], phrases=["Aria"]
            ))

    wavs = list(out_dir.glob("*.wav"))
    assert len(wavs) >= 1
    data, sr = sf.read(str(wavs[0]))
    assert sr == 16000
