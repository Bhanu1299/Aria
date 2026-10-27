"""Generate the voiceover clips in public/vo/ from voiceover.json.

Runs fully offline with Kokoro (open-weights TTS, Apache-2.0):
    pip install kokoro-onnx soundfile
    python scripts/make-voice.py --models /path/to/kokoro-models
where the models folder holds kokoro-v1.0.onnx and voices-v1.0.bin from
https://github.com/thewh1teagle/kokoro-onnx/releases

To use your own voice instead, record each line as public/vo/<id>.mp3 and skip this.
"""
import argparse, json, subprocess, tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

root = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)
ap.add_argument("--voice", help="override the voice in voiceover.json")
args = ap.parse_args()

cfg = json.loads((root / "voiceover.json").read_text())
voice = args.voice or cfg["voice"]
models = Path(args.models)
tts = Kokoro(str(models / "kokoro-v1.0.onnx"), str(models / "voices-v1.0.bin"))
out = root / "public" / "vo"
out.mkdir(parents=True, exist_ok=True)

TARGET_LUFS = -14.0


def run(a):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *a], check=True)


def loudness(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", path, "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    lines = [l for l in r.stderr.splitlines() if l.strip().startswith("I:")]
    return float(lines[-1].split()[1])


durations = {}
for line in cfg["lines"]:
    samples, sr = tts.create(line["text"], voice=voice, speed=line.get("speed", cfg["speed"]), lang="en-us")
    # Trim leading/trailing silence so `at` lands on the first syllable.
    idx = np.where(np.abs(samples) > 0.01)[0]
    samples = samples[max(0, idx[0] - int(0.02 * sr)) : idx[-1] + int(0.12 * sr)]
    with tempfile.TemporaryDirectory() as tmp:
        raw, comp = f"{tmp}/raw.wav", f"{tmp}/comp.wav"
        sf.write(raw, samples, sr)
        # Even out the level, then measure and lift to the target loudness.
        run(["-i", raw, "-af", "highpass=f=80,acompressor=threshold=-24dB:ratio=3:attack=5:release=90:makeup=6dB", comp])
        gain = TARGET_LUFS - loudness(comp)
        run(["-i", comp, "-af", f"volume={gain:.2f}dB,alimiter=limit=0.8:level=false", "-ar", "48000", "-b:a", "192k",
             str(out / f"{line['id']}.mp3")])
    durations[line["id"]] = round(len(samples) / sr, 2)
    print(f"{line['id']:<12} {durations[line['id']]:>5}s  {line['text']}")

(out / "durations.json").write_text(json.dumps(durations, indent=2) + "\n")

# captions.srt for upload, timed from the clips; numbers written as digits for reading.
starts, t = {}, 0.0
for s in json.loads((root / "timeline.json").read_text())["scenes"]:
    starts[s["id"]], t = t, t + s["seconds"]
READ = {"twenty-four": "24", "eighty-seven": "87", "three dollars": "$3.01", "twenty-one cents": "$0.21"}


def stamp(x):
    ms = int(round(x * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


cues = []
for line in cfg["lines"]:
    if line["scene"] not in starts:
        continue
    a = starts[line["scene"]] + line["at"] / 30
    text = line["text"]
    for k, val in READ.items():
        text = text.replace(k, val)
    cues.append((a, a + durations[line["id"]], text))
srt = "\n".join(f"{i}\n{stamp(a)} --> {stamp(b)}\n{text}\n" for i, (a, b, text) in enumerate(sorted(cues), 1))
(root / "captions.srt").write_text(srt)
