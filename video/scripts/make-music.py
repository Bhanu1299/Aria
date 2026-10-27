"""Synthesize public/music.mp3: a minimal, low-key 120 BPM bed that sits under the text.

Made entirely in code (numpy/scipy synthesis, no samples), so it's royalty-free.
    pip install numpy scipy soundfile
    python scripts/make-music.py

Steady low pulse, soft filtered pads, a few quiet synth notes. No vocals, no lead melody,
no drop. Sections follow timeline.json, so re-run this after changing scene order or lengths:
    hook -> voice        sparse: low pulse and pad only
    coldstart -> ready   a light beat joins (soft kick, shaker)
    cost                 the one small lift: pad opens up, a quiet high arp
    never                beat stops; everything fades to silence on "I do."
    endcard              silence
"""
import json, subprocess, tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

SR = 44100
BEAT = 0.5  # 120 BPM: one beat = 15 frames at 30 fps
root = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(5)

_scenes = json.loads((root / "timeline.json").read_text())["scenes"]
START, DUR = {}, {}
_t = 0.0
for _s in _scenes:
    START[_s["id"]], DUR[_s["id"]] = _t, _s["seconds"]
    _t += _s["seconds"]
LEN = _t
END = {k: START[k] + DUR[k] for k in START}
ORDER = [s["id"] for s in _scenes]

N = int(SR * (LEN + 3))
mix = np.zeros((N, 2))
verb_send = np.zeros((N, 2))


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def tt(dur):
    return np.arange(int(dur * SR)) / SR


def add(sig, t, gain=1.0, pan=0.0, verb=0.0):
    i = int(round(t * SR))
    if i >= N:
        return
    n = min(len(sig), N - i)
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    for buf, g in ((mix, gain), (verb_send, gain * verb)):
        buf[i : i + n, 0] += sig[:n] * g * l
        buf[i : i + n, 1] += sig[:n] * g * r


def lowpass(x, cutoff, order=2):
    b, a = butter(order, min(cutoff, SR / 2 - 100) / (SR / 2))
    return lfilter(b, a, x)


def highpass(x, cutoff, order=2):
    b, a = butter(order, cutoff / (SR / 2), btype="high")
    return lfilter(b, a, x)


def soft_saw(f, dur, harmonics=10):
    t = tt(dur)
    return sum(np.sin(2 * np.pi * f * h * t + h) / h ** 1.3 for h in range(1, harmonics + 1) if f * h < SR / 2.5)


# --- sounds ----------------------------------------------------------------------------------
def pulse(m, gain=0.22):
    """Muted low synth pulse: short, round, felt more than heard."""
    t = tt(0.24)
    s = np.sin(2 * np.pi * hz(m) * t) + 0.35 * np.sin(2 * np.pi * hz(m) * 2 * t)
    env = np.minimum(1, t / 0.006) * np.exp(-t * 14)
    return lowpass(s * env, 500) * gain


def pad(notes, dur, cutoff=1100, gain=0.05):
    out = sum(soft_saw(hz(m + d), dur) for m in notes for d in (-0.06, 0.06))
    out = lowpass(out, cutoff)
    n = len(out)
    t = np.arange(n) / SR
    env = np.minimum(1, t / 0.8) * np.minimum(1, (dur - t) / 0.8)
    return out * env * gain / len(notes)


def blip(m, gain=0.05):
    """A quiet, soft synth note. Used sparingly, never as a melody."""
    t = tt(0.9)
    s = np.sin(2 * np.pi * hz(m) * t) + 0.25 * np.sin(2 * np.pi * hz(m) * 3 * t) * np.exp(-t * 8)
    return s * np.minimum(1, t / 0.004) * np.exp(-t * 4) * gain


def soft_kick(gain=0.35):
    t = tt(0.3)
    f = 50 + 45 * np.exp(-t * 35)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 11)
    return lowpass(s, 900) * gain


def shaker(gain=0.035):
    t = tt(0.07)
    s = highpass(rng.standard_normal(len(t)), 6000)
    return s * np.minimum(1, t / 0.01) * np.exp(-t * 60) * gain


def rim(gain=0.03):
    t = tt(0.05)
    s = np.sin(2 * np.pi * 1700 * t) * np.exp(-t * 120) + highpass(rng.standard_normal(len(t)), 3000) * np.exp(-t * 150) * 0.3
    return s * gain


# --- harmony -----------------------------------------------------------------------------------
# Am9 - Fmaj7 - Cmaj7 - Em7, two bars (4 s) each, voiced low and close.
CHORDS = [
    ([57, 60, 64, 71], 45),
    ([53, 57, 60, 64], 41),
    ([48, 55, 59, 64], 36),
    ([52, 55, 59, 62], 40),
]


def chord_at(t):
    return CHORDS[int((t + 1e-6) // 4.0) % len(CHORDS)]


def scene_of(t):
    for sid in ORDER:
        if START[sid] <= t < END[sid]:
            return sid
    return ORDER[-1]


never_t = START["never"]
# The closing line ("That's mine.") is the last voiceover line in the never scene.
_vo = [l for l in json.loads((root / "voiceover.json").read_text())["lines"] if l["scene"] == "never"]
ido = START["never"] + max(l["at"] for l in _vo) / 30
lift0, lift1 = START["cost"], END["cost"]
beat_scenes = set(ORDER[ORDER.index("coldstart") : ORDER.index("never")])

# Pads, one per chord span, up to "I do."
t = 0.0
while t < ido:
    notes, _ = chord_at(t)
    span = min(4.0, ido + 0.8 - t)
    cutoff = 2600 if lift0 <= t < lift1 else 1100
    add(pad(notes, span + 0.6, cutoff, 0.06 if lift0 <= t < lift1 else 0.05), t, verb=0.6)
    t += 4.0

# Beat grid
b = 0
while b * BEAT < never_t - 1e-6:
    bt = b * BEAT
    sid = scene_of(bt)
    notes, bass = chord_at(bt)
    # steady low pulse on 8ths, every scene
    for e in range(2):
        add(pulse(bass, 0.2 if e == 0 else 0.13), bt + e * BEAT / 2)
    # a few quiet notes: one chord tone every other bar, never a tune
    if b % 8 == 4:
        add(blip(notes[(b // 8) % len(notes)] + 12), bt, pan=0.3 if (b // 8) % 2 else -0.3, verb=0.8)
    if sid in beat_scenes:
        if b % 2 == 0:
            add(soft_kick(), bt)
        else:
            add(rim(), bt, pan=0.15, verb=0.3)
        for e in range(2):
            add(shaker(0.03 if e == 0 else 0.045), bt + e * BEAT / 2, pan=-0.35)
    if lift0 <= bt < lift1:
        # the one small lift: a quiet high arp in 8ths, rising slightly in level
        p = (bt - lift0) / (lift1 - lift0)
        for e in range(2):
            m = notes[(b * 2 + e) % len(notes)] + 24
            add(blip(m, 0.018 + 0.02 * p), bt + e * BEAT / 2, pan=0.4 * (-1) ** e, verb=0.9)
    b += 1

# --- reverb, master ----------------------------------------------------------------------------
ir_n = int(2.4 * SR)
ir_t = np.arange(ir_n) / SR
ir = rng.standard_normal((ir_n, 2)) * np.exp(-ir_t * 2.6)[:, None]
ir[: int(0.015 * SR)] = 0
nfft = 1 << (N + ir_n - 1).bit_length()
wet = np.zeros_like(mix)
for ch in range(2):
    wet[:, ch] = np.fft.irfft(np.fft.rfft(verb_send[:, ch], nfft) * np.fft.rfft(ir[:, ch], nfft), nfft)[:N]
wet *= np.max(np.abs(verb_send)) / (np.max(np.abs(wet)) + 1e-9) * 0.5
out = mix + wet

# Fade to silence across "It never submits.", reaching zero on "I do."; silent after.
tline = np.arange(N) / SR
fade = np.clip((ido - tline) / (ido - never_t), 0, 1) ** 1.5
fade[tline < never_t] = 1
out *= fade[:, None]

out = out[: int(LEN * SR)]
out = np.tanh(out * 1.2) / np.tanh(1.2)
out *= 0.8 / np.max(np.abs(out))

dest = root / "public" / "music.mp3"
with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
    sf.write(tmp.name, out.astype(np.float32), SR)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp.name, "-b:a", "256k", str(dest)], check=True)
print(f"wrote {dest.relative_to(root)} ({LEN}s)")
