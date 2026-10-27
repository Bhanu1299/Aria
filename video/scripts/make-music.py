"""Synthesize public/music.mp3: a 120 BPM score cut to the video's scenes.

Made entirely in code (numpy additive synthesis, no samples), so it's royalty-free.
    pip install numpy soundfile
    python scripts/make-music.py

Sections follow timeline.json, so re-run this after changing scene lengths:
    hook       ticking 16ths, tension
    tired      low boom as the monitor goes dark, drone
    combine    pad chords enter, a pluck per tile
    feed       light ticks as postings land
    voice ...  kick and arpeggio join (through system)
    numbers    one hit per number
    never      everything stops; "I do." gets one warm chord
    endcard    plucks over the chord, fade
"""
import json, subprocess, tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
BEAT = 0.5  # 120 BPM
root = Path(__file__).resolve().parent.parent
_scenes = json.loads((root / "timeline.json").read_text())["scenes"]
START, DUR = {}, {}
_t = 0.0
for _s in _scenes:
    START[_s["id"]], DUR[_s["id"]] = _t, _s["seconds"]
    _t += _s["seconds"]
LEN = _t
rng = np.random.default_rng(7)
mix = np.zeros((int(SR * (LEN + 3)), 2))


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def add(sig, t, gain=1.0, pan=0.0):
    i = int(t * SR)
    n = min(len(sig), len(mix) - i)
    if n <= 0:
        return
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    mix[i : i + n, 0] += sig[:n] * gain * l
    mix[i : i + n, 1] += sig[:n] * gain * r


def env(n, a, r, sustain=True):
    t = np.arange(n) / SR
    e = np.minimum(1, t / max(a, 1e-4))
    if sustain:
        e *= np.minimum(1, (n / SR - t) / max(r, 1e-4))
    else:
        e *= np.exp(-t / r)
    return np.clip(e, 0, 1)


def lowpass(x, cutoff):
    a = np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):  # one-pole; fine for short notes
        acc = (1 - a) * v + a * acc
        y[i] = acc
    return y


def pad(notes, dur, gain=0.05):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for m in notes:
        for det in (-0.08, 0, 0.08):
            f = hz(m + det)
            for h in range(1, 7):
                out += np.sin(2 * np.pi * f * h * t + rng.uniform(0, 6)) / (h ** 1.6)
    return out * env(n, 0.9, 1.2) * gain / len(notes)


def pluck(m, dur=0.6, gain=0.12):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = hz(m)
    s = sum(np.sin(2 * np.pi * f * h * t) * np.exp(-t * (4 + 3 * h)) / h for h in range(1, 6))
    return s * env(n, 0.004, 0.25, sustain=False) * gain


def kick(gain=0.6, length=0.45):
    n = int(length * SR)
    t = np.arange(n) / SR
    f = 45 + 80 * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-t * 7) * gain


def tick(gain=0.05, bright=6000):
    n = int(0.04 * SR)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    noise = noise - lowpass(noise, bright)  # crude high-pass
    return noise * np.exp(-t * 120) * gain


def sub(m, dur, gain=0.18):
    n = int(dur * SR)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * hz(m) * t) * env(n, 0.05, 0.3) * gain


def boom(gain=0.7):
    n = int(1.6 * SR)
    t = np.arange(n) / SR
    f = 38 + 60 * np.exp(-t * 10)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.2) * gain


# --- hook: 16th ticks over a sub pulse --------------------------------------------------
hook_beats = int(DUR["hook"] / BEAT)
for k in range(hook_beats * 4):
    t = k * BEAT / 4
    add(tick(0.06 if k % 4 == 0 else 0.035), t, pan=0.3 if k % 2 else -0.3)
    if k % 2 == 0:
        add(pluck(81 + (k // 2) % 4 * 2, 0.15, 0.03), t, pan=0.2)
for k in range(hook_beats):
    add(sub(33, BEAT * 0.9, 0.1 + 0.08 * k / hook_beats), k * BEAT)

# --- tired: boom, then a low drone --------------------------------------------------------
add(boom(), START["tired"])
add(pad([45, 52], DUR["tired"] + 0.3, 0.06), START["tired"])

# --- combine -> system: chords every 2 bars ----------------------------------------------------
chords = [
    ([53, 57, 60, 64], 41),  # Fmaj7
    ([57, 60, 64, 67], 45),  # Am7
    ([55, 60, 64, 67], 48),  # C
    ([55, 59, 62, 67], 43),  # G
]
t0 = START["combine"]
build_end = START["numbers"]
kick_from = START["voice"]
hats_from = START["resume"]
c = 0
while t0 + c * 4 < build_end - 0.01:
    notes, bass = chords[c % 4]
    start = t0 + c * 4
    length = min(4.0, build_end - start)
    add(pad(notes, length + 0.4, 0.055), start, pan=-0.15)
    add(pad([n + 12 for n in notes[1:3]], length + 0.4, 0.02), start, pan=0.25)
    add(sub(bass, length, 0.12), start)
    for b in range(int(length / BEAT)):
        bt = start + b * BEAT
        if bt >= kick_from:
            add(kick(0.42 if b % 2 == 0 else 0.3), bt)
        if bt >= hats_from or START["feed"] <= bt < kick_from:
            add(tick(0.025), bt + BEAT / 2, pan=0.4)
        if bt >= kick_from:
            arp = notes + [notes[0] + 12, notes[2] + 12]
            for e in range(2):
                m = arp[(b * 2 + e) % len(arp)] + 12
                add(pluck(m, 0.5, 0.05), bt + e * BEAT / 2, pan=-0.35 if e else 0.35)
    c += 1

# one pluck per tile in "combine"
for k, m in enumerate([72, 76, 79]):
    add(pluck(m, 0.9, 0.07), START["combine"] + (14 + k * 8) / 30, pan=-0.3 + k * 0.3)

# lift into the numbers
riser_n = int(1.0 * SR)
riser = rng.standard_normal(riser_n)
riser = lowpass(riser, 3000) * np.linspace(0, 1, riser_n) ** 2 * 0.06
add(riser, START["numbers"] - 1.0)

# --- numbers: four hits ----------------------------------------------------------------------
stabs = [[57, 60, 64], [53, 57, 60], [55, 60, 64], [55, 59, 62, 67]]
for k in range(4):
    ht = START["numbers"] + k * DUR["numbers"] / 4
    add(kick(0.7, 0.6), ht)
    add(boom(0.25), ht)
    for m in stabs[k]:
        add(pluck(m + 12, 0.9, 0.09), ht, pan=(m % 3 - 1) * 0.3)
    add(pad(stabs[k], 1.0, 0.04), ht)
    for b in range(1, 4):
        add(tick(0.03), ht + b * BEAT / 2, pan=0.3)

# --- never: silence, then one warm chord on "I do." ------------------------------------------
add(sub(36, 1.2, 0.06), START["never"])
ido = START["never"] + 36 / 30
add(pad([48, 55, 60, 64, 71], LEN - ido + 1, 0.075), ido)  # Cmaj7(add9)
add(sub(36, LEN - ido, 0.12), ido)
for k, m in enumerate([72, 76, 79, 83]):
    add(pluck(m, 1.2, 0.05), ido + k * 0.12, pan=-0.3 + k * 0.2)

# --- end card: gentle plucks over the chord -------------------------------------------------
for k, m in enumerate([76, 79, 74, 72]):
    add(pluck(m, 1.4, 0.035), START["endcard"] + k * 0.8, pan=0.25 * (-1) ** k)

# --- reverb (FFT convolution with a decaying noise tail) + master -------------------------
ir_n = int(2.2 * SR)
ir_t = np.arange(ir_n) / SR
ir = rng.standard_normal((ir_n, 2)) * np.exp(-ir_t * 2.8)[:, None]
ir[0] = 0
wet = np.zeros_like(mix)
L = len(mix) + ir_n
nfft = 1 << (L - 1).bit_length()
for ch in range(2):
    wet[:, ch] = np.fft.irfft(np.fft.rfft(mix[:, ch], nfft) * np.fft.rfft(ir[:, ch], nfft), nfft)[: len(mix)]
wet /= np.max(np.abs(wet)) + 1e-9
out = mix + wet * np.max(np.abs(mix)) * 0.22

out = out[: int(LEN * SR)]
fade = int(1.5 * SR)
out[-fade:] *= np.linspace(1, 0, fade)[:, None] ** 1.5
out = np.tanh(out * 1.6) / np.tanh(1.6)
out *= 0.89 / np.max(np.abs(out))

dest = root / "public" / "music.mp3"
with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
    sf.write(tmp.name, out.astype(np.float32), SR)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp.name, "-b:a", "256k", str(dest)], check=True)
print(f"wrote {dest.relative_to(root)} ({LEN}s)")
