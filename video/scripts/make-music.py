"""Synthesize public/music.mp3: a driving 120 BPM track cut to the video's scenes.

Made entirely in code (numpy/scipy synthesis, no samples), so it's royalty-free.
    pip install numpy scipy soundfile
    python scripts/make-music.py

Sections follow timeline.json, so re-run this after changing scene order or lengths:
    hook      beat from frame one, filtered lead riff opening up, snare roll into the cut
    tired     hard stop: one impact, a low drone, a riser back in
    combine   the drop: four-on-the-floor, claps, pumping bass, chord stabs, lead hook
    ... cost  groove carries through (lead steps back while the voice talks), builds into "never"
    never     dead silence; "I do." lands on a big hit and chord
    endcard   the hook once more, final hit
"""
import json, subprocess, tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

SR = 44100
BEAT = 0.5  # 120 BPM: one beat = 15 frames at 30 fps
S16 = BEAT / 4
root = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(11)

_scenes = json.loads((root / "timeline.json").read_text())["scenes"]
START, DUR = {}, {}
_t = 0.0
for _s in _scenes:
    START[_s["id"]], DUR[_s["id"]] = _t, _s["seconds"]
    _t += _s["seconds"]
LEN = _t
END = {k: START[k] + DUR[k] for k in START}

N = int(SR * (LEN + 3))
bus = {name: np.zeros((N, 2)) for name in ("drums", "bass", "music", "lead", "fx")}


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def add(name, sig, t, gain=1.0, pan=0.0):
    i = int(round(t * SR))
    if i >= N or i + len(sig) <= 0:
        return
    if i < 0:
        sig, i = sig[-i:], 0
    n = min(len(sig), N - i)
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    bus[name][i : i + n, 0] += sig[:n] * gain * l
    bus[name][i : i + n, 1] += sig[:n] * gain * r


def filt(x, cutoff, kind="low", order=2):
    cutoff = np.clip(cutoff, 20, SR / 2 - 100)
    if kind == "band":
        b, a = butter(order, [c / (SR / 2) for c in cutoff], btype="band")
    else:
        b, a = butter(order, cutoff / (SR / 2), btype=kind)
    return lfilter(b, a, x)


def tt(dur):
    return np.arange(int(dur * SR)) / SR


def saw(f, dur, harmonics=24):
    t = tt(dur)
    out = np.zeros_like(t)
    for h in range(1, harmonics + 1):
        if f * h > SR / 2.2:
            break
        out += np.sin(2 * np.pi * f * h * t) / h
    return out


def adsr(n, a=0.005, d=0.15, s=0.0, r=0.05):
    t = np.arange(n) / SR
    e = np.where(t < a, t / a, s + (1 - s) * np.exp(-(t - a) / max(d, 1e-4)))
    tail = int(r * SR)
    if tail and n > tail:
        e[-tail:] *= np.linspace(1, 0, tail)
    return e


# --- instruments -----------------------------------------------------------------------------
def kick(gain=1.0):
    t = tt(0.42)
    f = 48 + 110 * np.exp(-t * 30)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 6.5)
    click = filt(rng.standard_normal(len(t)), 3000, "high") * np.exp(-t * 300) * 0.4
    return np.tanh((body + click) * 1.6) * gain


def clap(gain=0.5):
    t = tt(0.35)
    noise = filt(rng.standard_normal(len(t)), [900, 3500], "band")
    env = np.zeros_like(t)
    for k, off in enumerate((0, 0.011, 0.022)):
        env += np.exp(-np.clip(t - off, 0, None) * (160 if k < 2 else 18)) * (t >= off)
    return noise * env * gain


def snare(gain=0.4):
    t = tt(0.22)
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30)
    noise = filt(rng.standard_normal(len(t)), 1800, "high") * np.exp(-t * 22)
    return (tone * 0.6 + noise) * gain


def hat(gain=0.12, open_=False):
    t = tt(0.25 if open_ else 0.05)
    noise = filt(rng.standard_normal(len(t)), 7000, "high")
    return noise * np.exp(-t * (14 if open_ else 90)) * gain


def bass(m, dur, gain=0.32, cutoff=700):
    s = saw(hz(m), dur, 16) + 0.6 * np.sin(2 * np.pi * hz(m) * tt(dur))
    return filt(s, cutoff) * adsr(len(s), 0.004, 0.18, 0.55, 0.02) * gain


def stab(notes, dur=0.32, gain=0.16, cutoff=2600):
    out = sum(saw(hz(m + d), dur, 20) for m in notes for d in (-0.07, 0.07))
    return filt(out, cutoff) * adsr(len(out), 0.003, 0.12, 0.0, 0.02) * gain / len(notes)


def lead(m, dur, gain=0.16, cutoff=4000):
    s = saw(hz(m), dur, 28) + 0.5 * saw(hz(m + 0.1), dur, 28)
    s = filt(s, cutoff)
    return s * adsr(len(s), 0.003, 0.2, 0.25, 0.03) * gain


def supersaw(notes, dur, gain=0.07, cutoff=3000):
    out = sum(saw(hz(m + d), dur, 18) for m in notes for d in (-0.12, -0.04, 0.04, 0.12))
    out = filt(out, cutoff)
    n = len(out)
    env = np.minimum(1, np.arange(n) / (0.02 * SR)) * np.minimum(1, (n - np.arange(n)) / (0.4 * SR))
    return out * env * gain / len(notes)


def impact(gain=0.9):
    t = tt(2.2)
    f = 36 + 70 * np.exp(-t * 9)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.4)
    crash = filt(rng.standard_normal(len(t)), 4000, "high") * np.exp(-t * 2.8) * 0.35
    return (boom + crash) * gain


def riser(dur, gain=0.14):
    t = tt(dur)
    noise = rng.standard_normal(len(t))
    out = np.zeros_like(t)
    chunk = int(0.05 * SR)
    for i in range(0, len(t), chunk):  # band-pass sweeping up
        c = 400 + 7000 * (i / len(t)) ** 2
        out[i : i + chunk] = filt(noise[i : i + chunk], [c, c * 1.6], "band", 1)
    return out * (t / dur) ** 2 * gain


# --- harmony -----------------------------------------------------------------------------------
CHORDS = [  # i - VI - III - VII in A minor, one bar (4 beats) each
    ([57, 60, 64], 33),  # Am
    ([53, 57, 60], 29),  # F
    ([55, 60, 64], 36),  # C
    ([55, 59, 62], 31),  # G
]
# The hook: one bar of 16ths (None = rest). Minor-pentatonic so it sits over all four chords.
RIFF = [81, None, 76, 79, None, 81, None, 84, 83, None, 79, None, 76, 79, 74, None]
RIFF_B = [81, None, 76, 79, None, 81, None, 84, 86, None, 84, None, 83, 79, 81, None]


def chord_at(t):
    bar = int((t + 1e-6) // (4 * BEAT))
    return CHORDS[bar % 4]


kicks = []


def groove(t0, t1, *, full=True, lead_gain=1.0, lead_cut=4500, clap_on=True, stabs=True, hats16=True):
    """Lay the main groove between t0 and t1 (seconds), on the beat grid."""
    b0 = int(np.ceil(t0 / BEAT - 1e-6))
    b1 = int(np.floor(t1 / BEAT - 1e-6))
    for b in range(b0, b1 + 1):
        bt = b * BEAT
        if bt >= t1 - 1e-6:
            break
        notes, root_ = chord_at(bt)
        add("drums", kick(0.95), bt)
        kicks.append(bt)
        if clap_on and b % 2 == 1:
            add("drums", clap(0.5), bt, pan=0.05)
        for k in range(4):
            st = bt + k * S16
            if st >= t1:
                break
            if hats16 or k % 2 == 0:
                add("drums", hat(0.10 if k % 2 else 0.06), st, pan=0.35)
            if k == 2:
                add("drums", hat(0.07, open_=True), st, pan=-0.3)
            # pumping bass on the off-16ths
            if full and k in (1, 2, 3):
                add("bass", bass(root_ + (12 if k == 3 else 0), S16 * 0.95), st)
        if stabs and full and b % 4 in (0, 2):
            add("music", stab([n + 12 for n in notes]), bt + BEAT * 0.5, pan=-0.2)
            add("music", stab([n + 12 for n in notes]), bt + BEAT * 1.5, gain=0.7, pan=0.2)
        # lead riff, one 16th at a time
        if lead_gain > 0:
            bar_i = int((bt + 1e-6) // (4 * BEAT))
            riff = RIFF if bar_i % 2 == 0 else RIFF_B
            for k in range(4):
                st = bt + k * S16
                if st >= t1:
                    break
                m = riff[(b % 4) * 4 + k]
                if m is not None:
                    add("lead", lead(m, S16 * 1.7, cutoff=lead_cut), st, gain=lead_gain, pan=0.1 * (-1) ** k)
    # sustained supersaw bed per bar
    if full:
        bar0 = int(t0 // (4 * BEAT))
        bar1 = int(np.ceil(t1 / (4 * BEAT)))
        for bar in range(bar0, bar1):
            s = max(t0, bar * 4 * BEAT)
            e = min(t1, (bar + 1) * 4 * BEAT)
            if e - s > 0.05:
                notes, _ = chord_at(s)
                add("music", supersaw(notes, e - s + 0.1), s)


# --- arrangement -----------------------------------------------------------------------------
# hook: beat from the first frame, lead filtered and opening, snare roll into the cut
h0, h1 = START["hook"], END["hook"]
groove(h0, h1, full=True, lead_gain=0.8, lead_cut=1400, stabs=False)
for k in range(8):  # snare roll on the last beat, doubling up
    add("drums", snare(0.22 + 0.04 * k), h1 - BEAT + k * BEAT / 8)
add("fx", riser(DUR["hook"]), h0, gain=0.8)

# tired: hard stop, one impact, low drone, riser back in
add("fx", impact(), START["tired"])
add("music", supersaw([45, 52], DUR["tired"], 0.05, 900), START["tired"])
add("fx", riser(1.0, 0.2), END["tired"] - 1.0)

# combine -> cost: the drop and the groove; lead steps back under the voice
drop = START["combine"]
add("fx", impact(0.7), drop)
add("music", supersaw([57, 60, 64, 69], 1.2, 0.09, 5000), drop)
voice_heavy = {"voice", "coldstart", "gateway", "ready", "cost"}
order = [s["id"] for s in _scenes]
for sid in order[order.index("combine") : order.index("never")]:
    lg = 0.35 if sid in voice_heavy else 1.0
    groove(START[sid], END[sid], lead_gain=lg, lead_cut=5200 if lg == 1.0 else 3000)
    if sid != "combine":
        add("drums", hat(0.12, open_=True), START[sid])  # mark each cut

# build into "never": snare roll and riser across the last bar of the cost scene
b_end = END["cost"]
for k in range(16):  # 8ths, then 16ths for the last beat
    add("drums", snare(0.15 + 0.02 * k), b_end - 2.0 + k * BEAT / 2 if k < 12 else b_end - BEAT + (k - 12) * S16)
add("fx", riser(2.0, 0.22), b_end - 2.0)

# never: silence. "I do." lands at frame 38 of the scene (with the voice), big hit + chord
ido = START["never"] + 38 / 30
add("fx", impact(1.0), ido)
add("drums", kick(1.0), ido)
add("music", supersaw([57, 64, 69, 72, 76], END["never"] - ido + 0.6, 0.11, 4200), ido)
add("bass", bass(33, END["never"] - ido, 0.3, 300), ido)

# endcard: the hook once more, then a final hit
groove(START["endcard"], END["endcard"] - 1.0, lead_gain=0.9, lead_cut=5200)
add("fx", impact(0.8), END["endcard"] - 1.0)
add("music", supersaw([57, 60, 64, 69], 1.2, 0.08, 3000), END["endcard"] - 1.0)

# --- mix ---------------------------------------------------------------------------------------
# sidechain pump: bass and music duck on every kick
t = np.arange(N) / SR
duck = np.ones(N)
for k in sorted(set(round(x, 4) for x in kicks)):
    i = int(k * SR)
    seg = np.arange(min(int(0.3 * SR), N - i)) / SR
    duck[i : i + len(seg)] = np.minimum(duck[i : i + len(seg)], 1 - 0.7 * np.exp(-seg / 0.09))
for name in ("bass", "music"):
    bus[name] *= duck[:, None]

mix = bus["drums"] * 0.9 + bus["bass"] + bus["music"] + bus["lead"] + bus["fx"]

# short room reverb on a send (lead, music, claps live in it)
send = bus["lead"] * 0.5 + bus["music"] * 0.4 + bus["drums"] * 0.08
ir_n = int(1.4 * SR)
ir_t = np.arange(ir_n) / SR
ir = rng.standard_normal((ir_n, 2)) * np.exp(-ir_t * 4.5)[:, None]
ir[: int(0.01 * SR)] = 0
nfft = 1 << (N + ir_n - 1).bit_length()
wet = np.zeros_like(mix)
for ch in range(2):
    wet[:, ch] = np.fft.irfft(np.fft.rfft(send[:, ch], nfft) * np.fft.rfft(ir[:, ch], nfft), nfft)[:N]
wet *= np.max(np.abs(send)) / (np.max(np.abs(wet)) + 1e-9) * 0.35
mix += wet

# hard silence for "It never submits." (everything stops), up to "I do."
s0, s1 = int(START["never"] * SR), int(ido * SR)
mix[s0 + int(0.02 * SR) : s1] *= 0.0
mix[s0 : s0 + int(0.02 * SR)] *= np.linspace(1, 0, int(0.02 * SR))[:, None]

out = mix[: int(LEN * SR)]
fade = int(0.8 * SR)
out[-fade:] *= np.linspace(1, 0, fade)[:, None] ** 1.5
out = np.tanh(out * 1.8) / np.tanh(1.8)
out *= 0.9 / np.max(np.abs(out))

dest = root / "public" / "music.mp3"
with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
    sf.write(tmp.name, out.astype(np.float32), SR)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp.name, "-b:a", "256k", str(dest)], check=True)
print(f"wrote {dest.relative_to(root)} ({LEN}s)")
