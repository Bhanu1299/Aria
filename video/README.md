# "I stopped applying to jobs by hand": the video

A 47.5-second motion-graphics video, built entirely in code with [Remotion](https://www.remotion.dev)
(React + TypeScript → MP4), with a score synthesized in code and an offline-TTS voiceover.
No screen recording, no footage: every frame, including the macOS bar,
Chrome window, job form, resume, Coldstart card and email draft, is drawn with React, CSS and SVG.

| Composition | Size | Use |
|---|---|---|
| `Vertical` | 1080×1920, 30 fps | Reels / Shorts / LinkedIn mobile |
| `Landscape` | 1920×1080, 30 fps | X / LinkedIn desktop (re-laid out side by side, not cropped) |

## Quick start

```bash
cd video
npm install
npm run studio     # opens Remotion Studio (live preview, scrub, per-scene compositions)
npm run render     # writes out/aria-vertical.mp4 and out/aria-landscape.mp4
npm run stills     # writes one PNG per scene per format to out/stills/
```

`npm run stills -- Vertical` renders only one format.

## Edit text and numbers

Everything on screen lives in **`src/config.ts`**:

- `TEXT`: every string, per scene. `\n` forces a line break in headlines.
- `TEXT.numbers`: the four stats of scene 8. The first number in each value counts up from 0.
- `PLACEHOLDERS`:
  - `JOBS_TRACKED` is a visible amber `{{JOBS_TRACKED}}` chip in the hook. Set it to a number
    (for example `180`) and the counter counts up to it instead.
  - `GATELLM_LINK` / `COLDSTART_LINK` are `null` (hidden). Set them to a URL once those repos are
    public and they show up on the end card.
- `PERSON`, `COMPANIES`: the fictional data inside the recreated UIs.
- `COLORS`: palette. The coral accent `#D97757` is used only for the diagram lines and "I do."

Any string written as `{{SOMETHING}}` renders as an amber dashed placeholder, so it can't slip into
a final cut unnoticed.

## Music and voiceover

Both ship with the project and play automatically.

- **Music**: `public/music.mp3` is synthesized by `scripts/make-music.py` (numpy, no samples), so it's
  royalty-free and "made entirely in code" too. It follows the scene map: ticking hook, a boom on
  "I got tired of this.", a chord build with kick and arpeggio, one hit per number, then silence
  for "It never submits." and a single warm chord on "I do." Want a different track? Drop any mp3 at
  `public/music.mp3`.
- **Voiceover**: the on-screen words, read aloud. Lines and timing live in `voiceover.json` (`at` is
  the frame inside the scene). The clips in `public/vo/` come from `scripts/make-voice.py`, which runs
  [Kokoro](https://github.com/thewh1teagle/kokoro-onnx), an open-weights TTS model, **offline**
  (no API calls). It's still an AI voice. To use your own, record each line as
  `public/vo/<id>.mp3` with the same names. The music ducks under the voice automatically
  (`VOICEOVER.duckTo` in config).

```bash
pip install numpy soundfile kokoro-onnx
python scripts/make-music.py
python scripts/make-voice.py --models /path/to/kokoro-models   # kokoro-v1.0.onnx + voices-v1.0.bin
python scripts/make-voice.py --models ... --voice am_michael     # try another Kokoro voice
```

The voice script trims silence, levels each clip to -14 LUFS, and writes `public/vo/durations.json`,
which the video reads to time the ducking. The final mix comes out around -16 LUFS.

Optional UI sounds go in `public/sfx/` (`type.mp3`, `tick.mp3`, `whoosh.mp3`). Cues are listed in `SFX`.
Missing files are skipped.

## Retime to the beat

1. Scene lengths are in `SCENE_SECONDS`.
2. `BEATS` is a list of beat frames (default: a 120 BPM grid, one beat every 15 frames, matching the
   generated music). With `SNAP_CUTS_TO_BEATS = true`, every scene cut moves to the nearest beat.
   For another track, change `BPM` / `FIRST_BEAT`, or paste the exact cut frames into `BEATS`.
3. If you change scene lengths, update the section times in `scripts/make-music.py` and the `at`
   frames in `voiceover.json`, then re-run both scripts.

Scene 8's four numbers split their scene evenly, so each lands on every second beat.

## Structure

```
src/
  config.ts        all text, numbers, placeholders, timing, beats, colors, audio levels
  layout.ts        9:16 vs 16:9 switch, safe area (120 top / 320 bottom / 80 sides on 9:16)
  theme.ts         fonts (Inter + JetBrains Mono, bundled in public/fonts)
  Video.tsx        hard cuts between scenes + optional music/SFX
  Root.tsx         Vertical, Landscape, plus every scene as its own composition
  scenes/          S01Hook … S10EndCard, one component per scene
  ui/              MacTopBar, ChromeWindow, JobForm, ResumePage, EmailCard, SystemDiagram, motion helpers
scripts/stills.mjs one PNG per scene
scripts/make-music.py / make-voice.py   generate public/music.mp3 and public/vo/*.mp3
voiceover.json     voiceover lines, timing, TTS voice
```

## Rules the code keeps

- **The Submit button never animates.** `SubmitButton` in `src/ui/JobForm.tsx` takes no progress
  props. Fields fill and tick green around it; the toast says "Filled. Waiting for you."
- **Email is a Draft.** The `Draft` pill is always visible, and the Send button is grey and static.
- **Only fictional data:** Jane Doe, `jane@example.com`, a 555-01xx phone number, `*.example` domains,
  Acme Robotics / Northwind / Globex / Initech / Umbrella Labs, lettered tiles instead of logos.
- **Only verified claims:** 2,035 tests, < 2 ms, 5 / 5 sites, 14× cheaper, ATS 98 median, on-device
  speech-to-text, never submits on its own.

## Fonts

Inter and JetBrains Mono (both SIL Open Font License) are bundled as variable fonts in
`public/fonts/`. They're the same files `@remotion/google-fonts` downloads, but kept locally so
renders work offline and look the same on every machine.

## Browser

Remotion downloads its own headless Chrome the first time you render. To use a local
Chrome/Chromium instead: `REMOTION_BROWSER=/path/to/chrome npm run render`.
