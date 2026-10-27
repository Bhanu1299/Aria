# "I stopped applying to jobs by hand": the video

A 47.5-second motion-graphics video, built entirely in code with [Remotion](https://www.remotion.dev)
(React + TypeScript → MP4). No screen recording, no footage: every frame, including the macOS bar,
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

## Retime to the music

1. Drop your royalty-free track at `public/music.mp3`. It's picked up automatically and fades
   out over the last 1.5 s (`AUDIO` in config). Optional UI sounds go in `public/sfx/`
   (`type.mp3`, `tick.mp3`, `whoosh.mp3`). Cues are listed in `SFX`. Missing files are skipped.
2. Scene lengths are in `SCENE_SECONDS`.
3. `BEATS` is a list of beat frames (default: a 120 BPM grid, one beat every 15 frames). With
   `SNAP_CUTS_TO_BEATS = true`, every scene cut moves to the nearest beat. To sync:
   - change `BPM` / `FIRST_BEAT` if the track has a steady tempo, or
   - scrub the track in Studio, write down the frames where you want cuts, and paste them into
     `BEATS`.

Scene 8's four numbers split their scene evenly, so with a 4-second scene at 120 BPM each number
lands on every second beat.

> Note: the repo's root `.gitignore` ignores `*.mp3`, so your music stays local.

## Structure

```
src/
  config.ts        all text, numbers, placeholders, timing, beats, colors
  layout.ts        9:16 vs 16:9 switch, safe area (120 top / 320 bottom / 80 sides on 9:16)
  theme.ts         fonts (Inter + JetBrains Mono, bundled in public/fonts)
  Video.tsx        hard cuts between scenes + optional music/SFX
  Root.tsx         Vertical, Landscape, plus every scene as its own composition
  scenes/          S01Hook … S10EndCard, one component per scene
  ui/              MacTopBar, ChromeWindow, JobForm, ResumePage, EmailCard, SystemDiagram, motion helpers
scripts/stills.mjs one PNG per scene
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
