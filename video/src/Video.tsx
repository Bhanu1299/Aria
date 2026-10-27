import React from 'react';
import {AbsoluteFill, getStaticFiles, Html5Audio, interpolate, Sequence, staticFile} from 'remotion';
import {AUDIO, COLORS, FPS, SFX, TIMELINE, TOTAL_FRAMES, VOICEOVER} from './config';
import {SCENES} from './scenes';

const hasFile = (name: string) => getStaticFiles().some((f) => f.name === name);
const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;
const sceneStart = (id: string) => TIMELINE.find((s) => s.id === id)!.from;
const inTimeline = (id: string) => TIMELINE.some((s) => s.id === id);

const voLines = VOICEOVER.lines
  .filter((l) => hasFile(l.file) && inTimeline(l.scene))
  .map((l) => ({...l, from: sceneStart(l.scene) + l.at, frames: Math.ceil(l.seconds * FPS)}));

/** Music level at frame f: ducked under the voice, faded out at the end. */
const musicVolume = (f: number) => {
  const d = VOICEOVER.duckFrames;
  const duck = voLines.reduce((m, l) => {
    const inside = interpolate(f, [l.from - d, l.from, l.from + l.frames, l.from + l.frames + d * 2], [0, 1, 1, 0], clamp);
    return Math.max(m, inside);
  }, 0);
  const fade = interpolate(f, [TOTAL_FRAMES - AUDIO.musicFadeOut, TOTAL_FRAMES], [1, 0], clamp);
  return AUDIO.musicVolume * (1 - duck * (1 - VOICEOVER.duckTo)) * fade;
};

/** Hard cuts between scenes, on the beat. Sound is optional. */
export const Video: React.FC = () => {
  return (
    <AbsoluteFill style={{background: COLORS.bg}}>
      {TIMELINE.map((s) => {
        const Scene = SCENES[s.id];
        return (
          <Sequence key={s.id} name={s.id} from={s.from} durationInFrames={s.duration} premountFor={30}>
            <Scene />
          </Sequence>
        );
      })}

      {AUDIO.musicEnabled && hasFile(AUDIO.music) ? <Html5Audio src={staticFile(AUDIO.music)} volume={musicVolume} /> : null}

      {voLines.map((l) => (
        <Sequence key={l.id} name={`vo ${l.id}`} from={l.from} durationInFrames={l.frames + 10} layout="none">
          <Html5Audio src={staticFile(l.file)} volume={VOICEOVER.volume} />
        </Sequence>
      ))}

      {SFX.filter((c) => hasFile(c.file) && inTimeline(c.scene)).map((c, i) => (
        <Sequence key={i} from={sceneStart(c.scene) + c.at} durationInFrames={45} layout="none">
          <Html5Audio src={staticFile(c.file)} volume={AUDIO.sfxVolume} />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};
