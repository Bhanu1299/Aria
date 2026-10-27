import React from 'react';
import {AbsoluteFill, getStaticFiles, Html5Audio, interpolate, Sequence, staticFile} from 'remotion';
import {AUDIO, COLORS, SFX, TIMELINE, TOTAL_FRAMES} from './config';
import {SCENES} from './scenes';

const hasFile = (name: string) => getStaticFiles().some((f) => f.name === name);

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

      {hasFile(AUDIO.music) ? (
        <Html5Audio
          src={staticFile(AUDIO.music)}
          volume={(f) =>
            AUDIO.musicVolume *
            interpolate(f, [TOTAL_FRAMES - AUDIO.musicFadeOut, TOTAL_FRAMES], [1, 0], {
              extrapolateLeft: 'clamp',
              extrapolateRight: 'clamp',
            })
          }
        />
      ) : null}

      {SFX.filter((c) => hasFile(c.file)).map((c, i) => {
        const scene = TIMELINE.find((s) => s.id === c.scene)!;
        return (
          <Sequence key={i} from={scene.from + c.at} durationInFrames={45} layout="none">
            <Html5Audio src={staticFile(c.file)} volume={AUDIO.sfxVolume} />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};
