import React from 'react';
import {Composition, Folder} from 'remotion';
import {FPS, LANDSCAPE, SceneId, TIMELINE, TOTAL_FRAMES, VERTICAL} from './config';
import {SCENES} from './scenes';
import {Video} from './Video';

/** One scene on its own. `still` is the frame `npm run stills` captures. */
const SceneOnly: React.FC<{scene: SceneId; still: number}> = ({scene}) => {
  const Scene = SCENES[scene];
  return <Scene />;
};

export const RemotionRoot: React.FC = () => (
  <>
    <Composition id="Vertical" component={Video} durationInFrames={TOTAL_FRAMES} fps={FPS} {...VERTICAL} />
    <Composition id="Landscape" component={Video} durationInFrames={TOTAL_FRAMES} fps={FPS} {...LANDSCAPE} />
    {/* Each scene on its own, for quick iteration in Studio and for stills. */}
    {(['Vertical', 'Landscape'] as const).map((format) => (
      <Folder key={format} name={`${format}-scenes`}>
        {TIMELINE.map((s, i) => (
          <Composition
            key={s.id}
            id={`${format[0]}${String(i + 1).padStart(2, '0')}-${s.id}`}
            component={SceneOnly}
            defaultProps={{scene: s.id, still: Math.min(s.still, s.duration - 1)}}
            durationInFrames={s.duration}
            fps={FPS}
            {...(format === 'Vertical' ? VERTICAL : LANDSCAPE)}
          />
        ))}
      </Folder>
    ))}
  </>
);
