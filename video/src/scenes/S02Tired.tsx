import React from 'react';
import {AbsoluteFill, interpolate, useCurrentFrame} from 'remotion';
import {COLORS, TEXT} from '../config';
import {useLayout} from '../layout';
import {headline} from '../theme';
import {DeskScene, deskGeom} from '../ui/DeskScene';
import {clamp, Reveal} from '../ui/motion';
import {CYCLES, Screen} from './S01Hook';

/** The monitor switches off on the last form; the room goes dark; then the line. */
export const S02Tired: React.FC = () => {
  const frame = useCurrentFrame();
  const L = useLayout();
  const g = deskGeom(L);
  const off = interpolate(frame, [0, 8], [0, 1], {...clamp, easing: (t) => t * t});
  const room = interpolate(frame, [6, 16], [1, 0], clamp);
  const last = CYCLES.length - 1;

  return (
    <AbsoluteFill style={{background: COLORS.bg}}>
      {room > 0 ? (
        <div style={{position: 'absolute', inset: 0, opacity: room}}>
          <DeskScene
            L={L}
            annoy={1}
            facepalm={1}
            glow={1 - off}
            screen={
              <div
                style={{
                  width: '100%',
                  height: '100%',
                  transform: `scaleY(${interpolate(off, [0, 0.7, 1], [1, 0.02, 0.02])}) scaleX(${interpolate(off, [0.6, 1], [1, 0], clamp)})`,
                  filter: `brightness(${1 + off * 2})`,
                }}
              >
                <Screen w={g.screen.w} h={g.screen.h} cycle={last} local={99} duration={CYCLES[last]} />
              </div>
            }
          />
        </div>
      ) : null}
      <AbsoluteFill
        style={{
          alignItems: 'center',
          justifyContent: 'center',
          padding: `${L.safe.top}px ${L.safe.right}px ${L.safe.bottom}px ${L.safe.left}px`,
        }}
      >
        <div style={{...headline(L.h1), color: COLORS.text, textAlign: 'center', maxWidth: L.box.w}}>
          <Reveal text={TEXT.tired.headline} delay={12} stagger={4} />
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
