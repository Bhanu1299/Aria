import React from 'react';
import {AbsoluteFill, interpolate, useCurrentFrame} from 'remotion';
import {COLORS, TEXT} from '../config';
import {useLayout} from '../layout';
import {headline} from '../theme';
import {clamp, Reveal} from '../ui/motion';
import {formWidth, HandForm} from './S01Hook';

/** The last form breaks into strips and falls away; then the line. */
export const S02Tired: React.FC = () => {
  const frame = useCurrentFrame();
  const L = useLayout();
  const width = formWidth(L.vertical);
  const strips = 5;
  const exit = interpolate(frame, [0, 16], [0, 1], {...clamp, easing: (t) => t * t});

  return (
    <AbsoluteFill style={{background: COLORS.bg}}>
      {exit < 1 ? (
        <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center', paddingTop: L.safe.top, paddingBottom: L.safe.bottom}}>
          <div style={{position: 'relative', transform: 'scale(1.07)'}}>
            {Array.from({length: strips}).map((_, i) => {
              const dir = i % 2 === 0 ? -1 : 1;
              const top = (i / strips) * 100;
              const bottom = 100 - ((i + 1) / strips) * 100;
              return (
                <div
                  key={i}
                  style={{
                    position: i === 0 ? 'relative' : 'absolute',
                    inset: i === 0 ? undefined : 0,
                    clipPath: `inset(${top}% 0 ${bottom}% 0 round 30px)`,
                    transform: `translate(${dir * exit * (160 + i * 50)}px, ${exit * (300 + i * 140)}px) rotate(${dir * exit * (4 + i * 2)}deg)`,
                    opacity: 1 - exit,
                    filter: `blur(${exit * 10}px)`,
                  }}
                >
                  <div style={{borderRadius: 30, overflow: 'hidden'}}>
                    <HandForm cycle={2} local={99} duration={20} width={width} />
                  </div>
                </div>
              );
            })}
          </div>
        </AbsoluteFill>
      ) : null}
      <AbsoluteFill
        style={{
          alignItems: 'center',
          justifyContent: 'center',
          padding: `${L.safe.top}px ${L.safe.right}px ${L.safe.bottom}px ${L.safe.left}px`,
        }}
      >
        <div style={{...headline(L.h1), color: COLORS.text, textAlign: 'center', maxWidth: L.box.w}}>
          <Reveal text={TEXT.tired.headline} delay={10} stagger={4} />
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
