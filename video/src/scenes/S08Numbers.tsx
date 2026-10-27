import React from 'react';
import {AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig} from 'remotion';
import {COLORS, TEXT} from '../config';
import {useLayout} from '../layout';
import {FONT, headline} from '../theme';
import {clamp, ease, enterAt} from '../ui/motion';

/** Count the first number in a string up from zero, keeping its formatting. */
const countUp = (value: string, p: number) =>
  value.replace(/\d[\d,]*/, (m) => {
    const n = Number(m.replace(/,/g, ''));
    const cur = Math.round(n * p);
    return m.includes(',') ? cur.toLocaleString('en-US') : String(cur);
  });

/** One huge number per beat, hard cuts between them. */
export const S08Numbers: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps, durationInFrames} = useVideoConfig();
  const L = useLayout();
  const items = TEXT.numbers;
  const per = durationInFrames / items.length;
  const i = Math.min(items.length - 1, Math.floor(frame / per));
  const local = frame - i * per;
  const item = items[i];

  const p = enterAt(local, fps, 0, 10);
  const count = ease(local, [0, 12], [0, 1]);
  const label = enterAt(local, fps, 5, 10);
  const size = L.vertical ? 250 : 270;

  return (
    <AbsoluteFill
      style={{
        background: COLORS.bg,
        fontFamily: FONT,
        alignItems: 'center',
        justifyContent: 'center',
        padding: `${L.safe.top}px ${L.safe.right}px ${L.safe.bottom}px ${L.safe.left}px`,
      }}
    >
      <div style={{textAlign: 'center'}}>
        <div
          style={{
            ...headline(size),
            fontWeight: 800,
            letterSpacing: '-0.05em',
            color: COLORS.text,
            fontVariantNumeric: 'tabular-nums',
            whiteSpace: 'nowrap',
            transform: `scale(${interpolate(p, [0, 1], [1.12, 1])})`,
            filter: `blur(${interpolate(p, [0, 1], [14, 0], clamp)}px)`,
            opacity: p,
          }}
        >
          {countUp(item.value, count)}
        </div>
        <div
          style={{
            fontSize: L.vertical ? 46 : 44,
            fontWeight: 500,
            color: COLORS.secondary,
            letterSpacing: '-0.01em',
            marginTop: 26,
            opacity: label,
            transform: `translateY(${(1 - label) * 16}px)`,
          }}
        >
          {item.label}
        </div>
      </div>
      {/* Beat dots */}
      <div style={{position: 'absolute', bottom: L.safe.bottom + 10, display: 'flex', gap: 14}}>
        {items.map((_, k) => (
          <div
            key={k}
            style={{
              width: k === i ? 34 : 10,
              height: 10,
              borderRadius: 5,
              background: k === i ? COLORS.text : COLORS.surface2,
            }}
          />
        ))}
      </div>
    </AbsoluteFill>
  );
};
