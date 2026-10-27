import React from 'react';
import {AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig} from 'remotion';
import {COLORS, TEXT} from '../config';
import {useLayout} from '../layout';
import {FONT, headline, MONO} from '../theme';
import {clamp, ease, enterAt, Reveal} from '../ui/motion';

/** The 87-task stress test: what the same work cost each way. */
export const S14Cost: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const L = useLayout();
  const v = L.vertical;
  const t = TEXT.cost;
  const max = Math.max(...t.rows.map((r) => r.cost));

  const panel = v ? {x: L.safe.left, y: 720, w: L.box.w} : {x: 900, y: 260, w: 900};
  const labelIn = enterAt(frame, fps, 6, 12);
  const barMax = panel.w - 230;

  return (
    <AbsoluteFill style={{background: COLORS.bg, fontFamily: FONT}}>
      <div style={{position: 'absolute', left: L.safe.left, top: v ? L.safe.top + 40 : 340, width: v ? L.box.w : 760}}>
        <div style={{...headline(v ? 140 : 104), color: COLORS.text}}>
          <Reveal text={t.headline[0]} delay={2} stagger={4} />
        </div>
        <div style={{...headline(v ? 140 : 104), color: COLORS.green}}>
          <Reveal text={t.headline[1]} delay={58} stagger={4} />
        </div>
        <div
          style={{
            marginTop: v ? 20 : 16,
            fontSize: v ? 40 : 34,
            fontWeight: 600,
            color: COLORS.secondary,
            letterSpacing: '-0.01em',
            opacity: enterAt(frame, fps, 66, 12),
          }}
        >
          {t.vs}
        </div>
      </div>

      <div style={{position: 'absolute', left: panel.x, top: panel.y, width: panel.w}}>
        <div style={{fontSize: v ? 30 : 28, fontWeight: 600, color: COLORS.secondary, opacity: labelIn, marginBottom: 40}}>{t.test}</div>
        {t.rows.map((r, i) => {
          const start = 14 + i * 16;
          const grow = ease(frame, [start, start + 26], [0, 1]);
          const shown = r.cost * grow;
          const ours = i === t.rows.length - 1;
          const color = ours ? COLORS.green : COLORS.secondary;
          return (
            <div key={r.name} style={{marginBottom: 54, opacity: enterAt(frame, fps, start - 4, 10)}}>
              <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 16}}>
                <span style={{fontSize: v ? 40 : 36, fontWeight: 700, color: COLORS.text}}>{r.name}</span>
                <span style={{fontSize: v ? 26 : 24, color: COLORS.secondary}}>{r.passed}</span>
              </div>
              <div style={{display: 'flex', alignItems: 'center', gap: 22}}>
                <div
                  style={{
                    height: 56,
                    width: Math.max(14, (barMax * shown) / max),
                    borderRadius: 14,
                    background: color,
                    opacity: ours ? 1 : 0.55,
                    flexShrink: 0,
                  }}
                />
                <span
                  style={{
                    fontFamily: MONO,
                    fontWeight: 700,
                    fontSize: v ? 44 : 40,
                    color: ours ? COLORS.green : COLORS.text,
                    fontVariantNumeric: 'tabular-nums',
                    whiteSpace: 'nowrap',
                    opacity: interpolate(grow, [0, 0.2], [0, 1], clamp),
                  }}
                >
                  ${shown.toFixed(2)}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
