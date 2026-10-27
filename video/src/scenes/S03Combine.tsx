import React from 'react';
import {AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig} from 'remotion';
import {COLORS, TEXT} from '../config';
import {useLayout} from '../layout';
import {FONT, headline} from '../theme';
import {AriaGlyph} from '../ui/MacTopBar';
import {clamp, enterAt, Reveal} from '../ui/motion';
import {ColdIcon, GateIcon} from '../ui/SystemDiagram';

const ICONS = [<AriaGlyph key="a" state="idle" size={44} />, <ColdIcon key="c" />, <GateIcon key="g" />];

/** Three projects I built, snapping together into one. */
export const S03Combine: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const L = useLayout();
  const v = L.vertical;
  const t = TEXT.combine;

  const tileW = v ? L.box.w : 760;
  const tileH = v ? 150 : 150;
  const gap = v ? 44 : 40;
  const x0 = v ? L.safe.left : 1040;
  const y0 = v ? 760 : 190;
  const sub = enterAt(frame, fps, 62, 14);

  return (
    <AbsoluteFill style={{background: COLORS.bg, fontFamily: FONT}}>
      <div style={{position: 'absolute', left: L.safe.left, top: v ? L.safe.top + 40 : 300, width: v ? L.box.w : 820}}>
        <div style={{...headline(v ? 112 : 100), color: COLORS.text}}>
          <Reveal text={t.headline} stagger={3} />
        </div>
      </div>

      {/* Connectors draw between the tiles once all three are in */}
      <svg width={L.W} height={L.H} style={{position: 'absolute', inset: 0}}>
        {[0, 1].map((i) => {
          const p = interpolate(frame, [44 + i * 6, 56 + i * 6], [0, 1], clamp);
          const x = x0 + 70;
          const ya = y0 + i * (tileH + gap) + tileH;
          return (
            <line
              key={i}
              x1={x}
              y1={ya}
              x2={x}
              y2={ya + gap * p}
              stroke={COLORS.accent}
              strokeWidth={4}
              strokeLinecap="round"
            />
          );
        })}
      </svg>

      {t.tiles.map((tile, i) => {
        const p = enterAt(frame, fps, 14 + i * 8, 14);
        const lit = interpolate(frame, [52 + i * 4, 60 + i * 4], [0, 1], clamp);
        return (
          <div
            key={tile.name}
            style={{
              position: 'absolute',
              left: x0 + (1 - p) * (i % 2 ? 120 : -120),
              top: y0 + i * (tileH + gap),
              width: tileW,
              height: tileH,
              opacity: p,
              borderRadius: 32,
              background: COLORS.surface,
              border: `1.5px solid rgba(217,119,87,${0.55 * lit})`,
              boxShadow: '0 24px 70px rgba(0,0,0,0.6)',
              display: 'flex',
              alignItems: 'center',
              gap: 28,
              padding: '0 32px',
              boxSizing: 'border-box',
            }}
          >
            <div
              style={{
                width: 76,
                height: 76,
                borderRadius: 22,
                background: COLORS.surface2,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0,
              }}
            >
              {ICONS[i]}
            </div>
            <div style={{display: 'flex', alignItems: 'baseline', gap: 18, flexWrap: 'wrap'}}>
              <span style={{fontSize: 52, fontWeight: 700, letterSpacing: '-0.02em', color: COLORS.text}}>{tile.name}</span>
              <span style={{fontSize: 32, fontWeight: 500, color: COLORS.secondary}}>{tile.note}</span>
            </div>
          </div>
        );
      })}

      <div
        style={{
          position: 'absolute',
          left: v ? L.safe.left : x0,
          width: v ? L.box.w : tileW,
          top: y0 + 3 * (tileH + gap) + 10,
          textAlign: v ? 'center' : 'left',
          ...headline(v ? 60 : 54),
          fontWeight: 600,
          color: COLORS.text,
          opacity: sub,
          transform: `translateY(${(1 - sub) * 16}px)`,
        }}
      >
        {t.subtitle}
      </div>
    </AbsoluteFill>
  );
};
