import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {COLORS} from '../config';
import {FONT} from '../theme';
import {Check, clamp} from './motion';

export type GlyphState = 'idle' | 'listening' | 'thinking' | 'done';

/**
 * Aria's menu-bar glyph: a dot that becomes a small waveform.
 * idle: dot · listening: bars pulse with a ring · thinking: shimmer sweeps the bars · done: check.
 */
export const AriaGlyph: React.FC<{state: GlyphState; size?: number; color?: string}> = ({
  state,
  size = 30,
  color = COLORS.text,
}) => {
  const frame = useCurrentFrame();
  if (state === 'done') {
    return <Check progress={1} size={size} />;
  }
  const bars = 5;
  const bw = size * 0.12;
  const gap = size * 0.07;
  const total = bars * bw + (bars - 1) * gap;
  return (
    <div style={{position: 'relative', width: size, height: size}}>
      {state === 'listening' ? (
        <div
          style={{
            position: 'absolute',
            inset: -size * 0.35 * ((frame % 24) / 24),
            borderRadius: '50%',
            border: `${Math.max(1, size * 0.05)}px solid ${color}`,
            opacity: 0.5 * (1 - (frame % 24) / 24),
          }}
        />
      ) : null}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          borderRadius: '50%',
          background: state === 'idle' ? 'transparent' : 'rgba(255,255,255,0.12)',
        }}
      />
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{position: 'absolute', inset: 0}}>
        {Array.from({length: bars}).map((_, i) => {
          let h: number;
          if (state === 'idle') {
            h = size * [0.16, 0.3, 0.46, 0.3, 0.16][i];
          } else if (state === 'listening') {
            const wave = Math.sin(frame * 0.55 + i * 1.3) * 0.5 + 0.5;
            const env = [0.45, 0.75, 1, 0.75, 0.45][i];
            h = size * (0.14 + 0.46 * wave * env);
          } else {
            h = size * [0.22, 0.32, 0.4, 0.32, 0.22][i];
          }
          const x = (size - total) / 2 + i * (bw + gap);
          let op = 1;
          if (state === 'thinking') {
            const pos = ((frame % 20) / 20) * (bars + 2) - 1;
            op = 0.35 + 0.65 * Math.max(0, 1 - Math.abs(pos - i));
          }
          return (
            <rect
              key={i}
              x={x}
              y={(size - h) / 2}
              width={bw}
              height={h}
              rx={bw / 2}
              fill={color}
              opacity={op}
            />
          );
        })}
      </svg>
    </div>
  );
};

const Battery: React.FC<{s: number}> = ({s}) => (
  <svg width={34 * s} height={16 * s} viewBox="0 0 34 16">
    <rect x={1} y={1} width={28} height={14} rx={4} fill="none" stroke={COLORS.text} strokeOpacity={0.6} strokeWidth={1.5} />
    <rect x={3.5} y={3.5} width={20} height={9} rx={2} fill={COLORS.text} />
    <rect x={30.5} y={5.5} width={2.5} height={5} rx={1} fill={COLORS.text} fillOpacity={0.6} />
  </svg>
);

const Wifi: React.FC<{s: number}> = ({s}) => (
  <svg width={22 * s} height={16 * s} viewBox="0 0 22 16">
    {[14, 9.5, 5].map((r, i) => (
      <path
        key={r}
        d={`M ${11 - r * 0.72} ${14.5 - r * 0.69} A ${r} ${r} 0 0 1 ${11 + r * 0.72} ${14.5 - r * 0.69}`}
        fill="none"
        stroke={COLORS.text}
        strokeWidth={2.2}
        strokeLinecap="round"
        opacity={i === 0 ? 0.9 : 1}
      />
    ))}
    <circle cx={11} cy={14} r={1.6} fill={COLORS.text} />
  </svg>
);

const CLOCK_W = 160;

/** Where the Aria glyph's centre sits inside a bar of this size (for fly-in shots). */
export const glyphCenter = (width: number, height = 56, insetX = 24) => {
  const s = height / 56;
  // From the right: clock, gap, battery, gap, wifi, gap, then half the glyph box (30 + 2 * 5).
  return {x: width - insetX - (CLOCK_W + 22 + 34 + 22 + 22 + 22 + 20) * s, y: height / 2};
};

/** Thin translucent macOS-style menu bar. Text stays inside `insetX`. */
export const MacTopBar: React.FC<{
  width: number;
  height?: number;
  insetX?: number;
  glyph: GlyphState;
  glyphOpacity?: number;
  glyphHighlight?: number;
  clock?: string;
}> = ({width, height = 56, insetX = 24, glyph, glyphOpacity = 1, glyphHighlight = 0, clock = 'Tue 9:41 AM'}) => {
  const s = height / 56;
  return (
    <div
      style={{
        width,
        height,
        background: 'rgba(40,40,44,0.55)',
        backdropFilter: 'blur(30px)',
        borderBottom: '1px solid rgba(255,255,255,0.08)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: `0 ${insetX}px`,
        boxSizing: 'border-box',
        fontFamily: FONT,
        color: COLORS.text,
        fontSize: 22 * s,
        fontWeight: 500,
      }}
    >
      <div style={{display: 'flex', gap: 26 * s, alignItems: 'center'}}>
        <div style={{width: 18 * s, height: 18 * s, borderRadius: '50%', background: COLORS.text, opacity: 0.9}} />
        <span style={{fontWeight: 700}}>Finder</span>
        <span style={{opacity: 0.85}}>File</span>
        <span style={{opacity: 0.85}}>Edit</span>
        <span style={{opacity: 0.85}}>View</span>
      </div>
      <div style={{display: 'flex', gap: 22 * s, alignItems: 'center'}}>
        <div
          style={{
            opacity: glyphOpacity,
            transform: `scale(${interpolate(glyphOpacity, [0, 1], [0.4, 1], clamp)})`,
            padding: 5 * s,
            borderRadius: 9 * s,
            background: `rgba(255,255,255,${0.16 * glyphHighlight})`,
            boxShadow: glyphHighlight > 0 ? `0 0 ${30 * glyphHighlight * s}px rgba(255,255,255,${0.35 * glyphHighlight})` : 'none',
          }}
        >
          <AriaGlyph state={glyph} size={30 * s} />
        </div>
        <Wifi s={s} />
        <Battery s={s} />
        <span style={{whiteSpace: 'nowrap', width: CLOCK_W * s, textAlign: 'right'}}>{clock}</span>
      </div>
    </div>
  );
};
