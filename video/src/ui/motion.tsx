import React from 'react';
import {Easing, interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {COLORS} from '../config';
import {MONO} from '../theme';

/** High-damping spring: settles fast, never bounces. */
export const useEnter = (delay = 0, durationInFrames = 12) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  return spring({frame: frame - delay, fps, config: {damping: 200, mass: 0.6}, durationInFrames});
};

export const enterAt = (frame: number, fps: number, delay = 0, durationInFrames = 12) =>
  spring({frame: frame - delay, fps, config: {damping: 200, mass: 0.6}, durationInFrames});

export const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;

export const ease = (frame: number, input: [number, number], output: [number, number]) =>
  interpolate(frame, input, output, {...clamp, easing: Easing.bezier(0.22, 1, 0.36, 1)});

/**
 * Word-by-word mask-up reveal. Each word slides up out of its own clip box.
 * `stagger` is frames between words; each word takes ~10 frames.
 */
export const Reveal: React.FC<{
  text: string;
  delay?: number;
  stagger?: number;
  style?: React.CSSProperties;
  wordStyle?: (i: number) => React.CSSProperties;
}> = ({text, delay = 0, stagger = 3, style, wordStyle}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  // "\n" in the text forces a line break.
  const words = text.replace(/\n/g, ' \n ').split(' ').filter(Boolean);
  let k = -1;
  return (
    <span style={{display: 'inline', ...style}}>
      {words.map((w, i) => {
        if (w === '\n') return <br key={i} />;
        k++;
        const p = enterAt(frame, fps, delay + k * stagger, 10);
        return (
          <span
            key={i}
            style={{
              display: 'inline-block',
              overflow: 'hidden',
              verticalAlign: 'bottom',
              paddingBottom: '0.12em',
              marginBottom: '-0.12em',
              marginRight: i < words.length - 1 && words[i + 1] !== '\n' ? '0.25em' : 0,
            }}
          >
            <span
              style={{
                display: 'inline-block',
                transform: `translateY(${(1 - p) * 105}%)`,
                ...(wordStyle ? wordStyle(k) : {}),
              }}
            >
              {w}
            </span>
          </span>
        );
      })}
    </span>
  );
};

/** Words that appear one at a time like a live transcript (no slide). */
export const Transcript: React.FC<{
  text: string;
  delay?: number;
  perWord?: number;
  style?: React.CSSProperties;
}> = ({text, delay = 0, perWord = 6, style}) => {
  const frame = useCurrentFrame();
  const words = text.split(' ');
  return (
    <span style={style}>
      {words.map((w, i) => {
        const t = frame - delay - i * perWord;
        const o = interpolate(t, [0, 5], [0, 1], clamp);
        const blur = interpolate(t, [0, 6], [8, 0], clamp);
        const y = interpolate(t, [0, 8], [14, 0], {...clamp, easing: Easing.out(Easing.cubic)});
        return (
          <span
            key={i}
            style={{
              display: 'inline-block',
              opacity: o,
              filter: `blur(${blur}px)`,
              transform: `translateY(${y}px)`,
              marginRight: '0.24em',
            }}
          >
            {w}
          </span>
        );
      })}
    </span>
  );
};

/** Slow camera push used on UI shots. */
export const CameraPush: React.FC<{
  children: React.ReactNode;
  from?: number;
  to?: number;
  origin?: string;
}> = ({children, from = 1, to = 1.05, origin = '50% 50%'}) => {
  const frame = useCurrentFrame();
  const {durationInFrames} = useVideoConfig();
  const s = interpolate(frame, [0, durationInFrames], [from, to], clamp);
  return (
    <div style={{position: 'absolute', inset: 0, transform: `scale(${s})`, transformOrigin: origin}}>
      {children}
    </div>
  );
};

/** Green circular check that draws itself in. */
export const Check: React.FC<{progress: number; size?: number; color?: string}> = ({
  progress,
  size = 28,
  color = COLORS.green,
}) => {
  const p = Math.max(0, Math.min(1, progress));
  const path = 'M7 14.5 L12 19 L21 9.5';
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 28 28"
      style={{transform: `scale(${0.6 + 0.4 * p})`, opacity: p > 0 ? 1 : 0, flexShrink: 0}}
    >
      <circle cx={14} cy={14} r={13} fill={color} opacity={Math.min(1, p * 2)} />
      <path
        d={path}
        fill="none"
        stroke="#fff"
        strokeWidth={3}
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeDasharray={22}
        strokeDashoffset={22 * (1 - interpolate(p, [0.3, 1], [0, 1], clamp))}
      />
    </svg>
  );
};

/** Visible placeholder chip: amber, dashed, monospace. Impossible to miss. */
export const Placeholder: React.FC<{text: string; size?: number; style?: React.CSSProperties}> = ({
  text,
  size = 28,
  style,
}) => (
  <span
    style={{
      fontFamily: MONO,
      fontSize: size,
      fontWeight: 700,
      color: COLORS.placeholder,
      border: `2px dashed ${COLORS.placeholder}`,
      borderRadius: size * 0.3,
      padding: `${size * 0.1}px ${size * 0.35}px`,
      background: 'rgba(255,176,32,0.08)',
      letterSpacing: 0,
      whiteSpace: 'nowrap',
      ...style,
    }}
  >
    {text}
  </span>
);

export const isPlaceholder = (v: unknown): v is string =>
  typeof v === 'string' && /^\{\{.*\}\}$/.test(v);

/** Typing helper: how many characters of `text` are visible at progress 0..1. */
export const typed = (text: string, progress: number) =>
  text.slice(0, Math.round(Math.max(0, Math.min(1, progress)) * text.length));

export const Caret: React.FC<{height: number; color?: string; blink?: boolean}> = ({
  height,
  color = '#0A84FF',
  blink = true,
}) => {
  const frame = useCurrentFrame();
  const on = !blink || Math.floor(frame / 8) % 2 === 0;
  return (
    <span
      style={{
        display: 'inline-block',
        width: Math.max(2, height * 0.08),
        height,
        background: color,
        marginLeft: 2,
        verticalAlign: 'middle',
        opacity: on ? 1 : 0,
        borderRadius: 1,
      }}
    />
  );
};
