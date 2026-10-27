import React from 'react';
import {AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig} from 'remotion';
import {COLORS, TEXT} from '../config';
import {barGeom, useLayout} from '../layout';
import {FONT, headline} from '../theme';
import {AriaGlyph, glyphCenter, MacTopBar} from '../ui/MacTopBar';
import {clamp, ease, enterAt, Reveal} from '../ui/motion';

/** Wordmark reveal; the big glyph shrinks and flies up into the menu bar. */
export const S03Aria: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const L = useLayout();
  const bar = barGeom(L);
  const t = TEXT.aria;

  const barIn = enterAt(frame, fps, 0, 14);
  const glyphIn = enterAt(frame, fps, 2, 14);
  const fly = ease(frame, [46, 64], [0, 1]);
  const landed = frame >= 64;
  const highlight = interpolate(frame, [64, 70, 88], [0, 1, 0.35], clamp);
  const sub = enterAt(frame, fps, 30, 14);

  const cx = L.W / 2;
  const cy = L.safe.top + L.box.h / 2;
  const v = L.vertical;
  const big = v ? 170 : 140;
  const g0 = {x: cx, y: cy - (v ? 270 : 230)};
  const gc = glyphCenter(L.W, bar.height, bar.insetX);
  const g1 = {x: gc.x, y: bar.top + gc.y};
  const small = (30 * bar.height) / 56 + 10;
  const size = interpolate(fly, [0, 1], [big, small]);
  const gx = interpolate(fly, [0, 1], [g0.x, g1.x]);
  // Slight arc on the way up.
  const gy = interpolate(fly, [0, 1], [g0.y, g1.y]) - Math.sin(fly * Math.PI) * 60;

  return (
    <AbsoluteFill style={{background: COLORS.bg, fontFamily: FONT}}>
      <div style={{position: 'absolute', top: bar.top - (1 - barIn) * (bar.height + bar.top), left: 0}}>
        <MacTopBar width={L.W} height={bar.height} insetX={bar.insetX} glyph="idle" glyphOpacity={landed ? 1 : 0} glyphHighlight={highlight} />
      </div>

      {!landed ? (
        <div
          style={{
            position: 'absolute',
            left: gx - size / 2,
            top: gy - size / 2,
            width: size,
            height: size,
            borderRadius: '50%',
            background: `rgba(255,255,255,${0.1 * (1 - fly)})`,
            border: `1px solid rgba(255,255,255,${0.16 * (1 - fly)})`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            opacity: glyphIn,
            transform: `scale(${0.7 + 0.3 * glyphIn})`,
            boxShadow: `0 0 ${80 * (1 - fly)}px rgba(255,255,255,${0.08 * (1 - fly)})`,
          }}
        >
          <AriaGlyph state="idle" size={size * 0.62} />
        </div>
      ) : null}

      <div style={{position: 'absolute', left: L.safe.left, right: L.safe.right, top: cy - (v ? 140 : 120), textAlign: 'center'}}>
        <div style={{...headline(L.h2), color: COLORS.secondary, fontWeight: 600}}>
          <Reveal text={t.lead} delay={8} stagger={3} />
        </div>
        <div style={{...headline(v ? 280 : 230), color: COLORS.text, fontWeight: 800, marginTop: v ? 0 : -6}}>
          <Reveal text={`${t.wordmark}.`} delay={16} />
        </div>
        <div
          style={{
            fontSize: v ? 44 : 38,
            color: COLORS.secondary,
            fontWeight: 500,
            letterSpacing: '-0.01em',
            marginTop: v ? 22 : 14,
            opacity: sub,
            transform: `translateY(${(1 - sub) * 16}px)`,
          }}
        >
          {t.subtitle}
        </div>
      </div>
    </AbsoluteFill>
  );
};
