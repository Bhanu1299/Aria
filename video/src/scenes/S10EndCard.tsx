import React from 'react';
import {AbsoluteFill, useCurrentFrame, useVideoConfig} from 'remotion';
import {COLORS, PLACEHOLDERS, TEXT} from '../config';
import {useLayout} from '../layout';
import {FONT, headline, MONO} from '../theme';
import {AriaGlyph} from '../ui/MacTopBar';
import {enterAt, isPlaceholder, Placeholder, Reveal} from '../ui/motion';

/** End card. Everything is in by ~frame 20, then holds. */
export const S10EndCard: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const L = useLayout();
  const v = L.vertical;
  const t = TEXT.endcard;
  const fade = (d: number) => {
    const p = enterAt(frame, fps, d, 12);
    return {opacity: p, transform: `translateY(${(1 - p) * 14}px)`};
  };
  const extra = [PLACEHOLDERS.GATELLM_LINK, PLACEHOLDERS.COLDSTART_LINK].filter((x): x is string => !!x);

  return (
    <AbsoluteFill
      style={{
        background: COLORS.bg,
        fontFamily: FONT,
        alignItems: 'center',
        justifyContent: 'center',
        padding: `${L.safe.top}px ${L.safe.right}px ${L.safe.bottom}px ${L.safe.left}px`,
        textAlign: 'center',
        color: COLORS.text,
      }}
    >
      <div style={{...fade(0), marginBottom: v ? 40 : 28}}>
        <AriaGlyph state="idle" size={v ? 72 : 60} />
      </div>
      <div style={{...headline(v ? 130 : 120)}}>
        <Reveal text={t.name} delay={2} stagger={4} />
      </div>
      <div
        style={{
          ...fade(8),
          marginTop: v ? 40 : 30,
          fontFamily: MONO,
          fontSize: v ? 44 : 42,
          fontWeight: 500,
          background: COLORS.surface,
          border: '1px solid rgba(255,255,255,0.12)',
          borderRadius: 18,
          padding: v ? '18px 30px' : '14px 28px',
          whiteSpace: 'nowrap',
        }}
      >
        {t.link}
      </div>
      {extra.map((x, i) => (
        <div key={x} style={{...fade(10 + i * 2), marginTop: 16, fontFamily: MONO, fontSize: v ? 32 : 30, color: COLORS.secondary}}>
          {isPlaceholder(x) ? <Placeholder text={x} size={v ? 28 : 26} /> : x}
        </div>
      ))}
      <div style={{...fade(12), marginTop: v ? 44 : 32, fontSize: v ? 40 : 38, fontWeight: 500, color: COLORS.text, letterSpacing: '-0.01em'}}>
        {t.roles}
      </div>
      <div style={{...fade(18), marginTop: v ? 70 : 48, fontSize: v ? 28 : 26, color: COLORS.secondary, letterSpacing: '0.02em'}}>
        {t.madeWith}
      </div>
    </AbsoluteFill>
  );
};
