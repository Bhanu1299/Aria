import React from 'react';
import {AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig} from 'remotion';
import {COLORS, TEXT} from '../config';
import {useLayout} from '../layout';
import {FONT, headline, MONO} from '../theme';
import {EmailCard} from '../ui/EmailCard';
import {CameraPush, Check, clamp, ease, enterAt, Reveal} from '../ui/motion';

const T = {card: 12, flick: 26, perPattern: 5, lock: 54, email: 78, chip: 92, chipLand: 108, body: 110, caption: 128};

export const S07Coldstart: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const L = useLayout();
  const v = L.vertical;
  const t = TEXT.coldstart;

  const cardIn = enterAt(frame, fps, T.card, 16);
  const emailIn = enterAt(frame, fps, T.email, 18);
  const locked = frame >= T.lock;
  const lockP = interpolate(frame, [T.lock, T.lock + 10], [0, 1], clamp);
  const idx = Math.min(t.patterns.length - 1, Math.max(0, Math.floor((frame - T.flick) / T.perPattern)));
  const current = frame < T.flick ? '' : locked ? t.found : t.patterns[idx];
  const flickPhase = ((frame - T.flick) % T.perPattern) / T.perPattern;

  const box = v
    ? {card: {x: 80, y: 430, w: 920}, email: {x: 80, y: 790, w: 920}, caption: 1450}
    : {card: {x: 860, y: 110, w: 940}, email: {x: 900, y: 430, w: 860}, caption: 640};

  // The verified address flies from the Coldstart card into the draft's To field.
  const fly = ease(frame, [T.chip, T.chipLand], [0, 1]);
  const toFilled = frame >= T.chipLand;
  const bodyProgress = interpolate(frame, [T.body, T.body + 24], [0, 1], clamp);
  const emailS = box.email.w / 760;
  const fromY = box.card.y + 196;
  const toY = box.email.y + (1 - emailIn) * 140 + 62 * emailS + 16 * emailS;
  const fromX = box.card.x + 40;
  const toX = box.email.x + 28 * emailS + 96 * emailS;

  return (
    <AbsoluteFill style={{background: COLORS.bg, fontFamily: FONT}}>
      <CameraPush from={1} to={1.04} origin="50% 55%">
        {/* Coldstart card */}
        <div
          style={{
            position: 'absolute',
            left: box.card.x,
            top: box.card.y + (1 - cardIn) * 80,
            width: box.card.w,
            opacity: cardIn,
            background: COLORS.surface,
            borderRadius: 30,
            border: '1px solid rgba(255,255,255,0.1)',
            boxShadow: '0 30px 90px rgba(0,0,0,0.6)',
            padding: 36,
            boxSizing: 'border-box',
          }}
        >
          <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between'}}>
            <div style={{display: 'flex', alignItems: 'center', gap: 22}}>
              <div
                style={{
                  width: 76,
                  height: 76,
                  borderRadius: '50%',
                  background: 'linear-gradient(135deg, #3A3A3C, #636366)',
                  color: COLORS.text,
                  fontWeight: 700,
                  fontSize: 30,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                JD
              </div>
              <div>
                <div style={{fontSize: 36, fontWeight: 700, color: COLORS.text, letterSpacing: '-0.02em'}}>{t.personName}</div>
                <div style={{fontSize: 24, color: COLORS.secondary}}>{t.personRole}</div>
              </div>
            </div>
            <div style={{fontSize: 20, fontWeight: 700, color: COLORS.secondary, letterSpacing: '0.1em', textTransform: 'uppercase'}}>
              {t.appName}
            </div>
          </div>
          <div
            style={{
              marginTop: 28,
              height: 76,
              borderRadius: 18,
              background: COLORS.surface2,
              border: `2px solid ${locked ? `rgba(48,209,88,${0.7 * lockP})` : 'transparent'}`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '0 24px',
              fontFamily: MONO,
              fontSize: v ? 30 : 28,
              color: locked ? COLORS.text : COLORS.secondary,
              overflow: 'hidden',
            }}
          >
            <span
              style={{
                whiteSpace: 'nowrap',
                opacity: fly > 0 && !toFilled ? 0.35 : 1,
                transform: locked ? 'none' : `translateY(${interpolate(flickPhase, [0, 0.35], [-18, 0], clamp)}px)`,
                filter: locked ? 'none' : `blur(${interpolate(flickPhase, [0, 0.35], [3, 0], clamp)}px)`,
              }}
            >
              {current}
            </span>
            {locked ? (
              <span style={{display: 'flex', alignItems: 'center', gap: 10, fontFamily: FONT, fontSize: 24, fontWeight: 700, color: COLORS.green}}>
                <Check progress={lockP} size={32} />
                <span style={{opacity: lockP}}>{t.verified}</span>
              </span>
            ) : null}
          </div>
        </div>

        {/* Draft email */}
        <div style={{position: 'absolute', left: box.email.x, top: box.email.y + (1 - emailIn) * 140, opacity: emailIn}}>
          <EmailCard
            width={box.email.w}
            pill={t.draftPill}
            to={<span style={{opacity: toFilled ? 1 : 0}}>{t.to}</span>}
            subject={t.subject}
            bodyProgress={bodyProgress}
          />
        </div>

        {fly > 0 && !toFilled ? (
          <div
            style={{
              position: 'absolute',
              left: interpolate(fly, [0, 1], [fromX, toX]),
              top: interpolate(fly, [0, 1], [fromY, toY]) - Math.sin(fly * Math.PI) * 40,
              fontFamily: MONO,
              fontSize: interpolate(fly, [0, 1], [v ? 30 : 28, 21 * emailS]),
              color: COLORS.pageText,
              background: '#fff',
              padding: '6px 14px',
              borderRadius: 12,
              boxShadow: '0 16px 40px rgba(0,0,0,0.5)',
              whiteSpace: 'nowrap',
            }}
          >
            {t.found}
          </div>
        ) : null}
      </CameraPush>

      <div
        style={{
          position: 'absolute',
          left: L.safe.left,
          top: v ? L.safe.top + 30 : 160,
          width: v ? L.box.w : 640,
          ...headline(v ? 104 : 100),
          color: COLORS.text,
        }}
      >
        <Reveal text={t.headline} stagger={3} />
      </div>
      <div
        style={{
          position: 'absolute',
          left: L.safe.left,
          width: v ? L.box.w : 640,
          top: box.caption,
          textAlign: v ? 'center' : 'left',
          ...headline(v ? 58 : 56),
          fontWeight: 600,
          color: COLORS.secondary,
        }}
      >
        <Reveal text={t.caption} delay={T.caption} stagger={3} />
      </div>
    </AbsoluteFill>
  );
};
