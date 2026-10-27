import React from 'react';
import {AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig} from 'remotion';
import {COLORS, COMPANIES, POSTINGS_BY_FIT, TEXT} from '../config';
import {barGeom, useLayout} from '../layout';
import {FONT, headline} from '../theme';
import {ChromeWindow, Tab} from '../ui/ChromeWindow';
import {JobForm} from '../ui/JobForm';
import {AriaGlyph, GlyphState, glyphCenter, MacTopBar} from '../ui/MacTopBar';
import {CameraPush, Check, clamp, ease, enterAt, Reveal, Transcript} from '../ui/motion';

// Beats inside this 10 s scene (frames).
const T = {
  listen: 6,
  words: 12,
  think: 62,
  move: 92,
  window: 104,
  tabs: [114, 128, 140, 150, 158],
  fill: 132,
  perField: 13,
  why: 200,
  whyLen: 36,
  done: 246,
  caption1: 188,
  caption2: 218,
  toast: 252,
};

export const S05Voice: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const L = useLayout();
  const v = L.vertical;
  const bar = barGeom(L);
  const t = TEXT.voice;

  const glyph: GlyphState =
    frame < T.listen ? 'idle' : frame < T.think ? 'listening' : frame < T.done ? 'thinking' : 'done';

  // Spoken caption: big and centred, then tucks up out of the way.
  const move = ease(frame, [T.move, T.move + 16], [0, 1]);
  const spokenSize = v ? interpolate(move, [0, 1], [104, 52]) : interpolate(move, [0, 1], [96, 64]);
  const spokenTop = v
    ? interpolate(move, [0, 1], [L.safe.top + 520, bar.top + bar.height + 44])
    : interpolate(move, [0, 1], [330, 150]);
  const spokenLeft = v ? L.safe.left : L.safe.left;
  const spokenWidth = v ? L.box.w : 700;

  // Browser window
  const winIn = enterAt(frame, fps, T.window, 18);
  const win = v
    ? {x: L.safe.left, y: 430, w: L.box.w, h: 960}
    : {x: 860, y: 110, w: 960, h: 900};
  const formW = v ? 870 : 860;

  // Tabs open best fit first, the order the feed was sorted in.
  const tabs: Tab[] = POSTINGS_BY_FIT.map((p, i) => {
    const c = COMPANIES[p.company];
    return {title: c.name, color: c.color, letter: c.letter, appearAt: T.tabs[i]};
  });
  const front = COMPANIES[POSTINGS_BY_FIT[0].company];
  // The pipeline's three parts run side by side; each ticks when it's done.
  const stepsDone = [T.done, 214, 236];

  const fields = t.fields;
  const progress = fields.map((_, i) =>
    interpolate(frame, [T.fill + i * T.perField, T.fill + i * T.perField + T.perField - 4], [0, 1], clamp),
  );
  const checks = fields.map((_, i) =>
    interpolate(frame, [T.fill + (i + 1) * T.perField - 4, T.fill + (i + 1) * T.perField + 4], [0, 1], clamp),
  );
  const whyProgress = interpolate(frame, [T.why, T.why + T.whyLen], [0, 1], clamp);
  const whyCheck = interpolate(frame, [T.why + T.whyLen + 2, T.why + T.whyLen + 10], [0, 1], clamp);
  const active =
    frame < T.fill ? -1 : frame < T.why ? Math.min(fields.length - 1, Math.floor((frame - T.fill) / T.perField)) : frame < T.why + T.whyLen ? fields.length : -1;

  const toast = enterAt(frame, fps, T.toast, 14);
  const gc = glyphCenter(L.W, bar.height, bar.insetX);

  return (
    <AbsoluteFill style={{background: COLORS.bg, fontFamily: FONT}}>
      <CameraPush from={1} to={1.05} origin={v ? '50% 60%' : '70% 50%'}>
        {/* Spoken line, transcribed on-device */}
        <div
          style={{
            position: 'absolute',
            left: spokenLeft,
            top: spokenTop,
            width: spokenWidth,
            textAlign: v ? 'center' : 'left',
            // On 9:16 the notification needs this space later.
            opacity: v ? interpolate(frame, [T.toast - 10, T.toast], [1, 0], clamp) : 1,
          }}
        >
          <div style={{...headline(spokenSize), color: COLORS.text, fontWeight: 700}}>
            <Transcript text={`“${t.spoken}”`} delay={T.words} perWord={6} />
          </div>
          <div style={{position: 'relative', marginTop: interpolate(move, [0, 1], [34, 16]), height: v ? 56 : 50}}>
            <div
              style={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                justifyContent: v ? 'center' : 'flex-start',
                alignItems: 'center',
                gap: 12,
                fontSize: v ? 28 : 24,
                color: COLORS.secondary,
                fontWeight: 500,
                opacity: interpolate(frame, [T.words + 10, T.words + 20], [0, 1], clamp) * (1 - move),
              }}
            >
              <AriaGlyph state={frame < T.think ? 'listening' : 'idle'} size={v ? 34 : 30} color={COLORS.secondary} />
              {t.sttNote}
            </div>
            <div
              style={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                justifyContent: v ? 'center' : 'flex-start',
                flexWrap: v ? 'nowrap' : 'wrap',
                alignItems: 'center',
                gap: v ? 12 : 10,
                opacity: move,
              }}
            >
              {t.steps.map((step, i) => {
                const done = interpolate(frame, [stepsDone[i], stepsDone[i] + 8], [0, 1], clamp);
                return (
                  <div
                    key={step}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 10,
                      fontSize: v ? 24 : 22,
                      fontWeight: 600,
                      color: done > 0.5 ? COLORS.text : COLORS.secondary,
                      background: COLORS.surface,
                      border: `1.5px solid ${done > 0.5 ? 'rgba(48,209,88,0.5)' : 'rgba(255,255,255,0.1)'}`,
                      borderRadius: 999,
                      padding: v ? '9px 16px 9px 12px' : '7px 14px 7px 10px',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {done > 0 ? <Check progress={done} size={v ? 26 : 24} /> : <AriaGlyph state="thinking" size={v ? 26 : 24} color={COLORS.secondary} />}
                    {step}
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Background browser */}
        <div
          style={{
            position: 'absolute',
            left: win.x,
            top: win.y + (1 - winIn) * 260,
            opacity: winIn,
            filter: `saturate(${0.85})`,
          }}
        >
          <ChromeWindow width={win.w} height={win.h} tabs={tabs} active={0} url={t.url} dim={0.16} chip={t.backgroundChip}>
            <div style={{position: 'absolute', left: (win.w - formW) / 2, top: 0}}>
              <JobForm
                width={formW}
                company={front}
                title={TEXT.hook.formTitle}
                role={POSTINGS_BY_FIT[0].role}
                fields={fields}
                progress={progress}
                checks={checks}
                active={active}
                whyLabel={t.whyLabel}
                whyLines={t.whyLines}
                whyProgress={whyProgress}
                whyCheck={whyCheck}
                submit={t.submit}
                compact
              />
            </div>
          </ChromeWindow>
        </div>

        {/* Caption */}
        <div
          style={{
            position: 'absolute',
            left: L.safe.left,
            width: v ? L.box.w : 700,
            top: v ? win.y + win.h + 40 : 560,
            textAlign: v ? 'center' : 'left',
            ...headline(v ? 50 : 56),
            fontWeight: 600,
          }}
        >
          <div style={{color: COLORS.text}}>
            <Reveal text={t.caption[0]} delay={T.caption1} stagger={2} />
          </div>
          <div style={{color: COLORS.secondary, marginTop: 8}}>
            <Reveal text={t.caption[1]} delay={T.caption2} stagger={2} />
          </div>
        </div>
      </CameraPush>

      {/* Menu bar stays put: the screen never moves. */}
      <div style={{position: 'absolute', top: bar.top, left: 0}}>
        <MacTopBar
          width={L.W}
          height={bar.height}
          insetX={bar.insetX}
          glyph={glyph}
          glyphHighlight={frame >= T.listen && frame < T.done ? 1 : 0.3}
        />
      </div>

      {/* Notification: filled, waiting */}
      <div
        style={{
          position: 'absolute',
          top: bar.top + bar.height + 14,
          left: Math.min(gc.x - 40, L.W - L.safe.right - (v ? 470 : 420)),
          opacity: toast,
          transform: `translateY(${(1 - toast) * -16}px)`,
          background: 'rgba(44,44,46,0.92)',
          border: '1px solid rgba(255,255,255,0.12)',
          borderRadius: 18,
          padding: v ? '16px 22px' : '12px 18px',
          display: 'flex',
          alignItems: 'center',
          gap: 14,
          fontSize: v ? 28 : 24,
          fontWeight: 600,
          color: COLORS.text,
          boxShadow: '0 20px 50px rgba(0,0,0,0.6)',
          whiteSpace: 'nowrap',
        }}
      >
        <AriaGlyph state="done" size={v ? 30 : 26} />
        {t.waiting}
      </div>
    </AbsoluteFill>
  );
};
