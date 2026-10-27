import React from 'react';
import {AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig} from 'remotion';
import {COLORS, COMPANIES, PLACEHOLDERS, TEXT} from '../config';
import {useLayout} from '../layout';
import {FONT, headline} from '../theme';
import {DeskScene, deskGeom} from '../ui/DeskScene';
import {JobForm} from '../ui/JobForm';
import {clamp, Reveal} from '../ui/motion';

/** Each pass through the form is shorter than the last. */
export const CYCLES = [26, 18, 14, 32];

/** The same form, typed by hand, again and again. */
export const HandForm: React.FC<{cycle: number; local: number; duration: number; width: number}> = ({
  cycle,
  local,
  duration,
  width,
}) => {
  const t = TEXT.hook;
  const n = t.fields.length + 1;
  const typing = duration * 0.9;
  const prog = (i: number) => interpolate(local, [(i / n) * typing, ((i + 1) / n) * typing], [0, 1], clamp);
  const progress = t.fields.map((_, i) => prog(i));
  const active = Math.min(n - 1, Math.floor((local / typing) * n));
  return (
    <JobForm
      width={width}
      company={COMPANIES[cycle % COMPANIES.length]}
      title={t.formTitle}
      role={t.roles[cycle % t.roles.length]}
      fields={t.fields}
      progress={progress}
      active={active}
      whyLabel={t.whyLabel}
      whyLines={[t.whyValue]}
      whyProgress={prog(n - 1)}
      submit={t.submit}
      compact
    />
  );
};

/** The monitor's contents: a plain browser page with the form on it. */
export const Screen: React.FC<{w: number; h: number; cycle: number; local: number; duration: number}> = ({
  w,
  h,
  cycle,
  local,
  duration,
}) => {
  const bar = 34;
  const formW = Math.min(w * 0.72, (h - bar - 16) / 0.86);
  return (
    <div style={{width: w, height: h, background: '#F2F2F7', position: 'relative'}}>
      <div style={{height: bar, background: '#E5E5EA', display: 'flex', alignItems: 'center', gap: 8, paddingLeft: 14}}>
        {['#FF5F57', '#FEBC2E', '#28C840'].map((c) => (
          <div key={c} style={{width: 11, height: 11, borderRadius: '50%', background: c}} />
        ))}
      </div>
      <div style={{position: 'absolute', top: bar + 8, left: (w - formW) / 2, borderRadius: 14, overflow: 'hidden'}}>
        <HandForm cycle={cycle} local={local} duration={duration} width={formW} />
      </div>
    </div>
  );
};

export const cycleAt = (frame: number) => {
  let start = 0;
  let cycle = 0;
  while (cycle < CYCLES.length - 1 && frame >= start + CYCLES[cycle]) {
    start += CYCLES[cycle];
    cycle++;
  }
  return {cycle, local: frame - start, duration: CYCLES[cycle]};
};

/** On the fourth reset I stop typing and give up. */
export const FACEPALM_AT = 60;
/** The line lands at ~1 s; the monitor clicks off at the end of the shot. */
const LINE_AT = 28;
const OFF_AT = 74;

/**
 * Me at the computer, filling the same form again, getting more fed up each time.
 * "I got tired of this." lands at ~1 s; then the hand goes up and the monitor clicks off.
 */
export const S01Hook: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const L = useLayout();
  const g = deskGeom(L);
  const c = cycleAt(frame);
  const {cycle, duration} = c;
  const local = cycle === CYCLES.length - 1 ? Math.min(c.local, 1) : c.local;

  const annoy = interpolate(frame, [0, CYCLES[0], CYCLES[0] + CYCLES[1] + CYCLES[2]], [0, 0.35, 1], clamp);
  const off = interpolate(frame, [OFF_AT, OFF_AT + 8], [0, 1], {...clamp, easing: (x) => x * x});
  const room = interpolate(frame, [OFF_AT + 6, OFF_AT + 16], [1, 0.0], clamp);
  const facepalm = interpolate(frame, [FACEPALM_AT, FACEPALM_AT + 12], [0, 1], {...clamp, easing: (x) => 1 - (1 - x) ** 3});
  // A frustrated head jerk on every reset, plus one sigh.
  const shake = cycle > 0 ? interpolate(local, [0, 2, 5, 9], [0, -5, 3, 0], clamp) * (0.6 + cycle * 0.3) : 0;
  const sighStart = CYCLES[0] + 4;
  const sigh = interpolate(frame, [sighStart, sighStart + 8, sighStart + 20], [0, 1, 0], clamp);
  const flicker = cycle > 0 ? interpolate(local, [0, 2, 5], [1, 0.75, 1], clamp) : 1;

  const target = PLACEHOLDERS.JOBS_TRACKED;
  const bump = cycle > 0 ? interpolate(local, [0, 3, 8], [1, 1.18, 1], clamp) : 1;
  const count = target !== null ? Math.round(interpolate(frame, [0, fps * 3], [0, target], clamp)) : null;

  return (
    <AbsoluteFill style={{background: COLORS.bg, fontFamily: FONT}}>
      <DeskScene
        L={L}
        annoy={annoy}
        facepalm={facepalm}
        shake={shake}
        sigh={sigh}
        glow={flicker * (1 - off)}
        screen={
          <div
            style={{
              width: '100%',
              height: '100%',
              transform: `scaleY(${interpolate(off, [0, 0.7, 1], [1, 0.02, 0.02])}) scaleX(${interpolate(off, [0.6, 1], [1, 0], clamp)})`,
              filter: `brightness(${1 + off * 2})`,
            }}
          >
            <Screen w={g.screen.w} h={g.screen.h} cycle={cycle} local={local} duration={duration} />
          </div>
        }
      />
      {/* the room goes dark after the monitor */}
      <AbsoluteFill style={{background: COLORS.bg, opacity: 1 - room}} />

      {/* The line, over the dark of my back so it never covers the screen */}
      <div
        style={{
          position: 'absolute',
          left: L.safe.left,
          right: L.safe.right,
          top: L.vertical ? 1330 : 850,
          textAlign: 'center',
          ...headline(L.vertical ? 124 : 76),
          color: COLORS.text,
          textShadow: '0 4px 40px rgba(0,0,0,0.9)',
        }}
      >
        <Reveal text={TEXT.tired.headline} delay={LINE_AT} stagger={4} />
      </div>

      {/* Counter, top-right inside the safe area. Hidden until JOBS_TRACKED is set. */}
      {count !== null ? (
        <div
          style={{
            position: 'absolute',
            top: L.safe.top + 8,
            right: L.safe.right,
            display: 'flex',
            alignItems: 'center',
            gap: 14,
            color: COLORS.secondary,
            fontSize: L.vertical ? 28 : 24,
            fontWeight: 600,
            transform: `scale(${bump})`,
            transformOrigin: '100% 50%',
          }}
        >
          <span>{TEXT.hook.counterLabel}</span>
          <span style={{color: COLORS.text, fontVariantNumeric: 'tabular-nums', fontWeight: 700, fontSize: L.vertical ? 40 : 34}}>
            {count}
          </span>
        </div>
      ) : null}
    </AbsoluteFill>
  );
};
