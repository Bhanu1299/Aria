import React from 'react';
import {AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig} from 'remotion';
import {COLORS, COMPANIES, PLACEHOLDERS, TEXT} from '../config';
import {useLayout} from '../layout';
import {FONT} from '../theme';
import {JobForm} from '../ui/JobForm';
import {clamp, isPlaceholder, Placeholder} from '../ui/motion';

/** Each pass through the form is shorter than the last. */
const CYCLES = [34, 22, 19];

export const formWidth = (vertical: boolean) => (vertical ? 860 : 760);

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
      company={COMPANIES[cycle % 3]}
      title={t.formTitle}
      role={t.roles[cycle % t.roles.length]}
      fields={t.fields}
      progress={progress}
      active={active}
      whyLabel={t.whyLabel}
      whyLines={[t.whyValue]}
      whyProgress={prog(n - 1)}
      submit={t.submit}
    />
  );
};

export const S01Hook: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const L = useLayout();
  const width = formWidth(L.vertical);

  let start = 0;
  let cycle = 0;
  while (cycle < CYCLES.length - 1 && frame >= start + CYCLES[cycle]) {
    start += CYCLES[cycle];
    cycle++;
  }
  const local = frame - start;
  const dur = CYCLES[cycle];

  // Reset jolt: the form snaps up and back when it clears.
  const jolt = cycle > 0 ? interpolate(local, [0, 2, 7], [0, -26, 0], clamp) : 0;
  const escalate = 1 + cycle * 0.035;

  const target = PLACEHOLDERS.JOBS_TRACKED;
  const bump = cycle > 0 ? interpolate(local, [0, 3, 8], [1, 1.18, 1], clamp) : 1;
  const count =
    typeof target === 'number' ? Math.round(interpolate(frame, [0, fps * 2.2], [0, target], clamp)) : null;

  return (
    <AbsoluteFill style={{background: COLORS.bg, fontFamily: FONT}}>
      <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center', paddingTop: L.safe.top, paddingBottom: L.safe.bottom}}>
        <div style={{position: 'relative', transform: `translateY(${jolt}px) scale(${escalate})`}}>
          {/* Stack of forms already filled by hand */}
          {Array.from({length: cycle}).map((_, i) => {
            const depth = cycle - i;
            return (
              <div
                key={i}
                style={{
                  position: 'absolute',
                  inset: 0,
                  borderRadius: 30,
                  background: '#D1D1D6',
                  transform: `translateY(${-depth * 30}px) scale(${1 - depth * 0.05})`,
                  opacity: 0.5 / depth,
                }}
              />
            );
          })}
          <div style={{borderRadius: 30, overflow: 'hidden', position: 'relative', boxShadow: '0 40px 120px rgba(0,0,0,0.8)'}}>
            <HandForm cycle={cycle} local={local} duration={dur} width={width} />
          </div>
        </div>
      </AbsoluteFill>

      {/* Counter, top-right inside the safe area */}
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
        {count !== null ? (
          <span style={{color: COLORS.text, fontVariantNumeric: 'tabular-nums', fontWeight: 700, fontSize: L.vertical ? 40 : 34}}>
            {count}
          </span>
        ) : isPlaceholder(target) ? (
          <Placeholder text={target} size={L.vertical ? 28 : 24} />
        ) : (
          <span style={{color: COLORS.text}}>{target}</span>
        )}
      </div>
    </AbsoluteFill>
  );
};
