import React from 'react';
import {COLORS} from '../config';
import {FONT} from '../theme';
import {Caret, Check, typed} from './motion';

export type FormField = {label: string; value: string};

/**
 * The Submit button is deliberately frame-independent: it takes no progress
 * props and never changes. That's the point of the video.
 */
export const SubmitButton: React.FC<{label: string; s: number}> = React.memo(({label, s}) => (
  <div
    style={{
      alignSelf: 'flex-start',
      background: COLORS.pageText,
      color: '#fff',
      fontWeight: 600,
      fontSize: 24 * s,
      padding: `${16 * s}px ${30 * s}px`,
      borderRadius: 14 * s,
      letterSpacing: '-0.01em',
    }}
  >
    {label}
  </div>
));

export const JobForm: React.FC<{
  width: number;
  company: {name: string; color: string; letter: string};
  title: string;
  role: string;
  fields: readonly FormField[];
  /** typed progress 0..1 per field */
  progress: number[];
  /** check progress 0..1 per field (optional) */
  checks?: number[];
  active?: number;
  whyLabel: string;
  whyLines: readonly string[];
  whyProgress: number;
  whyCheck?: number;
  submit: string;
  /** Hide fields after this index (for short scenes). */
  compact?: boolean;
}> = ({
  width,
  company,
  title,
  role,
  fields,
  progress,
  checks = [],
  active = -1,
  whyLabel,
  whyLines,
  whyProgress,
  whyCheck = 0,
  submit,
  compact,
}) => {
  const s = width / 920;
  const pad = 44 * s;
  const caretColor = '#0A84FF';
  const fieldH = (compact ? 58 : 64) * s;

  // Paragraph fills line by line.
  const totalChars = whyLines.reduce((a, l) => a + l.length, 0);
  let budget = Math.round(Math.max(0, Math.min(1, whyProgress)) * totalChars);
  const shown = whyLines.map((l) => {
    const n = Math.min(l.length, budget);
    budget -= n;
    return l.slice(0, n);
  });
  const lastTyped = shown.reduce((acc, l, i) => (l.length ? i : acc), 0);

  return (
    <div
      style={{
        width,
        background: COLORS.page,
        color: COLORS.pageText,
        fontFamily: FONT,
        padding: pad,
        boxSizing: 'border-box',
        display: 'flex',
        flexDirection: 'column',
        gap: (compact ? 16 : 22) * s,
      }}
    >
      <div style={{display: 'flex', alignItems: 'center', gap: 16 * s}}>
        <div
          style={{
            width: 52 * s,
            height: 52 * s,
            borderRadius: 14 * s,
            background: company.color,
            color: '#fff',
            fontWeight: 800,
            fontSize: 28 * s,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          {company.letter}
        </div>
        <div style={{display: 'flex', flexDirection: 'column'}}>
          <div style={{fontSize: 30 * s, fontWeight: 700, letterSpacing: '-0.02em'}}>{title}</div>
          <div style={{fontSize: 21 * s, color: COLORS.pageMuted}}>
            {company.name} · {role}
          </div>
        </div>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          columnGap: 18 * s,
          rowGap: (compact ? 14 : 18) * s,
        }}
      >
        {fields.map((f, i) => {
          const p = progress[i] ?? 0;
          const c = checks[i] ?? 0;
          const wide = i >= 2;
          return (
            <div
              key={f.label}
              style={{gridColumn: wide ? 'span 2' : undefined, display: 'flex', flexDirection: 'column', gap: 8 * s}}
            >
              <div style={{fontSize: 19 * s, fontWeight: 600, color: COLORS.pageMuted}}>{f.label}</div>
              <div
                style={{
                  height: fieldH,
                  borderRadius: 12 * s,
                  background: COLORS.pageField,
                  border: `${2 * s}px solid ${
                    c > 0.5 ? 'rgba(48,209,88,0.55)' : active === i ? caretColor : 'transparent'
                  }`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: `0 ${18 * s}px`,
                  fontSize: 25 * s,
                  fontWeight: 500,
                  boxSizing: 'border-box',
                }}
              >
                <span style={{whiteSpace: 'nowrap', overflow: 'hidden'}}>
                  {typed(f.value, p)}
                  {active === i ? <Caret height={28 * s} color={caretColor} blink={false} /> : null}
                </span>
                <Check progress={c} size={30 * s} />
              </div>
            </div>
          );
        })}
      </div>

      <div style={{display: 'flex', flexDirection: 'column', gap: 8 * s}}>
        <div style={{fontSize: 19 * s, fontWeight: 600, color: COLORS.pageMuted}}>{whyLabel}</div>
        <div
          style={{
            borderRadius: 12 * s,
            background: COLORS.pageField,
            border: `${2 * s}px solid ${
              whyCheck > 0.5 ? 'rgba(48,209,88,0.55)' : active === fields.length ? caretColor : 'transparent'
            }`,
            padding: `${14 * s}px ${18 * s}px`,
            fontSize: 23 * s,
            lineHeight: 1.45,
            minHeight: (compact ? 104 : 136) * s,
            boxSizing: 'border-box',
            position: 'relative',
          }}
        >
          {shown.map((l, i) => (
            <div key={i} style={{minHeight: 23 * s * 1.45, whiteSpace: 'nowrap'}}>
              {l}
              {active === fields.length && i === lastTyped ? (
                <Caret height={26 * s} color={caretColor} blink={false} />
              ) : null}
            </div>
          ))}
          <div style={{position: 'absolute', right: 18 * s, bottom: 14 * s}}>
            <Check progress={whyCheck} size={30 * s} />
          </div>
        </div>
      </div>

      <SubmitButton label={submit} s={s} />
    </div>
  );
};
