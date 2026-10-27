import React from 'react';
import {COLORS} from '../config';
import {FONT} from '../theme';

const HL = 'rgba(48,209,88,0.32)';

/** A skill word with a highlighter sweep (progress 0..1, left to right). */
export const Highlight: React.FC<{word: string; progress: number; s: number; dark?: boolean}> = ({
  word,
  progress,
  s,
  dark,
}) => {
  const p = Math.max(0, Math.min(1, progress));
  return (
    <span style={{position: 'relative', display: 'inline-block', padding: `0 ${5 * s}px`, whiteSpace: 'nowrap'}}>
      <span
        style={{
          position: 'absolute',
          left: 0,
          top: '8%',
          bottom: '4%',
          width: `${p * 100}%`,
          background: HL,
          borderRadius: 6 * s,
        }}
      />
      <span style={{position: 'relative', fontWeight: 700, color: dark ? COLORS.text : COLORS.pageText}}>{word}</span>
    </span>
  );
};

export const GreyLine: React.FC<{w: number | string; s: number; h?: number; color?: string}> = ({
  w,
  s,
  h = 12,
  color = COLORS.pageLine,
}) => <div style={{width: w, height: h * s, borderRadius: h * s, background: color, flexShrink: 0}} />;

/** White resume page: name header, Experience / Projects / Skills. Fake content. */
export const ResumePage: React.FC<{
  width: number;
  name: string;
  contact: string;
  sections: readonly string[];
  skills: readonly string[];
  /** highlight progress per skill */
  highlight: number[];
}> = ({width, name, contact, sections, skills, highlight}) => {
  const s = width / 640;
  const height = width * 1.294; // US letter
  const Section: React.FC<{title: string; children: React.ReactNode}> = ({title, children}) => (
    <div style={{display: 'flex', flexDirection: 'column', gap: 12 * s}}>
      <div
        style={{
          fontSize: 17 * s,
          fontWeight: 700,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          color: COLORS.pageText,
          borderBottom: `${1.5 * s}px solid ${COLORS.pageText}`,
          paddingBottom: 6 * s,
        }}
      >
        {title}
      </div>
      {children}
    </div>
  );
  const Entry: React.FC<{lines: (number | string)[]; inline?: React.ReactNode}> = ({lines, inline}) => (
    <div style={{display: 'flex', flexDirection: 'column', gap: 9 * s}}>
      <div style={{display: 'flex', justifyContent: 'space-between'}}>
        <GreyLine w={200 * s} s={s} h={13} color="#C7C7CC" />
        <GreyLine w={80 * s} s={s} h={11} />
      </div>
      {lines.map((l, i) => (
        <GreyLine key={i} w={l} s={s} h={10} />
      ))}
      {inline}
    </div>
  );

  return (
    <div
      style={{
        width,
        height,
        background: COLORS.page,
        borderRadius: 10 * s,
        padding: `${50 * s}px ${52 * s}px`,
        boxSizing: 'border-box',
        fontFamily: FONT,
        color: COLORS.pageText,
        display: 'flex',
        flexDirection: 'column',
        gap: 26 * s,
        boxShadow: `0 ${30 * s}px ${90 * s}px rgba(0,0,0,0.6)`,
      }}
    >
      <div style={{textAlign: 'center', display: 'flex', flexDirection: 'column', gap: 6 * s}}>
        <div style={{fontSize: 40 * s, fontWeight: 800, letterSpacing: '-0.02em'}}>{name}</div>
        <div style={{fontSize: 16 * s, color: COLORS.pageMuted}}>{contact}</div>
      </div>
      <Section title={sections[0]}>
        <Entry
          lines={['96%', '88%']}
          inline={
            <div style={{fontSize: 17 * s, color: COLORS.pageMuted, display: 'flex', alignItems: 'center', gap: 4 * s, flexWrap: 'wrap'}}>
              <span>Built</span>
              <Highlight word={skills[2]} progress={highlight[2]} s={s} />
              <span>with</span>
              <Highlight word={skills[0]} progress={highlight[0]} s={s} />
            </div>
          }
        />
        <Entry lines={['92%', '70%']} />
      </Section>
      <Section title={sections[1]}>
        <Entry
          lines={['90%']}
          inline={
            <div style={{fontSize: 17 * s, color: COLORS.pageMuted, display: 'flex', alignItems: 'center', gap: 4 * s, flexWrap: 'wrap'}}>
              <span>Job agent on</span>
              <Highlight word={skills[1]} progress={highlight[1]} s={s} />
              <span>+</span>
              <Highlight word={skills[3]} progress={highlight[3]} s={s} />
            </div>
          }
        />
        <Entry lines={['84%', '60%']} />
      </Section>
      <Section title={sections[2]}>
        <div style={{display: 'flex', flexWrap: 'wrap', gap: 8 * s, fontSize: 17 * s, alignItems: 'center'}}>
          {skills.map((sk, i) => (
            <Highlight key={sk} word={sk} progress={highlight[i]} s={s} />
          ))}
          <GreyLine w={90 * s} s={s} h={10} />
          <GreyLine w={70 * s} s={s} h={10} />
        </div>
      </Section>
    </div>
  );
};

/** A dark job-posting card with the same skills highlighted. */
export const JobPosting: React.FC<{
  width: number;
  company: {name: string; color: string; letter: string};
  role: string;
  heading: string;
  skills: readonly string[];
  highlight: number[];
}> = ({width, company, role, heading, skills, highlight}) => {
  const s = width / 640;
  const bullets: [string, number, string][] = [
    ['Ship production services every week', -1, ''],
    ['Build', 2, 'that use real tools'],
    ['Browser automation with', 1, ''],
    ['APIs in', 3, 'and Postgres'],
  ];
  return (
    <div
      style={{
        width,
        background: COLORS.surface,
        borderRadius: 26 * s,
        padding: 40 * s,
        boxSizing: 'border-box',
        fontFamily: FONT,
        color: COLORS.text,
        display: 'flex',
        flexDirection: 'column',
        gap: 22 * s,
        border: '1px solid rgba(255,255,255,0.08)',
        boxShadow: `0 ${30 * s}px ${90 * s}px rgba(0,0,0,0.6)`,
      }}
    >
      <div style={{display: 'flex', alignItems: 'center', gap: 16 * s}}>
        <div
          style={{
            width: 52 * s,
            height: 52 * s,
            borderRadius: 14 * s,
            background: company.color,
            fontWeight: 800,
            fontSize: 26 * s,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          {company.letter}
        </div>
        <div>
          <div style={{fontSize: 28 * s, fontWeight: 700, letterSpacing: '-0.02em'}}>{role}</div>
          <div style={{fontSize: 19 * s, color: COLORS.secondary}}>{company.name}</div>
        </div>
      </div>
      <GreyLine w="92%" s={s} h={10} color={COLORS.surface2} />
      <GreyLine w="76%" s={s} h={10} color={COLORS.surface2} />
      <div style={{fontSize: 21 * s, fontWeight: 700}}>{heading}</div>
      <div style={{display: 'flex', flexDirection: 'column', gap: 12 * s, fontSize: 21 * s, color: COLORS.secondary}}>
        {bullets.map(([pre, k, post]) => (
          <div key={pre} style={{display: 'flex', alignItems: 'center', gap: 6 * s, flexWrap: 'wrap'}}>
            <span style={{color: COLORS.secondary}}>•</span>
            <span>{pre}</span>
            {k >= 0 ? <Highlight word={skills[k]} progress={highlight[k]} s={s} dark /> : null}
            {post ? <span>{post}</span> : null}
          </div>
        ))}
        <div style={{display: 'flex', alignItems: 'center', gap: 6 * s}}>
          <span>•</span>
          <span>Strong</span>
          <Highlight word={skills[0]} progress={highlight[0]} s={s} dark />
        </div>
      </div>
    </div>
  );
};
