import React from 'react';
import {AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig} from 'remotion';
import {COLORS, COMPANIES, TEXT} from '../config';
import {useLayout} from '../layout';
import {FONT, headline} from '../theme';
import {CameraPush, clamp, ease, enterAt, Reveal} from '../ui/motion';
import {JobPosting, ResumePage} from '../ui/ResumePage';

const AtsBadge: React.FC<{size: number; value: number; progress: number; appear: number}> = ({
  size,
  value,
  progress,
  appear,
}) => {
  const t = TEXT.resume;
  const r = size / 2 - 12;
  const c = 2 * Math.PI * r;
  return (
    <div
      style={{
        width: size,
        height: size,
        position: 'relative',
        opacity: appear,
        transform: `scale(${0.85 + 0.15 * appear})`,
        fontFamily: FONT,
      }}
    >
      <div style={{position: 'absolute', inset: 0, borderRadius: '50%', background: COLORS.surface, boxShadow: '0 30px 80px rgba(0,0,0,0.6)'}} />
      <svg width={size} height={size} style={{position: 'absolute', inset: 0, transform: 'rotate(-90deg)'}}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={COLORS.surface2} strokeWidth={14} />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={COLORS.green}
          strokeWidth={14}
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - (progress * value) / 100)}
        />
      </svg>
      <div
        style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          color: COLORS.text,
        }}
      >
        <div style={{fontSize: size * 0.1, fontWeight: 700, color: COLORS.secondary, letterSpacing: '0.12em'}}>{t.atsLabel}</div>
        <div style={{fontSize: size * 0.36, fontWeight: 800, letterSpacing: '-0.04em', lineHeight: 1, fontVariantNumeric: 'tabular-nums'}}>
          {Math.round(progress * value)}
        </div>
        <div style={{fontSize: size * 0.075, fontWeight: 500, color: COLORS.secondary, marginTop: size * 0.02}}>{t.atsNote}</div>
      </div>
    </div>
  );
};

export const S05Resume: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const L = useLayout();
  const v = L.vertical;
  const t = TEXT.resume;

  const postIn = enterAt(frame, fps, 14, 16);
  const resIn = enterAt(frame, fps, 24, 18);
  const badgeIn = enterAt(frame, fps, 120, 14);
  const count = ease(frame, [124, 176], [0, 1]);

  // Each skill lights up in the posting, then in the resume.
  const hlPost = t.skills.map((_, i) => interpolate(frame, [48 + i * 18, 58 + i * 18], [0, 1], clamp));
  const hlRes = t.skills.map((_, i) => interpolate(frame, [56 + i * 18, 66 + i * 18], [0, 1], clamp));

  const pos = v
    ? {post: {x: 80, y: 470, w: 600}, res: {x: 420, y: 800, w: 580}, badge: {x: 90, y: 1010, size: 300}}
    : {post: {x: 760, y: 170, w: 560}, res: {x: 1250, y: 250, w: 560}, badge: {x: 120, y: 560, size: 330}};

  return (
    <AbsoluteFill style={{background: COLORS.bg, fontFamily: FONT}}>
      <CameraPush from={1} to={1.04} origin="60% 60%">
        <div style={{position: 'absolute', left: pos.post.x, top: pos.post.y + (1 - postIn) * 80, opacity: postIn}}>
          <JobPosting
            width={pos.post.w}
            company={COMPANIES[1]}
            role={t.postingRole}
            heading={t.postingHeading}
            skills={t.skills}
            highlight={hlPost}
          />
        </div>
        <div
          style={{
            position: 'absolute',
            left: pos.res.x,
            top: pos.res.y + (1 - resIn) * 120,
            opacity: resIn,
            transform: `rotate(${(1 - resIn) * 3}deg)`,
          }}
        >
          <ResumePage width={pos.res.w} name={t.resumeName} contact={t.resumeContact} sections={t.sections} skills={t.skills} highlight={hlRes} />
        </div>
        <div style={{position: 'absolute', left: pos.badge.x, top: pos.badge.y}}>
          <AtsBadge size={pos.badge.size} value={t.ats} progress={count} appear={badgeIn} />
        </div>
      </CameraPush>

      <div
        style={{
          position: 'absolute',
          left: L.safe.left,
          top: v ? L.safe.top + 30 : 170,
          width: v ? L.box.w : 620,
          ...headline(v ? 124 : 84),
          color: COLORS.text,
        }}
      >
        <Reveal text={t.headline} stagger={3} />
      </div>
    </AbsoluteFill>
  );
};
