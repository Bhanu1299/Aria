import React from 'react';
import {AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig} from 'remotion';
import {COLORS, COMPANIES, POSTINGS, POSTINGS_BY_FIT, TEXT} from '../config';
import {useLayout} from '../layout';
import {FONT, headline} from '../theme';
import {CameraPush, clamp, ease, enterAt, Reveal} from '../ui/motion';

const T = {card: 6, rows: 16, perRow: 7, sort: 66, sortLen: 20};

/** The last 24 hours of postings land, then re-sort best fit first. */
export const S04Feed: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const L = useLayout();
  const v = L.vertical;
  const t = TEXT.feed;

  const card = v ? {x: L.safe.left, y: 520, w: L.box.w} : {x: 880, y: 150, w: 920};
  const rowH = 132;
  const headH = 92;
  const cardIn = enterAt(frame, fps, T.card, 14);
  const sort = ease(frame, [T.sort, T.sort + T.sortLen], [0, 1]);
  const chip = enterAt(frame, fps, T.sort - 6, 12);

  return (
    <AbsoluteFill style={{background: COLORS.bg, fontFamily: FONT}}>
      <CameraPush from={1} to={1.04} origin="60% 50%">
        <div
          style={{
            position: 'absolute',
            left: card.x,
            top: card.y + (1 - cardIn) * 60,
            width: card.w,
            height: headH + rowH * POSTINGS.length + 20,
            opacity: cardIn,
            borderRadius: 32,
            background: COLORS.surface,
            border: '1px solid rgba(255,255,255,0.1)',
            boxShadow: '0 30px 90px rgba(0,0,0,0.6)',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              height: headH,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '0 32px',
              borderBottom: '1px solid rgba(255,255,255,0.08)',
            }}
          >
            <span style={{fontSize: 32, fontWeight: 700, color: COLORS.text}}>{t.listTitle}</span>
            <span
              style={{
                fontSize: 24,
                fontWeight: 600,
                color: COLORS.green,
                background: 'rgba(48,209,88,0.12)',
                border: '1.5px solid rgba(48,209,88,0.45)',
                borderRadius: 999,
                padding: '6px 18px',
                opacity: chip,
                transform: `scale(${0.9 + 0.1 * chip})`,
              }}
            >
              {t.sortLabel}
            </span>
          </div>

          {POSTINGS.map((p, i) => {
            const inP = enterAt(frame, fps, T.rows + i * T.perRow, 12);
            const target = POSTINGS_BY_FIT.indexOf(p);
            const slot = interpolate(sort, [0, 1], [i, target]);
            const top = target === 0 ? sort : 0;
            const c = COMPANIES[p.company];
            return (
              <div
                key={c.name}
                style={{
                  position: 'absolute',
                  left: 16,
                  right: 16,
                  top: headH + 10 + slot * rowH - (1 - inP) * 40,
                  height: rowH - 12,
                  opacity: inP,
                  borderRadius: 22,
                  background: `rgba(48,209,88,${0.08 * top})`,
                  border: `1.5px solid rgba(48,209,88,${0.4 * top})`,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 24,
                  padding: '0 22px',
                  boxSizing: 'border-box',
                  zIndex: target === 0 ? 2 : 1,
                }}
              >
                <div
                  style={{
                    width: 64,
                    height: 64,
                    borderRadius: 18,
                    background: c.color,
                    color: '#fff',
                    fontWeight: 800,
                    fontSize: 32,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                  }}
                >
                  {c.letter}
                </div>
                <div style={{flex: 1, minWidth: 0}}>
                  <div style={{fontSize: 32, fontWeight: 700, color: COLORS.text, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis'}}>
                    {p.role}
                  </div>
                  <div style={{fontSize: 24, color: COLORS.secondary, marginTop: 2}}>
                    {c.name} · {p.posted}
                  </div>
                </div>
                <div
                  style={{
                    fontSize: 26,
                    fontWeight: 700,
                    fontVariantNumeric: 'tabular-nums',
                    color: target === 0 && sort > 0.5 ? COLORS.green : COLORS.text,
                    background: COLORS.surface2,
                    borderRadius: 14,
                    padding: '8px 16px',
                    whiteSpace: 'nowrap',
                  }}
                >
                  <span style={{color: COLORS.secondary, fontWeight: 500, marginRight: 8}}>{t.fitLabel}</span>
                  {p.fit}
                </div>
              </div>
            );
          })}
        </div>
      </CameraPush>

      <div
        style={{
          position: 'absolute',
          left: L.safe.left,
          top: v ? L.safe.top + 40 : 330,
          width: v ? L.box.w : 700,
          ...headline(v ? 120 : 88),
          color: COLORS.text,
        }}
      >
        <Reveal text={t.headline} stagger={3} />
      </div>
    </AbsoluteFill>
  );
};
