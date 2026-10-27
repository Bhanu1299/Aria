import React from 'react';
import {COLORS} from '../config';
import {FONT, MONO} from '../theme';
import {GreyLine} from './ResumePage';

/** Gmail-style compose card. Always a Draft; the Send button never animates. */
export const EmailCard: React.FC<{
  width: number;
  pill: string;
  to: React.ReactNode;
  subject: string;
  bodyProgress?: number;
}> = ({width, pill, to, subject, bodyProgress = 1}) => {
  const s = width / 760;
  const lines = ['94%', '88%', '97%', '62%', '0', '90%', '45%'];
  const shownLines = Math.round(bodyProgress * lines.length);
  const Row: React.FC<{label: string; children: React.ReactNode}> = ({label, children}) => (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 16 * s,
        padding: `${16 * s}px 0`,
        borderBottom: `1px solid ${COLORS.pageLine}`,
        fontSize: 22 * s,
      }}
    >
      <span style={{color: COLORS.pageMuted, width: 80 * s, flexShrink: 0}}>{label}</span>
      <div style={{flex: 1, minWidth: 0}}>{children}</div>
    </div>
  );
  return (
    <div
      style={{
        width,
        borderRadius: 22 * s,
        overflow: 'hidden',
        background: COLORS.page,
        fontFamily: FONT,
        color: COLORS.pageText,
        boxShadow: `0 ${36 * s}px ${110 * s}px rgba(0,0,0,0.7)`,
      }}
    >
      <div
        style={{
          background: '#F2F2F7',
          padding: `${18 * s}px ${28 * s}px`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: 22 * s,
          fontWeight: 600,
        }}
      >
        <span>New message</span>
        <span
          style={{
            background: 'rgba(255,159,10,0.16)',
            color: '#B25000',
            border: '1.5px solid rgba(255,159,10,0.6)',
            borderRadius: 999,
            padding: `${5 * s}px ${18 * s}px`,
            fontSize: 20 * s,
            fontWeight: 700,
          }}
        >
          {pill}
        </span>
      </div>
      <div style={{padding: `0 ${28 * s}px`}}>
        <Row label="To">
          <span style={{fontFamily: MONO, fontSize: 21 * s}}>{to}</span>
        </Row>
        <Row label="Subject">
          <span style={{fontWeight: 600, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', display: 'block'}}>
            {subject}
          </span>
        </Row>
        <div style={{display: 'flex', flexDirection: 'column', gap: 14 * s, padding: `${24 * s}px 0`, minHeight: 210 * s}}>
          {lines.slice(0, shownLines).map((w, i) =>
            w === '0' ? <div key={i} style={{height: 6 * s}} /> : <GreyLine key={i} w={w} s={s} h={13} />,
          )}
        </div>
        <div style={{display: 'flex', alignItems: 'center', gap: 18 * s, padding: `0 0 ${26 * s}px`}}>
          <div
            style={{
              background: '#E5E5EA',
              color: COLORS.pageMuted,
              fontWeight: 600,
              fontSize: 21 * s,
              padding: `${12 * s}px ${30 * s}px`,
              borderRadius: 999,
            }}
          >
            Send
          </div>
          <span style={{fontSize: 18 * s, color: COLORS.pageMuted}}>Saved to drafts</span>
        </div>
      </div>
    </div>
  );
};
