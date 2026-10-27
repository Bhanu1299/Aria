import React from 'react';
import {AbsoluteFill} from 'remotion';
import {COLORS, TEXT} from '../config';
import {useLayout} from '../layout';
import {FONT, headline} from '../theme';
import {Reveal} from '../ui/motion';

/** Full black. Everything stops. The longest hold in the video. */
export const S09Never: React.FC = () => {
  const L = useLayout();
  return (
    <AbsoluteFill
      style={{
        background: COLORS.bg,
        fontFamily: FONT,
        alignItems: 'center',
        justifyContent: 'center',
        padding: `${L.safe.top}px ${L.safe.right}px ${L.safe.bottom}px ${L.safe.left}px`,
        textAlign: 'center',
      }}
    >
      <div style={{...headline(L.vertical ? 160 : 150), color: COLORS.text}}>
        <Reveal text={TEXT.never.line1} stagger={3} />
      </div>
      <div style={{...headline(L.vertical ? 190 : 170), color: COLORS.accent, marginTop: L.vertical ? 40 : 24}}>
        <Reveal text={TEXT.never.line2} delay={36} stagger={4} />
      </div>
    </AbsoluteFill>
  );
};
