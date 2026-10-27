import React from 'react';
import {AbsoluteFill} from 'remotion';
import {COLORS, TEXT} from '../config';
import {useLayout} from '../layout';
import {FONT, headline} from '../theme';
import {Reveal} from '../ui/motion';

/** Full black. Everything stops. The longest hold in the video. */
export const S10Never: React.FC = () => {
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
      {/* 9:16 is too narrow for the two-line break: let the browser balance three lines instead. */}
      <div style={{...headline(L.vertical ? 116 : 112), color: COLORS.text, textWrap: 'balance'}}>
        <Reveal text={L.vertical ? TEXT.never.line1.replace('\n', ' ') : TEXT.never.line1} stagger={3} />
      </div>
      <div style={{...headline(L.vertical ? 148 : 150), color: COLORS.accent, whiteSpace: 'nowrap', marginTop: L.vertical ? 56 : 36}}>
        <Reveal text={TEXT.never.line2} delay={TEXT.never.line2At} stagger={4} />
      </div>
    </AbsoluteFill>
  );
};
