import React from 'react';
import {AbsoluteFill} from 'remotion';
import {COLORS, TEXT} from '../config';
import {useLayout} from '../layout';
import {FONT, headline} from '../theme';
import {CameraPush, Reveal} from '../ui/motion';
import {DIAGRAM_W, SystemDiagram} from '../ui/SystemDiagram';

export const S07System: React.FC = () => {
  const L = useLayout();
  const v = L.vertical;
  const scale = v ? L.box.w / DIAGRAM_W : 0.86;
  const at = v ? {x: L.safe.left, y: 500} : {x: 980, y: 110};
  const [a, b] = TEXT.system.headline.split('. ');

  return (
    <AbsoluteFill style={{background: COLORS.bg, fontFamily: FONT}}>
      <CameraPush from={1} to={1.03}>
        <div style={{position: 'absolute', left: at.x, top: at.y}}>
          <SystemDiagram scale={scale} />
        </div>
      </CameraPush>
      <div
        style={{
          position: 'absolute',
          left: L.safe.left,
          top: v ? L.safe.top + 30 : 300,
          width: v ? L.box.w : 760,
          ...headline(v ? 130 : 120),
        }}
      >
        <div style={{color: COLORS.text}}>
          <Reveal text={`${a}.`} stagger={3} />
        </div>
        <div style={{color: COLORS.secondary}}>
          <Reveal text={b} delay={10} stagger={3} />
        </div>
      </div>
    </AbsoluteFill>
  );
};
