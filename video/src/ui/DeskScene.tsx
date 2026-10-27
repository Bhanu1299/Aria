import React from 'react';
import {interpolate} from 'remotion';
import {Layout} from '../layout';

/** Where the monitor's screen sits for each format (the hook draws the form into it). */
export const deskGeom = (L: Layout) =>
  L.vertical
    ? {screen: {x: 100, y: 420, w: 880, h: 580}, head: {x: 540, y: 1175}, s: 1.05}
    : {screen: {x: 530, y: 150, w: 860, h: 520}, head: {x: 960, y: 790}, s: 0.95};

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const pt = (a: [number, number], b: [number, number], t: number) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t)];

/**
 * Me at the desk, seen from behind, lit by the monitor.
 * annoy 0..1: shoulders drop, head sinks. facepalm 0..1: hand comes up to the forehead.
 * shake: small head jerk in degrees. glow 0..1: monitor light.
 */
export const DeskScene: React.FC<{
  L: Layout;
  annoy: number;
  facepalm: number;
  shake?: number;
  sigh?: number;
  glow: number;
  screen: React.ReactNode;
}> = ({L, annoy, facepalm, shake = 0, sigh = 0, glow, screen}) => {
  const g = deskGeom(L);
  const {screen: sc, head, s} = g;
  const bezel = 18;
  const deskTop = sc.y + sc.h + bezel + 120;

  // Arm: shoulder → elbow → hand. Rest is down at the keyboard; facepalm brings the hand to the brow.
  const shoulder: [number, number] = [235, 235];
  const elbow = pt([300, 560], [330, 130], facepalm);
  const hand = pt([210, 700], [78, -30], facepalm);

  const headTilt = annoy * 5 + facepalm * 11 + shake;
  const headDrop = annoy * 18 + facepalm * 26;
  const slump = annoy * 16 + facepalm * 10 - sigh * 14;
  const rim = `rgba(170,195,255,${0.12 + 0.3 * glow})`;
  const body = '#0A0A0C';

  return (
    <div style={{position: 'absolute', inset: 0, overflow: 'hidden'}}>
      {/* Room light from the screen */}
      <div
        style={{
          position: 'absolute',
          left: sc.x + sc.w / 2 - sc.w,
          top: sc.y + sc.h / 2 - sc.h * 1.1,
          width: sc.w * 2,
          height: sc.h * 2.2,
          borderRadius: '50%',
          background: `radial-gradient(closest-side, rgba(150,180,255,${0.16 * glow}), rgba(150,180,255,0))`,
        }}
      />

      {/* Desk */}
      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          top: deskTop,
          bottom: 0,
          background: `linear-gradient(180deg, rgba(60,62,70,${0.35 + 0.35 * glow}) 0%, #050506 40%)`,
          borderTop: `1px solid rgba(255,255,255,${0.05 + 0.1 * glow})`,
        }}
      />

      {/* Monitor */}
      <div
        style={{
          position: 'absolute',
          left: sc.x + sc.w / 2 - 50,
          top: sc.y + sc.h + bezel,
          width: 100,
          height: deskTop - (sc.y + sc.h + bezel) + 6,
          background: 'linear-gradient(90deg, #1a1a1d, #2a2a2e, #1a1a1d)',
        }}
      />
      <div
        style={{
          position: 'absolute',
          left: sc.x + sc.w / 2 - 170,
          top: deskTop - 4,
          width: 340,
          height: 22,
          borderRadius: 11,
          background: '#202024',
        }}
      />
      <div
        style={{
          position: 'absolute',
          left: sc.x - bezel,
          top: sc.y - bezel,
          width: sc.w + bezel * 2,
          height: sc.h + bezel * 2,
          borderRadius: 26,
          background: '#151517',
          border: '1px solid rgba(255,255,255,0.08)',
          boxShadow: `0 0 ${120 * glow}px rgba(150,180,255,${0.22 * glow})`,
        }}
      />
      <div
        style={{
          position: 'absolute',
          left: sc.x,
          top: sc.y,
          width: sc.w,
          height: sc.h,
          borderRadius: 10,
          overflow: 'hidden',
          background: '#000',
        }}
      >
        {screen}
      </div>

      {/* Me */}
      <svg
        width={L.W}
        height={L.H}
        style={{position: 'absolute', inset: 0}}
        viewBox={`0 0 ${L.W} ${L.H}`}
      >
        <g transform={`translate(${head.x} ${head.y}) scale(${s})`}>
          {/* raised arm: behind the head and shoulders, outlined by the screen light */}
          <g transform={`translate(0 ${slump})`}>
            {[{w: 98, c: rim}, {w: 92, c: body}].map(({w, c}) => (
              <path
                key={w}
                d={`M ${shoulder[0]} ${shoulder[1]} L ${elbow[0]} ${elbow[1]} L ${hand[0]} ${hand[1]}`}
                fill="none"
                stroke={c}
                strokeWidth={w}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            ))}
          </g>
          {/* head: bowing forward sinks it behind the shoulders */}
          <g transform={`translate(0 ${headDrop + slump * 0.6}) rotate(${headTilt} 0 110)`}>
            <rect x={-48} y={40} width={96} height={170} rx={30} fill={body} />
            <ellipse cx={-90} cy={18} rx={16} ry={28} fill={body} stroke={rim} strokeWidth={2} />
            <ellipse cx={90} cy={18} rx={16} ry={28} fill={body} stroke={rim} strokeWidth={2} />
            <ellipse cx={0} cy={0} rx={94} ry={112} fill={body} stroke={rim} strokeWidth={3} />
            <path d="M -93 -10 C -96 -110 -40 -126 0 -126 C 52 -126 100 -104 94 -8 C 66 -54 -44 -62 -93 -10 Z" fill="#040405" />
          </g>
          {/* shoulders and back */}
          <g transform={`translate(0 ${slump})`}>
            <path
              d="M -340 900 L -340 470 C -340 285 -262 205 -128 182 L 128 182 C 262 205 340 285 340 470 L 340 900 Z"
              fill={body}
              stroke={rim}
              strokeWidth={3}
            />
          </g>
        </g>
      </svg>

      {/* Darken everything at the edges so the screen reads first */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          background: `radial-gradient(ellipse at 50% 35%, rgba(0,0,0,0) 40%, rgba(0,0,0,${interpolate(glow, [0, 1], [0.9, 0.55])}) 100%)`,
        }}
      />
    </div>
  );
};
