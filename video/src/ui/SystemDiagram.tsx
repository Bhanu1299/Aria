import React from 'react';
import {interpolate, useCurrentFrame, useVideoConfig} from 'remotion';
import {COLORS, TEXT} from '../config';
import {FONT, MONO} from '../theme';
import {AriaGlyph} from './MacTopBar';
import {Check, clamp, enterAt} from './motion';

/** Design space; the component is scaled to fit. */
export const DIAGRAM_W = 920;
export const DIAGRAM_H = 1000;

type P = {x: number; y: number};
const NODES: Record<'aria' | 'gate' | 'cold', P> = {
  aria: {x: 210, y: 150},
  cold: {x: 710, y: 150},
  gate: {x: 460, y: 470},
};

type Edge = {a: P; b: P; c: P; delay: number};
const mid = (a: P, b: P, bend: P): P => ({x: (a.x + b.x) / 2 + bend.x, y: (a.y + b.y) / 2 + bend.y});
const EDGES: Edge[] = [
  {a: NODES.aria, b: NODES.gate, c: mid(NODES.aria, NODES.gate, {x: -40, y: 40}), delay: 22},
  {a: NODES.cold, b: NODES.gate, c: mid(NODES.cold, NODES.gate, {x: 40, y: 40}), delay: 28},
  {a: NODES.aria, b: NODES.cold, c: mid(NODES.aria, NODES.cold, {x: 0, y: -70}), delay: 34},
];

const qPoint = (e: Edge, t: number): P => ({
  x: (1 - t) * (1 - t) * e.a.x + 2 * (1 - t) * t * e.c.x + t * t * e.b.x,
  y: (1 - t) * (1 - t) * e.a.y + 2 * (1 - t) * t * e.c.y + t * t * e.b.y,
});

const NodeBox: React.FC<{p: P; name: string; note: string; appear: number; icon: React.ReactNode}> = ({
  p,
  name,
  note,
  appear,
  icon,
}) => (
  <div
    style={{
      position: 'absolute',
      left: p.x - 140,
      top: p.y - 58,
      width: 280,
      height: 116,
      borderRadius: 28,
      background: COLORS.surface,
      border: '1.5px solid rgba(255,255,255,0.12)',
      boxShadow: '0 20px 60px rgba(0,0,0,0.6)',
      display: 'flex',
      alignItems: 'center',
      gap: 18,
      padding: '0 24px',
      boxSizing: 'border-box',
      opacity: appear,
      transform: `translateY(${(1 - appear) * 30}px) scale(${0.94 + 0.06 * appear})`,
      fontFamily: FONT,
    }}
  >
    <div
      style={{
        width: 60,
        height: 60,
        borderRadius: 18,
        background: COLORS.surface2,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
      }}
    >
      {icon}
    </div>
    <div>
      <div style={{fontSize: 36, fontWeight: 700, letterSpacing: '-0.02em', color: COLORS.text}}>{name}</div>
      <div style={{fontSize: 21, color: COLORS.secondary, marginTop: 2}}>{note}</div>
    </div>
  </div>
);

export const GateIcon = () => (
  <svg width={34} height={34} viewBox="0 0 34 34">
    {[6, 15, 24].map((y, i) => (
      <rect key={y} x={4 + i * 3} y={y} width={26 - i * 6} height={5} rx={2.5} fill={COLORS.text} opacity={1 - i * 0.25} />
    ))}
  </svg>
);
export const ColdIcon = () => (
  <svg width={34} height={34} viewBox="0 0 34 34">
    <rect x={3} y={8} width={28} height={19} rx={4} fill="none" stroke={COLORS.text} strokeWidth={2.4} />
    <path d="M4 10 L17 19 L30 10" fill="none" stroke={COLORS.text} strokeWidth={2.4} strokeLinejoin="round" />
  </svg>
);

export const SystemDiagram: React.FC<{scale: number}> = ({scale}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const n = TEXT.system.nodes;
  const tiers = TEXT.system.tiers;

  const appear = (d: number) => enterAt(frame, fps, d, 14);
  const panel = appear(52);
  // The request chip drops from GateLLM into the tier list and lands on "small".
  const drop = enterAt(frame, fps, 66, 16);
  const landed = interpolate(frame, [80, 92], [0, 1], clamp);
  const cost = enterAt(frame, fps, 88, 12);

  const panelTop = 600;
  const rowH = 96;

  return (
    <div style={{width: DIAGRAM_W, height: DIAGRAM_H, transform: `scale(${scale})`, transformOrigin: '0 0', position: 'relative'}}>
      <svg width={DIAGRAM_W} height={DIAGRAM_H} style={{position: 'absolute', inset: 0}}>
        {/* GateLLM → tier list connector */}
        <line
          x1={NODES.gate.x}
          y1={NODES.gate.y + 58}
          x2={NODES.gate.x}
          y2={panelTop}
          stroke={COLORS.accent}
          strokeWidth={3}
          pathLength={1}
          strokeDasharray={1}
          strokeDashoffset={1 - interpolate(frame, [48, 60], [0, 1], clamp)}
          opacity={0.8}
        />
        {EDGES.map((e, i) => {
          const p = interpolate(frame, [e.delay, e.delay + 18], [0, 1], {...clamp});
          return (
            <path
              key={i}
              d={`M ${e.a.x} ${e.a.y} Q ${e.c.x} ${e.c.y} ${e.b.x} ${e.b.y}`}
              fill="none"
              stroke={COLORS.accent}
              strokeWidth={3.5}
              strokeLinecap="round"
              pathLength={1}
              strokeDasharray={1}
              strokeDashoffset={1 - p}
            />
          );
        })}
        {/* Pulses: tiny dots travelling along the lines once drawn */}
        {EDGES.flatMap((e, i) =>
          [0, 0.5].map((phase) => {
            const start = e.delay + 20;
            if (frame < start) return null;
            const t = (((frame - start) / 34 + phase + i * 0.17) % 1 + 1) % 1;
            const pt = qPoint(e, t);
            const fade = Math.sin(t * Math.PI);
            return <circle key={`${i}-${phase}`} cx={pt.x} cy={pt.y} r={7} fill={COLORS.accent} opacity={fade} />;
          }),
        )}
      </svg>

      <NodeBox p={NODES.aria} name={n.aria.name} note={n.aria.note} appear={appear(4)} icon={<AriaGlyph state="idle" size={40} />} />
      <NodeBox p={NODES.cold} name={n.cold.name} note={n.cold.note} appear={appear(9)} icon={<ColdIcon />} />
      <NodeBox p={NODES.gate} name={n.gate.name} note={n.gate.note} appear={appear(14)} icon={<GateIcon />} />

      {/* Tier list */}
      <div
        style={{
          position: 'absolute',
          left: 170,
          top: panelTop,
          width: 580,
          borderRadius: 26,
          background: 'rgba(28,28,30,0.9)',
          border: '1.5px solid rgba(255,255,255,0.1)',
          padding: 18,
          boxSizing: 'border-box',
          opacity: panel,
          transform: `translateY(${(1 - panel) * 24}px)`,
          fontFamily: MONO,
        }}
      >
        {tiers.map((t, i) => {
          const isTarget = i === 0;
          return (
            <div
              key={t}
              style={{
                height: rowH - 12,
                marginBottom: i < tiers.length - 1 ? 12 : 0,
                borderRadius: 16,
                background: isTarget && landed > 0 ? `rgba(48,209,88,${0.12 * landed})` : COLORS.surface2,
                border: `1.5px solid ${isTarget && landed > 0 ? `rgba(48,209,88,${0.6 * landed})` : 'transparent'}`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '0 24px',
                fontSize: 30,
                color: i === 0 ? COLORS.text : COLORS.secondary,
                opacity: i === 0 ? 1 : 0.55,
              }}
            >
              <span>
                <span style={{color: COLORS.secondary, marginRight: 16}}>{i + 1}</span>
                {t}
              </span>
              {isTarget ? (
                <span style={{display: 'flex', alignItems: 'center', gap: 14}}>
                  <span
                    style={{
                      fontSize: 28,
                      fontWeight: 700,
                      color: COLORS.green,
                      opacity: cost,
                      transform: `translateX(${(1 - cost) * 16}px)`,
                    }}
                  >
                    {TEXT.system.cost}
                  </span>
                  <Check progress={landed} size={34} />
                </span>
              ) : null}
            </div>
          );
        })}
      </div>

      {/* Request chip */}
      <div
        style={{
          position: 'absolute',
          left: NODES.gate.x - 70,
          top: interpolate(drop, [0, 1], [NODES.gate.y + 30, panelTop + 18 + (rowH - 12) / 2 - 22]),
          width: 140,
          height: 44,
          borderRadius: 22,
          background: COLORS.text,
          color: COLORS.bg,
          fontFamily: MONO,
          fontWeight: 700,
          fontSize: 20,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          opacity: interpolate(frame, [64, 68, 84, 94], [0, 1, 1, 0], clamp),
          transform: `scale(${interpolate(landed, [0, 1], [1, 0.8])})`,
          boxShadow: '0 10px 30px rgba(0,0,0,0.5)',
        }}
      >
        request
      </div>
    </div>
  );
};
