import React from 'react';
import {AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig} from 'remotion';
import {COLORS, TEXT} from '../config';
import {useLayout} from '../layout';
import {FONT, headline, MONO} from '../theme';
import {AriaGlyph} from '../ui/MacTopBar';
import {Check, clamp, ease, enterAt, Reveal} from '../ui/motion';
import {ColdIcon, GateIcon} from '../ui/SystemDiagram';

/** Design space for the visual; scaled to fit each format. */
const W = 920;
const H = 800;
const GATE = {x: 230, y: 230, w: 460, h: 120};
const TIER_Y = 430;
const ROW = 110;

/**
 * Calls from Aria and Coldstart flow into GateLLM. One lands on the small model;
 * one fails the small model's check and escalates to frontier.
 */
export const S12Gateway: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const L = useLayout();
  const v = L.vertical;
  const t = TEXT.gateway;

  const scale = v ? L.box.w / W : 0.9;
  const at = v ? {x: L.safe.left, y: 520} : {x: 960, y: 120};
  const gateIn = enterAt(frame, fps, 4, 14);
  const tiersIn = enterAt(frame, fps, 14, 14);
  const sub = enterAt(frame, fps, 104, 14);

  // Request 1: frames 30-60 drops into "small" and lands.
  // Request 2: frames 62-112 tries "small", fails the check, drops to "frontier".
  const rowCenter = (i: number) => TIER_Y + 18 + i * ROW + (ROW - 14) / 2;
  const r1 = ease(frame, [30, 48], [0, 1]);
  const r1Land = interpolate(frame, [48, 56], [0, 1], clamp);
  const r2a = ease(frame, [62, 78], [0, 1]);
  const r2Fail = interpolate(frame, [78, 84, 92], [0, 1, 0], clamp);
  const r2b = ease(frame, [90, 106], [0, 1]);
  const r2Land = interpolate(frame, [106, 114], [0, 1], clamp);
  const y1 = interpolate(r1, [0, 1], [GATE.y + GATE.h + 24, rowCenter(0)]);
  const y2 = r2b > 0 ? interpolate(r2b, [0, 1], [rowCenter(0), rowCenter(2)]) : interpolate(r2a, [0, 1], [GATE.y + GATE.h + 24, rowCenter(0)]);

  const lineIn = (d: number) => interpolate(frame, [d, d + 12], [0, 1], clamp);

  const chip = (label: string, y: number, x: number, opacity: number, shake = 0) => (
    <div
      style={{
        position: 'absolute',
        left: x - 90 + shake,
        top: y - 24,
        width: 180,
        height: 48,
        borderRadius: 24,
        background: COLORS.text,
        color: COLORS.bg,
        fontFamily: MONO,
        fontWeight: 700,
        fontSize: 20,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        opacity,
        boxShadow: '0 12px 30px rgba(0,0,0,0.5)',
        zIndex: 3,
      }}
    >
      {label}
    </div>
  );

  return (
    <AbsoluteFill style={{background: COLORS.bg, fontFamily: FONT}}>
      <div style={{position: 'absolute', left: at.x, top: at.y, width: W, height: H, transform: `scale(${scale})`, transformOrigin: '0 0'}}>
        {/* Sources */}
        <svg width={W} height={H} style={{position: 'absolute', inset: 0}}>
          {[0, 1].map((i) => {
            const sx = i === 0 ? 170 : W - 170;
            const p = lineIn(8 + i * 4);
            return (
              <path
                key={i}
                d={`M ${sx} 110 Q ${sx} ${GATE.y + GATE.h / 2} ${i === 0 ? GATE.x : GATE.x + GATE.w} ${GATE.y + GATE.h / 2}`}
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
          <line
            x1={W / 2}
            y1={GATE.y + GATE.h}
            x2={W / 2}
            y2={TIER_Y}
            stroke={COLORS.accent}
            strokeWidth={3.5}
            pathLength={1}
            strokeDasharray={1}
            strokeDashoffset={1 - lineIn(16)}
          />
        </svg>
        {t.sources.map((name, i) => {
          const p = enterAt(frame, fps, i * 4, 12);
          return (
            <div
              key={name}
              style={{
                position: 'absolute',
                left: i === 0 ? 40 : W - 300,
                top: 40,
                width: 260,
                height: 70,
                borderRadius: 22,
                background: COLORS.surface,
                border: '1px solid rgba(255,255,255,0.1)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 14,
                fontSize: 30,
                fontWeight: 700,
                color: COLORS.text,
                opacity: p,
              }}
            >
              {i === 0 ? <AriaGlyph state="idle" size={34} /> : <ColdIcon />}
              {name}
            </div>
          );
        })}

        {/* GateLLM */}
        <div
          style={{
            position: 'absolute',
            left: GATE.x,
            top: GATE.y,
            width: GATE.w,
            height: GATE.h,
            borderRadius: 30,
            background: COLORS.surface,
            border: '1.5px solid rgba(217,119,87,0.6)',
            boxShadow: '0 0 60px rgba(217,119,87,0.15), 0 24px 70px rgba(0,0,0,0.6)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 20,
            opacity: gateIn,
            transform: `scale(${0.92 + 0.08 * gateIn})`,
          }}
        >
          <div style={{width: 64, height: 64, borderRadius: 18, background: COLORS.surface2, display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
            <GateIcon />
          </div>
          <span style={{fontSize: 48, fontWeight: 800, letterSpacing: '-0.02em', color: COLORS.text}}>GateLLM</span>
        </div>

        {/* Tiers */}
        <div
          style={{
            position: 'absolute',
            left: 120,
            top: TIER_Y,
            width: W - 240,
            borderRadius: 28,
            background: 'rgba(28,28,30,0.9)',
            border: '1.5px solid rgba(255,255,255,0.1)',
            padding: 18,
            boxSizing: 'border-box',
            fontFamily: MONO,
            opacity: tiersIn,
            transform: `translateY(${(1 - tiersIn) * 24}px)`,
          }}
        >
          {t.tiers.map((name, i) => {
            const land = i === 0 ? r1Land : i === 2 ? r2Land : 0;
            const fail = i === 0 ? r2Fail : 0;
            return (
              <div
                key={name}
                style={{
                  height: ROW - 14,
                  marginBottom: i < t.tiers.length - 1 ? 14 : 0,
                  borderRadius: 18,
                  background: fail > 0 ? `rgba(255,69,58,${0.18 * fail})` : land > 0 ? `rgba(48,209,88,${0.12 * land})` : COLORS.surface2,
                  border: `1.5px solid ${land > 0 ? `rgba(48,209,88,${0.55 * land})` : 'transparent'}`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '0 26px',
                  fontSize: 32,
                  color: COLORS.text,
                }}
              >
                <span>
                  <span style={{color: COLORS.secondary, marginRight: 16}}>{i + 1}</span>
                  {name}
                </span>
                <span style={{display: 'flex', alignItems: 'center', gap: 14, fontSize: 24}}>
                  {i === 0 && r1Land > 0 ? <span style={{color: COLORS.green, fontWeight: 700, opacity: r1Land}}>{t.requests[0].cost}</span> : null}
                  {i === 2 && r2Land > 0 ? <span style={{color: COLORS.secondary, opacity: r2Land}}>{t.escalated}</span> : null}
                  <Check progress={land} size={36} />
                </span>
              </div>
            );
          })}
        </div>

        {frame >= 30 && r1Land < 1 ? chip(t.requests[0].label, y1, W / 2, interpolate(frame, [30, 34, 50, 56], [0, 1, 1, 0], clamp)) : null}
        {frame >= 62 && r2Land < 1
          ? chip(t.requests[1].label, y2, W / 2, interpolate(frame, [62, 66, 106, 114], [0, 1, 1, 0], clamp), Math.sin(frame * 2.2) * 8 * r2Fail)
          : null}
      </div>

      <div style={{position: 'absolute', left: L.safe.left, top: v ? L.safe.top + 40 : 300, width: v ? L.box.w : 760}}>
        <div style={{...headline(v ? 104 : 92), color: COLORS.text}}>
          <Reveal text={t.headline} stagger={3} />
        </div>
        <div
          style={{
            marginTop: 24,
            fontSize: v ? 34 : 32,
            fontWeight: 500,
            color: COLORS.secondary,
            lineHeight: 1.3,
            opacity: sub,
            transform: `translateY(${(1 - sub) * 12}px)`,
            ...(v ? {position: 'absolute', top: 520 + H * scale - L.safe.top - 20, width: L.box.w, textAlign: 'center'} : {}),
          }}
        >
          {t.subtitle}
        </div>
      </div>
    </AbsoluteFill>
  );
};
