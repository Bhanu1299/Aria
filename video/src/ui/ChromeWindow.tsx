import React from 'react';
import {interpolate, useCurrentFrame, useVideoConfig} from 'remotion';
import {COLORS} from '../config';
import {FONT, MONO} from '../theme';
import {clamp, enterAt} from './motion';

export type Tab = {title: string; color: string; letter: string; appearAt: number};

/**
 * Rounded browser window. New tabs slide in from the right; the window itself
 * never comes forward (it's dimmed and labelled "background").
 */
export const ChromeWindow: React.FC<{
  width: number;
  height: number;
  tabs: Tab[];
  active: number;
  url: string;
  dim?: number;
  chip?: string;
  children: React.ReactNode;
}> = ({width, height, tabs, active, url, dim = 0, chip, children}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = width / 920;
  const stripH = 58 * s;
  const barH = 60 * s;
  const tabW = Math.min(210 * s, (width - 120 * s) / Math.max(1, tabs.length));

  return (
    <div
      style={{
        width,
        height,
        borderRadius: 22 * s,
        overflow: 'hidden',
        background: '#202124',
        boxShadow: `0 ${40 * s}px ${120 * s}px rgba(0,0,0,0.7), 0 0 0 1px rgba(255,255,255,0.08)`,
        fontFamily: FONT,
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Tab strip */}
      <div style={{height: stripH, position: 'relative', background: '#1A1A1D', flexShrink: 0}}>
        <div style={{position: 'absolute', left: 22 * s, top: stripH / 2 - 7 * s, display: 'flex', gap: 10 * s}}>
          {['#FF5F57', '#FEBC2E', '#28C840'].map((c) => (
            <div key={c} style={{width: 14 * s, height: 14 * s, borderRadius: '50%', background: c, opacity: 0.55}} />
          ))}
        </div>
        {tabs.map((t, i) => {
          const p = enterAt(frame, fps, t.appearAt, 14);
          if (frame < t.appearAt) return null;
          const x = 100 * s + i * tabW;
          const fromX = width + 40 * s;
          const isActive = i === active;
          return (
            <div
              key={t.title}
              style={{
                position: 'absolute',
                left: interpolate(p, [0, 1], [fromX, x]),
                top: 10 * s,
                width: tabW - 6 * s,
                height: stripH - 10 * s,
                borderRadius: `${12 * s}px ${12 * s}px 0 0`,
                background: isActive ? '#35363A' : 'rgba(255,255,255,0.04)',
                display: 'flex',
                alignItems: 'center',
                gap: 10 * s,
                padding: `0 ${14 * s}px`,
                boxSizing: 'border-box',
                color: isActive ? COLORS.text : COLORS.secondary,
                fontSize: 19 * s,
                fontWeight: 500,
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                opacity: interpolate(p, [0, 0.4], [0, 1], clamp),
              }}
            >
              <div
                style={{
                  width: 20 * s,
                  height: 20 * s,
                  borderRadius: 6 * s,
                  background: t.color,
                  color: '#fff',
                  fontSize: 12 * s,
                  fontWeight: 800,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                }}
              >
                {t.letter}
              </div>
              <span style={{overflow: 'hidden', textOverflow: 'ellipsis'}}>{t.title}</span>
            </div>
          );
        })}
      </div>

      {/* Address bar */}
      <div
        style={{
          height: barH,
          background: '#35363A',
          display: 'flex',
          alignItems: 'center',
          gap: 18 * s,
          padding: `0 ${20 * s}px`,
          flexShrink: 0,
        }}
      >
        <div style={{display: 'flex', gap: 16 * s, color: COLORS.secondary, fontSize: 24 * s}}>
          <span>‹</span>
          <span>›</span>
        </div>
        <div
          style={{
            flex: 1,
            height: 40 * s,
            borderRadius: 20 * s,
            background: '#202124',
            display: 'flex',
            alignItems: 'center',
            padding: `0 ${18 * s}px`,
            fontFamily: MONO,
            fontSize: 18 * s,
            color: '#BDC1C6',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
          }}
        >
          <svg width={14 * s} height={18 * s} viewBox="0 0 14 18" style={{marginRight: 12 * s, opacity: 0.7, flexShrink: 0}}>
            <rect x={1} y={8} width={12} height={9} rx={2} fill="#BDC1C6" />
            <path d="M3.5 8 V5.5 a3.5 3.5 0 0 1 7 0 V8" fill="none" stroke="#BDC1C6" strokeWidth={2} />
          </svg>
          {url}
        </div>
        {chip ? (
          <div
            style={{
              fontSize: 17 * s,
              fontWeight: 600,
              color: COLORS.secondary,
              border: `1px solid rgba(255,255,255,0.18)`,
              borderRadius: 20 * s,
              padding: `${5 * s}px ${14 * s}px`,
              display: 'flex',
              alignItems: 'center',
              gap: 8 * s,
              whiteSpace: 'nowrap',
            }}
          >
            <div style={{width: 8 * s, height: 8 * s, borderRadius: '50%', background: COLORS.secondary}} />
            {chip}
          </div>
        ) : null}
      </div>

      {/* Page */}
      <div style={{flex: 1, position: 'relative', overflow: 'hidden', background: COLORS.page}}>{children}</div>

      {/* Dim: this window is in the background */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          background: `rgba(0,0,0,${dim})`,
          pointerEvents: 'none',
        }}
      />
    </div>
  );
};
