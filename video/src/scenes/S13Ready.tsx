import React from 'react';
import {AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig} from 'remotion';
import {COLORS, COMPANIES, POSTINGS_BY_FIT, TEXT} from '../config';
import {useLayout} from '../layout';
import {FONT, headline} from '../theme';
import {ChromeWindow, Tab} from '../ui/ChromeWindow';
import {JobForm} from '../ui/JobForm';
import {CameraPush, Check, clamp, enterAt, Reveal} from '../ui/motion';
import {GreyLine} from '../ui/ResumePage';

/** Inbox full of drafts, browser full of filled tabs: all waiting for me. */
export const S13Ready: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const L = useLayout();
  const v = L.vertical;
  const t = TEXT.ready;
  const tv = TEXT.voice;

  const box = v
    ? {inbox: {x: 80, y: 470, w: 920}, browser: {x: 80, y: 900, w: 920, h: 520}, caption: 1450}
    : {inbox: {x: 860, y: 90, w: 940}, browser: {x: 860, y: 480, w: 940, h: 510}, caption: 640};

  const inboxIn = enterAt(frame, fps, 8, 16);
  const browserIn = enterAt(frame, fps, 20, 16);
  const tabs: Tab[] = POSTINGS_BY_FIT.map((p) => {
    const c = COMPANIES[p.company];
    return {title: c.name, color: c.color, letter: c.letter, appearAt: -100};
  });
  const done = tv.fields.map(() => 1);
  const formW = box.browser.w - 80;

  return (
    <AbsoluteFill style={{background: COLORS.bg, fontFamily: FONT}}>
      <CameraPush from={1} to={1.04} origin="60% 55%">
        {/* Inbox */}
        <div
          style={{
            position: 'absolute',
            left: box.inbox.x,
            top: box.inbox.y + (1 - inboxIn) * 60,
            width: box.inbox.w,
            opacity: inboxIn,
            borderRadius: 28,
            background: COLORS.page,
            color: COLORS.pageText,
            overflow: 'hidden',
            boxShadow: '0 30px 90px rgba(0,0,0,0.6)',
          }}
        >
          <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '18px 28px', background: '#F2F2F7'}}>
            <span style={{fontSize: 28, fontWeight: 700}}>{t.inbox}</span>
            <span style={{fontSize: 22, color: COLORS.pageMuted, fontWeight: 600}}>{t.drafts.length}</span>
          </div>
          {t.drafts.map((d, i) => {
            const p = enterAt(frame, fps, 16 + i * 6, 12);
            const c = COMPANIES[d.company];
            return (
              <div
                key={i}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 20,
                  padding: '18px 28px',
                  borderTop: `1px solid ${COLORS.pageLine}`,
                  opacity: p,
                  transform: `translateX(${(1 - p) * 30}px)`,
                }}
              >
                <span
                  style={{
                    background: 'rgba(255,159,10,0.16)',
                    color: '#B25000',
                    border: '1.5px solid rgba(255,159,10,0.6)',
                    borderRadius: 999,
                    padding: '3px 14px',
                    fontSize: 19,
                    fontWeight: 700,
                    flexShrink: 0,
                  }}
                >
                  Draft
                </span>
                <div style={{minWidth: 0, flex: 1}}>
                  <div style={{fontSize: 24, fontWeight: 700, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis'}}>
                    {d.to} · {c.name}
                  </div>
                  <div style={{fontSize: 20, color: COLORS.pageMuted, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis'}}>
                    {d.subject}
                  </div>
                </div>
                <GreyLine w={60} s={1} h={10} />
              </div>
            );
          })}
        </div>

        {/* Browser: every tab filled, waiting */}
        <div style={{position: 'absolute', left: box.browser.x, top: box.browser.y + (1 - browserIn) * 80, opacity: browserIn}}>
          <ChromeWindow width={box.browser.w} height={box.browser.h} tabs={tabs} active={0} url={tv.url} chip={t.tabsLabel}>
            <div style={{position: 'absolute', left: 40, top: -6}}>
              <JobForm
                width={formW}
                company={COMPANIES[POSTINGS_BY_FIT[0].company]}
                title={TEXT.hook.formTitle}
                role={POSTINGS_BY_FIT[0].role}
                fields={tv.fields}
                progress={done}
                checks={done}
                whyLabel={tv.whyLabel}
                whyLines={tv.whyLines}
                whyProgress={1}
                whyCheck={1}
                submit={tv.submit}
                compact
              />
            </div>
          </ChromeWindow>
          {/* every tab gets a ready tick */}
          <div style={{position: 'absolute', left: (100 * box.browser.w) / 920, top: -18, display: 'flex'}}>
            {tabs.map((tab, i) => (
              <div key={tab.title} style={{width: Math.min(210, (box.browser.w - 120) / tabs.length) * (box.browser.w / 920), display: 'flex', justifyContent: 'flex-end', paddingRight: 10}}>
                <Check progress={interpolate(frame, [40 + i * 5, 48 + i * 5], [0, 1], clamp)} size={30} />
              </div>
            ))}
          </div>
        </div>
      </CameraPush>

      <div style={{position: 'absolute', left: L.safe.left, top: v ? L.safe.top + 40 : 160, width: v ? L.box.w : 680}}>
        <div style={{...headline(v ? 120 : 100), color: COLORS.text}}>
          <Reveal text={t.headline} stagger={3} />
        </div>
      </div>
      <div
        style={{
          position: 'absolute',
          left: L.safe.left,
          width: v ? L.box.w : 680,
          top: box.caption,
          textAlign: v ? 'center' : 'left',
          ...headline(v ? 52 : 54),
          fontWeight: 600,
        }}
      >
        <div style={{color: COLORS.text}}>
          <Reveal text={t.caption[0]} delay={60} stagger={3} />
        </div>
        <div style={{color: COLORS.secondary, marginTop: 6}}>
          <Reveal text={t.caption[1]} delay={80} stagger={3} />
        </div>
      </div>
    </AbsoluteFill>
  );
};
