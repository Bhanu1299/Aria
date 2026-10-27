/**
 * Every on-screen string, number, placeholder and scene duration lives here.
 * Change text or timing in this file only; scenes read from it.
 *
 * Anything wrapped in {{DOUBLE_BRACES}} is rendered on screen as a visible
 * amber placeholder so it can't slip into a final render unnoticed.
 */

import timeline from '../timeline.json';
import voiceover from '../voiceover.json';
import voDurations from '../public/vo/durations.json';

export const FPS = 30;

export const VERTICAL = {width: 1080, height: 1920} as const;
export const LANDSCAPE = {width: 1920, height: 1080} as const;

/** Text must stay inside these margins. 9:16 numbers come from the Instagram UI. */
export const SAFE = {
  vertical: {top: 120, bottom: 320, left: 80, right: 80},
  landscape: {top: 80, bottom: 80, left: 120, right: 120},
} as const;

export const COLORS = {
  bg: '#000000',
  text: '#F5F5F7',
  secondary: '#86868B',
  surface: '#1C1C1E',
  surface2: '#2C2C2E',
  green: '#30D158',
  /** The one accent. Used only for the diagram lines and "I do." */
  accent: '#D97757',
  placeholder: '#FFB020',
  // Light "web page" palette used inside recreated browser/document UIs.
  page: '#FFFFFF',
  pageText: '#1D1D1F',
  pageMuted: '#6E6E73',
  pageLine: '#E5E5EA',
  pageField: '#F5F5F7',
} as const;

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

/**
 * Beat grid, in frames. Default: 120 BPM (one beat = 15 frames at 30 fps).
 * To sync to your track, either change BPM / FIRST_BEAT, or replace BEATS with
 * the exact frame numbers of the beats you tapped out in Remotion Studio.
 */
export const BPM = 120;
export const FIRST_BEAT = 0;
const TOTAL_GUESS = 3000; // just needs to exceed the video length
export const BEATS: number[] = Array.from(
  {length: Math.ceil(TOTAL_GUESS / ((60 / BPM) * FPS))},
  (_, i) => Math.round(FIRST_BEAT + i * (60 / BPM) * FPS),
);

/** When true, every scene cut is moved to the nearest frame in BEATS. */
export const SNAP_CUTS_TO_BEATS = true;

export type SceneId =
  | 'hook'
  | 'tired'
  | 'combine'
  | 'feed'
  | 'voice'
  | 'resume'
  | 'coldstart'
  | 'system'
  | 'numbers'
  | 'gateway'
  | 'ready'
  | 'cost'
  | 'never'
  | 'endcard';

/**
 * Scene order and lengths live in timeline.json (the music script reads it too).
 * Every scene in SceneId is available; timeline.json picks which ones play.
 */
export const SCENE_SECONDS = timeline.scenes.map((s) => ({...s, id: s.id as SceneId}));

const nearestBeat = (f: number) =>
  BEATS.reduce((best, b) => (Math.abs(b - f) < Math.abs(best - f) ? b : best), BEATS[0]);

export type TimelineEntry = {id: SceneId; from: number; duration: number; still: number};

export const TIMELINE: TimelineEntry[] = (() => {
  const out: TimelineEntry[] = [];
  let cursor = 0;
  let exact = 0;
  for (const s of SCENE_SECONDS) {
    exact += s.seconds * FPS;
    let end = Math.round(exact);
    if (SNAP_CUTS_TO_BEATS) end = Math.max(cursor + 1, nearestBeat(end));
    out.push({id: s.id, from: cursor, duration: end - cursor, still: Math.round(s.still * FPS)});
    cursor = end;
  }
  return out;
})();

export const TOTAL_FRAMES = TIMELINE[TIMELINE.length - 1].from + TIMELINE[TIMELINE.length - 1].duration;

// ---------------------------------------------------------------------------
// Placeholders (fill these in; leave the braces to keep them visible)
// ---------------------------------------------------------------------------

export const PLACEHOLDERS = {
  /** Jobs in the author's tracker (about 180). Set to a number, e.g. 180, to count up to it. */
  JOBS_TRACKED: '{{JOBS_TRACKED}}' as string | number,
  /** Only shown on the end card once the repo is public. Keep null to hide. */
  GATELLM_LINK: null as string | null,
  COLDSTART_LINK: null as string | null,
};

// ---------------------------------------------------------------------------
// Fictional data used inside recreated UIs
// ---------------------------------------------------------------------------

export const PERSON = {
  first: 'Jane',
  last: 'Doe',
  email: 'jane@example.com',
  phone: '+1 (555) 010-0142', // 555-01xx is reserved for fiction
  linkedin: 'linkedin.example/in/jane-doe',
};

export const COMPANIES = [
  {name: 'Acme Robotics', domain: 'acme-robotics.example', color: '#5E5CE6', letter: 'A'},
  {name: 'Northwind', domain: 'northwind.example', color: '#0A84FF', letter: 'N'},
  {name: 'Globex', domain: 'globex.example', color: '#FF9F0A', letter: 'G'},
  {name: 'Initech', domain: 'initech.example', color: '#64D2FF', letter: 'I'},
  {name: 'Umbrella Labs', domain: 'umbrella-labs.example', color: '#FF375F', letter: 'U'},
];

/** Fictional postings from the last 24 hours, in arrival order. `fit` is display-only. */
export const POSTINGS = [
  {company: 0, role: 'Software Engineer', posted: '2h ago', fit: 96},
  {company: 1, role: 'AI Engineer', posted: '5h ago', fit: 84},
  {company: 2, role: 'Backend Engineer', posted: '9h ago', fit: 91},
  {company: 3, role: 'ML Platform Engineer', posted: '14h ago', fit: 78},
  {company: 4, role: 'Software Engineer, Agents', posted: '20h ago', fit: 88},
];
/** Best fit first: the order Aria opens the tabs in. */
export const POSTINGS_BY_FIT = [...POSTINGS].sort((a, b) => b.fit - a.fit);

// ---------------------------------------------------------------------------
// On-screen copy, per scene
// ---------------------------------------------------------------------------

export const TEXT = {
  hook: {
    formTitle: 'Apply for this job',
    roles: ['Software Engineer', 'AI Engineer', 'Backend Engineer'],
    fields: [
      {label: 'First name', value: PERSON.first},
      {label: 'Last name', value: PERSON.last},
      {label: 'Email', value: PERSON.email},
      {label: 'Phone', value: PERSON.phone},
      {label: 'LinkedIn', value: PERSON.linkedin},
    ],
    whyLabel: 'Why do you want to work here?',
    whyValue: 'I have followed your work for a long time and',
    counterLabel: 'jobs tracked',
    submit: 'Submit application',
  },
  tired: {
    headline: 'I got tired\nof this.',
  },
  combine: {
    headline: 'So I combined\nthree of my projects.',
    subtitle: 'to automate my job search',
    tiles: [
      {name: 'Aria', note: 'my voice assistant'},
      {name: 'Coldstart', note: 'outreach'},
      {name: 'GateLLM', note: 'LLM gateway'},
    ],
  },
  feed: {
    headline: 'Every 24 hours:\nnew postings.',
    listTitle: 'New postings',
    sortLabel: 'best fit first',
    fitLabel: 'fit',
  },
  voice: {
    spoken: 'Aria, run the job pipeline.',
    sttNote: 'on-device speech-to-text',
    caption: ['Every 24 hours, Aria fills the new postings.', 'Best fit first. In the background.'],
    steps: ['Fill applications', 'Find contact emails', 'Draft emails'],
    backgroundChip: 'background',
    role: 'Software Engineer',
    url: 'jobs.acme-robotics.example/apply',
    fields: [
      {label: 'First name', value: PERSON.first},
      {label: 'Last name', value: PERSON.last},
      {label: 'Email', value: PERSON.email},
      {label: 'Phone', value: PERSON.phone},
      {label: 'LinkedIn', value: PERSON.linkedin},
    ],
    whyLabel: 'Why Acme?',
    whyLines: [
      'I build voice agents and browser automation,',
      'and your robotics work is where I want to',
      'apply that next.',
    ],
    submit: 'Submit application',
    waiting: 'Ready for review. Nothing submitted.',
  },
  resume: {
    headline: 'Every job gets\nits own resume.',
    postingCompany: 'Northwind',
    postingRole: 'Software Engineer, AI Tools',
    postingHeading: 'What you’ll bring',
    skills: ['Python', 'Playwright', 'LLM agents', 'FastAPI'],
    resumeName: 'Jane Doe',
    resumeContact: 'jane@example.com',
    sections: ['Experience', 'Projects', 'Skills'],
    ats: 98,
    atsLabel: 'ATS',
    atsNote: 'median score',
  },
  coldstart: {
    headline: 'Coldstart finds\nwho to email.',
    appName: 'Coldstart',
    personName: 'Jane Doe',
    personRole: 'Engineering Manager · Northwind',
    patterns: [
      'jane@northwind.example',
      'jdoe@northwind.example',
      'doe.jane@northwind.example',
      'j.doe@northwind.example',
      'janed@northwind.example',
    ],
    found: 'jane.doe@northwind.example',
    verified: 'verified',
    draftPill: 'Draft',
    to: 'jane.doe@northwind.example',
    subject: 'Software Engineer, AI Tools at Northwind',
    caption: 'And drafts it. Never sent without me.',
  },
  system: {
    headline: 'Three projects. One system.',
    nodes: {
      aria: {name: 'Aria', note: 'voice agent'},
      gate: {name: 'GateLLM', note: 'every AI call'},
      cold: {name: 'Coldstart', note: 'right person'},
    },
    tiers: ['small', 'strong', 'frontier'],
    cost: '$0.0004',
  },
  gateway: {
    headline: 'Every AI call goes\nthrough GateLLM.',
    subtitle: 'Cheapest model first. Fallback, retries and cost on every call.',
    sources: ['Aria', 'Coldstart'],
    tiers: ['small', 'strong', 'frontier'],
    requests: [
      {label: 'form field', lands: 0, cost: '$0.0004'},
      {label: 'why us?', lands: 2, cost: ''},
    ],
    escalated: 'checked, escalated',
  },
  ready: {
    headline: 'Everything\nwaits for me.',
    inbox: 'Drafts',
    drafts: [
      {to: 'Engineering Manager', company: 1, subject: 'Software Engineer, AI Tools at Northwind'},
      {to: 'Hiring Manager', company: 0, subject: 'Software Engineer at Acme Robotics'},
      {to: 'Tech Lead', company: 2, subject: 'Backend Engineer at Globex'},
    ],
    tabsLabel: 'Ready for review',
    caption: ['Drafts in my inbox.', 'Filled tabs in my browser.'],
  },
  cost: {
    headline: ['Same work.', '14\u00d7 cheaper.'],
    test: 'Same 87-task stress test',
    rows: [
      {name: 'Claude agent', passed: '78 / 87 passed', cost: 3.01},
      {name: 'DeepSeek via GateLLM', passed: '80 / 87 passed', cost: 0.21},
    ],
  },
  numbers: [
    {value: '2,035', label: 'automated tests passing'},
    {value: '< 2 ms', label: 'everyday commands, no AI call'},
    {value: '5 / 5', label: 'real job sites, filled end to end'},
    {value: '14×', label: 'cheaper on an 87-task stress test'},
  ],
  never: {
    line1: 'It never submits.',
    line2: 'I do.',
  },
  endcard: {
    name: 'Bhanu Teja',
    link: 'github.com/Bhanu1299/Aria',
    roles: 'Open to software / AI engineering roles',
    madeWith: 'made entirely in code',
  },
} as const;

// ---------------------------------------------------------------------------
// Sound (all optional; missing files are simply skipped)
// ---------------------------------------------------------------------------

export const AUDIO = {
  music: 'music.mp3',
  musicVolume: 0.5,
  /** Fade the music out over this many frames at the end. */
  musicFadeOut: 45,
  sfxVolume: 0.35,
};

/** UI sound cues, as [scene, frame-inside-scene, file]. */
export const SFX: {scene: SceneId; at: number; file: string}[] = [
  {scene: 'hook', at: 0, file: 'sfx/type.mp3'},
  {scene: 'tired', at: 0, file: 'sfx/whoosh.mp3'},
  {scene: 'combine', at: 10, file: 'sfx/tick.mp3'},
  {scene: 'voice', at: 140, file: 'sfx/whoosh.mp3'},
  {scene: 'voice', at: 190, file: 'sfx/tick.mp3'},
  {scene: 'resume', at: 150, file: 'sfx/tick.mp3'},
  {scene: 'coldstart', at: 60, file: 'sfx/tick.mp3'},
  {scene: 'system', at: 0, file: 'sfx/whoosh.mp3'},
  {scene: 'numbers', at: 0, file: 'sfx/tick.mp3'},
  {scene: 'numbers', at: 30, file: 'sfx/tick.mp3'},
  {scene: 'numbers', at: 60, file: 'sfx/tick.mp3'},
  {scene: 'numbers', at: 90, file: 'sfx/tick.mp3'},
];

// ---------------------------------------------------------------------------
// Voiceover
// ---------------------------------------------------------------------------

/**
 * Lines, timing and TTS voice live in voiceover.json (shared with scripts/make-voice.py).
 * `at` is the frame inside the scene where the line starts. Clips are public/vo/<id>.mp3;
 * replace any of them with your own recording and keep the name.
 */
export const VOICEOVER = {
  volume: 1,
  /** Music level while a line plays (fraction of musicVolume). */
  duckTo: 0.4,
  duckFrames: 6,
  lines: voiceover.lines.map((l) => ({
    ...l,
    scene: l.scene as SceneId,
    file: `vo/${l.id}.mp3`,
    seconds: (voDurations as Record<string, number>)[l.id] ?? 2,
  })),
};
