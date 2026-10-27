import {useVideoConfig} from 'remotion';
import {SAFE} from './config';

export type Layout = {
  vertical: boolean;
  W: number;
  H: number;
  safe: {top: number; bottom: number; left: number; right: number};
  /** The rectangle text is allowed to live in. */
  box: {x: number; y: number; w: number; h: number};
  /** Headline size for this format. */
  h1: number;
  h2: number;
  body: number;
};

/** Layout switches on aspect ratio, so both compositions share every scene. */
export const useLayout = (): Layout => {
  const {width: W, height: H} = useVideoConfig();
  const vertical = H > W;
  const safe = vertical ? SAFE.vertical : SAFE.landscape;
  return {
    vertical,
    W,
    H,
    safe,
    box: {x: safe.left, y: safe.top, w: W - safe.left - safe.right, h: H - safe.top - safe.bottom},
    h1: vertical ? 150 : 140,
    h2: vertical ? 92 : 80,
    body: vertical ? 40 : 34,
  };
};

/** Where the recreated macOS menu bar sits in each format. */
export const barGeom = (L: Layout) =>
  L.vertical
    ? {top: L.safe.top, height: 64, insetX: L.safe.left}
    : {top: 0, height: 52, insetX: 40};
