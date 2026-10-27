import {loadFont} from '@remotion/fonts';
import {staticFile} from 'remotion';

// Inter and JetBrains Mono (both SIL OFL) are bundled in public/fonts as
// variable fonts, so renders are offline and identical on every machine.
// Same files @remotion/google-fonts would fetch from Google.
export const FONT = 'Inter';
export const MONO = 'JetBrains Mono';

loadFont({family: FONT, url: staticFile('fonts/Inter-Variable-latin.woff2'), weight: '100 900'});
loadFont({family: MONO, url: staticFile('fonts/JetBrainsMono-Variable-latin.woff2'), weight: '100 800'});

/** Headline type: Inter 700, tight tracking. */
export const headline = (size: number): React.CSSProperties => ({
  fontFamily: FONT,
  fontWeight: 700,
  fontSize: size,
  letterSpacing: '-0.03em',
  lineHeight: 1.02,
});
