// Writes one PNG per scene, per format, to out/stills/ for a quick review.
// Usage: npm run stills                 (both formats)
//        npm run stills -- Vertical     (one format)
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {mkdirSync} from 'node:fs';
import {bundle} from '@remotion/bundler';
import {getCompositions, renderStill} from '@remotion/renderer';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const outDir = path.join(root, 'out', 'stills');
mkdirSync(outDir, {recursive: true});

const only = process.argv[2];
const browserExecutable = process.env.REMOTION_BROWSER || null;

const serveUrl = await bundle({entryPoint: path.join(root, 'src', 'index.ts')});
const comps = await getCompositions(serveUrl, {browserExecutable});

// Per-scene compositions are named like V01-hook / L01-hook.
const scenes = comps.filter((c) => /^[VL]\d\d-/.test(c.id) && (!only || c.id[0] === only[0]));

for (const composition of scenes) {
  const format = composition.id[0] === 'V' ? 'vertical' : 'landscape';
  const output = path.join(outDir, `${format}-${composition.id.slice(1)}.png`);
  const frame = composition.props.still;
  await renderStill({composition, serveUrl, output, frame, browserExecutable});
  console.log(`✓ ${path.relative(root, output)}  (frame ${frame})`);
}
