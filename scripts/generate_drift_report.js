#!/usr/bin/env node
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import pixelmatch from 'pixelmatch';
import { PNG } from 'pngjs';

const DEFAULT_THRESHOLD = 0.02;
const PIXELMATCH_OPTIONS = { threshold: 0.1, includeAA: false };
const VALID_MODES = new Set(['pyside-baseline', 'designer-reference']);

function usage() {
  const script = path.basename(fileURLToPath(import.meta.url));
  return [
    `Usage: node scripts/${script} <actual_dir> <expected_dir> <output_dir> [--threshold 0.02] [--advisory] [--mode pyside-baseline|designer-reference]`,
    '',
    'Compares PNGs by exact basename. Advisory mode records threshold failures but exits 0.',
  ].join('\n');
}

function parseArgs(argv) {
  const positional = [];
  const options = {
    threshold: DEFAULT_THRESHOLD,
    advisory: false,
    mode: 'designer-reference',
    allowMissing: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--advisory') {
      options.advisory = true;
      continue;
    }
    if (arg === '--threshold') {
      index += 1;
      if (index >= argv.length) {
        throw new Error('--threshold requires a value');
      }
      options.threshold = Number(argv[index]);
      if (!Number.isFinite(options.threshold) || options.threshold < 0 || options.threshold > 1) {
        throw new Error('--threshold must be a number between 0 and 1');
      }
      continue;
    }
    if (arg === '--mode') {
      index += 1;
      if (index >= argv.length) {
        throw new Error('--mode requires a value');
      }
      options.mode = argv[index];
      if (!VALID_MODES.has(options.mode)) {
        throw new Error(`--mode must be one of: ${[...VALID_MODES].join(', ')}`);
      }
      continue;
    }
    if (arg === '--allow-missing') {
      options.allowMissing = true;
      continue;
    }
    if (arg.startsWith('--')) {
      throw new Error(`Unknown option ${arg}`);
    }
    positional.push(arg);
  }
  if (positional.length !== 3) {
    throw new Error(usage());
  }
  return {
    actualDir: path.resolve(positional[0]),
    expectedDir: path.resolve(positional[1]),
    outputDir: path.resolve(positional[2]),
    ...options,
  };
}

async function listPngs(directory) {
  const entries = await fs.readdir(directory, { withFileTypes: true });
  return entries
    .filter((entry) => entry.isFile() && entry.name.toLowerCase().endsWith('.png'))
    .map((entry) => entry.name)
    .sort();
}

async function readPng(filePath) {
  const buffer = await fs.readFile(filePath);
  return PNG.sync.read(buffer);
}

async function writePng(filePath, png) {
  await fs.writeFile(filePath, PNG.sync.write(png));
}

function htmlEscape(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

async function comparePair(actualPath, expectedPath, outputPath) {
  const actual = await readPng(actualPath);
  const expected = await readPng(expectedPath);
  if (actual.width !== expected.width || actual.height !== expected.height) {
    throw new Error(
      `${path.basename(actualPath)} dimension mismatch: actual ${actual.width}x${actual.height}, expected ${expected.width}x${expected.height}`,
    );
  }
  const diff = new PNG({ width: actual.width, height: actual.height });
  const mismatchPixels = pixelmatch(
    actual.data,
    expected.data,
    diff.data,
    actual.width,
    actual.height,
    PIXELMATCH_OPTIONS,
  );
  await writePng(outputPath, diff);
  const totalPixels = actual.width * actual.height;
  return {
    mismatchPixels,
    totalPixels,
    mismatchRatio: totalPixels === 0 ? 0 : mismatchPixels / totalPixels,
    width: actual.width,
    height: actual.height,
  };
}

function renderHtmlReport(report) {
  const failingCount = report.results.filter((result) => result.aboveThreshold).length;
  const passingCount = report.results.length - failingCount;
  const expectedCaptureKind = report.mode === 'pyside-baseline' ? 'pyside-baseline' : 'designer-reference';
  const thresholdPercent = (report.threshold * 100).toFixed(3);
  const findings = report.results
    .map((result) => {
      const status = result.aboveThreshold ? 'Review needed' : 'Within threshold';
      const statusClass = result.aboveThreshold ? 'bad' : 'ok';
      const mismatchPercent = (result.mismatchRatio * 100).toFixed(3);
      const actualPath = `../../captures/runtime/${result.file}`;
      const expectedPath = `../../captures/${expectedCaptureKind}/${result.file}`;
      return `<article class="find">
    <header>
      <div>
        <p class="eyebrow">${htmlEscape(result.width)}x${htmlEscape(result.height)} capture</p>
        <h2>${htmlEscape(result.file)}</h2>
      </div>
      <span class="status ${statusClass}">${htmlEscape(status)}</span>
    </header>
    <dl class="metrics">
      <div><dt>Mismatch</dt><dd>${htmlEscape(mismatchPercent)}%</dd></div>
      <div><dt>Pixels</dt><dd>${htmlEscape(result.mismatchPixels)} / ${htmlEscape(result.totalPixels)}</dd></div>
      <div><dt>Threshold</dt><dd>${htmlEscape(thresholdPercent)}%</dd></div>
    </dl>
    <div class="capture-grid">
      <figure>
        <img src="${htmlEscape(actualPath)}" alt="Runtime capture for ${htmlEscape(result.file)}" loading="lazy">
        <figcaption>PySide runtime</figcaption>
      </figure>
      <figure>
        <img src="${htmlEscape(expectedPath)}" alt="Expected reference capture for ${htmlEscape(result.file)}" loading="lazy">
        <figcaption>${htmlEscape(expectedCaptureKind)}</figcaption>
      </figure>
      <figure>
        <img src="${htmlEscape(result.diffFile)}" alt="Pixel drift mask for ${htmlEscape(result.file)}" loading="lazy">
        <figcaption>Pixel drift mask</figcaption>
      </figure>
    </div>
  </article>`;
    })
    .join('\n');

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="robots" content="noindex, nofollow">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Operator Console Visual Drift Report</title>
  <link rel="stylesheet" href="../../tokens.css">
  <style>
    :root { color-scheme: dark; }
    body {
      margin: 0;
      background: var(--color-background, Canvas);
      color: var(--color-text-primary, CanvasText);
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    }
    main { max-width: 1180px; margin: 0 auto; padding: 40px 24px 64px; }
    a { color: var(--color-accent, LinkText); }
    .hero, .audit, .find {
      border: 1px solid color-mix(in srgb, CanvasText 16%, transparent);
      border-radius: 20px;
      background: color-mix(in srgb, Canvas 88%, CanvasText 12%);
      box-shadow: 0 18px 48px color-mix(in srgb, CanvasText 20%, transparent);
    }
    .hero { padding: 28px; margin-bottom: 20px; }
    .eyebrow {
      margin: 0 0 8px;
      color: var(--color-text-muted, GrayText);
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }
    h1, h2, h3 { margin: 0; line-height: 1.1; }
    h1 { font-size: clamp(2rem, 4vw, 4rem); max-width: 820px; }
    .lede { color: var(--color-text-muted, GrayText); font-size: 1.05rem; max-width: 820px; }
    .meta { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }
    .chip, .status {
      border: 1px solid color-mix(in srgb, currentColor 26%, transparent);
      border-radius: 999px;
      padding: 7px 11px;
      font-size: 0.82rem;
      font-weight: 700;
    }
    .status.ok { color: var(--color-status-ok, CanvasText); }
    .status.bad { color: var(--color-status-bad, CanvasText); }
    .scorecard {
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      margin: 20px 0;
    }
    .scorecard div {
      border: 1px solid color-mix(in srgb, CanvasText 14%, transparent);
      border-radius: 18px;
      padding: 18px;
      background: color-mix(in srgb, Canvas 82%, CanvasText 18%);
    }
    .scorecard dt { color: var(--color-text-muted, GrayText); font-size: 0.8rem; font-weight: 700; text-transform: uppercase; }
    .scorecard dd { margin: 8px 0 0; font-size: 2rem; font-weight: 800; }
    .audit { margin: 20px 0 28px; padding: 22px; }
    .audit p { color: var(--color-text-muted, GrayText); margin-bottom: 0; }
    .find { margin-top: 18px; padding: 20px; }
    .find header { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; }
    .metrics {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      margin: 18px 0;
    }
    .metrics div { border-left: 3px solid currentColor; padding-left: 12px; }
    .metrics dt { color: var(--color-text-muted, GrayText); font-size: 0.78rem; font-weight: 700; text-transform: uppercase; }
    .metrics dd { margin: 5px 0 0; font-size: 1.05rem; font-weight: 700; }
    .capture-grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
    figure {
      margin: 0;
      overflow: hidden;
      border: 1px solid color-mix(in srgb, CanvasText 14%, transparent);
      border-radius: 16px;
      background: Canvas;
    }
    img { display: block; width: 100%; height: auto; }
    figcaption {
      border-top: 1px solid color-mix(in srgb, CanvasText 14%, transparent);
      padding: 10px 12px;
      color: var(--color-text-muted, GrayText);
      font-size: 0.82rem;
      font-weight: 700;
    }
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <p class="eyebrow">Operator Console visual drift</p>
      <h1>Runtime-to-reference screenshot review</h1>
      <p class="lede">This advisory report compares synthetic PySide runtime captures against the ${htmlEscape(expectedCaptureKind)} capture set. Above-threshold rows are review signals, not blocking failures while visual baselines stabilize.</p>
      <div class="meta">
        <span class="chip">Mode: ${htmlEscape(report.mode)}</span>
        <span class="chip">Advisory: ${report.advisory ? 'yes' : 'no'}</span>
        <span class="chip">Threshold: ${htmlEscape(thresholdPercent)}%</span>
        <span class="chip">Generated: ${htmlEscape(report.generatedAt)}</span>
      </div>
    </section>

    <dl class="scorecard">
      <div><dt>Total captures</dt><dd>${htmlEscape(report.results.length)}</dd></div>
      <div><dt>Within threshold</dt><dd>${htmlEscape(passingCount)}</dd></div>
      <div><dt>Review needed</dt><dd>${htmlEscape(failingCount)}</dd></div>
    </dl>

    <section class="audit">
      <p class="eyebrow">AI Audit Placeholder</p>
      <h2>Human review remains authoritative</h2>
      <p>Future automated commentary can attach here after the capture contract is stable. For now, use the scorecard, side-by-side captures, and pixel drift masks as review evidence.</p>
    </section>

    ${findings}
  </main>
</body>
</html>
`;
}

async function main(argv) {
  const args = parseArgs(argv);
  await fs.mkdir(args.outputDir, { recursive: true });
  const actualPngs = await listPngs(args.actualDir);
  const expectedPngs = await listPngs(args.expectedDir);
  const expectedSet = new Set(expectedPngs);
  const missingExpected = actualPngs.filter((file) => !expectedSet.has(file));
  if (missingExpected.length > 0 && !args.allowMissing) {
    throw new Error(`Missing expected PNGs: ${missingExpected.join(', ')}`);
  }
  const pairs = actualPngs.filter((file) => expectedSet.has(file));
  if (pairs.length === 0) {
    throw new Error('No matching PNG basenames found');
  }

  const results = [];
  for (const file of pairs) {
    const diffFile = `diff-${file}`;
    const comparison = await comparePair(
      path.join(args.actualDir, file),
      path.join(args.expectedDir, file),
      path.join(args.outputDir, diffFile),
    );
    results.push({
      file,
      diffFile,
      ...comparison,
      threshold: args.threshold,
      aboveThreshold: comparison.mismatchRatio > args.threshold,
    });
  }

  const report = {
    version: 1,
    mode: args.mode,
    advisory: args.advisory,
    threshold: args.threshold,
    actualSource: path.basename(args.actualDir),
    expectedSource: path.basename(args.expectedDir),
    generatedAt: new Date().toISOString(),
    results,
    summary: {
      compared: results.length,
      aboveThreshold: results.filter((result) => result.aboveThreshold).length,
    },
  };
  await fs.writeFile(
    path.join(args.outputDir, 'drift-report.json'),
    `${JSON.stringify(report, null, 2)}\n`,
    'utf8',
  );
  await fs.writeFile(path.join(args.outputDir, 'runtime-drift.html'), renderHtmlReport(report), 'utf8');
  console.log(
    `Compared ${report.summary.compared} PNGs; ${report.summary.aboveThreshold} above threshold in ${args.outputDir}`,
  );
  if (!args.advisory && report.summary.aboveThreshold > 0) {
    process.exitCode = 1;
  }
}

main(process.argv.slice(2)).catch((error) => {
  console.error(`ERROR: ${error.message}`);
  process.exitCode = 1;
});
