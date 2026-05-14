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
  const advisoryLabel = report.advisory ? 'advisory' : 'blocking';
  const scoreTiles = [
    { number: 'S1', label: 'Compared captures', value: report.results.length, tags: ['runtime', expectedCaptureKind] },
    { number: 'S2', label: 'Within threshold', value: passingCount, tags: ['accepted', `${thresholdPercent}%`] },
    { number: 'S3', label: 'Review needed', value: failingCount, tags: [advisoryLabel, 'visual evidence'] },
  ]
    .map(
      (tile) => `<article class="tile metric-tile">
        <div class="top"><span class="num">${htmlEscape(tile.number)}</span><span>${htmlEscape(tile.label)}</span></div>
        <h3>${htmlEscape(tile.value)}</h3>
        <p>${htmlEscape(tile.label)} in the latest synthetic capture comparison.</p>
        <div class="tags">${tile.tags.map((tag) => `<span class="tag">${htmlEscape(tag)}</span>`).join('')}</div>
      </article>`,
    )
    .join('\n');
  const findings = report.results
    .map((result, index) => {
      const status = result.aboveThreshold ? 'Review needed' : 'Within threshold';
      const statusClass = result.aboveThreshold ? 'status-bad' : 'status-ok';
      const mismatchPercent = (result.mismatchRatio * 100).toFixed(3);
      const actualPath = `../../captures/runtime/${result.file}`;
      const expectedPath = `../../captures/${expectedCaptureKind}/${result.file}`;
      return `<article class="tile capture-tile">
        <div class="top"><span class="num">C${htmlEscape(index + 1)}</span><span>${htmlEscape(result.width)}x${htmlEscape(result.height)} capture</span><span class="arrow ${statusClass}">${htmlEscape(status)}</span></div>
        <h3>${htmlEscape(result.file)}</h3>
        <p>Mismatch ${htmlEscape(mismatchPercent)}% across ${htmlEscape(result.mismatchPixels)} of ${htmlEscape(result.totalPixels)} pixels.</p>
        <div class="tags">
          <span class="tag">threshold ${htmlEscape(thresholdPercent)}%</span>
          <span class="tag">${htmlEscape(expectedCaptureKind)}</span>
          <span class="tag ${statusClass}">${htmlEscape(status)}</span>
        </div>
        <div class="rules metric-rules">
          <div class="rule"><span class="num">M1</span><span class="text">Mismatch ratio <strong>${htmlEscape(mismatchPercent)}%</strong></span></div>
          <div class="rule"><span class="num">M2</span><span class="text">Pixels <strong>${htmlEscape(result.mismatchPixels)} / ${htmlEscape(result.totalPixels)}</strong></span></div>
          <div class="rule"><span class="num">M3</span><span class="text">Threshold <strong>${htmlEscape(thresholdPercent)}%</strong></span></div>
        </div>
        <div class="capture-grid">
          <figure class="src capture-frame">
            <img src="${htmlEscape(actualPath)}" alt="Runtime capture for ${htmlEscape(result.file)}" loading="lazy">
            <figcaption>PySide runtime</figcaption>
          </figure>
          <figure class="src capture-frame">
            <img src="${htmlEscape(expectedPath)}" alt="Expected reference capture for ${htmlEscape(result.file)}" loading="lazy">
            <figcaption>${htmlEscape(expectedCaptureKind)}</figcaption>
          </figure>
          <figure class="src capture-frame">
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
    body { padding: 0; min-height: 100vh; }
    .doc { max-width: 1180px; margin: 0 auto; padding: 64px 56px 96px; }
    .hero {
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 56px;
      align-items: end;
      padding-bottom: 56px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 56px;
    }
    .eyebrow {
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-muted);
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-bottom: 14px;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .eyebrow .dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--accent);
    }
    h1 {
      font-size: 44px;
      font-weight: 600;
      letter-spacing: -0.02em;
      margin: 0 0 18px;
      line-height: 1.05;
    }
    .lede {
      font-size: 15px;
      color: var(--text-muted);
      line-height: 1.6;
      max-width: 560px;
    }
    .hero-meta {
      display: flex;
      flex-direction: column;
      gap: 12px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 20px;
    }
    .hero-meta .row {
      display: grid;
      grid-template-columns: 100px 1fr;
      gap: 12px;
      align-items: baseline;
      font-size: 12px;
      border-bottom: 1px dashed var(--border);
      padding-bottom: 10px;
    }
    .hero-meta .row:last-child { border-bottom: none; padding-bottom: 0; }
    .hero-meta .k {
      color: var(--text-muted);
      font-family: var(--font-mono);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .hero-meta .v {
      color: var(--text-primary);
      line-height: 1.5;
      font-family: var(--font-mono);
    }
    h2 {
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--text-muted);
      margin: 56px 0 18px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border);
    }
    h2:first-of-type { margin-top: 0; }
    h2 + .desc { margin-top: -8px; }
    .desc {
      color: var(--text-muted);
      font-size: 13px;
      line-height: 1.6;
      max-width: 760px;
      margin: 0 0 22px;
    }
    .pages {
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(2, 1fr);
    }
    .tile {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 22px 24px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      text-decoration: none;
      color: inherit;
      cursor: pointer;
      transition: border-color 80ms linear, background 80ms linear;
      min-height: 168px;
    }
    .tile:hover {
      border-color: var(--accent);
      background: var(--surface-raised);
    }
    .tile .top {
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--text-muted);
      font-family: var(--font-mono);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }
    .tile .top .num {
      background: var(--surface-raised);
      padding: 2px 8px;
      border-radius: 3px;
      color: var(--text-primary);
    }
    .tile h3 {
      margin: 0;
      font-size: 18px;
      font-weight: 600;
      color: var(--text-primary);
    }
    .tile p {
      margin: 0;
      font-size: 13px;
      color: var(--text-muted);
      line-height: 1.55;
      flex: 1;
    }
    .tile .tags {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 6px;
    }
    .tile .tag {
      font-family: var(--font-mono);
      font-size: 11px;
      background: var(--bg);
      border: 1px solid var(--border);
      color: var(--text-muted);
      padding: 2px 8px;
      border-radius: 3px;
    }
    .tile:hover .tag {
      border-color: var(--accent);
      color: var(--text-primary);
    }
    .tile .arrow {
      margin-left: auto;
      color: var(--text-muted);
      font-family: var(--font-mono);
    }
    .tile:hover .arrow { color: var(--accent); }
    .rules {
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      overflow: hidden;
    }
    .rules .rule {
      display: grid;
      grid-template-columns: 50px 1fr;
      align-items: baseline;
      padding: 16px 20px;
      border-bottom: 1px solid var(--border);
      gap: 14px;
    }
    .rules .rule:last-child { border-bottom: none; }
    .rules .num {
      font-family: var(--font-mono);
      font-size: 12px;
      color: var(--text-muted);
      background: var(--surface-raised);
      padding: 4px 10px;
      border-radius: 3px;
      width: fit-content;
    }
    .rules .text {
      font-size: 13px;
      color: var(--text-primary);
      line-height: 1.55;
    }
    .src {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 12px 14px;
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-muted);
      line-height: 1.5;
    }
    .foot {
      margin-top: 80px;
      padding-top: 28px;
      border-top: 1px solid var(--border);
      color: var(--text-muted);
      font-size: 12px;
    }
    .metric-tile { min-height: 126px; cursor: default; }
    .metric-tile h3 { font-size: 34px; font-family: var(--font-mono); }
    .finding-pages { grid-template-columns: 1fr; }
    .capture-tile { min-height: 0; cursor: default; }
    .capture-tile .arrow { max-width: none; }
    .status-ok { color: var(--status-ok) !important; }
    .status-bad { color: var(--status-bad) !important; }
    .metric-rules { margin-top: 18px; }
    .metric-rules .text strong {
      color: var(--text-primary);
      font-family: var(--font-mono);
      font-weight: 500;
    }
    .capture-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-top: 18px;
    }
    .capture-frame {
      margin: 0;
      overflow: hidden;
      padding: 0;
    }
    .capture-frame img {
      display: block;
      width: 100%;
      height: auto;
      background: var(--bg);
    }
    .capture-frame figcaption {
      border-top: 1px solid var(--border);
      color: var(--text-muted);
      font-family: var(--font-mono);
      font-size: 11px;
      letter-spacing: 0.06em;
      padding: 10px 12px;
      text-transform: uppercase;
    }
    @media (max-width: 900px) {
      .doc { padding: 40px 22px 72px; }
      .hero { grid-template-columns: 1fr; gap: 28px; }
      .pages { grid-template-columns: 1fr; }
      .capture-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="doc">
    <section class="hero">
      <div>
        <div class="eyebrow"><span class="dot"></span><span>Operator Console visual drift</span></div>
        <h1>Runtime-to-reference screenshot review</h1>
        <p class="lede">This ${htmlEscape(advisoryLabel)} report compares synthetic PySide runtime captures against the ${htmlEscape(expectedCaptureKind)} capture set. Above-threshold rows are review signals while visual baselines stabilize.</p>
      </div>
      <div class="hero-meta">
        <div class="row"><span class="k">Mode</span><span class="v">${htmlEscape(report.mode)}</span></div>
        <div class="row"><span class="k">Gate behavior</span><span class="v">${htmlEscape(advisoryLabel)}</span></div>
        <div class="row"><span class="k">Threshold</span><span class="v">${htmlEscape(thresholdPercent)}%</span></div>
        <div class="row"><span class="k">Generated</span><span class="v">${htmlEscape(report.generatedAt)}</span></div>
      </div>
    </section>

    <h2>Summary</h2>
    <p class="desc">The report uses the same tile, tag, rule, and source-card vocabulary as the designer reference so generated evidence reads as part of the system.</p>
    <div class="pages score-pages">
      ${scoreTiles}
    </div>

    <h2>AI audit placeholder</h2>
    <p class="desc">Human review remains authoritative. Future automated commentary can attach here after the capture contract is stable.</p>
    <div class="rules">
      <div class="rule"><span class="num">A1</span><span class="text">Use the side-by-side captures and pixel drift masks as review evidence.</span></div>
      <div class="rule"><span class="num">A2</span><span class="text">Designer-reference drift is advisory because Chromium and Qt render text differently.</span></div>
    </div>

    <h2>Capture findings</h2>
    <p class="desc">Each capture keeps runtime, expected reference, and pixel mask artifacts together for visual review.</p>
    <div class="pages finding-pages">
      ${findings}
    </div>

    <p class="foot">Generated from synthetic seeded captures only. Do not publish real session data, logs, raw biometric media, or private capture artifacts.</p>
  </div>
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
