#!/usr/bin/env node
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium } from 'playwright';

const VIEWPORT = { width: 1024, height: 768 };
const DEFAULT_ROUTE_CAPTURES = [
  { route: 'overview', output: 'overview-medium-1024x768.png' },
  { route: 'live-session', output: 'live_session-medium-1024x768.png' },
  { route: 'experiments', output: 'experiments-medium-1024x768.png' },
  { route: 'physiology', output: 'physiology-medium-1024x768.png' },
  { route: 'health', output: 'health-medium-1024x768.png' },
  { route: 'sessions', output: 'sessions-medium-1024x768.png' },
];

function usage() {
  const script = path.basename(fileURLToPath(import.meta.url));
  return `Usage: node scripts/${script} <designer_export_dir> <output_png_dir>`;
}

async function readJson(filePath) {
  const raw = await fs.readFile(filePath, 'utf8');
  const parsed = JSON.parse(raw);
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(`${filePath} must contain a JSON object`);
  }
  return parsed;
}

async function pathExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function sha256(filePath) {
  const content = await fs.readFile(filePath);
  return `sha256:${createHash('sha256').update(content).digest('hex')}`;
}

async function verifyExport(exportDir) {
  const manifestPath = path.join(exportDir, 'export_manifest.json');
  if (!(await pathExists(manifestPath))) {
    return;
  }
  const manifest = await readJson(manifestPath);
  const hashes = manifest.file_hashes ?? manifest.contract_hashes ?? {};
  if (hashes === null || typeof hashes !== 'object' || Array.isArray(hashes)) {
    throw new Error(`${manifestPath} hashes must be an object`);
  }
  for (const [relativePath, expectedHash] of Object.entries(hashes)) {
    if (typeof expectedHash !== 'string' || !expectedHash.startsWith('sha256:')) {
      continue;
    }
    const target = path.join(exportDir, relativePath);
    const actualHash = await sha256(target);
    if (actualHash !== expectedHash) {
      throw new Error(`${relativePath} hash mismatch: expected ${expectedHash}, got ${actualHash}`);
    }
  }
}

async function loadCaptureManifest(exportDir) {
  const manifestPath = path.join(exportDir, 'contract', 'reference_capture_manifest.json');
  if (!(await pathExists(manifestPath))) {
    return { captures: [] };
  }
  const manifest = await readJson(manifestPath);
  const captures = manifest.captures;
  if (!Array.isArray(captures)) {
    throw new Error(`${manifestPath} must define captures[]`);
  }
  return { captures };
}

async function findInteractiveHtml(exportDir) {
  const candidates = [
    path.join(exportDir, 'reference', 'operator-console-interactive.html'),
    path.join(exportDir, 'operator-console-interactive.html'),
  ];
  for (const candidate of candidates) {
    if (await pathExists(candidate)) {
      return candidate;
    }
  }
  throw new Error(`Missing operator-console-interactive.html under ${exportDir}`);
}

async function htmlPath(exportDir, source) {
  const candidates = [path.join(exportDir, 'reference', source), path.join(exportDir, source)];
  for (const candidate of candidates) {
    if (await pathExists(candidate)) {
      return candidate;
    }
  }
  throw new Error(`Missing HTML source ${source}`);
}

async function activateRoute(page, route) {
  const selectors = [
    `[data-route="${route}"]`,
    `[data-page="${route}"]`,
    `[data-route-target="${route}"]`,
    `[href="#${route}"]`,
    `#${route}`,
  ];
  for (const selector of selectors) {
    const locator = page.locator(selector).first();
    if ((await locator.count()) === 0) {
      continue;
    }
    const tagName = await locator.evaluate((node) => node.tagName.toLowerCase()).catch(() => '');
    if (['button', 'a', 'input'].includes(tagName) || selector.startsWith('[href=')) {
      await locator.click();
    } else {
      await locator.evaluate((node) => {
        node.dispatchEvent(new MouseEvent('click', { bubbles: true }));
      });
    }
    await page.waitForTimeout(100);
    return;
  }
  throw new Error(`Could not activate route ${route}`);
}

async function captureRoutes(browser, exportDir, outputDir) {
  const interactiveHtml = await findInteractiveHtml(exportDir);
  const page = await browser.newPage({ viewport: VIEWPORT });
  await page.goto(pathToFileURL(interactiveHtml).href, { waitUntil: 'networkidle' });
  for (const capture of DEFAULT_ROUTE_CAPTURES) {
    await activateRoute(page, capture.route);
    await page.screenshot({ path: path.join(outputDir, capture.output), fullPage: false });
  }
  await page.close();
}

async function captureManifestEntries(browser, exportDir, outputDir, captures) {
  const pages = new Map();
  try {
    for (const capture of captures) {
      const { source, selector, output } = capture;
      if (typeof source !== 'string' || typeof selector !== 'string' || typeof output !== 'string') {
        throw new Error('Each capture manifest entry must define source, selector, and output strings');
      }
      const sourcePath = await htmlPath(exportDir, source);
      let page = pages.get(sourcePath);
      if (page === undefined) {
        page = await browser.newPage({ viewport: VIEWPORT });
        await page.goto(pathToFileURL(sourcePath).href, { waitUntil: 'networkidle' });
        pages.set(sourcePath, page);
      }
      const locator = page.locator(selector).first();
      if ((await locator.count()) === 0) {
        throw new Error(`${source} missing selector ${selector}`);
      }
      await locator.screenshot({ path: path.join(outputDir, output) });
    }
  } finally {
    await Promise.all([...pages.values()].map((page) => page.close()));
  }
}

async function main(argv) {
  if (argv.length !== 2) {
    throw new Error(usage());
  }
  const [rawExportDir, rawOutputDir] = argv;
  const exportDir = path.resolve(rawExportDir);
  const outputDir = path.resolve(rawOutputDir);
  await fs.mkdir(outputDir, { recursive: true });
  await verifyExport(exportDir);
  const manifest = await loadCaptureManifest(exportDir);
  const browser = await chromium.launch();
  try {
    await captureRoutes(browser, exportDir, outputDir);
    await captureManifestEntries(browser, exportDir, outputDir, manifest.captures);
  } finally {
    await browser.close();
  }
  console.log(`Captured designer-reference screenshots in ${outputDir}`);
}

main(process.argv.slice(2)).catch((error) => {
  console.error(`ERROR: ${error.message}`);
  process.exitCode = 1;
});
