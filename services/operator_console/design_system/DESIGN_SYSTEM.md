# Operator Console Design System

This package is the **design-system scaffolding** for `services/operator_console`.

It is grounded in the current-state audit at `docs/artifacts/OPERATOR_CONSOLE_UI_UX_AUDIT.md` and exists to make future UI work more retrievable, consistent, and reviewable.

## Canonical files

- `tokens.py` — generated typed palette tokens.
- `tokens.json` — generated machine-readable token export.
- `designer_export_manifest.json` — hash/pointer record for the consumed designer export contract.
- `baselines_manifest.json` — pointer/hash record for approved PySide baseline artifacts; screenshot PNGs stay out of `main`.
- `qss_builder.py` — the only place the Operator Console and launcher setup stylesheets are composed or installed.
- `design_system.json` — machine-readable manifest of shells, components, and registered object names.
- `components.py` — Python registry of shared primitives and compounds.
- `shells.py` — Python registry of current shell patterns.
- `audit.py` — helper utilities used by the design-system verifier.

## Current shared primitives

- `SectionHeader`
- `MetricCard`
- `StatusPill`
- `AlertBanner`
- `EmptyStateWidget`
- `EventTimelineWidget`
- `ResponsiveMetricGrid`
- `LauncherSetupSurface`

## Current compound surface

- `ActionBar`

## Current shell patterns

- `SidebarStackShell`
- `MetricGridPlusTimelineShell`
- `TableWithDrillDownShell`

## Hard rules

1. Do not compose QSS inline in `views/` or `widgets/`.
2. Keep hex literals in `tokens.py` only.
3. Add any new QSS `#ObjectName` selector to `design_system.json`.
4. Use `UiStatusKind` for status mapping rather than ad-hoc string categories.
5. Keep views bound to viewmodels and formatter helpers, not API clients.

## Runtime entry points

The active desktop runtime still enters styling through the existing theme API:
- `services.operator_console.theme.build_stylesheet()`
- `services.operator_console.theme.STYLESHEET`

Those symbols now delegate to this package so current imports keep working while new work can target `services.operator_console.design_system` directly.

## Designer export workflow

Designer-facing HTML/CSS references live outside `main` and must be labeled as `designer-export` or `designer-reference`, not as the PySide design system itself. Engineers consume only a verified export contract: `designer_export_manifest.json` records the export identity and SHA-256 hashes, and `scripts/generate_pyside_tokens.py` refuses to generate when the checked-out export does not match that manifest. The current consumed export is `bootstrap-2026-05-14` from `design-system-spec@6e8a746d06f8e850f9de190779130d301f200dac`.

Use the local check before committing token changes:

```powershell
uv run python scripts/generate_pyside_tokens.py --export-dir services/operator_console/design_system --manifest services/operator_console/design_system/designer_export_manifest.json --check
```

For a future tagged designer export checkout, pass that export directory instead of the bootstrap design-system directory. Do not scrape `tokens.css`; the machine contract is `contract/tokens.json` or `tokens.json` verified by hash.

### How to publish a new designer export

When design changes are ready for engineering, package them into a new immutable export on the `design-system-spec` branch. Designers do not need to calculate hashes by hand.

Designer-ready folder checklist:

- Updated reference files such as `index.html`, `operator-console-interactive.html`, `tokens.css`, and related `.md` notes.
- Machine contracts in either `contract/` or the folder root:
  - `tokens.json`
  - `reference_capture_manifest.json`
  - `reference_to_qt_mapping.json`
- Only approved synthetic/reference images under `uploads/`; do not include real session data, logs, raw captures, audio, video, or biometric media.

Packaging steps for a maintainer or designer:

1. Create or update a local folder that contains the `design-system-spec` branch. This is just a second checkout of this repository, placed anywhere convenient, with the export branch checked out:

   ```powershell
   git clone <repository-url> "$env:USERPROFILE\work\lsie-design-system-spec"
   git -C "$env:USERPROFILE\work\lsie-design-system-spec" checkout design-system-spec
   ```

   If you already have that folder, reuse it and pull the latest `design-system-spec` before packaging.

2. Run the packaging tool from the normal `main` checkout. `--spec-checkout` is the path to the folder from step 1:

   ```powershell
   uv run python scripts/publish_designer_export.py --source "$env:USERPROFILE\Downloads\Design System" --export-id "refresh-2026-06" --spec-checkout "$env:USERPROFILE\work\lsie-design-system-spec"
   ```

The tool copies the reference files, injects `noindex` into HTML pages, validates the three JSON machine contracts, computes SHA-256 hashes, and writes `exports/<export-id>/export_manifest.json`. It refuses to overwrite an existing export id.

Then publish the new immutable export from the `design-system-spec` checkout:

```powershell
git add exports/refresh-2026-06
git commit -m "Publish designer export refresh-2026-06"
git push origin design-system-spec
```

Engineers then switch back to `main`, update `services/operator_console/design_system/designer_export_manifest.json` with the new export id, commit ref, export path, and contract hashes, and run the token generator/check against that export before implementing the matching PySide changes.

## Visual drift workflow

Visual drift has two modes:

- `designer-reference` compares current PySide runtime screenshots against Chromium-rendered designer reference screenshots. This is advisory evidence for design review because Chromium and Qt render text and antialiasing differently.
- `pyside-baseline` compares current PySide runtime screenshots against approved PySide baseline artifacts. This is the only mode that can become blocking after baseline stability is proven.

PySide baseline PNGs must not be committed to `main`. Keep baseline images in an explicitly approved immutable artifact location and record only their hashes and artifact pointer in `baselines_manifest.json`.

Local advisory examples:

```powershell
node scripts/capture_html_reference.js <designer-export-dir> <tmp>\designer-reference
$env:QT_QPA_PLATFORM = "offscreen"; $env:LSIE_SCREENSHOT_DIR = "<tmp>\runtime"; uv run python scripts/render_console_screens.py
node scripts/generate_drift_report.js <tmp>\runtime <tmp>\designer-reference <tmp>\designer-reference-report --mode designer-reference --advisory --threshold 0.02
```

Pull-request CI must remain read-only and artifact-only. `.github/workflows/deploy_designer_portal.yml` handles trusted publication only on `push` to `main` with `contents: read`, `pages: write`, and `id-token: write`; it checks out the pinned designer export commit, regenerates advisory reports, assembles the portal, and deploys with GitHub Pages actions.
