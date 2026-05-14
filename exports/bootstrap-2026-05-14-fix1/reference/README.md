# Operator Console — Design System

A reference for the LSIE-MLF Operator Console design system, grounded in `services/operator_console/design_system/` and `docs/artifacts/OPERATOR_CONSOLE_UI_UX_AUDIT.md` on the `feature/v4-desktop` branch.

## Pages

- **`index.html`** — entry point and overview
- **`foundations.html`** — palette, type, surfaces, status semantics, spacing, radii
- **`primitives.html`** — SectionHeader, MetricCard, StatusPill, AlertBanner, EmptyState, EventTimeline, health rows
- **`action-bar.html`** — the persistent stimulus rail and its six lifecycle states
- **`shells.html`** — SidebarStackShell, MetricGridPlusTimelineShell, TableWithDrillDownShell
- **`operator-language.html`** — readback principles, wrong-vs-right copy, the four enum vocabularies, phrase library
- **`setup-launcher.html`** — first-run + repair surface, now unified with the console palette via `install_setup_stylesheet()`

## Reports

- **`handoff-to-engineering.html`** — UX-01..UX-22 (shipped at commit `a67e1ee`)
- **`runtime-drift.html`** — v2: 7 residual findings observed in 1024×768 runtime captures

## Tokens

`tokens.css` mirrors `services/operator_console/design_system/tokens.py`. All pages reference it. Hex literals live nowhere else.

## Source

- Audit: <https://github.com/ramiradwan/LSIE-MLF/blob/feature/v4-desktop/docs/artifacts/OPERATOR_CONSOLE_UI_UX_AUDIT.md>
- Design system package: <https://github.com/ramiradwan/LSIE-MLF/tree/feature/v4-desktop/services/operator_console/design_system>
