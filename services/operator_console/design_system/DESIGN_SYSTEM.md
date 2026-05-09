# Operator Console Design System

This package is the **design-system scaffolding** for `services/operator_console`.

It is grounded in the current-state audit at `docs/artifacts/OPERATOR_CONSOLE_UI_UX_AUDIT.md` and exists to make future UI work more retrievable, consistent, and reviewable.

## Canonical files

- `tokens.py` — typed palette tokens.
- `tokens.json` — machine-readable token export.
- `qss_builder.py` — the only place the main Operator Console stylesheet is composed or installed.
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
