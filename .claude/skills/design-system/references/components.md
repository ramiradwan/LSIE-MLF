# Shared components and shells

Canonical source: `services/operator_console/design_system/design_system.json`

## Shells
- `SidebarStackShell` — sidebar, stacked page body, persistent ActionBar, status bar
- `MetricGridPlusTimelineShell` — header + summary grid + timeline/follow-up surface
- `TableWithDrillDownShell` — header + table/list + detail/manage region

## Primitives
- `MetricCard`
- `StatusPill`
- `AlertBanner`
- `EmptyStateWidget`
- `EventTimelineWidget`
- `SectionHeader`
- `ResponsiveMetricGrid`

## Compound
- `ActionBar`

## Current selector-only surfaces
These are not standalone widgets but do carry stable QSS object names:
- `ContentSurface`
- `SidebarNav`
- `SidebarTitle`
- `SidebarSubtitle`
- `NavButton`
- `Panel`
- `PanelTitle`
- `PanelSubtitle`
- `StatusBarLabel`
- `HealthRowOk`
- `HealthRowWarn`
- `HealthRowBad`
- `HealthRowDegraded`
- `HealthRowRecovering`
