# Shell catalog

Canonical sources:
- `services/operator_console/design_system/shells.py`
- `services/operator_console/design_system/design_system.json`
- `docs/artifacts/OPERATOR_CONSOLE_UI_UX_AUDIT.md`

## `SidebarStackShell`
Use for full top-level routes that live inside the main Operator Console shell.

Regions:
- sidebar
- stack
- action_bar
- status_bar

## `MetricGridPlusTimelineShell`
Use for monitoring-heavy pages that combine summary cards with timeline or explanatory follow-up surfaces.

Current routes:
- overview
- live_session
- physiology
- health

## `TableWithDrillDownShell`
Use for data tables or lists that stay adjacent to detail, drill-down, or management UI.

Current routes:
- live_session
- experiments
- sessions
