# Design-system anti-patterns

Block these patterns during UI work:

- Inline `setStyleSheet()` in views or widgets
- New hex literals outside `design_system/tokens.py`
- New `#ObjectName` selectors added in QSS but not represented in `design_system.json`
- Ad-hoc status strings instead of `UiStatusKind`
- Views importing `services.operator_console.api_client`
- Widget-local copy for operator readback when a formatter helper should own the language
- Runtime style changes that mutate stylesheet strings instead of changing properties and repolishing
