---
name: design-system
description: Use when editing Operator Console UI, PySide6 widgets, theming, QSS, object names, accessibility, or responsive layout. Load the Operator Console design-system artifacts before changing views or widgets.
---

# Operator Console design system

Use this skill whenever work touches:
- `services/operator_console/views/`
- `services/operator_console/widgets/`
- `services/operator_console/theme.py`
- `services/operator_console/design_system/`

## Load these sources first
- `services/operator_console/design_system/DESIGN_SYSTEM.md`
- `services/operator_console/design_system/design_system.json`
- `services/operator_console/design_system/tokens.py`
- `docs/artifacts/OPERATOR_CONSOLE_UI_UX_AUDIT.md`

## Hard rules
1. Do not call `setStyleSheet()` outside `services/operator_console/design_system/`.
2. Do not add hex literals outside `services/operator_console/design_system/tokens.py`.
3. Every QSS-facing widget must use a stable `objectName()` and be represented in `design_system.json`.
4. Prefer shared primitives and compounds before inventing a new widget.
5. Status visuals must map from `packages.schemas.operator_console.UiStatusKind`.
6. Views bind to viewmodels and formatter helpers, not API clients.
7. If a runtime property change affects QSS, use `repolish()` from `design_system.qss_builder`.

## Workflow
1. Confirm whether the change fits an existing shell, primitive, or compound.
2. If it does, reuse the existing object-name and token surface.
3. If it does not, update the manifest and human docs before or alongside the code.
4. Run `python scripts/audit/verifiers/design_system.py --paths services/operator_console` before finishing.

## Reference files
- `references/components.md`
- `references/tokens.md`
- `references/anti-patterns.md`
