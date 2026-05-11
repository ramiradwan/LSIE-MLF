---
name: ui-implementation
description: Use when implementing a validated Operator Console UX plan. Translate the plan into PySide6 view, viewmodel, formatter, and test changes without inventing new design-system primitives.
---

# UI implementation for the Operator Console

Use this skill after a UX plan already exists and validates. This skill consumes the validated UX plan or local work item; it does not invent UX scope or silently revise the plan.

## Load these sources first
- `automation/schemas/ux_plan.py`
- `automation/work-items/templates/ui_implementation.yaml`
- `services/operator_console/design_system/DESIGN_SYSTEM.md`
- `services/operator_console/design_system/design_system.json`
- `references/file-layout.md`
- `references/review-checklist.md`
- `references/spec-refs-rules.md`

## Implementation rules
1. Treat the UX plan as the contract.
2. If the plan is invalid or insufficient, stop and return to UX planning instead of changing scope silently.
3. Reuse registered components and selectors before adding new ones.
4. Keep all styling changes inside `services/operator_console/design_system/`.
5. Keep formatter helpers pure and Qt-free.
6. Keep views bound to viewmodels instead of API clients.
7. Add or update unit tests alongside the implementation.
8. Do not stage generated files under `automation/work-items/active/`.

## Required verification
- `python scripts/audit/verifiers/design_system.py --paths services/operator_console`
- target unit tests under `tests/unit/operator_console/`
- any UX-plan schema tests affected by the change
- **screenshot pass via `uv run python scripts/render_console_screens.py`**;
  read the PNGs in `/tmp/console-screens/` (or `$LSIE_SCREENSHOT_DIR`)
  to confirm the rendered widget matches the UX plan at narrow,
  medium, and wide widths. Unit tests do not catch QSS-vs-CSS
  divergence, font fallbacks, layout cutoff, hover states, or
  background bleed-through.
