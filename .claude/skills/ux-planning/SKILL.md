---
name: ux-planning
description: Use when planning a new or modified Operator Console page/section. Produce a validated UX plan that only uses registered shells and components from the design-system manifest.
---

# UX planning for the Operator Console

This skill plans UI work before implementation. It does not write the view directly.

## Load these sources first
- `docs/artifacts/OPERATOR_CONSOLE_UI_UX_AUDIT.md`
- `services/operator_console/design_system/design_system.json`
- `automation/schemas/ux_plan.py`
- `references/shell-catalog.md`
- `references/ux-plan-schema.md`

## Planning rules
1. Stay grounded in the current shell and shared component inventory.
2. Use only shell names present in `design_system.json`.
3. Use only component names present in `design_system.json`.
4. Every component instance needs a unique PascalCase `object_name`.
5. Every component instance needs accessibility metadata.
6. Every region needs explicit responsive behavior.
7. Keep operator-facing language routed through formatter helpers.

## Output contract
- Emit a JSON payload that validates with `python automation/schemas/ux_plan.py <path>`.
- Include `spec_refs`, `source_artifacts`, `target_files`, and `target_symbols`.
- Use route values from the current console (`overview`, `live_session`, `experiments`, `physiology`, `health`, `sessions`).

## Handoff
After validation, use the UX plan to create a `ui_implementation` work item based on `automation/work-items/templates/ui_implementation.yaml`.

## Visual verification expectation
The implementer is required to render the actual widgets via
`uv run python scripts/render_console_screens.py` and read the PNGs in
`/tmp/console-screens/` before declaring the plan delivered. Plans that
hinge on a visual claim (alignment, colour, hover state, responsive
band, etc.) should call out the screenshot evidence the implementer
needs to produce.
