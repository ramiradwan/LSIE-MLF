---
name: implement-phase
description: Execute a validated multi-file local spec-work-item packet. Use only when the user explicitly invokes /implement-phase for a spec-driven implementation slice.
argument-hint: "<automation/work-items/active/*.json | phase label>"
disable-model-invocation: true
---

# Implement phase from a local spec work item

This skill owns multi-file implementation execution from a validated local spec-work-item packet. It does not create durable planning artifacts, infer unrelated scope, or handle single-file packets when `/implement-file` is the better fit.

## Workflow
1. Verify the single signed `docs/tech-spec-v*.pdf` before implementation. If verification fails, stop.
2. Treat `$ARGUMENTS` as either a local work-item path or a phase-like label.
3. If `$ARGUMENTS` is not a path to an existing local packet, create or ask for a local packet under `automation/work-items/active/` before editing code.
4. Validate the packet with `uv run python automation/schemas/spec_work_item.py <path>`.
5. Resolve the packet's `spec_refs` with `scripts/spec_ref_check.py`; use `--validate` when broad validation is needed.
6. Implement only the packet's `target_files`, `acceptance_criteria`, and `forbidden_changes`.
7. Add code comments only for durable, non-obvious WHY constraints.
8. Run the packet's `required_gates` and any relevant repo gates.
9. Before finishing, confirm no generated files under `automation/work-items/active/` are staged.

## Boundaries
- Use `/implement-file` for a packet scoped to one target file.
- Use `spec-ref-check` for reference validation details.
- Use `ux-planning` to produce UX plans and `ui-implementation` to consume validated Operator Console UX plans.
