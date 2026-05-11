---
name: implement-file
description: Execute a validated local spec-work-item packet for one target file. Use only when the user explicitly invokes /implement-file for a file-scoped implementation slice.
argument-hint: "<target file | automation/work-items/active/*.json>"
disable-model-invocation: true
---

# Implement one file from a local spec work item

This skill owns file-scoped implementation execution from a validated local spec-work-item packet. It does not infer a broader phase, edit unrelated files, or create committed planning artifacts.

## Workflow
1. Verify the single signed `docs/tech-spec-v*.pdf` before implementation. If verification fails, stop.
2. Treat `$ARGUMENTS` as either a target file or a local work-item path.
3. If only a target file is supplied, locate or ask for the local active packet that lists it in `target_files`.
4. Validate the packet with `uv run python automation/schemas/spec_work_item.py <path>`.
5. Confirm the target file is listed in the packet's `target_files`.
6. Resolve the packet's `spec_refs` with `scripts/spec_ref_check.py` before relying on them.
7. Implement only that target file and the corresponding tests/gates named by the packet.
8. Add code comments only for durable, non-obvious WHY constraints.
9. Before finishing, confirm no generated files under `automation/work-items/active/` are staged.

## Boundaries
- Use `/implement-phase` for multi-file packets.
- Use `spec-ref-check` for reference validation details.
- Use `schema-consistency` when the packet changes schema-bearing surfaces.
