---
name: software-engineering-team
description: Create an experimental agent team from validated local spec-work-item packets. Use only when the user explicitly invokes /software-engineering-team or asks to create a software engineering team for spec-driven implementation.
argument-hint: "<automation/work-items/active/*.json | brief team goal>"
disable-model-invocation: true
---

# Software engineering team from local spec work items

This skill starts a coordinated agent-team workflow for LSIE-MLF implementation cycles. The team lead remains the only code-writing owner by default; teammates provide read-only review and direction. The lead iterates implementation based on teammate feedback until the whole team agrees the result is ready.

## Preconditions

1. Agent teams must be enabled with `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` in the environment or Claude Code settings.
2. Work must be represented by one or more local JSON packets under `automation/work-items/active/`.
3. Each packet must validate with `uv run python automation/schemas/spec_work_item.py <path>`.
4. Each packet's `spec_refs` must resolve through `scripts/spec_ref_check.py` or the `spec-ref-check` workflow.
5. Generated active packets are local-only and must not be committed.

If the user supplies a brief goal instead of a packet path, create or ask for local packets before starting implementation work. Do not create committed planning docs.

## Team shape

Default reviewer teammates:

- `ux-ui-specialist` — Opus, read-only. Reviews Operator Console/UI implications, accessibility, design-system fit, and visual verification needs.
- `product-designer` — Opus, read-only. Reviews operator-facing flow, terminology, product coherence, and whether the work solves the intended user problem.
- `spec-compliance-reviewer` — Opus, read-only. Reviews signed-spec constraints, canonical terminology, work-item scope, forbidden changes, and gate alignment.

Optional additional teammates may be spawned only when the work justifies them:

- `spec-work-item-auditor` for packet quality/hygiene.
- `dormant-surface-auditor` for guarded activation risks.
- `gate-failure-triager` for noisy failing gates.

By default, do not spawn code-writing teammates. If the user explicitly requests implementation teammates, split files so each writer owns disjoint paths and require plan approval before edits.

## Workflow

1. Validate the packet(s) and resolve spec references before `TeamCreate`.
2. Create a team named for the implementation slice.
3. Convert packet contents into shared tasks:
   - one implementation task for the lead covering the target files;
   - one review task per default reviewer teammate;
   - one verification task covering `acceptance_criteria.required_gates` and hygiene checks.
4. Spawn reviewer teammates using the project subagent definitions named above.
5. Give each reviewer the same packet path(s), the specific review lens, and a requirement to return actionable feedback with file paths and gate references.
6. The lead implements the packet target files in the main session or lead context only.
7. After each implementation pass, send the diff summary to all reviewer teammates and ask whether they approve or have blocking feedback.
8. Iterate until all reviewers either approve or identify only non-blocking follow-ups accepted by the user.
9. Run the packet's `required_gates`, relevant repo gates, and `git check-ignore -v` for active packet paths.
10. Before finishing, confirm no files under `automation/work-items/active/` are staged.
11. Shut down teammates gracefully and clean up the team when work is complete.

## Non-overlap rules

- This skill coordinates a team; it does not replace `/implement-file` or `/implement-phase` for execution boundaries.
- `spec-ref-check` owns § reference validation details.
- `schema-consistency` owns schema/DDL/Pydantic consistency.
- `ux-planning` owns UX plan JSON production.
- `ui-implementation` owns implementation from validated Operator Console UX plans.
- `design-system` owns design-system rules and visual verification guidance.
- `tests/unit/automation/test_deferred_integration_guards.py` and `scripts/run_audit.py --strict` own dormant-surface executable protection.

## Consensus rule

The lead must not call the implementation complete while any reviewer has unresolved blocking feedback. If reviewers disagree, the lead summarizes the disagreement, chooses the smallest spec-compliant path, and asks the dissenting reviewer for a final check. If disagreement remains, stop and ask the user for the decision rather than merging conflicting guidance silently.

## Output

When the team finishes, report:

- packet path(s) implemented;
- teammates spawned;
- reviewer approval/blocker status;
- files changed;
- gates run and results;
- active work-item hygiene status;
- any non-blocking follow-ups.
