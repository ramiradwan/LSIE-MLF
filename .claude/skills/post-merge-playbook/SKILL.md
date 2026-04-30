---
name: post-merge-playbook
description: Execute the eight Standing Post-Merge Chores plus any cycle-specific chores from docs/POST_MERGE_PLAYBOOK.md after a feature branch lands on main. Triggers include "after merge", "post-merge chores", "run the playbook", "merge cleanup", or "what do I run now that PR X is merged".
---

# Post-Merge Playbook Execution (`docs/POST_MERGE_PLAYBOOK.md`)

## When to invoke

- Immediately after a feature branch is merged to `main` (PR closed, merge commit on `main`).
- When the user says "run the post-merge playbook", "post-merge chores", "after-merge work", or asks what to do now that a merge has happened.
- When the user is preparing the baseline for the ADO agent's next feature cycle.

Do NOT invoke for force-pushes that rewrite history, for hotfix cherry-picks that are not full merges, or for documentation-only PRs (the playbook is overkill for those — instead, run only Standing Chore #2 Documentation Reconciliation).

## Required inputs

Before starting, confirm:

1. The merge commit SHA on `main` (e.g. `git log -1 --first-parent main`).
2. The merge diff (`git diff <prev>..<merge> --stat` and full `git diff <prev>..<merge>`).
3. Which feature branch was merged (PR title / number).

These three facts feed the cycle log and Part 2 (Merge-Specific Chores) of the playbook.

## Execution order

Run the eight Standing Post-Merge Chores **in order**. Each one feeds the next; do not parallelize:

1. **Canonical Terminology Sweep** — run the §0.3 retired-synonym grep from `CLAUDE.md` against `services/`, `packages/`, and `docker-compose.yml`. Zero matches in code/docstrings is the gate. README narrative prose is exempt; the executable §13 audit harness in Chore #8 is the authoritative automated gate for canonical-name verifier results.
2. **Documentation Reconciliation** — diff merge against `README.md`, `.claude/skills/*/SKILL.md`, `CLAUDE.md`, `docs/SPEC_REFERENCE.md`, and `docs/SPEC_AMENDMENTS.md`. Every new container, env var, port, schema field, or Redis key in code must surface in at least one doc. New spec deviations → register a SPEC-AMEND entry.
3. **Schema-Code Consistency Verification** — run `python scripts/check_schema_consistency.py` (see the `schema-consistency` skill) AND verify SQL DDL ↔ Pydantic ↔ Python DDL string ↔ content.json alignment for every persistence table touched. `mypy --strict` must pass.
4. **Test Coverage Gap Analysis** — list new/modified files lacking corresponding tests under `tests/unit/` or `tests/integration/`. Each gap → follow-up test commit OR ADO work item if infra is missing.
5. **Logging and Observability Audit** — grep for emoji log lines, bare `except: pass`, `logger.error()` on §12-recoverable conditions, and log statements at WARNING+ that lack `session_id`/`segment_id`/`subject_role`. Confirm no raw biometric payloads in any log line.
6. **Deferred Integration Inventory Refresh** — walk the diff for new public symbols. Each is either (a) imported by a runtime entrypoint, or (b) added to `docs/DEFERRED_INTEGRATIONS.md` with name, files, gating dependency, deferred-since date, and justification. Re-run the four search recipes from that file's "Search methodology" section.
7. **Performance Baseline Refresh** — only required if the merge touched `orchestrator.py`, `inference.py`, `analytics.py`, `transcription.py`, `face_mesh.py`, or any IPC path. Append a row to `docs/artifacts/performance_baseline.md` with date, commit SHA, segment-assembly p50/p95, ML inference p50/p95, AU12 per-frame p50, and (if physiology touched) Co-Modulation Index window-compute time. Regression > 20% → ADO work item.
8. **§13 Audit Checklist Execution** — run the strict harness and append its Markdown stdout report verbatim to the cycle log:

   ```bash
   python scripts/run_audit.py --strict > /tmp/section13-audit.md
   cat /tmp/section13-audit.md >> docs/artifacts/<cycle-log>.md
   ```

   The harness report is a deterministic Markdown table in runtime-enumerated spec order with stable single-line cells, so adjacent cycle logs should produce meaningful diffs. Every fail must point to a follow-up commit OR a SPEC-AMEND entry.

After Standing Chores complete, execute every chore in **Part 2 — Merge-Specific Chores (Current Cycle)** from the playbook. These are tied to what was just merged and rotate per cycle.

## Outputs to produce

- A short cycle-log block (markdown) summarizing: merge SHA, branch, Standing chores 1–8 pass/fail, Merge-Specific chores M1–MN pass/fail, follow-up commits, and any new ADO work items filed.
- For Standing Chore #6 specifically, an updated `docs/DEFERRED_INTEGRATIONS.md` if any new dormant code was found.
- For Standing Chore #2, any required docs updates committed (or a doc-only fast-follow PR if the merge was already shipped).
- For Standing Chore #8, the strict harness Markdown report from `python scripts/run_audit.py --strict` appended verbatim to the cycle log.

After Part 2 is done, **rewrite the Merge-Specific Chores section of `docs/POST_MERGE_PLAYBOOK.md`** with entries scoped to the next merge — leave it ready for the next cycle.

## Failure handling

- A chore failure does NOT stop the playbook. Record the failure, file the follow-up, and proceed to the next chore. The cycle is closed only when every fail has either a fix commit or a justified deferral.
- If `scripts/check.sh` fails for unrelated reasons (Docker not running, etc.), note it as a `warn` and proceed — do not let environment friction block the audit trail.
- Never bypass a chore by editing the playbook to remove it. Standing chores are stable; if one is genuinely obsolete, that is a separate change with its own justification.

## Cross-references

- Source of truth: `docs/POST_MERGE_PLAYBOOK.md`
- §13 audit checklist: `.claude/commands/audit.md`
- Schema gate: see the `schema-consistency` skill
- Spec deviation registry: `docs/SPEC_AMENDMENTS.md`
- Deferred-integration inventory: `docs/DEFERRED_INTEGRATIONS.md`
