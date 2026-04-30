# LSIE-MLF Implementation Plan

This file holds the **active phased implementation plan** for whatever feature cycle is currently in flight. It is the source of truth consumed by the `/implement-phase` and `/implement-file` slash commands — they read this file to discover which phase to implement and which files belong to it.

**There is no active plan right now.** Populate this file before starting a new multi-phase feature. When the feature ships, leave the plan in place until its cycle is documented (PR merged, post-merge playbook run), then wipe this file back to the scaffold below so the next cycle starts clean.

Historical plans from prior cycles are archived under `docs/artifacts/` — for example `docs/artifacts/HARDENING_SUMMARY.md` records the baseline-hardening sprint that preceded the ADO agent's first feature cycle.

---

## How the slash commands use this file

- `/implement-phase N` — runs the TRUST GATE (`scripts/verify_spec_signature.py`), then reads this file, locates the `## Phase N` heading, implements every file listed under that phase, runs `mypy --strict` per file, adds or updates tests, and runs `pytest -x -q`. The command definition is `.claude/commands/implement-phase.md`.
- `/implement-file <path>` — same trust gate, then implements a single file by tracing back through this file to find which phase it belongs to and confirming its upstream dependencies are implemented. Definition in `.claude/commands/implement-file.md`.

Both commands assume that **every implementation decision traces to a spec section** — the phases you write here must cite the §-references that govern each file. The signed spec is the single `docs/tech-spec-v*.pdf` match; the extracted index is `docs/content.json`; spec amendments live in `docs/SPEC_AMENDMENTS.md`.

---

## Template for an active plan

When you're ready to start a cycle, replace the "no active plan" paragraph above and everything below the `---` with a block shaped like this:

```markdown
**Cycle:** <short feature name> (<YYYY-MM-DD>)
**Goal:** <one sentence — what this cycle delivers and why>
**Spec anchors:** <§-refs the cycle implements, e.g. §4.B.2, §7C>

## Phase 1 — <Phase name>

**Purpose.** <One or two sentences on what this phase produces and why it comes first.>

**Files.**
- `path/to/file_a.py` — <what it implements, spec ref>
- `path/to/file_b.py` — <what it implements, spec ref>

**Depends on.** <Earlier phases or existing modules this phase needs, or "nothing" for phase 1.>

**Done when.** <Concrete, testable criterion — e.g. "all files pass mypy --strict and tests/unit/<area>/ passes".>

## Phase 2 — <Phase name>
...
```

Keep each phase small enough that `/implement-phase N` can plausibly finish it in one run. A phase that touches more than ~6 files or crosses more than 2 modules is usually two phases pretending to be one.

---

## What does NOT belong here

- **Feature cycle retrospectives** — those go in `docs/artifacts/` (frozen point-in-time records) once the cycle closes.
- **Post-merge work for an already-merged PR** — that belongs in `docs/POST_MERGE_PLAYBOOK.md` under Part 2, scoped to the specific merge.
- **Ongoing deferred integrations** — registered in `docs/DEFERRED_INTEGRATIONS.md`, not here.
- **Spec amendments** — registered in `docs/SPEC_AMENDMENTS.md`, not here.

This file is for *work that is actively being planned or implemented in the current cycle*. Once the cycle closes, wipe and reset.
