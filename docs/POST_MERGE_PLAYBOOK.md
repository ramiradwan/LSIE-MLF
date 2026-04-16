# LSIE-MLF Post-Merge Playbook

Repeatable chores to run after every feature branch is merged into `main`. The playbook has two parts:

1. **Standing Post-Merge Chores** — eight permanent chores that run on every merge regardless of what was merged. These are stable and should not be edited except to refine wording or add new permanent chores.
2. **Merge-Specific Chores (Current Cycle)** — chores tied to what was *just* merged. This section is **rewritten at the end of each merge cycle** so it always describes the most recent merge. Once those chores are completed, the section is wiped and re-populated for the next cycle.

The playbook is operator-facing: each chore should be runnable from a clean checkout of `main` with no prior conversation context.

---

## Part 1 — Standing Post-Merge Chores

### 1. Canonical Terminology Sweep

**Purpose.** Enforce §0.3 of the spec: only the canonical identifiers listed in `CLAUDE.md` (IPC Pipe, Ephemeral Vault, InferenceHandoffPayload, ML Worker, API Server, Message Broker, Persistent Store, Capture Container, Physiological Sample Event, Physiological Context, Physiological State Buffer, subject_role, Co-Modulation Index) may appear in code and docstrings. Retired synonyms (Celery node, GPU worker, FIFO, named pipe, 24-hour vault, handoff schema, FastAPI server, etc.) erode shared vocabulary and confuse the ADO agent's refactor planner.

**Inputs.** The retired-synonym grep from `CLAUDE.md` and the longer extended list in `.claude/commands/audit.md` (item 15), run against `services/`, `packages/`, and `docker-compose.yml`.

**Outputs.** Either a clean (zero-match) report committed to the merge cycle log, or a follow-up commit that renames the offenders. README narrative prose is exempt; code, comments, and docstrings are not.

**Success.** `grep` returns no matches outside of explicitly whitelisted README sections. **Failure.** Any retired synonym found in code or docstrings, or any newly-coined synonym for a §0.3 concept that is not yet in the canonical list.

### 2. Documentation Reconciliation

**Purpose.** Keep `README.md`, `.claude/skills/*/SKILL.md`, `CLAUDE.md`, and `docs/SPEC_REFERENCE.md` aligned with what the merged code actually does. The ADO agent's workspace-server loads these files as ground truth; drift here causes the agent to generate code against a fictional baseline.

**Inputs.** The merge diff, the current contents of all skill files, the README topology table and data-flow diagram, and `docs/SPEC_AMENDMENTS.md`.

**Outputs.** Updated docs in the same merge or a fast-follow doc-only PR. Any new amendment to the signed v3.1 spec (`docs/tech-spec-v3.1.pdf`) must be registered in `docs/SPEC_AMENDMENTS.md` with ID, section, original text, new behavior, rationale, and affected files.

**Success.** Every container, env var, port, schema field, and Redis key referenced in the merged code is described in at least one of the four doc surfaces. **Failure.** Any new artifact that exists in code but is invisible to the workspace-server, or any spec deviation not registered in `SPEC_AMENDMENTS.md`.

### 3. Schema-Code Consistency Verification

**Purpose.** Enforce the hard rule that all inter-module payloads are validated via Pydantic before dispatch and that no untyped dicts cross module boundaries. Catches schemas in `packages/schemas/` that have drifted from their producers/consumers, and SQL DDL in `data/sql/` that no longer matches the Python ORM/insert layer.

**Inputs.** `packages/schemas/`, `services/api/db/schema.py`, `data/sql/*.sql`, and every call site that builds or consumes a payload (orchestrator dispatch, Celery task signatures, API route bodies/responses).

**Outputs.** A column-by-column reconciliation between SQL DDL and Python schema for every persistence table touched, plus a producer/consumer field check for every Pydantic model. Mismatches become bug-fix commits.

**Success.** `mypy packages/ services/ --python-version 3.11 --strict` passes, and every SQL column has a corresponding Pydantic field with a compatible type. **Failure.** A field exists in DDL but not in the Pydantic schema (or vice versa), a producer sends a key the consumer ignores, or a consumer reads a key the producer never sets.

### 4. Test Coverage Gap Analysis

**Purpose.** Every behavior in the merge that the ADO agent might modify during future feature work needs a test that breaks when the behavior changes. Untested code is a silent-regression vector.

**Inputs.** The merge diff (`git diff main~1..main --stat`), the test file inventory under `tests/unit/` and `tests/integration/`, and the §13 audit checklist's coverage criterion.

**Outputs.** A short list of new or modified files that lack corresponding tests. Each entry becomes a follow-up test commit (preferred) or an ADO work item (if the test requires infrastructure not yet available).

**Success.** Every new module under `services/` and `packages/` has at least one unit test exercising its public surface; every new API route has at least one route-level test with mocked dependencies; every new SQL table is exercised by at least one persistence test. **Failure.** New code merged with zero tests, or tests that only assert on import-success without exercising behavior.

### 5. Logging and Observability Audit

**Purpose.** Standardize log levels and structure so that production debugging does not depend on conversational context. Re-checks the Phase 4 cleanup rules: emoji logging is banned, `logger.error()` is reserved for unrecoverable conditions, expected drops/retries use `logger.debug()`, and bare `except Exception: pass` is forbidden.

**Inputs.** `services/`, `packages/`, plus the §12 error-handling matrix from the spec.

**Outputs.** A log-line audit report listing any new emoji entries, any new bare-except blocks, any `logger.error()` calls for §12-recoverable conditions, and any new log statement that does not include enough structured context (session_id, segment_id, subject_role) to be debuggable from a log aggregator alone.

**Success.** Zero emoji log lines, zero silent `except: pass`, and every new log statement at WARNING+ level includes the relevant identifiers. **Failure.** Any of the above, or any new log line that prints a raw payload containing biometric content (would violate the no-raw-media rule even in transient logs).

### 6. Deferred Integration Inventory Refresh

**Purpose.** Track code that is implemented but not yet wired into the runtime path. The canonical example is Module B (`GroundTruthIngester`), which is fully implemented but not called by the orchestrator because it depends on an external EulerStream signature provider. New merges frequently add similar stubs (new routes not mounted, new analytics not invoked, new schemas with no producer). The ADO agent must know which of these are intentionally dormant vs. wiring bugs.

**Inputs.** `docs/DEFERRED_INTEGRATIONS.md`, the merge diff, and a grep for the new modules' import sites.

**Outputs.** An updated `docs/DEFERRED_INTEGRATIONS.md` with one entry per dormant module: name, files, gating dependency, deferred-since date, and justification.

**Success.** Every public symbol added in the merge is either (a) imported by at least one runtime entrypoint, or (b) listed in the deferred inventory with a stated blocker. **Failure.** A new module is dead code with no entry in the deferred inventory and no runtime caller — that is a bug, not a deferral.

### 7. Performance Baseline Refresh

**Purpose.** Re-establish the latency and throughput numbers that the spec's 30 ms ML latency target and Module D segment cadence depend on. Merges that touch `orchestrator.py`, `inference.py`, `analytics.py`, `transcription.py`, `face_mesh.py`, or any IPC path can shift the baseline silently.

**Inputs.** The standard 30 s segment benchmark, the Whisper INT8 inference timer in `services/worker/tasks/inference.py`, the AU12 per-frame timer, and the orchestrator's segment-assembly wall-clock log line. If physiology persistence was touched, also the Co-Modulation Index window-compute timer.

**Outputs.** A small markdown row appended to `docs/artifacts/performance_baseline.md` (create if absent) with date, commit SHA, segment-assembly p50/p95, ML inference p50/p95, AU12 per-frame p50, and any physiology-window compute time. Regressions >20% versus the previous row are flagged for investigation before the cycle is closed.

**Success.** New baseline row recorded and within tolerance of the previous row, OR a regression is filed as an ADO work item with reproduction steps. **Failure.** No baseline run executed, or a regression silently accepted.

### 8. §13 Audit Checklist Execution

**Purpose.** Run the 21-item autonomous implementation audit from §13 of the spec v3.1 (see `.claude/commands/audit.md`). This is the final gate on every merge — the sum of items 1–7 above plus the items they do not cover (canonical name list integrity, requirements pin compliance, image-separation enforcement, `from __future__ import annotations` usage, and the six physiology-specific items 16–21).

**Inputs.** The current state of `main` after the merge, plus any follow-up commits triggered by chores 1–7.

**Outputs.** A 21-line pass/fail table appended to the merge cycle log. Any fail must point to either a follow-up commit or a SPEC-AMEND entry that justifies the deviation.

**Success.** All 21 items pass, OR every fail is justified by a registered amendment. **Failure.** Any unjustified fail, or the audit was not run.

---

## Part 2 — Merge-Specific Chores (Current Cycle)

> **Cycle:** _Awaiting next merge._ The previous cycle (PR 91, `feature/v31-physio-comodulation`, commit `60be7ec`, closed 2026-04-16) is complete — see the cycle log committed under that commit's post-merge run.
>
> When the next feature branch merges into `main`, replace this block in its entirety with a new `> **Cycle:** PR <n> — <branch> → main (commit <sha>, <date>)` header and a fresh set of **M1…Mk** chores derived from that merge's diff. Each M-chore must be self-contained, runnable from a clean checkout, and scoped to exactly what was introduced by the merge (do not restate Standing chores 1–8 here).
>
> **How to populate.** After running the eight Standing chores, walk the merge diff (`git diff <prev>..<merge>`) and create one M-chore per merge-specific risk surface. Typical categories:
>
> - New API routes → verify each response branch and add any missing route tests.
> - New Redis keys, queues, or pub/sub channels → confirm producer/consumer pairing and bounded drain semantics.
> - New SQL tables or columns → column-by-column reconciliation against Pydantic + Python insert layer (this overlaps Standing chore #3 but goes deeper on the new table).
> - New Celery tasks → confirm the worker `-I` include list loads them and that a producer exists.
> - New SPEC-AMEND entries → cross-check affected-files lists against the actual merge scope.
> - Any hard-rule surface touched (no raw biometric persistence, API/worker image separation, `from __future__ import annotations`, canonical names) → a targeted check scoped to the merge's new code paths.
>
> Keep each chore tight: one paragraph describing the risk, the file(s) to inspect, the success criterion, and where the test lives (or should live). When every M-chore has passed or been converted into a follow-up commit / ADO work item, close the cycle and leave this block reset for the merge after next.
