# LSIE-MLF Post-Merge Playbook

Repeatable chores to run after every feature branch is merged into `main`. The playbook has two parts:

1. **Standing Post-Merge Chores** — eight permanent chores that run on every merge regardless of what was merged. These are stable and should not be edited except to refine wording or add new permanent chores.
2. **Merge-Specific Chores (Current Cycle)** — chores tied to what was *just* merged. This section is **rewritten at the end of each merge cycle** so it always describes the most recent merge. Once those chores are completed, the section is wiped and re-populated for the next cycle.

The playbook is operator-facing: each chore should be runnable from a clean checkout of `main` with no prior conversation context.

---

## Part 1 — Standing Post-Merge Chores

### 1. Canonical Terminology Sweep

**Purpose.** Enforce §0.3 of the spec: only the canonical identifiers listed in `CLAUDE.md` (IPC Pipe, Ephemeral Vault, InferenceHandoffPayload, ML Worker, API Server, Message Broker, Persistent Store, Capture Container, Physiological Sample Event, Physiological Context, Physiological State Buffer, subject_role, Co-Modulation Index) may appear in code and docstrings. Retired synonyms (Celery node, GPU worker, FIFO, named pipe, 24-hour vault, handoff schema, FastAPI server, etc.) erode shared vocabulary and confuse the ADO agent's refactor planner.

**Inputs.** The retired-synonym grep from `CLAUDE.md` and the longer Phase 6 list in `IMPLEMENTATION_PLAN.md`, run against `services/`, `packages/`, and `docker-compose.yml`.

**Outputs.** Either a clean (zero-match) report committed to the merge cycle log, or a follow-up commit that renames the offenders. README narrative prose is exempt; code, comments, and docstrings are not.

**Success.** `grep` returns no matches outside of explicitly whitelisted README sections. **Failure.** Any retired synonym found in code or docstrings, or any newly-coined synonym for a §0.3 concept that is not yet in the canonical list.

### 2. Documentation Reconciliation

**Purpose.** Keep `README.md`, `.claude/skills/*/SKILL.md`, `CLAUDE.md`, and `docs/SPEC_REFERENCE.md` aligned with what the merged code actually does. The ADO agent's workspace-server loads these files as ground truth; drift here causes the agent to generate code against a fictional baseline.

**Inputs.** The merge diff, the current contents of all skill files, the README topology table and data-flow diagram, and `docs/SPEC_AMENDMENTS.md`.

**Outputs.** Updated docs in the same merge or a fast-follow doc-only PR. Any new amendment to the signed v3.0 spec must be registered in `docs/SPEC_AMENDMENTS.md` with ID, section, original text, new behavior, rationale, and affected files.

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

**Inputs.** The "What This Plan Does NOT Cover" section of `IMPLEMENTATION_PLAN.md`, the merge diff, and a grep for the new modules' import sites.

**Outputs.** An updated deferred-integration list in `IMPLEMENTATION_PLAN.md` (or a sibling `DEFERRED.md`), with one entry per dormant module: name, location, what it does, what blocks integration, and the ADO work item ID once filed.

**Success.** Every public symbol added in the merge is either (a) imported by at least one runtime entrypoint, or (b) listed in the deferred inventory with a stated blocker. **Failure.** A new module is dead code with no entry in the deferred inventory and no runtime caller — that is a bug, not a deferral.

### 7. Performance Baseline Refresh

**Purpose.** Re-establish the latency and throughput numbers that the spec's 30 ms ML latency target and Module D segment cadence depend on. Merges that touch `orchestrator.py`, `inference.py`, `analytics.py`, `transcription.py`, `face_mesh.py`, or any IPC path can shift the baseline silently.

**Inputs.** The standard 30 s segment benchmark, the Whisper INT8 inference timer in `services/worker/tasks/inference.py`, the AU12 per-frame timer, and the orchestrator's segment-assembly wall-clock log line. If physiology persistence was touched, also the Co-Modulation Index window-compute timer.

**Outputs.** A small markdown row appended to `docs/artifacts/performance_baseline.md` (create if absent) with date, commit SHA, segment-assembly p50/p95, ML inference p50/p95, AU12 per-frame p50, and any physiology-window compute time. Regressions >20% versus the previous row are flagged for investigation before the cycle is closed.

**Success.** New baseline row recorded and within tolerance of the previous row, OR a regression is filed as an ADO work item with reproduction steps. **Failure.** No baseline run executed, or a regression silently accepted.

### 8. §13 Audit Checklist Execution

**Purpose.** Run the 21-item autonomous implementation audit from §13 of the spec (see `.claude/commands/audit.md`). This is the final gate on every merge — the sum of items 1–7 above plus the few §13 items they do not cover (canonical name list integrity, requirements pin compliance, image-separation enforcement, `from __future__ import annotations` usage, etc.).

**Inputs.** The current state of `main` after the merge, plus any follow-up commits triggered by chores 1–7.

**Outputs.** A 15-line pass/fail table appended to the merge cycle log. Any fail must point to either a follow-up commit or a SPEC-AMEND entry that justifies the deviation.

**Success.** All 15 items pass, OR every fail is justified by a registered amendment. **Failure.** Any unjustified fail, or the audit was not run.

---

## Part 2 — Merge-Specific Chores (Current Cycle)

> **Cycle:** PR 91 — `feature/v31-physio-comodulation` → `main` (commit `60be7ec`, 2026-04-16).
> This section is rewritten at the end of each merge cycle. The chores below are specific to what PR 91 introduced and should be executed in addition to the eight Standing chores above.

### M1. Verify Oura Webhook Signature Path End-to-End

The new `POST /api/v1/ingest/oura/webhook` route in `services/api/routes/physiology.py` validates `x-oura-signature` using `OURA_WEBHOOK_SECRET`. Confirm the secret is documented in `.env.example`, that signature mismatch returns 401 (not 422), that duplicate `event_id` collisions via `physio:seen:*` `SETNX` return the documented response, and that a missing `subject_role` returns 422. Add a route-level test for each branch if missing from `tests/unit/api/test_physiology.py`.

### M2. Confirm Redis `physio:events` Drain is Bounded and Non-Blocking

Per SPEC-AMEND-007 and §4.C.4, the orchestrator's drain from `physio:events` must be non-blocking `LPOP`-based and bounded per tick — it cannot stall the segment-assembly loop if the list grows. Read `services/worker/pipeline/orchestrator.py` around the new physiology-drain code, verify the drain has a per-tick cap, and confirm there is no `BLPOP` or unbounded `while True: LPOP` pattern. Add a test that proves the drain returns control to the run loop within a fixed iteration count even when the list is artificially long.

### M3. Validate `freshness_s` Is Wall-Clock, Not Device Time

§4.C.4 specifies that `freshness_s` is computed at `assemble_segment()` wall-clock time, **not** ADB drift-corrected device time, and that stale snapshots remain present with `is_stale=True` rather than being dropped. Inspect the freshness computation site in `orchestrator.py` and the `Physiological Context` builder in `packages/schemas/physiology.py`. Confirm the timestamp source is `datetime.now(tz=UTC)` (or equivalent) and that `is_stale` is set rather than the snapshot being elided. Add a unit test that injects a stale snapshot and asserts it propagates with `is_stale=True`.

### M4. Reconcile `physiology_log` and `comodulation_log` DDL with Python Insert Layer

PR 91 added `data/sql/03-physiology.sql` and the corresponding insert paths in `services/api/db/schema.py` and `services/worker/pipeline/analytics.py`. Run a column-by-column diff: every SQL column must have a producer in the Python layer and a corresponding field on the relevant Pydantic schema, and every Pydantic field that should land in PostgreSQL must have a column. Pay particular attention to `rmssd_ms`, `heart_rate_bpm`, `freshness_s`, `is_stale`, `provider`, `source_timestamp`, and the rolling Co-Modulation Index column types (DOUBLE PRECISION, TIMESTAMPTZ).

### M5. Confirm Co-Modulation Index Returns Null for Insufficient Pairs

§7C requires that the rolling Pearson correlation over aligned 5-minute RMSSD bins for `streamer` and `operator` returns null when there are insufficient aligned non-stale pairs for the window. Read the new analytics code in `services/worker/pipeline/analytics.py`, identify the threshold constant, and verify there is a unit test in `tests/unit/worker/pipeline/test_physiology_analytics.py` that covers (a) below-threshold returns null, (b) at-threshold returns a value, (c) any pair flagged `is_stale=True` is excluded from the count.

### M6. Verify API/Worker Image Separation Was Not Violated

The hard rule in `CLAUDE.md` is that the API Server image excludes ML dependencies and the Worker image excludes web assets. PR 91 added physiology code to both sides. Inspect `services/api/Dockerfile` and `services/worker/Dockerfile` (and the corresponding requirements files) to confirm that no Pydantic schema imported by the API route pulls in a transitive ML dependency (mediapipe, faster-whisper, parselmouth, ctranslate2, torch), and that no analytics module imported by the worker pulls in a web framework (fastapi, uvicorn, starlette).

### M7. Update `IMPLEMENTATION_PLAN.md` Deferred List

Module B (TikTok ground-truth ingester) is still deferred pending EulerStream access. PR 91 may have added new deferred items — for example, if a physiology consumer module was scaffolded but is not yet fired by the orchestrator, or if `comodulation_log` is written but not yet read by any downstream analytics surface. Walk the diff for new public symbols that lack runtime callers and add them to the deferred inventory.

### M8. Refresh Performance Baseline With Physiology Enabled

PR 91 changes the segment-assembly path (physiology drain + context injection) and adds the Co-Modulation Index window compute on the persistence side. Run the standard 30 s segment benchmark twice — once with `physio:events` empty and once with synthetic events at the expected Oura cadence — and record both rows in `docs/artifacts/performance_baseline.md`. Confirm segment-assembly p95 is unchanged within tolerance and that the new analytics compute fits within the Module E budget.

### M9. Verify SPEC-AMEND-007 Cross-References

`docs/SPEC_AMENDMENTS.md` entry SPEC-AMEND-007 lists the affected files. Cross-check that every file listed actually contains physiology code in the merged tree, that no physiology-bearing file added by the merge is missing from the amendment's affected-files list, and that `docs/SPEC_REFERENCE.md` §4.B.2 / §4.C.4 / §4.E.2 / §7C summaries match the implemented behavior verbatim on the load-bearing details (Redis key names, table names, response shapes, idempotency mechanism).

### M10. Confirm No Raw Webhook Bodies Are Persisted

The hard rule against persisting raw biometric media extends to wearable telemetry: only normalized, anonymized metrics may land in PostgreSQL. Inspect the API route, the Redis enqueue payload, and the `physiology_log` insert path to confirm that the raw Oura webhook body (including any user identifiers, device IDs, or vendor-specific fields not part of the canonical Physiological Sample Event schema) is not stored in either Redis beyond the transient queue entry or in PostgreSQL at all. Anything beyond `subject_role`, `rmssd_ms`, `heart_rate_bpm`, freshness metadata, provider tag, and source timestamp must be dropped at the API boundary.
