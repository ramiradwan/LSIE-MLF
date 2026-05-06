# LSIE-MLF Post-Merge Playbook

Repeatable chores to run after every feature branch is merged into `main`. The playbook has two parts:

1. **Standing Post-Merge Chores** — eight permanent chores that run on every merge regardless of what was merged. These are stable and should not be edited except to refine wording or add new permanent chores.
2. **Merge-Specific Chores (Current Cycle)** — chores tied to what was *just* merged. This section is **rewritten at the end of each merge cycle** so it always describes the most recent merge. Once those chores are completed, the section is wiped and re-populated for the next cycle.

The playbook is operator-facing: each chore should be runnable from a clean checkout of `main` with no prior conversation context.

---

## Part 1 — Standing Post-Merge Chores

### 1. Canonical Terminology Sweep

**Purpose.** Enforce §0.3 of the spec: only the canonical identifiers listed in `CLAUDE.md` (IPC Pipe, Ephemeral Vault, InferenceHandoffPayload, ML Worker, API Server, Message Broker, Persistent Store, Capture Container, PhysiologicalChunkEvent, Physiological Context, Physiological State Buffer, subject_role, Co-Modulation Index) may appear in code and docstrings. Retired synonyms (Celery node, GPU worker, FIFO, named pipe, 24-hour vault, handoff schema, FastAPI server, etc.) erode shared vocabulary and confuse the ADO agent's refactor planner.

**Inputs.** The retired-synonym grep from `CLAUDE.md`, run against `services/`, `packages/`, and `scripts/`. The executable §13 audit harness in Chore #8 is the authoritative automated gate for canonical-name verifier results. There is no active `docker-compose.yml` input for the v4 desktop runtime.

**Outputs.** Either a clean (zero-match) report committed to the merge cycle log, or a follow-up commit that renames the offenders. README narrative prose is exempt; code, comments, and docstrings are not.

**Success.** `grep` returns no matches outside of explicitly whitelisted README sections. **Failure.** Any retired synonym found in code or docstrings, or any newly-coined synonym for a §0.3 concept that is not yet in the canonical list.

### 2. Documentation Reconciliation

**Purpose.** Keep `README.md`, `.claude/skills/*/SKILL.md`, `CLAUDE.md`, and `docs/SPEC_REFERENCE.md` aligned with what the merged code actually does. The ADO agent's workspace-server loads these files as ground truth; drift here causes the agent to generate code against a fictional baseline.

**Inputs.** The merge diff, the current contents of all skill files, the README topology table and data-flow diagram, and `docs/SPEC_AMENDMENTS.md`.

**Outputs.** Updated docs in the same merge or a fast-follow doc-only PR. Any new amendment to the signed spec (`docs/tech-spec-v*.pdf`) must be registered in `docs/SPEC_AMENDMENTS.md` with ID, section, original text, new behavior, rationale, and affected files.

**Success.** Every desktop process, env var, port, schema field, local SQLite table, IPC channel, and Redis key referenced in the merged code is described in at least one of the four doc surfaces. Container manifests are expected only if a future spec change reintroduces them. **Failure.** Any new artifact that exists in code but is invisible to the workspace-server, or any spec deviation not registered in `SPEC_AMENDMENTS.md`.

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

**Purpose.** Re-establish deterministic v4 desktop latency numbers for Module C dispatch, shared-memory IPC handoff, `gpu_ml_worker` analytics publication, and `analytics_state_worker` SQLite persistence. Merges that touch `services/desktop_app/processes/module_c_orchestrator.py`, `services/desktop_app/processes/gpu_ml_worker.py`, `services/desktop_app/processes/analytics_state_worker.py`, `services/desktop_app/ipc/`, or Operator Console read paths can shift the baseline silently.

**Inputs.** Run `uv run python scripts/run_fixture_benchmark.py <fixture_path> --segments 3` against the checked-in deterministic capture fixture or an explicitly generated deterministic fixture. The benchmark is fixture-driven and does not use live Android, ADB, scrcpy, retained worker paths, brokers, or cloud services.

**Outputs.** A small markdown row appended to `docs/artifacts/performance_baseline.md` with date, commit SHA, scenario, segment count, dispatch p50/p95, ML publish p50/p95, analytics-state p50/p95, visual AU12 tick p50, and end-to-end p95. Regressions >20% versus the previous `v4-fixture:@` row are flagged for investigation before the cycle is closed.

**Success.** New v4 baseline row recorded and within tolerance of the previous row, OR a regression is filed as an ADO work item with reproduction steps. **Failure.** No baseline run executed, or a regression silently accepted.

### 8. §13 Audit Checklist Execution

**Purpose.** Run the autonomous implementation audit from §13 of the current spec through the executable harness (see `.claude/commands/audit.md`). This is the final gate on every merge — the sum of items 1–7 above plus the items they do not cover (canonical name list integrity, requirements pin compliance, image-separation enforcement, `from __future__ import annotations` usage, and extension-specific audit items).

**Inputs.** The current state of `main` after the merge, plus any follow-up commits triggered by chores 1–7.

**Run.** Capture the harness-generated Markdown report from stdout and append it verbatim to the merge cycle log:

```bash
python scripts/run_audit.py --strict > /tmp/section13-audit.md
cat /tmp/section13-audit.md >> docs/artifacts/<cycle-log>.md
```

The harness renders a deterministic Markdown table in runtime-enumerated spec order with stable single-line cells, so adjacent cycle logs should produce meaningful diffs when the spec, verifiers, or evidence changes.

**Outputs.** The appended harness Markdown report for every current §13 audit item. Any fail must point to either a follow-up commit or a SPEC-AMEND entry that justifies the deviation.

**Success.** All current §13 items pass, OR every fail is justified by a registered amendment. **Failure.** Any unjustified fail, or the audit was not run.

---

## Part 2 — Merge-Specific Chores (Current Cycle)

> **Cycle:** _Awaiting next merge._ The previous cycle (PR 166, `feature/audit-as-code`, commit `8874737`, closed 2026-04-30) is complete. The cycle's M-chores were:
>
> - **M1 — §13 audit-harness placeholder verifiers (resolved).** PR 166 shipped the executable §13 audit harness with `--strict` wired into CI (`.github/workflows/ci.yml` `audit` job) and Standing Chore #8, but six items (§13.1, §13.2, §13.3, §13.5, §13.9, §13.10) initially landed backed by `register_placeholder_verifiers()` and kept `--strict` red on `main`. Resolved by adding concrete verifiers in `scripts/audit/verifiers/mechanical.py`: `verify_directory_structure` checks every `codebase_architecture.directory_hierarchy` path exists; `verify_docker_topology` parses `docker-compose.yml` and matches each spec container's image / network / restart policy / depends_on plus every spec volume's mount; `verify_ipc_lifecycle` greps Module A's spec lifecycle-step `<func>`/`<path>`/`<env>`/`<cmd>` tokens against `services/stream_ingest/entrypoint.sh` after expanding shell variable assignments so `$AUDIO_PIPE` resolves to its literal `/tmp/ipc/audio_stream.raw` path; `verify_drift_correction` AST-parses `services/worker/pipeline/orchestrator.py` and matches each `DRIFT_*` constant plus the `drift_offset = host_utc - android_epoch` formula; `verify_schema_validation` AST-parses `packages/schemas/inference_handoff.py`, asserts the `InferenceHandoffPayload(BaseModel)` declaration covers every spec-required field (carving out `_x_max` and any other field whose description is marked `diagnostic` / `calibration telemetry` / `compatibility` / `does not divide`), confirms `extra="forbid"`, and confirms `services/worker/` invokes the model at a dispatch boundary; `verify_module_contracts` extracts the structured `<path>`/`<func>`/`<env>`/`<class>`/`<cmd>`/`<flag>`/`<field>` tokens from each Module A–F `contract.{inputs,outputs,dependencies,side_effects,failure_modes}` and resolves them against module-specific search roots, with the same `_x_max` carve-out and a separate carve-out for deployment-only env vars (`SDL_VIDEODRIVER`, `XDG_RUNTIME_DIR`, `ADB_SERVER_SOCKET`) so they can resolve in `docker-compose.yml` / `.env.example` instead of Python source. Each verifier extracts its expected surface from `spec_content` and is unit-tested under `tests/unit/scripts/audit/test_mechanical_verifiers.py`. `python scripts/run_audit.py --strict` now exits 0 with 31/31 §13 items PASS.
> - **M2 — §13.13 `mark_data_tier` dict-literal emissions (resolved).** The new `services/worker/pipeline/analytics.py` `_evaluation_insert_params` / `_physiology_insert_params` helpers wrap dict literals in `mark_data_tier(...)`; the variable-traceability AST verifier walked those keys as `mark_data_tier.<key>` emissions and reported them unmapped after the analytics surface filter stripped the canonical match. Resolved by adding `is_match`, `source_kind`, `window_s`, `validity_ratio`, `is_valid` to the `out_of_scope` set in `tests/integration/test_variable_traceability.py` (alongside the existing `freshness_s` / `is_stale` / `rmssd_ms` passthrough entries) so the analytics persistence surface is treated as a sink for upstream-produced variables.
> - **M3 — §13.15 retired-synonym matches (resolved).** PR 166's analytics persistence wrapping introduced four `purpose=` strings containing `attribution event` / `outcome row` / `attribution link` / `score row` — all retired §0.3 synonyms. The Module A capture loop comment also still said `kernel pipe`. Resolved by replacing the four `mark_data_tier(..., purpose=...)` strings in `services/worker/pipeline/analytics.py` with the canonical `AttributionEvent` / `OutcomeEvent` / `EventOutcomeLink` / `AttributionScore` row labels and updating `services/stream_ingest/entrypoint.sh` to say `IPC Pipe`.
> - **M4 — Windows-portable canonical-terminology verifier output (resolved).** `scripts/audit/verifiers/mechanical.py` renders the `§13.15` mismatch lines as `f"§0.3/§13.15 {rel_path}:{line_number} matched ..."`; on Windows `rel_path` interpolates with `\` separators, so the unit-test assertions for the Ubuntu-style `services/bad.py:1` evidence broke for any developer running the local suite on Windows. Resolved by emitting `rel_path.as_posix()` so evidence is identical across platforms.
> - **M5 — `DataTier` annotation surface documentation (deferred).** The new `packages/schemas/data_tiers.py` exports (`DataTier`, `DataTierAnnotation`, `mark_data_tier`, `data_tier`, `data_tier_context`, `get_data_tier_annotation`) are reachable through every persistence INSERT call site in `services/` and audited by `scripts/audit/verifiers/data_classification.py` / §13.14, but they are not described in `README.md`, `CLAUDE.md`, `docs/SPEC_REFERENCE.md`, or any `.claude/skills/*/SKILL.md`. Follow-up cycle should add the §5.2 data-tier annotation surface to one of those doc surfaces (preferred location: a new section inside `docs/SPEC_REFERENCE.md` near the §5.2 references, plus a brief mention in the canonical-names list in `CLAUDE.md`).
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
