# LSIE-MLF Deferred Integrations Inventory

This file inventories code that is **implemented in the repository but not wired into the active runtime path**. These integrations exist as importable, tested modules but are not instantiated, called, or enqueued by any production code path. Each entry is gated by an external dependency or a future spec amendment.

**Why this file exists.** When the ADO agent (or any contributor) does feature work, it must not "fix" a deferred integration by wiring it in without first satisfying the gating dependency. Activating any of these without the gate would either (a) crash at runtime against a missing external service, (b) violate the signed spec (`docs/tech-spec-v*.pdf`), or (c) silently change the semantics of a stable subsystem.

**Scope.** This is not a TODO list. Operator tooling that is intentionally launched manually (CLI, Operator Console / Debug Studio) is out of scope. Read endpoints that have not been built yet but where the underlying data is already persisted are also out of scope. The criterion is strict: *the code is present, but no production code path reaches it*.

**Maintenance.** This file is refreshed during Standing Post-Merge Chore #6 (`docs/POST_MERGE_PLAYBOOK.md`). Every PR that adds a new public symbol must either wire it into a runtime entrypoint or add an entry here.

---

## Entries

### 1. Module B — TikTok Ground Truth Ingester

| Field | Value |
|---|---|
| **Files** | `services/worker/pipeline/ground_truth.py` (full implementation, ~234 lines), `packages/schemas/events.py` (`LiveEvent`, `GiftEvent`, `ComboEvent`, `GroundTruthRecord`), `tests/unit/worker/pipeline/test_ground_truth.py` (unit tests in isolation) |
| **Gating dependency** | EulerStream third-party WebSocket signature API (`SignatureProvider` protocol, default impl `EulerStreamSigner`). The signing service is unavailable in the current environment. A local patchright-based signer using the existing Module F Chromium stack is also feasible (see `services/worker/tasks/enrichment.py`) but has not been implemented. |
| **Deferred since** | 2026-04-16 (Hardening cycle, recorded in `IMPLEMENTATION_PLAN.md` "Module B clarification" and "What This Plan Does NOT Cover") |
| **Justification** | The `GroundTruthIngester` class, `EulerStreamSigner`, exponential-backoff parameters (§12.1 Module B), WebSocket reconnection logic, and Action_Combo constraint are all complete and unit-tested. The ingester is simply never instantiated by the Orchestrator Container because the EulerStream API requires paid credentials that the current environment does not have. Wiring it in without the signer would cause the Orchestrator Container startup to fail on the first WebSocket auth attempt. Activation requires either provisioning EulerStream credentials or implementing the local patchright signer per the `SignatureProvider` protocol. |

### 2. Physiological Reward Modulation

| Field | Value |
|---|---|
| **Files** | `services/worker/pipeline/reward.py` (current AU12-only `compute_reward()` signature would need an RMSSD parameter), `services/worker/pipeline/orchestrator.py` (assemble_segment already injects `_physiological_context` into the payload — the data is available downstream), `services/worker/tasks/inference.py` (the persist dispatch already forwards `_physiological_context`), `packages/schemas/physiology.py` (`PhysiologicalSnapshot.rmssd_ms` is the input field) |
| **Gating dependency** | A future amendment to spec §7B (Thompson Sampling) explicitly permitting RMSSD to enter the gated reward computation. The current §7B reward formula is `r_t = P90_AU12 × G_semantic` and contains no physiology term. SPEC-AMEND-007 added the *transport* of physiological context through Modules C/D/E and the *persistence* of co-modulation analytics, but explicitly stopped short of modifying the reward path. |
| **Deferred since** | 2026-04-16 (physiology merge, PR 91 commit `60be7ec`) |
| **Justification** | The full physiological data path now reaches the persistence layer (`physiology_log`, `comodulation_log`) and the per-segment `_physiological_context` dict is present in the InferenceHandoffPayload at the point where `reward.compute_reward()` is called. Adding RMSSD as a multiplicative or additive term to the gated reward would change the Thompson Sampling posterior update semantics for every prior arm in the `experiments` table — a change that must be governed by a written spec amendment so that the review agent treats it as accepted rather than a violation. Until that amendment exists, `compute_reward()` must remain AU12 + semantic-gate only. |

### 3. Module F — Context Enrichment Task

| Field | Value |
|---|---|
| **Files** | `services/worker/tasks/enrichment.py` (Celery task `scrape_context`, fully decorated and registered), `tests/unit/worker/tasks/test_enrichment.py` (isolated tests), `services/api/db/schema.py` (the `context` table DDL that would receive the scraped output), `data/sql/01-schema.sql` (same `context` table) |
| **Gating dependency** | (a) A producer that calls `scrape_context.delay(...)` — currently no production code path enqueues this task. (b) The ML Worker compose command at `docker-compose.yml` uses `-I services.worker.tasks.inference` which explicitly includes only the inference task; the enrichment task is not loaded by the running ML Worker even though `celery_app.autodiscover_tasks` would otherwise pick it up. (c) Module B integration (entry #1) is the natural producer — TikTok user profiles surfaced by the ingester are the intended scrape targets. |
| **Deferred since** | 2026-04-16 (carried forward from the original build; never wired during the 8-phase implementation) |
| **Justification** | The `scrape_context` Celery task implements §4.F.1 (patchright-based ephemeral browser scraping) end-to-end, including the §12 retry semantics. The downstream `context` table is created by the SQL bootstrap. However, no Orchestrator Container or API Server code path enqueues the task, and the ML Worker does not load the enrichment task module. Wiring requires (1) a producer — most naturally the Module B ingester emitting unique_ids, (2) updating the ML Worker `-I` list in compose, and (3) an INSERT path from the Celery task return value into the `context` table. Activating any one of these in isolation would either dead-letter the work or silently drop scraped data on the floor. |

### 4. Ephemeral Vault 24-Hour Secure Deletion Cron

| Field | Value |
|---|---|
| **Files** | `services/worker/vault_cron.py` (`run_vault_cron()` infinite loop calling `EphemeralVault.secure_delete` on `/data/raw/` and `/data/interim/` every 24 hours), `packages/ml_core/encryption.py` (`EphemeralVault.secure_delete` implementation), `docker-compose.yml` (no service runs this — neither the ML Worker nor Orchestrator Container command includes `vault_cron`) |
| **Gating dependency** | A dedicated container or sidecar in `docker-compose.yml` whose command is `python3.11 -m services.worker.vault_cron`, plus a host bind-mount of `/data/raw/` and `/data/interim/` into that container with appropriate write/delete permissions. Adding this without aligning the host filesystem layout would shred either nothing (mounts missing) or unintended files (mounts misaligned). |
| **Deferred since** | 2026-04-16 (carried forward from the original build; the function is invokable as `python -m services.worker.vault_cron` but never started by any container) |
| **Justification** | §5.1 mandates a 24-hour secure deletion policy on transient media buffers under `/data/raw/` and `/data/interim/`. The implementation exists and runs the documented `shred -vfz -n 3` via `EphemeralVault.secure_delete`. However, no container in `docker-compose.yml` invokes `run_vault_cron`, so the policy is currently enforced only when an operator manually runs the module. This is a known data-governance gap. Wiring requires both a compose-level service definition and a confirmed host-side directory layout — activation without both would either be a no-op or destroy unrelated data. |

### 5. Attribution Offline Finalization / Backfill Replay

| Field | Value |
|---|---|
| **Files** | `packages/ml_core/attribution.py` (`build_attribution_ledger_records(..., finality=...)`, replay-stable UUID helpers, and score identity that excludes finality so finalization can upsert the same rows), `packages/schemas/attribution.py` (`AttributionEvent`, `OutcomeEvent`, `EventOutcomeLink`, `AttributionScore` allow `online_provisional` and `offline_final`), `services/worker/pipeline/analytics.py` (`MetricsStore.persist_attribution_ledger()` / `_write_attribution_ledger()` upsert all four attribution tables), `data/sql/05-attribution.sql` (finality constraints and read-path indexes) |
| **Gating dependency** | A Module E finalization/replay producer that selects attribution events whose horizons have closed, rebuilds the ledger with `finality="offline_final"`, and invokes the existing idempotent upsert path. No production job, Celery task, CLI, API route, or scheduler currently calls the builder with `offline_final` or replays closed horizons. |
| **Deferred since** | 2026-04-29 (attribution ledger/backfill support, introduced during the baseline-cleanup merge train and first inventoried in this post-cleanup refresh) |
| **Justification** | The online attribution ledger path is wired from `persist_metrics()` through `build_attribution_ledger_records()` into `MetricsStore.persist_attribution_ledger()`, but it uses the builder default `online_provisional`. The schemas, SQL constraints, deterministic IDs, and upsert implementation already support the §7E `online_provisional` → `offline_final` lifecycle after attribution horizons close. The missing surface is the runtime finalization/replay driver that decides when horizons are closed and feeds finalized records back through the upsert path. Wiring finalization without that horizon scan and replay contract would either mark records final too early or leave analytics consumers unable to distinguish provisional data from closed-horizon attribution. |

---

## Searched-But-Not-Deferred (negative findings)

The following were considered during the deferral scan and rejected as **not** deferred integrations:

- **Co-Modulation Index read API** — the write path (`analytics.compute_comodulation_index` → `comodulation_log`) is fully wired in the active Orchestrator Container / Module E path. A `GET /api/v1/comodulation/{session_id}` read endpoint is now live (see `services/api/routes/comodulation.py`).
- **Mounted read API routes not consumed by the Operator Console** — `metrics`, `sessions`, `encounters`, `experiments`, `physiology`, and `comodulation` routers are mounted in `services/api/main.py` under `/api/v1`. Even when the PySide6 console prefers the aggregate `operator` API, these routes are production FastAPI entrypoints rather than dormant code.
- **Attribution online-provisional ledger writes** — the streaming path (`services/worker/tasks/inference.py` → `packages/ml_core/attribution.build_attribution_ledger_records()` → `services/worker/pipeline/analytics.MetricsStore.persist_attribution_ledger()`) is wired for `online_provisional` records. Only the closed-horizon `offline_final` replay/finalization path is deferred in entry #5.
- **Semantic shadow mode** — §8 / §13.27 define a parallel scorer that records candidate outputs without changing live `is_match`, `confidence_score`, reward, or Thompson Sampling updates. The in-process `SemanticEvaluator` shadow sidecar is wired for observational evaluation; no external service, persistence, or promotion workflow is treated as dormant integration here.
- **PySide6 Debug Studio** (`scripts/debug_studio.py`) — intentionally launched manually by operators per §4.E.1. Operator tooling, not deferred integration. Supersedes the retired Streamlit operator tooling surface (`services/worker/dashboard.py`, deleted 2026-04-17).
- **Operator CLI** (`scripts/lsie_cli.py`) — same reasoning as Debug Studio.
- **`SPEC-AMEND-001` / `SPEC-AMEND-002` references in production code** — these are *accepted* amendments documented in `docs/SPEC_AMENDMENTS.md`. Not pending; not deferred.

## Search methodology

The following sweeps were used to populate this file and should be re-run during Standing Post-Merge Chore #6:

```bash
# 1. Disabled feature flags / killswitches
grep -rn "if False\|if 0\|enabled\s*=\s*False\|ENABLED\s*=\s*False\|DISABLED\|disabled\s*=\s*True" services/ packages/

# 2. Commented-out integration calls (await/dispatch/register patterns at line start)
grep -rnE "^\s*#\s*(await|self\.|orchestrator\.|register|wire|init|\.delay\(|\.apply_async)" services/ packages/

# 3. TODO/FIXME/HACK markers, especially those referencing future spec amendments
grep -rniE "TODO|FIXME|XXX|HACK|DEFERRED|future amendment|pending amendment|when the spec|amendment.*allows" services/ packages/

# 4. Functions defined in services/ or packages/ that have no caller anywhere in
#    the production tree (excluding tests). For each public symbol introduced by
#    a merge, confirm at least one runtime caller exists outside its own module
#    and outside tests/.
```

Hits from these sweeps that cannot be classified as "wired" or as legitimate operator tooling MUST be added to this file with the same five fields — name, files, gating dependency, deferred-since date, justification.
