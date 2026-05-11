# LSIE-MLF Implementation Plan

**Cycle:** Desktop attribution diagnostics parity (2026-05-11)
**Goal:** Make the desktop runtime preserve full `online_provisional` attribution-ledger state for operator readback using the existing IPC and SQLite contracts, while keeping `offline_final` replay deferred until a closed-horizon producer exists.
**Spec anchors:** §6.1, §6.4.1, §7B, §7E, §9.1, §9.2

## Phase 1 — Desktop attribution pass-through and local ledger persistence

**Purpose.** Extend the active desktop analytics path so attribution inputs already allowed by the IPC contract survive `gpu_ml_worker` pass-through, land in `analytics_state_worker`, and upsert the full local attribution ledger (`AttributionEvent`, `OutcomeEvent`, `EventOutcomeLink`, `AttributionScore`). This phase comes first because the desktop SQLite schema and operator read paths already expect attribution-ledger tables, while the runtime previously emitted only the cloud-facing `AttributionEvent`.

**Files.**
- `services/desktop_app/processes/gpu_ml_worker.py` — forward attribution input keys already admitted by `InferenceControlMessage.forward_fields` into `AnalyticsResultMessage.attribution` without inventing a new producer (§6.1, §7E, §9.2).
- `services/desktop_app/processes/analytics_state_worker.py` — build `AttributionLedgerRecords`, persist all four attribution tables in SQLite, and keep cloud enqueue ordering as segment → optional `online_provisional` `AttributionEvent` → optional `PosteriorDelta` (§6.1, §6.4.1, §7B, §7E, §9.2).
- `tests/unit/desktop_app/processes/test_gpu_ml_worker.py` — lock the desktop IPC attribution pass-through behavior.
- `tests/unit/desktop_app/processes/test_analytics_state_worker.py` — verify local attribution-ledger persistence and the unchanged cloud enqueue contract.
- `tests/integration/desktop_app/test_cloud_offline_replay.py` — keep exact-once outbox replay expectations grounded on the mixed desktop cloud path (§9.1, §9.2).

**Depends on.** Existing desktop IPC models, SQLite attribution schema, `packages/ml_core.attribution.build_attribution_ledger_records(...)`, and `CloudOutbox`.

**Done when.** Attribution inputs present on the desktop IPC path are preserved into `AnalyticsResultMessage`, `analytics_state_worker` upserts all four attribution tables locally, targeted `mypy`/`pytest` gates pass, and segment-before-attribution-before-delta ordering remains intact.

## Phase 2 — Documentation refresh

**Purpose.** Bring the active planning and deferred-integration docs in sync with the shipped desktop attribution slice so future cycles do not assume the runtime already has a live attribution-input producer or an `offline_final` replay job.

**Files.**
- `IMPLEMENTATION_PLAN.md` — record this attribution cycle so `/implement-phase` resolves the current work.
- `docs/DEFERRED_INTEGRATIONS.md` — keep only the still-deferred `offline_final` replay/finalization path in the deferred inventory while documenting the shipped desktop `online_provisional` pass-through and local persistence behavior.

**Depends on.** Phase 1 landing in the tree.

**Done when.** Active docs describe the shipped desktop pass-through/local-persistence behavior accurately and still mark closed-horizon `offline_final` replay/finalization as deferred.

## Not in scope for this cycle

- A new desktop outcome or ground-truth producer.
- Automatic closed-horizon `offline_final` replay/finalization.
- New operator-console attribution UI beyond the existing read surfaces.
- Any spec amendment to the §7B reward formula.
