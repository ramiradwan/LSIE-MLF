# LSIE-MLF Implementation Plan

**Cycle:** Desktop cloud telemetry parity and operator sync controls (2026-05-10)
**Goal:** Finish the active desktop cloud path so the runtime can upload typed telemetry and let operators trigger cloud sign-in and verified experiment bundle refresh from the current shell.
**Spec anchors:** §4.E.1, §5.1.6, §6.1, §6.4.1, §7B, §7E, §9.1, §9.2

## Phase 1 — Desktop telemetry producers

**Purpose.** Complete the desktop-local producer side of the cloud sync path by emitting `InferenceHandoffPayload`, online-provisional `AttributionEvent`, and `PosteriorDelta` from `analytics_state_worker` using existing typed contracts. This phase comes first because cloud upload ordering depends on the segment payload existing before any posterior delta reaches the cloud API.

**Files.**
- `services/desktop_app/processes/analytics_state_worker.py` — build a typed enqueue plan and enqueue segment telemetry, optional attribution event, then posterior delta using the existing cloud outbox (§6.1, §6.4.1, §7B, §7E, §9.2).
- `tests/unit/desktop_app/processes/test_analytics_state_worker.py` — lock the enqueue-plan shape, segment-before-delta ordering, and online-provisional attribution event emission (§6.1, §6.4.1, §7B, §7E).
- `tests/integration/desktop_app/test_cloud_offline_replay.py` — keep exact-once replay expectations grounded on the mixed desktop cloud outbox path (§9.1, §9.2).

**Depends on.** Existing `CloudOutbox`, `CloudSyncWorker`, and desktop analytics IPC contracts.

**Done when.** `analytics_state_worker` uploads handoff payloads before posterior deltas, emits only `online_provisional` attribution events, and the targeted desktop telemetry tests pass.

## Phase 2 — Operator cloud actions

**Purpose.** Expose the existing cloud auth and experiment bundle clients through the current Health page action-binding pattern so operators can recover cloud connectivity and refresh verified experiment bundles from the desktop shell. This follows Phase 1 because both phases complete the same operator-visible cloud workflow from producer through upload/auth surface.

**Files.**
- `services/operator_console/views/health_view.py` — surface Health header actions and inline status for repair install, cloud sign-in, and experiment bundle refresh (§4.E.1).
- `services/operator_console/viewmodels/health_vm.py` — bind one-shot Health actions and track in-progress state without creating a new route (§4.E.1).
- `services/operator_console/views/main_window.py` — wire the Health view-model actions to the polling coordinator at the existing composition root (§4.E.1, §9.1).
- `services/operator_console/polling.py` — run loopback PKCE sign-in and signed experiment bundle refresh using the desktop secret store and verified bundle cache (§5.1.6, §9.1).
- `services/operator_console/design_system/design_system.json` — register the new Health action controls in the current design-system inventory.
- `tests/unit/operator_console/test_health_view.py` — verify Health action affordances and inline operator status behavior.
- `tests/unit/operator_console/test_main_window.py` — verify MainWindow wires Health actions to the coordinator.
- `tests/unit/operator_console/test_polling.py` — verify cloud sign-in, bundle refresh, prerequisite failure, and follow-up readback refresh behavior.

**Depends on.** Existing `DesktopAuthFlow`, `ExperimentBundleClient`, `ExperimentBundleStore`, secret-store helpers, and Health page architecture.

**Done when.** Operators can trigger cloud sign-in and experiment bundle refresh through the Health page pattern, and the targeted operator-console tests and design-system verification pass.

## Phase 3 — Documentation refresh

**Purpose.** Bring the active planning and deferral documentation back in sync with the shipped desktop/runtime behavior so future implementation cycles do not inherit stale assumptions. This is last because the docs should describe the landed behavior rather than the intended behavior.

**Files.**
- `IMPLEMENTATION_PLAN.md` — record the active cycle so `/implement-phase` and `/implement-file` resolve the current work correctly.
- `docs/DEFERRED_INTEGRATIONS.md` — narrow desktop cloud deferral language to the still-deferred surfaces now that telemetry producers and operator-triggered auth/bundle refresh exist.
- `docs/artifacts/OPERATOR_CONSOLE_UI_UX_AUDIT.md` — remove stale statements that cloud sign-in and bundle refresh are absent from the current shell.
- `docs/artifacts/OPERATOR_CONSOLE_UI_UX_SPECIALIST_BRIEF.md` — remove stale briefing language that says operator-facing cloud sign-in/sync UI is not present.
- `docs/artifacts/v4_pivot_handoff_2026-05-01.md` — refresh any stale cloud-activation claims carried forward from the pivot handoff.

**Depends on.** Phase 1 and Phase 2 landing in the tree.

**Done when.** No active doc still claims the desktop shell lacks operator-triggered cloud sign-in/bundle refresh or that the desktop runtime cannot enqueue cloud telemetry from `analytics_state_worker`.
