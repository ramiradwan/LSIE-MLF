# LSIE-MLF Implementation Plan

This file holds the **active phased implementation plan** for whatever feature cycle is currently in flight. It is the source of truth consumed by the `/implement-phase` and `/implement-file` slash commands — they read this file to discover which phase to implement and which files belong to it.

Historical plans from prior cycles are archived under `docs/artifacts/`.

---

## How the slash commands use this file

- `/implement-phase N` — runs the TRUST GATE (`scripts/verify_spec_signature.py`), then reads this file, locates the `## Phase N` heading, implements every file listed under that phase, runs `mypy --strict` per file, adds or updates tests, and runs `pytest -x -q`. Definition: `.claude/commands/implement-phase.md`.
- `/implement-file <path>` — same trust gate, then implements a single file by tracing back through this file to find which phase it belongs to and confirming its upstream dependencies are implemented.

Every implementation decision traces to a spec section. The signed spec is `docs/tech-spec-v3.1.pdf`; amendments live in `docs/SPEC_AMENDMENTS.md`.

---

**Cycle:** Operator Console — production-grade PySide6 dashboard (2026-04-17)
**Goal:** Replace the retired Streamlit surface with a modular, non-blocking, operator-grade PySide6 console that surfaces adaptive experiment state, stimulus lifecycle, physiology freshness, co-modulation validity, and degraded-but-recovering subsystem states — all through the API Server, per SPEC-AMEND-008.
**Spec anchors:** §4.E.1 (Execution Details), §4.C (Orchestrator `_active_arm` / `_expected_greeting` / `_stimulus_time`), §4.C.4 (Physiological State Buffer), §4.E.2 (Physiology Persistence), §7B (Thompson Sampling + reward = `p90_intensity × semantic_gate`), §7C (Co-Modulation Index, null-valid), §12 (error-handling matrix), SPEC-AMEND-007, SPEC-AMEND-008.

**Source artifact.** The file-by-file coding checklist with concrete class names, method signatures, and field lists lives at `docs/artifacts/operator_console_checklist.md` (non-tracked — read-only reference). When a phase says "governs §X", the checklist has the exact signatures that phase realizes. This plan is the executable view; the checklist is the contract detail.

**Framing (from the agent research report):**
1. Keep thread-based transport for v1 — structure code so transport can later be swapped without rewriting the UI.
2. Surface **degraded-but-recovering** states (ADB drift freeze/reset, FFmpeg restart, Azure retry-then-null, DB buffer/retry) instead of flattening them into generic errors.
3. Operator trust hinges on exposing the exact reward-explanation fields the pipeline uses: `p90_intensity`, `semantic_gate`, `gated_reward`, `n_frames_in_window`, `baseline_b_neutral`.
4. Physiology surfaces must explicitly distinguish fresh / stale / absent / null-valid states; co-modulation null is a legitimate outcome, not an error.
5. No raw biometric media on operator-facing paths — only derived analytics.

**Refactor note.** The current scaffold (`services/operator_console/config.py:ConsoleConfig`, `api_client.py:ApiClient`, `views/main_window.py`, `views/sessions_view.py`, `views/placeholder_view.py`, `widgets/status_pill.py`, `workers.py`, `app.py`) is intentionally thin. Phases 3–6 reshape it into the final architecture; `SessionsView` is repurposed as the session-history page in Phase 10.

---

## Phase 1 — Shared Operator DTOs

**Purpose.** One pure shared contract module exposing operator-facing read/action payloads. Everything downstream (API routes, client validation, viewmodels, tests) consumes these types. Pure dataclasses/Pydantic — no backend imports, no ORM, no raw media.

**Files.**
- `packages/schemas/operator_console.py` — Enums (`UiStatusKind`, `AlertSeverity`, `AlertKind`, `EncounterState`, `StimulusActionState`, `HealthState`) and models (`SessionSummary`, `OverviewSnapshot`, `LatestEncounterSummary`, `EncounterSummary`, `ExperimentSummary`, `ArmSummary`, `ExperimentDetail`, `PhysiologyCurrentSnapshot`, `CoModulationSummary`, `SessionPhysiologySnapshot`, `HealthSubsystemStatus`, `HealthSnapshot`, `AlertEvent`, `StimulusRequest`, `StimulusAccepted`). Every timestamp field is UTC-aware. Reward-explanation fields on `EncounterSummary`: `active_arm`, `expected_greeting`, `stimulus_time_utc`, `semantic_gate`, `semantic_confidence`, `p90_intensity`, `gated_reward`, `n_frames_in_window`, `baseline_b_neutral`, `physiology_attached`, `physiology_stale`, `notes`. `ArmSummary` includes `posterior_alpha`, `posterior_beta`, `evaluation_variance`. `CoModulationSummary` includes `null_reason`. `HealthSubsystemStatus` includes `recovery_mode` and `operator_action_hint`. Governs §4.E.1, §4.C, §7B, §7C, §12, SPEC-AMEND-007.
- `tests/unit/schemas/test_operator_console_dtos.py` — DTO validation tests: UTC-aware timestamps required; `null_reason` surfaces when `co_modulation_index` is None; `StimulusRequest.client_action_id` dedup key present; enum round-trip.

**Depends on.** nothing (pure contracts).

**Done when.** `mypy --strict packages/schemas/operator_console.py` is clean; `pytest tests/unit/schemas/test_operator_console_dtos.py -x -q` green; no import of `services/` or DB code from this module.

---

## Phase 2 — API Server Operator Aggregate Routes + Services

**Purpose.** Add the `/api/v1/operator/*` aggregate endpoints the console will poll. Reads go through a dedicated read-service; the single write path (stimulus POST) goes through an action-service with idempotency by `client_action_id`. No UI formatting in route handlers; shared DTOs from Phase 1 cross the boundary.

**Files.**
- `services/api/repos/operator_queries.py` — DB row fetchers (`fetch_recent_sessions`, `fetch_session_by_id`, `fetch_session_encounters`, `fetch_experiment_arms`, `fetch_latest_physiology_rows`, `fetch_latest_comodulation_row`, `fetch_health_rollup`, `fetch_alert_feed`). Only add functions that existing repos don't already provide — re-use where possible.
- `services/api/services/operator_read_service.py` — `OperatorReadService` aggregating DB rows + runtime health into DTOs. Private helpers `_build_latest_encounter_summary`, `_build_experiment_summary`, `_build_co_modulation_summary` (must set `null_reason` when index is None), `_build_health_rows` (must populate `recovery_mode` and `operator_action_hint`).
- `services/api/services/operator_action_service.py` — `OperatorActionService.submit_stimulus()`. Dedup by `client_action_id`; reject with 409 when session state disallows stimulus. Never trust the GUI wall clock — authoritative `stimulus_time` is set by the orchestrator, not the API.
- `services/api/routes/operator.py` — `APIRouter(prefix="/operator", tags=["operator"])` with `GET /overview`, `GET /sessions`, `GET /sessions/{id}`, `GET /sessions/{id}/encounters`, `GET /experiments/{id}`, `GET /sessions/{id}/physiology`, `GET /health`, `GET /alerts`, `POST /sessions/{id}/stimulus`. 404 on missing, 409 on conflicting state, operator-safe error payloads.
- `services/api/main.py` — register the router under the existing `/api/v1` prefix.
- `tests/unit/api/test_operator_routes.py` — route-level tests mirroring `tests/unit/api/routes/test_encounters.py` pattern.
- `tests/unit/api/test_operator_read_service.py` — service-level tests for null-reason propagation, recovery-mode inclusion, latest-encounter field completeness.

**Depends on.** Phase 1.

**Done when.** `mypy --strict services/api/` clean; new route + service tests green; `scripts/check_schema_consistency.py` still green (new DTOs reference only existing DB columns); `docker compose config --quiet` unchanged.

---

## Phase 3 — Frontend Contracts & Transport

**Purpose.** Reshape the existing thin scaffold into the final transport layer. `api_client.py` grows a `Transport` protocol, validates every response into Phase-1 DTOs, and raises `ApiError` with a `retryable` flag. `config.py` is renamed `OperatorConsoleConfig` and expanded to cover every poll cadence. `formatters.py` centralizes operator-language translation so widgets never format strings inline.

**Files.**
- `services/operator_console/config.py` — rename `ConsoleConfig` → `OperatorConsoleConfig`; add `environment_label`, `overview_poll_ms`, `session_header_poll_ms`, `live_encounters_poll_ms`, `experiments_poll_ms`, `physiology_poll_ms`, `comodulation_poll_ms`, `health_poll_ms`, `alerts_poll_ms`, `sessions_poll_ms`; `load_config(environ=None)` validates every interval > 0.
- `services/operator_console/api_client.py` — introduce `Transport` Protocol and `UrllibTransport` impl. Extend `ApiError` with `endpoint` and `retryable`. Add typed methods: `get_overview`, `list_sessions`, `get_session`, `list_session_encounters`, `get_experiment_detail`, `get_session_physiology`, `get_health`, `list_alerts`, `post_stimulus`. Every response validated via the Phase-1 Pydantic DTOs; no `dict[str, Any]` past the boundary.
- `services/operator_console/formatters.py` — pure string helpers for timestamps, duration, freshness (fresh/stale wording), reward, percentage, semantic gate, semantic confidence, health state, co-modulation value + null reason, expected-greeting truncation, `build_reward_explanation(encounter)` (using the §7B fields), `build_physiology_explanation(snapshot)`, `build_health_detail(row)`, `ui_status_for_health(row)`. No widget creation, no DB/API calls.
- `tests/unit/operator_console/test_api_client.py` — update existing tests: DTO validation success, `ApiError.retryable` flag semantics, `post_stimulus()`, encounter/alert querystring assembly.
- `tests/unit/operator_console/test_formatters.py` — semantic-gate closed/open strings, reward-explanation for `semantic_gate=0`, reward-explanation for `n_frames_in_window=0`, freshness-stale wording, co-modulation null-reason text.

**Depends on.** Phase 1.

**Done when.** `mypy --strict services/operator_console/` clean; both test files green; no `Any` leaks past the client boundary; `ApiClient` never raises `SystemExit`.

---

## Phase 4 — State Store & Polling Coordinator

**Purpose.** Decouple views from the network layer. `OperatorStore` is the single app-scoped state holder (Qt signals out, no I/O in). `PollingCoordinator` owns the job lifecycle, route-scopes polls so hidden pages don't poll, and triggers immediate refreshes after stimulus submission. `workers.py` gains a `job_name` so one signal bus can serve many jobs.

**Files.**
- `services/operator_console/state.py` — `AppRoute` enum, `StimulusUiContext` dataclass, `OperatorStore(QObject)` with signals `route_changed`, `selected_session_changed`, `overview_changed`, `sessions_changed`, `live_session_changed`, `encounters_changed`, `experiment_changed`, `physiology_changed`, `health_changed`, `alerts_changed`, `stimulus_state_changed`, `error_changed`. Getters + setters only — no network logic. Preserve selected session id across route changes.
- `services/operator_console/workers.py` — modify: `PollingWorker` takes `job_name` and emits `started(job)`, `data_ready(job, payload)`, `error(job, ApiError)`, `stopped(job)`. `OneShotSignals` emits `succeeded(job, payload)`, `failed(job, ApiError)`, `finished(job)`. `run_one_shot(job_name, fn)` dispatches on `QThreadPool`.
- `services/operator_console/polling.py` — `PollJobSpec` (name, interval, optional route scope), `PollingCoordinator(QObject)` with `start()`, `stop()`, `on_route_changed()`, `on_selected_session_changed()`, `refresh_now(job_name)`, `submit_stimulus(session_id, request) -> OneShotSignals`. Private `_make_fetch_*` factories bind the client. Stimulus submission triggers immediate refreshes of overview/live/alerts.
- `tests/unit/operator_console/test_store.py` — signal-emission tests using `qtbot`: route change, selected-session change, encounters replace.
- `tests/unit/operator_console/test_polling.py` — job-start/stop on route change; stimulus-success fan-out triggers overview/live/alerts refresh; error surfaces via `error_changed`.

**Depends on.** Phase 3.

**Done when.** `mypy --strict` clean; both test files green; no widget imports in `state.py`, `workers.py`, or `polling.py`; route-scoped jobs measurably stop when their page hides (covered by a test).

---

## Phase 5 — Widget Primitives

**Purpose.** Reusable presentation widgets with no business logic, no network calls. Each takes data in, emits simple signals out. Theme-agnostic styling via object names.

**Files.**
- `services/operator_console/widgets/metric_card.py` — `MetricCard(title)` with `set_primary_text`, `set_secondary_text`, `set_status(UiStatusKind, text=None)`, `set_clickable(bool)`.
- `services/operator_console/widgets/action_bar.py` — `ActionBar` emitting `stimulus_requested(note)`. Methods: `set_session_context(session_id, active_arm, expected_greeting)`, `set_action_state(StimulusUiContext)`, `set_countdown_remaining(seconds)`, `set_last_message(text)`. Button disabled during in-flight submission; visual state progresses idle→submitting→accepted→measuring→completed. No direct API calls.
- `services/operator_console/widgets/alert_banner.py` — `AlertBanner.set_alert(severity, message)`. Severity drives color + icon only.
- `services/operator_console/widgets/empty_state.py` — `EmptyStateWidget` with `set_title`/`set_message`.
- `services/operator_console/widgets/section_header.py` — `SectionHeader(title, subtitle=None)` with setters.
- `services/operator_console/widgets/event_timeline.py` — `EventTimelineWidget` wrapping a `QListView`/`QTableView` with `set_model(QAbstractItemModel)` and `scroll_to_latest()`.
- `services/operator_console/widgets/status_pill.py` — modify existing to accept `UiStatusKind` enum instead of string kinds.

**Depends on.** Phase 1 (for `UiStatusKind`, `AlertSeverity`, `StimulusUiContext`).

**Done when.** `mypy --strict services/operator_console/widgets/` clean; each widget renders under a minimal qtbot smoke check in an integration test added in Phase 11; no widget file imports `ApiClient`, `OperatorStore`, or DB modules.

---

## Phase 6 — Shell, Theme, Navigation

**Purpose.** Wire the store and coordinator into `app.py`, rebuild `MainWindow` with eager page instantiation, a persistent action bar, sidebar navigation, and a stylesheet that covers every new widget. Close-event stops polling and joins worker threads cleanly.

**Files.**
- `services/operator_console/theme.py` — extend stylesheet: `MetricCard`, `ActionBar`, alert-banner severities, health-row states (ok/warn/bad/degraded/recovering), empty-state. Keep palette stable; `build_stylesheet()` factory callable.
- `services/operator_console/app.py` — factory functions `build_api_client`, `build_store`, `build_polling_coordinator`, `build_main_window`; `main(argv=None)` instantiates store before views, starts polling after `window.show()`, stops polling on shutdown.
- `services/operator_console/views/main_window.py` — rewrite around `OperatorStore` + `PollingCoordinator`. Eager-instantiate all pages in a `QStackedWidget`, mount a single persistent `ActionBar` below the content area, sidebar with nav buttons for Overview / Live Session / Experiments / Physiology / Health / Sessions. `_on_route_selected` forwards to store and coordinator; `_on_stimulus_requested` calls `coordinator.submit_stimulus`. `closeEvent` calls `coordinator.stop()` and joins threads.
- `services/operator_console/__init__.py` — minor: bump docstring to reflect multi-page layout.

**Depends on.** Phases 3, 4, 5.

**Done when.** `python -m services.operator_console` boots to a shell with all six nav entries, action bar disabled while no session is selected, statusline shows the configured API URL; `mypy --strict` clean; close does not leave dangling QThreads (verified via a `pytest` teardown check where feasible).

---

## Phase 7 — Table Models + ViewModel Base

**Purpose.** Qt item models for each tabular surface, all on top of the Phase-1 DTOs. `ViewModelBase` gives every VM a consistent `changed` / `toast_requested` / `error_changed` signal surface.

**Files.**
- `services/operator_console/viewmodels/base.py` — `ViewModelBase(QObject)` with `changed`, `toast_requested`, `error_changed` signals and `emit_changed` / `set_error` / `emit_toast` helpers.
- `services/operator_console/table_models/encounters_table_model.py` — `EncountersTableModel(QAbstractTableModel)`: columns including `segment_timestamp_utc`, `state`, `active_arm`, `semantic_gate`, `p90_intensity`, `gated_reward`, `n_frames_in_window`, `physiology_attached`, `physiology_stale`. `set_rows` preserves selection when row identity is stable; `dataChanged` used for same-identity updates.
- `services/operator_console/table_models/experiments_table_model.py` — arms table (`arm_id`, `greeting_text`, `posterior_alpha`, `posterior_beta`, `evaluation_variance`, `selection_count`, `recent_reward_mean`).
- `services/operator_console/table_models/alerts_table_model.py` — timestamped alert feed; `append_rows` in addition to `set_rows` for streaming-style updates.
- `services/operator_console/table_models/sessions_table_model.py` — recent sessions with `status`, `active_arm`, `experiment_id`, `latest_reward`, `duration_s`.
- `services/operator_console/table_models/health_table_model.py` — subsystem rollup including `recovery_mode` column so degraded-but-recovering states read visually distinct from errors.

**Depends on.** Phase 1.

**Done when.** `mypy --strict` clean; `tests/unit/operator_console/test_encounters_table_model.py` (new, added alongside) and `tests/unit/operator_console/test_health_table_model.py` (new) cover row/column counts, display-role values, selection preservation, recovery-mode rendering.

---

## Phase 8 — ViewModels

**Purpose.** One viewmodel per operator page, subscribing to the store and exposing read-only getters to the view. The Live Session VM carries the reward-explanation and stimulus-lifecycle logic — that is where operator trust lives.

**Files.**
- `services/operator_console/viewmodels/overview_vm.py` — subscribes to overview/alerts/health signals; exposes `snapshot`, `active_session`, `latest_encounter`, `experiment_summary`, `physiology_summary`, `health_summary`, `alerts`.
- `services/operator_console/viewmodels/live_session_vm.py` — owns `EncountersTableModel`, exposes `session`, `selected_encounter`, `select_encounter(id)`, `active_arm`, `expected_greeting`, `stimulus_ui_context`, `set_stimulus_submitting(note)`, `apply_stimulus_accepted(accepted)`, `reconcile_authoritative_stimulus_time()`, `measurement_window_remaining_s(now)`, `reward_explanation()`. Countdown derives from authoritative readback, not click time.
- `services/operator_console/viewmodels/experiments_vm.py` — owns `ExperimentsTableModel`; `detail`, `active_arm_id`, `latest_update_summary`. Must not imply semantic confidence changes reward math.
- `services/operator_console/viewmodels/physiology_vm.py` — `snapshot`, `operator_snapshot`, `streamer_snapshot`, `comodulation`, `comodulation_explanation()`. Explicitly distinguishes fresh/stale/absent/null-valid.
- `services/operator_console/viewmodels/health_vm.py` — owns `HealthTableModel` and `AlertsTableModel`; `snapshot`, `degraded_count`.
- `tests/unit/operator_console/test_viewmodels.py` — reward-explanation text when `semantic_gate=0`, when `n_frames_in_window=0`; physiology null-reason propagation; experiments active-arm; live-session measurement-window countdown arithmetic.

**Depends on.** Phases 3, 4, 7.

**Done when.** `mypy --strict` clean; `test_viewmodels.py` green; no network calls in any VM.

---

## Phase 9 — Overview + Live Session Views

**Purpose.** Build the two operator-critical pages. Overview is the first thing the operator sees; Live Session is where reward explanation and stimulus actions live.

**Files.**
- `services/operator_console/views/overview_view.py` — `OverviewView(vm)` with cards for Active Session, Experiment, Physiology, Health, Latest Encounter, and an attention queue fed by alerts. `on_activated` / `on_deactivated` hooks for coordinator route scoping. `_render_*` helpers per card.
- `services/operator_console/views/live_session_view.py` — `LiveSessionView(vm)` with header (session / arm / expected-greeting / readiness), encounter table using `EncountersTableModel`, detail pane showing reward explanation (`p90_intensity`, `semantic_gate`, `gated_reward`, `n_frames_in_window`, `baseline_b_neutral`) plus physiology freshness for the segment. Selection is by encounter id.

**Depends on.** Phases 5, 6, 7, 8.

**Done when.** Both pages render real data when pointed at a running API; action bar submits stimulus and progresses through the lifecycle; no inline string formatting (`formatters.py` only); `mypy --strict` clean.

---

## Phase 10 — Experiments + Physiology + Health + Sessions Views

**Purpose.** Build the remaining pages. Repurpose the existing `SessionsView` as the history/recent-sessions page.

**Files.**
- `services/operator_console/views/experiments_view.py` — active experiment summary card, arm table, latest posterior update summary, plain-language explainer. No semantic-confidence reward math implication.
- `services/operator_console/views/physiology_view.py` — operator/streamer side-by-side cards, freshness + stale badge, co-modulation value + null-reason line. No fake high-frequency animation.
- `services/operator_console/views/health_view.py` — subsystem table + alert timeline; recovery-mode and operator-action-hint columns surfaced.
- `services/operator_console/views/sessions_view.py` — modify: rename current implementation's role to "recent sessions / history", wire to `SessionsTableModel` from Phase 7, emit `session_selected(id)` into the store instead of talking to the API directly.
- `services/operator_console/views/placeholder_view.py` — keep as-is for any future surface.

**Depends on.** Phases 5, 6, 7, 8.

**Done when.** All four pages render real data; `mypy --strict` clean; `SessionsView` no longer owns a `QThread` or `ApiClient` (that lives in the coordinator now).

---

## Phase 11 — Integration Tests, Packaging, Docs

**Purpose.** Lock behavior with integration tests, ship a one-dir PyInstaller build, and align docs/amendments.

**Files.**
- `tests/integration/operator_console/test_overview_view.py` — overview renders active-session card with seeded store.
- `tests/integration/operator_console/test_live_session_action_bar.py` — action bar disables during submit; selection updates detail pane.
- `tests/integration/operator_console/test_experiments_view.py` — arm table renders posterior + evaluation variance.
- `tests/integration/operator_console/test_physiology_view.py` — stale badge visible when `is_stale=True`; null-valid co-modulation renders `null_reason` without appearing broken.
- `tests/integration/operator_console/test_health_view.py` — degraded and recovering states render distinctly from errors.
- `build/operator_console.spec` — PyInstaller one-dir build; include `services/operator_console`, `packages/schemas`, theme assets/icons; exclude API-Server-only packages. Smoke-test builds on Windows.
- `requirements/cli.txt` — confirm `PySide6` pin; add `pydantic` explicitly if the operator host doesn't pull it transitively. Do **not** add `qasync`/`httpx` in this cycle.
- `README.md` — launch command, env vars, API-Server dependency, explicit note that the operator host does not couple directly to Postgres/Redis.
- `docs/SPEC_REFERENCE.md` — replace the §4.E.1 Streamlit-dashboard wording with the Operator Console page map (Overview / Live Session / Experiments / Physiology / Health / Sessions) and action-rail lifecycle.
- `docs/SPEC_AMENDMENTS.md` — update SPEC-AMEND-008 entry to enumerate the six pages and the stimulus-lifecycle contract that the console relies on.
- `.claude/skills/module-contracts/SKILL.md` — extend Module E section to mention the new `/api/v1/operator/*` aggregate surface.

**Depends on.** Phases 1–10.

**Done when.** Full `scripts/check.sh` green (ruff, mypy, pytest, schema gate, compose config, canonical-terminology audit, pin check); operator-console boots; PyInstaller one-dir build runs; docs and amendment text match behavior.

---

## Completion criteria for the cycle

The cycle closes when every item below holds:

- `python -m services.operator_console` boots into a working shell.
- Overview page renders real API data.
- Action bar can submit stimulus and show accepted → measuring → completed lifecycle driven by authoritative readback, not click time.
- Live Session page explains reward with `p90_intensity`, `semantic_gate`, `gated_reward`, `n_frames_in_window`, `baseline_b_neutral`.
- Experiments page shows arm / posterior (α, β) / evaluation-variance readback.
- Physiology page distinguishes fresh / stale / absent / null-valid.
- Co-modulation null is rendered as a legitimate outcome with a readable `null_reason`, not an error.
- Health page shows degraded vs recovering vs error states with operator-action hints.
- No raw biometric or raw wearable payloads appear in any operator-facing data path.
- `scripts/check.sh` is green on a clean checkout.

---

## What does NOT belong here

- Feature-cycle retrospectives — `docs/artifacts/` once closed.
- Post-merge chores — `docs/POST_MERGE_PLAYBOOK.md`.
- Deferred integrations — `docs/DEFERRED_INTEGRATIONS.md`.
- Spec amendments — `docs/SPEC_AMENDMENTS.md`.

This file is for the currently-in-flight cycle only. Wipe back to scaffold when shipped.
