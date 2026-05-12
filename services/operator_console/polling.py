"""
Polling coordinator.

Single authority over the console's polling job lifecycle. Views and
viewmodels never talk to `ApiClient` directly; they read from
`OperatorStore`, and the coordinator is the only thing that drives
store mutations in response to network fetches. This separation keeps
widgets as pure presentation components.

Design notes:
  - One `PollingWorker` + one `QThread` per job. Workers move to their
    own thread on start and quit on stop; joining happens in `stop()`
    so `MainWindow.closeEvent` can terminate cleanly.
  - Route scoping (`PollJobSpec.route_scoped`) is enforced on every
    route change: jobs whose scope differs from the new route stop,
    and jobs that should now be active start. `None`-scoped jobs run
    continuously — alerts are the canonical example because the
    attention queue is always visible.
  - Session-scoped jobs (live-session header, encounters, physiology)
    tear down and re-start when the selected session id changes so
    the fetch callable captures the current id.
  - Stimulus and experiment-management submissions are one-shots via
    `run_one_shot`; on success the coordinator updates the store and
    queues the relevant refreshes so the UI does not wait for the next
    polling interval.
  - Cloud sign-in and experiment-bundle refresh run through the shared
    loopback API/client surface so GUI and CLI callers do not duplicate
    cloud helper logic or touch SQLite write helpers directly.

Spec references:
  §4.C           — stimulus idempotency lives in the loopback API/client boundary, not here
  §4.E.1         — operator-facing aggregate and experiment admin endpoints
  §12            — retryable vs non-retryable errors flow through the
                   store's per-scope error signal
"""

from __future__ import annotations

from collections.abc import Callable, Container
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from time import monotonic
from typing import Any
from uuid import UUID, uuid4

from PySide6.QtCore import QMetaObject, QObject, Qt, QThread, QTimer, Signal, Slot

from packages.schemas.experiments import (
    ExperimentAdminResponse,
    ExperimentArmAdminResponse,
    ExperimentArmCreateRequest,
    ExperimentArmDeleteResponse,
    ExperimentArmPatchRequest,
    ExperimentCreateRequest,
)
from packages.schemas.operator_console import (
    AlertEvent,
    ArmSummary,
    CloudActionStatus,
    CloudAuthStatus,
    CloudExperimentRefreshStatus,
    CloudOutboxSummary,
    EncounterSummary,
    ExperimentBundleRefreshRequest,
    ExperimentDetail,
    ExperimentSummary,
    HealthSnapshot,
    OperatorEventEnvelope,
    OverviewSnapshot,
    SessionCreateRequest,
    SessionEndRequest,
    SessionLifecycleAccepted,
    SessionPhysiologySnapshot,
    SessionSummary,
    StimulusRequest,
)
from services.desktop_launcher.repair import repair_runtime
from services.operator_console.api_client import ApiClient, ApiError
from services.operator_console.config import OperatorConsoleConfig
from services.operator_console.event_client import OperatorEventClient
from services.operator_console.state import AppRoute, OperatorStore
from services.operator_console.workers import (
    EventStreamHandle,
    OneShotSignals,
    PollingWorker,
    run_one_shot,
    start_event_stream_worker,
)

# ----------------------------------------------------------------------
# Job identity constants — also used as error-scope strings so the
# store's per-scope error signal matches the job name that produced it.
# ----------------------------------------------------------------------

JOB_OVERVIEW = "overview"
JOB_SESSIONS = "sessions"
JOB_LIVE_SESSION = "live_session"
JOB_ENCOUNTERS = "encounters"
JOB_EXPERIMENT = "experiment"
JOB_EXPERIMENT_SUMMARIES = "experiment_summaries"
JOB_PHYSIOLOGY = "physiology"
JOB_HEALTH = "health"
JOB_ALERTS = "alerts"
JOB_STIMULUS = "stimulus"
JOB_SESSION_START = "session_start"
JOB_SESSION_END = "session_end"
JOB_REPAIR_INSTALL = "repair_install"
JOB_CLOUD_AUTH_STATUS = "cloud_auth_status"
JOB_CLOUD_OUTBOX = "cloud_outbox"
JOB_CLOUD_SIGN_IN = "cloud_sign_in"
JOB_EXPERIMENT_BUNDLE_REFRESH = "experiment_bundle_refresh"
JOB_EVENT_STREAM = "event_stream"

_HIGH_FREQUENCY_SSE_JOBS = frozenset(
    {
        JOB_OVERVIEW,
        JOB_HEALTH,
        JOB_ALERTS,
    }
)
SSE_RECONNECT_BASE_MS = 500
SSE_RECONNECT_MAX_MS = 8000


@dataclass(frozen=True)
class PollJobSpec:
    """Static description of a polling job.

    `route_scoped=None` means "always run while the coordinator is
    started". A single route or route collection means the job only runs
    while one of those routes is active; switching away stops it until the
    operator returns.
    """

    name: str
    interval_ms: int
    route_scoped: AppRoute | Container[AppRoute] | None = None
    session_scoped: bool = False


class _JobHandle:
    """Bookkeeping for one live job: its worker, its thread, its spec."""

    __slots__ = ("spec", "worker", "thread")

    def __init__(self, spec: PollJobSpec, worker: PollingWorker, thread: QThread) -> None:
        self.spec = spec
        self.worker = worker
        self.thread = thread


class PollingCoordinator(QObject):
    """Orchestrates poll jobs plus one-shot operator/admin write paths.

    Emits `job_failed(scope, message)` as a convenience for tests and
    for any listener that wants a flattened error feed; the primary
    error surface is still the store's per-scope `error_changed`.
    """

    job_failed = Signal(str, str)

    def __init__(
        self,
        config: OperatorConsoleConfig,
        client: ApiClient,
        store: OperatorStore,
        parent: QObject | None = None,
        event_client: OperatorEventClient | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._client = client
        self._event_client = (
            event_client
            if event_client is not None
            else OperatorEventClient(config.api_base_url, config.api_timeout_seconds)
        )
        self._store = store
        self._specs: dict[str, PollJobSpec] = self._register_jobs()
        self._jobs: dict[str, _JobHandle] = {}
        # Jobs that have been told to stop but whose worker thread may
        # still be draining a slow urlopen. They drain on their own time
        # so route-change teardown does not block the UI; final shutdown
        # joins anything still alive.
        self._orphan_jobs: list[_JobHandle] = []
        self._inflight_stimulus: dict[str, OneShotSignals] = {}
        self._inflight_session_lifecycle: dict[str, OneShotSignals] = {}
        self._inflight_experiment_mutations: dict[str, OneShotSignals] = {}
        self._inflight_repairs: dict[str, OneShotSignals] = {}
        self._inflight_cloud_actions: dict[str, OneShotSignals] = {}
        self._event_stream: EventStreamHandle | None = None
        self._orphan_event_streams: list[EventStreamHandle] = []
        self._event_stream_connected = False
        self._last_event_id: str | None = None
        self._sse_reconnect_ms = SSE_RECONNECT_BASE_MS
        self._sse_reconnect_timer = QTimer(self)
        self._sse_reconnect_timer.setSingleShot(True)
        self._sse_reconnect_timer.timeout.connect(self._start_event_stream)
        self._started = False

        # Wire store-driven job lifecycle
        store.route_changed.connect(self._on_route_changed_str)
        store.selected_session_changed.connect(self.on_selected_session_changed)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start every job that should run given the current route."""
        if self._started:
            return
        self._started = True
        self._sync_jobs_for_current_state()
        self._start_event_stream()

    def stop(self) -> None:
        """Stop every job and join its thread. Safe to call from
        `MainWindow.closeEvent`.

        Route-change teardown leaves orphans behind (their workers may
        still be in a slow urlopen); final shutdown is the one place we
        actually block waiting for those threads to finish so the
        process can exit cleanly.
        """
        if not self._started:
            return
        self._started = False
        self._sse_reconnect_timer.stop()
        self._stop_event_stream()
        for name in list(self._jobs):
            self._stop_job(name)
        self._drain_orphan_event_streams()
        self._drain_orphan_jobs()

    # ------------------------------------------------------------------
    # Route / selection callbacks
    # ------------------------------------------------------------------

    @Slot(str)
    def _on_route_changed_str(self, route_value: str) -> None:
        # signal payload is the enum's string value
        self.on_route_changed(AppRoute(route_value))

    def on_route_changed(self, route: AppRoute) -> None:
        del route  # the current route is read via self._store.route()
        if self._started:
            self._sync_jobs_for_current_state()

    @Slot(object)
    def on_selected_session_changed(self, session_id: UUID | None) -> None:
        del session_id
        if not self._started:
            return
        # Session-scoped jobs capture the current session in their fetch
        # closure; restart so the closure rebinds to the new id.
        for name, handle in list(self._jobs.items()):
            if handle.spec.session_scoped:
                self._stop_job(name)
        self._sync_jobs_for_current_state()

    # ------------------------------------------------------------------
    # Event stream lifecycle
    # ------------------------------------------------------------------

    @Slot()
    def _start_event_stream(self) -> None:
        if not self._started or self._event_stream is not None:
            return
        handle = start_event_stream_worker(
            self._event_client,
            initial_last_event_id=self._last_event_id,
        )
        handle.worker.connected.connect(self._handle_event_stream_connected)
        handle.worker.event_ready.connect(self._handle_event_stream_event)
        handle.worker.error.connect(self._handle_event_stream_error)
        handle.worker.stopped.connect(self._handle_event_stream_stopped)
        handle.thread.finished.connect(
            lambda stream_handle=handle: self._prune_event_stream_orphan(stream_handle)
        )
        self._event_stream = handle

    def _stop_event_stream(self) -> None:
        handle = self._event_stream
        if handle is None:
            return
        self._disconnect_event_stream_signals(handle)
        handle.worker.stop()
        self._event_stream = None
        self._event_stream_connected = False
        if handle.thread.isRunning():
            self._orphan_event_streams.append(handle)

    @Slot()
    def _handle_event_stream_connected(self) -> None:
        self._event_stream_connected = True
        self._sse_reconnect_ms = SSE_RECONNECT_BASE_MS
        self._store.clear_error(JOB_EVENT_STREAM)
        self._sync_jobs_for_current_state()

    @Slot(object)
    def _handle_event_stream_event(self, envelope: object) -> None:
        if not isinstance(envelope, OperatorEventEnvelope):
            self._handle_event_stream_error(
                ApiError(
                    message="unexpected event stream payload shape",
                    endpoint=None,
                    retryable=True,
                )
            )
            return
        self._last_event_id = envelope.event_id
        self.apply_event_payload(envelope.event_type, envelope.payload)

    @Slot(object)
    def _handle_event_stream_error(self, error: object) -> None:
        message = str(error) if not isinstance(error, ApiError) else error.message
        self._store.set_error(JOB_EVENT_STREAM, message)
        self.job_failed.emit(JOB_EVENT_STREAM, message)
        self._event_stream_connected = False
        self._sync_jobs_for_current_state()

    @Slot()
    def _handle_event_stream_stopped(self) -> None:
        handle = self._event_stream
        if handle is not None:
            self._disconnect_event_stream_signals(handle)
        self._event_stream = None
        was_connected = self._event_stream_connected
        self._event_stream_connected = False
        if self._started:
            self._sync_jobs_for_current_state()
            self._schedule_event_stream_reconnect(reset=was_connected)
        elif handle is not None and handle.thread.isRunning():
            handle.thread.quit()

    def _schedule_event_stream_reconnect(self, *, reset: bool = False) -> None:
        if reset:
            self._sse_reconnect_ms = SSE_RECONNECT_BASE_MS
        delay_ms = self._sse_reconnect_ms
        self._sse_reconnect_ms = min(self._sse_reconnect_ms * 2, SSE_RECONNECT_MAX_MS)
        self._sse_reconnect_timer.start(delay_ms)

    def _disconnect_event_stream_signals(self, handle: EventStreamHandle) -> None:
        for signal, slot in (
            (handle.worker.connected, self._handle_event_stream_connected),
            (handle.worker.event_ready, self._handle_event_stream_event),
            (handle.worker.error, self._handle_event_stream_error),
            (handle.worker.stopped, self._handle_event_stream_stopped),
        ):
            with suppress(RuntimeError, TypeError):
                signal.disconnect(slot)

    def _prune_event_stream_orphan(self, handle: EventStreamHandle) -> None:
        self._orphan_event_streams = [h for h in self._orphan_event_streams if h is not handle]

    def _prune_completed_event_stream_orphans(self) -> None:
        self._orphan_event_streams = [
            handle for handle in self._orphan_event_streams if handle.thread.isRunning()
        ]

    def _drain_orphan_event_streams(self) -> None:
        self._prune_completed_event_stream_orphans()
        graceful_deadline = monotonic() + 0.5
        remaining_orphans: list[EventStreamHandle] = []
        for handle in self._orphan_event_streams:
            remaining_ms = int(max(0.0, graceful_deadline - monotonic()) * 1000)
            if remaining_ms > 0:
                handle.thread.wait(remaining_ms)
            if handle.thread.isRunning():
                remaining_orphans.append(handle)
        self._orphan_event_streams = remaining_orphans

    # ------------------------------------------------------------------
    # On-demand refresh
    # ------------------------------------------------------------------

    def refresh_now(self, job_name: str) -> None:
        """Trigger an off-cadence fetch for a running job.

        The worker lives on its own QThread; calling `refresh_now`
        directly would run the network fetch on the UI thread (and
        violate Qt's "QObject lives on its thread" contract). Queue
        the slot so it executes on the worker thread.
        """
        handle = self._jobs.get(job_name)
        if handle is None:
            return
        # PySide6's runtime invokeMethod takes the slot name as `str`;
        # the bundled type stubs incorrectly require `bytes`, so the
        # ignore is for mypy strict mode only.
        QMetaObject.invokeMethod(  # type: ignore[call-overload]
            handle.worker, "refresh_now", Qt.ConnectionType.QueuedConnection
        )

    # ------------------------------------------------------------------
    # Stimulus (one-shot write path)
    # ------------------------------------------------------------------

    def submit_stimulus(self, session_id: UUID, request: StimulusRequest) -> OneShotSignals:
        """Dispatch a session-scoped stimulus POST on the thread pool. §4.C.

        On success, overview / live-session / alerts refresh immediately
        so the UI reflects the new lifecycle state without waiting for
        the next tick. On failure the store's `JOB_STIMULUS` error
        scope is populated and `job_failed` fires.
        """

        def fn() -> object:
            return self._client.post_stimulus(session_id, request)

        signals = run_one_shot(JOB_STIMULUS, fn)
        # Track the signals object so Qt does not GC it mid-flight.
        handle_key = str(request.client_action_id)
        self._inflight_stimulus[handle_key] = signals

        def on_succeeded(_job: str, _payload: object) -> None:
            self._store.clear_error(JOB_STIMULUS)
            # Operator-visible refresh fan-out: the encounter that this
            # stimulus belongs to needs the next read to include the
            # accepted state, and the attention queue may gain a new
            # alert entry.
            for target in (JOB_OVERVIEW, JOB_LIVE_SESSION, JOB_ENCOUNTERS, JOB_ALERTS):
                self.refresh_now(target)

        def on_failed(_job: str, error: object) -> None:
            message = str(error) if not isinstance(error, ApiError) else error.message
            self._store.set_error(JOB_STIMULUS, message)
            self.job_failed.emit(JOB_STIMULUS, message)

        def on_finished(_job: str) -> None:
            self._inflight_stimulus.pop(handle_key, None)

        signals.succeeded.connect(on_succeeded)
        signals.failed.connect(on_failed)
        signals.finished.connect(on_finished)
        return signals

    def request_session_start(self, request: SessionCreateRequest) -> OneShotSignals:
        """Dispatch `POST /api/v1/sessions` via the one-shot write path."""

        def fn() -> object:
            return self._client.post_session_start(request)

        signals = run_one_shot(JOB_SESSION_START, fn)
        handle_key = str(request.client_action_id)
        self._inflight_session_lifecycle[handle_key] = signals

        def on_succeeded(_job: str, payload: object) -> None:
            self._store.clear_error(JOB_SESSION_START)
            if (
                isinstance(payload, SessionLifecycleAccepted)
                and payload.accepted
                and payload.action == "start"
            ):
                self._store.set_selected_session_id(payload.session_id)
            for target in (
                JOB_OVERVIEW,
                JOB_SESSIONS,
                JOB_LIVE_SESSION,
                JOB_ENCOUNTERS,
                JOB_ALERTS,
            ):
                self.refresh_now(target)

        def on_failed(_job: str, error: object) -> None:
            message = str(error) if not isinstance(error, ApiError) else error.message
            self._store.set_error(JOB_SESSION_START, message)
            self.job_failed.emit(JOB_SESSION_START, message)

        def on_finished(_job: str) -> None:
            self._inflight_session_lifecycle.pop(handle_key, None)

        signals.succeeded.connect(on_succeeded)
        signals.failed.connect(on_failed)
        signals.finished.connect(on_finished)
        return signals

    def request_session_end(
        self,
        session_id: UUID,
        request: SessionEndRequest,
    ) -> OneShotSignals:
        """Dispatch `POST /api/v1/sessions/{id}/end` via the one-shot path."""

        def fn() -> object:
            return self._client.post_session_end(session_id, request)

        signals = run_one_shot(JOB_SESSION_END, fn)
        handle_key = str(request.client_action_id)
        self._inflight_session_lifecycle[handle_key] = signals

        def on_succeeded(_job: str, _payload: object) -> None:
            self._store.clear_error(JOB_SESSION_END)
            for target in (
                JOB_OVERVIEW,
                JOB_SESSIONS,
                JOB_LIVE_SESSION,
                JOB_ENCOUNTERS,
                JOB_HEALTH,
                JOB_ALERTS,
            ):
                self.refresh_now(target)

        def on_failed(_job: str, error: object) -> None:
            message = str(error) if not isinstance(error, ApiError) else error.message
            self._store.set_error(JOB_SESSION_END, message)
            self.job_failed.emit(JOB_SESSION_END, message)

        def on_finished(_job: str) -> None:
            self._inflight_session_lifecycle.pop(handle_key, None)

        signals.succeeded.connect(on_succeeded)
        signals.failed.connect(on_failed)
        signals.finished.connect(on_finished)
        return signals

    # ------------------------------------------------------------------
    # Experiment management writes (one-shot admin API calls)
    # ------------------------------------------------------------------

    def create_experiment(self, request: ExperimentCreateRequest) -> OneShotSignals:
        """Dispatch `POST /api/v1/experiments` via the one-shot pattern."""

        def fn() -> object:
            return self._client.create_experiment(request)

        return self._run_experiment_mutation(fn)

    def add_experiment_arm(
        self,
        experiment_id: str,
        request: ExperimentArmCreateRequest,
    ) -> OneShotSignals:
        """Dispatch `POST /api/v1/experiments/{id}/arms` as a one-shot."""

        def fn() -> object:
            return self._client.add_experiment_arm(experiment_id, request)

        return self._run_experiment_mutation(fn)

    def patch_experiment_arm(
        self,
        experiment_id: str,
        arm_id: str,
        request: ExperimentArmPatchRequest,
    ) -> OneShotSignals:
        """Dispatch a supported arm patch as a one-shot admin write."""

        def fn() -> object:
            return self._client.patch_experiment_arm(experiment_id, arm_id, request)

        return self._run_experiment_mutation(fn)

    def rename_experiment_arm(
        self,
        experiment_id: str,
        arm_id: str,
        greeting_text: str,
    ) -> OneShotSignals:
        """Rename arm greeting text; posterior-owned fields are not writable."""
        return self.patch_experiment_arm(
            experiment_id,
            arm_id,
            ExperimentArmPatchRequest(greeting_text=greeting_text),
        )

    def disable_experiment_arm(self, experiment_id: str, arm_id: str) -> OneShotSignals:
        """Disable an arm using the only supported enabled-state mutation."""
        return self.patch_experiment_arm(
            experiment_id,
            arm_id,
            ExperimentArmPatchRequest(enabled=False),
        )

    def delete_experiment_arm(self, experiment_id: str, arm_id: str) -> OneShotSignals:
        """Dispatch guarded arm DELETE as a one-shot admin write."""

        def fn() -> object:
            return self._client.delete_experiment_arm(experiment_id, arm_id)

        return self._run_experiment_mutation(fn)

    def repair_install(self) -> OneShotSignals:
        signals = run_one_shot(JOB_REPAIR_INSTALL, repair_runtime)
        handle_key = str(uuid4())
        self._inflight_repairs[handle_key] = signals

        def on_succeeded(_job: str, _payload: object) -> None:
            self._store.clear_error(JOB_REPAIR_INSTALL)
            for target in (JOB_HEALTH, JOB_ALERTS):
                self.refresh_now(target)

        def on_failed(_job: str, error: object) -> None:
            message = str(error) if not isinstance(error, ApiError) else error.message
            self._store.set_error(JOB_REPAIR_INSTALL, message)
            self.job_failed.emit(JOB_REPAIR_INSTALL, message)

        def on_finished(_job: str) -> None:
            self._inflight_repairs.pop(handle_key, None)

        signals.succeeded.connect(on_succeeded)
        signals.failed.connect(on_failed)
        signals.finished.connect(on_finished)
        return signals

    def cloud_sign_in(self) -> OneShotSignals:
        def fn() -> object:
            result = self._client.post_cloud_sign_in()
            if result.status is CloudActionStatus.FAILED:
                raise ApiError(message=result.message, retryable=result.retryable)
            return result

        return self._run_cloud_action(JOB_CLOUD_SIGN_IN, fn)

    def preview_experiment_bundle_refresh(self) -> OneShotSignals:
        def fn() -> object:
            result = self._client.post_experiment_bundle_refresh_preview()
            if result.status is CloudActionStatus.FAILED:
                raise ApiError(message=result.message, retryable=result.retryable)
            return result

        return self._run_cloud_action(JOB_EXPERIMENT_BUNDLE_REFRESH, fn, refresh_on_success=False)

    def refresh_experiment_bundle(
        self,
        request: ExperimentBundleRefreshRequest,
    ) -> OneShotSignals:
        def fn() -> object:
            result = self._client.post_experiment_bundle_refresh(request)
            if result.status is CloudExperimentRefreshStatus.FAILED:
                raise ApiError(message=result.message, retryable=result.retryable)
            return result

        return self._run_cloud_action(JOB_EXPERIMENT_BUNDLE_REFRESH, fn)

    def _run_cloud_action(
        self,
        job_name: str,
        fn: Callable[[], object],
        *,
        refresh_on_success: bool = True,
    ) -> OneShotSignals:
        signals = run_one_shot(job_name, fn)
        handle_key = str(uuid4())
        self._inflight_cloud_actions[handle_key] = signals

        def on_succeeded(_job: str, _payload: object) -> None:
            self._store.clear_error(job_name)
            if not refresh_on_success:
                return
            for target in (JOB_EXPERIMENT, JOB_EXPERIMENT_SUMMARIES, JOB_HEALTH, JOB_ALERTS):
                self.refresh_now(target)

        def on_failed(_job: str, error: object) -> None:
            message = str(error) if not isinstance(error, ApiError) else error.message
            self._store.set_error(job_name, message)
            self.job_failed.emit(job_name, message)

        def on_finished(_job: str) -> None:
            self._inflight_cloud_actions.pop(handle_key, None)

        signals.succeeded.connect(on_succeeded)
        signals.failed.connect(on_failed)
        signals.finished.connect(on_finished)
        return signals

    def _run_experiment_mutation(self, fn: Callable[[], object]) -> OneShotSignals:
        signals = run_one_shot(JOB_EXPERIMENT, fn)
        handle_key = str(uuid4())
        self._inflight_experiment_mutations[handle_key] = signals

        def on_succeeded(_job: str, payload: object) -> None:
            self._store.clear_error(JOB_EXPERIMENT)
            self._apply_experiment_mutation_payload(payload)
            self.refresh_now(JOB_EXPERIMENT)

        def on_failed(_job: str, error: object) -> None:
            message = str(error) if not isinstance(error, ApiError) else error.message
            self._store.set_error(JOB_EXPERIMENT, message)
            self.job_failed.emit(JOB_EXPERIMENT, message)

        def on_finished(_job: str) -> None:
            self._inflight_experiment_mutations.pop(handle_key, None)

        signals.succeeded.connect(on_succeeded)
        signals.failed.connect(on_failed)
        signals.finished.connect(on_finished)
        return signals

    def _apply_experiment_mutation_payload(self, payload: object) -> None:
        current = self._store.experiment()
        if isinstance(payload, ExperimentAdminResponse):
            self._store.set_experiment(_detail_from_admin_response(payload, current))
        elif isinstance(payload, ExperimentArmAdminResponse):
            self._store.set_experiment(_detail_with_admin_arm(payload, current))
        elif isinstance(payload, ExperimentArmDeleteResponse):
            self._store.set_managed_experiment_id(payload.experiment_id)
            if payload.deleted:
                self._store.set_experiment(_detail_without_admin_arm(payload, current))
            elif payload.arm_state is not None:
                self._store.set_experiment(_detail_with_admin_arm(payload.arm_state, current))

    # ------------------------------------------------------------------
    # Spec registration — one place that maps jobs to cadences.
    # ------------------------------------------------------------------

    def _register_jobs(self) -> dict[str, PollJobSpec]:
        cfg = self._config
        specs: list[PollJobSpec] = [
            # Overview card — only polls on the OVERVIEW route.
            PollJobSpec(JOB_OVERVIEW, cfg.overview_poll_ms, AppRoute.OVERVIEW),
            # Recent sessions table — SESSIONS route only.
            PollJobSpec(JOB_SESSIONS, cfg.sessions_poll_ms, AppRoute.SESSIONS),
            # Live-session header + encounters — LIVE_SESSION route.
            PollJobSpec(
                JOB_LIVE_SESSION,
                cfg.session_header_poll_ms,
                AppRoute.LIVE_SESSION,
                session_scoped=True,
            ),
            PollJobSpec(
                JOB_ENCOUNTERS,
                cfg.live_encounters_poll_ms,
                AppRoute.LIVE_SESSION,
                session_scoped=True,
            ),
            PollJobSpec(
                JOB_EXPERIMENT_SUMMARIES,
                cfg.experiments_poll_ms,
                AppRoute.LIVE_SESSION,
            ),
            # Experiment detail — EXPERIMENTS route.
            PollJobSpec(JOB_EXPERIMENT, cfg.experiments_poll_ms, AppRoute.EXPERIMENTS),
            # Physiology — PHYSIOLOGY route.
            PollJobSpec(
                JOB_PHYSIOLOGY,
                cfg.physiology_poll_ms,
                AppRoute.PHYSIOLOGY,
                session_scoped=True,
            ),
            # Health rollup — visible on LIVE_SESSION and HEALTH routes.
            PollJobSpec(
                JOB_HEALTH,
                cfg.health_poll_ms,
                frozenset({AppRoute.LIVE_SESSION, AppRoute.HEALTH}),
            ),
            PollJobSpec(JOB_CLOUD_AUTH_STATUS, cfg.health_poll_ms, AppRoute.HEALTH),
            PollJobSpec(JOB_CLOUD_OUTBOX, cfg.health_poll_ms, AppRoute.HEALTH),
            # Alerts — always on; attention queue must stay current on
            # every page, per the §4.E.1 multi-page layout.
            PollJobSpec(JOB_ALERTS, cfg.alerts_poll_ms, route_scoped=None),
        ]
        return {spec.name: spec for spec in specs}

    # ------------------------------------------------------------------
    # Sync: start/stop workers so the running set matches the current
    # route + session selection.
    # ------------------------------------------------------------------

    def _sync_jobs_for_current_state(self) -> None:
        self._prune_completed_orphans()
        current_route = self._store.route()
        selected = self._store.selected_session_id()
        want: set[str] = set()
        for name, spec in self._specs.items():
            if not _route_matches(spec.route_scoped, current_route):
                continue
            if self._event_stream_connected and name in _HIGH_FREQUENCY_SSE_JOBS:
                continue
            if spec.session_scoped and selected is None:
                continue
            want.add(name)
        for name in list(self._jobs):
            if name not in want:
                self._stop_job(name)
        for name in want:
            if name not in self._jobs:
                self._start_job(self._specs[name])

    # ------------------------------------------------------------------
    # Per-job start/stop
    # ------------------------------------------------------------------

    def _start_job(self, spec: PollJobSpec) -> None:
        fetch = self._make_fetch(spec.name)
        thread = QThread(self)
        worker = PollingWorker(spec.name, spec.interval_ms, fetch)
        worker.moveToThread(thread)
        worker.data_ready.connect(self._handle_job_data)
        worker.error.connect(self._handle_job_error)
        # Tear-down chain (canonical Qt pattern). When the worker emits
        # `stopped` from inside its own `stop()` slot, two cleanups
        # fire in order:
        #   1. `worker.deleteLater` — same-thread direct connection,
        #      which posts a DeferredDelete event to the worker thread
        #      so the worker (and its child QTimer) are destroyed on
        #      the right thread. Without this, Python GC eventually
        #      runs the worker's destructor on the main thread and Qt
        #      raises "Timers cannot be stopped from another thread".
        #   2. `thread.quit` — `QThread.quit()` is thread-safe, so we
        #      pin DirectConnection. The default AutoConnection would
        #      queue it back to the main thread, which deadlocks the
        #      shutdown drain (`_drain_orphan_jobs` blocks main on
        #      `thread.wait()`, so a queued slot on main never runs).
        worker.stopped.connect(worker.deleteLater)
        worker.stopped.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(lambda handle_worker=worker: self._prune_orphan(handle_worker))
        thread.started.connect(worker.run)
        thread.start()
        self._jobs[spec.name] = _JobHandle(spec, worker, thread)

    def _stop_job(self, job_name: str) -> None:
        """Tear down a job without blocking the UI.

        The worker's QTimer was created on the worker thread, so
        `stop()` must run on that thread — invoking it from the UI
        thread emits Qt's "Timers cannot be stopped from another
        thread" warning and leaks the timer. We queue the slot and
        park the handle on `_orphan_jobs`; the `stopped → deleteLater
        → thread.quit` chain wired in `_start_job` does the rest. The
        UI never blocks waiting because route changes need to feel
        instant even when the API is unreachable and the worker is
        mid-urlopen. Final shutdown drains the orphan list with a
        real wait.
        """
        handle = self._jobs.pop(job_name, None)
        if handle is None:
            return
        self._disconnect_worker_signals(handle.worker)
        QMetaObject.invokeMethod(  # type: ignore[call-overload]
            handle.worker, "stop", Qt.ConnectionType.QueuedConnection
        )
        # Note: do NOT call `handle.thread.quit()` here. The connect
        # in `_start_job` quits the thread once the worker has
        # finished its stop slot; quitting first races the queued
        # stop slot and can exit the event loop before the worker is
        # safely deleted on its own thread.
        self._orphan_jobs.append(handle)

    def _disconnect_worker_signals(self, worker: PollingWorker) -> None:
        for signal, slot in (
            (worker.data_ready, self._handle_job_data),
            (worker.error, self._handle_job_error),
        ):
            with suppress(RuntimeError, TypeError):
                signal.disconnect(slot)

    def _prune_orphan(self, worker: PollingWorker) -> None:
        self._orphan_jobs = [h for h in self._orphan_jobs if h.worker is not worker]

    def _prune_completed_orphans(self) -> None:
        self._orphan_jobs = [h for h in self._orphan_jobs if h.thread.isRunning()]

    def _drain_orphan_jobs(self) -> None:
        """Give orphaned worker threads a short cooperative shutdown window."""
        graceful_deadline = monotonic() + 0.5
        remaining_orphans: list[_JobHandle] = []
        for handle in self._orphan_jobs:
            remaining_ms = int(max(0.0, graceful_deadline - monotonic()) * 1000)
            if remaining_ms > 0:
                handle.thread.wait(remaining_ms)
            if handle.thread.isRunning():
                remaining_orphans.append(handle)
        self._orphan_jobs = remaining_orphans

    # ------------------------------------------------------------------
    # Slot endpoints
    # ------------------------------------------------------------------

    @Slot(str, object)
    def _handle_job_data(self, job_name: str, payload: object) -> None:
        # Any data arrival clears the error scope for this job; the
        # next fetch attempt re-populates it if the failure persists.
        self.apply_payload(job_name, payload)

    @Slot(str, object)
    def _handle_job_error(self, job_name: str, error: object) -> None:
        message = str(error) if not isinstance(error, ApiError) else error.message
        # Non-retryable errors are the ones worth surfacing on the card
        # immediately; retryable errors (URLError/Timeout/5xx) get the
        # same treatment; add a grace-period suppression here if needed.
        # Either way the error_changed signal carries the job/scope so
        # the UI can attribute it correctly.
        self._store.set_error(job_name, message)
        self.job_failed.emit(job_name, message)

    # ------------------------------------------------------------------
    # Payload dispatch: route each job's DTO to the right store setter.
    # ------------------------------------------------------------------

    def apply_payload(self, job_name: str, payload: object) -> None:
        self._store.clear_error(job_name)
        self._apply_payload(job_name, payload)

    def apply_event_payload(self, job_name: str, payload: object) -> None:
        if not self._event_payload_matches_current_scope(job_name, payload):
            return
        self._store.clear_error(job_name)
        self._apply_event_payload(job_name, payload)

    def _apply_payload(self, job_name: str, payload: object) -> None:
        if job_name == JOB_OVERVIEW and isinstance(payload, OverviewSnapshot):
            self._store.set_overview(payload)
            # Overview composes several surfaces, so reflect its sub-
            # components into their dedicated store slots too.
            self._store.set_live_session(payload.active_session)
            if payload.health is not None:
                self._store.set_health(payload.health)
            if payload.physiology is not None:
                self._store.set_physiology(payload.physiology)
            if payload.alerts:
                self._store.set_alerts(list(payload.alerts))
        elif job_name == JOB_SESSIONS and isinstance(payload, list):
            self._store.set_sessions(_as_list(payload, SessionSummary))
        elif job_name == JOB_LIVE_SESSION:
            if isinstance(payload, SessionSummary):
                self._store.set_live_session(payload)
            elif payload == []:
                self._store.set_live_session(None)
        elif job_name == JOB_ENCOUNTERS and isinstance(payload, list):
            self._store.set_encounters(_as_list(payload, EncounterSummary))
        elif job_name == JOB_EXPERIMENT_SUMMARIES and isinstance(payload, list):
            self._store.set_experiment_summaries(_as_list(payload, ExperimentSummary))
        elif job_name == JOB_EXPERIMENT and isinstance(payload, ExperimentDetail):
            self._store.set_experiment(payload)
        elif job_name == JOB_PHYSIOLOGY:
            if isinstance(payload, SessionPhysiologySnapshot):
                self._store.set_physiology(payload)
            elif payload == []:
                self._store.set_physiology(None)
        elif job_name == JOB_HEALTH and isinstance(payload, HealthSnapshot):
            self._store.set_health(payload)
        elif job_name == JOB_CLOUD_AUTH_STATUS and isinstance(payload, CloudAuthStatus):
            self._store.set_cloud_auth_status(payload)
        elif job_name == JOB_CLOUD_OUTBOX and isinstance(payload, CloudOutboxSummary):
            self._store.set_cloud_outbox_summary(payload)
        elif job_name == JOB_ALERTS and isinstance(payload, list):
            self._store.set_alerts(_as_list(payload, AlertEvent))
        else:
            # Shape mismatch is a bug, not a recoverable error. Surface
            # it via the same error scope so tests see it.
            self._store.set_error(job_name, f"unexpected payload shape for {job_name}")

    def _apply_event_payload(self, job_name: str, payload: object) -> None:
        if job_name == JOB_OVERVIEW and isinstance(payload, OverviewSnapshot):
            self._store.set_overview(payload)
            if payload.health is not None:
                self._store.set_health(payload.health)
            if payload.alerts:
                self._store.set_alerts(list(payload.alerts))
        elif job_name == JOB_EXPERIMENT:
            if isinstance(payload, ExperimentDetail):
                self._store.set_experiment_readback(payload)
        else:
            self._apply_payload(job_name, payload)

    def _event_payload_matches_current_scope(self, job_name: str, payload: object) -> bool:
        selected = self._store.selected_session_id()
        if job_name == JOB_LIVE_SESSION:
            if isinstance(payload, SessionSummary):
                return selected is not None and payload.session_id == selected
            return selected is None and payload == []
        if job_name == JOB_ENCOUNTERS and isinstance(payload, list):
            encounters = _as_list(payload, EncounterSummary)
            if not encounters:
                return selected is None
            return selected is not None and all(row.session_id == selected for row in encounters)
        if job_name == JOB_PHYSIOLOGY:
            if isinstance(payload, SessionPhysiologySnapshot):
                return selected is not None and payload.session_id == selected
            return selected is None and payload == []
        if job_name == JOB_EXPERIMENT:
            if payload == []:
                return True
            if isinstance(payload, ExperimentDetail):
                managed = self._store.managed_experiment_id()
                return managed is not None and payload.experiment_id == managed
            return False
        return True

    # ------------------------------------------------------------------
    # Fetch-callable factories (one per job). Each captures client +
    # current session id at construction time; the coordinator rebuilds
    # the worker on session change so the id stays current.
    # ------------------------------------------------------------------

    def _make_fetch(self, job_name: str) -> Callable[[], object]:
        if job_name == JOB_OVERVIEW:
            return self._make_fetch_overview()
        if job_name == JOB_SESSIONS:
            return self._make_fetch_sessions()
        if job_name == JOB_LIVE_SESSION:
            return self._make_fetch_live_session()
        if job_name == JOB_ENCOUNTERS:
            return self._make_fetch_encounters()
        if job_name == JOB_EXPERIMENT_SUMMARIES:
            return self._make_fetch_experiment_summaries()
        if job_name == JOB_EXPERIMENT:
            return self._make_fetch_experiment()
        if job_name == JOB_PHYSIOLOGY:
            return self._make_fetch_physiology()
        if job_name == JOB_HEALTH:
            return self._make_fetch_health()
        if job_name == JOB_CLOUD_AUTH_STATUS:
            return self._make_fetch_cloud_auth_status()
        if job_name == JOB_CLOUD_OUTBOX:
            return self._make_fetch_cloud_outbox()
        if job_name == JOB_ALERTS:
            return self._make_fetch_alerts()
        raise ValueError(f"unknown job: {job_name}")

    def _make_fetch_overview(self) -> Callable[[], OverviewSnapshot]:
        client = self._client

        def fetch() -> OverviewSnapshot:
            return client.get_overview()

        return fetch

    def _make_fetch_sessions(self) -> Callable[[], list[SessionSummary]]:
        client = self._client

        def fetch() -> list[SessionSummary]:
            return client.list_sessions()

        return fetch

    def _make_fetch_live_session(self) -> Callable[[], SessionSummary]:
        client = self._client
        session_id = self._store.selected_session_id()
        if session_id is None:
            raise RuntimeError("live-session job started without a selected session")
        captured_id = session_id

        def fetch() -> SessionSummary:
            return client.get_session(captured_id)

        return fetch

    def _make_fetch_encounters(self) -> Callable[[], list[EncounterSummary]]:
        client = self._client
        session_id = self._store.selected_session_id()
        if session_id is None:
            raise RuntimeError("encounters job started without a selected session")
        captured_id = session_id

        def fetch() -> list[EncounterSummary]:
            return client.list_session_encounters(captured_id)

        return fetch

    def _make_fetch_experiment_summaries(self) -> Callable[[], list[ExperimentSummary]]:
        client = self._client

        def fetch() -> list[ExperimentSummary]:
            return client.list_experiments()

        return fetch

    def _make_fetch_experiment(self) -> Callable[[], ExperimentDetail]:
        client = self._client
        store = self._store
        default_experiment_id = self._config.default_experiment_id

        def fetch() -> ExperimentDetail:
            experiment_id = store.managed_experiment_id() or default_experiment_id
            return client.get_experiment_detail(experiment_id)

        return fetch

    def _make_fetch_physiology(self) -> Callable[[], SessionPhysiologySnapshot]:
        client = self._client
        session_id = self._store.selected_session_id()
        if session_id is None:
            raise RuntimeError("physiology job started without a selected session")
        captured_id = session_id

        def fetch() -> SessionPhysiologySnapshot:
            return client.get_session_physiology(captured_id)

        return fetch

    def _make_fetch_health(self) -> Callable[[], HealthSnapshot]:
        client = self._client

        def fetch() -> HealthSnapshot:
            return client.get_health()

        return fetch

    def _make_fetch_cloud_auth_status(self) -> Callable[[], CloudAuthStatus]:
        client = self._client

        def fetch() -> CloudAuthStatus:
            return client.get_cloud_auth_status()

        return fetch

    def _make_fetch_cloud_outbox(self) -> Callable[[], CloudOutboxSummary]:
        client = self._client

        def fetch() -> CloudOutboxSummary:
            return client.get_cloud_outbox_summary()

        return fetch

    def _make_fetch_alerts(self) -> Callable[[], list[AlertEvent]]:
        client = self._client

        def fetch() -> list[AlertEvent]:
            return client.list_alerts()

        return fetch


def _route_matches(
    route_scoped: AppRoute | Container[AppRoute] | None,
    current_route: AppRoute,
) -> bool:
    if route_scoped is None:
        return True
    if isinstance(route_scoped, AppRoute):
        return route_scoped == current_route
    return current_route in route_scoped


def _detail_from_admin_response(
    payload: ExperimentAdminResponse,
    current: ExperimentDetail | None,
) -> ExperimentDetail:
    current_detail = (
        current if current is not None and current.experiment_id == payload.experiment_id else None
    )
    return ExperimentDetail(
        experiment_id=payload.experiment_id,
        label=payload.label,
        active_arm_id=current_detail.active_arm_id if current_detail is not None else None,
        arms=[_arm_from_admin_response(arm, None) for arm in payload.arms],
        last_update_summary=(
            current_detail.last_update_summary if current_detail is not None else None
        ),
        last_updated_utc=_latest_admin_timestamp(payload.arms),
    )


def _detail_with_admin_arm(
    payload: ExperimentArmAdminResponse,
    current: ExperimentDetail | None,
) -> ExperimentDetail:
    current_detail = (
        current if current is not None and current.experiment_id == payload.experiment_id else None
    )
    existing_arms = list(current_detail.arms) if current_detail is not None else []
    merged: list[ArmSummary] = []
    matched = False
    for existing in existing_arms:
        if existing.arm_id == payload.arm:
            merged.append(_arm_from_admin_response(payload, existing))
            matched = True
        else:
            merged.append(existing)
    if not matched:
        merged.append(_arm_from_admin_response(payload, None))
    return ExperimentDetail(
        experiment_id=payload.experiment_id,
        label=payload.label
        if payload.label
        else (current_detail.label if current_detail is not None else None),
        active_arm_id=current_detail.active_arm_id if current_detail is not None else None,
        arms=merged,
        last_update_summary=(
            current_detail.last_update_summary if current_detail is not None else None
        ),
        last_updated_utc=payload.updated_at
        if payload.updated_at is not None
        else (current_detail.last_updated_utc if current_detail is not None else None),
    )


def _detail_without_admin_arm(
    payload: ExperimentArmDeleteResponse,
    current: ExperimentDetail | None,
) -> ExperimentDetail:
    current_detail = (
        current if current is not None and current.experiment_id == payload.experiment_id else None
    )
    remaining_arms = (
        [arm for arm in current_detail.arms if arm.arm_id != payload.arm]
        if current_detail is not None
        else []
    )
    return ExperimentDetail(
        experiment_id=payload.experiment_id,
        label=current_detail.label if current_detail is not None else None,
        active_arm_id=current_detail.active_arm_id if current_detail is not None else None,
        arms=remaining_arms,
        last_update_summary=(
            current_detail.last_update_summary if current_detail is not None else None
        ),
        last_updated_utc=(current_detail.last_updated_utc if current_detail is not None else None),
    )


def _arm_from_admin_response(
    payload: ExperimentArmAdminResponse,
    existing: ArmSummary | None,
) -> ArmSummary:
    total = payload.alpha_param + payload.beta_param
    variance = (
        (payload.alpha_param * payload.beta_param) / (total * total * (total + 1.0))
        if total > 0
        else None
    )
    return ArmSummary(
        arm_id=payload.arm,
        greeting_text=payload.greeting_text,
        posterior_alpha=payload.alpha_param,
        posterior_beta=payload.beta_param,
        evaluation_variance=variance,
        selection_count=existing.selection_count if existing is not None else 0,
        recent_reward_mean=existing.recent_reward_mean if existing is not None else None,
        recent_semantic_pass_rate=(
            existing.recent_semantic_pass_rate if existing is not None else None
        ),
        enabled=payload.enabled,
        end_dated_at=payload.end_dated_at,
    )


def _latest_admin_timestamp(rows: list[ExperimentArmAdminResponse]) -> datetime | None:
    latest: datetime | None = None
    for row in rows:
        updated = row.updated_at
        if updated is None:
            continue
        if latest is None or updated > latest:
            latest = updated
    return latest


def _as_list(payload: object, item_type: type[Any]) -> list[Any]:
    """Narrow `list[object]` payloads to `list[item_type]`.

    The `ApiClient` already validated each item via its TypeAdapter; this
    helper is a defensive pass for the handful of places where a slot
    argument arrives typed as `object`.
    """
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, item_type)]
