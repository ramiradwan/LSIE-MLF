"""
Polling coordinator — Phase 4 of the Operator Console cycle.

Single authority over the console's polling job lifecycle. Views and
viewmodels never talk to `ApiClient` directly; they read from
`OperatorStore`, and the coordinator is the only thing that drives
store mutations in response to network fetches. This separation is
what lets Phase 5+ build widgets that are pure presentation.

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
  - Stimulus submission is a one-shot via `run_one_shot`; on success
    the coordinator immediately refreshes overview, live-session, and
    alerts so the UI reflects the new lifecycle state without waiting
    for the next tick.

Spec references:
  §4.C           — stimulus is the only write surface; idempotency
                   lives in the API Server, not here
  §4.E.1         — operator-facing aggregate endpoints
  §12            — retryable vs non-retryable errors flow through the
                   store's per-scope error signal
  SPEC-AMEND-008 — PySide6 Operator Console
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import Any
from uuid import UUID

from PySide6.QtCore import QMetaObject, QObject, Qt, QThread, Signal, Slot

from packages.schemas.operator_console import (
    AlertEvent,
    EncounterSummary,
    ExperimentDetail,
    HealthSnapshot,
    OverviewSnapshot,
    SessionPhysiologySnapshot,
    SessionSummary,
    StimulusRequest,
)
from services.operator_console.api_client import ApiClient, ApiError
from services.operator_console.config import OperatorConsoleConfig
from services.operator_console.state import AppRoute, OperatorStore
from services.operator_console.workers import (
    OneShotSignals,
    PollingWorker,
    run_one_shot,
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
JOB_PHYSIOLOGY = "physiology"
JOB_HEALTH = "health"
JOB_ALERTS = "alerts"
JOB_STIMULUS = "stimulus"


@dataclass(frozen=True)
class PollJobSpec:
    """Static description of a polling job.

    `route_scoped=None` means "always run while the coordinator is
    started". A non-None value means the job only runs while the named
    route is active; switching away stops it until the operator returns.
    """

    name: str
    interval_ms: int
    route_scoped: AppRoute | None = None
    session_scoped: bool = False


class _JobHandle:
    """Bookkeeping for one live job: its worker, its thread, its spec."""

    __slots__ = ("spec", "worker", "thread")

    def __init__(self, spec: PollJobSpec, worker: PollingWorker, thread: QThread) -> None:
        self.spec = spec
        self.worker = worker
        self.thread = thread


class PollingCoordinator(QObject):
    """Orchestrates every poll job plus the single write path (stimulus).

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
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._client = client
        self._store = store
        self._specs: dict[str, PollJobSpec] = self._register_jobs()
        self._jobs: dict[str, _JobHandle] = {}
        # Jobs that have been told to stop but whose worker thread may
        # still be draining a slow urlopen. They drain on their own time
        # so route-change teardown does not block the UI; final shutdown
        # joins anything still alive.
        self._orphan_jobs: list[_JobHandle] = []
        self._inflight_stimulus: dict[str, OneShotSignals] = {}
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
        for name in list(self._jobs):
            self._stop_job(name)
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
    # Stimulus (single write path)
    # ------------------------------------------------------------------

    def submit_stimulus(self, session_id: UUID, request: StimulusRequest) -> OneShotSignals:
        """Dispatch a `POST /stimulus` on the thread pool. §4.C.

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
            for target in (JOB_OVERVIEW, JOB_LIVE_SESSION, JOB_ALERTS):
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
            # Experiment detail — EXPERIMENTS route.
            PollJobSpec(JOB_EXPERIMENT, cfg.experiments_poll_ms, AppRoute.EXPERIMENTS),
            # Physiology — PHYSIOLOGY route.
            PollJobSpec(
                JOB_PHYSIOLOGY,
                cfg.physiology_poll_ms,
                AppRoute.PHYSIOLOGY,
                session_scoped=True,
            ),
            # Health rollup — HEALTH route.
            PollJobSpec(JOB_HEALTH, cfg.health_poll_ms, AppRoute.HEALTH),
            # Alerts — always on; attention queue must stay current on
            # every page, per SPEC-AMEND-008's multi-page layout.
            PollJobSpec(JOB_ALERTS, cfg.alerts_poll_ms, route_scoped=None),
        ]
        return {spec.name: spec for spec in specs}

    # ------------------------------------------------------------------
    # Sync: start/stop workers so the running set matches the current
    # route + session selection.
    # ------------------------------------------------------------------

    def _sync_jobs_for_current_state(self) -> None:
        current_route = self._store.route()
        selected = self._store.selected_session_id()
        want: set[str] = set()
        for name, spec in self._specs.items():
            if spec.route_scoped is not None and spec.route_scoped != current_route:
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
        QMetaObject.invokeMethod(  # type: ignore[call-overload]
            handle.worker, "stop", Qt.ConnectionType.QueuedConnection
        )
        # Note: do NOT call `handle.thread.quit()` here. The connect
        # in `_start_job` quits the thread once the worker has
        # finished its stop slot; quitting first races the queued
        # stop slot and can exit the event loop before the worker is
        # safely deleted on its own thread.
        # Opportunistically prune orphans whose threads have already
        # finished so the list does not grow without bound across
        # many route changes.
        self._orphan_jobs = [h for h in self._orphan_jobs if h.thread.isRunning()]
        if handle.thread.isRunning():
            self._orphan_jobs.append(handle)

    def _drain_orphan_jobs(self) -> None:
        """Wait briefly for orphaned worker threads to finish.

        Called from `stop()` only. Both wait windows are *shared*
        across all orphans rather than budgeted per-thread — with
        several orphans accumulated from rapid route changes, a per-
        thread wait of even a few hundred ms multiplies out into a
        perceptible freeze on the close button. Anything still
        running past the shared grace is forcibly terminated; the
        process is exiting, so a forceful kill is acceptable — the
        OS reclaims the socket and `terminate()` is the only lever
        Qt gives us against a Python thread blocked in C-level
        network I/O.
        """
        # Phase 1: shared graceful window. Enough for an idle worker
        # to process its queued stop slot and exit on its own.
        graceful_deadline = monotonic() + 0.5
        for handle in self._orphan_jobs:
            remaining_ms = int(max(0.0, graceful_deadline - monotonic()) * 1000)
            if remaining_ms > 0:
                handle.thread.wait(remaining_ms)
        # Phase 2: fire terminate on every still-running thread in a
        # single batch so they die concurrently. `terminate()` is
        # async on Windows, so don't pair each one with its own wait.
        stuck = [h for h in self._orphan_jobs if h.thread.isRunning()]
        for handle in stuck:
            handle.thread.terminate()
        # Phase 3: shared short wait for the terminated threads to
        # actually clear. ~300ms total is plenty for a TerminateThread
        # call to settle.
        terminate_deadline = monotonic() + 0.3
        for handle in stuck:
            remaining_ms = int(max(0.0, terminate_deadline - monotonic()) * 1000)
            if remaining_ms > 0:
                handle.thread.wait(remaining_ms)
        self._orphan_jobs.clear()

    # ------------------------------------------------------------------
    # Slot endpoints
    # ------------------------------------------------------------------

    @Slot(str, object)
    def _handle_job_data(self, job_name: str, payload: object) -> None:
        # Any data arrival clears the error scope for this job; the
        # next fetch attempt re-populates it if the failure persists.
        self._store.clear_error(job_name)
        self._apply_payload(job_name, payload)

    @Slot(str, object)
    def _handle_job_error(self, job_name: str, error: object) -> None:
        message = str(error) if not isinstance(error, ApiError) else error.message
        # Non-retryable errors are the ones worth surfacing on the card
        # immediately; retryable errors (URLError/Timeout/5xx) get the
        # same treatment for v1 — Phase 6 can add a grace-period
        # suppression if needed. Either way the error_changed signal
        # carries the job/scope so the UI can attribute it correctly.
        self._store.set_error(job_name, message)
        self.job_failed.emit(job_name, message)

    # ------------------------------------------------------------------
    # Payload dispatch: route each job's DTO to the right store setter.
    # ------------------------------------------------------------------

    def _apply_payload(self, job_name: str, payload: object) -> None:
        if job_name == JOB_OVERVIEW and isinstance(payload, OverviewSnapshot):
            self._store.set_overview(payload)
            # Overview composes several surfaces, so reflect its sub-
            # components into their dedicated store slots too.
            if payload.active_session is not None:
                self._store.set_live_session(payload.active_session)
            if payload.health is not None:
                self._store.set_health(payload.health)
            if payload.physiology is not None:
                self._store.set_physiology(payload.physiology)
            if payload.alerts:
                self._store.set_alerts(list(payload.alerts))
        elif job_name == JOB_SESSIONS and isinstance(payload, list):
            self._store.set_sessions(_as_list(payload, SessionSummary))
        elif job_name == JOB_LIVE_SESSION and isinstance(payload, SessionSummary):
            self._store.set_live_session(payload)
        elif job_name == JOB_ENCOUNTERS and isinstance(payload, list):
            self._store.set_encounters(_as_list(payload, EncounterSummary))
        elif job_name == JOB_EXPERIMENT and isinstance(payload, ExperimentDetail):
            self._store.set_experiment(payload)
        elif job_name == JOB_PHYSIOLOGY and isinstance(payload, SessionPhysiologySnapshot):
            self._store.set_physiology(payload)
        elif job_name == JOB_HEALTH and isinstance(payload, HealthSnapshot):
            self._store.set_health(payload)
        elif job_name == JOB_ALERTS and isinstance(payload, list):
            self._store.set_alerts(_as_list(payload, AlertEvent))
        else:
            # Shape mismatch is a bug, not a recoverable error. Surface
            # it via the same error scope so tests see it.
            self._store.set_error(job_name, f"unexpected payload shape for {job_name}")

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
        if job_name == JOB_EXPERIMENT:
            return self._make_fetch_experiment()
        if job_name == JOB_PHYSIOLOGY:
            return self._make_fetch_physiology()
        if job_name == JOB_HEALTH:
            return self._make_fetch_health()
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

    def _make_fetch_experiment(self) -> Callable[[], ExperimentDetail]:
        client = self._client
        experiment_id = self._config.default_experiment_id

        def fetch() -> ExperimentDetail:
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

    def _make_fetch_alerts(self) -> Callable[[], list[AlertEvent]]:
        client = self._client

        def fetch() -> list[AlertEvent]:
            return client.list_alerts()

        return fetch


def _as_list(payload: object, item_type: type[Any]) -> list[Any]:
    """Narrow `list[object]` payloads to `list[item_type]`.

    The `ApiClient` already validated each item via its TypeAdapter; this
    helper is a defensive pass for the handful of places where a slot
    argument arrives typed as `object`.
    """
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, item_type)]
