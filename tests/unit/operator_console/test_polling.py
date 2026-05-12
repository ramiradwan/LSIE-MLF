"""Tests for `PollingCoordinator` — Phase 4.

The coordinator owns job lifecycle, route scoping, session scoping,
error routing, and the stimulus one-shot fan-out. Tests avoid spinning
up real QThreads (flaky, slow) by patching `_start_job` / `_stop_job`
so we exercise the routing logic without the transport layer.

Stimulus fan-out is exercised by replacing `run_one_shot` with an
in-process `FakeOneShot` that fires `succeeded` / `failed` immediately
on the test thread.

`TestNonBlockingTeardown` is the one section that spins up real
QThread workers — those tests guard the regression where route-change
teardown blocked the UI on `thread.wait(2000)` and emitted Qt's
"Timers cannot be stopped from another thread" warning.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from PySide6.QtCore import QCoreApplication, QElapsedTimer, QEventLoop, Qt, QThread

from packages.schemas.experiments import (
    ExperimentAdminResponse,
    ExperimentArmAdminResponse,
    ExperimentArmDeleteResponse,
    ExperimentArmPatchRequest,
    ExperimentArmSeedRequest,
    ExperimentCreateRequest,
)
from packages.schemas.operator_console import (
    ArmSummary,
    CloudActionStatus,
    CloudAuthState,
    CloudAuthStatus,
    CloudExperimentRefreshStatus,
    CloudOutboxSummary,
    CloudSignInResult,
    EncounterState,
    EncounterSummary,
    ExperimentBundleRefreshPreview,
    ExperimentBundleRefreshRequest,
    ExperimentBundleRefreshResult,
    ExperimentDetail,
    ExperimentSummary,
    HealthSnapshot,
    HealthState,
    OperatorEventEnvelope,
    OverviewSnapshot,
    SessionCreateRequest,
    SessionEndRequest,
    SessionLifecycleAccepted,
    SessionPhysiologySnapshot,
    SessionSummary,
    StimulusAccepted,
    StimulusRequest,
)
from services.operator_console.api_client import ApiClient, ApiError
from services.operator_console.config import OperatorConsoleConfig, load_config
from services.operator_console.polling import (
    JOB_ALERTS,
    JOB_CLOUD_AUTH_STATUS,
    JOB_CLOUD_OUTBOX,
    JOB_CLOUD_SIGN_IN,
    JOB_ENCOUNTERS,
    JOB_EVENT_STREAM,
    JOB_EXPERIMENT,
    JOB_EXPERIMENT_BUNDLE_REFRESH,
    JOB_EXPERIMENT_SUMMARIES,
    JOB_HEALTH,
    JOB_LIVE_SESSION,
    JOB_OVERVIEW,
    JOB_PHYSIOLOGY,
    JOB_REPAIR_INSTALL,
    JOB_SESSION_END,
    JOB_SESSION_START,
    JOB_SESSIONS,
    JOB_STIMULUS,
    PollingCoordinator,
    PollJobSpec,
)
from services.operator_console.state import AppRoute, OperatorStore
from services.operator_console.table_models.encounters_table_model import EncountersTableModel
from services.operator_console.viewmodels.live_session_vm import LiveSessionViewModel
from services.operator_console.workers import (
    EventStreamHandle,
    EventStreamWorker,
    OneShotSignals,
    PollingWorker,
    run_one_shot,
)

pytestmark = pytest.mark.usefixtures("qt_app")


# ----------------------------------------------------------------------
# Fixtures + helpers
# ----------------------------------------------------------------------


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def _drain_events_until(app: QCoreApplication, predicate: Callable[[], bool]) -> None:
    timer = QElapsedTimer()
    timer.start()
    while not predicate() and timer.elapsed() < 1000:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 25)


@pytest.fixture
def cfg() -> OperatorConsoleConfig:
    # `load_config` with an empty mapping gives us the documented defaults
    return load_config({})


@pytest.fixture
def client() -> ApiClient:
    # Transport is never exercised in these tests (patched _start_job
    # means workers never run), but we still pass a real ApiClient so
    # the factory functions build cleanly.
    return ApiClient("http://api.test")


@dataclass
class _FakeJobHandle:
    """Minimal stand-in for the real _JobHandle so `refresh_now` works."""

    spec: Any
    worker: Any


@dataclass
class _CoordinatorHarness:
    """Bundles the coordinator with recordings of start/stop calls."""

    coordinator: PollingCoordinator
    started: list[str] = field(default_factory=list)
    stopped: list[str] = field(default_factory=list)
    refresh_calls: list[str] = field(default_factory=list)


@pytest.fixture
def harness(
    qt_app: QCoreApplication,
    cfg: OperatorConsoleConfig,
    client: ApiClient,
) -> Iterator[_CoordinatorHarness]:
    del qt_app
    store = OperatorStore()
    coord = PollingCoordinator(cfg, client, store)
    rec = _CoordinatorHarness(coord)

    # Patch lifecycle helpers so no real QThread/QTimer is created.
    def fake_start(spec: Any) -> None:
        rec.started.append(spec.name)
        # Record a handle so `refresh_now` finds the job
        coord._jobs[spec.name] = _FakeJobHandle(spec=spec, worker=None)  # type: ignore[assignment]

    def fake_stop(job_name: str) -> None:
        rec.stopped.append(job_name)
        coord._jobs.pop(job_name, None)

    def fake_refresh(job_name: str) -> None:
        rec.refresh_calls.append(job_name)

    # Patch instance methods directly
    coord._start_job = fake_start  # type: ignore[method-assign]
    coord._stop_job = fake_stop  # type: ignore[method-assign]
    coord._start_event_stream = lambda: None  # type: ignore[method-assign]
    coord.refresh_now = fake_refresh  # type: ignore[method-assign]

    yield rec


# ----------------------------------------------------------------------
# Route scoping
# ----------------------------------------------------------------------


class TestRouteScoping:
    def test_start_registers_overview_and_alerts_on_overview_route(
        self, harness: _CoordinatorHarness
    ) -> None:
        coord = harness.coordinator
        coord.start()
        # Overview route → overview job + always-on alerts; no others.
        assert set(harness.started) == {JOB_OVERVIEW, JOB_ALERTS}
        assert harness.stopped == []

    def test_route_change_stops_old_and_starts_new(self, harness: _CoordinatorHarness) -> None:
        coord = harness.coordinator
        store = coord._store  # exposed for the test
        coord.start()
        harness.started.clear()
        harness.stopped.clear()
        # Moving to SESSIONS route should stop JOB_OVERVIEW (scope
        # mismatch) and start JOB_SESSIONS. JOB_ALERTS stays (always on).
        store.set_route(AppRoute.SESSIONS)
        assert JOB_OVERVIEW in harness.stopped
        assert JOB_SESSIONS in harness.started
        assert JOB_ALERTS not in harness.stopped

    def test_live_session_requires_selected_session(self, harness: _CoordinatorHarness) -> None:
        coord = harness.coordinator
        store = coord._store
        coord.start()
        harness.started.clear()
        # Switching to LIVE_SESSION without a selected id should NOT
        # start the session-scoped jobs.
        store.set_route(AppRoute.LIVE_SESSION)
        assert JOB_LIVE_SESSION not in harness.started
        assert JOB_ENCOUNTERS not in harness.started

    def test_live_session_starts_health_job_without_selected_session(
        self, harness: _CoordinatorHarness
    ) -> None:
        coord = harness.coordinator
        store = coord._store
        coord.start()
        harness.started.clear()
        store.set_route(AppRoute.LIVE_SESSION)
        assert JOB_HEALTH in harness.started
        assert JOB_CLOUD_AUTH_STATUS not in harness.started
        assert JOB_CLOUD_OUTBOX not in harness.started
        assert JOB_EXPERIMENT_SUMMARIES in harness.started

    def test_health_route_starts_health_and_cloud_readback_jobs(
        self, harness: _CoordinatorHarness
    ) -> None:
        coord = harness.coordinator
        store = coord._store
        coord.start()
        harness.started.clear()
        store.set_route(AppRoute.HEALTH)
        assert JOB_HEALTH in harness.started
        assert JOB_CLOUD_AUTH_STATUS in harness.started
        assert JOB_CLOUD_OUTBOX in harness.started

    def test_selecting_session_starts_session_scoped_jobs(
        self, harness: _CoordinatorHarness
    ) -> None:
        coord = harness.coordinator
        store = coord._store
        coord.start()
        store.set_route(AppRoute.LIVE_SESSION)
        harness.started.clear()
        store.set_selected_session_id(uuid4())
        assert JOB_LIVE_SESSION in harness.started
        assert JOB_ENCOUNTERS in harness.started

    def test_selection_change_restarts_session_scoped_jobs(
        self, harness: _CoordinatorHarness
    ) -> None:
        coord = harness.coordinator
        store = coord._store
        coord.start()
        store.set_route(AppRoute.LIVE_SESSION)
        store.set_selected_session_id(uuid4())
        harness.started.clear()
        harness.stopped.clear()
        # Selecting a different session should stop the session-scoped
        # workers (their closures captured the old id) and start fresh.
        store.set_selected_session_id(uuid4())
        assert JOB_LIVE_SESSION in harness.stopped
        assert JOB_ENCOUNTERS in harness.stopped
        assert JOB_LIVE_SESSION in harness.started
        assert JOB_ENCOUNTERS in harness.started

    def test_leaving_route_stops_its_jobs(self, harness: _CoordinatorHarness) -> None:
        coord = harness.coordinator
        store = coord._store
        coord.start()
        store.set_route(AppRoute.HEALTH)
        harness.stopped.clear()
        store.set_route(AppRoute.OVERVIEW)
        assert JOB_HEALTH in harness.stopped

    def test_stop_tears_down_everything(self, harness: _CoordinatorHarness) -> None:
        coord = harness.coordinator
        coord.start()
        harness.stopped.clear()
        coord.stop()
        # Everything that was started must be stopped.
        assert set(harness.stopped) >= {JOB_OVERVIEW, JOB_ALERTS}


# ----------------------------------------------------------------------
# Payload dispatch — _apply_payload routes DTOs to the store
# ----------------------------------------------------------------------


class TestEventStreamCoordination:
    def test_event_dispatch_reuses_payload_path(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        store = OperatorStore()
        coord = PollingCoordinator(cfg, client, store)
        generated_at = _utc(2026, 5, 6, 12, 0)
        overview = OverviewSnapshot(
            generated_at_utc=generated_at,
            health=HealthSnapshot(generated_at_utc=generated_at, overall_state=HealthState.OK),
        )
        envelope = OperatorEventEnvelope(
            event_id="overview:1",
            event_type="overview",
            cursor="overview:1",
            generated_at_utc=generated_at,
            payload=overview,
        )

        coord._handle_event_stream_event(envelope)

        assert store.overview() is overview
        assert store.health() is overview.health
        assert coord._last_event_id == "overview:1"

    def test_stream_connected_pauses_unscoped_high_frequency_polling(
        self, harness: _CoordinatorHarness
    ) -> None:
        coord = harness.coordinator
        store = coord._store
        store.set_route(AppRoute.LIVE_SESSION)
        store.set_selected_session_id(uuid4())
        coord.start()
        harness.started.clear()
        harness.stopped.clear()

        coord._handle_event_stream_connected()

        assert JOB_LIVE_SESSION not in harness.stopped
        assert JOB_ENCOUNTERS not in harness.stopped
        assert JOB_HEALTH in harness.stopped
        assert JOB_ALERTS in harness.stopped
        assert JOB_EXPERIMENT_SUMMARIES not in harness.stopped

    def test_low_frequency_reconciliation_remains_active_while_stream_is_healthy(
        self, harness: _CoordinatorHarness
    ) -> None:
        coord = harness.coordinator
        store = coord._store
        store.set_route(AppRoute.LIVE_SESSION)
        store.set_selected_session_id(uuid4())
        coord.start()
        harness.stopped.clear()

        coord._handle_event_stream_connected()

        assert JOB_EXPERIMENT_SUMMARIES in coord._jobs
        assert JOB_EXPERIMENT_SUMMARIES not in harness.stopped

    def test_open_failure_does_not_mark_stream_connected_or_reset_backoff(
        self, harness: _CoordinatorHarness
    ) -> None:
        coord = harness.coordinator
        coord._sse_reconnect_ms = 4000
        coord._handle_event_stream_error(
            ApiError(message="open failed", endpoint=None, retryable=True)
        )

        assert coord._event_stream_connected is False
        assert coord._sse_reconnect_ms == 4000

    def test_stop_event_stream_calls_worker_stop_directly(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        coord = PollingCoordinator(cfg, client, OperatorStore())

        worker = _StubWorker()
        thread = _StubThread(running=False)
        coord._event_stream = EventStreamHandle(worker, thread)  # type: ignore[arg-type]

        coord._stop_event_stream()

        assert worker.stop_calls == 1
        assert coord._event_stream is None

    def test_event_stream_worker_stop_closes_active_response(self) -> None:
        class _Response:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

            def __iter__(self) -> Any:
                return iter(())

        class _Client:
            def __init__(self) -> None:
                self.response = _Response()

            def stream_events(
                self,
                *,
                last_event_id: str | None = None,
                on_open: Callable[[object], None] | None = None,
            ) -> Iterator[OperatorEventEnvelope]:
                del last_event_id
                if on_open is not None:
                    on_open(self.response)
                yield from ()

        client = _Client()
        worker = EventStreamWorker(client)  # type: ignore[arg-type]
        worker._handle_open(client.response)

        worker.stop()

        assert client.response.closed is True

    def test_live_session_empty_event_clears_current_session(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        store = OperatorStore()
        stale = SessionSummary(
            session_id=uuid4(),
            status="active",
            started_at_utc=_utc(2026, 5, 6, 12, 0),
        )
        store.set_live_session(stale)
        coord = PollingCoordinator(cfg, client, store)

        coord.apply_payload(JOB_LIVE_SESSION, [])

        assert store.live_session() is None

    def test_physiology_empty_event_clears_current_snapshot(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        store = OperatorStore()
        physiology = SessionPhysiologySnapshot(
            session_id=uuid4(),
            generated_at_utc=_utc(2026, 5, 6, 12, 0),
        )
        store.set_physiology(physiology)
        coord = PollingCoordinator(cfg, client, store)

        coord.apply_payload(JOB_PHYSIOLOGY, [])

        assert store.physiology() is None

    def test_event_payload_ignores_non_selected_session_state(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        selected_session_id = uuid4()
        active_session_id = uuid4()
        store = OperatorStore()
        store.set_selected_session_id(selected_session_id)
        selected_session = SessionSummary(
            session_id=selected_session_id,
            status="ended",
            started_at_utc=_utc(2026, 5, 6, 11, 0),
        )
        store.set_live_session(selected_session)
        selected_physiology = SessionPhysiologySnapshot(
            session_id=selected_session_id,
            generated_at_utc=_utc(2026, 5, 6, 11, 0),
        )
        store.set_physiology(selected_physiology)
        coord = PollingCoordinator(cfg, client, store)

        coord.apply_event_payload(
            JOB_LIVE_SESSION,
            SessionSummary(
                session_id=active_session_id,
                status="active",
                started_at_utc=_utc(2026, 5, 6, 12, 0),
            ),
        )
        coord.apply_event_payload(
            JOB_ENCOUNTERS,
            [
                EncounterSummary(
                    encounter_id="active-encounter",
                    session_id=active_session_id,
                    segment_timestamp_utc=_utc(2026, 5, 6, 12, 1),
                    state=EncounterState.COMPLETED,
                )
            ],
        )
        coord.apply_event_payload(
            JOB_PHYSIOLOGY,
            SessionPhysiologySnapshot(
                session_id=active_session_id,
                generated_at_utc=_utc(2026, 5, 6, 12, 0),
            ),
        )

        assert store.live_session() is selected_session
        assert store.encounters() == []
        assert store.physiology() is selected_physiology

    def test_event_experiment_does_not_change_managed_experiment(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        store = OperatorStore()
        store.set_managed_experiment_id("operator-managed")
        current = ExperimentDetail(experiment_id="operator-managed")
        store.set_experiment_readback(current)
        coord = PollingCoordinator(cfg, client, store)

        coord.apply_event_payload(JOB_EXPERIMENT, ExperimentDetail(experiment_id="active-default"))

        assert store.managed_experiment_id() == "operator-managed"
        assert store.experiment() is current

    def test_cloud_readback_payloads_update_store(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        store = OperatorStore()
        coord = PollingCoordinator(cfg, client, store)
        auth = CloudAuthStatus(
            state=CloudAuthState.SIGNED_IN,
            checked_at_utc=_utc(2026, 5, 6, 12, 0),
            message="Cloud sign-in is active.",
        )
        outbox = CloudOutboxSummary(
            generated_at_utc=_utc(2026, 5, 6, 12, 1),
            pending_count=2,
            in_flight_count=1,
            retry_scheduled_count=1,
            dead_letter_count=1,
            redacted_count=3,
        )

        coord.apply_payload(JOB_CLOUD_AUTH_STATUS, auth)
        coord.apply_payload(JOB_CLOUD_OUTBOX, outbox)

        assert store.cloud_auth_status() is auth
        assert store.cloud_outbox_summary() is outbox

    def test_event_experiment_requires_explicit_managed_experiment(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        store = OperatorStore()
        current = ExperimentDetail(experiment_id="default")
        store.set_experiment_readback(current)
        coord = PollingCoordinator(cfg, client, store)

        coord.apply_event_payload(JOB_EXPERIMENT, ExperimentDetail(experiment_id="active-default"))

        assert store.managed_experiment_id() is None
        assert store.experiment() is current
        assert store.error(JOB_EXPERIMENT) is None

    def test_event_experiment_empty_payload_is_noop(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        store = OperatorStore()
        store.set_managed_experiment_id("operator-managed")
        current = ExperimentDetail(experiment_id="operator-managed")
        store.set_experiment_readback(current)
        coord = PollingCoordinator(cfg, client, store)

        coord.apply_event_payload(JOB_EXPERIMENT, [])

        assert store.managed_experiment_id() == "operator-managed"
        assert store.experiment() is current
        assert store.error(JOB_EXPERIMENT) is None

    def test_polling_resumes_on_stream_failure(self, harness: _CoordinatorHarness) -> None:
        coord = harness.coordinator
        store = coord._store
        store.set_route(AppRoute.OVERVIEW)
        coord.start()
        coord._handle_event_stream_connected()
        harness.started.clear()

        coord._handle_event_stream_error(
            ApiError(message="stream broke", endpoint=None, retryable=True)
        )

        assert JOB_OVERVIEW in harness.started
        assert JOB_ALERTS in harness.started
        assert store.error(JOB_EVENT_STREAM) == "stream broke"


class TestPayloadDispatch:
    def test_overview_payload_populates_overview_and_sub_slots(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        store = OperatorStore()
        coord = PollingCoordinator(cfg, client, store)
        snap = OverviewSnapshot(
            generated_at_utc=_utc(2026, 4, 18, 10, 0),
            health=HealthSnapshot(
                generated_at_utc=_utc(2026, 4, 18, 10, 0),
                overall_state=HealthState.OK,
            ),
        )
        coord._handle_job_data(JOB_OVERVIEW, snap)
        assert store.overview() is snap
        assert store.health() is not None
        assert store.live_session() is None

    def test_overview_payload_clears_stale_live_session_when_no_active_session(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        store = OperatorStore()
        stale = SessionSummary(
            session_id=uuid4(),
            status="active",
            started_at_utc=_utc(2026, 4, 18, 9, 55),
        )
        store.set_live_session(stale)
        coord = PollingCoordinator(cfg, client, store)
        snap = OverviewSnapshot(generated_at_utc=_utc(2026, 4, 18, 10, 0))

        coord._handle_job_data(JOB_OVERVIEW, snap)

        assert store.overview() is snap
        assert store.live_session() is None

    def test_encounters_list_routes_to_encounters_slot(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        store = OperatorStore()
        coord = PollingCoordinator(cfg, client, store)
        sid = uuid4()
        rows = [
            EncounterSummary(
                encounter_id="e1",
                session_id=sid,
                segment_timestamp_utc=_utc(2026, 4, 18, 10, 5),
                state=EncounterState.COMPLETED,
            )
        ]
        coord._handle_job_data(JOB_ENCOUNTERS, rows)
        assert len(store.encounters()) == 1

    def test_unexpected_shape_sets_error_scope(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        store = OperatorStore()
        coord = PollingCoordinator(cfg, client, store)
        coord._handle_job_data(JOB_HEALTH, "not-a-health-snapshot")
        assert store.error(JOB_HEALTH) is not None
        assert "unexpected payload" in (store.error(JOB_HEALTH) or "")

    def test_data_arrival_clears_previous_error(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        store = OperatorStore()
        coord = PollingCoordinator(cfg, client, store)
        store.set_error(JOB_HEALTH, "earlier failure")
        snap = HealthSnapshot(
            generated_at_utc=_utc(2026, 4, 18, 10, 0),
            overall_state=HealthState.OK,
        )
        coord._handle_job_data(JOB_HEALTH, snap)
        assert store.error(JOB_HEALTH) is None


# ----------------------------------------------------------------------
# Error handling — ApiError surfaces through the store
# ----------------------------------------------------------------------


class TestErrorHandling:
    def test_api_error_populates_scoped_error_and_emits_job_failed(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        store = OperatorStore()
        coord = PollingCoordinator(cfg, client, store)
        emissions: list[tuple[str, str]] = []
        coord.job_failed.connect(lambda *args: emissions.append(tuple(args)))
        coord._handle_job_error(
            JOB_PHYSIOLOGY,
            ApiError(message="pool exhausted", retryable=True),
        )
        assert store.error(JOB_PHYSIOLOGY) == "pool exhausted"
        assert emissions == [(JOB_PHYSIOLOGY, "pool exhausted")]

    def test_non_apierror_still_recorded(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        store = OperatorStore()
        coord = PollingCoordinator(cfg, client, store)
        coord._handle_job_error(JOB_ALERTS, "something broke")
        assert store.error(JOB_ALERTS) == "something broke"


# ----------------------------------------------------------------------
# Stimulus one-shot — success fans out refreshes; failure surfaces
# ----------------------------------------------------------------------


class _FakeOneShot:
    """In-process replacement for `run_one_shot` that fires immediately."""

    def __init__(self, job_name: str, fn: Callable[[], object], mode: str) -> None:
        self.signals = OneShotSignals()
        # Connect the coordinator's slots first — the caller wires them
        # after this constructor returns. Fire after a short deferred
        # hop so the caller has time to connect.
        self._job_name = job_name
        self._fn = fn
        self._mode = mode

    def fire(self) -> None:
        if self._mode == "success":
            try:
                result = self._fn()
            except ApiError as exc:
                self.signals.failed.emit(self._job_name, exc)
            except Exception as exc:
                self.signals.failed.emit(self._job_name, ApiError(message=str(exc)))
            else:
                self.signals.succeeded.emit(self._job_name, result)
        elif self._mode == "failure":
            self.signals.failed.emit(self._job_name, ApiError(message="boom", retryable=True))
        self.signals.finished.emit(self._job_name)


class TestStimulusOneShot:
    def _patch_one_shot(self, mode: str) -> tuple[Any, list[_FakeOneShot]]:
        registry: list[_FakeOneShot] = []

        def fake_run_one_shot(job_name: str, fn: Callable[[], object]) -> OneShotSignals:
            inst = _FakeOneShot(job_name, fn, mode)
            registry.append(inst)
            return inst.signals

        return fake_run_one_shot, registry

    def test_success_triggers_refresh_fan_out(self, harness: _CoordinatorHarness) -> None:
        coord = harness.coordinator
        session_id = uuid4()
        action_id = uuid4()
        coord._store.set_selected_session_id(session_id)
        fake, registry = self._patch_one_shot("success")

        # Replace the client POST to avoid transport
        def fake_post(_sid: UUID, _req: StimulusRequest) -> StimulusAccepted:
            return StimulusAccepted(
                session_id=_sid,
                client_action_id=_req.client_action_id,
                accepted=True,
                received_at_utc=_utc(2026, 4, 18, 10, 2),
            )

        coord._client.post_stimulus = fake_post  # type: ignore[assignment,method-assign]

        with patch("services.operator_console.polling.run_one_shot", fake, create=False):
            req = StimulusRequest(client_action_id=action_id, operator_note="hi")
            coord.submit_stimulus(session_id, req)
            assert registry, "run_one_shot not invoked"
            registry[0].fire()

        # Success should have fanned out to overview/live/encounters/alerts
        assert JOB_OVERVIEW in harness.refresh_calls
        assert JOB_LIVE_SESSION in harness.refresh_calls
        assert JOB_ENCOUNTERS in harness.refresh_calls
        assert JOB_ALERTS in harness.refresh_calls
        # And cleared any stimulus error scope
        assert coord._store.error(JOB_STIMULUS) is None

    def test_failure_surfaces_error_and_job_failed(self, harness: _CoordinatorHarness) -> None:
        coord = harness.coordinator
        session_id = uuid4()
        action_id = uuid4()
        fake, registry = self._patch_one_shot("failure")
        emissions: list[tuple[str, str]] = []
        coord.job_failed.connect(lambda *args: emissions.append(tuple(args)))

        with patch("services.operator_console.polling.run_one_shot", fake, create=False):
            req = StimulusRequest(client_action_id=action_id)
            coord.submit_stimulus(session_id, req)
            registry[0].fire()

        assert coord._store.error(JOB_STIMULUS) == "boom"
        assert (JOB_STIMULUS, "boom") in emissions
        # No refreshes on failure
        assert JOB_OVERVIEW not in harness.refresh_calls


# ----------------------------------------------------------------------
# Repair one-shot — success refreshes health; failure surfaces
# ----------------------------------------------------------------------


class TestRepairOneShot:
    def _patch_one_shot(self, mode: str) -> tuple[Any, list[_FakeOneShot]]:
        registry: list[_FakeOneShot] = []

        def fake_run_one_shot(job_name: str, fn: Callable[[], object]) -> OneShotSignals:
            inst = _FakeOneShot(job_name, fn, mode)
            registry.append(inst)
            return inst.signals

        return fake_run_one_shot, registry

    def test_success_clears_error_and_refreshes_health_surfaces(
        self, harness: _CoordinatorHarness
    ) -> None:
        coord = harness.coordinator
        coord._store.set_error(JOB_REPAIR_INSTALL, "previous repair failure")
        fake, registry = self._patch_one_shot("success")

        with (
            patch("services.operator_console.polling.repair_runtime", lambda: "repaired"),
            patch("services.operator_console.polling.run_one_shot", fake, create=False),
        ):
            coord.repair_install()
            assert len(coord._inflight_repairs) == 1
            assert registry, "run_one_shot not invoked"
            registry[0].fire()

        assert coord._store.error(JOB_REPAIR_INSTALL) is None
        assert JOB_HEALTH in harness.refresh_calls
        assert JOB_ALERTS in harness.refresh_calls
        assert coord._inflight_repairs == {}

    def test_failure_surfaces_error_and_job_failed(self, harness: _CoordinatorHarness) -> None:
        coord = harness.coordinator
        fake, registry = self._patch_one_shot("failure")
        emissions: list[tuple[str, str]] = []
        coord.job_failed.connect(lambda *args: emissions.append(tuple(args)))

        with patch("services.operator_console.polling.run_one_shot", fake, create=False):
            coord.repair_install()
            registry[0].fire()

        assert coord._store.error(JOB_REPAIR_INSTALL) == "boom"
        assert (JOB_REPAIR_INSTALL, "boom") in emissions
        assert JOB_HEALTH not in harness.refresh_calls
        assert coord._inflight_repairs == {}


# ----------------------------------------------------------------------
# Cloud one-shots — success refreshes cloud-adjacent readbacks
# ----------------------------------------------------------------------


class TestCloudOneShot:
    def _patch_one_shot(self, mode: str) -> tuple[Any, list[_FakeOneShot]]:
        registry: list[_FakeOneShot] = []

        def fake_run_one_shot(job_name: str, fn: Callable[[], object]) -> OneShotSignals:
            inst = _FakeOneShot(job_name, fn, mode)
            registry.append(inst)
            return inst.signals

        return fake_run_one_shot, registry

    def test_cloud_sign_in_success_refreshes_health_and_experiment_readbacks(
        self, harness: _CoordinatorHarness
    ) -> None:
        coord = harness.coordinator
        coord._store.set_error(JOB_CLOUD_SIGN_IN, "previous sign-in failure")
        fake, registry = self._patch_one_shot("success")

        coord._client.post_cloud_sign_in = (  # type: ignore[method-assign]
            lambda: CloudSignInResult(
                status=CloudActionStatus.SUCCEEDED,
                auth_state=CloudAuthState.SIGNED_IN,
                completed_at_utc=_utc(2026, 5, 2, 12, 0),
                message="Cloud sign-in completed.",
            )
        )

        with patch("services.operator_console.polling.run_one_shot", fake, create=False):
            coord.cloud_sign_in()
            assert registry[0]._job_name == JOB_CLOUD_SIGN_IN
            assert len(coord._inflight_cloud_actions) == 1
            registry[0].fire()

        assert coord._store.error(JOB_CLOUD_SIGN_IN) is None
        assert JOB_EXPERIMENT in harness.refresh_calls
        assert JOB_EXPERIMENT_SUMMARIES in harness.refresh_calls
        assert JOB_HEALTH in harness.refresh_calls
        assert JOB_ALERTS in harness.refresh_calls
        assert coord._inflight_cloud_actions == {}

    def test_cloud_sign_in_failure_surfaces_error(self, harness: _CoordinatorHarness) -> None:
        coord = harness.coordinator
        fake, registry = self._patch_one_shot("failure")
        emissions: list[tuple[str, str]] = []
        coord.job_failed.connect(lambda *args: emissions.append(tuple(args)))

        with patch("services.operator_console.polling.run_one_shot", fake, create=False):
            coord.cloud_sign_in()
            registry[0].fire()

        assert coord._store.error(JOB_CLOUD_SIGN_IN) == "boom"
        assert (JOB_CLOUD_SIGN_IN, "boom") in emissions
        assert JOB_HEALTH not in harness.refresh_calls
        assert coord._inflight_cloud_actions == {}

    def test_cloud_sign_in_failed_dto_surfaces_error_without_refreshing(
        self, harness: _CoordinatorHarness
    ) -> None:
        coord = harness.coordinator
        fake, registry = self._patch_one_shot("success")
        emissions: list[tuple[str, str]] = []
        coord.job_failed.connect(lambda *args: emissions.append(tuple(args)))
        coord._client.post_cloud_sign_in = (  # type: ignore[method-assign]
            lambda: CloudSignInResult(
                status=CloudActionStatus.FAILED,
                auth_state=CloudAuthState.REFRESH_FAILED,
                completed_at_utc=_utc(2026, 5, 2, 12, 0),
                message="Cloud authorization failed.",
                retryable=True,
            )
        )

        with patch("services.operator_console.polling.run_one_shot", fake, create=False):
            coord.cloud_sign_in()
            registry[0].fire()

        assert coord._store.error(JOB_CLOUD_SIGN_IN) == "Cloud authorization failed."
        assert (JOB_CLOUD_SIGN_IN, "Cloud authorization failed.") in emissions
        assert JOB_HEALTH not in harness.refresh_calls
        assert coord._inflight_cloud_actions == {}

    def test_experiment_bundle_refresh_success_fetches_through_api_client(
        self, harness: _CoordinatorHarness
    ) -> None:
        coord = harness.coordinator
        coord._store.set_error(JOB_EXPERIMENT_BUNDLE_REFRESH, "previous refresh failure")
        fake, registry = self._patch_one_shot("success")
        emissions: list[tuple[str, str]] = []
        coord.job_failed.connect(lambda *args: emissions.append(tuple(args)))

        seen_requests: list[ExperimentBundleRefreshRequest] = []

        def refresh(
            request: ExperimentBundleRefreshRequest,
        ) -> ExperimentBundleRefreshResult:
            seen_requests.append(request)
            return ExperimentBundleRefreshResult(
                status=CloudExperimentRefreshStatus.APPLIED,
                completed_at_utc=_utc(2026, 5, 2, 12, 0),
                message="Experiment bundle refreshed.",
                bundle_id="bundle-a",
                experiment_count=1,
            )

        coord._client.post_experiment_bundle_refresh = refresh  # type: ignore[method-assign]

        request = ExperimentBundleRefreshRequest(preview_token="preview-token-a")
        with patch("services.operator_console.polling.run_one_shot", fake, create=False):
            coord.refresh_experiment_bundle(request)
            assert registry[0]._job_name == JOB_EXPERIMENT_BUNDLE_REFRESH
            assert len(coord._inflight_cloud_actions) == 1
            registry[0].fire()

        assert seen_requests == [request]
        assert coord._store.error(JOB_EXPERIMENT_BUNDLE_REFRESH) is None
        assert emissions == []
        assert harness.refresh_calls == [
            JOB_EXPERIMENT,
            JOB_EXPERIMENT_SUMMARIES,
            JOB_HEALTH,
            JOB_ALERTS,
        ]
        assert coord._inflight_cloud_actions == {}

    def test_experiment_bundle_preview_success_does_not_refresh_readbacks(
        self, harness: _CoordinatorHarness
    ) -> None:
        coord = harness.coordinator
        coord._store.set_error(JOB_EXPERIMENT_BUNDLE_REFRESH, "previous preview failure")
        fake, registry = self._patch_one_shot("success")
        coord._client.post_experiment_bundle_refresh_preview = (  # type: ignore[method-assign]
            lambda: ExperimentBundleRefreshPreview(
                status=CloudActionStatus.SUCCEEDED,
                checked_at_utc=_utc(2026, 5, 2, 12, 0),
                message="Preview ready.",
                preview_token="preview-token-a",
                bundle_id="bundle-a",
                experiment_count=1,
                added_count=1,
            )
        )

        with patch("services.operator_console.polling.run_one_shot", fake, create=False):
            coord.preview_experiment_bundle_refresh()
            assert registry[0]._job_name == JOB_EXPERIMENT_BUNDLE_REFRESH
            assert len(coord._inflight_cloud_actions) == 1
            registry[0].fire()

        assert coord._store.error(JOB_EXPERIMENT_BUNDLE_REFRESH) is None
        assert harness.refresh_calls == []
        assert coord._inflight_cloud_actions == {}

    def test_experiment_bundle_refresh_failed_dto_surfaces_error_without_refreshing(
        self, harness: _CoordinatorHarness
    ) -> None:
        coord = harness.coordinator
        fake, registry = self._patch_one_shot("success")
        emissions: list[tuple[str, str]] = []
        coord.job_failed.connect(lambda *args: emissions.append(tuple(args)))

        def refresh(
            request: ExperimentBundleRefreshRequest,
        ) -> ExperimentBundleRefreshResult:
            del request
            return ExperimentBundleRefreshResult(
                status=CloudExperimentRefreshStatus.FAILED,
                completed_at_utc=_utc(2026, 5, 2, 12, 0),
                message="Cloud experiment service is offline.",
                retryable=True,
            )

        coord._client.post_experiment_bundle_refresh = refresh  # type: ignore[method-assign]

        request = ExperimentBundleRefreshRequest(preview_token="preview-token-a")
        with patch("services.operator_console.polling.run_one_shot", fake, create=False):
            coord.refresh_experiment_bundle(request)
            registry[0].fire()

        assert (
            coord._store.error(JOB_EXPERIMENT_BUNDLE_REFRESH)
            == "Cloud experiment service is offline."
        )
        assert (
            JOB_EXPERIMENT_BUNDLE_REFRESH,
            "Cloud experiment service is offline.",
        ) in emissions
        assert harness.refresh_calls == []
        assert coord._inflight_cloud_actions == {}


# ----------------------------------------------------------------------
# Experiment management one-shots — success updates store and refreshes
# ----------------------------------------------------------------------


class TestExperimentManagementOneShot:
    def _patch_one_shot(self, mode: str) -> tuple[Any, list[_FakeOneShot]]:
        registry: list[_FakeOneShot] = []

        def fake_run_one_shot(job_name: str, fn: Callable[[], object]) -> OneShotSignals:
            inst = _FakeOneShot(job_name, fn, mode)
            registry.append(inst)
            return inst.signals

        return fake_run_one_shot, registry

    def test_create_experiment_updates_store_and_refreshes(
        self, harness: _CoordinatorHarness
    ) -> None:
        coord = harness.coordinator
        fake, registry = self._patch_one_shot("success")

        def fake_create(request: ExperimentCreateRequest) -> ExperimentAdminResponse:
            assert request.experiment_id == "exp-new"
            return ExperimentAdminResponse(
                experiment_id="exp-new",
                label="Greeting v2",
                arms=[
                    ExperimentArmAdminResponse(
                        experiment_id="exp-new",
                        label="Greeting v2",
                        arm="arm-a",
                        greeting_text="Hei",
                        alpha_param=1.0,
                        beta_param=1.0,
                        enabled=True,
                    )
                ],
            )

        coord._client.create_experiment = fake_create  # type: ignore[method-assign]
        request = ExperimentCreateRequest(
            experiment_id="exp-new",
            label="Greeting v2",
            arms=[ExperimentArmSeedRequest(arm="arm-a", greeting_text="Hei")],
        )
        with patch("services.operator_console.polling.run_one_shot", fake, create=False):
            coord.create_experiment(request)
            registry[0].fire()

        detail = coord._store.experiment()
        assert detail is not None
        assert detail.experiment_id == "exp-new"
        assert detail.arms[0].posterior_alpha == 1.0
        assert coord._store.managed_experiment_id() == "exp-new"
        assert JOB_EXPERIMENT in harness.refresh_calls

    def test_rename_arm_updates_store_readback(self, harness: _CoordinatorHarness) -> None:
        coord = harness.coordinator
        coord._store.set_experiment(
            ExperimentDetail(
                experiment_id="exp1",
                arms=[
                    ArmSummary(
                        arm_id="a1",
                        greeting_text="old",
                        posterior_alpha=2.0,
                        posterior_beta=3.0,
                        selection_count=9,
                    )
                ],
            )
        )
        fake, registry = self._patch_one_shot("success")

        def fake_patch(
            experiment_id: str,
            arm_id: str,
            request: ExperimentArmPatchRequest,
        ) -> ExperimentArmAdminResponse:
            assert (experiment_id, arm_id) == ("exp1", "a1")
            assert request.greeting_text == "new"
            assert request.enabled is None
            return ExperimentArmAdminResponse(
                experiment_id="exp1",
                label="exp1",
                arm="a1",
                greeting_text="new",
                alpha_param=2.0,
                beta_param=3.0,
                enabled=True,
            )

        coord._client.patch_experiment_arm = fake_patch  # type: ignore[method-assign]
        with patch("services.operator_console.polling.run_one_shot", fake, create=False):
            coord.rename_experiment_arm("exp1", "a1", "new")
            registry[0].fire()

        detail = coord._store.experiment()
        assert detail is not None
        assert detail.arms[0].greeting_text == "new"
        assert detail.arms[0].selection_count == 9

    def test_disable_arm_uses_allowed_enabled_false_patch(
        self, harness: _CoordinatorHarness
    ) -> None:
        coord = harness.coordinator
        coord._store.set_experiment(
            ExperimentDetail(
                experiment_id="exp1",
                arms=[
                    ArmSummary(
                        arm_id="a1",
                        greeting_text="old",
                        posterior_alpha=2.0,
                        posterior_beta=3.0,
                    )
                ],
            )
        )
        fake, registry = self._patch_one_shot("success")

        def fake_patch(
            experiment_id: str,
            arm_id: str,
            request: ExperimentArmPatchRequest,
        ) -> ExperimentArmAdminResponse:
            assert (experiment_id, arm_id) == ("exp1", "a1")
            assert request.enabled is False
            assert request.greeting_text is None
            return ExperimentArmAdminResponse(
                experiment_id="exp1",
                label="exp1",
                arm="a1",
                greeting_text="old",
                alpha_param=2.0,
                beta_param=3.0,
                enabled=False,
            )

        coord._client.patch_experiment_arm = fake_patch  # type: ignore[method-assign]
        with patch("services.operator_console.polling.run_one_shot", fake, create=False):
            coord.disable_experiment_arm("exp1", "a1")
            registry[0].fire()

        detail = coord._store.experiment()
        assert detail is not None
        assert detail.arms[0].enabled is False

    def test_delete_arm_removes_unused_arm_and_pins_managed_experiment(
        self, harness: _CoordinatorHarness
    ) -> None:
        coord = harness.coordinator
        coord._store.set_experiment(
            ExperimentDetail(
                experiment_id="exp-non-default",
                arms=[
                    ArmSummary(
                        arm_id="unused",
                        greeting_text="old",
                        posterior_alpha=1.0,
                        posterior_beta=1.0,
                    ),
                    ArmSummary(
                        arm_id="kept",
                        greeting_text="hi",
                        posterior_alpha=2.0,
                        posterior_beta=3.0,
                    ),
                ],
            )
        )
        fake, registry = self._patch_one_shot("success")

        def fake_delete(experiment_id: str, arm_id: str) -> ExperimentArmDeleteResponse:
            assert (experiment_id, arm_id) == ("exp-non-default", "unused")
            return ExperimentArmDeleteResponse(
                experiment_id="exp-non-default",
                arm="unused",
                deleted=True,
                posterior_preserved=False,
                reason="unused arm hard-deleted",
            )

        coord._client.delete_experiment_arm = fake_delete  # type: ignore[method-assign]
        with patch("services.operator_console.polling.run_one_shot", fake, create=False):
            coord.delete_experiment_arm("exp-non-default", "unused")
            registry[0].fire()

        detail = coord._store.experiment()
        assert detail is not None
        assert detail.experiment_id == "exp-non-default"
        assert [arm.arm_id for arm in detail.arms] == ["kept"]
        assert coord._store.managed_experiment_id() == "exp-non-default"
        assert JOB_EXPERIMENT in harness.refresh_calls


# ----------------------------------------------------------------------
# Fetch factories — make sure they bind the right ApiClient method
# ----------------------------------------------------------------------


class TestSessionLifecycleOneShot:
    def _patch_one_shot(self, mode: str) -> tuple[Any, list[_FakeOneShot]]:
        registry: list[_FakeOneShot] = []

        def fake_run_one_shot(job_name: str, fn: Callable[[], object]) -> OneShotSignals:
            inst = _FakeOneShot(job_name, fn, mode)
            registry.append(inst)
            return inst.signals

        return fake_run_one_shot, registry

    def test_session_start_success_selects_session_before_live_refresh(
        self,
        harness: _CoordinatorHarness,
    ) -> None:
        coord = harness.coordinator
        coord._store.set_route(AppRoute.LIVE_SESSION)
        coord.start()
        harness.started.clear()
        harness.refresh_calls.clear()
        fake, registry = self._patch_one_shot("success")

        session_id = uuid4()

        def fake_post(request: SessionCreateRequest) -> SessionLifecycleAccepted:
            return SessionLifecycleAccepted(
                action="start",
                session_id=session_id,
                client_action_id=request.client_action_id,
                accepted=True,
                received_at_utc=_utc(2026, 4, 18, 10, 3),
            )

        coord._client.post_session_start = fake_post  # type: ignore[method-assign]

        with patch("services.operator_console.polling.run_one_shot", fake, create=False):
            req = SessionCreateRequest(
                stream_url="rtmp://example/live",
                experiment_id="greeting_line_v1",
                client_action_id=uuid4(),
            )
            coord.request_session_start(req)
            registry[0].fire()

        assert coord._store.selected_session_id() == session_id
        assert JOB_LIVE_SESSION in harness.started
        assert JOB_ENCOUNTERS in harness.started
        assert JOB_OVERVIEW in harness.refresh_calls
        assert JOB_SESSIONS in harness.refresh_calls
        assert JOB_LIVE_SESSION in harness.refresh_calls
        assert JOB_ENCOUNTERS in harness.refresh_calls
        assert JOB_ALERTS in harness.refresh_calls
        assert coord._store.error(JOB_SESSION_START) is None

    def test_fast_session_start_signal_reaches_late_connected_vm_slot(
        self,
        qt_app: QCoreApplication,
    ) -> None:
        session_id = uuid4()
        request = SessionCreateRequest(
            stream_url="rtmp://example/live",
            experiment_id="greeting_line_v1",
            client_action_id=uuid4(),
        )

        signals = run_one_shot(
            JOB_SESSION_START,
            lambda: SessionLifecycleAccepted(
                action="start",
                session_id=session_id,
                client_action_id=request.client_action_id,
                accepted=True,
                received_at_utc=_utc(2026, 4, 18, 10, 3),
            ),
        )
        payloads: list[SessionLifecycleAccepted] = []
        signals.succeeded.connect(lambda _job, payload: payloads.append(payload))

        _drain_events_until(qt_app, lambda: bool(payloads))

        assert payloads and payloads[0].session_id == session_id

    def test_manual_start_reaches_ready_viewmodel_state(
        self,
        cfg: OperatorConsoleConfig,
        qt_app: QCoreApplication,
    ) -> None:
        store = OperatorStore()
        store.set_route(AppRoute.LIVE_SESSION)
        model = EncountersTableModel()
        vm = LiveSessionViewModel(store, model)
        session_id = uuid4()

        class _Client:
            def post_session_start(
                self,
                request: SessionCreateRequest,
            ) -> SessionLifecycleAccepted:
                return SessionLifecycleAccepted(
                    action="start",
                    session_id=session_id,
                    client_action_id=request.client_action_id,
                    accepted=True,
                    received_at_utc=_utc(2026, 4, 18, 10, 3),
                )

            def get_session(self, requested_session_id: UUID) -> SessionSummary:
                assert requested_session_id == session_id
                return SessionSummary(
                    session_id=session_id,
                    status="active",
                    started_at_utc=_utc(2026, 4, 18, 10, 3),
                    experiment_id="greeting_line_v1",
                )

            def list_session_encounters(
                self,
                requested_session_id: UUID,
                *,
                limit: int = 100,
                before_utc: datetime | None = None,
            ) -> list[EncounterSummary]:
                del limit, before_utc
                assert requested_session_id == session_id
                return []

            def list_experiments(self) -> list[ExperimentSummary]:
                return [ExperimentSummary(experiment_id="greeting_line_v1", arm_count=1)]

        coord = PollingCoordinator(cfg, _Client(), store)  # type: ignore[arg-type]
        coord._start_event_stream = lambda: None  # type: ignore[method-assign]
        vm.bind_session_lifecycle_actions(coord.request_session_start, coord.request_session_end)
        coord.start()

        action_id = vm.start_new_session("greeting_line_v1")
        _drain_events_until(qt_app, lambda: vm.ttv_state() == "READY")
        coord.stop()

        assert action_id is not None
        assert store.selected_session_id() == session_id
        assert vm.session() is not None
        assert vm.ttv_state() == "READY"
        assert vm.session_start_in_progress() is False

    def test_one_shot_success_is_delivered_on_main_qt_thread(
        self,
        qt_app: QCoreApplication,
    ) -> None:
        main_thread = qt_app.thread()
        delivered_threads: list[QThread] = []

        signals = run_one_shot(JOB_SESSION_START, lambda: "ok")
        signals.succeeded.connect(
            lambda _job, _payload: delivered_threads.append(QThread.currentThread())
        )

        _drain_events_until(qt_app, lambda: bool(delivered_threads))

        assert delivered_threads == [main_thread]

    def test_session_end_success_refreshes_live_session_surfaces(
        self,
        harness: _CoordinatorHarness,
    ) -> None:
        coord = harness.coordinator
        fake, registry = self._patch_one_shot("success")
        session_id = uuid4()

        def fake_post(_session_id: UUID, request: SessionEndRequest) -> SessionLifecycleAccepted:
            return SessionLifecycleAccepted(
                action="end",
                session_id=_session_id,
                client_action_id=request.client_action_id,
                accepted=True,
                received_at_utc=_utc(2026, 4, 18, 10, 4),
            )

        coord._client.post_session_end = fake_post  # type: ignore[assignment,method-assign]

        with patch("services.operator_console.polling.run_one_shot", fake, create=False):
            req = SessionEndRequest(client_action_id=uuid4())
            coord.request_session_end(session_id, req)
            registry[0].fire()

        assert JOB_OVERVIEW in harness.refresh_calls
        assert JOB_SESSIONS in harness.refresh_calls
        assert JOB_LIVE_SESSION in harness.refresh_calls
        assert JOB_ENCOUNTERS in harness.refresh_calls
        assert JOB_HEALTH in harness.refresh_calls
        assert JOB_ALERTS in harness.refresh_calls
        assert coord._store.error(JOB_SESSION_END) is None

    def test_session_lifecycle_failure_surfaces_scoped_error(
        self,
        harness: _CoordinatorHarness,
    ) -> None:
        coord = harness.coordinator
        fake, registry = self._patch_one_shot("failure")
        emissions: list[tuple[str, str]] = []
        coord.job_failed.connect(lambda *args: emissions.append(tuple(args)))

        with patch("services.operator_console.polling.run_one_shot", fake, create=False):
            req = SessionCreateRequest(
                stream_url="rtmp://example/live",
                experiment_id="greeting_line_v1",
                client_action_id=uuid4(),
            )
            coord.request_session_start(req)
            registry[0].fire()

        assert coord._store.error(JOB_SESSION_START) == "boom"
        assert (JOB_SESSION_START, "boom") in emissions


class TestFetchFactories:
    def test_overview_fetch_calls_client_get_overview(self, cfg: OperatorConsoleConfig) -> None:
        calls: list[str] = []

        class _SpyClient:
            def get_overview(self) -> OverviewSnapshot:
                calls.append("get_overview")
                return OverviewSnapshot(generated_at_utc=_utc(2026, 4, 18, 10, 0))

        store = OperatorStore()
        coord = PollingCoordinator(cfg, _SpyClient(), store)  # type: ignore[arg-type]
        fetch = coord._make_fetch_overview()
        fetch()
        assert calls == ["get_overview"]

    def test_encounters_fetch_captures_selected_session(self, cfg: OperatorConsoleConfig) -> None:
        called_with: list[UUID] = []

        class _SpyClient:
            def list_session_encounters(
                self, sid: UUID, *, limit: int = 100, before_utc: datetime | None = None
            ) -> list[EncounterSummary]:
                del limit, before_utc
                called_with.append(sid)
                return []

        store = OperatorStore()
        sid = uuid4()
        store.set_selected_session_id(sid)
        coord = PollingCoordinator(cfg, _SpyClient(), store)  # type: ignore[arg-type]
        fetch = coord._make_fetch_encounters()
        fetch()
        assert called_with == [sid]

    def test_encounters_fetch_raises_without_selection(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        store = OperatorStore()
        coord = PollingCoordinator(cfg, client, store)
        with pytest.raises(RuntimeError, match="selected session"):
            coord._make_fetch_encounters()

    def test_experiment_fetch_uses_config_default_without_managed_id(
        self, cfg: OperatorConsoleConfig
    ) -> None:
        calls: list[str] = []

        class _SpyClient:
            def get_experiment_detail(self, eid: str) -> Any:
                calls.append(eid)

        store = OperatorStore()
        coord = PollingCoordinator(cfg, _SpyClient(), store)  # type: ignore[arg-type]
        fetch = coord._make_fetch_experiment()
        fetch()
        assert calls == [cfg.default_experiment_id]

    def test_experiment_fetch_reads_current_store_managed_id(
        self, cfg: OperatorConsoleConfig
    ) -> None:
        calls: list[str] = []

        class _SpyClient:
            def get_experiment_detail(self, eid: str) -> Any:
                calls.append(eid)

        store = OperatorStore()
        coord = PollingCoordinator(cfg, _SpyClient(), store)  # type: ignore[arg-type]
        fetch = coord._make_fetch_experiment()
        store.set_managed_experiment_id("exp-non-default")
        fetch()
        assert calls == ["exp-non-default"]


# ----------------------------------------------------------------------
# PollingWorker shutdown edge cases
# ----------------------------------------------------------------------


class TestPollingWorkerShutdown:
    def test_stop_before_run_emits_stopped_once(self) -> None:
        worker = PollingWorker(JOB_HEALTH, 1000, lambda: object())
        stopped: list[str] = []
        worker.stopped.connect(stopped.append)

        worker.stop()
        worker.stop()

        assert stopped == [JOB_HEALTH]

    def test_stop_before_queued_first_fetch_prevents_timer_creation(
        self, qt_app: QCoreApplication
    ) -> None:
        fetch_count = 0

        def fetch() -> object:
            nonlocal fetch_count
            fetch_count += 1
            return object()

        worker = PollingWorker(JOB_HEALTH, 1000, fetch)
        stopped: list[str] = []
        payloads: list[object] = []
        worker.stopped.connect(stopped.append)
        worker.data_ready.connect(lambda _job, payload: payloads.append(payload))

        worker.run()
        worker.stop()
        qt_app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 25)

        assert stopped == [JOB_HEALTH]
        assert fetch_count == 0
        assert payloads == []
        assert worker._timer is None

    def test_refresh_after_stop_does_not_fetch(self) -> None:
        fetch_count = 0

        def fetch() -> object:
            nonlocal fetch_count
            fetch_count += 1
            return object()

        worker = PollingWorker(JOB_HEALTH, 1000, fetch)
        worker.stop()
        worker.refresh_now()

        assert fetch_count == 0

    def test_overlapping_fetch_tick_is_skipped(self) -> None:
        fetch_count = 0

        def fetch() -> object:
            nonlocal fetch_count
            fetch_count += 1
            worker._run_once()
            return object()

        worker = PollingWorker(JOB_HEALTH, 1000, fetch)
        worker._run_once()

        assert fetch_count == 1


# ----------------------------------------------------------------------
# SessionSummary round-trip through live_session job
# ----------------------------------------------------------------------


class TestSessionDispatch:
    def test_live_session_payload_routes_to_live_session_slot(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        store = OperatorStore()
        coord = PollingCoordinator(cfg, client, store)
        summary = SessionSummary(
            session_id=uuid4(),
            status="live",
            started_at_utc=_utc(2026, 4, 18, 10, 0),
        )
        coord._handle_job_data(JOB_LIVE_SESSION, summary)
        assert store.live_session() is summary


# ----------------------------------------------------------------------
# Non-blocking teardown — regression for the route-change UI freeze
# ----------------------------------------------------------------------


@dataclass
class _StubThread:
    """Just-enough QThread stand-in for the orphan-list bookkeeping."""

    running: bool = True
    quit_called: int = 0
    wait_calls: list[int] = field(default_factory=list)
    terminate_called: int = 0

    def isRunning(self) -> bool:  # noqa: N802 — Qt API mimic
        return self.running

    def quit(self) -> None:
        self.quit_called += 1

    def wait(self, ms: int) -> bool:
        self.wait_calls.append(ms)
        # Default behaviour: nothing actually finishes during the
        # graceful wait — tests can flip `running` themselves to
        # simulate a clean exit.
        return not self.running

    def terminate(self) -> None:
        self.terminate_called += 1
        self.running = False  # post-terminate the thread is dead


class _StubSignal:
    def __init__(self) -> None:
        self.disconnected: list[object] = []

    def disconnect(self, slot: object) -> None:
        self.disconnected.append(slot)


class _StubWorker:
    """Stand-in worker exposing only what `_stop_job` / `refresh_now`
    poke at via QMetaObject.invokeMethod (which is patched out)."""

    def __init__(self) -> None:
        self.connected = _StubSignal()
        self.event_ready = _StubSignal()
        self.data_ready = _StubSignal()
        self.error = _StubSignal()
        self.stopped = _StubSignal()
        self.refresh_calls = 0
        self.stop_calls = 0

    def refresh_now(self) -> None:
        self.refresh_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


def _install_orphan_handle(
    coord: PollingCoordinator,
    job_name: str,
    *,
    running: bool = True,
) -> tuple[_StubWorker, _StubThread]:
    """Drop a stub job into `_jobs` so `_stop_job` can park it."""
    spec = coord._specs[job_name]
    worker = _StubWorker()
    thread = _StubThread(running=running)
    coord._jobs[job_name] = _JobHandle(spec, worker, thread)  # type: ignore[arg-type]
    return worker, thread


# Late import so the dataclass type is in scope for the helper above.
from services.operator_console.polling import _JobHandle  # noqa: E402


class TestNonBlockingTeardown:
    """Route-change teardown must not freeze the UI thread."""

    def test_stop_job_queues_worker_stop_and_does_not_quit_thread(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        # The previous implementation called `handle.thread.quit()`
        # synchronously from the main thread, which raced the queued
        # stop slot. Verify we no longer do that — `quit` only fires
        # via the `worker.stopped → thread.quit` connection wired in
        # `_start_job`.
        coord = PollingCoordinator(cfg, client, OperatorStore())
        worker, thread = _install_orphan_handle(coord, JOB_OVERVIEW)

        invocations: list[tuple[object, str, Qt.ConnectionType]] = []

        def fake_invoke(obj: object, method: str, conn: Qt.ConnectionType) -> bool:
            invocations.append((obj, method, conn))
            return True

        with patch(
            "services.operator_console.polling.QMetaObject.invokeMethod",
            side_effect=fake_invoke,
        ):
            coord._stop_job(JOB_OVERVIEW)

        assert worker.stop_calls == 0, "stop must be queued, not called inline"
        assert worker.data_ready.disconnected == [coord._handle_job_data]
        assert worker.error.disconnected == [coord._handle_job_error]
        assert thread.quit_called == 0, "thread.quit must come from the signal chain"
        assert invocations == [
            (worker, "stop", Qt.ConnectionType.QueuedConnection),
        ]
        assert JOB_OVERVIEW not in coord._jobs
        # Still-running thread parks on the orphan list for shutdown drain.
        assert len(coord._orphan_jobs) == 1
        assert coord._orphan_jobs[0].thread is thread  # type: ignore[comparison-overlap]

    def test_thread_finished_prunes_orphan_handle(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        coord = PollingCoordinator(cfg, client, OperatorStore())
        worker = _StubWorker()
        thread = _StubThread(running=False)
        coord._orphan_jobs.append(
            _JobHandle(coord._specs[JOB_HEALTH], worker, thread)  # type: ignore[arg-type]
        )

        coord._prune_orphan(worker)  # type: ignore[arg-type]

        assert coord._orphan_jobs == []

    def test_route_sync_prunes_completed_orphan_handles(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        coord = PollingCoordinator(cfg, client, OperatorStore())
        completed = _JobHandle(coord._specs[JOB_HEALTH], _StubWorker(), _StubThread(False))  # type: ignore[arg-type]
        running = _JobHandle(coord._specs[JOB_ALERTS], _StubWorker(), _StubThread(True))  # type: ignore[arg-type]
        coord._orphan_jobs.extend([completed, running])
        coord._start_job = lambda _spec: None  # type: ignore[assignment,method-assign]
        coord._stop_job = lambda _job_name: None  # type: ignore[assignment,method-assign]

        coord._sync_jobs_for_current_state()

        assert completed not in coord._orphan_jobs
        assert running in coord._orphan_jobs

    def test_live_session_to_overview_parks_stopped_jobs_as_orphans(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        store = OperatorStore()
        store.set_route(AppRoute.LIVE_SESSION)
        session_id = uuid4()
        store.set_selected_session_id(session_id)
        coord = PollingCoordinator(cfg, client, store)

        stopped: list[str] = []
        started: list[str] = []

        def fake_stop(job_name: str) -> None:
            stopped.append(job_name)
            coord._jobs.pop(job_name, None)
            worker = _StubWorker()
            thread = _StubThread(running=True)
            coord._orphan_jobs.append(
                _JobHandle(coord._specs[job_name], worker, thread)  # type: ignore[arg-type]
            )

        def fake_start(spec: PollJobSpec) -> None:
            started.append(spec.name)
            worker = _StubWorker()
            thread = _StubThread(running=True)
            coord._jobs[spec.name] = _JobHandle(spec, worker, thread)  # type: ignore[arg-type]

        coord._stop_job = fake_stop  # type: ignore[method-assign]
        coord._start_job = fake_start  # type: ignore[method-assign]
        coord._start_event_stream = lambda: None  # type: ignore[method-assign]
        coord.start()
        started.clear()

        store.set_route(AppRoute.OVERVIEW)

        assert set(stopped) == {
            JOB_LIVE_SESSION,
            JOB_ENCOUNTERS,
            JOB_EXPERIMENT_SUMMARIES,
            JOB_HEALTH,
        }
        assert started == [JOB_OVERVIEW]
        assert set(coord._jobs) == {JOB_ALERTS, JOB_OVERVIEW}
        assert {handle.spec.name for handle in coord._orphan_jobs} == {
            JOB_LIVE_SESSION,
            JOB_ENCOUNTERS,
            JOB_EXPERIMENT_SUMMARIES,
            JOB_HEALTH,
        }

    def test_stop_event_stream_parks_running_thread_as_orphan(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        coord = PollingCoordinator(cfg, client, OperatorStore())
        worker = _StubWorker()
        thread = _StubThread(running=True)
        handle = EventStreamHandle(worker, thread)  # type: ignore[arg-type]
        coord._event_stream = handle

        coord._stop_event_stream()

        assert worker.stop_calls == 1
        assert coord._event_stream is None
        assert coord._orphan_event_streams == [handle]

    def test_drain_orphan_event_streams_retains_connect_phase_thread(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        coord = PollingCoordinator(cfg, client, OperatorStore())
        thread = _StubThread(running=True)
        handle = EventStreamHandle(_StubWorker(), thread)  # type: ignore[arg-type]
        coord._orphan_event_streams.append(handle)

        coord._drain_orphan_event_streams()

        assert thread.wait_calls
        assert coord._orphan_event_streams == [handle]

    def test_drain_orphan_event_streams_prunes_clean_exit(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        coord = PollingCoordinator(cfg, client, OperatorStore())
        handle = EventStreamHandle(_StubWorker(), _StubThread(running=False))  # type: ignore[arg-type]
        coord._orphan_event_streams.append(handle)

        coord._drain_orphan_event_streams()

        assert coord._orphan_event_streams == []

    def test_final_stop_drains_event_stream_orphans(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        coord = PollingCoordinator(cfg, client, OperatorStore())
        worker = _StubWorker()
        thread = _StubThread(running=True)
        coord._event_stream = EventStreamHandle(worker, thread)  # type: ignore[arg-type]
        coord._started = True
        coord._drain_orphan_jobs = lambda: None  # type: ignore[method-assign]

        coord.stop()

        assert worker.stop_calls == 1
        assert thread.wait_calls
        assert len(coord._orphan_event_streams) == 1

    def test_drain_orphan_jobs_retains_stuck_threads_without_terminating(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        coord = PollingCoordinator(cfg, client, OperatorStore())

        stuck = _StubThread(running=True)
        handle = _JobHandle(coord._specs[JOB_OVERVIEW], _StubWorker(), stuck)  # type: ignore[arg-type]
        coord._orphan_jobs.append(handle)

        coord._drain_orphan_jobs()

        assert stuck.wait_calls
        assert stuck.terminate_called == 0
        assert coord._orphan_jobs == [handle]

    def test_drain_orphan_jobs_skips_terminate_for_clean_exits(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        coord = PollingCoordinator(cfg, client, OperatorStore())

        clean = _StubThread(running=False)  # already done
        coord._orphan_jobs.append(
            _JobHandle(coord._specs[JOB_OVERVIEW], _StubWorker(), clean)  # type: ignore[arg-type]
        )

        coord._drain_orphan_jobs()

        assert clean.terminate_called == 0
        assert coord._orphan_jobs == []

    def test_refresh_now_uses_queued_invocation(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        coord = PollingCoordinator(cfg, client, OperatorStore())
        worker, _thread = _install_orphan_handle(coord, JOB_OVERVIEW)

        invocations: list[tuple[object, str, Qt.ConnectionType]] = []

        def fake_invoke(obj: object, method: str, conn: Qt.ConnectionType) -> bool:
            invocations.append((obj, method, conn))
            return True

        with patch(
            "services.operator_console.polling.QMetaObject.invokeMethod",
            side_effect=fake_invoke,
        ):
            coord.refresh_now(JOB_OVERVIEW)

        assert worker.refresh_calls == 0, "refresh_now must not call the worker inline"
        assert invocations == [
            (worker, "refresh_now", Qt.ConnectionType.QueuedConnection),
        ]
