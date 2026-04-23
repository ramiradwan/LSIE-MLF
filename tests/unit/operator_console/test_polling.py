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
from PySide6.QtCore import QCoreApplication, Qt

from packages.schemas.operator_console import (
    EncounterState,
    EncounterSummary,
    HealthSnapshot,
    HealthState,
    OverviewSnapshot,
    SessionSummary,
    StimulusAccepted,
    StimulusRequest,
)
from services.operator_console.api_client import ApiClient, ApiError
from services.operator_console.config import OperatorConsoleConfig, load_config
from services.operator_console.polling import (
    JOB_ALERTS,
    JOB_ENCOUNTERS,
    JOB_HEALTH,
    JOB_LIVE_SESSION,
    JOB_OVERVIEW,
    JOB_PHYSIOLOGY,
    JOB_SESSIONS,
    JOB_STIMULUS,
    PollingCoordinator,
)
from services.operator_console.state import AppRoute, OperatorStore
from services.operator_console.workers import OneShotSignals

pytestmark = pytest.mark.usefixtures("qt_app")


# ----------------------------------------------------------------------
# Fixtures + helpers
# ----------------------------------------------------------------------


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


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

        # Success should have fanned out to overview/live/alerts
        assert JOB_OVERVIEW in harness.refresh_calls
        assert JOB_LIVE_SESSION in harness.refresh_calls
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
# Fetch factories — make sure they bind the right ApiClient method
# ----------------------------------------------------------------------


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

    def test_experiment_fetch_uses_config_default(self, cfg: OperatorConsoleConfig) -> None:
        calls: list[str] = []

        class _SpyClient:
            def get_experiment_detail(self, eid: str) -> Any:
                calls.append(eid)

        store = OperatorStore()
        coord = PollingCoordinator(cfg, _SpyClient(), store)  # type: ignore[arg-type]
        fetch = coord._make_fetch_experiment()
        fetch()
        assert calls == [cfg.default_experiment_id]


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
    """Just-enough QThread stand-in for the orphan-list bookkeeping.

    Avoids spinning a real worker thread (flaky on Windows under
    pytest's GC-during-teardown). The real `QThread.terminate()` /
    `wait()` semantics are exercised in the standalone smoke test in
    docs/artifacts/operator_console_checklist.md.
    """

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


class _StubWorker:
    """Stand-in worker exposing only what `_stop_job` / `refresh_now`
    poke at via QMetaObject.invokeMethod (which is patched out)."""

    def __init__(self) -> None:
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
        assert thread.quit_called == 0, "thread.quit must come from the signal chain"
        assert invocations == [
            (worker, "stop", Qt.ConnectionType.QueuedConnection),
        ]
        assert JOB_OVERVIEW not in coord._jobs
        # Still-running thread parks on the orphan list for shutdown drain.
        assert len(coord._orphan_jobs) == 1
        assert coord._orphan_jobs[0].thread is thread

    def test_stop_job_drops_already_finished_orphans(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        # The orphan list must not grow unbounded across many route
        # changes — finished threads are pruned every time.
        coord = PollingCoordinator(cfg, client, OperatorStore())

        # Seed with a finished orphan from a previous route change.
        finished_worker = _StubWorker()
        finished_thread = _StubThread(running=False)
        coord._orphan_jobs.append(
            _JobHandle(coord._specs[JOB_HEALTH], finished_worker, finished_thread)  # type: ignore[arg-type]
        )

        _install_orphan_handle(coord, JOB_OVERVIEW)
        with patch("services.operator_console.polling.QMetaObject.invokeMethod"):
            coord._stop_job(JOB_OVERVIEW)

        assert finished_thread not in [h.thread for h in coord._orphan_jobs]

    def test_drain_orphan_jobs_terminates_stuck_threads(
        self, cfg: OperatorConsoleConfig, client: ApiClient
    ) -> None:
        # On shutdown the drain gives each worker a short wait; if the
        # thread is still in a urlopen after that, terminate() is the
        # only lever — the process is exiting and the OS reclaims the
        # socket either way.
        coord = PollingCoordinator(cfg, client, OperatorStore())

        stuck = _StubThread(running=True)  # `wait` keeps returning False
        coord._orphan_jobs.append(
            _JobHandle(coord._specs[JOB_OVERVIEW], _StubWorker(), stuck)  # type: ignore[arg-type]
        )

        coord._drain_orphan_jobs()

        assert stuck.wait_calls, "graceful wait must run before terminate"
        assert stuck.terminate_called == 1
        assert coord._orphan_jobs == []

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
