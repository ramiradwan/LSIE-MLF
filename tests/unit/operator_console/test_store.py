"""Tests for `OperatorStore` — Phase 4.

Store is a dumb signal bus plus getters/setters. Tests assert that:
  - every top-level replacement emits the correct signal with the
    expected payload
  - idempotent setters do not emit a redundant signal
  - selected session id persists across a route change
  - per-scope error signal carries the scope + message tuple
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from PySide6.QtCore import QCoreApplication

from packages.schemas.operator_console import (
    AlertEvent,
    AlertKind,
    AlertSeverity,
    AttributionSummary,
    EncounterState,
    EncounterSummary,
    ExperimentDetail,
    HealthSnapshot,
    HealthState,
    ObservationalAcousticSummary,
    OverviewSnapshot,
    SemanticEvaluationSummary,
    SessionSummary,
    StimulusActionState,
)
from services.operator_console.state import (
    AppRoute,
    OperatorStore,
    StimulusUiContext,
)

pytestmark = pytest.mark.usefixtures("qt_app")


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


# ----------------------------------------------------------------------
# Signal capture helper (no pytest-qt)
# ----------------------------------------------------------------------


@dataclass
class _SignalSpy:
    """Capture signal emissions synchronously via a direct connection."""

    emissions: list[tuple[Any, ...]] = field(default_factory=list)

    def __call__(self, *args: Any) -> None:
        self.emissions.append(args)


@pytest.fixture
def spy_bank(qt_app: QCoreApplication) -> Iterator[dict[str, _SignalSpy]]:
    del qt_app
    yield {}


# ----------------------------------------------------------------------
# Helpers to build minimally-populated DTOs
# ----------------------------------------------------------------------


def _session(session_id: UUID | None = None) -> SessionSummary:
    return SessionSummary(
        session_id=session_id or uuid4(),
        status="live",
        started_at_utc=_utc(2026, 4, 18, 10, 0),
    )


def _encounter(session_id: UUID) -> EncounterSummary:
    return EncounterSummary(
        encounter_id="e1",
        session_id=session_id,
        segment_timestamp_utc=_utc(2026, 4, 18, 10, 5),
        state=EncounterState.COMPLETED,
        observational_acoustic=ObservationalAcousticSummary(
            f0_valid_measure=True,
            f0_valid_baseline=True,
            voiced_coverage_measure_s=2.0,
            voiced_coverage_baseline_s=2.5,
            f0_delta_semitones=1.25,
        ),
        semantic_evaluation=SemanticEvaluationSummary(
            reasoning="gray_band_llm_match",
            is_match=True,
            confidence_score=0.68,
            semantic_method="llm_gray_band",
            semantic_method_version="gray-v1",
        ),
        attribution=AttributionSummary(
            finality="online_provisional",
            soft_reward_candidate=0.33,
            au12_lift_p90=0.49,
            sync_peak_corr=0.21,
            outcome_link_lag_s=31.0,
        ),
    )


def _alert() -> AlertEvent:
    return AlertEvent(
        alert_id="a1",
        severity=AlertSeverity.WARNING,
        kind=AlertKind.PHYSIOLOGY_STALE,
        message="operator stale",
        emitted_at_utc=_utc(2026, 4, 18, 10, 1),
    )


# ----------------------------------------------------------------------
# Route + selection
# ----------------------------------------------------------------------


class TestRouting:
    def test_default_route_is_overview(self) -> None:
        store = OperatorStore()
        assert store.route() is AppRoute.OVERVIEW

    def test_route_change_emits_string_value(self) -> None:
        store = OperatorStore()
        spy = _SignalSpy()
        store.route_changed.connect(spy)
        store.set_route(AppRoute.LIVE_SESSION)
        assert spy.emissions == [("live_session",)]
        assert store.route() is AppRoute.LIVE_SESSION

    def test_idempotent_route_does_not_emit(self) -> None:
        store = OperatorStore()
        spy = _SignalSpy()
        store.route_changed.connect(spy)
        store.set_route(AppRoute.OVERVIEW)
        assert spy.emissions == []

    def test_selected_session_persists_across_route_change(self) -> None:
        store = OperatorStore()
        sid = uuid4()
        store.set_selected_session_id(sid)
        store.set_route(AppRoute.LIVE_SESSION)
        store.set_route(AppRoute.EXPERIMENTS)
        store.set_route(AppRoute.OVERVIEW)
        assert store.selected_session_id() == sid

    def test_selected_session_emits_on_change(self) -> None:
        store = OperatorStore()
        spy = _SignalSpy()
        store.selected_session_changed.connect(spy)
        sid = uuid4()
        store.set_selected_session_id(sid)
        assert spy.emissions == [(sid,)]

    def test_selected_session_idempotent(self) -> None:
        store = OperatorStore()
        sid = uuid4()
        store.set_selected_session_id(sid)
        spy = _SignalSpy()
        store.selected_session_changed.connect(spy)
        store.set_selected_session_id(sid)
        assert spy.emissions == []

    def test_managed_experiment_persists_across_route_change(self) -> None:
        store = OperatorStore()
        store.set_managed_experiment_id("exp-non-default")
        store.set_route(AppRoute.EXPERIMENTS)
        store.set_route(AppRoute.OVERVIEW)
        assert store.managed_experiment_id() == "exp-non-default"

    def test_setting_experiment_updates_managed_experiment_id(self) -> None:
        store = OperatorStore()
        spy = _SignalSpy()
        store.managed_experiment_changed.connect(spy)
        detail = ExperimentDetail(experiment_id="exp-loaded")
        store.set_experiment(detail)
        assert store.managed_experiment_id() == "exp-loaded"
        assert spy.emissions == [("exp-loaded",)]


# ----------------------------------------------------------------------
# Data replacement signals
# ----------------------------------------------------------------------


class TestDataReplacements:
    def test_overview_replacement_emits_snapshot(self) -> None:
        store = OperatorStore()
        spy = _SignalSpy()
        store.overview_changed.connect(spy)
        snap = OverviewSnapshot(generated_at_utc=_utc(2026, 4, 18, 10, 0))
        store.set_overview(snap)
        assert spy.emissions == [(snap,)]
        assert store.overview() is snap

    def test_sessions_replacement_copies_list(self) -> None:
        store = OperatorStore()
        rows = [_session(), _session()]
        spy = _SignalSpy()
        store.sessions_changed.connect(spy)
        store.set_sessions(rows)
        assert len(spy.emissions) == 1
        (emitted,) = spy.emissions[0]
        # store returns a copy so mutation cannot leak back in
        assert emitted == rows
        emitted.clear()
        assert len(store.sessions()) == 2

    def test_encounters_replacement(self) -> None:
        store = OperatorStore()
        sid = uuid4()
        rows = [_encounter(sid)]
        spy = _SignalSpy()
        store.encounters_changed.connect(spy)
        store.set_encounters(rows)
        assert len(spy.emissions) == 1
        (emitted,) = spy.emissions[0]
        assert isinstance(emitted[0], EncounterSummary)
        assert emitted[0].observational_acoustic is not None
        assert emitted[0].observational_acoustic.f0_delta_semitones == 1.25
        assert emitted[0].semantic_evaluation is not None
        assert emitted[0].semantic_evaluation.reasoning == "gray_band_llm_match"
        assert emitted[0].attribution is not None
        assert emitted[0].attribution.finality == "online_provisional"
        stored = store.encounters()
        assert isinstance(stored[0], EncounterSummary)
        assert stored[0].encounter_id == "e1"
        assert stored[0].observational_acoustic is not None
        assert stored[0].observational_acoustic.f0_valid_measure is True
        assert stored[0].semantic_evaluation is not None
        assert stored[0].semantic_evaluation.semantic_method == "llm_gray_band"
        assert stored[0].attribution is not None
        assert stored[0].attribution.outcome_link_lag_s == 31.0

    def test_alerts_replacement(self) -> None:
        store = OperatorStore()
        spy = _SignalSpy()
        store.alerts_changed.connect(spy)
        store.set_alerts([_alert()])
        assert len(spy.emissions) == 1
        assert store.alerts()[0].alert_id == "a1"

    def test_health_replacement(self) -> None:
        store = OperatorStore()
        spy = _SignalSpy()
        store.health_changed.connect(spy)
        snap = HealthSnapshot(
            generated_at_utc=_utc(2026, 4, 18, 10, 0),
            overall_state=HealthState.OK,
        )
        store.set_health(snap)
        assert spy.emissions == [(snap,)]


# ----------------------------------------------------------------------
# Stimulus UI context
# ----------------------------------------------------------------------


class TestStimulusContext:
    def test_default_context_is_idle(self) -> None:
        store = OperatorStore()
        ctx = store.stimulus_ui_context()
        assert ctx.state is StimulusActionState.IDLE
        assert ctx.client_action_id is None

    def test_replacing_context_emits(self) -> None:
        store = OperatorStore()
        spy = _SignalSpy()
        store.stimulus_state_changed.connect(spy)
        action_id = uuid4()
        ctx = StimulusUiContext(
            state=StimulusActionState.SUBMITTING,
            client_action_id=action_id,
            operator_note="hi",
        )
        store.set_stimulus_ui_context(ctx)
        assert spy.emissions == [(ctx,)]
        assert store.stimulus_ui_context().state is StimulusActionState.SUBMITTING


# ----------------------------------------------------------------------
# Error scopes
# ----------------------------------------------------------------------


class TestErrorScopes:
    def test_set_error_emits_scope_and_message(self) -> None:
        store = OperatorStore()
        spy = _SignalSpy()
        store.error_changed.connect(spy)
        store.set_error("overview", "boom")
        assert spy.emissions == [("overview", "boom")]
        assert store.error("overview") == "boom"

    def test_idempotent_error_does_not_emit(self) -> None:
        store = OperatorStore()
        store.set_error("overview", "boom")
        spy = _SignalSpy()
        store.error_changed.connect(spy)
        store.set_error("overview", "boom")
        assert spy.emissions == []

    def test_two_scopes_coexist(self) -> None:
        store = OperatorStore()
        store.set_error("overview", "boom")
        store.set_error("alerts", "flaky")
        assert store.error("overview") == "boom"
        assert store.error("alerts") == "flaky"

    def test_clear_error_emits_cleared(self) -> None:
        store = OperatorStore()
        store.set_error("overview", "boom")
        spy = _SignalSpy()
        store.error_cleared.connect(spy)
        store.clear_error("overview")
        assert spy.emissions == [("overview",)]
        assert store.error("overview") is None

    def test_clear_unknown_scope_is_silent(self) -> None:
        store = OperatorStore()
        spy = _SignalSpy()
        store.error_cleared.connect(spy)
        store.clear_error("nope")
        assert spy.emissions == []
