"""Tests for `OverviewView` — Phase 9.

Locks the six-card render path and the active-session click →
`session_activated(UUID)` shortcut the shell uses to jump to Live
Session.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from packages.schemas.operator_console import (
    AlertEvent,
    AlertKind,
    AlertSeverity,
    AttributionSummary,
    EncounterState,
    ExperimentSummary,
    HealthSnapshot,
    HealthState,
    HealthSubsystemStatus,
    LatestEncounterSummary,
    OverviewSnapshot,
    PhysiologyCurrentSnapshot,
    SemanticEvaluationSummary,
    SessionPhysiologySnapshot,
    SessionSummary,
    UiStatusKind,
)
from services.operator_console.state import OperatorStore
from services.operator_console.viewmodels.overview_vm import OverviewViewModel
from services.operator_console.views.overview_view import OverviewView

pytestmark = pytest.mark.usefixtures("qt_app")


_NOW = datetime(2026, 4, 17, 12, 0, 0, tzinfo=UTC)


def _session(session_id: UUID | None = None) -> SessionSummary:
    return SessionSummary(
        session_id=session_id or uuid4(),
        status="active",
        started_at_utc=_NOW,
        active_arm="greeting_v1",
        expected_greeting="hei rakas",
        duration_s=120.0,
    )


def _snap_with_session(session: SessionSummary) -> OverviewSnapshot:
    return OverviewSnapshot(
        generated_at_utc=_NOW,
        active_session=session,
        latest_encounter=LatestEncounterSummary(
            encounter_id="e1",
            session_id=session.session_id,
            segment_timestamp_utc=_NOW,
            state=EncounterState.COMPLETED,
            semantic_gate=1,
            p90_intensity=0.5,
            gated_reward=0.5,
            n_frames_in_window=150,
            semantic_evaluation=SemanticEvaluationSummary(
                reasoning="cross_encoder_high_match",
                is_match=True,
                confidence_score=0.91,
                semantic_method="cross_encoder",
                semantic_method_version="ce-v1",
            ),
            attribution=AttributionSummary(finality="online_provisional"),
        ),
        experiment_summary=ExperimentSummary(
            experiment_id="exp1",
            label="greeting line v1",
            active_arm_id="a2",
            arm_count=3,
            latest_reward=0.42,
        ),
        physiology=SessionPhysiologySnapshot(
            session_id=session.session_id,
            streamer=PhysiologyCurrentSnapshot(
                subject_role="streamer",
                rmssd_ms=45.0,
                heart_rate_bpm=72,
                is_stale=False,
                freshness_s=4.0,
            ),
            generated_at_utc=_NOW,
        ),
        health=HealthSnapshot(
            generated_at_utc=_NOW,
            overall_state=HealthState.OK,
            subsystems=[
                HealthSubsystemStatus(
                    subsystem_key="orchestrator",
                    label="Orchestrator",
                    state=HealthState.OK,
                ),
            ],
        ),
    )


def test_overview_view_renders_no_session_state() -> None:
    store = OperatorStore()
    vm = OverviewViewModel(store)
    view = OverviewView(vm)
    assert view._active_session_card._primary.text() == "No active session"  # type: ignore[attr-defined]
    # No click navigates when there's nothing to activate.
    assert view._active_session_id is None


def test_overview_view_populates_all_cards_from_snapshot() -> None:
    store = OperatorStore()
    vm = OverviewViewModel(store)
    view = OverviewView(vm)
    session = _session()
    store.set_overview(_snap_with_session(session))

    active_primary = view._active_session_card._primary.text()  # type: ignore[attr-defined]
    assert str(session.session_id) not in active_primary
    assert active_primary.startswith(str(session.session_id)[:8])
    assert active_primary.endswith(str(session.session_id)[-4:])
    active_secondary = view._active_session_card._secondary.text()  # type: ignore[attr-defined]
    assert str(session.session_id) in active_secondary
    assert "stimulus strategy greeting_v1" in active_secondary
    assert "expected response" in active_secondary
    assert "active strategy" not in view._experiment_card._secondary.text()  # type: ignore[attr-defined]
    assert "greeting line v1" in view._experiment_card._primary.text()  # type: ignore[attr-defined]
    assert "ok" in view._health_card._primary.text()  # type: ignore[attr-defined]
    # Physiology card reads live when heart variability is present.
    assert view._physiology_card._primary.text() == "Live heart data"  # type: ignore[attr-defined]
    # Latest encounter card surfaces the reward plus compact confirmation/follow-up summary.
    latest_secondary = view._latest_encounter_card._secondary.text()  # type: ignore[attr-defined]
    assert "reward" in view._latest_encounter_card._primary.text()  # type: ignore[attr-defined]
    assert "Stimulus confirmed? yes" in latest_secondary
    assert "stimulus confirmation" in latest_secondary
    assert "stimulus confirmation confidence 91%" in latest_secondary
    assert "follow-up signals online provisional" in latest_secondary


def test_overview_view_marks_active_conflict_as_error() -> None:
    store = OperatorStore()
    vm = OverviewViewModel(store)
    view = OverviewView(vm)
    session = _session().model_copy(update={"status": "active conflict"})
    store.set_overview(_snap_with_session(session))

    assert view._active_session_card._status._kind is UiStatusKind.ERROR  # type: ignore[attr-defined]
    assert view._active_session_card._status._label.text() == "active conflict"  # type: ignore[attr-defined]


def test_overview_view_attention_card_counts_alerts() -> None:
    store = OperatorStore()
    vm = OverviewViewModel(store)
    view = OverviewView(vm)
    store.set_alerts(
        [
            AlertEvent(
                alert_id="a1",
                severity=AlertSeverity.CRITICAL,
                kind=AlertKind.SUBSYSTEM_ERROR,
                message="GPU lost",
                emitted_at_utc=_NOW,
            ),
            AlertEvent(
                alert_id="a2",
                severity=AlertSeverity.WARNING,
                kind=AlertKind.PHYSIOLOGY_STALE,
                message="Streamer strap disconnected",
                emitted_at_utc=_NOW,
            ),
        ]
    )
    assert "2 open" in view._attention_card._primary.text()  # type: ignore[attr-defined]
    assert "1 critical" in view._attention_card._secondary.text()  # type: ignore[attr-defined]


def test_overview_view_active_session_click_emits_session_id() -> None:
    store = OperatorStore()
    vm = OverviewViewModel(store)
    view = OverviewView(vm)
    session = _session()
    store.set_overview(_snap_with_session(session))

    received: list[UUID] = []
    view.session_activated.connect(lambda sid: received.append(sid))
    # Invoke the click handler directly — testing the card's
    # `mousePressEvent` requires a visible widget in the offscreen QPA.
    view._on_active_session_clicked()
    assert received == [session.session_id]


def test_overview_view_physiology_stale_flag_surfaces_on_pill() -> None:
    store = OperatorStore()
    vm = OverviewViewModel(store)
    view = OverviewView(vm)
    session = _session()
    physiology = SessionPhysiologySnapshot(
        session_id=session.session_id,
        streamer=PhysiologyCurrentSnapshot(
            subject_role="streamer",
            rmssd_ms=45.0,
            heart_rate_bpm=72,
            is_stale=True,
            freshness_s=90.0,
        ),
        generated_at_utc=_NOW,
    )
    # The overview VM renders via the overview snapshot; a stale
    # streamer embedded there drives the physiology pill to "stale".
    snapshot = OverviewSnapshot(
        generated_at_utc=_NOW,
        active_session=session,
        physiology=physiology,
    )
    store.set_overview(snapshot)
    assert view._physiology_card._status._label.text() == "stale"  # type: ignore[attr-defined]


def test_overview_view_error_changed_shows_alert_banner() -> None:
    store = OperatorStore()
    vm = OverviewViewModel(store)
    view = OverviewView(vm)
    store.set_error("overview", "backend unreachable")
    # Offscreen QPA: parents aren't shown, so `isVisible()` is False even when
    # the banner was explicitly un-hidden. `isHidden()` reads the local flag.
    assert view._error_banner.isHidden() is False  # type: ignore[attr-defined]
