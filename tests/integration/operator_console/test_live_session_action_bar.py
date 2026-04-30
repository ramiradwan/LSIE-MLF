"""Integration: Live Session action bar + detail pane round-trip.

Two composition-level assertions Phase 9 unit tests can't make on their
own:

  1. The ActionBar disables its submit button while a stimulus is in
     flight — the view sees `StimulusUiContext(state=SUBMITTING)` from
     the store and must not let the operator double-fire.
  2. Selecting an encounter row on the table replaces the detail pane's
     reward-explanation text with the selected encounter's fields — the
     §7B reward-explanation bundle the operator trusts lives in that
     pane and must update on selection.

Spec references:
  §4.C           — authoritative stimulus_time is orchestrator-owned;
                   UI state derives from `StimulusUiContext`
  §7B            — reward explanation uses p90_intensity × semantic_gate,
                   n_frames_in_window, gated_reward, au12_baseline_pre
  §12            — SUBMITTING is a lifecycle state, not an error
  SPEC-AMEND-008 — PySide6 Operator Console
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from packages.schemas.operator_console import (
    EncounterState,
    EncounterSummary,
    ObservationalAcousticSummary,
    SessionCreateRequest,
    SessionEndRequest,
    SessionLifecycleAccepted,
    SessionSummary,
    StimulusActionState,
)
from services.operator_console.api_client import ApiError
from services.operator_console.formatters import operator_ready_for_submit
from services.operator_console.state import OperatorStore, StimulusUiContext
from services.operator_console.table_models.encounters_table_model import (
    EncountersTableModel,
)
from services.operator_console.viewmodels.live_session_vm import LiveSessionViewModel
from services.operator_console.views.live_session_view import LiveSessionView
from services.operator_console.widgets.action_bar import ActionBar
from services.operator_console.widgets.metric_card import MetricCard
from services.operator_console.workers import OneShotSignals

pytestmark = pytest.mark.usefixtures("qt_app")


_NOW = datetime(2026, 4, 17, 12, 0, 0, tzinfo=UTC)


def _seed_session(
    store: OperatorStore,
    *,
    is_calibrating: bool | None = None,
    calibration_frames_accumulated: int | None = None,
    calibration_frames_required: int | None = None,
) -> SessionSummary:
    session = SessionSummary(
        session_id=uuid4(),
        status="active",
        started_at_utc=_NOW,
        active_arm="greeting_v1",
        expected_greeting="hei rakas",
        is_calibrating=is_calibrating,
        calibration_frames_accumulated=calibration_frames_accumulated,
        calibration_frames_required=calibration_frames_required,
        duration_s=90.0,
    )
    store.set_selected_session_id(session.session_id)
    store.set_live_session(session)
    return session


@dataclass
class _FakeSessionStartDispatcher:
    calls: list[SessionCreateRequest] = field(default_factory=list)
    signals: list[OneShotSignals] = field(default_factory=list)

    def __call__(self, request: SessionCreateRequest) -> OneShotSignals:
        self.calls.append(request)
        bus = OneShotSignals()
        self.signals.append(bus)
        return bus


@dataclass
class _FakeSessionEndCall:
    session_id: UUID
    request: SessionEndRequest


@dataclass
class _FakeSessionEndDispatcher:
    calls: list[_FakeSessionEndCall] = field(default_factory=list)
    signals: list[OneShotSignals] = field(default_factory=list)

    def __call__(self, session_id: UUID, request: SessionEndRequest) -> OneShotSignals:
        self.calls.append(_FakeSessionEndCall(session_id=session_id, request=request))
        bus = OneShotSignals()
        self.signals.append(bus)
        return bus


def test_action_bar_disables_submit_while_stimulus_submitting() -> None:
    # A persistent ActionBar on the shell mirrors what the MainWindow
    # mounts once below the content area per SPEC-AMEND-008.
    store = OperatorStore()
    session = _seed_session(store)
    bar = ActionBar()
    bar.set_session_context(session.session_id, session.active_arm, session.expected_greeting)
    # Idle state — the button is enabled because a session is bound.
    assert bar._submit_button.isEnabled() is True  # type: ignore[attr-defined]

    # Submitting lands on the bar through the store → shell wiring.
    bar.set_action_state(
        StimulusUiContext(
            state=StimulusActionState.SUBMITTING,
            submitted_at_utc=_NOW,
        )
    )
    assert bar._submit_button.isEnabled() is False  # type: ignore[attr-defined]

    # Back to idle — re-enabled so the operator can retry.
    bar.set_action_state(StimulusUiContext(state=StimulusActionState.IDLE))
    assert bar._submit_button.isEnabled() is True  # type: ignore[attr-defined]


def test_action_bar_disables_submit_until_calibration_ready() -> None:
    store = OperatorStore()
    session = _seed_session(
        store,
        is_calibrating=True,
        calibration_frames_accumulated=12,
        calibration_frames_required=45,
    )
    bar = ActionBar()
    bar.set_session_context(
        session.session_id,
        session.active_arm,
        session.expected_greeting,
        operator_ready_for_submit=operator_ready_for_submit(session),
    )
    assert bar._submit_button.isEnabled() is False  # type: ignore[attr-defined]

    bar.set_action_state(StimulusUiContext(state=StimulusActionState.IDLE))
    assert bar._submit_button.isEnabled() is False  # type: ignore[attr-defined]

    ready_session = session.model_copy(update={"calibration_frames_accumulated": 45})
    assert ready_session.is_calibrating is True
    bar.set_session_context(
        ready_session.session_id,
        ready_session.active_arm,
        ready_session.expected_greeting,
        operator_ready_for_submit=operator_ready_for_submit(ready_session),
    )
    assert bar._submit_button.isEnabled() is True  # type: ignore[attr-defined]


def test_live_session_view_encounter_selection_updates_detail_pane() -> None:
    store = OperatorStore()
    session = _seed_session(store)
    model = EncountersTableModel()
    vm = LiveSessionViewModel(store, model)
    view = LiveSessionView(vm)

    encounter = EncounterSummary(
        encounter_id="enc-1",
        session_id=session.session_id,
        segment_timestamp_utc=_NOW,
        state=EncounterState.COMPLETED,
        active_arm="greeting_v1",
        semantic_gate=1,
        semantic_confidence=0.82,
        p90_intensity=0.55,
        gated_reward=0.55,
        n_frames_in_window=150,
        au12_baseline_pre=0.12,
        observational_acoustic=ObservationalAcousticSummary(
            f0_valid_measure=True,
            f0_valid_baseline=True,
            perturbation_valid_measure=False,
            perturbation_valid_baseline=True,
            voiced_coverage_measure_s=3.1,
            voiced_coverage_baseline_s=2.4,
            f0_mean_measure_hz=221.0,
            f0_mean_baseline_hz=221.0,
            f0_delta_semitones=0.0,
        ),
    )
    store.set_encounters([encounter])

    # Selecting the encounter pushes it into the VM's selection,
    # which drives the detail pane's reward-explanation text.
    vm.select_encounter(encounter.encounter_id)
    detail_text = view._detail_panel._explanation.text()  # type: ignore[attr-defined]
    # The explanation mentions the two §7B reward inputs by name. The
    # exact formatting is the formatter's business; integration-level
    # we only assert both show up in the rendered string.
    assert "p90" in detail_text.lower()
    assert "gate" in detail_text.lower()
    # And the reward card reads a numeric value (the §7B gated reward).
    assert view._detail_panel._reward_card._primary.text() != "—"  # type: ignore[attr-defined]
    # The detail pane keeps the original six reward/physiology cards and adds
    # three §7D acoustic cards plus six compact §8/§7E diagnostic cards; the
    # reward assertions above remain untouched.
    assert len(view._detail_panel.findChildren(MetricCard)) == 15  # type: ignore[attr-defined]
    acoustic_text = view._detail_panel._acoustic_explanation.text()  # type: ignore[attr-defined]
    perturbation_text = (  # type: ignore[attr-defined]
        view._detail_panel._perturbation_validity_pill.text()
    )
    assert "F0 Δ +0.00 st" in acoustic_text
    assert "Jitter Δ not measured" in acoustic_text
    assert "measure perturbation window invalid" in perturbation_text


def test_live_session_view_end_button_waits_for_authoritative_end_readback() -> None:
    store = OperatorStore()
    session = _seed_session(store)
    model = EncountersTableModel()
    vm = LiveSessionViewModel(store, model)
    start_dispatcher = _FakeSessionStartDispatcher()
    end_dispatcher = _FakeSessionEndDispatcher()
    vm.bind_session_lifecycle_actions(start_dispatcher, end_dispatcher)
    view = LiveSessionView(vm)

    view._session_panel._end_button.click()  # type: ignore[attr-defined]
    assert len(end_dispatcher.calls) == 1
    assert view._session_panel._end_button.isEnabled() is False  # type: ignore[attr-defined]
    assert view._session_panel._end_button.text() == "Ending…"  # type: ignore[attr-defined]

    call = end_dispatcher.calls[0]
    end_dispatcher.signals[0].succeeded.emit(
        "session_end",
        SessionLifecycleAccepted(
            action="end",
            session_id=session.session_id,
            client_action_id=call.request.client_action_id,
            accepted=True,
            received_at_utc=_NOW,
        ),
    )
    assert view._session_panel._end_button.isEnabled() is False  # type: ignore[attr-defined]

    store.set_live_session(session.model_copy(update={"status": "ended", "ended_at_utc": _NOW}))
    assert view._session_panel._end_button.isHidden() is True  # type: ignore[attr-defined]


def test_live_session_view_session_start_failure_surfaces_error_banner() -> None:
    store = OperatorStore()
    _seed_session(store)
    model = EncountersTableModel()
    vm = LiveSessionViewModel(store, model)
    start_dispatcher = _FakeSessionStartDispatcher()
    end_dispatcher = _FakeSessionEndDispatcher()
    vm.bind_session_lifecycle_actions(start_dispatcher, end_dispatcher)
    view = LiveSessionView(vm)

    vm.start_new_session("rtmp://example/live", "greeting_line_v1")
    start_dispatcher.signals[0].failed.emit(
        "session_start",
        ApiError(message="broker unavailable", retryable=True),
    )

    assert view._error_banner.isHidden() is False  # type: ignore[attr-defined]
    assert view._error_banner._message.text() == "broker unavailable"  # type: ignore[attr-defined]
