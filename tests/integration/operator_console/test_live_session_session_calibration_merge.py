"""Integration coverage for the merged session-lifecycle + calibration seam.

These tests intentionally compose the real Store → ViewModel → View widgets
around fake coordinator one-shot dispatchers. That keeps the operator
console on its API/coordinator abstraction while proving calibration
readiness only gates stimulus submission, not session lifecycle controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from packages.schemas.operator_console import (
    SessionCreateRequest,
    SessionEndRequest,
    SessionSummary,
    UiStatusKind,
)
from services.operator_console.formatters import operator_ready_for_submit
from services.operator_console.state import OperatorStore
from services.operator_console.table_models.encounters_table_model import EncountersTableModel
from services.operator_console.viewmodels.live_session_vm import LiveSessionViewModel
from services.operator_console.views.live_session_view import LiveSessionView
from services.operator_console.widgets.action_bar import ActionBar
from services.operator_console.workers import OneShotSignals

pytestmark = pytest.mark.usefixtures("qt_app")


_NOW = datetime(2026, 4, 17, 12, 0, 0, tzinfo=UTC)


def _session(
    *,
    is_calibrating: bool | None = None,
    calibration_frames_accumulated: int | None = None,
    calibration_frames_required: int | None = None,
) -> SessionSummary:
    return SessionSummary(
        session_id=uuid4(),
        status="active",
        started_at_utc=_NOW,
        active_arm="greeting_v1",
        expected_greeting="Hei rakas — welcome to the stream!",
        is_calibrating=is_calibrating,
        calibration_frames_accumulated=calibration_frames_accumulated,
        calibration_frames_required=calibration_frames_required,
    )


@dataclass
class _FakeSessionStartDispatcher:
    calls: list[SessionCreateRequest] = field(default_factory=list)
    signals: list[OneShotSignals] = field(default_factory=list)

    def __call__(self, request: SessionCreateRequest) -> OneShotSignals:
        self.calls.append(request)
        signals = OneShotSignals()
        self.signals.append(signals)
        return signals


@dataclass
class _EndCall:
    session_id: UUID
    request: SessionEndRequest


@dataclass
class _FakeSessionEndDispatcher:
    calls: list[_EndCall] = field(default_factory=list)
    signals: list[OneShotSignals] = field(default_factory=list)

    def __call__(self, session_id: UUID, request: SessionEndRequest) -> OneShotSignals:
        self.calls.append(_EndCall(session_id=session_id, request=request))
        signals = OneShotSignals()
        self.signals.append(signals)
        return signals


def _build_live_session_view(
    session: SessionSummary,
) -> tuple[
    LiveSessionView,
    LiveSessionViewModel,
    _FakeSessionStartDispatcher,
    _FakeSessionEndDispatcher,
]:
    store = OperatorStore()
    store.set_selected_session_id(session.session_id)
    store.set_live_session(session)
    vm = LiveSessionViewModel(store, EncountersTableModel())
    start = _FakeSessionStartDispatcher()
    end = _FakeSessionEndDispatcher()
    vm.bind_session_lifecycle_actions(start, end)
    view = LiveSessionView(vm)
    return view, vm, start, end


def test_header_composes_calibration_status_with_session_lifecycle_controls() -> None:
    session = _session(
        is_calibrating=True,
        calibration_frames_accumulated=12,
        calibration_frames_required=45,
    )
    view, _vm, _start, end = _build_live_session_view(session)

    panel = view._session_panel  # type: ignore[attr-defined]
    assert "greeting_v1" in panel._arm_label.text()  # type: ignore[attr-defined]
    assert "Hei rakas" in panel._greeting_label.text()  # type: ignore[attr-defined]
    assert panel._calibration_pill.kind() == UiStatusKind.PROGRESS  # type: ignore[attr-defined]
    assert panel._calibration_pill.text() == "Calibrating · 12/45 frames"  # type: ignore[attr-defined]
    assert panel._controls_label.text() == "Session controls"  # type: ignore[attr-defined]
    assert panel._start_button.text() == "Start new session"  # type: ignore[attr-defined]
    assert panel._start_button.isEnabled() is True  # type: ignore[attr-defined]
    assert panel._end_button.isHidden() is False  # type: ignore[attr-defined]
    assert panel._end_button.isEnabled() is True  # type: ignore[attr-defined]

    panel._end_button.click()  # type: ignore[attr-defined]
    assert [call.session_id for call in end.calls] == [session.session_id]


def test_session_start_dispatch_uses_bound_coordinator_submitter_not_backend_clients() -> None:
    session = _session(
        is_calibrating=True,
        calibration_frames_accumulated=3,
        calibration_frames_required=45,
    )
    _view, vm, start, _end = _build_live_session_view(session)

    action_id = vm.start_new_session("  rtmp://example/live  ", "  greeting_line_v1  ")

    assert action_id is not None
    assert len(start.calls) == 1
    request = start.calls[0]
    assert request.client_action_id == action_id
    assert request.stream_url == "rtmp://example/live"
    assert request.experiment_id == "greeting_line_v1"


def test_false_and_none_calibration_states_leave_stimulus_submit_ready() -> None:
    for is_calibrating in (False, None):
        session = _session(is_calibrating=is_calibrating)
        bar = ActionBar()
        bar.set_session_context(
            session.session_id,
            session.active_arm,
            session.expected_greeting,
            operator_ready_for_submit=operator_ready_for_submit(session),
        )

        assert bar._submit_button.isEnabled() is True  # type: ignore[attr-defined]
