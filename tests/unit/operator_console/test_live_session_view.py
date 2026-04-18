"""Tests for `LiveSessionView` — Phase 9.

Locks the render path the operator relies on: empty-state when no
session is selected, header populated from the live-session DTO (not
from row data), and the detail pane surfacing the §7B reward
explanation after a row is selected.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from packages.schemas.operator_console import (
    EncounterState,
    EncounterSummary,
    SessionSummary,
    StimulusActionState,
)
from services.operator_console.state import OperatorStore, StimulusUiContext
from services.operator_console.table_models.encounters_table_model import (
    EncountersTableModel,
)
from services.operator_console.viewmodels.live_session_vm import LiveSessionViewModel
from services.operator_console.views.live_session_view import LiveSessionView

pytestmark = pytest.mark.usefixtures("qt_app")


_NOW = datetime(2026, 4, 17, 12, 0, 0, tzinfo=UTC)


def _session(session_id: UUID | None = None) -> SessionSummary:
    return SessionSummary(
        session_id=session_id or uuid4(),
        status="active",
        started_at_utc=_NOW,
        active_arm="greeting_v7",
        expected_greeting="hei rakas",
    )


def _encounter(
    encounter_id: str,
    *,
    state: EncounterState = EncounterState.COMPLETED,
    semantic_gate: int | None = 1,
    semantic_confidence: float | None = 0.9,
    p90: float | None = 0.42,
    gated_reward: float | None = 0.42,
    frames: int | None = 150,
    session_id: UUID | None = None,
) -> EncounterSummary:
    return EncounterSummary(
        encounter_id=encounter_id,
        session_id=session_id or uuid4(),
        segment_timestamp_utc=_NOW,
        state=state,
        active_arm="greeting_v7",
        expected_greeting="hei rakas",
        semantic_gate=semantic_gate,
        semantic_confidence=semantic_confidence,
        p90_intensity=p90,
        gated_reward=gated_reward,
        n_frames_in_window=frames,
        baseline_b_neutral=0.1,
    )


def _build_view() -> tuple[LiveSessionView, OperatorStore, LiveSessionViewModel]:
    store = OperatorStore()
    model = EncountersTableModel()
    vm = LiveSessionViewModel(store, model)
    view = LiveSessionView(vm)
    return view, store, vm


def test_live_session_view_shows_empty_state_without_session() -> None:
    view, _store, _vm = _build_view()
    # Offscreen QPA: `isVisible()` is False until the parent chain is shown.
    # `isHidden()` reads the local flag and tells us the page was wired to
    # show the empty state rather than the body container.
    assert view._empty_state.isHidden() is False  # type: ignore[attr-defined]
    assert view._body_container.isHidden() is True  # type: ignore[attr-defined]


def test_live_session_view_header_reads_from_live_session_dto() -> None:
    view, store, _vm = _build_view()
    store.set_live_session(_session())
    # Arm + greeting come from the live_session DTO, never from rows.
    panel = view._session_panel
    assert "greeting_v7" in panel._arm_label.text()  # type: ignore[attr-defined]
    assert "hei rakas" in panel._greeting_label.text()  # type: ignore[attr-defined]


def test_live_session_view_detail_pane_shows_reward_explanation() -> None:
    view, store, vm = _build_view()
    session = _session()
    store.set_live_session(session)
    store.set_encounters([_encounter("e1", session_id=session.session_id)])
    vm.select_encounter("e1")

    detail = view._detail_panel
    # P90 card reads the intensity; reward card reads the gated reward.
    assert "0.420" in detail._p90_card._primary.text()  # type: ignore[attr-defined]
    assert "0.420" in detail._reward_card._primary.text()  # type: ignore[attr-defined]
    assert "150" in detail._frames_card._primary.text()  # type: ignore[attr-defined]
    # Reward explanation sentence mentions the §7B inputs by name.
    assert "P90" in detail._explanation.text() or "p90" in detail._explanation.text().lower()  # type: ignore[attr-defined]


def test_live_session_view_detail_pane_flags_zero_frames() -> None:
    view, store, vm = _build_view()
    session = _session()
    store.set_live_session(session)
    store.set_encounters(
        [
            _encounter(
                "e1",
                state=EncounterState.REJECTED_NO_FRAMES,
                frames=0,
                p90=None,
                gated_reward=None,
                session_id=session.session_id,
            )
        ]
    )
    vm.select_encounter("e1")
    detail = view._detail_panel
    assert "No valid AU12 frames" in detail._explanation.text()  # type: ignore[attr-defined]


def test_live_session_view_detail_pane_flags_gate_closed() -> None:
    view, store, vm = _build_view()
    session = _session()
    store.set_live_session(session)
    store.set_encounters(
        [
            _encounter(
                "e1",
                semantic_gate=0,
                gated_reward=0.0,
                session_id=session.session_id,
            )
        ]
    )
    vm.select_encounter("e1")
    detail = view._detail_panel
    assert "gate closed" in detail._explanation.text().lower()  # type: ignore[attr-defined]


def test_live_session_view_countdown_timer_activates_on_measuring() -> None:
    view, store, _vm = _build_view()
    session = _session()
    store.set_live_session(session)
    # Anchor the stimulus clock to wall-clock "now" so the §7B 30-second
    # measurement window is in the future and the 1s tick does not auto-
    # stop itself on the zero-remaining boundary.
    store.set_stimulus_ui_context(
        StimulusUiContext(
            state=StimulusActionState.MEASURING,
            authoritative_stimulus_time_utc=datetime.now(UTC),
        )
    )
    assert view._countdown_timer.isActive() is True


def test_live_session_view_countdown_timer_stops_when_not_measuring() -> None:
    view, store, _vm = _build_view()
    session = _session()
    store.set_live_session(session)
    # Start measuring, then transition to COMPLETED — timer must stop.
    store.set_stimulus_ui_context(
        StimulusUiContext(
            state=StimulusActionState.MEASURING,
            authoritative_stimulus_time_utc=_NOW,
        )
    )
    store.set_stimulus_ui_context(StimulusUiContext(state=StimulusActionState.COMPLETED))
    assert view._countdown_timer.isActive() is False


def test_live_session_view_on_activated_does_not_crash_without_session() -> None:
    view, _store, _vm = _build_view()
    # Page may be activated before a session is selected — must be a no-op
    # that leaves the empty state visible.
    view.on_activated()
    assert view._empty_state.isHidden() is False  # type: ignore[attr-defined]
