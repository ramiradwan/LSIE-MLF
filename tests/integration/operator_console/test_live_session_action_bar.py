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
                   n_frames_in_window, gated_reward, baseline_b_neutral
  §12            — SUBMITTING is a lifecycle state, not an error
  SPEC-AMEND-008 — PySide6 Operator Console
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

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
from services.operator_console.widgets.action_bar import ActionBar

pytestmark = pytest.mark.usefixtures("qt_app")


_NOW = datetime(2026, 4, 17, 12, 0, 0, tzinfo=UTC)


def _seed_session(store: OperatorStore) -> SessionSummary:
    session = SessionSummary(
        session_id=uuid4(),
        status="active",
        started_at_utc=_NOW,
        active_arm="greeting_v1",
        expected_greeting="hei rakas",
        duration_s=90.0,
    )
    store.set_selected_session_id(session.session_id)
    store.set_live_session(session)
    return session


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
        baseline_b_neutral=0.12,
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
