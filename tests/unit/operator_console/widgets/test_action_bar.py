"""Tests for `ActionBar` — Phase 5.

Covers the bits that the operator actually relies on:
  - button disabled when no session is selected (§4.C guard)
  - button disabled while the action is in-flight (§4.C dedup)
  - visual state progression IDLE → SUBMITTING → ACCEPTED → MEASURING
    → COMPLETED reflected by the pill label + button label
  - `stimulus_requested` emits the operator note and does not fire
    when the button is disabled
  - countdown is shown/hidden with the MEASURING state
"""

from __future__ import annotations

from typing import cast
from uuid import uuid4

import pytest
from PySide6.QtWidgets import QWidget

from packages.schemas.operator_console import StimulusActionState
from services.operator_console.state import StimulusUiContext
from services.operator_console.widgets.action_bar import ActionBar

pytestmark = pytest.mark.usefixtures("qt_app")


def _layout_widget(layout: object, index: int) -> QWidget:
    item = layout.itemAt(index)  # type: ignore[attr-defined]
    assert item is not None
    widget = item.widget()
    assert widget is not None
    return cast(QWidget, widget)


def test_button_disabled_without_session() -> None:
    bar = ActionBar()
    # No session set → button disabled.
    assert bar._submit_button.isEnabled() is False  # type: ignore[attr-defined]


def test_action_bar_primary_controls_have_accessible_names() -> None:
    bar = ActionBar()
    assert bar._note_input.accessibleName() == "Operator note"  # type: ignore[attr-defined]
    assert bar._submit_button.accessibleName() == "Send stimulus"  # type: ignore[attr-defined]


def test_focus_note_input_moves_keyboard_focus() -> None:
    bar = ActionBar()
    bar.show()
    bar.focus_note_input()
    assert bar.focusWidget() is bar._note_input  # type: ignore[attr-defined]


def test_set_session_context_enables_button() -> None:
    bar = ActionBar()
    session_id = uuid4()
    bar.set_session_context(
        session_id=session_id,
        active_arm="greeting_v1",
        expected_response_text="hei rakas",
    )
    assert bar._submit_button.isEnabled() is True  # type: ignore[attr-defined]
    # Wide mode renders the abbreviated UUID — full UUIDs wrapped the
    # ActionBar onto two rows on 1024–1366 displays, which is the most
    # common operator screen, so we abbreviate consistently.
    expected_compact = f"{str(session_id)[:8]}…{str(session_id)[-4:]}"
    assert bar._session_label.text() == (  # type: ignore[attr-defined]
        f"Session {expected_compact} — stimulus strategy: greeting_v1"
    )
    assert bar._greeting_label.text() == "Expected response: “hei rakas”"  # type: ignore[attr-defined]


def test_operator_not_ready_session_context_disables_button() -> None:
    bar = ActionBar()
    bar.set_session_context(
        session_id=uuid4(),
        active_arm="greeting_v1",
        expected_response_text="hei rakas",
        operator_ready_for_submit=False,
    )
    assert bar._submit_button.isEnabled() is False  # type: ignore[attr-defined]


def test_legacy_unknown_readiness_defaults_to_enabled_for_bound_session() -> None:
    bar = ActionBar()
    bar.set_session_context(
        session_id=uuid4(),
        active_arm="greeting_v1",
        expected_response_text="hei rakas",
        operator_ready_for_submit=None,
    )
    assert bar._submit_button.isEnabled() is True  # type: ignore[attr-defined]


def test_set_session_context_none_disables_button() -> None:
    bar = ActionBar()
    bar.set_session_context(uuid4(), "arm1", "hello")
    bar.set_session_context(None, None, None)
    assert bar._submit_button.isEnabled() is False  # type: ignore[attr-defined]


def test_submitting_state_disables_button() -> None:
    bar = ActionBar()
    bar.set_session_context(uuid4(), "arm1", "hello")
    bar.set_action_state(StimulusUiContext(state=StimulusActionState.SUBMITTING))
    assert bar._submit_button.isEnabled() is False  # type: ignore[attr-defined]


def test_completed_state_stays_disabled_until_operator_ready() -> None:
    bar = ActionBar()
    session_id = uuid4()
    bar.set_session_context(session_id, "arm1", "hello", operator_ready_for_submit=False)
    bar.set_action_state(StimulusUiContext(state=StimulusActionState.COMPLETED))
    assert bar._submit_button.isEnabled() is False  # type: ignore[attr-defined]

    bar.set_session_context(session_id, "arm1", "hello", operator_ready_for_submit=True)
    assert bar._submit_button.isEnabled() is True  # type: ignore[attr-defined]


def test_accepted_measuring_completed_progression() -> None:
    bar = ActionBar()
    bar.set_session_context(uuid4(), "arm1", "hello")

    for state, expected_label in [
        (StimulusActionState.ACCEPTED, "Accepted"),
        (StimulusActionState.MEASURING, "Measuring…"),
        (StimulusActionState.COMPLETED, "Send Stimulus"),
    ]:
        bar.set_action_state(StimulusUiContext(state=state))
        assert bar._submit_button.text() == expected_label  # type: ignore[attr-defined]

    # Only COMPLETED re-enables the button for a fresh submission.
    bar.set_action_state(StimulusUiContext(state=StimulusActionState.COMPLETED))
    assert bar._submit_button.isEnabled() is True  # type: ignore[attr-defined]


def test_stimulus_requested_emits_note() -> None:
    bar = ActionBar()
    bar.set_session_context(uuid4(), "arm1", "hello")
    notes: list[str] = []
    bar.stimulus_requested.connect(notes.append)

    bar._note_input.setText("operator test")  # type: ignore[attr-defined]
    bar._on_submit_clicked()  # type: ignore[attr-defined]
    assert notes == ["operator test"]


def test_stimulus_requested_suppressed_when_disabled() -> None:
    bar = ActionBar()
    # No session context — button should be disabled, click suppressed.
    notes: list[str] = []
    bar.stimulus_requested.connect(notes.append)
    bar._note_input.setText("hi")  # type: ignore[attr-defined]
    bar._on_submit_clicked()  # type: ignore[attr-defined]
    assert notes == []


def test_countdown_visibility() -> None:
    bar = ActionBar()
    bar.set_countdown_remaining(None)
    assert bar._countdown_label.isHidden() is True  # type: ignore[attr-defined]

    bar.set_countdown_remaining(75)
    assert bar._countdown_label.isHidden() is False  # type: ignore[attr-defined]
    assert bar._countdown_label.text() == "01:15 left"  # type: ignore[attr-defined]

    bar.set_countdown_remaining(0)
    assert bar._countdown_label.isHidden() is True  # type: ignore[attr-defined]


def test_compact_mode_moves_submit_below_note_input() -> None:
    bar = ActionBar()
    session_id = uuid4()
    bar.set_session_context(session_id, "arm1", "hello")

    bar.set_compact_mode(True)

    assert bar._compact_mode is True  # type: ignore[attr-defined]
    assert (
        bar._session_label.text()
        == (  # type: ignore[attr-defined]
            f"Session {str(session_id)[:8]}…{str(session_id)[-4:]} · stimulus strategy arm1"
        )
    )
    assert bar._greeting_label.text() == "Expected response: “hello”"  # type: ignore[attr-defined]
    assert bar._input_layout.count() >= 1  # type: ignore[attr-defined]
    assert _layout_widget(bar._input_layout, 0) is bar._submit_button  # type: ignore[attr-defined]


def test_compact_mode_round_trips_to_wide_layout() -> None:
    bar = ActionBar()
    bar.set_session_context(uuid4(), "arm1", "hello")

    bar.set_compact_mode(True)
    bar.set_compact_mode(False)

    assert bar._compact_mode is False  # type: ignore[attr-defined]
    assert _layout_widget(bar._input_layout, 0) is bar._note_input  # type: ignore[attr-defined]
    assert _layout_widget(bar._input_layout, 1) is bar._submit_button  # type: ignore[attr-defined]
