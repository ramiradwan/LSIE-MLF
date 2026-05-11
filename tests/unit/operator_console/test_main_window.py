"""Tests for `MainWindow` and `app.build_*` factories — Phase 6.

Covers the bits the shell has to get right:
  * factories compose the store/coordinator/window in the documented
    order without side effects (no polling kicked off);
  * nav clicks forward to the store's route;
  * store `route_changed` re-syncs the stack + sidebar check state;
  * ActionBar context updates when `selected_session_id` + live session
    DTO are both present and identify the same session;
  * `closeEvent` stops the coordinator (drain polling threads);
  * stimulus request path transitions the store into SUBMITTING, dedup
    key is carried, and the ActionBar reacts to the state signal.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QLabel

from packages.schemas.operator_console import (
    ArmSummary,
    ExperimentDetail,
    SessionSummary,
    StimulusAccepted,
    StimulusActionState,
)
from services.operator_console.app import (
    build_main_window,
    build_polling_coordinator,
    build_store,
)
from services.operator_console.config import OperatorConsoleConfig
from services.operator_console.polling import PollingCoordinator
from services.operator_console.state import AppRoute, OperatorStore, StimulusUiContext
from services.operator_console.views.main_window import MainWindow
from services.operator_console.workers import OneShotSignals

pytestmark = pytest.mark.usefixtures("qt_app")


def _make_config() -> OperatorConsoleConfig:
    return OperatorConsoleConfig(
        api_base_url="http://localhost:8000",
        api_timeout_seconds=5.0,
        environment_label="test",
        overview_poll_ms=1000,
        session_header_poll_ms=1000,
        live_encounters_poll_ms=1000,
        experiments_poll_ms=1000,
        physiology_poll_ms=1000,
        comodulation_poll_ms=1000,
        health_poll_ms=1000,
        alerts_poll_ms=1000,
        sessions_poll_ms=1000,
        default_experiment_id="greeting_line_v1",
    )


def _make_window(
    coordinator: PollingCoordinator | None = None,
) -> tuple[MainWindow, OperatorStore, PollingCoordinator]:
    config = _make_config()
    store = build_store()
    coord: PollingCoordinator
    if coordinator is None:
        # Build a real coordinator; we never call .start() so no threads
        # are spun up, but the QObject + signal wiring is exercised.
        client = MagicMock()
        coord = build_polling_coordinator(config, client, store)
    else:
        coord = coordinator
    window = build_main_window(config, store, coord)
    return window, store, coord


def _make_session(
    session_id: UUID,
    *,
    active_arm: str | None = "greeting_v1",
    expected_greeting: str | None = "hei rakas",
    is_calibrating: bool | None = None,
    calibration_frames_accumulated: int | None = None,
    calibration_frames_required: int | None = None,
) -> SessionSummary:
    return SessionSummary(
        session_id=session_id,
        status="active",
        started_at_utc=datetime(2026, 4, 17, 12, 0, 0, tzinfo=UTC),
        active_arm=active_arm,
        expected_greeting=expected_greeting,
        is_calibrating=is_calibrating,
        calibration_frames_accumulated=calibration_frames_accumulated,
        calibration_frames_required=calibration_frames_required,
    )


# ---------------------------------------------------------------------
# Construction / factories
# ---------------------------------------------------------------------


def test_factories_compose_without_starting_polling() -> None:
    window, _store, coord = _make_window()
    # Coordinator is inert until .start() is called.
    assert not coord._started
    # Window wears the configured env label in its title.
    assert "test" in window.windowTitle()


def test_window_registers_six_pages_in_nav_order() -> None:
    window, _store, _coord = _make_window()
    pages = window._pages
    assert list(pages.keys()) == [
        AppRoute.OVERVIEW,
        AppRoute.LIVE_SESSION,
        AppRoute.EXPERIMENTS,
        AppRoute.PHYSIOLOGY,
        AppRoute.HEALTH,
        AppRoute.SESSIONS,
    ]
    stack = window._stack
    assert stack.count() == 6


def test_initial_route_is_live_session_and_action_bar_disabled() -> None:
    window, store, _coord = _make_window()
    assert store.route() == AppRoute.LIVE_SESSION
    assert window._stack.currentWidget() is window._pages[AppRoute.LIVE_SESSION]
    assert window._nav_buttons[AppRoute.LIVE_SESSION].isChecked() is True
    bar = window._action_bar
    # No session selected yet — submit disabled.
    assert bar._submit_button.isEnabled() is False


# ---------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------


def test_nav_click_forwards_to_store_route(qtbot_unused: None = None) -> None:  # noqa: ARG001
    window, store, _coord = _make_window()
    btn = window._nav_buttons[AppRoute.EXPERIMENTS]
    btn.click()
    assert store.route() == AppRoute.EXPERIMENTS


def test_sidebar_nav_buttons_expose_accessible_descriptions() -> None:
    window, _store, _coord = _make_window()
    live_button = window._nav_buttons[AppRoute.LIVE_SESSION]
    health_button = window._nav_buttons[AppRoute.HEALTH]

    assert live_button.accessibleName() == "Live Session"
    assert "stimulus workflow" in live_button.accessibleDescription()
    assert live_button.toolTip() == live_button.accessibleDescription()
    assert health_button.accessibleName() == "Health"
    assert "readiness checks" in health_button.accessibleDescription()


def test_sidebar_header_uses_registered_object_names() -> None:
    window, _store, _coord = _make_window()

    assert window.findChild(QLabel, "SidebarTitle") is not None
    assert window.findChild(QLabel, "SidebarSubtitle") is not None


def test_store_route_change_syncs_stack_and_sidebar() -> None:
    window, store, _coord = _make_window()
    store.set_route(AppRoute.PHYSIOLOGY)
    stack = window._stack
    expected_page = window._pages[AppRoute.PHYSIOLOGY]
    assert stack.currentWidget() is expected_page
    assert window._nav_buttons[AppRoute.PHYSIOLOGY].isChecked() is True


# ---------------------------------------------------------------------
# ActionBar context
# ---------------------------------------------------------------------


def test_action_bar_enables_when_session_and_live_match() -> None:
    window, store, _coord = _make_window()
    session_id = uuid4()
    # Selecting a session alone is not enough — live_session DTO must
    # describe that same session for arm/greeting to surface.
    store.set_selected_session_id(session_id)
    live = _make_session(session_id)
    store.set_live_session(live)

    bar = window._action_bar
    assert bar._submit_button.isEnabled() is True
    assert "greeting_v1" in bar._session_label.text()
    assert bar._greeting_label.isHidden() is False


def test_action_bar_disables_when_live_session_below_safe_submit_threshold() -> None:
    window, store, _coord = _make_window()
    session_id = uuid4()
    store.set_selected_session_id(session_id)
    store.set_live_session(
        _make_session(
            session_id,
            is_calibrating=True,
            calibration_frames_accumulated=12,
            calibration_frames_required=45,
        )
    )

    bar = window._action_bar
    assert bar._submit_button.isEnabled() is False


def test_action_bar_enables_at_safe_submit_threshold_while_lifecycle_calibrating() -> None:
    window, store, _coord = _make_window()
    session_id = uuid4()
    store.set_selected_session_id(session_id)
    store.set_live_session(
        _make_session(
            session_id,
            is_calibrating=True,
            calibration_frames_accumulated=45,
            calibration_frames_required=45,
        )
    )

    bar = window._action_bar
    assert bar._submit_button.isEnabled() is True


def test_action_bar_ignores_live_session_of_other_session() -> None:
    window, store, _coord = _make_window()
    selected = uuid4()
    other = uuid4()
    store.set_selected_session_id(selected)
    # A live_session DTO for a *different* session id must not leak its
    # arm/greeting into the action bar.
    store.set_live_session(_make_session(other))
    bar = window._action_bar
    # Button is still enabled (a session is selected) but the context
    # strings should not come from the mismatched live DTO.
    assert bar._active_arm is None
    assert bar._expected_greeting is None


# ---------------------------------------------------------------------
# Experiment management submit path
# ---------------------------------------------------------------------


def test_health_vm_actions_route_to_coordinator() -> None:
    config = _make_config()
    store = build_store()
    coord = build_polling_coordinator(config, MagicMock(), store)
    returned_signals = OneShotSignals()
    coord.repair_install = MagicMock(return_value=returned_signals)  # type: ignore[method-assign]
    coord.cloud_sign_in = MagicMock(return_value=OneShotSignals())  # type: ignore[method-assign]
    coord.refresh_experiment_bundle = MagicMock(return_value=OneShotSignals())  # type: ignore[method-assign]
    window = build_main_window(config, store, coord)

    assert window._health_vm.request_repair() is True
    assert window._health_vm.request_cloud_sign_in() is True
    assert window._health_vm.request_experiment_bundle_refresh() is True
    coord.repair_install.assert_called_once_with()
    coord.cloud_sign_in.assert_called_once_with()
    coord.refresh_experiment_bundle.assert_called_once_with()


def test_experiment_management_vm_signals_route_to_coordinator() -> None:
    config = _make_config()
    store = build_store()
    coord = build_polling_coordinator(config, MagicMock(), store)
    coord.create_experiment = MagicMock()  # type: ignore[method-assign]
    coord.add_experiment_arm = MagicMock()  # type: ignore[method-assign]
    coord.rename_experiment_arm = MagicMock()  # type: ignore[method-assign]
    coord.disable_experiment_arm = MagicMock()  # type: ignore[method-assign]
    window = build_main_window(config, store, coord)
    vm = window._experiments_vm

    assert vm.create_experiment("exp-new", "Greeting v2", "arm-a", "Hei") is True
    coord.create_experiment.assert_called_once()
    create_request = coord.create_experiment.call_args.args[0]
    assert create_request.experiment_id == "exp-new"

    store.set_experiment(
        ExperimentDetail(
            experiment_id="exp-new",
            arms=[
                ArmSummary(
                    arm_id="arm-a",
                    greeting_text="Hei",
                    posterior_alpha=1.0,
                    posterior_beta=1.0,
                )
            ],
        )
    )
    assert vm.add_arm("arm-b", "Moi") is True
    assert vm.rename_arm_greeting("arm-a", "Hei uusi") is True
    assert vm.disable_arm("arm-a") is True

    coord.add_experiment_arm.assert_called_once()
    add_experiment_id, add_request = coord.add_experiment_arm.call_args.args
    assert add_experiment_id == "exp-new"
    assert add_request.arm == "arm-b"
    coord.rename_experiment_arm.assert_called_once_with("exp-new", "arm-a", "Hei uusi")
    coord.disable_experiment_arm.assert_called_once_with("exp-new", "arm-a")


# ---------------------------------------------------------------------
# Stimulus submit path
# ---------------------------------------------------------------------


def test_stimulus_requested_transitions_store_to_submitting() -> None:
    window, store, coord = _make_window()
    # Stub out `submit_stimulus` so no thread is dispatched.
    returned_signals = OneShotSignals()
    coord.submit_stimulus = MagicMock(return_value=returned_signals)  # type: ignore[method-assign]

    session_id = uuid4()
    store.set_selected_session_id(session_id)
    store.set_live_session(_make_session(session_id))

    window._on_stimulus_requested("operator test note")

    ctx = store.stimulus_ui_context()
    assert ctx.state == StimulusActionState.SUBMITTING
    assert ctx.client_action_id is not None
    assert ctx.operator_note == "operator test note"
    # Coordinator was called with the selected session id and a real
    # StimulusRequest carrying the dedup key.
    coord.submit_stimulus.assert_called_once()
    call_session_id, call_request = coord.submit_stimulus.call_args.args
    assert call_session_id == session_id
    assert call_request.client_action_id == ctx.client_action_id


def test_stimulus_state_changed_reaches_action_bar() -> None:
    window, store, _coord = _make_window()
    session_id = uuid4()
    store.set_selected_session_id(session_id)
    store.set_live_session(_make_session(session_id))

    store.set_stimulus_ui_context(
        StimulusUiContext(
            state=StimulusActionState.MEASURING,
            authoritative_stimulus_time_utc=datetime.now(UTC),
        )
    )
    bar = window._action_bar
    assert bar._submit_button.text() == "Measuring…"
    assert bar._submit_button.isEnabled() is False
    assert bar._countdown_label.isHidden() is False
    assert "response window" in bar._message_label.text().lower()


def test_stimulus_success_with_time_starts_action_bar_countdown() -> None:
    window, store, _coord = _make_window()
    session_id = uuid4()
    action_id = uuid4()
    stimulus_time = datetime.now(UTC)
    store.set_selected_session_id(session_id)
    store.set_live_session(_make_session(session_id))

    window._on_stimulus_succeeded(
        "stimulus",
        StimulusAccepted(
            session_id=session_id,
            client_action_id=action_id,
            accepted=True,
            received_at_utc=stimulus_time,
            stimulus_time_utc=stimulus_time,
            message="Test message accepted.",
        ),
    )

    ctx = store.stimulus_ui_context()
    bar = window._action_bar
    assert ctx.state == StimulusActionState.MEASURING
    assert ctx.authoritative_stimulus_time_utc == stimulus_time
    assert bar._submit_button.text() == "Measuring…"
    assert bar._countdown_label.isHidden() is False


def test_stimulus_success_without_time_keeps_action_bar_accepted() -> None:
    window, store, _coord = _make_window()
    session_id = uuid4()
    action_id = uuid4()
    accepted_at = datetime.now(UTC)
    store.set_selected_session_id(session_id)
    store.set_live_session(_make_session(session_id))

    window._on_stimulus_succeeded(
        "stimulus",
        StimulusAccepted(
            session_id=session_id,
            client_action_id=action_id,
            accepted=True,
            received_at_utc=accepted_at,
            stimulus_time_utc=None,
            message="Accepted and waiting for phone timing.",
        ),
    )

    ctx = store.stimulus_ui_context()
    bar = window._action_bar
    assert ctx.state == StimulusActionState.ACCEPTED
    assert ctx.authoritative_stimulus_time_utc is None
    assert bar._submit_button.text() == "Accepted"
    assert bar._countdown_label.isHidden() is True


def test_stimulus_success_rejected_marks_action_bar_failed() -> None:
    window, store, _coord = _make_window()
    session_id = uuid4()
    action_id = uuid4()
    rejected_at = datetime.now(UTC)
    store.set_selected_session_id(session_id)
    store.set_live_session(_make_session(session_id))

    window._on_stimulus_succeeded(
        "stimulus",
        StimulusAccepted(
            session_id=session_id,
            client_action_id=action_id,
            accepted=False,
            received_at_utc=rejected_at,
            stimulus_time_utc=None,
            message="Phone did not accept the stimulus.",
        ),
    )

    ctx = store.stimulus_ui_context()
    bar = window._action_bar
    assert ctx.state == StimulusActionState.FAILED
    assert ctx.message == "Phone did not accept the stimulus."
    assert bar._submit_button.text() == "Retry"
    assert bar._countdown_label.isHidden() is True


# ---------------------------------------------------------------------
# Close / shutdown
# ---------------------------------------------------------------------


def test_window_sets_supported_minimum_size() -> None:
    window, _store, _coord = _make_window()
    assert window.minimumWidth() == 900
    assert window.minimumHeight() == 640


def test_sidebar_width_reflows_with_window_resize() -> None:
    window, _store, _coord = _make_window()

    sidebar = window._sidebar
    assert sidebar is not None

    window.resize(1280, 800)
    window._update_responsive_layout(window.width())
    wide_width = sidebar.width()

    window.resize(900, 640)
    window._update_responsive_layout(window.width())
    narrow_width = sidebar.width()

    assert wide_width == 220
    assert narrow_width == 162


def test_action_bar_switches_to_compact_mode_on_narrow_window() -> None:
    window, _store, _coord = _make_window()

    window.resize(900, 640)
    window._update_responsive_layout(window.width())
    assert window._action_bar._compact_mode is True

    window.resize(1280, 800)
    window._update_responsive_layout(window.width())
    assert window._action_bar._compact_mode is False


def test_close_event_stops_coordinator() -> None:
    window, _store, coord = _make_window()
    coord.stop = MagicMock()  # type: ignore[method-assign]
    event = QCloseEvent()
    window.closeEvent(event)
    coord.stop.assert_called_once()
    assert event.isAccepted()


def test_close_event_calls_shutdown_on_scaffold_pages() -> None:
    window, _store, _coord = _make_window()
    # Swap one page for a stub exposing shutdown() to confirm the
    # close handler still honours legacy scaffold pages.
    shutdown_called = {"called": False}

    class _ScaffoldPage:
        def shutdown(self) -> None:
            shutdown_called["called"] = True

    pages = cast(dict[AppRoute, object], window._pages)
    pages[AppRoute.SESSIONS] = _ScaffoldPage()

    window.closeEvent(QCloseEvent())
    assert shutdown_called["called"] is True
