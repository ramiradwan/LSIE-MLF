"""Tests for `HealthView` — Phase 10.

Locks the §12 distinction the operator must not lose:
  * `DEGRADED` → WARN pill
  * `RECOVERING` → PROGRESS pill (self-healing in flight, not an error)
  * `ERROR` → ERROR pill
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from PySide6.QtWidgets import QHeaderView

from packages.schemas.operator_console import (
    AlertEvent,
    AlertKind,
    AlertSeverity,
    CloudAuthState,
    CloudAuthStatus,
    CloudOutboxSummary,
    HealthProbeState,
    HealthSnapshot,
    HealthState,
    HealthSubsystemProbe,
    HealthSubsystemStatus,
    UiStatusKind,
)
from services.operator_console.state import OperatorStore
from services.operator_console.table_models.alerts_table_model import AlertsTableModel
from services.operator_console.table_models.health_table_model import HealthTableModel
from services.operator_console.viewmodels.health_vm import HealthViewModel
from services.operator_console.views.health_view import HealthView
from services.operator_console.workers import OneShotSignals

pytestmark = pytest.mark.usefixtures("qt_app")


_NOW = datetime(2026, 4, 17, 12, 0, 0, tzinfo=UTC)


def _view() -> tuple[HealthView, OperatorStore]:
    store = OperatorStore()
    health_model = HealthTableModel()
    alerts_model = AlertsTableModel()
    vm = HealthViewModel(store, health_model, alerts_model)
    return HealthView(vm), store


def _view_with_vm() -> tuple[HealthView, HealthViewModel, OperatorStore]:
    store = OperatorStore()
    health_model = HealthTableModel()
    alerts_model = AlertsTableModel()
    vm = HealthViewModel(store, health_model, alerts_model)
    return HealthView(vm), vm, store


def _snapshot(
    overall: HealthState,
    *,
    degraded: int = 0,
    recovering: int = 0,
    errors: int = 0,
    subsystems: list[HealthSubsystemStatus] | None = None,
    subsystem_probes: list[HealthSubsystemProbe] | None = None,
) -> HealthSnapshot:
    probes = {probe.subsystem_key: probe for probe in subsystem_probes or []}
    return HealthSnapshot(
        generated_at_utc=_NOW,
        overall_state=overall,
        subsystems=subsystems or [],
        subsystem_probes=probes,
        degraded_count=degraded,
        recovering_count=recovering,
        error_count=errors,
    )


def test_health_view_empty_until_snapshot_set() -> None:
    view, _store = _view()
    assert view._empty_state.isHidden() is False
    assert view._scroll.isHidden() is True
    assert "first readiness update" in view._empty_state._message.text().lower()


def test_health_view_header_buttons_track_action_bindings() -> None:
    view, vm, store = _view_with_vm()
    assert view._repair_button.isEnabled() is False
    assert view._cloud_sign_in_button.isEnabled() is False
    assert view._experiment_bundle_button.isEnabled() is False
    assert view._repair_button.accessibleName() == "Repair install"
    assert "desktop.sqlite" in view._repair_button.accessibleDescription()
    assert view._cloud_sign_in_button.accessibleName() == "Cloud sign-in"
    assert "cloud sync" in view._cloud_sign_in_button.accessibleDescription()
    assert view._experiment_bundle_button.accessibleName() == "Refresh experiments"
    assert "signed experiment bundle" in view._experiment_bundle_button.accessibleDescription()
    assert view._action_status.accessibleName() == "Health action status"
    vm.bind_repair_action(lambda: OneShotSignals())
    vm.bind_cloud_sign_in_action(lambda: OneShotSignals())
    vm.bind_experiment_bundle_refresh_action(lambda: OneShotSignals())
    store.set_health(_snapshot(HealthState.OK))

    assert view._repair_button.isEnabled() is True
    assert view._cloud_sign_in_button.isEnabled() is True
    assert view._experiment_bundle_button.isEnabled() is True


def test_health_view_repair_click_invokes_bound_action(monkeypatch: pytest.MonkeyPatch) -> None:
    view, vm, store = _view_with_vm()
    calls: list[str] = []

    def request_repair() -> OneShotSignals:
        calls.append("repair")
        return OneShotSignals()

    vm.bind_repair_action(request_repair)
    store.set_health(_snapshot(HealthState.OK))

    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "exec", lambda self: QMessageBox.StandardButton.Yes)

    view._repair_button.click()
    assert calls == ["repair"]
    assert view._action_status.isHidden() is False
    assert view._action_status.text() == "Installing runtime…"
    assert "Repair install in progress" in view._action_status.accessibleDescription()


def test_health_view_cloud_buttons_invoke_bound_actions() -> None:
    view, vm, store = _view_with_vm()
    calls: list[str] = []
    sign_in_signals = OneShotSignals()
    refresh_signals = OneShotSignals()

    def request_sign_in() -> OneShotSignals:
        calls.append("sign-in")
        return sign_in_signals

    def request_refresh() -> OneShotSignals:
        calls.append("refresh")
        return refresh_signals

    vm.bind_cloud_sign_in_action(request_sign_in)
    vm.bind_experiment_bundle_refresh_action(request_refresh)
    store.set_health(_snapshot(HealthState.OK))

    view._cloud_sign_in_button.click()
    assert calls == ["sign-in"]
    assert view._action_status.text() == "Waiting for sign-in…"
    assert "Browser sign-in" in view._action_status.accessibleDescription()

    sign_in_signals.succeeded.emit("cloud_sign_in", object())
    sign_in_signals.finished.emit("cloud_sign_in")
    assert view._action_status.text() == "Cloud sign-in completed"

    view._experiment_bundle_button.click()
    assert calls == ["sign-in", "refresh"]
    assert view._action_status.text() == "Refreshing experiments…"
    assert "Experiment bundle refresh" in view._action_status.accessibleDescription()


def test_health_view_hides_inline_status_when_cloud_sign_in_fails() -> None:
    view, vm, store = _view_with_vm()
    signals = OneShotSignals()
    vm.bind_cloud_sign_in_action(lambda: signals)
    store.set_health(_snapshot(HealthState.OK))

    view._cloud_sign_in_button.click()
    assert view._action_status.text() == "Waiting for sign-in…"

    store.set_error("cloud_sign_in", "browser closed")
    signals.failed.emit("cloud_sign_in", object())
    signals.finished.emit("cloud_sign_in")

    assert view._action_status.isHidden() is True
    assert view._error_banner.isHidden() is False
    assert view._error_banner._message.text() == "browser closed"


def test_health_view_hides_inline_status_when_refresh_prerequisite_fails() -> None:
    view, vm, store = _view_with_vm()
    signals = OneShotSignals()
    vm.bind_experiment_bundle_refresh_action(lambda: signals)
    store.set_health(_snapshot(HealthState.OK))

    view._experiment_bundle_button.click()
    assert view._action_status.text() == "Refreshing experiments…"

    store.set_error(
        "experiment_bundle_refresh",
        "cloud sign-in is required before refreshing experiments",
    )
    signals.failed.emit("experiment_bundle_refresh", object())
    signals.finished.emit("experiment_bundle_refresh")

    assert view._action_status.isHidden() is True
    assert view._error_banner.isHidden() is False
    assert view._error_banner._message.text() == (
        "Cloud sign-in is required before refreshing experiments."
    )


def test_health_view_renders_ok_snapshot() -> None:
    view, store = _view()
    store.set_health(_snapshot(HealthState.OK))
    assert view._overall_card._status._kind is UiStatusKind.OK
    assert view._degraded_card._primary.text() == "0"
    assert view._recovering_card._primary.text() == "0"
    assert view._error_card._primary.text() == "0"


def test_health_view_renders_cloud_auth_and_outbox_readbacks() -> None:
    view, store = _view()
    store.set_health(_snapshot(HealthState.OK))
    store.set_cloud_auth_status(
        CloudAuthStatus(
            state=CloudAuthState.SIGNED_IN,
            checked_at_utc=_NOW,
            message="Cloud sign-in is active.",
        )
    )
    store.set_cloud_outbox_summary(
        CloudOutboxSummary(
            generated_at_utc=_NOW,
            pending_count=2,
            in_flight_count=1,
            retry_scheduled_count=1,
            dead_letter_count=1,
            redacted_count=3,
        )
    )

    assert view._cloud_auth_card._primary.text() == "signed in"
    assert view._cloud_auth_card._secondary.text() == "Cloud sign-in is active."
    assert view._cloud_auth_card._status._kind is UiStatusKind.OK
    assert view._cloud_outbox_card._primary.text() == "4 active"
    assert view._cloud_outbox_card._secondary.text() == "1 dead-letter · 3 redacted"
    assert view._cloud_outbox_card._status._kind is UiStatusKind.ERROR


def test_health_view_probe_matrix_renders_not_configured_as_muted() -> None:
    view, store = _view()
    store.set_health(
        _snapshot(
            HealthState.OK,
            subsystem_probes=[
                HealthSubsystemProbe(
                    subsystem_key="azure_openai",
                    label="Azure OpenAI",
                    state=HealthProbeState.NOT_CONFIGURED,
                    latency_ms=None,
                    detail="missing AZURE_OPENAI_ENDPOINT",
                    checked_at_utc=_NOW,
                )
            ],
        )
    )
    pill = view._probe_matrix._state_pills[0]
    latency = view._probe_matrix._latency_labels[0]
    # UX-05: NOT_CONFIGURED maps to the explicit MUTED bucket so it
    # reads as "the system isn't expected to report here", visually
    # distinct from NEUTRAL ("nothing to say yet").
    assert pill.kind() is UiStatusKind.MUTED
    assert pill.text() == "not configured"
    assert latency.text() == "—"


def test_health_view_degraded_pill_is_warn_not_error() -> None:
    view, store = _view()
    store.set_health(_snapshot(HealthState.DEGRADED, degraded=2))
    assert view._overall_card._status._kind is UiStatusKind.WARN
    assert view._degraded_card._status._kind is UiStatusKind.WARN
    assert view._degraded_card._primary.text() == "2"


def test_health_view_recovering_pill_is_progress_not_warn() -> None:
    # §12: recovering is distinct from degraded — self-healing in flight.
    view, store = _view()
    store.set_health(_snapshot(HealthState.RECOVERING, recovering=1))
    assert view._overall_card._status._kind is UiStatusKind.PROGRESS
    assert view._recovering_card._status._kind is UiStatusKind.PROGRESS
    assert view._recovering_card._primary.text() == "1"


def test_health_view_error_card_surfaces_operator_action_hint() -> None:
    view, store = _view()
    store.set_health(
        _snapshot(
            HealthState.ERROR,
            errors=1,
            subsystems=[
                HealthSubsystemStatus(
                    subsystem_key="gpu_ml_worker",
                    label="GPU ML worker",
                    state=HealthState.ERROR,
                    operator_action_hint="restart local ML worker",
                ),
            ],
        )
    )
    assert view._error_card._status._kind is UiStatusKind.ERROR
    # The first error row's operator-action hint should be pulled through.
    assert "restart" in view._error_card._secondary.text()


def test_health_view_alerts_model_rows_insert_triggers_scroll() -> None:
    view, store = _view()
    store.set_health(_snapshot(HealthState.OK))
    # Append an alert — the view's rowsInserted hook should not raise.
    store.set_alerts(
        [
            AlertEvent(
                alert_id="a1",
                severity=AlertSeverity.WARNING,
                kind=AlertKind.SUBSYSTEM_DEGRADED,
                message="IPC queue degraded",
                emitted_at_utc=_NOW,
            ),
        ]
    )
    # Model now holds the row; the timeline widget exists.
    assert view._vm.alerts_model().rowCount() == 1


def test_health_view_uses_scroll_container_when_snapshot_present() -> None:
    view, store = _view()
    store.set_health(_snapshot(HealthState.OK))

    assert view._scroll.isHidden() is False
    assert view._scroll.widget() is view._body_container
    assert view._subsystem_table.accessibleName() == "Readiness details table"
    assert "next operator action" in view._subsystem_table.accessibleDescription()


def test_health_view_compacts_cards_and_tables_at_narrow_width() -> None:
    view, store = _view()
    store.set_health(
        _snapshot(
            HealthState.ERROR,
            degraded=1,
            recovering=1,
            errors=1,
            subsystems=[
                HealthSubsystemStatus(
                    subsystem_key="capture",
                    label="Capture",
                    state=HealthState.DEGRADED,
                    detail="ADB drift exceeded threshold for the active session.",
                    recovery_mode="drift-reset",
                    operator_action_hint="Confirm the phone stays attached during recovery.",
                )
            ],
            subsystem_probes=[
                HealthSubsystemProbe(
                    subsystem_key="azure_openai",
                    label="Azure OpenAI",
                    state=HealthProbeState.NOT_CONFIGURED,
                    latency_ms=None,
                    detail="missing AZURE_OPENAI_ENDPOINT",
                    checked_at_utc=_NOW,
                )
            ],
        )
    )

    view.resize(640, 900)
    view.show()
    view._apply_responsive_layout()

    assert view._cards_grid.column_count() == 1
    assert view._subsystem_table.isColumnHidden(2) is True
    assert view._subsystem_table.isColumnHidden(4) is True
    assert view._subsystem_table.isColumnHidden(5) is True
    assert (
        view._subsystem_table.horizontalHeader().sectionResizeMode(0)
        is QHeaderView.ResizeMode.ResizeToContents
    )
    assert view._alerts_timeline.current_width_band().value == "narrow"
    assert view._probe_panel._subtitle.isHidden() is True
    assert view._probe_matrix._header_labels[0].isHidden() is True
