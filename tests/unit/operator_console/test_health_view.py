"""Tests for `HealthView` — Phase 10.

Locks the §12 distinction the operator must not lose:
  * `DEGRADED` → WARN pill
  * `RECOVERING` → PROGRESS pill (self-healing in flight, not an error)
  * `ERROR` → ERROR pill
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from packages.schemas.operator_console import (
    AlertEvent,
    AlertKind,
    AlertSeverity,
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
    assert view._body_container.isHidden() is True


def test_health_view_repair_button_tracks_action_binding() -> None:
    view, vm, store = _view_with_vm()
    assert view._repair_button.isEnabled() is False
    vm.bind_repair_action(lambda: OneShotSignals())
    store.set_health(_snapshot(HealthState.OK))

    assert view._repair_button.isEnabled() is True


def test_health_view_repair_click_invokes_bound_action() -> None:
    view, vm, store = _view_with_vm()
    calls: list[str] = []

    def request_repair() -> OneShotSignals:
        calls.append("repair")
        return OneShotSignals()

    vm.bind_repair_action(request_repair)
    store.set_health(_snapshot(HealthState.OK))

    view._repair_button.click()
    assert calls == ["repair"]
    assert view._repair_status.isHidden() is False
    assert view._repair_status.text() == "Repair requested"


def test_health_view_renders_ok_snapshot() -> None:
    view, store = _view()
    store.set_health(_snapshot(HealthState.OK))
    assert view._overall_card._status._kind is UiStatusKind.OK
    assert view._degraded_card._primary.text() == "0"
    assert view._recovering_card._primary.text() == "0"
    assert view._error_card._primary.text() == "0"


def test_health_view_probe_matrix_renders_not_configured_as_neutral() -> None:
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
    assert pill.kind() is UiStatusKind.NEUTRAL
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
                    subsystem_key="worker",
                    label="Worker",
                    state=HealthState.ERROR,
                    operator_action_hint="restart worker container",
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
                message="Redis broker degraded",
                emitted_at_utc=_NOW,
            ),
        ]
    )
    # Model now holds the row; the timeline widget exists.
    assert view._vm.alerts_model().rowCount() == 1
