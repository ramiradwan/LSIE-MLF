"""Health page viewmodel.

Owns two tables: the subsystem rollup (§12 states with recovery-mode
and operator-action hints) and the alerts feed. Keeping them separate
is deliberate — §12 transitions emit both a subsystem state change and
an alert, and the operator reads them at different times (the rollup
is the "right now" snapshot, the feed is the audit trail).

`degraded_count` is a convenience accessor used by the shell's
navigation badge: if non-zero, the sidebar surfaces a small indicator.

The VM also accumulates a per-subsystem rolling history of the last
`_PROBE_HISTORY_CAPACITY` probe samples so the Health view can render a
flapping-aware sparkline without the API needing to send series data.
The buffer is in-memory and resets on app restart — no persistence,
no schema change.

Spec references:
  §4.E.1         — Health operator surface
  §12            — error-handling matrix, including
                   degraded-but-recovering paths
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

from PySide6.QtCore import QObject, Signal

from packages.schemas.operator_console import (
    CloudAuthStatus,
    CloudOutboxSummary,
    ExperimentBundleRefreshPreview,
    ExperimentBundleRefreshRequest,
    HealthProbeState,
    HealthSnapshot,
    HealthSubsystemProbe,
    UiStatusKind,
)
from services.operator_console.polling import (
    JOB_CLOUD_SIGN_IN,
    JOB_EXPERIMENT_BUNDLE_REFRESH,
    JOB_REPAIR_INSTALL,
)
from services.operator_console.state import OperatorStore
from services.operator_console.table_models.alerts_table_model import AlertsTableModel
from services.operator_console.table_models.health_table_model import HealthTableModel
from services.operator_console.viewmodels.base import ViewModelBase
from services.operator_console.widgets.probe_sparkline import ProbeSparklineCell
from services.operator_console.workers import OneShotSignals

_PROBE_ORDER: tuple[str, ...] = (
    "postgres",
    "redis",
    "azure_openai",
    "whisper_worker",
    "orchestrator",
)
_PROBE_HISTORY_CAPACITY = 60

_PROBE_UI_STATUS: dict[HealthProbeState, UiStatusKind] = {
    HealthProbeState.OK: UiStatusKind.OK,
    HealthProbeState.ERROR: UiStatusKind.ERROR,
    HealthProbeState.TIMEOUT: UiStatusKind.ERROR,
    HealthProbeState.NOT_CONFIGURED: UiStatusKind.MUTED,
    HealthProbeState.UNKNOWN: UiStatusKind.NEUTRAL,
}


HealthAction = Callable[[], OneShotSignals]
ExperimentBundleRefreshAction = Callable[[ExperimentBundleRefreshRequest], OneShotSignals]
HealthActionState = Literal[
    "idle",
    "repair_install_progress",
    "repair_install_success",
    "repair_install_failure",
    "cloud_sign_in_progress",
    "cloud_sign_in_success",
    "cloud_sign_in_failure",
    "experiment_bundle_refresh_progress",
    "experiment_bundle_refresh_success",
    "experiment_bundle_refresh_failure",
]


class HealthViewModel(ViewModelBase):
    """Owns the subsystem table + alerts feed."""

    repair_requested = Signal()
    cloud_sign_in_requested = Signal()
    experiment_bundle_refresh_preview_requested = Signal()
    experiment_bundle_refresh_preview_ready = Signal(object)
    experiment_bundle_refresh_requested = Signal()

    def __init__(
        self,
        store: OperatorStore,
        model: HealthTableModel,
        alerts_model: AlertsTableModel,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(store, parent)
        self._health_model = model
        self._alerts_model = alerts_model
        self._repair_action: HealthAction | None = None
        self._cloud_sign_in_action: HealthAction | None = None
        self._experiment_bundle_refresh_preview_action: HealthAction | None = None
        self._experiment_bundle_refresh_action: ExperimentBundleRefreshAction | None = None
        self._probe_history: dict[str, deque[ProbeSparklineCell]] = {}
        self._repair_in_progress = False
        self._cloud_sign_in_in_progress = False
        self._experiment_bundle_refresh_preview_in_progress = False
        self._experiment_bundle_refresh_in_progress = False
        self._action_state: HealthActionState = "idle"
        store.health_changed.connect(self._on_health_changed)
        store.alerts_changed.connect(self._on_alerts_changed)
        store.cloud_auth_changed.connect(self._on_cloud_readback_changed)
        store.cloud_outbox_changed.connect(self._on_cloud_readback_changed)
        store.error_changed.connect(self._on_error)
        store.error_cleared.connect(self._on_error_cleared)
        # Seed from whatever the store already holds.
        self._sync_health_rows(self._store.health())
        self._record_probe_history(self._store.health())
        self._alerts_model.set_rows(self._store.alerts())

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def snapshot(self) -> HealthSnapshot | None:
        return self._store.health()

    def cloud_auth_status(self) -> CloudAuthStatus | None:
        return self._store.cloud_auth_status()

    def cloud_outbox_summary(self) -> CloudOutboxSummary | None:
        return self._store.cloud_outbox_summary()

    def health_model(self) -> HealthTableModel:
        return self._health_model

    def alerts_model(self) -> AlertsTableModel:
        return self._alerts_model

    def subsystem_probes(self) -> list[HealthSubsystemProbe]:
        snap = self._store.health()
        if snap is None:
            return []
        probes = snap.subsystem_probes
        ordered = [probes[key] for key in _PROBE_ORDER if key in probes]
        ordered.extend(probe for key, probe in sorted(probes.items()) if key not in _PROBE_ORDER)
        return ordered

    def probe_history(self, subsystem_key: str) -> tuple[ProbeSparklineCell, ...]:
        """Return the last 60 samples observed for one subsystem.

        Empty when the page has not yet seen a probe for that key. The
        Health view pads the sparkline to its capacity, so callers do
        not need to add filler.
        """

        history = self._probe_history.get(subsystem_key)
        if history is None:
            return ()
        return tuple(history)

    def probe_history_capacity(self) -> int:
        return _PROBE_HISTORY_CAPACITY

    def bind_repair_action(self, action: HealthAction) -> None:
        self._repair_action = action

    def bind_cloud_sign_in_action(self, action: HealthAction) -> None:
        self._cloud_sign_in_action = action

    def bind_experiment_bundle_refresh_preview_action(self, action: HealthAction) -> None:
        self._experiment_bundle_refresh_preview_action = action

    def bind_experiment_bundle_refresh_action(
        self,
        action: ExperimentBundleRefreshAction,
    ) -> None:
        self._experiment_bundle_refresh_action = action

    def repair_available(self) -> bool:
        return self._repair_action is not None and not self._repair_in_progress

    def repair_in_progress(self) -> bool:
        return self._repair_in_progress

    def cloud_sign_in_available(self) -> bool:
        return self._cloud_sign_in_action is not None and not self._cloud_sign_in_in_progress

    def cloud_sign_in_in_progress(self) -> bool:
        return self._cloud_sign_in_in_progress

    def experiment_bundle_refresh_available(self) -> bool:
        return (
            self._experiment_bundle_refresh_action is not None
            and not self._experiment_bundle_refresh_in_progress
            and not self._experiment_bundle_refresh_preview_in_progress
        )

    def experiment_bundle_refresh_in_progress(self) -> bool:
        return self._experiment_bundle_refresh_in_progress

    def experiment_bundle_refresh_preview_available(self) -> bool:
        return (
            self._experiment_bundle_refresh_preview_action is not None
            and self.experiment_bundle_refresh_available()
        )

    def experiment_bundle_refresh_preview_in_progress(self) -> bool:
        return self._experiment_bundle_refresh_preview_in_progress

    def clear_experiment_bundle_refresh_preview(self) -> None:
        if self._action_state == "experiment_bundle_refresh_progress":
            self._action_state = "idle"
        self.emit_changed()

    def request_repair(self) -> bool:
        if self._repair_action is None or self._repair_in_progress:
            return False
        self.set_error(None)
        self._repair_in_progress = True
        self._action_state = "repair_install_progress"
        self.repair_requested.emit()
        signals = self._repair_action()
        signals.succeeded.connect(self._on_repair_succeeded)
        signals.failed.connect(self._on_repair_failed)
        signals.finished.connect(self._on_repair_finished)
        self.emit_changed()
        return True

    def request_cloud_sign_in(self) -> bool:
        if self._cloud_sign_in_action is None or self._cloud_sign_in_in_progress:
            return False
        self.set_error(None)
        self._cloud_sign_in_in_progress = True
        self._action_state = "cloud_sign_in_progress"
        self.cloud_sign_in_requested.emit()
        signals = self._cloud_sign_in_action()
        signals.succeeded.connect(self._on_cloud_sign_in_succeeded)
        signals.failed.connect(self._on_cloud_sign_in_failed)
        signals.finished.connect(self._on_cloud_sign_in_finished)
        self.emit_changed()
        return True

    def request_experiment_bundle_refresh_preview(self) -> bool:
        if (
            self._experiment_bundle_refresh_preview_action is None
            or self._experiment_bundle_refresh_preview_in_progress
            or self._experiment_bundle_refresh_in_progress
        ):
            return False
        self.set_error(None)
        self._experiment_bundle_refresh_preview_in_progress = True
        self._action_state = "experiment_bundle_refresh_progress"
        self.experiment_bundle_refresh_preview_requested.emit()
        signals = self._experiment_bundle_refresh_preview_action()
        signals.succeeded.connect(self._on_experiment_bundle_refresh_preview_succeeded)
        signals.failed.connect(self._on_experiment_bundle_refresh_preview_failed)
        signals.finished.connect(self._on_experiment_bundle_refresh_preview_finished)
        self.emit_changed()
        return True

    def request_experiment_bundle_refresh(
        self,
        request: ExperimentBundleRefreshRequest,
    ) -> bool:
        if (
            self._experiment_bundle_refresh_action is None
            or self._experiment_bundle_refresh_in_progress
            or self._experiment_bundle_refresh_preview_in_progress
        ):
            return False
        self.set_error(None)
        self._experiment_bundle_refresh_in_progress = True
        self._action_state = "experiment_bundle_refresh_progress"
        self.experiment_bundle_refresh_requested.emit()
        signals = self._experiment_bundle_refresh_action(request)
        signals.succeeded.connect(self._on_experiment_bundle_refresh_succeeded)
        signals.failed.connect(self._on_experiment_bundle_refresh_failed)
        signals.finished.connect(self._on_experiment_bundle_refresh_finished)
        self.emit_changed()
        return True

    def degraded_count(self) -> int:
        """Count of subsystems currently DEGRADED or RECOVERING.

        §12 draws a line between degraded-but-recovering and outright
        error; the navigation badge surfaces both so the operator sees
        "something needs attention" even when no subsystem is in hard
        error state. The snapshot's `degraded_count` already excludes
        errors by construction, so we add the recovering count here.
        """
        snap = self._store.health()
        if snap is None:
            return 0
        return snap.degraded_count + snap.recovering_count

    def action_state(self) -> HealthActionState:
        return self._action_state

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_health_changed(self, snap: object) -> None:
        if isinstance(snap, HealthSnapshot) or snap is None:
            resolved = snap if isinstance(snap, HealthSnapshot) else None
        else:
            resolved = self._store.health()
        self._sync_health_rows(resolved)
        self._record_probe_history(resolved)
        self.emit_changed()

    def _on_alerts_changed(self, rows: object) -> None:
        if isinstance(rows, list):
            self._alerts_model.set_rows(rows)
        else:
            self._alerts_model.set_rows(self._store.alerts())
        self.emit_changed()

    def _on_cloud_readback_changed(self, _payload: object) -> None:
        self.emit_changed()

    def _sync_health_rows(self, snap: HealthSnapshot | None) -> None:
        if snap is None:
            self._health_model.set_rows([])
        else:
            self._health_model.set_rows(list(snap.subsystems))

    def _record_probe_history(self, snap: HealthSnapshot | None) -> None:
        if snap is None:
            return
        timestamp = snap.generated_at_utc or datetime.now(UTC)
        for key, probe in snap.subsystem_probes.items():
            buffer = self._probe_history.setdefault(key, deque(maxlen=_PROBE_HISTORY_CAPACITY))
            buffer.append(
                ProbeSparklineCell(
                    timestamp_utc=timestamp,
                    state=_PROBE_UI_STATUS.get(probe.state, UiStatusKind.NEUTRAL),
                    probe_state=probe.state,
                )
            )

    def _on_repair_succeeded(self, _job: str, _payload: object) -> None:
        self._action_state = "repair_install_success"
        self.set_error(None)
        self.emit_changed()

    def _on_repair_failed(self, _job: str, _error: object) -> None:
        self._action_state = "repair_install_failure"
        self.emit_changed()

    def _on_repair_finished(self, _job: str) -> None:
        self._repair_in_progress = False
        self.emit_changed()

    def _on_cloud_sign_in_succeeded(self, _job: str, _payload: object) -> None:
        self._action_state = "cloud_sign_in_success"
        self.set_error(None)
        self.emit_changed()

    def _on_cloud_sign_in_failed(self, _job: str, _error: object) -> None:
        self._action_state = "cloud_sign_in_failure"
        self.emit_changed()

    def _on_cloud_sign_in_finished(self, _job: str) -> None:
        self._cloud_sign_in_in_progress = False
        self.emit_changed()

    def _on_experiment_bundle_refresh_preview_succeeded(
        self,
        _job: str,
        payload: object,
    ) -> None:
        self.set_error(None)
        self._experiment_bundle_refresh_preview_in_progress = False
        if isinstance(payload, ExperimentBundleRefreshPreview):
            self.experiment_bundle_refresh_preview_ready.emit(payload)
        self.emit_changed()

    def _on_experiment_bundle_refresh_preview_failed(self, _job: str, _error: object) -> None:
        self._action_state = "experiment_bundle_refresh_failure"
        self.emit_changed()

    def _on_experiment_bundle_refresh_preview_finished(self, _job: str) -> None:
        self._experiment_bundle_refresh_preview_in_progress = False
        self.emit_changed()

    def _on_experiment_bundle_refresh_succeeded(self, _job: str, _payload: object) -> None:
        self._action_state = "experiment_bundle_refresh_success"
        self.set_error(None)
        self.emit_changed()

    def _on_experiment_bundle_refresh_failed(self, _job: str, _error: object) -> None:
        self._action_state = "experiment_bundle_refresh_failure"
        self.emit_changed()

    def _on_experiment_bundle_refresh_finished(self, _job: str) -> None:
        self._experiment_bundle_refresh_in_progress = False
        self.emit_changed()

    def _on_error(self, scope: str, message: str) -> None:
        if scope in (
            "health",
            "alerts",
            JOB_REPAIR_INSTALL,
            JOB_CLOUD_SIGN_IN,
            JOB_EXPERIMENT_BUNDLE_REFRESH,
        ):
            if scope == JOB_REPAIR_INSTALL:
                self._action_state = "repair_install_failure"
            elif scope == JOB_CLOUD_SIGN_IN:
                self._action_state = "cloud_sign_in_failure"
            elif scope == JOB_EXPERIMENT_BUNDLE_REFRESH:
                self._action_state = "experiment_bundle_refresh_failure"
            self.set_error(message)
            self.emit_changed()

    def _on_error_cleared(self, scope: str) -> None:
        if scope in (
            "health",
            "alerts",
            JOB_REPAIR_INSTALL,
            JOB_CLOUD_SIGN_IN,
            JOB_EXPERIMENT_BUNDLE_REFRESH,
        ):
            self.set_error(None)
