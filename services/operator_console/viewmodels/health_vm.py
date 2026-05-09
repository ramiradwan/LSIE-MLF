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

from PySide6.QtCore import QObject, Signal

from packages.schemas.operator_console import (
    HealthProbeState,
    HealthSnapshot,
    HealthSubsystemProbe,
    UiStatusKind,
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


RepairAction = Callable[[], OneShotSignals]


class HealthViewModel(ViewModelBase):
    """Owns the subsystem table + alerts feed."""

    repair_requested = Signal()

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
        self._repair_action: RepairAction | None = None
        self._probe_history: dict[str, deque[ProbeSparklineCell]] = {}
        self._repair_in_progress = False
        store.health_changed.connect(self._on_health_changed)
        store.alerts_changed.connect(self._on_alerts_changed)
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

    def bind_repair_action(self, action: RepairAction) -> None:
        self._repair_action = action

    def repair_available(self) -> bool:
        return self._repair_action is not None and not self._repair_in_progress

    def repair_in_progress(self) -> bool:
        return self._repair_in_progress

    def request_repair(self) -> bool:
        if self._repair_action is None or self._repair_in_progress:
            return False
        self._repair_in_progress = True
        self.repair_requested.emit()
        signals = self._repair_action()
        signals.finished.connect(self._on_repair_finished)
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

    def _on_repair_finished(self, _job: str) -> None:
        self._repair_in_progress = False
        self.emit_changed()

    def _on_error(self, scope: str, message: str) -> None:
        if scope in ("health", "alerts"):
            self.set_error(message)

    def _on_error_cleared(self, scope: str) -> None:
        if scope in ("health", "alerts"):
            self.set_error(None)
