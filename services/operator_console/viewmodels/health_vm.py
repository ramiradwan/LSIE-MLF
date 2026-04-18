"""Health page viewmodel — Phase 8.

Owns two tables: the subsystem rollup (§12 states with recovery-mode
and operator-action hints) and the alerts feed. Keeping them separate
is deliberate — §12 transitions emit both a subsystem state change and
an alert, and the operator reads them at different times (the rollup
is the "right now" snapshot, the feed is the audit trail).

`degraded_count` is a convenience accessor used by the shell's
navigation badge: if non-zero, the sidebar surfaces a small indicator.

Spec references:
  §4.E.1         — Health operator surface
  §12            — error-handling matrix, including
                   degraded-but-recovering paths
  SPEC-AMEND-008 — PySide6 Operator Console
"""

from __future__ import annotations

from PySide6.QtCore import QObject

from packages.schemas.operator_console import HealthSnapshot
from services.operator_console.state import OperatorStore
from services.operator_console.table_models.alerts_table_model import AlertsTableModel
from services.operator_console.table_models.health_table_model import HealthTableModel
from services.operator_console.viewmodels.base import ViewModelBase


class HealthViewModel(ViewModelBase):
    """Owns the subsystem table + alerts feed."""

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
        store.health_changed.connect(self._on_health_changed)
        store.alerts_changed.connect(self._on_alerts_changed)
        store.error_changed.connect(self._on_error)
        store.error_cleared.connect(self._on_error_cleared)
        # Seed from whatever the store already holds.
        self._sync_health_rows(self._store.health())
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
            self._sync_health_rows(snap if isinstance(snap, HealthSnapshot) else None)
        else:
            self._sync_health_rows(self._store.health())
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

    def _on_error(self, scope: str, message: str) -> None:
        if scope in ("health", "alerts"):
            self.set_error(message)

    def _on_error_cleared(self, scope: str) -> None:
        if scope in ("health", "alerts"):
            self.set_error(None)
