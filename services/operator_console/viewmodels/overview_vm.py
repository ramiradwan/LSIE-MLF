"""Overview page viewmodel.

Binds the Overview page to the shared `OperatorStore`. All six Overview
cards (active session, experiment, physiology, health, latest encounter,
attention queue) read through this VM, so it only has to expose trivial
getters and re-emit a `changed` signal whenever any contributing store
slice mutates.

The store is the single source of truth; this VM deliberately holds no
cached copy. Re-reading on demand means the view always sees the latest
value and no drift between signal fan-out and cached state can occur.

Spec references:
  §4.E.1         — Overview operator surface (the six glance cards)
  §12            — alert feed drives the attention queue
"""

from __future__ import annotations

from PySide6.QtCore import QObject

from packages.schemas.operator_console import (
    AlertEvent,
    ExperimentSummary,
    HealthSnapshot,
    LatestEncounterSummary,
    OverviewSnapshot,
    SessionPhysiologySnapshot,
    SessionSummary,
)
from services.operator_console.formatters import (
    SemanticAttributionDiagnosticsDisplay,
    semantic_attribution_diagnostics_for_encounter,
)
from services.operator_console.state import OperatorStore
from services.operator_console.viewmodels.base import ViewModelBase


class OverviewViewModel(ViewModelBase):
    """Read-only accessor over `OverviewSnapshot` + alerts/health slices.

    Subscribes to the three store signals whose payloads feed the
    Overview cards. Alerts and health are also surfaced on other pages,
    but the Overview's attention queue and health card must stay in
    sync with the standalone surfaces, so we listen to the full set.
    """

    def __init__(self, store: OperatorStore, parent: QObject | None = None) -> None:
        super().__init__(store, parent)
        store.overview_changed.connect(self._on_any_change)
        store.alerts_changed.connect(self._on_any_change)
        store.health_changed.connect(self._on_any_change)
        store.error_changed.connect(self._on_error)
        store.error_cleared.connect(self._on_error_cleared)

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def snapshot(self) -> OverviewSnapshot | None:
        return self._store.overview()

    def active_session(self) -> SessionSummary | None:
        snap = self._store.overview()
        return snap.active_session if snap is not None else None

    def latest_encounter(self) -> LatestEncounterSummary | None:
        snap = self._store.overview()
        return snap.latest_encounter if snap is not None else None

    def experiment_summary(self) -> ExperimentSummary | None:
        snap = self._store.overview()
        return snap.experiment_summary if snap is not None else None

    def physiology_summary(self) -> SessionPhysiologySnapshot | None:
        snap = self._store.overview()
        # Prefer the composed overview snapshot's physiology; fall back
        # to the session-scoped physiology slice so the Overview card
        # still renders when only the session physiology job has fired.
        if snap is not None and snap.physiology is not None:
            return snap.physiology
        return self._store.physiology()

    def health_summary(self) -> HealthSnapshot | None:
        # The dedicated health slice wins over the overview-embedded
        # copy because the health poll cadence is independent and tends
        # to be fresher than the composed overview payload.
        health = self._store.health()
        if health is not None:
            return health
        snap = self._store.overview()
        return snap.health if snap is not None else None

    def alerts(self) -> list[AlertEvent]:
        return self._store.alerts()

    def latest_encounter_semantic_attribution_diagnostics(
        self,
    ) -> SemanticAttributionDiagnosticsDisplay:
        """Preformatted read-only §7E diagnostics from the latest encounter card."""

        return semantic_attribution_diagnostics_for_encounter(self.latest_encounter())

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_any_change(self, _payload: object) -> None:
        # One re-render signal no matter which slice moved; the view
        # pulls the fresh state through the getters above.
        self.emit_changed()

    def _on_error(self, scope: str, message: str) -> None:
        # Overview shows a page-level error slot for its own scopes
        # (overview / alerts / health); other scopes belong to other VMs.
        if scope in ("overview", "alerts", "health"):
            self.set_error(message)

    def _on_error_cleared(self, scope: str) -> None:
        if scope in ("overview", "alerts", "health"):
            self.set_error(None)
