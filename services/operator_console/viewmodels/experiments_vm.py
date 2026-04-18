"""Experiments page viewmodel — Phase 8.

Surfaces the §7B Thompson Sampling arm readback: posterior α/β per arm,
evaluation variance, selection counts, and the "last update summary"
line that the API's read service formats.

Semantic confidence is explicitly *not* modeled here as an input to the
reward. §7B defines the gated reward as `r_t = p90_intensity × semantic_gate`
with the gate being the integer 0/1 — confidence is informational on
the per-encounter surface and must not appear to move the posterior on
the Experiments page.

Spec references:
  §4.E.1         — Experiments operator surface
  §7B            — Thompson Sampling posterior; reward math is
                   intensity × gate, not confidence-scaled
  SPEC-AMEND-008 — PySide6 Operator Console
"""

from __future__ import annotations

from PySide6.QtCore import QObject

from packages.schemas.operator_console import ExperimentDetail
from services.operator_console.state import OperatorStore
from services.operator_console.table_models.experiments_table_model import (
    ExperimentsTableModel,
)
from services.operator_console.viewmodels.base import ViewModelBase


class ExperimentsViewModel(ViewModelBase):
    """Owns the arms table and exposes active-arm / last-update readbacks."""

    def __init__(
        self,
        store: OperatorStore,
        arms_model: ExperimentsTableModel,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(store, parent)
        self._arms_model = arms_model
        store.experiment_changed.connect(self._on_experiment_changed)
        store.error_changed.connect(self._on_error)
        store.error_cleared.connect(self._on_error_cleared)
        # Seed from whatever the store already holds.
        self._sync_rows(self._store.experiment())

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def detail(self) -> ExperimentDetail | None:
        return self._store.experiment()

    def arms_model(self) -> ExperimentsTableModel:
        return self._arms_model

    def active_arm_id(self) -> str | None:
        detail = self._store.experiment()
        return detail.active_arm_id if detail is not None else None

    def latest_update_summary(self) -> str:
        """Return the API-provided update summary, or a neutral placeholder.

        The update summary is pre-formatted server-side (see
        `OperatorReadService._build_experiment_summary`) so the
        operator sees the same phrasing the API logs use.
        """
        detail = self._store.experiment()
        if detail is None or detail.last_update_summary is None:
            return "No experiment update yet."
        return detail.last_update_summary

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_experiment_changed(self, detail: object) -> None:
        if isinstance(detail, ExperimentDetail) or detail is None:
            self._sync_rows(detail if isinstance(detail, ExperimentDetail) else None)
        else:
            self._sync_rows(self._store.experiment())
        self.emit_changed()

    def _sync_rows(self, detail: ExperimentDetail | None) -> None:
        if detail is None:
            self._arms_model.set_rows([])
        else:
            self._arms_model.set_rows(list(detail.arms))

    def _on_error(self, scope: str, message: str) -> None:
        if scope == "experiment":
            self.set_error(message)

    def _on_error_cleared(self, scope: str) -> None:
        if scope == "experiment":
            self.set_error(None)
