"""Sessions (history) page viewmodel.

Reads sessions from the shared `OperatorStore` while `PollingCoordinator`
drives the underlying fetch through a route-scoped job. The VM holds no
cached copy: it re-reads on demand and re-emits on store change so the
view stays aligned with shared operator state.

`session_selected(session_id)` is the VM's write path: clicking a row
pushes the selection into the store so the shell can carry it across
pages (most commonly Sessions → Live Session).

Spec references:
  §4.E.1         — Sessions / history operator surface
  §7B            — latest_reward is the §7B readback carried on SessionSummary
"""

from __future__ import annotations

from uuid import UUID

from PySide6.QtCore import QObject, Signal

from packages.schemas.operator_console import SessionSummary
from services.operator_console.state import OperatorStore
from services.operator_console.table_models.sessions_table_model import (
    SessionsTableModel,
)
from services.operator_console.viewmodels.base import ViewModelBase


class SessionsViewModel(ViewModelBase):
    """Owns the sessions table model + exposes a `session_selected` signal."""

    # fmt: off
    session_selected = Signal(object)  # UUID
    # fmt: on

    def __init__(
        self,
        store: OperatorStore,
        model: SessionsTableModel,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(store, parent)
        self._model = model
        store.sessions_changed.connect(self._on_sessions_changed)
        store.error_changed.connect(self._on_error)
        store.error_cleared.connect(self._on_error_cleared)
        # Seed the table from whatever the store already has.
        self._model.set_rows(self._store.sessions())

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def sessions_model(self) -> SessionsTableModel:
        return self._model

    def sessions(self) -> list[SessionSummary]:
        return self._store.sessions()

    def selected_session_id(self) -> UUID | None:
        return self._store.selected_session_id()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def select_session(self, session_id: UUID | None) -> None:
        """Push the selection into the store and announce it to the shell.

        The store is the authority — a second subscriber (e.g. the
        Overview page's action-bar context) reads from the store, not
        from this signal. The signal is a UX nudge so the shell can
        switch route without the view having to know about navigation.
        """
        self._store.set_selected_session_id(session_id)
        self.session_selected.emit(session_id)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_sessions_changed(self, rows: object) -> None:
        if isinstance(rows, list):
            self._model.set_rows(rows)
        else:
            self._model.set_rows(self._store.sessions())
        self.emit_changed()

    def _on_error(self, scope: str, message: str) -> None:
        if scope == "sessions":
            self.set_error(message)

    def _on_error_cleared(self, scope: str) -> None:
        if scope == "sessions":
            self.set_error(None)
