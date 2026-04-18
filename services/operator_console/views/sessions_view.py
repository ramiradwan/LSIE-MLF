"""Sessions page — history + recent sessions picker.

Phase 10 rewrites the scaffold SessionsView into a proper page: no more
per-page `QThread` or `ApiClient`, no more ad-hoc dict rows. The shared
`PollingCoordinator` drives the underlying fetch, `OperatorStore` holds
the state, and the view now emits `session_selected(UUID)` when an
operator picks a row so the shell can jump into Live Session with the
same session already wired through.

Columns come from `SessionsTableModel` (Phase 7) and render through
`formatters.py`; no inline formatting here.

Spec references:
  §4.E.1         — Sessions / history operator surface
  §7B            — latest_reward readback column
  SPEC-AMEND-008 — PySide6 Operator Console
"""

from __future__ import annotations

from uuid import UUID

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from packages.schemas.operator_console import AlertSeverity
from services.operator_console.viewmodels.sessions_vm import SessionsViewModel
from services.operator_console.widgets.alert_banner import AlertBanner
from services.operator_console.widgets.empty_state import EmptyStateWidget
from services.operator_console.widgets.section_header import SectionHeader


class SessionsView(QWidget):
    """Sessions history page: table only + session-selected signal."""

    # Emitted when the operator picks a row. The shell routes this into
    # the store + page change; other pages react through the store.
    session_selected = Signal(object)  # UUID

    def __init__(
        self,
        vm: SessionsViewModel,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ContentSurface")
        self._vm = vm

        self._header = SectionHeader(
            "Sessions",
            "Recent sessions — click a row to open it in Live Session.",
            self,
        )
        self._error_banner = AlertBanner(self)
        self._empty_state = EmptyStateWidget(self)
        self._empty_state.set_title("No sessions yet")
        self._empty_state.set_message(
            "Sessions will appear here as they are created by the capture stack."
        )

        self._table = self._build_table()

        self._body_container = QWidget(self)
        body = QVBoxLayout(self._body_container)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._table)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)
        layout.addWidget(self._header)
        layout.addWidget(self._error_banner)
        layout.addWidget(self._empty_state)
        layout.addWidget(self._body_container, 1)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._vm.changed.connect(self._refresh)
        self._vm.error_changed.connect(self._on_error_changed)
        # Forward the VM's selection signal to the shell.
        self._vm.session_selected.connect(self._on_vm_session_selected)

        self._refresh()

    # ------------------------------------------------------------------
    # Page lifecycle hooks
    # ------------------------------------------------------------------

    def on_activated(self) -> None:
        self._refresh()

    def on_deactivated(self) -> None:
        return None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_table(self) -> QTableView:
        table = QTableView(self)
        table.setObjectName("SessionsTable")
        table.setModel(self._vm.sessions_model())
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        vertical = table.verticalHeader()
        if vertical is not None:
            vertical.setVisible(False)
        horizontal = table.horizontalHeader()
        if horizontal is not None:
            horizontal.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        selection_model = table.selectionModel()
        if selection_model is not None:
            selection_model.selectionChanged.connect(self._on_table_selection_changed)
        # A double-click is the explicit "open this session" gesture —
        # single-click selects, double-click commits to the VM + shell.
        table.doubleClicked.connect(self._on_table_double_clicked)
        return table

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        has_rows = self._vm.sessions_model().rowCount() > 0
        self._empty_state.setVisible(not has_rows)
        self._body_container.setVisible(has_rows)

    # ------------------------------------------------------------------
    # Selection handling
    # ------------------------------------------------------------------

    def _on_table_selection_changed(self, *_: object) -> None:
        # Single-click updates the store's selected_session_id only —
        # the shell shouldn't auto-navigate on every keyboard nav.
        selection_model = self._table.selectionModel()
        if selection_model is None:
            return
        indexes = selection_model.selectedRows()
        if not indexes:
            return
        row = indexes[0].row()
        summary = self._vm.sessions_model().row_at(row)
        if summary is None:
            return
        self._vm.store.set_selected_session_id(summary.session_id)

    def _on_table_double_clicked(self, *_: object) -> None:
        selection_model = self._table.selectionModel()
        if selection_model is None:
            return
        indexes = selection_model.selectedRows()
        if not indexes:
            return
        row = indexes[0].row()
        summary = self._vm.sessions_model().row_at(row)
        if summary is None:
            return
        self._vm.select_session(summary.session_id)

    @Slot(object)
    def _on_vm_session_selected(self, session_id: object) -> None:
        if isinstance(session_id, UUID):
            self.session_selected.emit(session_id)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_error_changed(self, message: str) -> None:
        if message:
            self._error_banner.set_alert(AlertSeverity.WARNING, message)
        else:
            self._error_banner.set_alert(None, None)
