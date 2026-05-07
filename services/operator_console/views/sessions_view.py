"""Sessions page — history and recent sessions picker.

The shared `PollingCoordinator` drives the sessions fetch, `OperatorStore`
holds session state, and the view emits `session_selected(UUID)` when an
operator picks a row so the shell can open Live Session with that selection.

Columns come from `SessionsTableModel` and render through `formatters.py`.
A single-line filter sits above the table — the page is intentionally
table-led, so the filter is the only friction-removal we add and lives
on the page header rather than in a sidebar or modal.

Spec references:
  §4.E.1         — Sessions / history operator surface
  §7B            — latest_reward readback column
"""

from __future__ import annotations

from uuid import UUID

from PySide6.QtCore import (
    QModelIndex,
    QPersistentModelIndex,
    QSortFilterProxyModel,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLineEdit,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from packages.schemas.operator_console import AlertSeverity, SessionSummary
from services.operator_console.viewmodels.sessions_vm import SessionsViewModel
from services.operator_console.widgets.alert_banner import AlertBanner
from services.operator_console.widgets.empty_state import EmptyStateWidget
from services.operator_console.widgets.section_header import SectionHeader


class _SessionsFilterProxy(QSortFilterProxyModel):
    """Free-text filter across every visible column.

    Operators type "arm_b" or part of a session id; we fold case and
    match against the display text of every column so the filter does
    the obvious thing without the operator picking which field to search.
    """

    def filterAcceptsRow(  # noqa: N802 — Qt override
        self,
        source_row: int,
        source_parent: QModelIndex | QPersistentModelIndex,
    ) -> bool:
        pattern = self.filterRegularExpression().pattern()
        if not pattern:
            return True
        model = self.sourceModel()
        if model is None:
            return True
        column_count = model.columnCount(source_parent)
        for column in range(column_count):
            index = model.index(source_row, column, source_parent)
            value = model.data(index, Qt.ItemDataRole.DisplayRole)
            if value is None:
                continue
            if pattern.lower() in str(value).lower():
                return True
        return False


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
            "Recent sessions — double-click or press Enter to open one in Live Session.",
            self,
            level="page",
        )
        self._error_banner = AlertBanner(self)
        self._empty_state = EmptyStateWidget(self)
        self._empty_state.set_title("No sessions yet")
        self._empty_state.set_message(
            "Sessions will appear here after they are started from Live Session."
        )

        self._filter_input = QLineEdit(self)
        self._filter_input.setObjectName("SessionsFilterInput")
        self._filter_input.setPlaceholderText("Filter by session id, arm, or date…")
        self._filter_input.setClearButtonEnabled(True)
        self._filter_input.setAccessibleName("Filter sessions")
        self._filter_input.setAccessibleDescription(
            "Filter the sessions table by any visible column. Cmd/Ctrl-F focuses this input."
        )
        self._filter_input.setToolTip("Filter the sessions table. Cmd/Ctrl-F to focus.")

        self._proxy_model = _SessionsFilterProxy(self)
        self._proxy_model.setSourceModel(self._vm.sessions_model())
        self._proxy_model.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._filter_input.textChanged.connect(self._on_filter_text_changed)

        self._table = self._build_table()

        self._body_container = QWidget(self)
        body = QVBoxLayout(self._body_container)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(8)
        body.addWidget(self._filter_input)
        body.addWidget(self._table)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)
        layout.addWidget(self._header)
        layout.addWidget(self._error_banner)
        layout.addWidget(self._empty_state)
        layout.addWidget(self._body_container, 1)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Cmd/Ctrl-F focuses the filter input — keyboard parity with the
        # rest of the operator surface.
        focus_shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.Find), self)
        focus_shortcut.activated.connect(self._focus_filter)

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
        table.setModel(self._proxy_model)
        table.setSortingEnabled(False)
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
        table.setAccessibleName("Sessions table")
        table.setAccessibleDescription(
            "Select a session, then double-click or press Enter to open it in Live Session."
        )
        table.setToolTip("Double-click or press Enter to open the selected session.")
        # A double-click or activation is the explicit "open this session" gesture.
        table.doubleClicked.connect(self._on_table_double_clicked)
        table.activated.connect(self._on_table_double_clicked)
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

    def _row_to_summary(self, proxy_row: int) -> SessionSummary | None:
        source_index = self._proxy_model.mapToSource(self._proxy_model.index(proxy_row, 0))
        if not source_index.isValid():
            return None
        return self._vm.sessions_model().row_at(source_index.row())

    def _on_table_selection_changed(self, *_: object) -> None:
        # Single-click updates the store's selected_session_id only —
        # the shell shouldn't auto-navigate on every keyboard nav.
        selection_model = self._table.selectionModel()
        if selection_model is None:
            return
        indexes = selection_model.selectedRows()
        if not indexes:
            return
        summary = self._row_to_summary(indexes[0].row())
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
        summary = self._row_to_summary(indexes[0].row())
        if summary is None:
            return
        self._vm.select_session(summary.session_id)

    @Slot(object)
    def _on_vm_session_selected(self, session_id: object) -> None:
        if isinstance(session_id, UUID):
            self.session_selected.emit(session_id)

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def _on_filter_text_changed(self, text: str) -> None:
        self._proxy_model.setFilterFixedString(text.strip())

    def _focus_filter(self) -> None:
        self._filter_input.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self._filter_input.selectAll()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_error_changed(self, message: str) -> None:
        if message:
            self._error_banner.set_alert(AlertSeverity.WARNING, message)
        else:
            self._error_banner.set_alert(None, None)
