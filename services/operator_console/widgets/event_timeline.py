"""Event timeline — a QTableView wrapper that scrolls to latest on demand.

Backs the alerts feed (`AlertsTableModel`) and the encounter
timeline on the Live Session page. We wrap QTableView rather than
subclassing so consumers can swap in any QAbstractItemModel without
this widget needing to know the column layout.

Spec references:
  §4.E.1         — operator-facing event timelines
  §12            — alert feed surfaced on Health view
"""

from __future__ import annotations

from PySide6.QtCore import QAbstractItemModel, QModelIndex
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QTableView,
    QVBoxLayout,
    QWidget,
)


class EventTimelineWidget(QWidget):
    """Timeline list of events, newest at the bottom.

    `scroll_to_latest()` scrolls to the last row in the attached model,
    which is what operator-grade timelines actually want (you're
    watching for what just happened).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("EventTimeline")

        self._table = QTableView(self)
        self._table.setObjectName("EventTimelineTable")
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        vertical = self._table.verticalHeader()
        if vertical is not None:
            vertical.setVisible(False)
        horizontal = self._table.horizontalHeader()
        if horizontal is not None:
            horizontal.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._table)

    def set_model(self, model: QAbstractItemModel) -> None:
        self._table.setModel(model)
        # Reset the stretch policy after a model swap — Qt drops header
        # resize modes when a new model is attached.
        horizontal = self._table.horizontalHeader()
        if horizontal is not None:
            horizontal.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def model(self) -> QAbstractItemModel | None:
        return self._table.model()

    def scroll_to_latest(self) -> None:
        model = self._table.model()
        if model is None:
            return
        rows = model.rowCount(QModelIndex())
        if rows <= 0:
            return
        last_index = model.index(rows - 1, 0, QModelIndex())
        if last_index.isValid():
            self._table.scrollTo(last_index, QAbstractItemView.ScrollHint.PositionAtBottom)
