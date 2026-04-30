"""Sessions (recent) table model.

Backs the Sessions page's history table. Columns surface the session
identity, current status, which arm/experiment was running, the latest
reward, and duration so the operator can pick a row to drill into.

Spec references:
  §4.E.1         — Sessions / history operator surface
  §7B            — latest_reward readback
"""

from __future__ import annotations

from typing import Any, ClassVar

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, QPersistentModelIndex, Qt

from packages.schemas.operator_console import SessionSummary
from services.operator_console.formatters import (
    format_duration,
    format_reward,
    format_timestamp,
)

_EM_DASH = "—"


class SessionsTableModel(QAbstractTableModel):
    """Recent sessions history."""

    COLUMNS: ClassVar[tuple[str, ...]] = (
        "Started (UTC)",
        "Status",
        "Experiment",
        "Active arm",
        "Latest reward",
        "Duration",
    )

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._rows: list[SessionSummary] = []

    def set_rows(self, rows: list[SessionSummary]) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def row_at(self, row: int) -> SessionSummary | None:
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None

    def rowCount(  # noqa: N802 — Qt override
        self,
        parent: QModelIndex | QPersistentModelIndex = QModelIndex(),  # noqa: B008
    ) -> int:
        if parent.isValid():
            return 0
        return len(self._rows)

    def columnCount(  # noqa: N802 — Qt override
        self,
        parent: QModelIndex | QPersistentModelIndex = QModelIndex(),  # noqa: B008
    ) -> int:
        if parent.isValid():
            return 0
        return len(self.COLUMNS)

    def headerData(  # noqa: N802 — Qt override
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self.COLUMNS):
                return self.COLUMNS[section]
            return None
        return section + 1

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid():
            return None
        row = self.row_at(index.row())
        if row is None:
            return None
        col = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            return self._display(row, col)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if col in (4, 5):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        return None

    def _display(self, row: SessionSummary, col: int) -> str:
        if col == 0:
            return format_timestamp(row.started_at_utc)
        if col == 1:
            return row.status
        if col == 2:
            return row.experiment_id or _EM_DASH
        if col == 3:
            return row.active_arm or _EM_DASH
        if col == 4:
            return format_reward(row.latest_reward)
        if col == 5:
            return format_duration(row.duration_s)
        return ""
