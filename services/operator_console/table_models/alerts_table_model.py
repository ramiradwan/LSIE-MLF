"""Alerts table model.

Backs the attention queue surfaces (Overview card + Health page feed).
Carries both `set_rows` (for full-refresh semantics) and `append_rows`
(for streaming-style updates) because §12 alerts arrive incrementally —
a §12 subsystem degradation and its paired recovery are two events, not
one, and the operator sees the sequence.

Duplicate filtering is by `alert_id` so that a re-emission during
append_rows does not double-post an alert the operator has already seen.

Spec references:
  §12            — subsystem state transitions, physiology staleness,
                   stimulus lifecycle failures, data gaps
  §4.E.1         — Health / attention-queue operator surfaces
"""

from __future__ import annotations

from typing import Any, ClassVar

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, QPersistentModelIndex, Qt

from packages.schemas.operator_console import AlertEvent, AlertSeverity
from services.operator_console.formatters import format_timestamp

_EM_DASH = "—"


class AlertsTableModel(QAbstractTableModel):
    """Timestamped alert feed with severity / kind / message / acknowledged.

    Newest rows render first — `set_rows` and `append_rows` both leave
    the caller's ordering intact so the store's policy (sort descending
    by `emitted_at_utc`) drives the final presentation rather than the
    model reshuffling.
    """

    COLUMNS: ClassVar[tuple[str, ...]] = (
        "Time (UTC)",
        "Severity",
        "Kind",
        "Subsystem",
        "Message",
        "Ack",
    )

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._rows: list[AlertEvent] = []

    def set_rows(self, rows: list[AlertEvent]) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def append_rows(self, rows: list[AlertEvent]) -> None:
        """Append rows skipping duplicates by `alert_id`.

        New rows go at the *start* of the list so the operator sees the
        most recent event first. De-duplication is important: §12
        subsystem transitions can re-emit the same alert while a state
        persists, and we don't want the queue to grow with clones.
        """
        existing = {row.alert_id for row in self._rows}
        fresh = [row for row in rows if row.alert_id not in existing]
        if not fresh:
            return
        self.beginInsertRows(QModelIndex(), 0, len(fresh) - 1)
        self._rows = list(fresh) + self._rows
        self.endInsertRows()

    def row_at(self, row: int) -> AlertEvent | None:
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
        if role == Qt.ItemDataRole.ToolTipRole and col == 4:
            return row.message
        return None

    def _display(self, row: AlertEvent, col: int) -> str:
        if col == 0:
            return format_timestamp(row.emitted_at_utc)
        if col == 1:
            return _severity_label(row.severity)
        if col == 2:
            return row.kind.value
        if col == 3:
            return row.subsystem_key or _EM_DASH
        if col == 4:
            return row.message
        if col == 5:
            return "yes" if row.acknowledged else "no"
        return ""


def _severity_label(severity: AlertSeverity) -> str:
    mapping = {
        AlertSeverity.INFO: "info",
        AlertSeverity.WARNING: "warning",
        AlertSeverity.CRITICAL: "critical",
    }
    return mapping[severity]
