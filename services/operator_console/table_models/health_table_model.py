"""Subsystem health table model.

Backs the Health page's subsystem rollup. §12 distinguishes
degraded-but-recovering states (ADB drift freeze/reset, FFmpeg restart,
Azure retry-then-null, DB write buffer) from generic error states, so
`recovery_mode` and `operator_action_hint` are full-fledged columns
rather than collapsed into the detail string — the table must read
visually distinct for a subsystem that is self-healing versus one that
requires operator intervention.

Spec references:
  §12            — error-handling matrix, including degraded-but-recovering
                   paths (network disconnects, hardware loss, worker
                   crashes, queue overload)
  §4.E.1         — Health operator surface
"""

from __future__ import annotations

from typing import Any, ClassVar

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, QPersistentModelIndex, Qt

from packages.schemas.operator_console import HealthSubsystemStatus
from services.operator_console.formatters import format_health_state, format_timestamp

_EM_DASH = "—"


class HealthTableModel(QAbstractTableModel):
    """Per-subsystem health rollup with recovery-mode column."""

    COLUMNS: ClassVar[tuple[str, ...]] = (
        "Area",
        "Readiness",
        "What is happening",
        "Detail",
        "Last healthy (UTC)",
        "Next action",
    )

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._rows: list[HealthSubsystemStatus] = []

    def set_rows(self, rows: list[HealthSubsystemStatus]) -> None:
        new_rows = list(rows)
        # Preserve QTableView selection across health-poll updates by
        # only resetting the model when the row identities (subsystem
        # keys + order) actually change. When the structure is stable
        # we emit `dataChanged` per row instead — the view keeps its
        # selection because the row indexes stay valid.
        old_keys = [row.subsystem_key for row in self._rows]
        new_keys = [row.subsystem_key for row in new_rows]
        if old_keys == new_keys and new_rows:
            self._rows = new_rows
            top_left = self.index(0, 0)
            bottom_right = self.index(len(new_rows) - 1, len(self.COLUMNS) - 1)
            self.dataChanged.emit(
                top_left,
                bottom_right,
                [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole],
            )
            return
        self.beginResetModel()
        self._rows = new_rows
        self.endResetModel()

    def row_at(self, row: int) -> HealthSubsystemStatus | None:
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None

    def subsystem_by_key(self, key: str) -> HealthSubsystemStatus | None:
        for row in self._rows:
            if row.subsystem_key == key:
                return row
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
        if role == Qt.ItemDataRole.ToolTipRole:
            if col == 3 and row.detail:
                return row.detail
            if col == 5 and row.operator_action_hint:
                return row.operator_action_hint
            return None
        return None

    def _display(self, row: HealthSubsystemStatus, col: int) -> str:
        if col == 0:
            return row.label or row.subsystem_key
        if col == 1:
            return format_health_state(row.state)
        if col == 2:
            return row.recovery_mode or _EM_DASH
        if col == 3:
            return row.detail or _EM_DASH
        if col == 4:
            return format_timestamp(row.last_success_utc)
        if col == 5:
            return row.operator_action_hint or _EM_DASH
        return ""
