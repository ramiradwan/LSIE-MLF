"""Experiments (arms) table model — Phase 7.

Backs the Experiments page's arm posterior table. Columns surface the
exact Thompson Sampling readbacks an operator needs to reason about
exploration vs exploitation: arm id, greeting text, posterior α/β,
evaluation variance, selection count, and the recent reward mean.

Semantic confidence is deliberately not a column here — §7B's reward is
`p90_intensity × semantic_gate`, not a confidence-scaled quantity, so
including it in the arm table would imply it moves the posterior. That
relationship belongs on the Live Session encounter table (where
`semantic_confidence` is informational, not reward-bearing).

Spec references:
  §7B            — Thompson Sampling posterior (α, β), evaluation variance
  §4.E.1         — Experiments operator surface
  SPEC-AMEND-008 — PySide6 Operator Console
"""

from __future__ import annotations

from typing import Any, ClassVar

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, QPersistentModelIndex, Qt

from packages.schemas.operator_console import ArmSummary
from services.operator_console.formatters import format_percentage, format_reward

_EM_DASH = "—"


class ExperimentsTableModel(QAbstractTableModel):
    """Thompson Sampling arms table."""

    COLUMNS: ClassVar[tuple[str, ...]] = (
        "Arm",
        "Greeting",
        "Posterior α",
        "Posterior β",
        "Eval variance",
        "Selections",
        "Recent reward mean",
        "Recent semantic pass",
    )

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._rows: list[ArmSummary] = []

    def set_rows(self, rows: list[ArmSummary]) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def row_at(self, row: int) -> ArmSummary | None:
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None

    def arm_by_id(self, arm_id: str) -> ArmSummary | None:
        for row in self._rows:
            if row.arm_id == arm_id:
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
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if col in (2, 3, 4, 5, 6, 7):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ToolTipRole and col == 1:
            return row.greeting_text
        return None

    def _display(self, row: ArmSummary, col: int) -> str:
        if col == 0:
            return row.arm_id
        if col == 1:
            return row.greeting_text
        if col == 2:
            return format_reward(row.posterior_alpha)
        if col == 3:
            return format_reward(row.posterior_beta)
        if col == 4:
            return format_reward(row.evaluation_variance)
        if col == 5:
            return str(row.selection_count)
        if col == 6:
            return format_reward(row.recent_reward_mean)
        if col == 7:
            return format_percentage(row.recent_semantic_pass_rate, digits=0)
        return ""
