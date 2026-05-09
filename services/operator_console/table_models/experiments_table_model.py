"""Experiments (arms) table model.

Backs the Experiments page's arm posterior table. Columns surface the
exact Thompson Sampling readbacks an operator needs to reason about
exploration vs exploitation: arm id, greeting text, posterior α/β,
evaluation variance, selection count, and the recent reward mean.

Only human-owned arm metadata is editable from this model. The greeting
column emits a rename request, and the Enabled checkbox only emits the
one-way disable request; posterior-owned numeric fields remain read-only
and cannot be edited through the operator console.

Semantic confidence is deliberately not a column here — §7B's reward is
`p90_intensity × semantic_gate`, not a confidence-scaled quantity, so
including it in the arm table would imply it moves the posterior. That
relationship belongs on the Live Session encounter table (where
`semantic_confidence` is informational, not reward-bearing).

Spec references:
  §7B            — Thompson Sampling posterior (α, β), evaluation variance
  §4.E.1         — Experiments operator surface
"""

from __future__ import annotations

from typing import Any, ClassVar

from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    Qt,
    Signal,
)

from packages.schemas.operator_console import ArmSummary
from services.operator_console.formatters import format_percentage, format_reward

_EM_DASH = "—"
_ENABLED_COL = 2
_GREETING_COL = 1
_POSTERIOR_ALPHA_COL = 3


class ExperimentsTableModel(QAbstractTableModel):
    """Thompson Sampling arms table plus safe arm-management intents."""

    greeting_edit_requested = Signal(str, str)  # arm_id, greeting_text
    disable_requested = Signal(str)  # arm_id

    COLUMNS: ClassVar[tuple[str, ...]] = (
        "Arm",
        "Confirmation text",
        "Enabled",
        "Posterior α",
        "Posterior β",
        "Eval variance",
        "Selections",
        "Recent reward mean",
        "Recent stimulus confirmed",
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

    def flags(  # noqa: N802 — Qt override
        self,
        index: QModelIndex | QPersistentModelIndex,
    ) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        row = self.row_at(index.row())
        if row is None:
            return Qt.ItemFlag.NoItemFlags
        flags = super().flags(index)
        if index.column() == _GREETING_COL:
            return flags | Qt.ItemFlag.ItemIsEditable
        if index.column() == _ENABLED_COL and row.enabled:
            return flags | Qt.ItemFlag.ItemIsUserCheckable
        return flags

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
        if role == Qt.ItemDataRole.CheckStateRole and col == _ENABLED_COL:
            return Qt.CheckState.Checked if row.enabled else Qt.CheckState.Unchecked
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if col == _ENABLED_COL:
                return int(Qt.AlignmentFlag.AlignCenter)
            if col in (3, 4, 5, 6, 7, 8):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ToolTipRole:
            if col == _GREETING_COL:
                return "Double-click or press F2/Enter to edit confirmation text."
            if col == _ENABLED_COL and row.enabled:
                return "Uncheck to disable this arm. Disabled arms cannot be re-enabled here."
            if col == _ENABLED_COL:
                return "Arm disabled; re-enable is not supported by the operator flow."
            if col == _POSTERIOR_ALPHA_COL:
                return (
                    f"posterior α {format_reward(row.posterior_alpha)} · "
                    f"posterior β {format_reward(row.posterior_beta)}"
                )
        if role == Qt.ItemDataRole.UserRole and col == _POSTERIOR_ALPHA_COL:
            return (row.posterior_alpha, row.posterior_beta)
        return None

    def setData(  # noqa: N802 — Qt override
        self,
        index: QModelIndex | QPersistentModelIndex,
        value: Any,
        role: int = Qt.ItemDataRole.EditRole,
    ) -> bool:
        if not index.isValid():
            return False
        row = self.row_at(index.row())
        if row is None:
            return False
        col = index.column()
        if col == _GREETING_COL and role == Qt.ItemDataRole.EditRole:
            greeting_text = str(value).strip()
            if not greeting_text:
                return False
            if greeting_text == row.greeting_text:
                return True
            self.greeting_edit_requested.emit(row.arm_id, greeting_text)
            return True
        if col == _ENABLED_COL and role == Qt.ItemDataRole.CheckStateRole:
            requested_state = _coerce_check_state(value)
            if requested_state is None:
                return False
            # One-way disable only. A checked request would imply
            # re-enabling, which the backend intentionally rejects.
            if row.enabled and requested_state == Qt.CheckState.Unchecked:
                self.disable_requested.emit(row.arm_id)
                return True
            return False
        return False

    def _display(self, row: ArmSummary, col: int) -> str:
        if col == 0:
            return row.arm_id
        if col == 1:
            return row.greeting_text
        if col == 2:
            return "enabled" if row.enabled else "disabled"
        if col == 3:
            return format_reward(row.posterior_alpha)
        if col == 4:
            return format_reward(row.posterior_beta)
        if col == 5:
            return format_reward(row.evaluation_variance)
        if col == 6:
            return str(row.selection_count)
        if col == 7:
            return format_reward(row.recent_reward_mean)
        if col == 8:
            return format_percentage(row.recent_semantic_pass_rate, digits=0)
        return ""


def _coerce_check_state(value: Any) -> Qt.CheckState | None:
    if isinstance(value, Qt.CheckState):
        return value
    try:
        return Qt.CheckState(int(value))
    except (TypeError, ValueError):
        return None
