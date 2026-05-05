"""Encounters table model.

Backs the Live Session page's encounter timeline. Columns expose the
exact §7B reward-explanation inputs the pipeline used — the operator
needs to see `p90_intensity`, `semantic_gate`, `gated_reward`, and the
frame count alongside every row, or the table becomes an opaque log.

`set_rows` preserves selection: when the incoming list shares row
identity (by `encounter_id`) with what is currently rendered,
`dataChanged` is emitted instead of a full structural reset. That way
the Live Session view does not lose the operator's selected encounter
every time the 1-second poll fires. Structural resets happen only when
the identity sequence changes — rows inserted/removed/reordered.

Spec references:
  §4.C       — stimulus_time_utc (authoritative, orchestrator-owned)
  §4.C.4     — physiology freshness / staleness on per-segment payload
  §4.E.1     — Live Session operator surface
  §7B        — reward = p90_intensity × semantic_gate; gate=0 → 0 reward
"""

from __future__ import annotations

from typing import Any, ClassVar

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, QPersistentModelIndex, Qt

from packages.schemas.operator_console import EncounterSummary
from services.operator_console.formatters import (
    format_reward,
    format_semantic_gate,
    format_timestamp,
    reward_detail_labels,
)

_EM_DASH = "—"
_REWARD_LABELS = reward_detail_labels()


class EncountersTableModel(QAbstractTableModel):
    """Per-segment encounter rows with §7B reward-explanation columns.

    Column order is deliberate: timestamp and state anchor the row, the
    arm identifies which greeting was under test, and the four
    reward-explanation fields come next in the order an operator reads
    the §7B formula (inputs → outputs). Physiology flags come last so
    the table reads as a causal narrative.
    """

    COLUMNS: ClassVar[tuple[str, ...]] = (
        "Timestamp (UTC)",
        "State",
        "Stimulus strategy",
        _REWARD_LABELS.gate_title,
        _REWARD_LABELS.p90_title,
        _REWARD_LABELS.reward_title,
        "Frames in window",
        "Physiology",
    )

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._rows: list[EncounterSummary] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_rows(self, rows: list[EncounterSummary]) -> None:
        """Replace all rows, preserving selection when row identities match.

        If the sequence of `encounter_id`s is unchanged, we emit
        `dataChanged` across the existing span instead of a
        beginResetModel/endResetModel pair — the Live Session view's
        selection model keys off row indices, and a reset would clear
        the operator's current selection on every poll tick.
        """

        incoming = list(rows)
        current_ids = [row.encounter_id for row in self._rows]
        incoming_ids = [row.encounter_id for row in incoming]
        if current_ids == incoming_ids and current_ids:
            self._rows = incoming
            top_left = self.index(0, 0)
            bottom_right = self.index(len(incoming) - 1, len(self.COLUMNS) - 1)
            self.dataChanged.emit(top_left, bottom_right)
            return
        self.beginResetModel()
        self._rows = incoming
        self.endResetModel()

    def row_at(self, row: int) -> EncounterSummary | None:
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None

    def encounter_by_id(self, encounter_id: str) -> EncounterSummary | None:
        for row in self._rows:
            if row.encounter_id == encounter_id:
                return row
        return None

    def index_of_encounter(self, encounter_id: str) -> int | None:
        for idx, row in enumerate(self._rows):
            if row.encounter_id == encounter_id:
                return idx
        return None

    # ------------------------------------------------------------------
    # QAbstractTableModel overrides
    # ------------------------------------------------------------------

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
        # Vertical header: 1-based row numbers for operator reference.
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
            # Numeric columns right-align; text columns stay default-left.
            if col in (3, 4, 5, 6):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ToolTipRole and col == 2:
            # Full expected response text on the strategy column — table width stays tight.
            return row.expected_greeting or ""
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _display(self, row: EncounterSummary, col: int) -> str:
        if col == 0:
            return format_timestamp(row.segment_timestamp_utc)
        if col == 1:
            return row.state.value
        if col == 2:
            return row.active_arm or _EM_DASH
        if col == 3:
            return format_semantic_gate(row.semantic_gate)
        if col == 4:
            return format_reward(row.p90_intensity)
        if col == 5:
            return format_reward(row.gated_reward)
        if col == 6:
            if row.n_frames_in_window is None:
                return _EM_DASH
            return str(row.n_frames_in_window)
        if col == 7:
            return self._physiology_cell(row)
        return ""

    @staticmethod
    def _physiology_cell(row: EncounterSummary) -> str:
        if not row.physiology_attached:
            return "absent"
        if row.physiology_stale is True:
            return "attached (stale)"
        return "attached"
