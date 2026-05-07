"""Probe sparkline — last 60 probe states for one subsystem.

A row-level visualisation that lets operators see flapping at a glance
without bouncing between the probe matrix and the alert timeline. Each
of the 60 cells is colored by `UiStatusKind`, the same vocabulary the
rest of the console uses, so recovering reads distinct from degraded
reads distinct from error.

Hovering a cell shows the timestamp and state via tooltip, sourced
from the buffer that the Health viewmodel already polls; this widget
itself owns no state.

Spec references:
  §4.E.1         — Health operator surface
  §12            — degraded vs recovering vs error
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent
from PySide6.QtWidgets import QSizePolicy, QToolTip, QWidget

from packages.schemas.operator_console import HealthProbeState, UiStatusKind
from services.operator_console.design_system.tokens import PALETTE


@dataclass(frozen=True)
class ProbeSparklineCell:
    """One probe sample.

    `state` is the §12-aligned `UiStatusKind` so the cell colour matches
    the rest of the console. `timestamp_utc` is purely informational —
    it shows up in the tooltip and is never persisted from this widget.
    """

    timestamp_utc: datetime | None
    state: UiStatusKind
    probe_state: HealthProbeState | None = None


_KIND_COLORS: dict[UiStatusKind, str] = {
    UiStatusKind.OK: PALETTE.status_ok,
    UiStatusKind.INFO: PALETTE.accent,
    UiStatusKind.WARN: PALETTE.status_warn,
    UiStatusKind.ERROR: PALETTE.status_bad,
    UiStatusKind.PROGRESS: PALETTE.status_recovering,
    UiStatusKind.NEUTRAL: PALETTE.text_muted,
    UiStatusKind.MUTED: PALETTE.text_muted,
}

_CELL_WIDTH = 6
_CELL_HEIGHT = 14
_CELL_SPACING = 2
_DEFAULT_CAPACITY = 60


class ProbeSparkline(QWidget):
    """Horizontal colored cells visualising the last N probe samples."""

    def __init__(
        self,
        *,
        capacity: int = _DEFAULT_CAPACITY,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ProbeSparkline")
        self._capacity = max(1, capacity)
        self._cells: list[ProbeSparklineCell] = []
        self._empty_color = QColor(PALETTE.surface_raised)
        self.setFixedHeight(_CELL_HEIGHT)
        target_width = self._capacity * (_CELL_WIDTH + _CELL_SPACING)
        self.setMinimumWidth(target_width)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self.setAccessibleName("Probe history sparkline")
        self.setAccessibleDescription(
            f"Last {self._capacity} probe samples for this subsystem; one cell per probe."
        )

    def set_cells(self, cells: Sequence[ProbeSparklineCell]) -> None:
        self._cells = list(cells)[-self._capacity :]
        self.update()

    def cells(self) -> tuple[ProbeSparklineCell, ...]:
        return tuple(self._cells)

    # ---- events --------------------------------------------------------

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 — Qt override
        del event
        painter = QPainter(self)
        painter.setPen(Qt.PenStyle.NoPen)
        for index in range(self._capacity):
            rect = self._cell_rect(index)
            cell_index = index - (self._capacity - len(self._cells))
            if 0 <= cell_index < len(self._cells):
                cell = self._cells[cell_index]
                painter.setBrush(QColor(_KIND_COLORS.get(cell.state, PALETTE.text_muted)))
            else:
                painter.setBrush(self._empty_color)
            painter.drawRect(rect)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt override
        cell = self._cell_at(event.position().toPoint())
        if cell is None:
            QToolTip.hideText()
            super().mouseMoveEvent(event)
            return
        ts_text = cell.timestamp_utc.isoformat() if cell.timestamp_utc is not None else "—"
        state_text = cell.probe_state.value if cell.probe_state is not None else cell.state.value
        QToolTip.showText(
            event.globalPosition().toPoint(),
            f"{state_text} · {ts_text}",
            self,
        )
        super().mouseMoveEvent(event)

    # ---- internals -----------------------------------------------------

    def _cell_rect(self, index: int) -> QRect:
        x = index * (_CELL_WIDTH + _CELL_SPACING)
        return QRect(x, 0, _CELL_WIDTH, _CELL_HEIGHT)

    def _cell_at(self, position: QPoint) -> ProbeSparklineCell | None:
        if position.y() < 0 or position.y() > _CELL_HEIGHT:
            return None
        index = position.x() // (_CELL_WIDTH + _CELL_SPACING)
        cell_index = index - (self._capacity - len(self._cells))
        if 0 <= cell_index < len(self._cells):
            return self._cells[cell_index]
        return None
