"""Probe sparkline — last 60 probe states for one subsystem.

A row-level visualisation that lets operators see flapping at a glance
without bouncing between the probe matrix and the alert timeline. Each
of the 60 cells is colored by `UiStatusKind`, the same vocabulary the
rest of the console uses, so recovering reads distinct from degraded
reads distinct from error.

Hovering the widget shows a static `setToolTip` summary built from the
most recent cell. Per-cell tooltips via `QToolTip.showText` were tried
and removed: with `setMouseTracking(True)` they pop a fresh floating
window on every mouse-move event, which read as a flickering box
loop on the Health page during normal hover.

Spec references:
  §4.E.1         — Health operator surface
  §12            — degraded vs recovering vs error
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent
from PySide6.QtWidgets import QSizePolicy, QWidget

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
        # Track a target width but allow the widget to shrink when the
        # parent panel is narrow; a 480px hard floor blew out the column
        # widths on the Health probe matrix at 1024px.
        target_width = self._capacity * (_CELL_WIDTH + _CELL_SPACING)
        self.setMinimumWidth(120)
        self.setMaximumWidth(target_width)
        self.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
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
        cell_width, cell_spacing = self._effective_cell_metrics()
        for index in range(self._capacity):
            rect = self._cell_rect(index, cell_width, cell_spacing)
            cell_index = index - (self._capacity - len(self._cells))
            if 0 <= cell_index < len(self._cells):
                cell = self._cells[cell_index]
                painter.setBrush(QColor(_KIND_COLORS.get(cell.state, PALETTE.text_muted)))
            else:
                painter.setBrush(self._empty_color)
            painter.drawRect(rect)

    def _effective_cell_metrics(self) -> tuple[int, int]:
        """Shrink cell width/spacing when the widget is narrower than ideal.

        Cells stay readable down to a 2px minimum width, and spacing
        scales linearly. The result is the natural metrics on a wide
        panel and a compressed-but-visible view on a narrow one.
        """

        available = max(self.width(), 1)
        natural = self._capacity * (_CELL_WIDTH + _CELL_SPACING)
        if available >= natural:
            return _CELL_WIDTH, _CELL_SPACING
        ratio = available / natural
        cell_width = max(2, int(_CELL_WIDTH * ratio))
        cell_spacing = max(0, int(_CELL_SPACING * ratio))
        return cell_width, cell_spacing

    # ---- internals -----------------------------------------------------

    def _cell_rect(
        self,
        index: int,
        cell_width: int = _CELL_WIDTH,
        cell_spacing: int = _CELL_SPACING,
    ) -> QRect:
        x = index * (cell_width + cell_spacing)
        return QRect(x, 0, cell_width, _CELL_HEIGHT)
