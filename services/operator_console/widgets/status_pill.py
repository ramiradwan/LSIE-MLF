"""Status indicator pill — coloured dot + short label.

Takes the shared `UiStatusKind` enum from the
operator DTOs rather than ad-hoc string kinds, so every surface
(overview cards, health rows, action-bar recovery hints) maps health
and lifecycle state through the same palette bucket.

The pill is a pure presentation widget: it does not subscribe to the
store, does not talk to `ApiClient`, and holds no state beyond the
values set on it from the outside.

Spec references:
  §4.E.1         — operator-facing status affordances
  §12            — subsystem state categories (OK/DEGRADED/RECOVERING/ERROR)
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from packages.schemas.operator_console import UiStatusKind
from services.operator_console.theme import PALETTE

# §12 status colouring. RECOVERING picks the accent colour so it reads
# as "self-healing in progress" rather than "warning" (which would
# conflate it with DEGRADED).
_KIND_COLORS: dict[UiStatusKind, str] = {
    UiStatusKind.OK: PALETTE.status_ok,
    UiStatusKind.INFO: PALETTE.accent,
    UiStatusKind.WARN: PALETTE.status_warn,
    UiStatusKind.ERROR: PALETTE.status_bad,
    UiStatusKind.NEUTRAL: PALETTE.text_muted,
    UiStatusKind.PROGRESS: PALETTE.accent,
}


class StatusPill(QWidget):
    """Coloured dot + text label, laid out horizontally."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._kind: UiStatusKind = UiStatusKind.NEUTRAL
        self._dot = _Dot(self._color(), self)
        self._label = QLabel("", self)
        self._label.setObjectName("StatusPillLabel")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._dot)
        layout.addWidget(self._label)
        layout.addStretch(1)

    def set_text(self, text: str) -> None:
        self._label.setText(text)

    def set_kind(self, kind: UiStatusKind) -> None:
        if kind == self._kind:
            return
        self._kind = kind
        self._dot.set_color(self._color())

    def kind(self) -> UiStatusKind:
        return self._kind

    def text(self) -> str:
        return self._label.text()

    def _color(self) -> QColor:
        return QColor(_KIND_COLORS.get(self._kind, PALETTE.text_muted))


class _Dot(QWidget):
    def __init__(self, color: QColor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = color
        self.setFixedSize(10, 10)

    def set_color(self, color: QColor) -> None:
        self._color = color
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 — Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._color)
        painter.drawEllipse(0, 0, self.width(), self.height())
