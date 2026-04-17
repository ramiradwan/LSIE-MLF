"""Minimal status indicator — colored dot + label.

Intentionally tiny replacement for the Debug Studio's ``StatusBadge`` —
the operator console prefers flat, minimal affordances over glass/glow.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from services.operator_console.theme import PALETTE


class StatusPill(QWidget):
    _COLORS: dict[str, str] = {
        "ok": PALETTE.status_ok,
        "warn": PALETTE.status_warn,
        "bad": PALETTE.status_bad,
        "idle": PALETTE.text_muted,
    }

    def __init__(self, text: str = "", kind: str = "idle", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._kind = kind
        self._dot = _Dot(self._color(), self)
        self._label = QLabel(text, self)
        self._label.setObjectName("StatusPillLabel")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._dot)
        layout.addWidget(self._label)
        layout.addStretch(1)

    def set_state(self, text: str, kind: str) -> None:
        self._kind = kind
        self._dot.set_color(self._color())
        self._label.setText(text)

    def _color(self) -> QColor:
        return QColor(self._COLORS.get(self._kind, PALETTE.text_muted))


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
