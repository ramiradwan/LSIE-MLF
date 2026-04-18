"""Empty-state placeholder — centered title + explanatory message.

Rendered in the middle of a page when there is no data to show yet
(e.g., the Live Session view before a session is selected). Keeps the
operator from staring at a literally blank surface.

Spec references:
  §4.E.1         — operator-facing UX affordances
  SPEC-AMEND-008 — PySide6 Operator Console
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class EmptyStateWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("EmptyState")

        self._title = QLabel("", self)
        self._title.setObjectName("EmptyStateTitle")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._message = QLabel("", self)
        self._message.setObjectName("EmptyStateMessage")
        self._message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._message.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(8)
        layout.addStretch(1)
        layout.addWidget(self._title)
        layout.addWidget(self._message)
        layout.addStretch(1)

    def set_title(self, title: str) -> None:
        self._title.setText(title)

    def set_message(self, message: str) -> None:
        self._message.setText(message)
