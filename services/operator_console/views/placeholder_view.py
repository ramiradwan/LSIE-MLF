"""Placeholder view for panels that are not yet implemented.

Kept deliberately trivial so the navigation shell can be exercised end
to end before the real panel classes land.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from services.operator_console.theme import PALETTE


class PlaceholderView(QWidget):
    def __init__(self, title: str, hint: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ContentSurface")

        heading = QLabel(title, self)
        heading.setStyleSheet(f"font-size: 18px; font-weight: 600; color: {PALETTE.text_primary};")

        subtitle = QLabel(hint or "Coming soon.", self)
        subtitle.setStyleSheet(f"color: {PALETTE.text_muted};")
        subtitle.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(6)
        layout.addWidget(heading)
        layout.addWidget(subtitle)
        layout.addStretch(1)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
