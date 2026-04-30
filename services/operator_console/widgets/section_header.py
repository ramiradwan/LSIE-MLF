"""Section header — title + optional subtitle.

Used at the top of each page and above grouped metric cards so the
operator can scan for the section they need. Layout only; no logic.

Spec references:
  §4.E.1         — operator-facing page structure
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class SectionHeader(QWidget):
    def __init__(
        self,
        title: str,
        subtitle: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("SectionHeader")

        self._title = QLabel(title, self)
        self._title.setObjectName("SectionHeaderTitle")
        self._subtitle = QLabel(subtitle or "", self)
        self._subtitle.setObjectName("SectionHeaderSubtitle")
        self._subtitle.setWordWrap(True)
        self._subtitle.setVisible(bool(subtitle))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._title)
        layout.addWidget(self._subtitle)

    def set_title(self, title: str) -> None:
        self._title.setText(title)

    def set_subtitle(self, subtitle: str | None) -> None:
        self._subtitle.setText(subtitle or "")
        self._subtitle.setVisible(bool(subtitle))
