"""Section header — title + optional subtitle.

Used at the top of each page and above grouped metric cards so the
operator can scan for the section they need. Layout only; no logic.

The `level` parameter expresses three explicit hierarchy steps:
  * `page`  — the page heading at the top of a route
  * `panel` — the heading at the top of a content panel
  * `sub`   — a muted divider within a panel (e.g., grouping cards)

Every level uses the same widget so a future heading change ripples to
one place. The label's `level` Qt property drives QSS sizing.

Spec references:
  §4.E.1         — operator-facing page structure
"""

from __future__ import annotations

from typing import Literal

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from services.operator_console.design_system.qss_builder import repolish

SectionHeaderLevel = Literal["page", "panel", "sub"]


class SectionHeader(QWidget):
    def __init__(
        self,
        title: str,
        subtitle: str | None = None,
        parent: QWidget | None = None,
        *,
        level: SectionHeaderLevel = "panel",
    ) -> None:
        super().__init__(parent)
        self.setObjectName("SectionHeader")

        self._title = QLabel(title, self)
        self._title.setObjectName("SectionHeaderTitle")
        self._subtitle = QLabel(subtitle or "", self)
        self._subtitle.setObjectName("SectionHeaderSubtitle")
        self._subtitle.setWordWrap(True)
        self._subtitle.setVisible(bool(subtitle))
        self.setAccessibleName(title)
        self.setAccessibleDescription(subtitle or "")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._title)
        layout.addWidget(self._subtitle)

        self._level: SectionHeaderLevel = level
        self._apply_level()

    def set_title(self, title: str) -> None:
        self._title.setText(title)
        self.setAccessibleName(title)

    def set_subtitle(self, subtitle: str | None) -> None:
        self._subtitle.setText(subtitle or "")
        self._subtitle.setVisible(bool(subtitle))
        self.setAccessibleDescription(subtitle or "")

    def set_level(self, level: SectionHeaderLevel) -> None:
        if level == self._level:
            return
        self._level = level
        self._apply_level()

    def level(self) -> SectionHeaderLevel:
        return self._level

    def _apply_level(self) -> None:
        self._title.setProperty("level", self._level)
        repolish(self._title)
