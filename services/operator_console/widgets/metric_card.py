"""Reusable metric card — title, primary value, secondary line, status pill.

Used on Overview (six cards: Active Session, Experiment, Physiology,
Health, Latest Encounter, Attention) and on the Live Session detail
pane for the reward-explanation breakdown.

The card is deliberately layout-only: it takes plain strings and a
`UiStatusKind`; the viewmodel/view layer is responsible for translating
DTOs into those strings (see `formatters.py`).

Spec references:
  §4.E.1         — operator-facing overview page
  §7B            — reward-explanation card on the Live Session surface
  SPEC-AMEND-008 — PySide6 Operator Console
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from packages.schemas.operator_console import UiStatusKind
from services.operator_console.widgets.status_pill import StatusPill


class MetricCard(QFrame):
    """A titled metric tile with optional status pill and click signal.

    Emits `clicked` only when `set_clickable(True)` is in effect, so a
    non-interactive card does not steal mouse events from its container.
    """

    clicked = Signal()

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("MetricCard")
        # Panel styling lives in `theme.STYLESHEET`; we only set the
        # object name so the stylesheet can target us.
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._clickable = False

        self._title = QLabel(title, self)
        self._title.setObjectName("MetricCardTitle")
        self._primary = QLabel("—", self)
        self._primary.setObjectName("MetricCardPrimary")
        self._primary.setWordWrap(True)
        self._secondary = QLabel("", self)
        self._secondary.setObjectName("MetricCardSecondary")
        self._secondary.setWordWrap(True)
        self._secondary.setVisible(False)
        self._status = StatusPill(self)
        self._status.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)
        layout.addWidget(self._title)
        layout.addWidget(self._primary)
        layout.addWidget(self._secondary)
        layout.addWidget(self._status)
        layout.addStretch(1)

    # ---- setters -------------------------------------------------------

    def set_title(self, text: str) -> None:
        self._title.setText(text)

    def set_primary_text(self, text: str) -> None:
        self._primary.setText(text)

    def set_secondary_text(self, text: str) -> None:
        self._secondary.setText(text)
        self._secondary.setVisible(bool(text))

    def set_status(self, kind: UiStatusKind, text: str | None = None) -> None:
        """Show the embedded pill with the given kind and optional text.

        Passing `text=None` hides the pill entirely — useful for cards
        that have nothing to report at a glance.
        """
        self._status.set_kind(kind)
        if text is None:
            self._status.set_text("")
            self._status.setVisible(False)
        else:
            self._status.set_text(text)
            self._status.setVisible(True)

    def set_clickable(self, clickable: bool) -> None:
        self._clickable = clickable
        if clickable:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.unsetCursor()

    # ---- events --------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt override
        if self._clickable and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)
