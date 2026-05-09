"""Slide-over container — right-side modal-to-page panel.

Hosts deliberate edit affordances (e.g., experiment management) so a
read-heavy page can stay pure readback by default. The slide-over is
modal to the parent page but never to the application: a click outside
closes it, Escape closes it, and `close()` closes it too. The widget is
purely presentational; consumers parent their existing form widgets
into the body and connect their own save/cancel signals.

Spec references:
  §4.E.1         — operator-facing readback / management split
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_SLIDE_OVER_WIDTH = 320


class SlideOver(QFrame):
    """Right-anchored slide-over panel.

    The widget paints two layers: a dim scrim covering the page and a
    framed panel anchored to the right edge that hosts the supplied
    body widget. The scrim itself is what catches clicks-outside and
    closes the panel; consumers do not need to wire that path.
    """

    closed = Signal()

    def __init__(
        self,
        title: str,
        parent: QWidget,
        *,
        body: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ExperimentsSlideOverScrim")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setVisible(False)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._panel = QFrame(self)
        self._panel.setObjectName("ExperimentsSlideOver")
        self._panel.setFrameShape(QFrame.Shape.NoFrame)
        self._panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._title = QLabel(title, self._panel)
        self._title.setObjectName("PanelTitle")
        self._close_button = QPushButton("✕", self._panel)
        self._close_button.setObjectName("ActionBarNoteToggle")
        self._close_button.setAccessibleName("Close panel")
        self._close_button.setAccessibleDescription("Close the slide-over.")
        self._close_button.setFlat(True)
        self._close_button.setFixedWidth(32)
        self._close_button.clicked.connect(self.close)

        self._body_container = QWidget(self._panel)
        self._body_layout = QVBoxLayout(self._body_container)
        self._body_layout.setContentsMargins(16, 8, 16, 16)
        self._body_layout.setSpacing(10)
        if body is not None:
            self._body_layout.addWidget(body)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(16, 12, 16, 8)
        title_row.setSpacing(8)
        title_row.addWidget(self._title, 1)
        title_row.addWidget(self._close_button)

        panel_layout = QVBoxLayout(self._panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)
        panel_layout.addLayout(title_row)
        panel_layout.addWidget(self._body_container, 1)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._panel.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._installed = False

    def set_body(self, body: QWidget) -> None:
        while self._body_layout.count() > 0:
            item = self._body_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        self._body_layout.addWidget(body)
        body.setParent(self._body_container)
        body.setVisible(True)

    def open(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        if not self._installed:
            parent.installEventFilter(self)
            self._installed = True
        self.setGeometry(0, 0, parent.width(), parent.height())
        self._reposition_panel()
        self.raise_()
        self.setVisible(True)
        self.setFocus(Qt.FocusReason.OtherFocusReason)

    def close(self) -> None:  # type: ignore[override]
        if not self.isVisible():
            return
        self.setVisible(False)
        self.closed.emit()

    def is_open(self) -> bool:
        return self.isVisible()

    # ---- events --------------------------------------------------------

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Resize and watched is self.parentWidget():
            parent = self.parentWidget()
            if parent is not None:
                self.setGeometry(0, 0, parent.width(), parent.height())
                self._reposition_panel()
        return super().eventFilter(watched, event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt override
        # Click outside the panel closes the slide-over. Inside the panel
        # children consume the event before this handler.
        if not self._panel.geometry().contains(event.position().toPoint()):
            self.close()
            event.accept()
            return
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 — Qt override
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)

    # ---- internals -----------------------------------------------------

    def _reposition_panel(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        width = min(_SLIDE_OVER_WIDTH, max(240, parent.width() - 80))
        self._panel.setGeometry(parent.width() - width, 0, width, parent.height())
