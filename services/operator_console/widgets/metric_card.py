"""Reusable metric card — title, primary value, secondary line, status pill.

Used on Overview (six cards: Active Session, Experiment, Physiology,
Health, Latest Encounter, Attention) and on the Live Session detail
pane for the reward-explanation breakdown.

The card is deliberately layout-only: it takes plain strings and a
`UiStatusKind`; the viewmodel/view layer is responsible for translating
DTOs into those strings (see `formatters.py`).

Clickable cards expose a trailing chevron in the title row so the
affordance is visible without depending on cursor hover, and the QSS
hover/focus rules tint the border in the accent color.

Spec references:
  §4.E.1         — operator-facing overview page
  §7B            — reward-explanation card on the Live Session surface
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent, QMouseEvent, QResizeEvent
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from packages.schemas.operator_console import UiStatusKind
from services.operator_console.design_system.qss_builder import repolish
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
        self.setProperty("clickable", "false")
        # Panel styling lives in `theme.STYLESHEET`; we only set the
        # object name so the stylesheet can target us. The card grows
        # vertically when secondary text wraps, so the size policy on
        # the vertical axis must allow expansion past `minimumHeight`
        # and must declare `heightForWidth` so the QGridLayout queries
        # the wrapped height instead of the single-line sizeHint.
        self.setFrameShape(QFrame.Shape.NoFrame)
        frame_policy = QSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.MinimumExpanding,
        )
        frame_policy.setHeightForWidth(True)
        self.setSizePolicy(frame_policy)
        # 168px gives room for two-line secondary text + status pill at
        # the narrow band where cards squeeze to ~250px wide. Cards with
        # less content read evenly across a row instead of jagging.
        # Setting both `minimumSize` and `minimumHeight` is intentional —
        # QFrame's `minimumSizeHint()` otherwise reports the layout's
        # natural sum, and the parent QVBoxLayout/QGridLayout uses *that*
        # to squeeze the card below 168 when total content exceeds the
        # available column height.
        self.setMinimumSize(220, 168)

        self._clickable = False
        self._click_destination: str | None = None

        # `Ignored × MinimumExpanding` on each text label is the
        # canonical Qt recipe for `setWordWrap(True)` actually wrapping
        # inside a parent container — Ignored tells the layout the label
        # is happy at any width, MinimumExpanding lets the height grow
        # vertically to fit the wrapped text.
        text_policy = QSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.MinimumExpanding,
        )

        self._title = QLabel(title, self)
        self._title.setObjectName("MetricCardTitle")
        self._title.setWordWrap(True)
        self._title.setSizePolicy(text_policy)
        self._title.setMinimumWidth(0)
        self._chevron = QLabel("›", self)
        self._chevron.setObjectName("MetricCardChevron")
        self._chevron.setVisible(False)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(0)
        title_row.addWidget(self._title, 1)
        title_row.addWidget(self._chevron, 0, Qt.AlignmentFlag.AlignRight)

        self._primary = QLabel("—", self)
        self._primary.setObjectName("MetricCardPrimary")
        self._primary.setWordWrap(True)
        self._primary.setSizePolicy(text_policy)
        self._primary.setMinimumWidth(0)
        self._secondary = QLabel("", self)
        self._secondary.setObjectName("MetricCardSecondary")
        self._secondary.setWordWrap(True)
        self._secondary.setSizePolicy(text_policy)
        self._secondary.setMinimumWidth(0)
        self._secondary.setVisible(False)
        self._status = StatusPill(self)
        self._status.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)
        layout.addLayout(title_row)
        layout.addWidget(self._primary)
        layout.addWidget(self._secondary)
        layout.addWidget(self._status)
        # Trailing stretch absorbs any extra height; the card itself
        # still claims its minimumHeight floor so the row equalises.
        layout.addStretch(1)
        self._sync_accessibility()

    # ---- setters -------------------------------------------------------

    def set_title(self, text: str) -> None:
        self._title.setText(text)
        self._sync_accessibility()

    def set_primary_text(self, text: str) -> None:
        self._primary.setText(text)
        self._sync_accessibility()

    def set_secondary_text(self, text: str) -> None:
        self._secondary.setText(text)
        self._secondary.setVisible(bool(text))
        self._sync_accessibility()
        # QGridLayout in the parent ResponsiveMetricGrid does not query
        # heightForWidth when laying out rows — it sizes each row to the
        # children's `sizeHint`/`minimumSize`. Bump our minimum height
        # to the wrap-aware height for the current width so the row
        # actually grows when secondary text spans multiple lines.
        if text and self.width() > 0:
            self.setMinimumHeight(self.heightForWidth(self.width()))
        self._secondary.updateGeometry()
        self.updateGeometry()

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
        self._sync_accessibility()

    def set_clickable(
        self,
        clickable: bool,
        *,
        destination: str | None = None,
    ) -> None:
        self._clickable = clickable
        self._click_destination = destination if clickable else None
        self.setProperty("clickable", "true" if clickable else "false")
        self._chevron.setVisible(clickable)
        if clickable:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        else:
            self.unsetCursor()
            self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        repolish(self)
        self._sync_accessibility()

    def _sync_accessibility(self) -> None:
        title = self._title.text()
        primary = self._primary.text()
        secondary = self._secondary.text()
        status = self._status.text()
        self.setAccessibleName(title)
        description_parts = [part for part in (primary, secondary, status) if part]
        self.setAccessibleDescription(". ".join(description_parts))
        if self._clickable:
            if self._click_destination is not None:
                self.setToolTip(f"Open in {self._click_destination}")
            else:
                self.setToolTip(f"Open {title}")
        else:
            self.setToolTip("")

    # ---- events --------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt override
        if self._clickable and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 — Qt override
        activation_keys = (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space)
        if self._clickable and event.key() in activation_keys:
            self.clicked.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 — Qt override
        super().resizeEvent(event)
        # Cards live inside a QGridLayout that ignores `heightForWidth`
        # when sizing rows. When our width changes, recompute the
        # wrap-aware minimum height so the next layout pass gives us
        # enough vertical room for two-line secondary text.
        if self._secondary.isVisible() and event.size().width() > 0:
            wanted = self.heightForWidth(event.size().width())
            if wanted > self.minimumHeight():
                self.setMinimumHeight(wanted)
                self.updateGeometry()

    def hasHeightForWidth(self) -> bool:  # noqa: N802 — Qt override
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 — Qt override
        """Sum the wrap-aware heights of every child so the QGridLayout
        gives the card enough vertical room for wrapped secondary text.

        QFrame's default never asks the layout, so a wordWrap-enabled
        QLabel inside it never gets a chance to claim more vertical
        space. We compute the height inline here, accounting for the
        16px horizontal padding inside the layout.
        """

        margins = self.contentsMargins()
        text_width = max(width - 32, 1)  # 16px left + 16px right padding
        spacing = 6  # matches QVBoxLayout setSpacing
        rows = 0
        height = 0
        for label in (self._title, self._primary, self._secondary):
            if label.isHidden():
                continue
            height += label.heightForWidth(text_width)
            rows += 1
        if not self._status.isHidden():
            height += self._status.sizeHint().height()
            rows += 1
        if rows > 1:
            height += spacing * (rows - 1)
        height += margins.top() + margins.bottom() + 28  # 14px top + 14px bottom inner padding
        return max(height, self.minimumHeight())
