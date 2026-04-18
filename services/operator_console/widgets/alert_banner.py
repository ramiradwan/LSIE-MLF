"""Alert banner — colour-coded by severity, hidden when no alert.

A thin presentation widget that shows the currently most-urgent alert
from the attention queue. Severity drives colour + icon glyph only;
the full alert feed is rendered by the Health view's timeline.

Spec references:
  §12            — subsystem alert categories drive the §12-aligned
                   severity palette
  §4.E.1         — operator-facing attention queue
  SPEC-AMEND-008 — PySide6 Operator Console
"""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget

from packages.schemas.operator_console import AlertSeverity

# Unicode glyphs so we don't need an icon font in the operator host.
_SEVERITY_GLYPH: dict[AlertSeverity, str] = {
    AlertSeverity.INFO: "ℹ",
    AlertSeverity.WARNING: "⚠",
    AlertSeverity.CRITICAL: "✖",
}

# Object-name suffix drives the stylesheet rule in `theme.STYLESHEET`.
_SEVERITY_SUFFIX: dict[AlertSeverity, str] = {
    AlertSeverity.INFO: "Info",
    AlertSeverity.WARNING: "Warning",
    AlertSeverity.CRITICAL: "Critical",
}


class AlertBanner(QFrame):
    """Single-line alert ribbon. Hidden by default and when cleared."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("AlertBanner")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setVisible(False)

        self._glyph = QLabel("", self)
        self._glyph.setObjectName("AlertBannerGlyph")
        self._message = QLabel("", self)
        self._message.setObjectName("AlertBannerMessage")
        self._message.setWordWrap(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)
        layout.addWidget(self._glyph)
        layout.addWidget(self._message, 1)

    def set_alert(
        self,
        severity: AlertSeverity | None,
        message: str | None,
    ) -> None:
        """Show an alert, or hide the banner when severity is None."""
        if severity is None or not message:
            self.setVisible(False)
            self._message.setText("")
            self._glyph.setText("")
            # Reset object name so the stylesheet can't keep colouring
            # a hidden widget.
            self.setObjectName("AlertBanner")
            return

        self._glyph.setText(_SEVERITY_GLYPH[severity])
        self._message.setText(message)
        self.setObjectName(f"AlertBanner{_SEVERITY_SUFFIX[severity]}")
        # Re-polish so the new object name's QSS rule kicks in.
        style = self.style()
        if style is not None:
            style.unpolish(self)
            style.polish(self)
        self.setVisible(True)
