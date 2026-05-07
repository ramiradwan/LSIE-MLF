"""Alert banner — colour-coded by severity, hidden when no alert.

A thin presentation widget that shows the currently most-urgent alert
from the attention queue. Severity drives colour + icon glyph only;
the full alert feed is rendered by the Health view's timeline.

Severity icons render through `QSvgWidget` so the glyph is recognisable
at a glance and does not depend on a font fallback chain — peripheral
recognition matters more than typographic compactness for an alerting
surface.

Spec references:
  §12            — subsystem alert categories drive the §12-aligned
                   severity palette
  §4.E.1         — operator-facing attention queue
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget

from packages.schemas.operator_console import AlertSeverity

_ICON_DIR = Path(__file__).resolve().parent.parent / "resources" / "icons"

# Object-name suffix drives the stylesheet rule in `theme.STYLESHEET`.
_SEVERITY_SUFFIX: dict[AlertSeverity, str] = {
    AlertSeverity.INFO: "Info",
    AlertSeverity.WARNING: "Warning",
    AlertSeverity.CRITICAL: "Critical",
}

_SEVERITY_ICON: dict[AlertSeverity, Path] = {
    AlertSeverity.INFO: _ICON_DIR / "alert_info.svg",
    AlertSeverity.WARNING: _ICON_DIR / "alert_warning.svg",
    AlertSeverity.CRITICAL: _ICON_DIR / "alert_critical.svg",
}

# Plain-text fallback so accessibility tooling and non-graphical readers
# still receive the severity word even when the SVG cannot paint.
_SEVERITY_TEXT: dict[AlertSeverity, str] = {
    AlertSeverity.INFO: "info",
    AlertSeverity.WARNING: "warning",
    AlertSeverity.CRITICAL: "critical",
}

_GLYPH_SIZE = QSize(16, 16)


class AlertBanner(QFrame):
    """Single-line alert ribbon. Hidden by default and when cleared."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("AlertBanner")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setVisible(False)

        self._glyph = QSvgWidget(self)
        self._glyph.setObjectName("AlertBannerGlyph")
        self._glyph.setFixedSize(_GLYPH_SIZE)
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
            self._glyph.load(b"")
            self._glyph.setAccessibleName("")
            self._glyph.setAccessibleDescription("")
            # Reset object name so the stylesheet can't keep colouring
            # a hidden widget.
            self.setObjectName("AlertBanner")
            return

        icon_path = _SEVERITY_ICON[severity]
        if icon_path.exists():
            self._glyph.load(str(icon_path))
        self._glyph.setAccessibleName(_SEVERITY_TEXT[severity])
        self._glyph.setAccessibleDescription(_SEVERITY_TEXT[severity])
        self._message.setText(message)
        self.setObjectName(f"AlertBanner{_SEVERITY_SUFFIX[severity]}")
        # Re-polish so the new object name's QSS rule kicks in.
        style = self.style()
        if style is not None:
            style.unpolish(self)
            style.polish(self)
        self.setVisible(True)
