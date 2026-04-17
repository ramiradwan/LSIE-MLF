"""Shell window — sidebar navigation + stacked content surface.

The nav-sidebar + QStackedWidget layout keeps panels isolated. Adding a
new section is a three-line change: create the widget, add a nav button,
append to the stack. Panels expose an optional ``shutdown()`` method
which the window calls on close so polling threads exit cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from services.operator_console.config import ConsoleConfig
from services.operator_console.theme import PALETTE
from services.operator_console.views.placeholder_view import PlaceholderView
from services.operator_console.views.sessions_view import SessionsView


@dataclass(frozen=True)
class _NavEntry:
    label: str
    widget: QWidget


class MainWindow(QMainWindow):
    def __init__(self, config: ConsoleConfig) -> None:
        super().__init__()
        self._config = config
        self.setWindowTitle("LSIE-MLF Operator Console")
        self.resize(1280, 800)

        sessions_view = SessionsView(config, self)
        experiments_view = PlaceholderView(
            "Experiments",
            "Thompson Sampling posteriors and per-arm encounter summaries. "
            "Coming in the next pass.",
        )
        physiology_view = PlaceholderView(
            "Physiology",
            "Per-role RMSSD, heart rate, freshness, and Co-Modulation Index. "
            "Coming in the next pass.",
        )
        device_view = PlaceholderView(
            "Device",
            "USB device status, Oura OAuth, and external-telemetry connectors. "
            "Future operator-facing configuration surface.",
        )

        self._entries: list[_NavEntry] = [
            _NavEntry("Sessions", sessions_view),
            _NavEntry("Experiments", experiments_view),
            _NavEntry("Physiology", physiology_view),
            _NavEntry("Device", device_view),
        ]

        self._stack = QStackedWidget(self)
        for entry in self._entries:
            self._stack.addWidget(entry.widget)

        sidebar = self._build_sidebar()

        content = QWidget(self)
        content.setObjectName("ContentSurface")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(self._stack)

        root = QWidget(self)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(sidebar)
        root_layout.addWidget(content, stretch=1)
        self.setCentralWidget(root)

        status_bar = QStatusBar(self)
        api_label = QLabel(f"API · {config.api_base_url}", status_bar)
        api_label.setObjectName("StatusBarLabel")
        status_bar.addWidget(api_label)
        self.setStatusBar(status_bar)

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame(self)
        sidebar.setObjectName("SidebarNav")
        sidebar.setFixedWidth(220)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(4)

        title = QLabel("LSIE-MLF", sidebar)
        title.setStyleSheet(
            "font-size: 16px; font-weight: 700; "
            f"color: {PALETTE.text_primary}; padding: 0 6px 4px 6px;"
        )
        subtitle = QLabel("Operator Console", sidebar)
        subtitle.setStyleSheet(f"color: {PALETTE.text_muted}; padding: 0 6px 18px 6px;")

        layout.addWidget(title)
        layout.addWidget(subtitle)

        group = QButtonGroup(sidebar)
        group.setExclusive(True)
        for index, entry in enumerate(self._entries):
            btn = QPushButton(entry.label, sidebar)
            btn.setObjectName("NavButton")
            btn.setCheckable(True)
            if index == 0:
                btn.setChecked(True)
            btn.clicked.connect(lambda _=False, idx=index: self._stack.setCurrentIndex(idx))
            group.addButton(btn, index)
            layout.addWidget(btn)

        layout.addStretch(1)
        return sidebar

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 — Qt override
        for entry in self._entries:
            shutdown = getattr(entry.widget, "shutdown", None)
            if callable(shutdown):
                shutdown()
        event.accept()
