"""QApplication bootstrap and ``main()`` entry point."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from services.operator_console.config import load_config
from services.operator_console.theme import STYLESHEET
from services.operator_console.views.main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    config = load_config()
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("LSIE-MLF Operator Console")
    app.setOrganizationName("[REDACTED]")
    app.setStyleSheet(STYLESHEET)

    window = MainWindow(config)
    window.show()
    return app.exec()
