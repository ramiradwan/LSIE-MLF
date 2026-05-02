"""PySide6 first-run setup UI for desktop runtime hydration."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from services.desktop_launcher import health_check
from services.desktop_launcher.install_manager import InstallManager


class SetupWindow(QMainWindow):
    def __init__(self, manager: InstallManager | None = None) -> None:
        super().__init__()
        self.manager = manager or InstallManager()
        self.setWindowTitle("LSIE-MLF Setup")
        self.setMinimumSize(680, 460)

        self.title_label = QLabel("Preparing LSIE-MLF")
        self.title_label.setObjectName("SetupTitle")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.status_label = QLabel("Downloading the desktop runtime…")
        self.status_label.setObjectName("SetupStatus")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("SetupLog")
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(400)

        self.retry_button = QPushButton("Retry setup")
        self.retry_button.setObjectName("SetupRetry")
        self.retry_button.setVisible(False)
        self.retry_button.clicked.connect(self.start_install)

        panel = QFrame()
        panel.setObjectName("SetupPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(32, 28, 32, 28)
        panel_layout.setSpacing(18)
        panel_layout.addWidget(self.title_label)
        panel_layout.addWidget(self.status_label)
        panel_layout.addWidget(self.progress_bar)
        panel_layout.addWidget(self.log_view)
        panel_layout.addWidget(self.retry_button, alignment=Qt.AlignmentFlag.AlignCenter)

        root = QWidget()
        root.setObjectName("SetupRoot")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(36, 36, 36, 36)
        root_layout.addWidget(panel)
        self.setCentralWidget(root)
        self.setStyleSheet(build_setup_stylesheet())

        signals = self.manager.signals
        signals.status_changed.connect(self.set_status)
        signals.progress_changed.connect(self.set_progress)
        signals.log_line.connect(self.append_log)
        signals.failed.connect(self.install_failed)
        signals.finished.connect(self.install_finished)

    def start_install(self) -> None:
        self.retry_button.setVisible(False)
        self.progress_bar.setValue(0)
        self.log_view.clear()
        self.set_status("Starting setup…")
        self.manager.start()

    def set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def set_progress(self, value: int) -> None:
        self.progress_bar.setValue(max(0, min(100, value)))

    def append_log(self, line: str) -> None:
        if line:
            self.log_view.appendPlainText(line)

    def install_failed(self, message: str) -> None:
        self.set_status("Setup needs attention")
        self.append_log(message)
        self.retry_button.setVisible(True)

    def install_finished(self, runtime_dir: Path) -> None:
        self.set_status("Setup complete — launching LSIE-MLF")
        self.progress_bar.setValue(100)
        health_check.launch_desktop_app(runtime_dir)
        self.close()


def build_setup_stylesheet() -> str:
    return """
QWidget#SetupRoot {
    background: #0f1115;
}
QFrame#SetupPanel {
    background: #171a21;
    border: 1px solid #262a33;
    border-radius: 14px;
}
QLabel#SetupTitle {
    color: #e6e8ed;
    font-size: 24px;
    font-weight: 700;
}
QLabel#SetupStatus {
    color: #a6adba;
    font-size: 14px;
}
QProgressBar {
    background: #10131a;
    border: 1px solid #303542;
    border-radius: 8px;
    color: #e6e8ed;
    min-height: 24px;
    text-align: center;
}
QProgressBar::chunk {
    background: #5b8def;
    border-radius: 7px;
}
QPlainTextEdit#SetupLog {
    background: #10131a;
    border: 1px solid #303542;
    border-radius: 8px;
    color: #c8ced8;
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 12px;
    padding: 8px;
}
QPushButton#SetupRetry {
    background: #5b8def;
    border: none;
    border-radius: 6px;
    color: white;
    font-weight: 600;
    padding: 8px 18px;
}
"""


def main() -> int:
    app = QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication(sys.argv)
        created_app = True
    window = SetupWindow()
    window.show()
    window.start_install()
    if created_app:
        return app.exec()
    return 0


if __name__ == "__main__":
    sys.exit(main())
