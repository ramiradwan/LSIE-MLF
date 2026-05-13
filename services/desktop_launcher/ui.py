"""PySide6 first-run setup UI for desktop runtime hydration."""

from __future__ import annotations

import sys
import time
from argparse import ArgumentParser
from collections.abc import Sequence
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

from services.desktop_launcher import health_check, install_manager
from services.desktop_launcher.install_manager import InstallManager
from services.operator_console.design_system.qss_builder import install_setup_stylesheet


class SetupWindow(QMainWindow):
    def __init__(self, manager: InstallManager | None = None) -> None:
        super().__init__()
        self.manager = manager or InstallManager()
        self._final_runtime_dir: Path | None = None
        self._final_app_root: Path | None = None
        self.setWindowTitle("LSIE-MLF Setup")
        self.setMinimumSize(680, 460)

        self.title_label = QLabel("Preparing LSIE-MLF")
        self.title_label.setObjectName("SetupTitle")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.status_label = QLabel("Downloading the desktop runtime…")
        self.status_label.setObjectName("SetupStatus")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("SetupProgress")
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

        self.launch_button = QPushButton("Launch LSIE-MLF")
        self.launch_button.setObjectName("SetupLaunch")
        self.launch_button.setVisible(False)
        self.launch_button.clicked.connect(self.execute_handoff)

        self.reinstall_button = QPushButton("Reinstall runtime")
        self.reinstall_button.setObjectName("SetupReinstall")
        self.reinstall_button.setVisible(False)
        self.reinstall_button.clicked.connect(self.start_install)

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
        panel_layout.addWidget(self.launch_button, alignment=Qt.AlignmentFlag.AlignCenter)
        panel_layout.addWidget(self.reinstall_button, alignment=Qt.AlignmentFlag.AlignCenter)

        root = QWidget()
        root.setObjectName("SetupRoot")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(36, 36, 36, 36)
        root_layout.addWidget(panel)
        self.setCentralWidget(root)
        install_setup_stylesheet(self)

        signals = self.manager.signals
        signals.status_changed.connect(self.set_status)
        signals.progress_changed.connect(self.set_progress)
        signals.log_line.connect(self.append_log)
        signals.failed.connect(self.install_failed)
        signals.finished.connect(self.install_finished)

    def start_install(self) -> None:
        self._final_runtime_dir = None
        self._final_app_root = None
        self.retry_button.setVisible(False)
        self.launch_button.setVisible(False)
        self.launch_button.setEnabled(True)
        self.reinstall_button.setVisible(False)
        self.progress_bar.setValue(0)
        self.log_view.clear()
        self.set_status("Starting setup…")
        self.manager.start()

    def launch_existing_install(self) -> bool:
        runtime_dir = self.manager.paths.active_runtime_dir
        if not install_manager.has_current_runtime(runtime_dir):
            return False
        self._final_runtime_dir = runtime_dir
        self._final_app_root = self.manager.paths.repo_root
        self.progress_bar.setValue(100)
        self.set_status("LSIE-MLF is installed and ready to launch.")
        self.launch_button.setVisible(True)
        self.launch_button.setEnabled(True)
        self.reinstall_button.setVisible(True)
        return True

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
        self._final_runtime_dir = runtime_dir
        self._final_app_root = self.manager.paths.repo_root
        self.set_status("Setup complete! Ready to launch.")
        self.progress_bar.setValue(100)
        self.launch_button.setVisible(True)

    def execute_handoff(self) -> None:
        if self._final_runtime_dir is None:
            return
        self.launch_button.setEnabled(False)
        self.set_status("Launching LSIE-MLF…")
        try:
            process = health_check.launch_desktop_app(
                self._final_runtime_dir,
                app_root=self._final_app_root,
            )
        except Exception as exc:
            self.launch_button.setEnabled(True)
            self.set_status("Launch failed")
            self.append_log(str(exc))
            return
        time.sleep(0.25)
        returncode = process.poll()
        if returncode is not None:
            self.launch_button.setEnabled(True)
            self.set_status("Launch failed")
            self.append_log(f"Desktop app exited during startup with code {returncode}")
            self.append_log("See desktop-launch.log in the LSIE-MLF app-data logs folder.")
            return
        self.close()


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="python -m services.desktop_launcher.ui")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Validate launcher preflight and desktop-app handoff without opening the setup UI.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoke:
        from services.desktop_app.__main__ import main as desktop_app_main

        return desktop_app_main(["--smoke"])
    app = QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication(sys.argv)
        created_app = True
    window = SetupWindow()
    window.show()
    if not window.launch_existing_install():
        window.start_install()
    if created_app:
        return app.exec()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
