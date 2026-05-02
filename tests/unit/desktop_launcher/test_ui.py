"""WS1 P2 — first-run setup UI tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.desktop_launcher.install_manager import InstallerSignals
from services.desktop_launcher.ui import SetupWindow

pytestmark = pytest.mark.usefixtures("qt_app")


class _FakeManager:
    def __init__(self) -> None:
        self.signals = InstallerSignals()
        self.started = False

    def start(self) -> None:
        self.started = True


def test_setup_window_starts_manager_and_updates_progress() -> None:
    manager = _FakeManager()
    window = SetupWindow(manager=manager)  # type: ignore[arg-type]

    window.start_install()
    manager.signals.status_changed.emit("Downloading scrcpy")
    manager.signals.progress_changed.emit(42)
    manager.signals.log_line.emit("downloaded chunk")

    assert manager.started is True
    assert window.status_label.text() == "Downloading scrcpy"
    assert window.progress_bar.value() == 42
    assert "downloaded chunk" in window.log_view.toPlainText()


def test_setup_window_failure_exposes_retry() -> None:
    manager = _FakeManager()
    window = SetupWindow(manager=manager)  # type: ignore[arg-type]

    manager.signals.failed.emit("network down")

    assert window.retry_button.isHidden() is False
    assert "network down" in window.log_view.toPlainText()


def test_setup_window_finished_launches_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launched: list[Path] = []
    monkeypatch.setattr(
        "services.desktop_launcher.health_check.launch_desktop_app",
        lambda runtime_dir: launched.append(runtime_dir),
    )
    manager = _FakeManager()
    window = SetupWindow(manager=manager)  # type: ignore[arg-type]

    manager.signals.finished.emit(tmp_path)

    assert launched == [tmp_path]
    assert window.progress_bar.value() == 100
