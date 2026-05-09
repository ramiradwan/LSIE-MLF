"""First-run setup UI tests."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from services.desktop_launcher.install_manager import InstallerSignals, LauncherPaths
from services.desktop_launcher.ui import SetupWindow

pytestmark = pytest.mark.usefixtures("qt_app")


class _FakeManager:
    def __init__(self) -> None:
        self.signals = InstallerSignals()
        self.paths = LauncherPaths(
            base_dir=Path("base"),
            downloads_dir=Path("base/downloads"),
            staging_dir=Path("base/runtime.staging"),
            active_runtime_dir=Path("base/runtime"),
            repo_root=Path("repo"),
        )
        self.started = False

    def start(self) -> None:
        self.started = True


def test_setup_window_launch_existing_install_skips_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _FakeManager()
    monkeypatch.setattr(
        "services.desktop_launcher.install_manager.has_current_runtime",
        lambda runtime_dir: runtime_dir == manager.paths.active_runtime_dir,
    )
    window = SetupWindow(manager=manager)  # type: ignore[arg-type]

    assert window.launch_existing_install() is True

    assert manager.started is False
    assert window.status_label.text() == "LSIE-MLF is installed and ready to launch."
    assert window.progress_bar.value() == 100
    assert window.launch_button.isHidden() is False
    assert window.reinstall_button.isHidden() is False


def test_setup_window_launch_existing_install_missing_runtime_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _FakeManager()
    monkeypatch.setattr(
        "services.desktop_launcher.install_manager.has_current_runtime",
        lambda _runtime_dir: False,
    )
    window = SetupWindow(manager=manager)  # type: ignore[arg-type]

    assert window.launch_existing_install() is False
    assert window.launch_button.isHidden() is True


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


def test_setup_window_finished_exposes_launch_handoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launched: list[Path] = []

    class FakeProcess:
        def poll(self) -> int | None:
            return None

    def fake_launch(runtime_dir: Path, app_root: Path | None = None) -> FakeProcess:
        del app_root
        launched.append(runtime_dir)
        return FakeProcess()

    monkeypatch.setattr(
        "services.desktop_launcher.health_check.launch_desktop_app",
        fake_launch,
    )
    manager = _FakeManager()
    window = SetupWindow(manager=manager)  # type: ignore[arg-type]

    manager.signals.finished.emit(tmp_path)

    assert launched == []
    assert window.status_label.text() == "Setup complete! Ready to launch."
    assert window.progress_bar.value() == 100
    assert window.launch_button.isHidden() is False

    window.execute_handoff()

    assert launched == [tmp_path]


def test_setup_window_handoff_failure_keeps_window_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_launch(_runtime_dir: Path, app_root: Path | None = None) -> subprocess.Popen[str]:
        del app_root
        raise RuntimeError("missing services.desktop_app")

    monkeypatch.setattr(
        "services.desktop_launcher.health_check.launch_desktop_app",
        fail_launch,
    )
    manager = _FakeManager()
    window = SetupWindow(manager=manager)  # type: ignore[arg-type]
    window.show()
    manager.signals.finished.emit(tmp_path)

    window.execute_handoff()

    assert window.isHidden() is False
    assert window.status_label.text() == "Launch failed"
    assert window.launch_button.isEnabled() is True
    assert "missing services.desktop_app" in window.log_view.toPlainText()


def test_setup_window_immediate_exit_keeps_window_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeProcess:
        def poll(self) -> int:
            return 1

    monkeypatch.setattr(
        "services.desktop_launcher.health_check.launch_desktop_app",
        lambda _runtime_dir, app_root=None: FakeProcess(),
    )
    manager = _FakeManager()
    window = SetupWindow(manager=manager)  # type: ignore[arg-type]
    window.show()
    manager.signals.finished.emit(tmp_path)

    window.execute_handoff()

    assert window.isHidden() is False
    assert window.status_label.text() == "Launch failed"
    assert window.launch_button.isEnabled() is True
    assert "Desktop app exited during startup with code 1" in window.log_view.toPlainText()


def test_main_smoke_delegates_to_desktop_app_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    smoke_calls: list[list[str]] = []

    def fake_main(argv: Sequence[str] | None = None) -> int:
        smoke_calls.append([] if argv is None else list(argv))
        return 0

    monkeypatch.setattr("services.desktop_app.__main__.main", fake_main)

    from services.desktop_launcher import ui

    assert ui.main(["--smoke"]) == 0
    assert smoke_calls == [["--smoke"]]
