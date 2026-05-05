"""Static Module A capture-supervisor contract checks."""

from __future__ import annotations

from pathlib import Path

from services.desktop_app.processes import capture_supervisor
from services.desktop_app.processes.capture_supervisor import (
    AUDIO_SCRCPY_PORT_RANGE,
    DEVICE_POLL_INTERVAL_S,
    DEVICE_POLL_TIMEOUT_S,
    RESTART_BACKOFF_S,
    VIDEO_SCRCPY_PORT_RANGE,
    CaptureLayout,
    _build_audio_scrcpy_args,
    _build_video_scrcpy_args,
)

CAPTURE_SUPERVISOR_PATH = Path("services/desktop_app/processes/capture_supervisor.py")
RETIRED_ENTRYPOINT_PATH = Path("services/stream_ingest/entrypoint.sh")


def _layout(tmp_path: Path) -> CaptureLayout:
    return CaptureLayout(
        capture_dir=tmp_path,
        audio_path=tmp_path / "audio_stream.wav",
        video_path=tmp_path / "video_stream.mkv",
    )


def test_retired_stream_ingest_entrypoint_is_not_restored() -> None:
    assert not RETIRED_ENTRYPOINT_PATH.exists()


def test_capture_supervisor_is_current_module_a_surface() -> None:
    source = CAPTURE_SUPERVISOR_PATH.read_text(encoding="utf-8")

    assert "Owns the desktop capture lifecycle" in source
    assert "§9.1, §9.3, and §12" in source
    assert "cleanup_capture_files(layout.capture_dir)" in source
    assert "record_capture_pid" in source


def test_hardware_device_loss_cadence_matches_section_12() -> None:
    assert DEVICE_POLL_INTERVAL_S == 2.0
    assert DEVICE_POLL_TIMEOUT_S == 60.0
    assert RESTART_BACKOFF_S == 2.0


def test_audio_scrcpy_args_are_headless_raw_wav(tmp_path: Path) -> None:
    args = _build_audio_scrcpy_args("scrcpy", _layout(tmp_path))

    assert args[0] == "scrcpy"
    assert "--no-video" in args
    assert "--no-playback" in args
    assert "--no-window" in args
    assert "--audio-codec=raw" in args
    assert "--audio-buffer=30" in args
    assert "--audio-dup" not in args
    assert "--record-format=wav" in args
    assert f"--port={AUDIO_SCRCPY_PORT_RANGE}" in args
    assert f"--record={tmp_path / 'audio_stream.wav'}" in args


def test_video_scrcpy_args_are_headless_h264_mkv(tmp_path: Path) -> None:
    args = _build_video_scrcpy_args("scrcpy", _layout(tmp_path))

    assert args[0] == "scrcpy"
    assert "--no-audio" in args
    assert "--no-playback" in args
    assert "--no-window" in args
    assert "--video-codec=h264" in args
    assert "--max-fps=30" in args
    assert "--record-format=mkv" in args
    assert f"--port={VIDEO_SCRCPY_PORT_RANGE}" in args
    assert f"--record={tmp_path / 'video_stream.mkv'}" in args


def test_capture_supervisor_uses_interruptible_shutdown_waits() -> None:
    source = CAPTURE_SUPERVISOR_PATH.read_text(encoding="utf-8")

    assert "shutdown_event.wait(timeout=SCRCPY_STAGGER_S)" in source
    assert "shutdown_event.wait(timeout=RESTART_BACKOFF_S)" in source
    assert "shutdown_event.wait(timeout=0.5)" in source
    assert "while not shutdown_event.is_set()" in source


def test_capture_supervisor_exports_expected_runtime_symbols() -> None:
    assert capture_supervisor.SQLITE_FILENAME == "desktop.sqlite"
    assert capture_supervisor.AUDIO_SCRCPY_PORT_RANGE == "27100:27199"
    assert capture_supervisor.VIDEO_SCRCPY_PORT_RANGE == "27200:27299"
