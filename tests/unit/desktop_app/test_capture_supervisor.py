"""capture_supervisor unit tests.

Validates the supervision contract without requiring a real Android
device or a real ``scrcpy`` install:

  - ``_wait_for_device`` returns True when ADB reports a device line.
  - ``_DriftPollThread`` ships ``drift_offset`` payloads onto
    ``IpcChannels.drift_updates`` and stops cleanly on shutdown.
  - The scrcpy command builders emit the expected dual-instance args
    (port ranges 27100:27199 / 27200:27299, audio raw + WAV record,
    video H.264 30 fps + MKV record).
  - ``_resolve_capture_layout`` honours ``LSIE_CAPTURE_DIR`` and falls
    back to ``%LOCALAPPDATA%`` / ``$HOME``.

Real-device + real-scrcpy supervision is exercised by
``tests/integration/desktop_app/test_capture_supervisor_smoke.py``
which only runs when ``LSIE_INTEGRATION_DEVICE=1``.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

from services.desktop_app.drift import DriftCorrector
from services.desktop_app.ipc import IpcChannels
from services.desktop_app.processes.capture_supervisor import (
    AUDIO_SCRCPY_PORT_RANGE,
    VIDEO_SCRCPY_PORT_RANGE,
    CaptureLayout,
    _build_audio_scrcpy_args,
    _build_video_scrcpy_args,
    _DriftPollThread,
    _resolve_capture_layout,
    _spawn_scrcpy_pair,
    _wait_for_device,
    run,
)
from services.desktop_app.state.sqlite_schema import bootstrap_schema


def _make_channels() -> IpcChannels:
    ctx = mp.get_context("spawn")
    return IpcChannels(
        ml_inbox=ctx.Queue(),
        drift_updates=ctx.Queue(),
        analytics_inbox=ctx.Queue(),
        pcm_acks=ctx.Queue(),
    )


def _bootstrap_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "desktop.sqlite"
    import sqlite3

    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        bootstrap_schema(conn)
    finally:
        conn.close()
    return db_path


def test_wait_for_device_returns_true_on_device_line(tmp_path: Path) -> None:
    devices_result = MagicMock(returncode=0)
    devices_result.stdout = "List of devices attached\n55281FDCR002LK\tdevice\n\n"
    model_result = MagicMock(returncode=0)
    model_result.stdout = "Pixel 8\n"
    window_result = MagicMock(returncode=0)
    window_result.stdout = (
        "mCurrentFocus=Window{abc u0 com.example.app/com.example.app.MainActivity}"
    )
    db_path = _bootstrap_db(tmp_path)
    with patch("subprocess.run", side_effect=[devices_result, model_result, window_result]):
        device = _wait_for_device("adb", db_path, deadline_s=2.0)
    assert device is not None
    assert device.serial == "55281FDCR002LK"
    assert device.model == "Pixel 8"
    assert device.active_app == "com.example.app"


def test_wait_for_device_times_out_when_no_device(tmp_path: Path) -> None:
    fake_result = MagicMock(returncode=0)
    fake_result.stdout = "List of devices attached\n\n"
    db_path = _bootstrap_db(tmp_path)
    with patch("subprocess.run", return_value=fake_result):
        assert _wait_for_device("adb", db_path, deadline_s=0.5) is None


def test_wait_for_device_returns_early_when_shutdown_is_requested(tmp_path: Path) -> None:
    shutdown = mp.get_context("spawn").Event()
    shutdown.set()
    db_path = _bootstrap_db(tmp_path)
    with patch("subprocess.run") as run_mock:
        assert _wait_for_device("adb", db_path, deadline_s=2.0, shutdown_event=shutdown) is None
    run_mock.assert_not_called()


def test_audio_scrcpy_args_match_spec() -> None:
    layout = CaptureLayout(
        capture_dir=Path("/tmp/lsie-mlf/capture"),
        audio_path=Path("/tmp/lsie-mlf/capture/audio_stream.wav"),
        video_path=Path("/tmp/lsie-mlf/capture/video_stream.mkv"),
    )
    args = _build_audio_scrcpy_args("scrcpy.exe", layout)

    assert args[0] == "scrcpy.exe"
    assert "--no-video" in args
    assert "--no-window" in args
    assert "--audio-codec=raw" in args
    assert "--audio-source=playback" in args
    assert "--audio-buffer=30" in args
    assert "--audio-dup" not in args
    assert f"--port={AUDIO_SCRCPY_PORT_RANGE}" in args
    assert any(arg.startswith("--record=") and "audio_stream.wav" in arg for arg in args)
    assert "--record-format=wav" in args


def test_video_scrcpy_args_match_spec() -> None:
    layout = CaptureLayout(
        capture_dir=Path("/tmp/lsie-mlf/capture"),
        audio_path=Path("/tmp/lsie-mlf/capture/audio_stream.wav"),
        video_path=Path("/tmp/lsie-mlf/capture/video_stream.mkv"),
    )
    args = _build_video_scrcpy_args("scrcpy.exe", layout)

    assert args[0] == "scrcpy.exe"
    assert "--no-audio" in args
    assert "--no-window" in args
    assert "--video-codec=h264" in args
    assert "--max-fps=30" in args
    assert f"--port={VIDEO_SCRCPY_PORT_RANGE}" in args
    assert any(arg.startswith("--record=") and "video_stream.mkv" in arg for arg in args)
    assert "--record-format=mkv" in args


def test_resolve_capture_layout_honours_env_override(tmp_path: Path, monkeypatch: Any) -> None:
    target = tmp_path / "custom-capture"
    monkeypatch.setenv("LSIE_CAPTURE_DIR", str(target))

    layout = _resolve_capture_layout()

    assert layout.capture_dir == target.resolve()
    assert target.is_dir()
    assert layout.audio_path == target.resolve() / "audio_stream.wav"
    assert layout.video_path == target.resolve() / "video_stream.mkv"


def test_resolve_capture_layout_falls_back_when_override_is_blank(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setenv("LSIE_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LSIE_CAPTURE_DIR", "   ")

    layout = _resolve_capture_layout()

    assert layout.capture_dir == (tmp_path / "capture").resolve()
    assert layout.audio_path == (tmp_path / "capture" / "audio_stream.wav").resolve()
    assert layout.video_path == (tmp_path / "capture" / "video_stream.mkv").resolve()


def test_drift_poll_thread_publishes_offsets() -> None:
    channels = _make_channels()
    corrector = DriftCorrector()
    corrector.drift_offset = 0.123

    shutdown = mp.get_context("spawn").Event()

    poll_called = MagicMock(return_value=0.123)
    corrector.poll = poll_called  # type: ignore[method-assign]

    thread = _DriftPollThread(
        corrector=corrector,
        channels=channels,
        shutdown_event=shutdown,
        interval_s=0.1,
    )
    thread.start()

    time.sleep(0.25)

    shutdown.set()
    thread.join(timeout=2.0)

    payloads: list[dict[str, float]] = []
    while True:
        try:
            raw = channels.drift_updates.get_nowait()
        except Exception:  # noqa: BLE001
            break
        assert isinstance(raw, dict)
        payloads.append(raw)

    assert len(payloads) >= 1
    assert payloads[0]["drift_offset"] == 0.123
    assert "ts_monotonic" in payloads[0]


def test_spawn_scrcpy_pair_aborts_during_stagger(tmp_path: Path) -> None:
    shutdown = mp.get_context("spawn").Event()
    layout = CaptureLayout(
        capture_dir=tmp_path,
        audio_path=tmp_path / "audio_stream.wav",
        video_path=tmp_path / "video_stream.mkv",
    )
    audio_proc = MagicMock()
    audio_proc.pid = 101

    def trigger_shutdown(timeout: float) -> bool:
        shutdown.set()
        return True

    with (
        patch(
            "services.desktop_app.processes.capture_supervisor.SupervisedProcess",
            side_effect=[audio_proc],
        ),
        patch("services.desktop_app.processes.capture_supervisor.record_capture_pid") as record_pid,
        patch("services.desktop_app.processes.capture_supervisor._kill_pair") as kill_pair,
        patch.object(shutdown, "wait", side_effect=trigger_shutdown),
    ):
        pair = _spawn_scrcpy_pair("scrcpy", layout, tmp_path / "desktop.sqlite", shutdown)

    assert pair is None
    record_pid.assert_called_once()
    kill_pair.assert_called_once_with(audio_proc, None, tmp_path / "desktop.sqlite")


def test_run_cleans_capture_files_on_startup_and_shutdown(tmp_path: Path) -> None:
    shutdown = mp.get_context("spawn").Event()
    shutdown.set()
    channels = _make_channels()
    layout = CaptureLayout(
        capture_dir=tmp_path,
        audio_path=tmp_path / "audio_stream.wav",
        video_path=tmp_path / "video_stream.mkv",
    )

    with (
        patch(
            "services.desktop_app.processes.capture_supervisor._resolve_capture_layout",
            return_value=layout,
        ),
        patch(
            "services.desktop_app.processes.capture_supervisor.cleanup_capture_files",
            side_effect=[([layout.audio_path], []), ([layout.video_path], [])],
        ) as cleanup,
        patch(
            "services.desktop_app.processes.capture_supervisor.HeartbeatRecorder",
        ) as heartbeat_cls,
        patch(
            "services.desktop_app.processes.capture_supervisor._DriftPollThread",
        ) as drift_cls,
        patch(
            "services.desktop_app.processes.capture_supervisor.find_executable",
            return_value="scrcpy",
        ),
    ):
        heartbeat = MagicMock()
        heartbeat_cls.return_value = heartbeat
        drift_thread = MagicMock()
        drift_cls.return_value = drift_thread

        run(shutdown_event=shutdown, channels=channels)

    assert cleanup.call_args_list == [call(layout.capture_dir), call(layout.capture_dir)]
    heartbeat.start.assert_called_once()
    heartbeat.stop.assert_called_once()
    drift_thread.start.assert_called_once()
    drift_thread.join.assert_called_once_with(timeout=5.0)


def test_kill_pair_keeps_manifest_entry_when_process_survives(tmp_path: Path) -> None:
    from services.desktop_app.processes.capture_supervisor import _kill_pair

    db_path = _bootstrap_db(tmp_path)
    proc = MagicMock()
    proc.pid = 101
    proc.is_alive.return_value = True

    proc.terminate_root.return_value = False

    with patch("services.desktop_app.processes.capture_supervisor.forget_capture_pid") as forget:
        _kill_pair(proc, None, db_path)

    proc.terminate_root.assert_called_once_with(grace_s=3.0)
    proc.terminate.assert_called_once_with(grace_s=3.0)
    forget.assert_not_called()


def test_kill_pair_forgets_manifest_entry_when_process_exits(tmp_path: Path) -> None:
    from services.desktop_app.processes.capture_supervisor import _kill_pair

    db_path = _bootstrap_db(tmp_path)
    proc = MagicMock()
    proc.pid = 101
    proc.is_alive.return_value = False

    proc.terminate_root.return_value = True

    with patch("services.desktop_app.processes.capture_supervisor.forget_capture_pid") as forget:
        _kill_pair(proc, None, db_path)

    proc.terminate_root.assert_called_once_with(grace_s=3.0)
    proc.terminate.assert_not_called()
    forget.assert_called_once_with(db_path, 101)
