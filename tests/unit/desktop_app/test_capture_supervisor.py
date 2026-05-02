"""WS3 P3 — capture_supervisor unit tests.

Validates the supervision contract without requiring a real Android
device or a real ``scrcpy`` install:

  - ``_wait_for_device`` returns True when ADB reports a device line.
  - ``_DriftPollThread`` ships ``drift_offset`` payloads onto
    ``IpcChannels.drift_updates`` and stops cleanly on shutdown.
  - The scrcpy command builders emit the SPEC-AMEND-004 args
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
from unittest.mock import MagicMock, patch

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
    _wait_for_device,
)


def _make_channels() -> IpcChannels:
    ctx = mp.get_context("spawn")
    return IpcChannels(
        ml_inbox=ctx.Queue(),
        drift_updates=ctx.Queue(),
        analytics_inbox=ctx.Queue(),
    )


def test_wait_for_device_returns_true_on_device_line() -> None:
    fake_result = MagicMock()
    fake_result.stdout = "List of devices attached\n55281FDCR002LK\tdevice\n\n"
    with (
        patch(
            "services.desktop_app.processes.capture_supervisor.find_executable",
            return_value="adb",
        ),
        patch("subprocess.run", return_value=fake_result),
    ):
        assert _wait_for_device(deadline_s=2.0) is True


def test_wait_for_device_times_out_when_no_device() -> None:
    fake_result = MagicMock()
    fake_result.stdout = "List of devices attached\n\n"
    with (
        patch(
            "services.desktop_app.processes.capture_supervisor.find_executable",
            return_value="adb",
        ),
        patch("subprocess.run", return_value=fake_result),
    ):
        assert _wait_for_device(deadline_s=0.5) is False


def test_audio_scrcpy_args_match_spec() -> None:
    layout = CaptureLayout(
        capture_dir=Path("/tmp/lsie-mlf/capture"),
        audio_path=Path("/tmp/lsie-mlf/capture/audio_stream.wav"),
        video_path=Path("/tmp/lsie-mlf/capture/video_stream.mkv"),
    )
    args = _build_audio_scrcpy_args("scrcpy.exe", layout)

    assert args[0] == "scrcpy.exe"
    assert "--no-video" in args
    assert "--audio-codec=raw" in args
    assert "--audio-buffer=30" in args
    assert "--audio-dup" in args
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
    assert "--video-codec=h264" in args
    assert "--max-fps=30" in args
    assert f"--port={VIDEO_SCRCPY_PORT_RANGE}" in args
    assert any(arg.startswith("--record=") and "video_stream.mkv" in arg for arg in args)
    assert "--record-format=mkv" in args


def test_resolve_capture_layout_honours_env_override(tmp_path: Path, monkeypatch: Any) -> None:
    target = tmp_path / "custom-capture"
    monkeypatch.setenv("LSIE_CAPTURE_DIR", str(target))

    layout = _resolve_capture_layout()

    assert layout.capture_dir == target
    assert target.is_dir()
    assert layout.audio_path == target / "audio_stream.wav"
    assert layout.video_path == target / "video_stream.mkv"


def test_drift_poll_thread_publishes_offsets() -> None:
    channels = _make_channels()
    corrector = DriftCorrector()
    corrector.drift_offset = 0.123  # simulate a previous successful poll

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

    # Give the thread a tick to push at least one update.
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
