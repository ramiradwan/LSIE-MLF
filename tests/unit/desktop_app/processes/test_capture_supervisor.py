from __future__ import annotations

import sqlite3
from pathlib import Path

from services.desktop_app.processes.capture_supervisor import (
    CaptureLayout,
    CaptureStatusRecord,
    _AdbDevice,
    _parse_active_app,
    _write_capture_statuses,
)
from services.desktop_app.state.sqlite_schema import bootstrap_schema


def _bootstrap(tmp_path: Path) -> Path:
    db = tmp_path / "desktop.sqlite"
    conn = sqlite3.connect(str(db), isolation_level=None)
    try:
        bootstrap_schema(conn)
    finally:
        conn.close()
    return db


def _capture_statuses(db: Path) -> dict[str, sqlite3.Row]:
    conn = sqlite3.connect(str(db), isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM capture_status").fetchall()
        return {str(row["status_key"]): row for row in rows}
    finally:
        conn.close()


def test_parse_active_app_from_window_dump() -> None:
    output = """
      mCurrentFocus=Window{abc u0 com.zhiliaoapp.musically/com.ss.android.ugc.aweme.MainActivity}
    """

    assert _parse_active_app(output) == "com.zhiliaoapp.musically"


def test_write_capture_statuses_records_disconnected_state(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)

    _write_capture_statuses(db, None, None, None, None)

    rows = _capture_statuses(db)
    assert rows["adb"]["state"] == "unknown"
    assert rows["adb"]["detail"] == "No Android device connected"
    assert rows["audio_capture"]["state"] == "unknown"
    assert rows["video_capture"]["state"] == "unknown"


def test_write_capture_statuses_records_device_and_stream_details(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    audio = tmp_path / "audio_stream.wav"
    video = tmp_path / "video_stream.mkv"
    audio.write_bytes(b"R" * 44 + b"a" * 84)
    video.write_bytes(b"v" * 256)
    layout = CaptureLayout(
        capture_dir=tmp_path,
        audio_path=audio,
        video_path=video,
    )
    device = _AdbDevice(
        serial="abc123",
        model="Pixel 8",
        active_app="com.zhiliaoapp.musically",
    )

    _write_capture_statuses(db, device, _AliveProc(), _AliveProc(), layout)

    rows = _capture_statuses(db)
    assert rows["adb"]["state"] == "ok"
    assert rows["adb"]["detail"] == (
        "Connected device: Pixel 8 (abc123) · Active app: com.zhiliaoapp.musically"
    )
    assert rows["audio_capture"]["state"] == "ok"
    assert rows["audio_capture"]["detail"] == "Audio stream recording: audio_stream.wav · 128 bytes"
    assert rows["video_capture"]["state"] == "ok"
    assert rows["video_capture"]["detail"] == "Video stream recording: video_stream.mkv · 256 bytes"


def test_write_capture_statuses_reports_silent_audio_stream(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    audio = tmp_path / "audio_stream.wav"
    video = tmp_path / "video_stream.mkv"
    audio.write_bytes(b"R" * 44 + b"\x00" * 960)
    video.write_bytes(b"v" * 256)
    layout = CaptureLayout(
        capture_dir=tmp_path,
        audio_path=audio,
        video_path=video,
    )
    device = _AdbDevice(
        serial="abc123",
        model="Pixel 8",
        active_app="com.zhiliaoapp.musically",
    )

    _write_capture_statuses(db, device, _AliveProc(), _AliveProc(), layout)

    rows = _capture_statuses(db)
    assert rows["audio_capture"]["state"] == "recovering"
    assert rows["audio_capture"]["detail"] == (
        "Audio stream is recording but the captured signal is silent"
    )
    assert rows["audio_capture"]["operator_action_hint"] == (
        "Make sure the phone is playing audible media; some apps block Android playback capture"
    )


def test_missing_capture_tooling_uses_shared_detail_and_hint(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    detail = "Missing required external tools: scrcpy is unavailable: PATH lookup failed"
    hint = "Install scrcpy or set LSIE_SCRCPY_PATH to the scrcpy executable path."

    from services.desktop_app.processes.capture_supervisor import _upsert_capture_statuses

    _upsert_capture_statuses(
        db,
        [
            CaptureStatusRecord(
                status_key="adb",
                state="unknown",
                label="Android Device Bridge",
                detail=detail,
                operator_action_hint=hint,
            ),
            CaptureStatusRecord(
                status_key="audio_capture",
                state="unknown",
                label="Audio Capture",
                detail=detail,
                operator_action_hint=hint,
            ),
            CaptureStatusRecord(
                status_key="video_capture",
                state="unknown",
                label="Video Capture",
                detail=detail,
                operator_action_hint=hint,
            ),
        ],
    )

    rows = _capture_statuses(db)
    assert rows["adb"]["detail"] == detail
    assert rows["adb"]["operator_action_hint"] == hint
    assert rows["audio_capture"]["detail"] == detail
    assert rows["audio_capture"]["operator_action_hint"] == hint
    assert rows["video_capture"]["detail"] == detail
    assert rows["video_capture"]["operator_action_hint"] == hint


class _AliveProc:
    def is_alive(self) -> bool:
        return True
