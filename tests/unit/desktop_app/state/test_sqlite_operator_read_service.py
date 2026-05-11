"""SqliteOperatorReadService end-to-end smoke tests.

These tests exercise the full path:

  ``OperatorReadService`` public surface  ──▶  SQLite query backend
                                          ──▶  ``SqliteReader`` cursor
                                          ──▶  Pydantic DTO

They verify that the same DTO assembly logic that the FastAPI route
layer drives in production renders correct payloads on top of the
desktop's local SQLite store, and that empty-DB cases still produce
valid (degenerate) DTOs rather than raising.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from packages.schemas.operator_console import (
    EncounterState,
    HealthState,
)
from services.desktop_app.state.sqlite_operator_read_service import (
    SqliteOperatorReadService,
)
from services.desktop_app.state.sqlite_reader import SqliteReader
from services.desktop_app.state.sqlite_writer import SqliteWriter

SESSION_A = UUID("00000000-0000-4000-8000-000000000001")
SESSION_B = UUID("00000000-0000-4000-8000-000000000002")


def _build_service(db: Path, *, now: datetime | None = None) -> SqliteOperatorReadService:
    reader = SqliteReader(db)
    service = SqliteOperatorReadService(reader)
    if now is not None:
        service._clock = lambda: now  # noqa: SLF001 — test injection
    return service


def test_get_overview_with_no_sessions_yields_seed_experiments(tmp_path: Path) -> None:
    """Fresh install: no sessions, but the four seed arms must be reachable."""
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    writer.close()

    service = _build_service(db)
    overview = service.get_overview()

    assert overview.active_session is None
    assert overview.latest_encounter is None
    assert overview.experiment_summary is None
    assert overview.physiology is None
    assert overview.health is not None
    assert overview.health.overall_state is HealthState.UNKNOWN

    detail = service.get_experiment_detail("greeting_line_v1")
    assert detail is not None
    assert {arm.arm_id for arm in detail.arms} == {
        "warm_welcome",
        "direct_question",
        "compliment_content",
        "simple_hello",
    }


def test_list_sessions_renders_seeded_rows(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        writer.enqueue(
            "sessions",
            {
                "session_id": str(SESSION_A),
                "stream_url": "test://1",
                "experiment_id": "greeting_line_v1",
                "started_at": "2026-04-01 12:00:00",
            },
        )
        writer.flush()
    finally:
        writer.close()

    service = _build_service(db)
    sessions = service.list_sessions(limit=10)
    assert len(sessions) == 1
    assert sessions[0].session_id == SESSION_A
    assert sessions[0].status == "active"
    assert sessions[0].experiment_id == "greeting_line_v1"
    assert sessions[0].started_at_utc == datetime(2026, 4, 1, 12, 0, tzinfo=UTC)


def test_list_sessions_renders_persisted_selection_before_live_state_or_encounters(
    tmp_path: Path,
) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        writer.enqueue(
            "sessions",
            {
                "session_id": str(SESSION_A),
                "stream_url": "test://1",
                "experiment_id": "greeting_line_v1",
                "active_arm": "warm_welcome",
                "expected_greeting": "Say hello to the creator",
                "started_at": "2026-04-01 12:00:00",
            },
        )
        writer.flush()
    finally:
        writer.close()

    service = _build_service(db)
    sessions = service.list_sessions(limit=10)
    detail = service.get_session(SESSION_A)
    overview = service.get_overview()
    experiment = service.get_experiment_detail("greeting_line_v1")

    assert len(sessions) == 1
    assert sessions[0].active_arm == "warm_welcome"
    assert sessions[0].expected_greeting == "Say hello to the creator"
    assert detail is not None
    assert detail.active_arm == "warm_welcome"
    assert detail.expected_greeting == "Say hello to the creator"
    assert overview.active_session is not None
    assert overview.active_session.active_arm == "warm_welcome"
    assert overview.active_session.expected_greeting == "Say hello to the creator"
    assert experiment is not None
    assert experiment.active_arm_id == "warm_welcome"


def test_list_sessions_marks_multiple_active_rows_as_conflict(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        writer.enqueue(
            "sessions",
            {
                "session_id": str(SESSION_A),
                "stream_url": "test://older",
                "experiment_id": "greeting_line_v1",
                "started_at": "2026-04-01 12:00:00",
            },
        )
        writer.enqueue(
            "sessions",
            {
                "session_id": str(SESSION_B),
                "stream_url": "test://newer",
                "experiment_id": "greeting_line_v1",
                "started_at": "2026-04-01 12:01:00",
            },
        )
        writer.flush()
    finally:
        writer.close()

    service = _build_service(db)
    sessions = service.list_sessions(limit=10)

    assert [session.status for session in sessions] == ["active conflict", "active conflict"]


def test_list_encounters_carries_reward_explanation(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        writer.enqueue(
            "sessions",
            {
                "session_id": str(SESSION_A),
                "stream_url": "test://reward",
                "started_at": "2026-04-01 12:00:00",
            },
        )
        segment_id = "a" * 64
        writer.enqueue(
            "encounter_log",
            {
                "session_id": str(SESSION_A),
                "segment_id": segment_id,
                "experiment_id": "greeting_line_v1",
                "arm": "warm_welcome",
                "timestamp_utc": "2026-04-01 12:01:00",
                "gated_reward": 0.42,
                "p90_intensity": 0.6,
                "semantic_gate": 1,
                "n_frames_in_window": 30,
                "au12_baseline_pre": 0.05,
                "stimulus_time": 1714737660.0,
            },
        )
        writer.enqueue(
            "transcripts",
            {
                "session_id": str(SESSION_A),
                "segment_id": segment_id,
                "timestamp_utc": "2026-04-01 12:01:00",
                "text": "hello welcome to the stream",
            },
        )
        writer.flush()
    finally:
        writer.close()

    service = _build_service(db)
    rows = service.list_encounters(SESSION_A, limit=10, before_utc=None)
    assert len(rows) == 1
    enc = rows[0]
    assert enc.state is EncounterState.COMPLETED
    assert enc.gated_reward == pytest.approx(0.42)
    assert enc.p90_intensity == pytest.approx(0.6)
    assert enc.semantic_gate == 1
    assert enc.n_frames_in_window == 30
    assert enc.au12_baseline_pre == pytest.approx(0.05)
    assert enc.transcription == "hello welcome to the stream"
    # No metrics / attribution rows yet → optional summaries omitted.
    assert enc.observational_acoustic is None
    assert enc.semantic_evaluation is None
    assert enc.attribution is None


def test_get_session_physiology_renders_null_comod(tmp_path: Path) -> None:
    """§7C null-valid co-modulation must surface through the SQLite path."""
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        writer.enqueue(
            "sessions",
            {
                "session_id": str(SESSION_A),
                "stream_url": "x",
                "started_at": "2026-04-01 12:00:00",
            },
        )
        writer.flush()
    finally:
        writer.close()

    service = _build_service(db)
    snap = service.get_session_physiology(SESSION_A)
    assert snap is not None
    assert snap.streamer is None
    assert snap.operator is None
    assert snap.comodulation is not None
    assert snap.comodulation.co_modulation_index is None
    assert snap.comodulation.null_reason == "no co-modulation window computed yet"


def test_get_health_uses_no_op_probes(tmp_path: Path) -> None:
    """Desktop runtime ships no Postgres/Redis; probes must be skipped."""
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    writer.close()

    service = _build_service(db)
    snapshot = asyncio.run(service.get_health())
    assert snapshot.subsystem_probes == {}
    states = {row.subsystem_key: row.state for row in snapshot.subsystems}
    assert states["live_analytics_producer"] is HealthState.UNKNOWN
    assert all(state is HealthState.UNKNOWN for state in states.values())


def test_get_health_surfaces_desktop_adb_and_ml_processes(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    writer.close()
    conn = sqlite3.connect(str(db), isolation_level=None)
    try:
        conn.execute(
            "INSERT INTO process_heartbeat "
            "(process_name, pid, started_at_utc, last_heartbeat_utc) "
            "VALUES (?, ?, ?, ?)",
            (
                "gpu_ml_worker",
                123,
                "2026-04-01 12:00:00",
                "2026-04-01 12:00:05",
            ),
        )
        conn.executemany(
            "INSERT INTO capture_status "
            "(status_key, state, label, detail, operator_action_hint, updated_at_utc) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    "adb",
                    "ok",
                    "Android Device Bridge",
                    "Connected device: Pixel 8 (abc123) · Active app: com.example.app",
                    None,
                    "2026-04-01 12:00:10",
                ),
                (
                    "audio_capture",
                    "ok",
                    "Audio Capture",
                    "Audio stream recording: audio_stream.wav · 1,024 bytes",
                    None,
                    "2026-04-01 12:00:11",
                ),
                (
                    "video_capture",
                    "ok",
                    "Video Capture",
                    "Video stream recording: video_stream.mkv · 2,048 bytes",
                    None,
                    "2026-04-01 12:00:12",
                ),
            ],
        )
    finally:
        conn.close()

    service = _build_service(db, now=datetime(2026, 4, 1, 12, 0, 6, tzinfo=UTC))
    snapshot = asyncio.run(service.get_health())
    states = {row.subsystem_key: row.state for row in snapshot.subsystems}
    assert set(states) == {
        "ui_api_shell",
        "capture_supervisor",
        "module_c_orchestrator",
        "gpu_ml_worker",
        "analytics_state_worker",
        "cloud_sync_worker",
        "adb",
        "audio_capture",
        "video_capture",
        "live_analytics_producer",
    }
    assert not {"metrics", "physiology", "comodulation", "encounters"} & set(states)
    assert states["adb"] is HealthState.OK
    assert states["audio_capture"] is HealthState.OK
    assert states["video_capture"] is HealthState.OK
    assert states["gpu_ml_worker"] is HealthState.OK
    assert states["live_analytics_producer"] is HealthState.UNKNOWN
    rows = {row.subsystem_key: row for row in snapshot.subsystems}
    assert rows["adb"].detail == "Connected device: Pixel 8 (abc123) · Active app: com.example.app"
    assert rows["audio_capture"].detail == "Audio stream recording: audio_stream.wav · 1,024 bytes"
    assert rows["video_capture"].detail == "Video stream recording: video_stream.mkv · 2,048 bytes"

    live_producer = next(
        row for row in snapshot.subsystems if row.subsystem_key == "live_analytics_producer"
    )
    assert live_producer.detail is not None
    assert live_producer.operator_action_hint is not None
    assert "Setup not ready" in live_producer.detail
    assert "Start or select a Live Session" in live_producer.operator_action_hint


def test_get_health_surfaces_preparing_live_analysis_status(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    writer.close()
    conn = sqlite3.connect(str(db), isolation_level=None)
    try:
        conn.execute(
            "INSERT INTO process_heartbeat "
            "(process_name, pid, started_at_utc, last_heartbeat_utc) "
            "VALUES (?, ?, ?, ?)",
            (
                "gpu_ml_worker",
                123,
                "2026-04-01 12:00:00",
                "2026-04-01 12:00:05",
            ),
        )
        conn.execute(
            "INSERT INTO capture_status "
            "(status_key, state, label, detail, operator_action_hint, updated_at_utc) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "gpu_ml_worker",
                "recovering",
                "GPU ML Worker",
                "Loading speech transcription model before live analytics can complete.",
                None,
                "2026-04-01 12:00:05",
            ),
        )
        conn.execute(
            "INSERT INTO sessions (session_id, stream_url, experiment_id, started_at) "
            "VALUES (?, ?, ?, ?)",
            (str(SESSION_A), "test://stream", "greeting_line_v1", "2026-04-01 12:00:00"),
        )
    finally:
        conn.close()

    service = _build_service(db, now=datetime(2026, 4, 1, 12, 0, 6, tzinfo=UTC))
    snapshot = asyncio.run(service.get_health())
    rows = {row.subsystem_key: row for row in snapshot.subsystems}

    assert rows["gpu_ml_worker"].state is HealthState.RECOVERING
    assert rows["gpu_ml_worker"].recovery_mode == "Preparing live analysis"
    assert rows["gpu_ml_worker"].detail is not None
    assert "Loading speech transcription model" in rows["gpu_ml_worker"].detail
    assert rows["live_analytics_producer"].state is HealthState.RECOVERING
    assert rows["live_analytics_producer"].detail == "Preparing live analysis."
    assert rows["live_analytics_producer"].operator_action_hint == (
        "Wait for the first result before sending another test message."
    )


def test_get_health_flags_multiple_active_sessions(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    writer.close()
    conn = sqlite3.connect(str(db), isolation_level=None)
    try:
        conn.executemany(
            "INSERT INTO sessions (session_id, stream_url, experiment_id, started_at) "
            "VALUES (?, ?, ?, ?)",
            [
                (str(SESSION_A), "test://older", "greeting_line_v1", "2026-04-01 12:00:00"),
                (str(SESSION_B), "test://newer", "greeting_line_v1", "2026-04-01 12:01:00"),
            ],
        )
    finally:
        conn.close()

    service = _build_service(db, now=datetime(2026, 4, 1, 12, 0, 6, tzinfo=UTC))
    snapshot = asyncio.run(service.get_health())
    rows = {row.subsystem_key: row for row in snapshot.subsystems}

    assert rows["live_analytics_producer"].state is HealthState.ERROR
    assert rows["live_analytics_producer"].detail is not None
    assert "2 active desktop sessions" in rows["live_analytics_producer"].detail


def test_list_sessions_renders_live_calibration_state(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        writer.enqueue(
            "sessions",
            {
                "session_id": str(SESSION_A),
                "stream_url": "test://1",
                "experiment_id": "greeting_line_v1",
                "started_at": "2026-04-01 12:00:00",
            },
        )
        writer.enqueue(
            "live_session_state",
            {
                "session_id": str(SESSION_A),
                "active_arm": "warm_welcome",
                "expected_greeting": "Say hello to the creator",
                "is_calibrating": 1,
                "calibration_frames_accumulated": 4,
                "calibration_frames_required": 10,
                "face_present": 1,
                "status": "calibrating",
                "updated_at_utc": "2026-04-01 12:00:05",
            },
        )
        writer.flush()
    finally:
        writer.close()

    service = _build_service(db)
    sessions = service.list_sessions(limit=10)

    assert len(sessions) == 1
    session = sessions[0]
    assert session.active_arm == "warm_welcome"
    assert session.expected_greeting == "Say hello to the creator"
    assert session.is_calibrating is True
    assert session.calibration_frames_accumulated == 4
    assert session.calibration_frames_required == 10


def test_get_health_marks_live_analytics_ok_from_fresh_visual_state(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        writer.enqueue(
            "sessions",
            {
                "session_id": str(SESSION_A),
                "stream_url": "test://1",
                "started_at": "2026-04-01 12:00:00",
            },
        )
        writer.enqueue(
            "live_session_state",
            {
                "session_id": str(SESSION_A),
                "is_calibrating": 0,
                "calibration_frames_accumulated": 10,
                "calibration_frames_required": 10,
                "face_present": 1,
                "status": "ready",
                "updated_at_utc": "2026-04-01 12:00:05",
            },
        )
        writer.flush()
    finally:
        writer.close()

    service = _build_service(db, now=datetime(2026, 4, 1, 12, 0, 6, tzinfo=UTC))
    snapshot = asyncio.run(service.get_health())
    rows = {row.subsystem_key: row for row in snapshot.subsystems}

    assert rows["live_analytics_producer"].state is HealthState.OK
    assert rows["live_analytics_producer"].detail == "Healthy: live analysis is updating."
    assert rows["live_analytics_producer"].operator_action_hint is None
    assert rows["live_analytics_producer"].last_success_utc == datetime(
        2026, 4, 1, 12, 0, 5, tzinfo=UTC
    )


def test_get_health_degrades_stale_live_analytics_for_active_session(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        writer.enqueue(
            "sessions",
            {
                "session_id": str(SESSION_A),
                "stream_url": "test://1",
                "started_at": "2026-04-01 12:00:00",
            },
        )
        writer.enqueue(
            "live_session_state",
            {
                "session_id": str(SESSION_A),
                "is_calibrating": 0,
                "calibration_frames_accumulated": 10,
                "calibration_frames_required": 10,
                "face_present": 1,
                "status": "ready",
                "updated_at_utc": "2026-04-01 12:00:05",
            },
        )
        writer.flush()
    finally:
        writer.close()

    service = _build_service(db, now=datetime(2026, 4, 1, 12, 3, 0, tzinfo=UTC))
    snapshot = asyncio.run(service.get_health())
    rows = {row.subsystem_key: row for row in snapshot.subsystems}

    assert rows["live_analytics_producer"].state is HealthState.DEGRADED
    assert rows["live_analytics_producer"].detail == "Live analysis has stopped updating."
    assert rows["live_analytics_producer"].operator_action_hint == (
        "Check that the face is visible, then send one test message."
    )


def test_get_overview_with_encounter_renders_experiment(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        writer.enqueue(
            "sessions",
            {
                "session_id": str(SESSION_A),
                "stream_url": "x",
                "started_at": "2026-04-01 12:00:00",
            },
        )
        writer.enqueue(
            "encounter_log",
            {
                "session_id": str(SESSION_A),
                "segment_id": "b" * 64,
                "experiment_id": "greeting_line_v1",
                "arm": "warm_welcome",
                "timestamp_utc": "2026-04-01 12:01:00",
                "gated_reward": 1.0,
                "p90_intensity": 0.7,
                "semantic_gate": 1,
                "n_frames_in_window": 30,
            },
        )
        writer.flush()
    finally:
        writer.close()

    # Pin "now" to a time inside the stale-physiology hour-window so the
    # default §12 freshness rollup classifies the encounter row as fresh.
    service = _build_service(db, now=datetime(2026, 4, 1, 12, 1, 1, tzinfo=UTC))
    overview = service.get_overview()

    assert overview.active_session is not None
    assert overview.active_session.session_id == SESSION_A
    assert overview.latest_encounter is not None
    assert overview.latest_encounter.gated_reward == pytest.approx(1.0)
    assert overview.experiment_summary is not None
    assert overview.experiment_summary.experiment_id == "greeting_line_v1"
    assert overview.experiment_summary.active_arm_id == "warm_welcome"


def test_get_overview_marks_active_conflict_when_multiple_rows_are_live(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        writer.enqueue(
            "sessions",
            {
                "session_id": str(SESSION_A),
                "stream_url": "test://older",
                "experiment_id": "greeting_line_v1",
                "started_at": "2026-04-01 12:00:00",
            },
        )
        writer.enqueue(
            "sessions",
            {
                "session_id": str(SESSION_B),
                "stream_url": "test://newer",
                "experiment_id": "greeting_line_v1",
                "started_at": "2026-04-01 12:01:00",
            },
        )
        writer.flush()
    finally:
        writer.close()

    service = _build_service(db)
    overview = service.get_overview()

    assert overview.active_session is not None
    assert overview.active_session.session_id == SESSION_B
    assert overview.active_session.status == "active conflict"


def test_list_alerts_returns_session_ended_alert(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        # ended_at within the last hour to satisfy the SQL filter.
        ended_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        writer.enqueue(
            "sessions",
            {
                "session_id": str(SESSION_A),
                "stream_url": "x",
                "started_at": "2026-04-01 12:00:00",
                "ended_at": ended_at,
            },
        )
        writer.flush()
    finally:
        writer.close()

    service = _build_service(db)
    alerts = service.list_alerts(limit=10, since_utc=None)
    assert len(alerts) == 1
    assert alerts[0].kind.value == "session_ended"
    assert alerts[0].session_id == SESSION_A


def test_since_utc_filter_excludes_old_alerts(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        ended_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        writer.enqueue(
            "sessions",
            {
                "session_id": str(SESSION_A),
                "stream_url": "x",
                "started_at": "2026-04-01 12:00:00",
                "ended_at": ended_at,
            },
        )
        writer.flush()
    finally:
        writer.close()

    service = _build_service(db)
    far_future = datetime.now(UTC) + timedelta(days=1)
    alerts = service.list_alerts(limit=10, since_utc=far_future)
    assert alerts == []
