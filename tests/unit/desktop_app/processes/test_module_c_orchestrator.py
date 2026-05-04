from __future__ import annotations

import multiprocessing as mp
import queue
import sqlite3
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from services.desktop_app.ipc import IpcChannels
from services.desktop_app.ipc.control_messages import (
    InferenceControlMessage,
    LiveSessionControlMessage,
    PcmBlockAckMessage,
)
from services.desktop_app.ipc.shared_buffers import read_pcm_block
from services.desktop_app.processes import module_c_orchestrator
from services.desktop_app.state.sqlite_schema import bootstrap_schema


def _make_channels() -> IpcChannels:
    ctx = mp.get_context("spawn")
    return IpcChannels(
        ml_inbox=ctx.Queue(),
        drift_updates=ctx.Queue(),
        analytics_inbox=ctx.Queue(),
        pcm_acks=ctx.Queue(),
        live_control=ctx.Queue(),
        segment_control=ctx.Queue(),
    )


def _bootstrap_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "desktop.sqlite"
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        bootstrap_schema(conn)
        conn.execute(
            """
            INSERT INTO sessions (session_id, stream_url, experiment_id, started_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                "00000000-0000-4000-8000-000000000001",
                "test://stream",
                "greeting_line_v1",
                "2026-05-02 12:00:00",
            ),
        )
    finally:
        conn.close()
    return db_path


def _write_audio_header(path: Path) -> None:
    path.write_bytes(b"R" * module_c_orchestrator.WAV_HEADER_BYTES)


def _append_audio(path: Path, frames: int) -> None:
    with path.open("ab") as audio_file:
        audio_file.write(b"\x01\x02" * frames)


def _segment() -> module_c_orchestrator.DesktopSegment:
    return module_c_orchestrator.DesktopSegment(
        session_id=uuid.UUID("00000000-0000-4000-8000-000000000001"),
        stream_url="test://stream",
        segment_window_start_utc=datetime(2026, 5, 2, 12, 0, tzinfo=UTC),
        pcm_s16le_16khz_mono=b"\x01\x02" * 16000,
        experiment_row_id=1,
        experiment_id="greeting_line_v1",
        active_arm="warm_welcome",
        expected_greeting="Say hello to the creator",
        stimulus_time_s=100.0,
    )


def test_dispatcher_writes_pcm_shared_memory_and_sends_control_message() -> None:
    inbox: queue.Queue[object] = queue.Queue()
    ack_queue: queue.Queue[object] = queue.Queue()
    dispatcher = module_c_orchestrator.DesktopSegmentDispatcher(inbox, ack_queue)
    segment = _segment()

    try:
        dispatcher.dispatch(segment, drift_offset_s=0.125)
        raw = inbox.get_nowait()
        message = InferenceControlMessage.model_validate(raw)
        audio = read_pcm_block(message.audio.to_metadata())
    finally:
        dispatcher.close_inflight_blocks()

    assert audio == segment.pcm_s16le_16khz_mono
    assert "_audio_data" not in message.handoff
    assert message.forward_fields == {}
    assert message.handoff["session_id"] == str(segment.session_id)
    assert message.handoff["_active_arm"] == "warm_welcome"
    assert message.handoff["_expected_greeting"] == "Say hello to the creator"
    assert message.handoff["_stimulus_time"] == 100.125
    assert message.handoff["_au12_series"] == []
    assert message.handoff["_bandit_decision_snapshot"]["selected_arm_id"] == "warm_welcome"


def test_dispatcher_drops_invalid_handoff_without_shared_memory() -> None:
    inbox: queue.Queue[object] = queue.Queue()
    ack_queue: queue.Queue[object] = queue.Queue()
    dispatcher = module_c_orchestrator.DesktopSegmentDispatcher(inbox, ack_queue)
    invalid = _segment()
    invalid = module_c_orchestrator.DesktopSegment(
        session_id=invalid.session_id,
        stream_url="123",
        segment_window_start_utc=invalid.segment_window_start_utc,
        pcm_s16le_16khz_mono=invalid.pcm_s16le_16khz_mono,
        experiment_row_id=invalid.experiment_row_id,
        experiment_id=invalid.experiment_id,
        active_arm=invalid.active_arm,
        expected_greeting=invalid.expected_greeting,
        stimulus_time_s=invalid.stimulus_time_s,
    )

    assert dispatcher.dispatch(invalid) is False
    assert inbox.empty()
    dispatcher.close_inflight_blocks()


def test_dispatcher_default_retention_covers_slow_model_load() -> None:
    inbox: queue.Queue[object] = queue.Queue()
    ack_queue: queue.Queue[object] = queue.Queue()
    dispatcher = module_c_orchestrator.DesktopSegmentDispatcher(inbox, ack_queue)

    assert dispatcher._max_inflight_blocks == module_c_orchestrator.DEFAULT_MAX_INFLIGHT_BLOCKS  # noqa: SLF001


def test_dispatcher_blocks_new_segments_until_pcm_ack_releases_oldest() -> None:
    inbox: queue.Queue[object] = queue.Queue()
    ack_queue: queue.Queue[object] = queue.Queue()
    dispatcher = module_c_orchestrator.DesktopSegmentDispatcher(
        inbox,
        ack_queue,
        max_inflight_blocks=1,
    )

    dispatcher.dispatch(_segment())
    first = InferenceControlMessage.model_validate(inbox.get_nowait())
    try:
        assert dispatcher.dispatch(_segment()) is False
        read_pcm_block(first.audio.to_metadata())
        ack_queue.put(PcmBlockAckMessage(name=first.audio.name).model_dump(mode="json"))
        assert dispatcher.dispatch(_segment()) is True
        second = InferenceControlMessage.model_validate(inbox.get_nowait())
        assert first.audio.name != second.audio.name
        with pytest.raises(FileNotFoundError):
            read_pcm_block(first.audio.to_metadata())
    finally:
        dispatcher.close_inflight_blocks()


def test_drain_segment_controls_tracks_start_stimulus_and_end() -> None:
    channels = IpcChannels(
        ml_inbox=cast("Any", queue.Queue()),
        drift_updates=cast("Any", queue.Queue()),
        analytics_inbox=cast("Any", queue.Queue()),
        pcm_acks=cast("Any", queue.Queue()),
        live_control=cast("Any", queue.Queue()),
        segment_control=cast("Any", queue.Queue()),
    )
    now = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    assert channels.segment_control is not None
    for control in (
        LiveSessionControlMessage(
            action="start",
            session_id=uuid.UUID("00000000-0000-4000-8000-000000000001"),
            stream_url="test://stream",
            experiment_id="greeting_line_v1",
            active_arm="warm_welcome",
            expected_greeting="Say hello",
            timestamp_utc=now,
        ),
        LiveSessionControlMessage(
            action="stimulus",
            session_id=uuid.UUID("00000000-0000-4000-8000-000000000001"),
            stream_url="test://stream",
            experiment_id="greeting_line_v1",
            active_arm="warm_welcome",
            expected_greeting="Say hello",
            stimulus_time_s=now.timestamp(),
            timestamp_utc=now,
        ),
    ):
        channels.segment_control.put(control.model_dump(mode="json"))

    active = module_c_orchestrator._drain_segment_controls(channels, None)

    assert active is not None
    assert active.stimulus_time_s == now.timestamp()

    channels.segment_control.put(
        LiveSessionControlMessage(
            action="end",
            session_id=active.session_id,
            timestamp_utc=now,
        ).model_dump(mode="json")
    )

    assert module_c_orchestrator._drain_segment_controls(channels, active) is None


def test_run_segment_loop_dispatches_audio_for_active_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = _bootstrap_db(tmp_path)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        conn.execute("UPDATE sessions SET ended_at = ?", ("2026-05-02 12:01:00",))
    finally:
        conn.close()
    audio_path = tmp_path / "audio_stream.wav"
    _write_audio_header(audio_path)
    channels = _make_channels()
    shutdown = mp.get_context("spawn").Event()
    dispatcher = MagicMock()
    drift_state = SimpleNamespace(drift_corrector=SimpleNamespace(drift_offset=0.125))
    monkeypatch.setattr(module_c_orchestrator, "SESSION_POLL_INTERVAL_S", 0.01)

    runner = threading.Thread(
        target=module_c_orchestrator._run_segment_loop,
        args=(shutdown, channels, dispatcher, drift_state),
        kwargs={"db_path": db_path, "audio_path": audio_path},
        daemon=True,
    )
    runner.start()
    try:
        assert channels.segment_control is not None
        channels.segment_control.put(
            LiveSessionControlMessage(
                action="start",
                session_id=uuid.UUID("00000000-0000-4000-8000-000000000001"),
                stream_url="test://stream",
                experiment_id="greeting_line_v1",
                active_arm="warm_welcome",
                expected_greeting="Say hello to the creator",
                timestamp_utc=datetime(2026, 5, 2, 12, 0, tzinfo=UTC),
            ).model_dump(mode="json")
        )
        _append_audio(
            audio_path,
            frames=module_c_orchestrator.AUDIO_SAMPLE_RATE_HZ
            * module_c_orchestrator.SEGMENT_WINDOW_SECONDS,
        )
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if dispatcher.dispatch.called:
                break
            time.sleep(0.01)
    finally:
        shutdown.set()
        runner.join(timeout=2.0)

    assert dispatcher.dispatch.called
    segment = dispatcher.dispatch.call_args.args[0]
    assert segment.session_id == uuid.UUID("00000000-0000-4000-8000-000000000001")
    assert len(segment.pcm_s16le_16khz_mono) == 16_000 * 2 * 30
    assert dispatcher.dispatch.call_args.kwargs == {
        "drift_offset_s": 0.125,
        "shutdown_event": shutdown,
    }


def test_drain_drift_updates_applies_numeric_offsets() -> None:
    channels = _make_channels()
    shutdown = mp.get_context("spawn").Event()
    orchestrator = SimpleNamespace(drift_corrector=SimpleNamespace(drift_offset=0.0))

    thread = threading.Thread(
        target=module_c_orchestrator._drain_drift_updates,
        args=(channels, orchestrator, shutdown),
        daemon=True,
    )
    thread.start()
    try:
        channels.drift_updates.put("ignore-me")
        channels.drift_updates.put({"drift_offset": 0.125})

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if orchestrator.drift_corrector.drift_offset == 0.125:
                break
            time.sleep(0.01)
    finally:
        shutdown.set()
        thread.join(timeout=2.0)

    assert orchestrator.drift_corrector.drift_offset == 0.125


def test_run_uses_desktop_dispatcher_and_never_constructs_orchestrator(tmp_path: Path) -> None:
    channels = _make_channels()
    shutdown = mp.get_context("spawn").Event()
    heartbeat = MagicMock()
    dispatcher = MagicMock()
    drift_corrector = SimpleNamespace(drift_offset=0.0)

    with (
        patch("services.desktop_app.os_adapter.resolve_state_dir", return_value=tmp_path),
        patch("services.desktop_app.os_adapter.resolve_capture_dir", return_value=tmp_path),
        patch("services.desktop_app.state.heartbeats.HeartbeatRecorder", return_value=heartbeat),
        patch(
            "services.desktop_app.processes.module_c_orchestrator.DesktopSegmentDispatcher",
            return_value=dispatcher,
        ) as dispatcher_cls,
        patch(
            "services.worker.pipeline.orchestrator.DriftCorrector",
            return_value=drift_corrector,
        ),
        patch(
            "services.desktop_app.processes.module_c_orchestrator._run_segment_loop",
            side_effect=lambda shutdown_event, *_args, **_kwargs: shutdown_event.wait(),
        ) as segment_loop,
        patch("services.worker.pipeline.orchestrator.Orchestrator") as orchestrator_cls,
    ):
        runner = threading.Thread(
            target=module_c_orchestrator.run,
            kwargs={"shutdown_event": shutdown, "channels": channels},
            daemon=True,
        )
        runner.start()
        try:
            channels.drift_updates.put({"drift_offset": 0.25})
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if drift_corrector.drift_offset == 0.25:
                    break
                time.sleep(0.01)
        finally:
            shutdown.set()
            runner.join(timeout=2.0)

    assert not runner.is_alive()
    assert drift_corrector.drift_offset == 0.25
    dispatcher_cls.assert_called_once_with(channels.ml_inbox, channels.pcm_acks)
    orchestrator_cls.assert_not_called()
    segment_loop.assert_called_once()
    dispatcher.close_inflight_blocks.assert_called_once_with()
    heartbeat.start.assert_called_once_with()
    heartbeat.stop.assert_called_once_with()
