from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import queue
import sqlite3
import struct
import threading
import time
import uuid
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from packages.schemas.evaluation import StimulusDefinition, StimulusPayload
from services.desktop_app.ipc import IpcChannels
from services.desktop_app.ipc.control_messages import (
    DESKTOP_LIVE_VISUAL_SOURCE_CONTRACT,
    InferenceControlMessage,
    LiveSessionControlMessage,
    PcmBlockAckMessage,
    VisualAnalyticsStateMessage,
)
from services.desktop_app.ipc.shared_buffers import read_pcm_block
from services.desktop_app.processes import module_c_orchestrator
from services.desktop_app.state.sqlite_schema import bootstrap_schema
from services.desktop_app.state.sqlite_session_lifecycle_service import _json_default


def _stimulus_definition() -> StimulusDefinition:
    return StimulusDefinition(
        stimulus_modality="spoken_greeting",
        stimulus_payload=StimulusPayload(
            content_type="text",
            text="Say hello to the creator",
        ),
        expected_stimulus_rule=(
            "Deliver the spoken greeting to the live streamer exactly as written."
        ),
        expected_response_rule=(
            "The live streamer acknowledges the greeting or responds to it on stream."
        ),
    )


_BANDIT_SNAPSHOT = {
    "selection_method": "thompson_sampling",
    "selection_time_utc": datetime(2026, 5, 2, 12, 0, tzinfo=UTC),
    "experiment_id": 1,
    "policy_version": "desktop_replay_v1",
    "selected_arm_id": "warm_welcome",
    "candidate_arm_ids": ["direct_question", "warm_welcome"],
    "posterior_by_arm": {
        "direct_question": {"alpha": 1.0, "beta": 5.0},
        "warm_welcome": {"alpha": 2.0, "beta": 3.0},
    },
    "sampled_theta_by_arm": {"direct_question": 0.2, "warm_welcome": 0.6},
    "stimulus_modality": _stimulus_definition().stimulus_modality,
    "stimulus_payload": _stimulus_definition().stimulus_payload.model_dump(mode="json"),
    "expected_stimulus_rule": _stimulus_definition().expected_stimulus_rule,
    "expected_response_rule": _stimulus_definition().expected_response_rule,
    "decision_context_hash": "a" * 64,
    "random_seed": 42,
}


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
            INSERT INTO sessions (
                session_id, stream_url, experiment_id, active_arm, stimulus_definition,
                bandit_decision_snapshot, started_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "00000000-0000-4000-8000-000000000001",
                "test://stream",
                "greeting_line_v1",
                "warm_welcome",
                _stimulus_definition().model_dump_json(),
                json.dumps(
                    _BANDIT_SNAPSHOT,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=_json_default,
                ),
                "2026-05-02 12:00:00",
            ),
        )
    finally:
        conn.close()
    return db_path


def _write_audio_header(
    path: Path,
    *,
    channels: int = 1,
    sample_rate_hz: int = module_c_orchestrator.AUDIO_SAMPLE_RATE_HZ,
) -> None:
    sample_width_bytes = module_c_orchestrator.AUDIO_SAMPLE_WIDTH_BYTES
    byte_rate = sample_rate_hz * channels * sample_width_bytes
    block_align = channels * sample_width_bytes
    header = b"".join(
        (
            b"RIFF",
            struct.pack("<I", 36),
            b"WAVE",
            b"fmt ",
            struct.pack("<I", 16),
            struct.pack("<HHIIHH", 1, channels, sample_rate_hz, byte_rate, block_align, 16),
            b"data",
            struct.pack("<I", 0),
        )
    )
    path.write_bytes(header)


def _append_audio(path: Path, frames: int, *, channels: int = 1) -> None:
    frame = b"\x01\x02" * channels
    with path.open("ab") as audio_file:
        audio_file.write(frame * frames)


def _segment(
    *,
    au12_series: tuple[dict[str, float], ...] = (),
) -> module_c_orchestrator.DesktopSegment:
    return module_c_orchestrator.DesktopSegment(
        session_id=uuid.UUID("00000000-0000-4000-8000-000000000001"),
        stream_url="test://stream",
        segment_window_start_utc=datetime(2026, 5, 2, 12, 0, tzinfo=UTC),
        pcm_s16le_16khz_mono=b"\x01\x02" * 16000,
        experiment_row_id=1,
        experiment_id="greeting_line_v1",
        active_arm="warm_welcome",
        stimulus_definition=_stimulus_definition(),
        bandit_decision_snapshot=_BANDIT_SNAPSHOT,
        stimulus_time_s=100.0,
        au12_series=au12_series,
    )


def test_dispatcher_writes_pcm_shared_memory_and_sends_control_message() -> None:
    inbox: queue.Queue[object] = queue.Queue()
    ack_queue: queue.Queue[object] = queue.Queue()
    dispatcher = module_c_orchestrator.DesktopSegmentDispatcher(inbox, ack_queue)
    segment = _segment(
        au12_series=({"timestamp_s": 1_777_373_401.0, "intensity": 0.72},),
    )

    try:
        dispatcher.dispatch(segment, drift_offset_s=0.125)
        raw = inbox.get_nowait()
        message = InferenceControlMessage.model_validate(raw)
        audio = read_pcm_block(message.audio.to_metadata())
    finally:
        dispatcher.close_inflight_blocks()

    assert audio == segment.pcm_s16le_16khz_mono
    assert "_audio_data" not in message.handoff
    assert "_visual_source_contract" not in message.handoff
    assert message.forward_fields == {"_drift_offset_s": 0.125}
    assert message.visual_source_contract == DESKTOP_LIVE_VISUAL_SOURCE_CONTRACT
    assert message.handoff["session_id"] == str(segment.session_id)
    assert message.handoff["_active_arm"] == "warm_welcome"
    assert message.handoff["_stimulus_modality"] == segment.stimulus_definition.stimulus_modality
    assert message.handoff[
        "_stimulus_payload"
    ] == segment.stimulus_definition.stimulus_payload.model_dump(mode="json")
    assert (
        message.handoff["_expected_stimulus_rule"]
        == segment.stimulus_definition.expected_stimulus_rule
    )
    assert (
        message.handoff["_expected_response_rule"]
        == segment.stimulus_definition.expected_response_rule
    )
    assert message.handoff["_stimulus_time"] == segment.stimulus_time_s
    assert message.handoff["_au12_series"] == [{"timestamp_s": 1_777_373_401.0, "intensity": 0.72}]
    snapshot = message.handoff["_bandit_decision_snapshot"]
    assert snapshot["selected_arm_id"] == "warm_welcome"
    assert snapshot["candidate_arm_ids"] == ["direct_question", "warm_welcome"]
    assert snapshot["posterior_by_arm"]["warm_welcome"] == {"alpha": 2.0, "beta": 3.0}
    assert snapshot["sampled_theta_by_arm"] == {"direct_question": 0.2, "warm_welcome": 0.6}


def test_segment_id_uses_only_session_and_canonical_window_bounds() -> None:
    segment = _segment()
    end = segment.segment_window_start_utc + timedelta(
        seconds=module_c_orchestrator.SEGMENT_WINDOW_SECONDS
    )
    stable_identity = "|".join(
        (
            str(segment.session_id),
            module_c_orchestrator._canonical_utc_timestamp(segment.segment_window_start_utc),
            module_c_orchestrator._canonical_utc_timestamp(end),
        )
    )
    expected = hashlib.sha256(stable_identity.encode("utf-8")).hexdigest()

    baseline = module_c_orchestrator._segment_id(segment)
    changed_runtime_fields = module_c_orchestrator.DesktopSegment(
        session_id=segment.session_id,
        stream_url=segment.stream_url,
        segment_window_start_utc=segment.segment_window_start_utc,
        pcm_s16le_16khz_mono=segment.pcm_s16le_16khz_mono + b"\x00\x00",
        experiment_row_id=999,
        experiment_id="different_experiment",
        active_arm="direct_question",
        stimulus_definition=segment.stimulus_definition,
        bandit_decision_snapshot=segment.bandit_decision_snapshot,
        stimulus_time_s=segment.stimulus_time_s,
    )

    assert baseline == expected
    assert module_c_orchestrator._segment_id(changed_runtime_fields) == expected


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
        stimulus_definition=invalid.stimulus_definition,
        bandit_decision_snapshot=invalid.bandit_decision_snapshot,
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


def test_capture_audio_conversion_handles_stereo_duration(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio_stream.wav"
    _write_audio_header(audio_path, channels=2)
    _append_audio(
        audio_path,
        frames=module_c_orchestrator.AUDIO_SAMPLE_RATE_HZ,
        channels=2,
    )
    audio_format = module_c_orchestrator._read_capture_audio_format(audio_path)  # noqa: SLF001
    assert audio_format is not None

    raw, cursor = module_c_orchestrator._read_new_capture_audio(  # noqa: SLF001
        audio_path,
        0,
        audio_format,
    )
    pcm_16k = module_c_orchestrator._source_pcm_to_16k_mono(raw, audio_format)  # noqa: SLF001

    assert cursor == audio_path.stat().st_size
    assert len(pcm_16k) == module_c_orchestrator.PCM_SAMPLE_RATE_HZ * 2
    assert module_c_orchestrator._audio_duration_for_bytes(len(pcm_16k)).total_seconds() == 1.0  # noqa: SLF001


def test_start_control_uses_persisted_lifecycle_selection(tmp_path: Path) -> None:
    db_path = _bootstrap_db(tmp_path)
    session_id = uuid.UUID("00000000-0000-4000-8000-000000000001")
    persisted_snapshot = {
        **_BANDIT_SNAPSHOT,
        "candidate_arm_ids": ["direct_question", "simple_hello", "warm_welcome"],
        "posterior_by_arm": {
            "direct_question": {"alpha": 1.0, "beta": 5.0},
            "simple_hello": {"alpha": 9.0, "beta": 1.0},
            "warm_welcome": {"alpha": 2.0, "beta": 3.0},
        },
        "sampled_theta_by_arm": {
            "direct_question": 0.2,
            "simple_hello": 0.99,
            "warm_welcome": 0.6,
        },
        "decision_context_hash": "b" * 64,
        "random_seed": 987654321,
    }
    persisted_snapshot_json = json.dumps(
        persisted_snapshot,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        conn.execute(
            """
            UPDATE sessions
            SET active_arm = ?, stimulus_definition = ?, bandit_decision_snapshot = ?
            WHERE session_id = ?
            """,
            (
                "warm_welcome",
                _stimulus_definition().model_dump_json(),
                persisted_snapshot_json,
                str(session_id),
            ),
        )
    finally:
        conn.close()

    active = module_c_orchestrator._resolve_control_session(  # noqa: SLF001
        db_path,
        LiveSessionControlMessage(
            action="start",
            session_id=session_id,
            stream_url="test://stream",
            experiment_id="greeting_line_v1",
            active_arm="simple_hello",
            stimulus_definition=StimulusDefinition(
                stimulus_modality="spoken_greeting",
                stimulus_payload=StimulusPayload(
                    content_type="text",
                    text="This control value must not override storage",
                ),
                expected_stimulus_rule=(
                    "Deliver the spoken greeting to the live streamer exactly as written."
                ),
                expected_response_rule=(
                    "The live streamer acknowledges the greeting or responds to it on stream."
                ),
            ),
            timestamp_utc=datetime(2026, 5, 2, 12, 0, tzinfo=UTC),
        ),
    )

    assert active is not None
    assert active.active_arm == "warm_welcome"
    assert active.stimulus_definition == _stimulus_definition()
    assert active.bandit_decision_snapshot == json.loads(persisted_snapshot_json)


def test_drain_visual_au12_updates_buffers_bounded_observations() -> None:
    channels = IpcChannels(
        ml_inbox=cast("Any", queue.Queue()),
        drift_updates=cast("Any", queue.Queue()),
        analytics_inbox=cast("Any", queue.Queue()),
        pcm_acks=cast("Any", queue.Queue()),
        visual_state_updates=cast("Any", queue.Queue()),
    )
    visual_buffer: deque[module_c_orchestrator._VisualAu12Telemetry] = deque(  # noqa: SLF001
        maxlen=module_c_orchestrator.VISUAL_AU12_BUFFER_MAXLEN,
    )
    assert channels.visual_state_updates is not None
    visual = VisualAnalyticsStateMessage.model_validate(
        {
            "message_id": "00000000-0000-4000-8000-000000000001",
            "session_id": "00000000-0000-4000-8000-000000000001",
            "timestamp_utc": "2026-05-02T12:00:01+00:00",
            "face_present": True,
            "is_calibrating": False,
            "calibration_frames_accumulated": 45,
            "calibration_frames_required": 45,
            "latest_au12_intensity": 0.72,
            "latest_au12_timestamp_s": 1_777_373_401.0,
            "status": "post_stimulus",
        }
    )
    channels.visual_state_updates.put(visual.model_dump(mode="json"))

    module_c_orchestrator._drain_visual_au12_updates(channels, visual_buffer)  # noqa: SLF001

    assert [observation.as_payload() for observation in visual_buffer] == [
        {"timestamp_s": 1_777_373_401.0, "intensity": 0.72}
    ]


def test_drain_segment_controls_tracks_start_stimulus_and_end(tmp_path: Path) -> None:
    db_path = _bootstrap_db(tmp_path)
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
            stimulus_definition=_stimulus_definition(),
            timestamp_utc=now,
        ),
        LiveSessionControlMessage(
            action="stimulus",
            session_id=uuid.UUID("00000000-0000-4000-8000-000000000001"),
            stream_url="test://stream",
            experiment_id="greeting_line_v1",
            active_arm="warm_welcome",
            stimulus_definition=_stimulus_definition(),
            stimulus_time_s=now.timestamp(),
            timestamp_utc=now,
        ),
    ):
        channels.segment_control.put(control.model_dump(mode="json"))

    active = module_c_orchestrator._drain_segment_controls(channels, None, db_path=db_path)

    assert active is not None
    assert active.experiment_row_id > 0
    assert active.experiment_id == "greeting_line_v1"
    assert active.active_arm in active.bandit_decision_snapshot["candidate_arm_ids"]
    assert active.active_arm == active.bandit_decision_snapshot["selected_arm_id"]
    assert active.bandit_decision_snapshot == json.loads(
        json.dumps(
            _BANDIT_SNAPSHOT,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        )
    )
    assert active.stimulus_time_s == now.timestamp()

    channels.segment_control.put(
        LiveSessionControlMessage(
            action="end",
            session_id=active.session_id,
            timestamp_utc=now,
        ).model_dump(mode="json")
    )

    assert module_c_orchestrator._drain_segment_controls(channels, active, db_path=db_path) is None


@pytest.mark.parametrize("channels", [1, 2])
def test_run_segment_loop_dispatches_audio_for_active_stimulus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    channels: int,
) -> None:
    db_path = _bootstrap_db(tmp_path)
    audio_path = tmp_path / "audio_stream.wav"
    _write_audio_header(audio_path, channels=channels)
    ipc_channels = _make_channels()
    shutdown = mp.get_context("spawn").Event()
    dispatcher = MagicMock()
    drift_state = SimpleNamespace(drift_corrector=SimpleNamespace(drift_offset=0.125))
    monkeypatch.setattr(module_c_orchestrator, "SESSION_POLL_INTERVAL_S", 0.01)

    runner = threading.Thread(
        target=module_c_orchestrator._run_segment_loop,
        args=(shutdown, ipc_channels, dispatcher, drift_state),
        kwargs={"db_path": db_path, "audio_path": audio_path},
        daemon=True,
    )
    runner.start()
    try:
        assert ipc_channels.segment_control is not None
        session_id = uuid.UUID("00000000-0000-4000-8000-000000000001")
        ipc_channels.segment_control.put(
            LiveSessionControlMessage(
                action="start",
                session_id=session_id,
                stream_url="test://stream",
                experiment_id="greeting_line_v1",
                active_arm="warm_welcome",
                stimulus_definition=_stimulus_definition(),
                timestamp_utc=datetime(2026, 5, 2, 12, 0, tzinfo=UTC),
            ).model_dump(mode="json")
        )
        _append_audio(
            audio_path,
            frames=module_c_orchestrator.AUDIO_SAMPLE_RATE_HZ
            * module_c_orchestrator.SEGMENT_WINDOW_SECONDS,
            channels=channels,
        )
        time.sleep(0.05)
        stimulus_time = (datetime.now(UTC) - timedelta(seconds=5)).timestamp()
        ipc_channels.segment_control.put(
            LiveSessionControlMessage(
                action="stimulus",
                session_id=session_id,
                stream_url="test://stream",
                experiment_id="greeting_line_v1",
                active_arm="warm_welcome",
                stimulus_definition=_stimulus_definition(),
                stimulus_time_s=stimulus_time,
                timestamp_utc=datetime.now(UTC),
            ).model_dump(mode="json")
        )
        _append_audio(
            audio_path,
            frames=module_c_orchestrator.AUDIO_SAMPLE_RATE_HZ
            * module_c_orchestrator.SEGMENT_WINDOW_SECONDS,
            channels=channels,
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
    assert segment.experiment_row_id > 0
    assert segment.experiment_id == "greeting_line_v1"
    assert segment.active_arm == segment.bandit_decision_snapshot["selected_arm_id"]
    assert segment.stimulus_definition.model_dump(mode="json") == {
        "stimulus_modality": segment.bandit_decision_snapshot["stimulus_modality"],
        "stimulus_payload": segment.bandit_decision_snapshot["stimulus_payload"],
        "expected_stimulus_rule": segment.bandit_decision_snapshot["expected_stimulus_rule"],
        "expected_response_rule": segment.bandit_decision_snapshot["expected_response_rule"],
    }
    assert segment.active_arm in segment.bandit_decision_snapshot["candidate_arm_ids"]
    assert segment.stimulus_time_s == pytest.approx(stimulus_time)
    assert len(segment.pcm_s16le_16khz_mono) == 16_000 * 2 * 30
    assert segment.segment_window_start_utc.timestamp() <= stimulus_time - 5.0
    assert segment.segment_window_start_utc.timestamp() + 30.0 >= stimulus_time + 5.0
    assert dispatcher.dispatch.call_args.kwargs == {
        "drift_offset_s": 0.125,
        "shutdown_event": shutdown,
    }


def test_run_segment_loop_does_not_dispatch_without_stimulus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = _bootstrap_db(tmp_path)
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
                stimulus_definition=_stimulus_definition(),
                timestamp_utc=datetime(2026, 5, 2, 12, 0, tzinfo=UTC),
            ).model_dump(mode="json")
        )
        _append_audio(
            audio_path,
            frames=module_c_orchestrator.AUDIO_SAMPLE_RATE_HZ
            * module_c_orchestrator.SEGMENT_WINDOW_SECONDS,
        )
        time.sleep(0.1)
    finally:
        shutdown.set()
        runner.join(timeout=2.0)

    assert not dispatcher.dispatch.called


def test_stimulus_time_attaches_only_to_segment_covering_baseline_and_measurement() -> None:
    active = module_c_orchestrator._ActiveSession(  # noqa: SLF001
        session_id=uuid.UUID("00000000-0000-4000-8000-000000000001"),
        stream_url="test://stream",
        experiment_id="greeting_line_v1",
        experiment_row_id=1,
        active_arm="warm_welcome",
        stimulus_definition=_stimulus_definition(),
        bandit_decision_snapshot=_BANDIT_SNAPSHOT,
        stimulus_time_s=datetime(2026, 5, 2, 12, 0, 10, tzinfo=UTC).timestamp(),
    )
    covering_start = datetime(2026, 5, 2, 11, 59, 45, tzinfo=UTC)
    too_late_start = datetime(2026, 5, 2, 12, 0, 30, tzinfo=UTC)

    covering = module_c_orchestrator._segment_from_active_session(  # noqa: SLF001
        active,
        segment_start_utc=covering_start,
        pcm_s16le_16khz_mono=b"\0\0" * 16000 * 30,
        drift_offset_s=0.125,
    )
    too_late = module_c_orchestrator._segment_from_active_session(  # noqa: SLF001
        active,
        segment_start_utc=too_late_start,
        pcm_s16le_16khz_mono=b"\0\0" * 16000 * 30,
        drift_offset_s=0.125,
    )

    assert covering.stimulus_time_s == active.stimulus_time_s
    assert too_late.stimulus_time_s is None


def test_stimulus_aligned_segment_start_ends_at_measurement_window_close() -> None:
    active = module_c_orchestrator._ActiveSession(  # noqa: SLF001
        session_id=uuid.UUID("00000000-0000-4000-8000-000000000001"),
        stream_url="test://stream",
        experiment_id="greeting_line_v1",
        experiment_row_id=1,
        active_arm="warm_welcome",
        stimulus_definition=_stimulus_definition(),
        bandit_decision_snapshot=_BANDIT_SNAPSHOT,
        stimulus_time_s=datetime(2026, 5, 2, 12, 0, 40, tzinfo=UTC).timestamp(),
    )

    start = module_c_orchestrator._stimulus_aligned_segment_start_utc(active)  # noqa: SLF001

    assert start == datetime(2026, 5, 2, 12, 0, 15, tzinfo=UTC)


def test_stimulus_aligned_segment_start_is_anchored_independent_of_buffer_start() -> None:
    active = module_c_orchestrator._ActiveSession(  # noqa: SLF001
        session_id=uuid.UUID("00000000-0000-4000-8000-000000000001"),
        stream_url="test://stream",
        experiment_id="greeting_line_v1",
        experiment_row_id=1,
        active_arm="warm_welcome",
        stimulus_definition=_stimulus_definition(),
        bandit_decision_snapshot=_BANDIT_SNAPSHOT,
        stimulus_time_s=datetime(2026, 5, 2, 12, 0, 8, tzinfo=UTC).timestamp(),
    )

    start = module_c_orchestrator._stimulus_aligned_segment_start_utc(active)  # noqa: SLF001

    # Segment end = stim + 5 s = 12:00:13. Segment start = end − 30 s
    # = 11:59:43, regardless of how much pre-stim audio the buffer
    # actually has — the run loop pre-pads silence when needed.
    assert start == datetime(2026, 5, 2, 11, 59, 43, tzinfo=UTC)


def test_build_stimulus_segment_pcm_pre_pads_silence_when_buffer_short(
    tmp_path: Path,  # noqa: ARG001
) -> None:
    segment_bytes = (
        module_c_orchestrator.SEGMENT_WINDOW_SECONDS
        * module_c_orchestrator.PCM_SAMPLE_RATE_HZ
        * module_c_orchestrator.AUDIO_SAMPLE_WIDTH_BYTES
        * module_c_orchestrator.PCM_CHANNELS
    )
    # Buffer holds 13 s of non-zero audio.
    real_audio_seconds = 13
    real_audio_bytes = (
        real_audio_seconds
        * module_c_orchestrator.PCM_SAMPLE_RATE_HZ
        * module_c_orchestrator.AUDIO_SAMPLE_WIDTH_BYTES
        * module_c_orchestrator.PCM_CHANNELS
    )
    audio_buffer = bytearray(b"\x7f\x01" * (real_audio_bytes // 2))
    buffer_start = datetime(2026, 5, 2, 12, 0, 0, tzinfo=UTC)
    # Stim at 12:00:08 → segment [11:59:43, 12:00:13]. Buffer covers
    # [12:00:00, 12:00:13]; segment needs 17 s of silence pre-pad.
    desired_segment_start = datetime(2026, 5, 2, 11, 59, 43, tzinfo=UTC)
    desired_end_offset = module_c_orchestrator._segment_audio_offset_bytes(  # noqa: SLF001
        segment_start_utc=buffer_start,
        target_start_utc=desired_segment_start
        + timedelta(seconds=module_c_orchestrator.SEGMENT_WINDOW_SECONDS),
    )

    pcm = module_c_orchestrator._build_stimulus_segment_pcm(  # noqa: SLF001
        audio_buffer,
        buffer_start_utc=buffer_start,
        desired_segment_start_utc=desired_segment_start,
        desired_segment_end_offset_bytes=desired_end_offset,
        segment_bytes=segment_bytes,
    )

    assert len(pcm) == segment_bytes
    silence_prefix_bytes = (
        17
        * module_c_orchestrator.PCM_SAMPLE_RATE_HZ
        * module_c_orchestrator.AUDIO_SAMPLE_WIDTH_BYTES
        * module_c_orchestrator.PCM_CHANNELS
    )
    assert pcm[:silence_prefix_bytes] == b"\x00" * silence_prefix_bytes
    # The next byte after the silence pad is real audio.
    assert pcm[silence_prefix_bytes : silence_prefix_bytes + 2] == b"\x7f\x01"


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
