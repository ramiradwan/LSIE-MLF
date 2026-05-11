from __future__ import annotations

import multiprocessing as mp
import queue
import uuid
from collections import deque
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import patch

import pytest

from services.desktop_app.ipc import IpcChannels
from services.desktop_app.ipc.control_messages import (
    AnalyticsResultMessage,
    AudioBlockRef,
    InferenceControlMessage,
    LiveSessionControlMessage,
    PcmBlockAckMessage,
    VisualAnalyticsStateMessage,
)
from services.desktop_app.ipc.shared_buffers import PcmBlock, PcmBlockMetadata, write_pcm_block
from services.desktop_app.processes import gpu_ml_worker
from services.desktop_app.processes.gpu_ml_worker import (
    DesktopScreenrecordCapture,
    LiveVisualTracker,
    _bgr_frame_from_raw,
    _default_video_capture_factory,
    _derive_segment_start_time_s,
    _drain_live_control,
    _handoff_window_s,
    _handoff_with_tracker_au12,
    _ml_inbox_poll_timeout,
    _publish_analytics_result,
)

SEGMENT_ID = "a" * 64
SESSION_ID = "00000000-0000-4000-8000-000000000001"
SESSION_UUID = uuid.UUID(SESSION_ID)
DECISION_CONTEXT_HASH = "b" * 64


class StubTranscriptionEngine:
    def transcribe(self, audio: object) -> str:
        del audio
        return "hello creator"


class StubTextPreprocessor:
    def preprocess(self, text: str) -> str:
        return text


class StubSemanticEvaluator:
    last_semantic_method = "cross_encoder"
    last_semantic_method_version = "test-v1"

    def evaluate(self, expected_greeting: str, actual_utterance: str) -> dict[str, Any]:
        del expected_greeting, actual_utterance
        return {
            "reasoning": "cross_encoder_high_match",
            "is_match": True,
            "confidence_score": 0.91,
        }


class FakeVideoCapture:
    def __init__(self, frames: list[object]) -> None:
        self.frames = frames
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def get_latest_frame(self) -> object | None:
        if not self.frames:
            return None
        return self.frames.pop(0)


class FakeFaceMesh:
    def __init__(self, landmarks: list[object | None]) -> None:
        self.landmarks = landmarks
        self.closed = False

    def extract_landmarks(self, frame: object) -> object | None:
        del frame
        if not self.landmarks:
            return None
        return self.landmarks.pop(0)

    def close(self) -> None:
        self.closed = True


class FakeAu12Normalizer:
    def __init__(self, intensities: list[float]) -> None:
        self.intensities = intensities
        self.calibration_buffer: list[float] = []

    def compute_bounded_intensity(self, landmarks: object, *, is_calibrating: bool) -> float:
        del landmarks
        if is_calibrating:
            self.calibration_buffer.append(0.4)
            return 0.0
        if not self.intensities:
            return 0.0
        return self.intensities.pop(0)


class _HangingThread:
    def __init__(self) -> None:
        self.join_timeout: float | None = None

    def is_alive(self) -> bool:
        return True

    def join(self, timeout: float | None = None) -> None:
        self.join_timeout = timeout


def _handoff() -> dict[str, Any]:
    timestamp = datetime(2026, 5, 2, 12, 0, tzinfo=UTC).isoformat()
    return {
        "session_id": SESSION_ID,
        "segment_id": SEGMENT_ID,
        "segment_window_start_utc": timestamp,
        "segment_window_end_utc": timestamp,
        "timestamp_utc": timestamp,
        "media_source": {
            "stream_url": "https://example.com/stream",
            "codec": "h264",
            "resolution": [1920, 1080],
        },
        "segments": [],
        "_active_arm": "warm_welcome",
        "_experiment_id": 1,
        "_expected_greeting": "Say hello to the creator",
        "_stimulus_time": 100.0,
        "_au12_series": [
            {"timestamp_s": 100.1, "intensity": 0.2},
            {"timestamp_s": 100.2, "intensity": 0.5},
        ],
        "_bandit_decision_snapshot": {
            "selection_method": "thompson_sampling",
            "selection_time_utc": timestamp,
            "experiment_id": 1,
            "policy_version": "ts-v1",
            "selected_arm_id": "warm_welcome",
            "candidate_arm_ids": ["warm_welcome", "direct_question"],
            "posterior_by_arm": {
                "warm_welcome": {"alpha": 1.0, "beta": 1.0},
                "direct_question": {"alpha": 1.0, "beta": 1.0},
            },
            "sampled_theta_by_arm": {
                "warm_welcome": 0.72,
                "direct_question": 0.44,
            },
            "expected_greeting": "Say hello to the creator",
            "decision_context_hash": DECISION_CONTEXT_HASH,
            "random_seed": 42,
        },
    }


def _control_message(
    *,
    forward_fields: dict[str, Any] | None = None,
) -> tuple[InferenceControlMessage, PcmBlock]:
    block = write_pcm_block(b"\0\0" * 16000)
    msg = InferenceControlMessage(
        handoff=_handoff(),
        audio=AudioBlockRef.from_metadata(block.metadata),
        forward_fields={} if forward_fields is None else forward_fields,
    )
    return msg, block


def _start_control(now: datetime) -> LiveSessionControlMessage:
    return LiveSessionControlMessage(
        action="start",
        session_id=SESSION_UUID,
        stream_url="test://stream",
        experiment_id="greeting_line_v1",
        active_arm="warm_welcome",
        expected_greeting="Say hello to the creator",
        timestamp_utc=now,
    )


def test_gpu_ml_worker_drains_live_control_to_visual_state_messages() -> None:
    channels = IpcChannels(
        ml_inbox=cast("Any", queue.Queue()),
        drift_updates=cast("Any", queue.Queue()),
        analytics_inbox=cast("Any", queue.Queue()),
        pcm_acks=cast("Any", queue.Queue()),
        live_control=cast("Any", queue.Queue()),
    )
    now = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    controls = [
        _start_control(now),
        LiveSessionControlMessage(
            action="stimulus",
            session_id=SESSION_UUID,
            stream_url="test://stream",
            experiment_id="greeting_line_v1",
            active_arm="warm_welcome",
            expected_greeting="Say hello to the creator",
            stimulus_time_s=100.0,
            timestamp_utc=now,
        ),
        LiveSessionControlMessage(
            action="end",
            session_id=SESSION_UUID,
            timestamp_utc=now,
        ),
    ]
    for control in controls:
        assert channels.live_control is not None
        channels.live_control.put(control.model_dump(mode="json"))

    tracker = LiveVisualTracker(
        video_capture_factory=lambda _path: FakeVideoCapture([]),
    )

    _drain_live_control(channels, tracker)

    visual = [
        VisualAnalyticsStateMessage.model_validate(channels.analytics_inbox.get(timeout=1.0))
        for _ in controls
    ]
    assert [message.status for message in visual] == [
        "waiting_for_face",
        "waiting_for_face",
        "no_session",
    ]
    assert visual[0].face_present is False
    assert visual[0].latest_au12_intensity is None
    assert visual[0].active_arm == "warm_welcome"
    assert visual[0].expected_greeting == "Say hello to the creator"
    assert visual[1].active_arm == "warm_welcome"
    assert visual[1].latest_au12_intensity is None


def test_gpu_ml_worker_visual_tick_restores_health_after_receiving_face_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_ml_worker, "_VISUAL_CALIBRATION_FRAMES_REQUIRED", 1)
    now = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    channels = IpcChannels(
        ml_inbox=cast("Any", queue.Queue()),
        drift_updates=cast("Any", queue.Queue()),
        analytics_inbox=cast("Any", queue.Queue()),
        pcm_acks=cast("Any", queue.Queue()),
        live_control=cast("Any", queue.Queue()),
    )
    tracker = LiveVisualTracker(
        video_capture_factory=lambda _path: FakeVideoCapture([object()]),
        face_mesh_factory=lambda: FakeFaceMesh([object()]),
        au12_factory=lambda: FakeAu12Normalizer([]),
    )
    statuses: list[gpu_ml_worker._GpuWorkerStatus] = []  # noqa: SLF001
    tracker.handle_control(_start_control(now))

    gpu_ml_worker._publish_visual_tick(channels, tracker, status_callback=statuses.append)  # noqa: SLF001

    assert statuses[-1].state == "ok"
    assert statuses[-1].detail == "Live visual tracking is receiving phone frames."


def test_gpu_ml_worker_stimulus_control_preserves_calibrated_visual_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_ml_worker, "_VISUAL_CALIBRATION_FRAMES_REQUIRED", 1)
    now = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    tracker = LiveVisualTracker(
        video_capture_factory=lambda _path: FakeVideoCapture([object()]),
        face_mesh_factory=lambda: FakeFaceMesh([object()]),
        au12_factory=lambda: FakeAu12Normalizer([]),
    )
    tracker.handle_control(_start_control(now))
    tracker.tick(now)

    visual = tracker.handle_control(
        LiveSessionControlMessage(
            action="stimulus",
            session_id=SESSION_UUID,
            stream_url="test://stream",
            experiment_id="greeting_line_v1",
            active_arm="warm_welcome",
            expected_greeting="Say hello to the creator",
            stimulus_time_s=now.timestamp(),
            timestamp_utc=now,
        )
    )

    assert visual.status == "post_stimulus"
    assert visual.face_present is True
    assert visual.is_calibrating is False
    assert visual.calibration_frames_accumulated == 1


def test_gpu_ml_worker_visual_tick_publishes_waiting_state_when_no_frame() -> None:
    now = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    capture = FakeVideoCapture([])
    tracker = LiveVisualTracker(
        video_capture_factory=lambda _path: capture,
        face_mesh_factory=lambda: FakeFaceMesh([]),
        au12_factory=lambda: FakeAu12Normalizer([]),
    )
    tracker.handle_control(_start_control(now))

    outcome = tracker.tick(now)
    visual = outcome.visual

    assert outcome.missing_frame is True
    assert visual is not None
    assert visual.status == "waiting_for_face"
    assert visual.face_present is False
    assert visual.calibration_frames_accumulated == 0
    assert visual.latest_au12_intensity is None


def test_gpu_ml_worker_visual_tick_waits_for_real_face() -> None:
    now = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    capture = FakeVideoCapture([object()])
    tracker = LiveVisualTracker(
        video_capture_factory=lambda _path: capture,
        face_mesh_factory=lambda: FakeFaceMesh([None]),
        au12_factory=lambda: FakeAu12Normalizer([]),
    )
    tracker.handle_control(_start_control(now))

    outcome = tracker.tick(now)
    visual = outcome.visual

    assert outcome.missing_frame is False
    assert visual is not None
    assert visual.status == "waiting_for_face"
    assert visual.face_present is False
    assert visual.calibration_frames_accumulated == 0
    assert visual.latest_au12_intensity is None


def test_gpu_ml_worker_visual_tick_calibrates_and_reaches_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_ml_worker, "_VISUAL_CALIBRATION_FRAMES_REQUIRED", 2)
    now = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    capture = FakeVideoCapture([object(), object()])
    au12 = FakeAu12Normalizer([])
    tracker = LiveVisualTracker(
        video_capture_factory=lambda _path: capture,
        face_mesh_factory=lambda: FakeFaceMesh([object(), object()]),
        au12_factory=lambda: au12,
    )
    tracker.handle_control(_start_control(now))

    calibrating = tracker.tick(now).visual
    ready = tracker.tick(now).visual

    assert calibrating is not None
    assert calibrating.status == "calibrating"
    assert calibrating.face_present is True
    assert calibrating.calibration_frames_accumulated == 1
    assert ready is not None
    assert ready.status == "ready"
    assert ready.face_present is True
    assert ready.is_calibrating is False
    assert ready.calibration_frames_accumulated == 2
    assert len(au12.calibration_buffer) == 2


def test_gpu_ml_worker_visual_tick_records_real_post_stimulus_au12(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_ml_worker, "_VISUAL_CALIBRATION_FRAMES_REQUIRED", 1)
    now = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    capture = FakeVideoCapture([object(), object()])
    tracker = LiveVisualTracker(
        video_capture_factory=lambda _path: capture,
        face_mesh_factory=lambda: FakeFaceMesh([object(), object()]),
        au12_factory=lambda: FakeAu12Normalizer([0.62]),
    )
    tracker.handle_control(_start_control(now))
    tracker.tick(now)
    tracker.handle_control(
        LiveSessionControlMessage(
            action="stimulus",
            session_id=SESSION_UUID,
            stream_url="test://stream",
            experiment_id="greeting_line_v1",
            stimulus_time_s=now.timestamp(),
            timestamp_utc=now,
        )
    )

    post_stimulus = tracker.tick(now).visual
    observations = tracker.drain_au12_observations(
        start_s=now.timestamp(),
        end_s=now.timestamp(),
    )

    assert post_stimulus is not None
    assert post_stimulus.status == "post_stimulus"
    assert post_stimulus.latest_au12_intensity == 0.62
    assert observations == [{"timestamp_s": now.timestamp(), "intensity": 0.62}]


def test_gpu_ml_worker_visual_tracker_keeps_pre_stimulus_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_ml_worker, "_VISUAL_CALIBRATION_FRAMES_REQUIRED", 1)
    now = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    baseline = datetime(2026, 5, 2, 12, 0, 6, tzinfo=UTC)
    response = datetime(2026, 5, 2, 12, 0, 10, tzinfo=UTC)
    capture = FakeVideoCapture([object(), object(), object(), object()])
    tracker = LiveVisualTracker(
        video_capture_factory=lambda _path: capture,
        face_mesh_factory=lambda: FakeFaceMesh([object(), object(), object(), object()]),
        au12_factory=lambda: FakeAu12Normalizer([0.25, 0.75, 0.9]),
    )
    tracker.handle_control(_start_control(now))
    tracker.tick(now)
    tracker.tick(baseline)
    tracker.handle_control(
        LiveSessionControlMessage(
            action="stimulus",
            session_id=SESSION_UUID,
            stream_url="test://stream",
            experiment_id="greeting_line_v1",
            stimulus_time_s=response.timestamp() - 1.0,
            timestamp_utc=response,
        )
    )
    tracker.tick(response)

    observations = tracker.drain_au12_observations(
        start_s=response.timestamp() - 5.0,
        end_s=response.timestamp(),
    )

    assert observations == [
        {"timestamp_s": baseline.timestamp(), "intensity": 0.25},
        {"timestamp_s": response.timestamp(), "intensity": 0.75},
    ]


def test_gpu_ml_worker_visual_tick_keeps_calibrated_state_without_new_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_ml_worker, "_VISUAL_CALIBRATION_FRAMES_REQUIRED", 1)
    now = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    capture = FakeVideoCapture([object()])
    tracker = LiveVisualTracker(
        video_capture_factory=lambda _path: capture,
        face_mesh_factory=lambda: FakeFaceMesh([object(), object()]),
        au12_factory=lambda: FakeAu12Normalizer([0.25]),
    )
    tracker.handle_control(_start_control(now))
    tracker.tick(now)
    post_stimulus = tracker.handle_control(
        LiveSessionControlMessage(
            action="stimulus",
            session_id=SESSION_UUID,
            stream_url="test://stream",
            experiment_id="greeting_line_v1",
            stimulus_time_s=now.timestamp(),
            timestamp_utc=now,
        )
    )
    visual = tracker.tick(now).visual

    assert post_stimulus.status == "post_stimulus"
    assert visual is not None
    assert visual.status == "post_stimulus"
    assert visual.face_present is True
    assert visual.is_calibrating is False


def test_gpu_ml_worker_inbox_poll_waits_only_until_next_visual_tick() -> None:
    now = 10.0

    assert _ml_inbox_poll_timeout(
        now + gpu_ml_worker._VISUAL_TICK_INTERVAL_S,  # noqa: SLF001
        now,
    ) == pytest.approx(gpu_ml_worker._VISUAL_TICK_INTERVAL_S)  # noqa: SLF001
    assert _ml_inbox_poll_timeout(now + 2.0, now) == gpu_ml_worker.INBOX_POLL_TIMEOUT_S
    assert _ml_inbox_poll_timeout(now - 0.1, now) == 0.0


def test_gpu_ml_worker_derives_acoustic_segment_start_from_handoff_clock() -> None:
    timestamp = datetime(2026, 5, 2, 12, 0, 30, tzinfo=UTC)
    audio = b"\0\0" * 16000 * 30
    start = _derive_segment_start_time_s(
        timestamp_utc=timestamp.isoformat(),
        audio_data=audio,
        sample_rate=16000,
    )

    assert start == pytest.approx(timestamp.timestamp() - 30.0)


def test_gpu_ml_worker_uses_handoff_clock_for_tracker_au12_window() -> None:
    timestamp = datetime(2026, 5, 2, 12, 0, 30, tzinfo=UTC)
    audio = b"\0\0" * 16000 * 30
    handoff = {
        "segment_window_start_utc": (timestamp.replace(second=0)).isoformat(),
        "segment_window_end_utc": timestamp.isoformat(),
    }

    window = _handoff_window_s(handoff, audio)

    assert window == pytest.approx(
        (
            timestamp.replace(second=0).timestamp(),
            timestamp.timestamp(),
        )
    )


def test_gpu_ml_worker_keeps_tracker_au12_timestamps_on_handoff_clock() -> None:
    timestamp = datetime(2026, 5, 2, 12, 0, 30, tzinfo=UTC)
    tracker = LiveVisualTracker()
    tracker._au12_observations.append(  # noqa: SLF001
        gpu_ml_worker._Au12Observation(  # noqa: SLF001
            timestamp_s=timestamp.timestamp(),
            intensity=0.7,
        )
    )
    handoff = {
        "segment_window_start_utc": timestamp.replace(second=0).isoformat(),
        "segment_window_end_utc": timestamp.isoformat(),
    }

    enriched = _handoff_with_tracker_au12(
        handoff,
        audio_data=b"\0\0" * 16000 * 30,
        tracker=tracker,
    )

    assert enriched["_au12_series"] == [{"timestamp_s": timestamp.timestamp(), "intensity": 0.7}]


def test_gpu_ml_worker_visual_tracker_releases_resources() -> None:
    now = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    capture = FakeVideoCapture([object()])
    face_mesh = FakeFaceMesh([None])
    tracker = LiveVisualTracker(
        video_capture_factory=lambda _path: capture,
        face_mesh_factory=lambda: face_mesh,
        au12_factory=lambda: FakeAu12Normalizer([]),
    )
    tracker.handle_control(_start_control(now))
    tracker.tick(now)

    visual = tracker.handle_control(
        LiveSessionControlMessage(
            action="end",
            session_id=SESSION_UUID,
            timestamp_utc=now,
        )
    )

    assert visual.status == "no_session"
    assert capture.stopped is True
    assert face_mesh.closed is True


def test_default_live_visual_capture_uses_screenrecord_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "services.desktop_app.processes.gpu_ml_worker.find_executable",
        lambda name, env_override=None: f"resolved-{name}",
    )

    capture = _default_video_capture_factory("ignored-video-stream.mkv")

    assert isinstance(capture, DesktopScreenrecordCapture)
    assert capture._adb_path == "resolved-adb"  # noqa: SLF001
    assert capture._ffmpeg_path == "resolved-ffmpeg"  # noqa: SLF001


def test_desktop_screenrecord_capture_starts_adb_and_ffmpeg_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned: list[tuple[list[str], dict[str, object]]] = []

    class FakeSupervisedProcess:
        def __init__(self, args: list[str], **kwargs: object) -> None:
            spawned.append((args, kwargs))
            self.stdout = object()
            self.closed = False

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(gpu_ml_worker, "SupervisedProcess", FakeSupervisedProcess)
    capture = DesktopScreenrecordCapture("adb", "ffmpeg")

    capture._start_stream()  # noqa: SLF001

    assert spawned[0][0][:4] == ["adb", "exec-out", "screenrecord", "--output-format=h264"]
    expected_size = (
        f"--size={gpu_ml_worker._VISUAL_FRAME_WIDTH}x"  # noqa: SLF001
        f"{gpu_ml_worker._VISUAL_FRAME_HEIGHT}"  # noqa: SLF001
    )
    assert expected_size in spawned[0][0]
    assert spawned[1][0][:8] == [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "h264",
        "-i",
        "pipe:0",
    ]
    assert "rawvideo" in spawned[1][0]
    assert spawned[1][1]["stdin"] is cast(Any, capture._source_proc).stdout  # noqa: SLF001


def test_bgr_frame_from_raw_returns_expected_shape() -> None:
    payload = b"\0" * gpu_ml_worker._VISUAL_FRAME_BYTES  # noqa: SLF001

    frame = _bgr_frame_from_raw(payload)

    assert frame.shape == (
        gpu_ml_worker._VISUAL_FRAME_HEIGHT,  # noqa: SLF001
        gpu_ml_worker._VISUAL_FRAME_WIDTH,  # noqa: SLF001
        3,
    )


def test_desktop_screenrecord_stop_does_not_block_on_hung_capture_thread() -> None:
    capture = DesktopScreenrecordCapture("adb", "ffmpeg")
    hanging = _HangingThread()
    capture._thread = cast("Any", hanging)
    capture._frame_buffer = deque([object()])

    capture.stop()

    assert hanging.join_timeout == gpu_ml_worker._DESKTOP_STREAM_STOP_TIMEOUT_S
    assert capture._thread is hanging
    assert not capture._frame_buffer


def test_gpu_ml_worker_discards_malformed_live_control() -> None:
    channels = IpcChannels(
        ml_inbox=cast("Any", queue.Queue()),
        drift_updates=cast("Any", queue.Queue()),
        analytics_inbox=cast("Any", queue.Queue()),
        pcm_acks=cast("Any", queue.Queue()),
        live_control=cast("Any", queue.Queue()),
    )
    assert channels.live_control is not None
    channels.live_control.put({"schema_version": "ws5.p4.live_session_control.v1"})

    _drain_live_control(channels, LiveVisualTracker())

    assert channels.analytics_inbox.empty()


def test_gpu_ml_worker_publishes_bounded_result_while_speech_model_warms() -> None:
    ctx = mp.get_context("spawn")
    channels = IpcChannels(
        ml_inbox=ctx.Queue(),
        drift_updates=ctx.Queue(),
        analytics_inbox=ctx.Queue(),
        pcm_acks=ctx.Queue(),
    )
    statuses: list[gpu_ml_worker._GpuWorkerStatus] = []  # noqa: SLF001
    msg, block = _control_message()
    try:
        with (
            patch("packages.ml_core.audio_pipe.pcm_to_wav_bytes", return_value=b"RIFF"),
            patch("packages.ml_core.transcription.TranscriptionEngine") as engine_cls,
        ):
            engine_cls.return_value.transcribe.side_effect = RuntimeError("not warm yet")
            _publish_analytics_result(channels, msg, status_callback=statuses.append)

        assert channels.pcm_acks is not None
        ack = PcmBlockAckMessage.model_validate(channels.pcm_acks.get(timeout=1.0))
        assert ack.name == msg.audio.name
        raw = channels.analytics_inbox.get(timeout=1.0)
        analytics = AnalyticsResultMessage.model_validate(raw)
        assert analytics.handoff.segment_id == SEGMENT_ID
        assert analytics.transcription == ""
        assert analytics.semantic.is_match is False
        assert analytics.semantic.reasoning == "semantic_error"
        assert [status.state for status in statuses] == ["ok"]
    finally:
        block.close_and_unlink()


def test_gpu_ml_worker_drops_expired_pcm_block_without_traceback() -> None:
    ctx = mp.get_context("spawn")
    channels = IpcChannels(
        ml_inbox=ctx.Queue(),
        drift_updates=ctx.Queue(),
        analytics_inbox=ctx.Queue(),
        pcm_acks=ctx.Queue(),
    )
    statuses: list[gpu_ml_worker._GpuWorkerStatus] = []  # noqa: SLF001
    msg = InferenceControlMessage(
        handoff=_handoff(),
        audio=AudioBlockRef.from_metadata(
            PcmBlockMetadata(name="lsie_ipc_pcm_missing", byte_length=2, sha256="0" * 64)
        ),
    )

    _publish_analytics_result(
        channels,
        msg,
        transcription_engine=StubTranscriptionEngine(),
        status_callback=statuses.append,
    )

    assert channels.pcm_acks is not None
    ack = PcmBlockAckMessage.model_validate(channels.pcm_acks.get(timeout=1.0))
    assert ack.name == msg.audio.name
    assert channels.analytics_inbox.empty()
    assert statuses == []


def test_gpu_ml_worker_publishes_analytics_result_with_null_physiology_omitted() -> None:
    ctx = mp.get_context("spawn")
    channels = IpcChannels(
        ml_inbox=ctx.Queue(),
        drift_updates=ctx.Queue(),
        analytics_inbox=ctx.Queue(),
        pcm_acks=ctx.Queue(),
    )
    msg, block = _control_message()
    msg.handoff["_physiological_context"] = None
    try:
        with (
            patch("packages.ml_core.audio_pipe.pcm_to_wav_bytes", return_value=b"RIFF"),
            patch("packages.ml_core.preprocessing.TextPreprocessor", StubTextPreprocessor),
            patch("packages.ml_core.semantic.SemanticEvaluator", StubSemanticEvaluator),
        ):
            _publish_analytics_result(
                channels,
                msg,
                transcription_engine=StubTranscriptionEngine(),
            )

        raw = channels.analytics_inbox.get(timeout=1.0)
        analytics = AnalyticsResultMessage.model_validate(raw)
        assert analytics.handoff.physiological_context is None
        assert "_physiological_context" not in analytics.handoff.model_dump(
            mode="json",
            by_alias=True,
            exclude_none=True,
        )
    finally:
        block.close_and_unlink()


def test_gpu_ml_worker_publishes_analytics_result_to_inbox() -> None:
    ctx = mp.get_context("spawn")
    channels = IpcChannels(
        ml_inbox=ctx.Queue(),
        drift_updates=ctx.Queue(),
        analytics_inbox=ctx.Queue(),
        pcm_acks=ctx.Queue(),
    )
    msg, block = _control_message()
    try:
        with (
            patch("packages.ml_core.audio_pipe.pcm_to_wav_bytes", return_value=b"RIFF"),
            patch("packages.ml_core.preprocessing.TextPreprocessor", StubTextPreprocessor),
            patch("packages.ml_core.semantic.SemanticEvaluator", StubSemanticEvaluator),
        ):
            _publish_analytics_result(
                channels,
                msg,
                transcription_engine=StubTranscriptionEngine(),
            )

        assert channels.pcm_acks is not None
        ack = PcmBlockAckMessage.model_validate(channels.pcm_acks.get(timeout=1.0))
        assert ack.name == msg.audio.name
        raw = channels.analytics_inbox.get(timeout=1.0)
        analytics = AnalyticsResultMessage.model_validate(raw)
        assert analytics.handoff.segment_id == SEGMENT_ID
        assert analytics.semantic.is_match is True
        assert analytics.transcription == "hello creator"
        dumped = analytics.model_dump(mode="json")
        assert "_audio_data" not in dumped
        assert "_frame_data" not in dumped
    finally:
        block.close_and_unlink()


def test_gpu_ml_worker_forwards_attribution_inputs_from_ipc_tail() -> None:
    ctx = mp.get_context("spawn")
    channels = IpcChannels(
        ml_inbox=ctx.Queue(),
        drift_updates=ctx.Queue(),
        analytics_inbox=ctx.Queue(),
        pcm_acks=ctx.Queue(),
    )
    msg, block = _control_message(
        forward_fields={
            "_creator_follow": True,
            "_creator_follow_outcome": {
                "outcome_type": "creator_follow",
                "outcome_time_utc": "2026-05-02T12:00:02+00:00",
                "source_system": "desktop_test",
                "source_event_ref": SEGMENT_ID,
                "confidence": 1.0,
            },
        }
    )
    try:
        with (
            patch("packages.ml_core.audio_pipe.pcm_to_wav_bytes", return_value=b"RIFF"),
            patch("packages.ml_core.preprocessing.TextPreprocessor", StubTextPreprocessor),
            patch("packages.ml_core.semantic.SemanticEvaluator", StubSemanticEvaluator),
        ):
            _publish_analytics_result(
                channels,
                msg,
                transcription_engine=StubTranscriptionEngine(),
            )

        raw = channels.analytics_inbox.get(timeout=1.0)
        analytics = AnalyticsResultMessage.model_validate(raw)
        assert analytics.attribution is not None
        assert analytics.attribution.creator_follow is True
        assert analytics.attribution.outcome_event is not None
        assert analytics.attribution.outcome_event["source_system"] == "desktop_test"
    finally:
        block.close_and_unlink()


def test_gpu_ml_worker_uses_real_tracker_au12_observations() -> None:
    timestamp = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    tracker = LiveVisualTracker()
    tracker._au12_observations.append(  # noqa: SLF001
        gpu_ml_worker._Au12Observation(timestamp_s=timestamp.timestamp(), intensity=0.7)  # noqa: SLF001
    )
    ctx = mp.get_context("spawn")
    channels = IpcChannels(
        ml_inbox=ctx.Queue(),
        drift_updates=ctx.Queue(),
        analytics_inbox=ctx.Queue(),
        pcm_acks=ctx.Queue(),
    )
    msg, block = _control_message()
    try:
        with (
            patch("packages.ml_core.audio_pipe.pcm_to_wav_bytes", return_value=b"RIFF"),
            patch("packages.ml_core.preprocessing.TextPreprocessor", StubTextPreprocessor),
            patch("packages.ml_core.semantic.SemanticEvaluator", StubSemanticEvaluator),
        ):
            _publish_analytics_result(
                channels,
                msg,
                tracker,
                transcription_engine=StubTranscriptionEngine(),
            )

        assert channels.pcm_acks is not None
        ack = PcmBlockAckMessage.model_validate(channels.pcm_acks.get(timeout=1.0))
        assert ack.name == msg.audio.name
        raw = channels.analytics_inbox.get(timeout=1.0)
        analytics = AnalyticsResultMessage.model_validate(raw)
        assert [sample.intensity for sample in analytics.handoff.au12_series] == [0.7]
    finally:
        block.close_and_unlink()


def test_gpu_ml_worker_publishes_bounded_result_for_empty_transcription() -> None:
    class EmptyTranscriptionEngine:
        def transcribe(self, audio: object) -> str:
            del audio
            return ""

    msg, block = _control_message()
    ctx = mp.get_context("spawn")
    try:
        channels = IpcChannels(
            ml_inbox=ctx.Queue(),
            drift_updates=ctx.Queue(),
            analytics_inbox=ctx.Queue(),
            pcm_acks=ctx.Queue(),
        )
        with patch("packages.ml_core.audio_pipe.pcm_to_wav_bytes", return_value=b"RIFF"):
            _publish_analytics_result(
                channels,
                msg,
                transcription_engine=EmptyTranscriptionEngine(),
            )

        assert channels.pcm_acks is not None
        ack = PcmBlockAckMessage.model_validate(channels.pcm_acks.get(timeout=1.0))
        assert ack.name == msg.audio.name
        raw = channels.analytics_inbox.get(timeout=1.0)
        analytics = AnalyticsResultMessage.model_validate(raw)
        assert analytics.transcription == ""
        assert analytics.semantic.is_match is False
        assert analytics.semantic.reasoning == "semantic_error"
        assert analytics.semantic.confidence_score == 0.0
    finally:
        block.close_and_unlink()
