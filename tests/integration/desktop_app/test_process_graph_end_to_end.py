from __future__ import annotations

import json
import multiprocessing as mp
import multiprocessing.synchronize as mpsync
import queue
import sqlite3
import struct
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from packages.schemas.evaluation import StimulusDefinition, StimulusPayload
from packages.schemas.inference_handoff import InferenceHandoffPayload
from services.desktop_app.cloud.outbox import CloudOutbox
from services.desktop_app.ipc import IpcChannels
from services.desktop_app.ipc.control_messages import (
    AnalyticsResultMessage,
    InferenceControlMessage,
    LiveSessionControlMessage,
    PcmBlockAckMessage,
)
from services.desktop_app.ipc.shared_buffers import read_pcm_block
from services.desktop_app.process_graph import ProcessGraph, process_modules_for_mode
from services.desktop_app.processes import module_c_orchestrator
from services.desktop_app.processes.analytics_state_worker import (
    LocalAnalyticsProcessor,
    QueueLike,
    _run_loop,
)
from services.desktop_app.processes.capture_supervisor import (
    CaptureLayout,
    _build_audio_scrcpy_args,
)
from services.desktop_app.processes.gpu_ml_worker import _publish_analytics_result
from services.desktop_app.processes.module_c_orchestrator import DesktopSegmentDispatcher
from services.desktop_app.state.sqlite_schema import bootstrap_schema

SESSION_ID = uuid.UUID("00000000-0000-4000-8000-000000000001")
CLIENT_ID = "desktop-process-graph-e2e"


class _StubTranscriptionEngine:
    def transcribe(self, audio: object) -> str:
        del audio
        return "hello creator"


class _StubTextPreprocessor:
    def preprocess(self, text: str) -> str:
        return text


class _StubSemanticEvaluator:
    last_semantic_method = "cross_encoder"
    last_semantic_method_version = "process-graph-e2e-v1"

    def evaluate(self, expected_response_rule: str, actual_utterance: str) -> dict[str, object]:
        del expected_response_rule, actual_utterance
        return {
            "reasoning": "cross_encoder_high_match",
            "is_match": True,
            "confidence_score": 0.91,
        }


class _OneMessageInbox:
    def __init__(self, raw: object, shutdown_event: mpsync.Event) -> None:
        self._raw = raw
        self._shutdown_event = shutdown_event

    def get(self, block: bool = True, timeout: float | None = None) -> object:
        del block, timeout
        self._shutdown_event.set()
        return self._raw


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
        expected_response_rule="The live streamer acknowledges the stimulus or responds to it.",
    )


def _bandit_snapshot(stimulus: StimulusDefinition) -> dict[str, object]:
    return {
        "selection_method": "thompson_sampling",
        "selection_time_utc": "2026-05-02T12:00:00Z",
        "experiment_id": 1,
        "policy_version": "desktop_replay_v1",
        "selected_arm_id": "warm_welcome",
        "candidate_arm_ids": ["direct_question", "warm_welcome"],
        "posterior_by_arm": {
            "direct_question": {"alpha": 1.0, "beta": 5.0},
            "warm_welcome": {"alpha": 2.0, "beta": 3.0},
        },
        "sampled_theta_by_arm": {"direct_question": 0.2, "warm_welcome": 0.6},
        "stimulus_modality": stimulus.stimulus_modality,
        "stimulus_payload": stimulus.stimulus_payload.model_dump(mode="json"),
        "expected_stimulus_rule": stimulus.expected_stimulus_rule,
        "expected_response_rule": stimulus.expected_response_rule,
        "decision_context_hash": "a" * 64,
        "random_seed": 42,
    }


def _channels() -> IpcChannels:
    return IpcChannels(
        ml_inbox=cast("Any", queue.Queue()),
        drift_updates=cast("Any", queue.Queue()),
        analytics_inbox=cast("Any", queue.Queue()),
        pcm_acks=cast("Any", queue.Queue()),
        live_control=cast("Any", queue.Queue()),
        segment_control=cast("Any", queue.Queue()),
        visual_state_updates=cast("Any", queue.Queue()),
    )


def _bootstrap_db(tmp_path: Path, stimulus: StimulusDefinition) -> Path:
    db_path = tmp_path / "desktop.sqlite"
    snapshot = _bandit_snapshot(stimulus)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        bootstrap_schema(conn)
        conn.execute(
            """
            INSERT INTO sessions (
                session_id, stream_url, experiment_id, active_arm, stimulus_definition,
                bandit_decision_snapshot, started_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(SESSION_ID),
                "replay://capture-file",
                "greeting_line_v1",
                "warm_welcome",
                stimulus.model_dump_json(),
                json.dumps(snapshot, sort_keys=True, separators=(",", ":")),
                "2026-05-02 12:00:00",
            ),
        )
    finally:
        conn.close()
    return db_path


def _write_audio_header(path: Path) -> None:
    channels = 1
    sample_rate_hz = module_c_orchestrator.AUDIO_SAMPLE_RATE_HZ
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


def _append_capture_audio(path: Path, frames: int) -> None:
    with path.open("ab") as audio_file:
        audio_file.write(b"\x01\x02" * frames)


def _next_ml_control(channels: IpcChannels) -> InferenceControlMessage:
    raw_control = channels.ml_inbox.get(timeout=2.0)
    control = InferenceControlMessage.model_validate(raw_control)
    return control.model_copy(
        update={
            "forward_fields": {
                **control.forward_fields,
                "_creator_follow": True,
            }
        }
    )


def _process_one_analytics_message(
    db_path: Path,
    raw_analytics: object,
    outbox: CloudOutbox,
) -> None:
    processor = LocalAnalyticsProcessor(db_path, client_id=CLIENT_ID)
    shutdown = mp.get_context("spawn").Event()
    try:
        _run_loop(
            shutdown,
            cast("QueueLike", _OneMessageInbox(raw_analytics, shutdown)),
            processor,
            outbox,
        )
    finally:
        processor.close()


def test_headless_capture_to_cloud_replay_preserves_segment_identity_and_outbox_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert process_modules_for_mode("operator_api")["ui_api_shell"].endswith("operator_api_runtime")
    assert ProcessGraph(runtime_mode="operator_api").runtime_mode == "operator_api"

    stimulus = _stimulus_definition()
    db_path = _bootstrap_db(tmp_path, stimulus)
    capture_layout = CaptureLayout(
        capture_dir=tmp_path,
        audio_path=tmp_path / "audio_stream.wav",
        video_path=tmp_path / "video_stream.mkv",
    )
    audio_path = capture_layout.audio_path
    assert f"--record={audio_path}" in _build_audio_scrcpy_args("scrcpy", capture_layout)
    _write_audio_header(audio_path)
    channels = _channels()
    dispatcher = DesktopSegmentDispatcher(channels.ml_inbox, channels.pcm_acks)
    shutdown = mp.get_context("spawn").Event()
    drift = type("Drift", (), {"drift_offset": 0.125})()
    drift_state = type("DriftState", (), {"drift_corrector": drift})()
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
                session_id=SESSION_ID,
                stream_url="replay://capture-file",
                experiment_id="greeting_line_v1",
                active_arm="warm_welcome",
                stimulus_definition=stimulus,
                timestamp_utc=datetime.now(UTC),
            ).model_dump(mode="json")
        )
        _append_capture_audio(
            audio_path,
            module_c_orchestrator.AUDIO_SAMPLE_RATE_HZ
            * module_c_orchestrator.SEGMENT_WINDOW_SECONDS,
        )
        time.sleep(0.05)
        stimulus_time = (datetime.now(UTC) - timedelta(seconds=5)).timestamp()
        channels.segment_control.put(
            LiveSessionControlMessage(
                action="stimulus",
                session_id=SESSION_ID,
                stream_url="replay://capture-file",
                experiment_id="greeting_line_v1",
                active_arm="warm_welcome",
                stimulus_definition=stimulus,
                stimulus_time_s=stimulus_time,
                timestamp_utc=datetime.now(UTC),
            ).model_dump(mode="json")
        )
        _append_capture_audio(
            audio_path,
            module_c_orchestrator.AUDIO_SAMPLE_RATE_HZ
            * module_c_orchestrator.SEGMENT_WINDOW_SECONDS,
        )

        ml_control = _next_ml_control(channels)
    finally:
        shutdown.set()
        runner.join(timeout=2.0)

    try:
        handoff = InferenceHandoffPayload.model_validate(ml_control.handoff)
        shared_pcm = read_pcm_block(ml_control.audio.to_metadata())
        assert len(shared_pcm) == 16_000 * 2 * module_c_orchestrator.SEGMENT_WINDOW_SECONDS
        assert handoff.segment_id == ml_control.handoff["segment_id"]
        assert handoff.session_id == SESSION_ID
        assert handoff.stimulus_time == pytest.approx(stimulus_time)
        assert handoff.active_arm == "warm_welcome"
        assert ml_control.forward_fields["_creator_follow"] is True

        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr("packages.ml_core.audio_pipe.pcm_to_wav_bytes", lambda _audio: b"RIFF")
            patcher.setattr(
                "packages.ml_core.preprocessing.TextPreprocessor",
                _StubTextPreprocessor,
            )
            patcher.setattr("packages.ml_core.semantic.SemanticEvaluator", _StubSemanticEvaluator)
            _publish_analytics_result(
                channels,
                ml_control,
                None,
                transcription_engine=_StubTranscriptionEngine(),
            )

        assert channels.pcm_acks is not None
        ack = PcmBlockAckMessage.model_validate(channels.pcm_acks.get(timeout=1.0))
        assert ack.name == ml_control.audio.name
        channels.pcm_acks.put(ack.model_dump(mode="json"))
        dispatcher.release_acked_blocks()
        with pytest.raises(FileNotFoundError):
            read_pcm_block(ml_control.audio.to_metadata())

        raw_analytics = channels.analytics_inbox.get(timeout=1.0)
        analytics = AnalyticsResultMessage.model_validate(raw_analytics)
        assert analytics.handoff.segment_id == handoff.segment_id
        assert analytics.attribution is not None
        assert analytics.attribution.creator_follow is True

        outbox = CloudOutbox(db_path)
        try:
            _process_one_analytics_message(db_path, raw_analytics, outbox)
            segment_uploads = outbox.fetch_ready_batch(
                "telemetry_segments",
                limit=10,
                now_utc="2030-01-01T00:00:00Z",
            )
            delta_uploads = outbox.fetch_ready_batch(
                "telemetry_posterior_deltas",
                limit=10,
                now_utc="2030-01-01T00:00:00Z",
            )
        finally:
            outbox.close()

        assert [upload.payload_type for upload in segment_uploads] == [
            "inference_handoff",
            "attribution_event",
        ]
        assert [upload.payload_type for upload in delta_uploads] == ["posterior_delta"]

        handoff_payload = json.loads(segment_uploads[0].payload_json)
        attribution_payload = json.loads(segment_uploads[1].payload_json)
        delta_payload = json.loads(delta_uploads[0].payload_json)
        assert handoff_payload["segment_id"] == handoff.segment_id
        assert attribution_payload["segment_id"] == handoff.segment_id
        assert delta_payload["segment_id"] == handoff.segment_id
        assert attribution_payload["event_type"] == "stimulus_interaction"
        assert attribution_payload["stimulus_modality"] == stimulus.stimulus_modality
        assert attribution_payload["response_registration_status"] == "observable_response"

        conn = sqlite3.connect(str(db_path), isolation_level=None)
        try:
            ledger_row = conn.execute(
                "SELECT segment_id FROM analytics_message_ledger WHERE segment_id = ?",
                (handoff.segment_id,),
            ).fetchone()
            attribution_row = conn.execute(
                "SELECT segment_id FROM attribution_event WHERE segment_id = ?",
                (handoff.segment_id,),
            ).fetchone()
            pending_types = [
                str(row[0])
                for row in conn.execute(
                    "SELECT payload_type FROM pending_uploads ORDER BY created_at_utc, "
                    "CASE payload_type "
                    "WHEN 'inference_handoff' THEN 0 "
                    "WHEN 'attribution_event' THEN 1 "
                    "WHEN 'posterior_delta' THEN 2 ELSE 3 END, upload_id"
                ).fetchall()
            ]
        finally:
            conn.close()
        assert ledger_row == (handoff.segment_id,)
        assert attribution_row == (handoff.segment_id,)
        assert pending_types == ["inference_handoff", "attribution_event", "posterior_delta"]
    finally:
        dispatcher.close_inflight_blocks()
