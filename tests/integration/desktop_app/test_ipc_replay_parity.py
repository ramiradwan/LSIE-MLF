"""IPC replay parity and desktop queue lifecycle proof.

Drives the v4.0 IPC dispatch end-to-end on a single host and proves
byte-identical reconstruction of the ``InferenceHandoffPayload`` body
that crosses the orchestrator → gpu_ml_worker boundary. The producer
side wraps ``services.worker.pipeline.orchestrator.Orchestrator._dispatch_payload``;
the consumer side mirrors what ``services.desktop_app.processes.gpu_ml_worker.run``
does after a queue ``get`` — validate the control message, attach to the
SharedMemory block, copy audio, verify SHA-256.

The Gate 0 corpus lives at ``tests/fixtures/v4_gate0/segment_*.json``.
Each fixture's ``_au12_series`` and other fields drive a synthetic
``assemble_segment`` call; the test then asserts that the reconstructed
handoff body matches the fixture (modulo schema-prunable optionals)
and that the audio bytes survive the SharedMemory round trip with
deterministic SHA-256.

The desktop lifecycle test also proves live-session controls fan out to
both local IPC consumers, PCM blocks are ACKed and released, and typed
visual and analytics messages reach SQLite through the worker queue path.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from queue import Queue
from typing import Any, cast

import pytest

from packages.schemas.evaluation import StimulusDefinition
from packages.schemas.inference_handoff import InferenceHandoffPayload
from services.desktop_app.ipc import IpcChannels
from services.desktop_app.ipc.control_messages import (
    AnalyticsResultMessage,
    InferenceControlMessage,
    LiveSessionControlMessage,
    PcmBlockAckMessage,
    VisualAnalyticsStateMessage,
)
from services.desktop_app.ipc.shared_buffers import (
    PcmBlockMetadata,
    read_pcm_block,
)
from services.desktop_app.processes.analytics_state_worker import (
    LocalAnalyticsProcessor,
    QueueLike,
    _run_loop,
)
from services.desktop_app.processes.gpu_ml_worker import (
    LiveVisualTracker,
    _drain_live_control,
    _publish_analytics_result,
)
from services.desktop_app.processes.module_c_orchestrator import (
    DesktopSegment,
    DesktopSegmentDispatcher,
    _drain_segment_controls,
)
from services.desktop_app.processes.operator_api_runtime import _QueueLiveSessionControlPublisher
from services.desktop_app.state.sqlite_schema import bootstrap_schema
from services.worker.pipeline.orchestrator import Orchestrator

GATE0_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "v4_gate0"


def _fixtures() -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(GATE0_FIXTURE_DIR.glob("segment_*.json"))]


def _audio_for_fixture(fixture: dict[str, Any]) -> bytes:
    seed = str(fixture["segment_id"]).encode("utf-8")
    repeats = (960_000 // len(seed)) + 1
    return (seed * repeats)[:960_000]


def _fixture_stimulus_definition(fixture: dict[str, Any]) -> StimulusDefinition:
    return StimulusDefinition.model_validate(
        {
            "stimulus_modality": fixture["_stimulus_modality"],
            "stimulus_payload": fixture["_stimulus_payload"],
            "expected_stimulus_rule": fixture["_expected_stimulus_rule"],
            "expected_response_rule": fixture["_expected_response_rule"],
        }
    )


class _StubTranscriptionEngine:
    def transcribe(self, audio: object) -> str:
        del audio
        return "hello creator"


class _StubTextPreprocessor:
    def preprocess(self, text: str) -> str:
        return text


class _StubSemanticEvaluator:
    last_semantic_method = "cross_encoder"
    last_semantic_method_version = "integration-test-v1"

    def evaluate(self, expected_response_rule: str, actual_utterance: str) -> dict[str, Any]:
        del expected_response_rule, actual_utterance
        return {
            "reasoning": "cross_encoder_high_match",
            "is_match": True,
            "confidence_score": 0.91,
        }


class _NoopCapture:
    def start(self) -> None:
        return

    def stop(self) -> None:
        return


class _OneMessageInbox:
    def __init__(self, raw: object, shutdown_event: _ShutdownEvent) -> None:
        self._raw = raw
        self._shutdown_event = shutdown_event

    def get(self, block: bool = True, timeout: float | None = None) -> object:
        del block, timeout
        self._shutdown_event.set()
        return self._raw


class _ShutdownEvent:
    def __init__(self) -> None:
        self._is_set = False

    def is_set(self) -> bool:
        return self._is_set

    def set(self) -> None:
        self._is_set = True


class _Outbox:
    def __init__(self) -> None:
        self.enqueued: list[tuple[str, object]] = []

    def enqueue_inference_handoff(self, handoff: object) -> None:
        self.enqueued.append(("handoff", handoff))

    def enqueue_attribution_event(self, event: object) -> None:
        self.enqueued.append(("attribution_event", event))

    def enqueue_posterior_delta(self, delta: object) -> None:
        self.enqueued.append(("posterior_delta", delta))


@pytest.mark.parametrize(
    "fixture",
    _fixtures(),
    ids=[p.stem for p in sorted(GATE0_FIXTURE_DIR.glob("segment_*.json"))],
)
def test_ipc_dispatch_round_trips_fixture(fixture: dict[str, Any]) -> None:
    """Synthesize a payload from each Gate 0 fixture, push through IPC, recover.

    The audio bytes are a synthetic deterministic pattern derived from
    the segment_id so every replay is byte-identical without needing
    the captured ``audio.wav``.
    """
    audio = _audio_for_fixture(fixture)
    assert len(audio) == 960_000  # 30 s × 16 kHz × s16le mono

    # Drive the orchestrator's dispatch path with a thread-safe Queue;
    # the producer/consumer ends are the same process for parity.
    ipc_queue: Queue[Any] = Queue()
    orch = Orchestrator(session_id=fixture["session_id"], ipc_queue=ipc_queue)
    try:
        # Inject the fixture's pre-computed orchestrator state so
        # assemble_segment emits a payload with matching identity.
        stimulus_definition = _fixture_stimulus_definition(fixture)
        orch._segment_window_anchor_utc = None
        orch._active_arm = fixture["_active_arm"]
        orch._stimulus_definition = stimulus_definition
        orch._stimulus_time = fixture["_stimulus_time"]
        orch._au12_series = list(fixture["_au12_series"])
        orch._bandit_decision_snapshot = dict(fixture["_bandit_decision_snapshot"])

        payload = orch.assemble_segment(audio, [])

        orch._dispatch_payload(payload)

        assert ipc_queue.qsize() == 1
        raw = ipc_queue.get_nowait()
        msg = InferenceControlMessage.model_validate(raw)

        # Handoff body validates against §6.1.
        validated = InferenceHandoffPayload.model_validate(msg.handoff)
        assert (
            validated.bandit_decision_snapshot.stimulus_modality
            == fixture["_bandit_decision_snapshot"]["stimulus_modality"]
        )
        assert (
            validated.bandit_decision_snapshot.stimulus_payload.model_dump(mode="json")
            == fixture["_bandit_decision_snapshot"]["stimulus_payload"]
        )
        assert (
            validated.bandit_decision_snapshot.expected_stimulus_rule
            == fixture["_bandit_decision_snapshot"]["expected_stimulus_rule"]
        )
        assert (
            validated.bandit_decision_snapshot.expected_response_rule
            == fixture["_bandit_decision_snapshot"]["expected_response_rule"]
        )
        assert validated.stimulus_modality == fixture["_stimulus_modality"]
        assert validated.stimulus_payload.model_dump(mode="json") == fixture["_stimulus_payload"]
        assert validated.expected_stimulus_rule == fixture["_expected_stimulus_rule"]
        assert validated.expected_response_rule == fixture["_expected_response_rule"]
        assert validated.au12_series[0].timestamp_s == fixture["_au12_series"][0]["timestamp_s"]

        # Audio survives the SharedMemory round trip byte-identically.
        recovered = read_pcm_block(
            PcmBlockMetadata(
                name=msg.audio.name,
                byte_length=msg.audio.byte_length,
                sha256=msg.audio.sha256,
            )
        )
        assert recovered == audio
        assert msg.audio.byte_length == 960_000
        assert len(msg.audio.sha256) == 64

        # forward_fields carries _experiment_code only (no _audio_data,
        # no _frame_data — the desktop path drops the base64 round trip).
        assert "_audio_data" not in msg.forward_fields
        assert "_frame_data" not in msg.forward_fields
    finally:
        orch.close_inflight_blocks()


def test_process_graph_queues_route_pcm_ack_and_analytics_to_local_state(
    tmp_path: Path,
) -> None:
    fixture = _fixtures()[0]
    stimulus_definition = _fixture_stimulus_definition(fixture)
    stimulus_definition_dump = stimulus_definition.model_dump(mode="json")
    audio = _audio_for_fixture(fixture)
    channels = IpcChannels(
        ml_inbox=cast("Any", Queue()),
        drift_updates=cast("Any", Queue()),
        analytics_inbox=cast("Any", Queue()),
        pcm_acks=cast("Any", Queue()),
        live_control=cast("Any", Queue()),
        segment_control=cast("Any", Queue()),
    )
    dispatcher = DesktopSegmentDispatcher(channels.ml_inbox, channels.pcm_acks)
    db = tmp_path / "desktop.sqlite"
    conn = sqlite3.connect(str(db), isolation_level=None)
    try:
        bootstrap_schema(conn)
        experiment_row = conn.execute(
            "SELECT id FROM experiments WHERE experiment_id = ? AND arm = ?",
            ("greeting_line_v1", fixture["_active_arm"]),
        ).fetchone()
        assert experiment_row is not None
        conn.execute(
            "INSERT OR IGNORE INTO sessions ("
            "session_id, stream_url, experiment_id, active_arm, stimulus_definition, "
            "bandit_decision_snapshot"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            (
                fixture["session_id"],
                "replay://synthetic-fixture",
                "greeting_line_v1",
                fixture["_active_arm"],
                _fixture_stimulus_definition(fixture).model_dump_json(),
                json.dumps(fixture["_bandit_decision_snapshot"]),
            ),
        )
    finally:
        conn.close()

    try:
        control_timestamp = datetime.fromisoformat(
            str(fixture["segment_window_start_utc"]).replace("Z", "+00:00")
        ).astimezone(UTC)
        publisher = _QueueLiveSessionControlPublisher(channels)
        assert channels.live_control is not None
        assert channels.segment_control is not None
        live_control_queue = channels.live_control
        segment_control_queue = channels.segment_control

        live_payloads: list[object] = []
        segment_payloads: list[object] = []

        def publish_and_capture(control: LiveSessionControlMessage) -> None:
            expected = control.model_dump(mode="json")
            publisher.publish(control)
            raw_live_control = live_control_queue.get(timeout=1.0)
            raw_segment_control = segment_control_queue.get(timeout=1.0)
            assert raw_live_control == expected
            assert raw_segment_control == expected
            live_payloads.append(raw_live_control)
            segment_payloads.append(raw_segment_control)

        start_control = LiveSessionControlMessage(
            action="start",
            session_id=fixture["session_id"],
            stream_url="replay://synthetic-fixture",
            experiment_id="greeting_line_v1",
            active_arm=str(fixture["_active_arm"]),
            stimulus_definition=stimulus_definition,
            timestamp_utc=control_timestamp,
        )
        stimulus_control = LiveSessionControlMessage(
            action="stimulus",
            session_id=fixture["session_id"],
            experiment_id="greeting_line_v1",
            active_arm=str(fixture["_active_arm"]),
            stimulus_definition=stimulus_definition,
            stimulus_time_s=float(fixture["_stimulus_time"]),
            timestamp_utc=control_timestamp,
        )
        end_control = LiveSessionControlMessage(
            action="end",
            session_id=fixture["session_id"],
            timestamp_utc=control_timestamp,
        )
        publish_and_capture(start_control)
        publish_and_capture(stimulus_control)
        for raw_live_control in live_payloads:
            live_control_queue.put(raw_live_control)
        for raw_segment_control in segment_payloads:
            segment_control_queue.put(raw_segment_control)

        tracker = LiveVisualTracker(video_capture_factory=lambda _path: _NoopCapture())
        _drain_live_control(channels, tracker)
        raw_visuals = [channels.analytics_inbox.get(timeout=1.0) for _ in range(2)]
        visual = [VisualAnalyticsStateMessage.model_validate(raw) for raw in raw_visuals]
        assert [message.session_id for message in visual] == [start_control.session_id] * 2
        assert [message.active_arm for message in visual] == [fixture["_active_arm"]] * 2
        assert [
            message.stimulus_definition.model_dump(mode="json")
            if message.stimulus_definition is not None
            else None
            for message in visual
        ] == [stimulus_definition_dump] * 2
        assert tracker.stimulus_time_s == fixture["_stimulus_time"]

        active = _drain_segment_controls(channels, None, db_path=db)
        assert active is not None
        assert active.session_id == start_control.session_id
        assert active.active_arm == fixture["_active_arm"]
        assert active.stimulus_definition.model_dump(mode="json") == stimulus_definition_dump
        assert active.bandit_decision_snapshot == fixture["_bandit_decision_snapshot"]
        assert active.stimulus_time_s == fixture["_stimulus_time"]

        segment = DesktopSegment(
            session_id=active.session_id,
            stream_url=active.stream_url,
            segment_window_start_utc=control_timestamp,
            pcm_s16le_16khz_mono=audio,
            experiment_row_id=active.experiment_row_id,
            experiment_id=active.experiment_id,
            active_arm=active.active_arm,
            stimulus_definition=active.stimulus_definition,
            bandit_decision_snapshot=active.bandit_decision_snapshot,
            stimulus_time_s=active.stimulus_time_s,
        )
        assert dispatcher.dispatch(segment)
        raw_control = channels.ml_inbox.get(timeout=1.0)
        control = InferenceControlMessage.model_validate(raw_control)
        recovered = read_pcm_block(control.audio.to_metadata())
        assert recovered == audio

        with (
            pytest.MonkeyPatch.context() as monkeypatch,
        ):
            monkeypatch.setattr(
                "packages.ml_core.audio_pipe.pcm_to_wav_bytes",
                lambda _audio: b"RIFF",
            )
            monkeypatch.setattr(
                "packages.ml_core.preprocessing.TextPreprocessor",
                _StubTextPreprocessor,
            )
            monkeypatch.setattr(
                "packages.ml_core.semantic.SemanticEvaluator",
                _StubSemanticEvaluator,
            )
            _publish_analytics_result(
                channels,
                control,
                tracker,
                transcription_engine=_StubTranscriptionEngine(),
            )

        assert channels.pcm_acks is not None
        ack = PcmBlockAckMessage.model_validate(channels.pcm_acks.get(timeout=1.0))
        assert ack.name == control.audio.name
        channels.pcm_acks.put(ack.model_dump(mode="json"))
        dispatcher.release_acked_blocks()
        with pytest.raises(FileNotFoundError):
            read_pcm_block(control.audio.to_metadata())

        raw_analytics = channels.analytics_inbox.get(timeout=1.0)
        analytics = AnalyticsResultMessage.model_validate(raw_analytics)
        assert analytics.handoff.segment_id == control.handoff["segment_id"]
        assert analytics.transcription == "hello creator"
        assert analytics.semantic.is_match is True

        processor = LocalAnalyticsProcessor(db, client_id="desktop-e2e")
        outbox = _Outbox()
        try:
            for raw_visual in raw_visuals:
                visual_shutdown = _ShutdownEvent()
                _run_loop(
                    cast("Any", visual_shutdown),
                    cast("QueueLike", _OneMessageInbox(raw_visual, visual_shutdown)),
                    processor,
                    cast("Any", outbox),
                )
            analytics_shutdown = _ShutdownEvent()
            _run_loop(
                cast("Any", analytics_shutdown),
                cast("QueueLike", _OneMessageInbox(raw_analytics, analytics_shutdown)),
                processor,
                cast("Any", outbox),
            )
        finally:
            processor.close()

        conn = sqlite3.connect(str(db), isolation_level=None)
        try:
            live_row = conn.execute(
                "SELECT active_arm, stimulus_definition "
                "FROM live_session_state WHERE session_id = ?",
                (str(start_control.session_id),),
            ).fetchone()
            ledger_row = conn.execute(
                "SELECT segment_id FROM analytics_message_ledger WHERE segment_id = ?",
                (analytics.handoff.segment_id,),
            ).fetchone()
        finally:
            conn.close()
        assert live_row is not None
        assert live_row[0] == fixture["_active_arm"]
        assert json.loads(str(live_row[1])) == stimulus_definition_dump
        assert ledger_row is not None
        assert [kind for kind, _payload in outbox.enqueued] == ["handoff", "posterior_delta"]

        live_payloads.clear()
        segment_payloads.clear()
        publish_and_capture(end_control)
        live_control_queue.put(live_payloads[0])
        segment_control_queue.put(segment_payloads[0])
        _drain_live_control(channels, tracker)
        end_visual = VisualAnalyticsStateMessage.model_validate(
            channels.analytics_inbox.get(timeout=1.0)
        )
        assert end_visual.status == "no_session"
        assert tracker.session_id is None
        assert _drain_segment_controls(channels, active, db_path=db) is None
    finally:
        dispatcher.close_inflight_blocks()
