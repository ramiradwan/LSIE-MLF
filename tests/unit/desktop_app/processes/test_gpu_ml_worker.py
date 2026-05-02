from __future__ import annotations

import multiprocessing as mp
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

from services.desktop_app.ipc import IpcChannels
from services.desktop_app.ipc.control_messages import (
    AnalyticsResultMessage,
    AudioBlockRef,
    InferenceControlMessage,
)
from services.desktop_app.ipc.shared_buffers import write_pcm_block
from services.desktop_app.processes.gpu_ml_worker import _publish_analytics_result

SEGMENT_ID = "a" * 64
SESSION_ID = "00000000-0000-4000-8000-000000000001"
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


def _control_message() -> tuple[InferenceControlMessage, object]:
    block = write_pcm_block(b"\0\0" * 16000)
    msg = InferenceControlMessage(
        handoff=_handoff(),
        audio=AudioBlockRef.from_metadata(block.metadata),
    )
    return msg, block


def test_gpu_ml_worker_publishes_analytics_result_to_inbox() -> None:
    ctx = mp.get_context("spawn")
    channels = IpcChannels(
        ml_inbox=ctx.Queue(),
        drift_updates=ctx.Queue(),
        analytics_inbox=ctx.Queue(),
    )
    msg, block = _control_message()
    try:
        with (
            patch("packages.ml_core.audio_pipe.pcm_to_wav_bytes", return_value=b"RIFF"),
            patch("packages.ml_core.transcription.TranscriptionEngine", StubTranscriptionEngine),
            patch("packages.ml_core.preprocessing.TextPreprocessor", StubTextPreprocessor),
            patch("packages.ml_core.semantic.SemanticEvaluator", StubSemanticEvaluator),
        ):
            _publish_analytics_result(channels, msg)

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


def test_gpu_ml_worker_does_not_publish_without_semantic_result() -> None:
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
        )
        with (
            patch("packages.ml_core.audio_pipe.pcm_to_wav_bytes", return_value=b"RIFF"),
            patch("packages.ml_core.transcription.TranscriptionEngine", EmptyTranscriptionEngine),
        ):
            _publish_analytics_result(channels, msg)

        assert channels.analytics_inbox.empty()
    finally:
        block.close_and_unlink()
