"""GPU ML worker process.

This is the **only** module in the desktop graph that may import
``torch``, ``mediapipe``, ``faster_whisper``, or ``ctranslate2``. The
parent process never imports this module — it is launched by string
through :func:`services.desktop_app.process_graph._launch`, so the ML
libraries are pulled into a dedicated child process and never into the
UI / API / orchestrator / state / cloud-sync surfaces.

The ML imports stay at module top level so the isolation canary can
prove they remain confined to this process. ``run`` stays intentionally
narrow around that boundary.
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import multiprocessing.synchronize as mpsync
import queue
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

# The four ML library imports below are intentional at module top
# level: they prove the v4.0 §9 isolation contract. The canary test
# imports each non-ML process module in a clean subprocess and asserts
# these names are absent from ``sys.modules``; re-importing this
# module DOES bring them in — that is the whole point of routing ML
# inference into a dedicated child process.
import ctranslate2  # noqa: F401
import faster_whisper  # noqa: F401
import mediapipe  # noqa: F401
import torch  # noqa: F401

from services.desktop_app.ipc import IpcChannels
from services.desktop_app.ipc.control_messages import (
    AnalyticsResultMessage,
    InferenceControlMessage,
)
from services.desktop_app.ipc.shared_buffers import read_pcm_block

logger = logging.getLogger(__name__)

INBOX_POLL_TIMEOUT_S = 0.5
SQLITE_FILENAME = "desktop.sqlite"
_PCM_SAMPLE_WIDTH_BYTES = 2


def _default_acoustic_payload() -> dict[str, Any]:
    return {
        "f0_valid_measure": False,
        "f0_valid_baseline": False,
        "perturbation_valid_measure": False,
        "perturbation_valid_baseline": False,
        "voiced_coverage_measure_s": 0.0,
        "voiced_coverage_baseline_s": 0.0,
        "f0_mean_measure_hz": None,
        "f0_mean_baseline_hz": None,
        "f0_delta_semitones": None,
        "jitter_mean_measure": None,
        "jitter_mean_baseline": None,
        "jitter_delta": None,
        "shimmer_mean_measure": None,
        "shimmer_mean_baseline": None,
        "shimmer_delta": None,
    }


def _finite_float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite_nonnegative_float_or_default(value: Any, default: float = 0.0) -> float:
    result = _finite_float_or_none(value)
    if result is None or result < 0.0:
        return default
    return result


def _serialize_acoustic_metrics(metrics: Any) -> dict[str, Any]:
    acoustic_payload = _default_acoustic_payload()
    acoustic_payload.update(asdict(metrics))
    for field in (
        "f0_mean_measure_hz",
        "f0_mean_baseline_hz",
        "f0_delta_semitones",
        "jitter_mean_measure",
        "jitter_mean_baseline",
        "jitter_delta",
        "shimmer_mean_measure",
        "shimmer_mean_baseline",
        "shimmer_delta",
    ):
        acoustic_payload[field] = _finite_float_or_none(acoustic_payload.get(field))
    for field in ("voiced_coverage_measure_s", "voiced_coverage_baseline_s"):
        acoustic_payload[field] = _finite_nonnegative_float_or_default(acoustic_payload.get(field))
    return acoustic_payload


def _derive_segment_start_time_s(
    *,
    timestamp_utc: str,
    audio_data: bytes,
    sample_rate: int,
) -> float | None:
    try:
        segment_end_time_s = (
            datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00")).astimezone(UTC).timestamp()
        )
    except ValueError:
        return None
    if sample_rate <= 0 or len(audio_data) < _PCM_SAMPLE_WIDTH_BYTES:
        return None
    segment_duration_s = len(audio_data) / (_PCM_SAMPLE_WIDTH_BYTES * sample_rate)
    return segment_end_time_s - segment_duration_s


def _normalize_semantic_result(
    semantic: dict[str, Any] | None,
    *,
    semantic_method: str | None = None,
    semantic_method_version: str | None = None,
) -> dict[str, Any] | None:
    if semantic is None:
        return None
    from packages.schemas.evaluation import SEMANTIC_METHODS, SEMANTIC_REASON_CODES

    method = semantic_method or semantic.get("semantic_method") or "cross_encoder"
    if method not in SEMANTIC_METHODS:
        method = "cross_encoder"
    reasoning = semantic.get("reasoning")
    if reasoning not in SEMANTIC_REASON_CODES:
        reasoning = "semantic_error"
    confidence = _finite_float_or_none(semantic.get("confidence_score"))
    if confidence is None:
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))
    return {
        "reasoning": reasoning,
        "is_match": bool(semantic.get("is_match", False)),
        "confidence_score": confidence,
        "semantic_method": method,
        "semantic_method_version": semantic_method_version
        or semantic.get("semantic_method_version")
        or "desktop-gpu-worker-v1",
    }


def _analytics_message_id(segment_id: str) -> uuid.UUID:
    digest = bytearray(hashlib.sha256(f"analytics-result:{segment_id}".encode()).digest()[:16])
    digest[6] = (digest[6] & 0x0F) | 0x40
    digest[8] = (digest[8] & 0x3F) | 0x80
    return uuid.UUID(bytes=bytes(digest))


def _build_analytics_result(msg: InferenceControlMessage) -> AnalyticsResultMessage | None:
    audio = read_pcm_block(msg.audio.to_metadata())
    handoff = msg.handoff
    segment_id = str(handoff.get("segment_id", "unknown"))
    timestamp_utc = str(handoff["timestamp_utc"])
    stimulus_time = handoff.get("_stimulus_time")

    transcription = ""
    try:
        from packages.ml_core.audio_pipe import pcm_to_wav_bytes
        from packages.ml_core.transcription import TranscriptionEngine

        transcription = TranscriptionEngine().transcribe(io.BytesIO(pcm_to_wav_bytes(audio)))
    except Exception:
        logger.warning("Transcription failed for %s", segment_id, exc_info=True)

    acoustic_payload: dict[str, Any]
    if stimulus_time is None:
        acoustic_payload = _default_acoustic_payload()
    else:
        try:
            from packages.ml_core.acoustic import AcousticAnalyzer

            segment_start_time_s = _derive_segment_start_time_s(
                timestamp_utc=timestamp_utc,
                audio_data=audio,
                sample_rate=16000,
            )
            acoustic_payload = (
                _default_acoustic_payload()
                if segment_start_time_s is None
                else _serialize_acoustic_metrics(
                    AcousticAnalyzer().analyze(
                        audio,
                        sample_rate=16000,
                        stimulus_time_s=float(stimulus_time),
                        segment_start_time_s=segment_start_time_s,
                    )
                )
            )
        except Exception:
            logger.warning("Acoustic analysis failed for %s", segment_id, exc_info=True)
            acoustic_payload = _default_acoustic_payload()

    if not transcription:
        return None

    semantic: dict[str, Any] | None = None
    try:
        from packages.ml_core.preprocessing import TextPreprocessor
        from packages.ml_core.semantic import SemanticEvaluator

        preprocessed_text = TextPreprocessor().preprocess(transcription)
        evaluator = SemanticEvaluator()
        live_semantic = evaluator.evaluate(
            str(handoff.get("_expected_greeting", "Hello, welcome to the stream!")),
            preprocessed_text,
        )
        semantic = _normalize_semantic_result(
            live_semantic,
            semantic_method=getattr(evaluator, "last_semantic_method", None),
            semantic_method_version=getattr(evaluator, "last_semantic_method_version", None),
        )
    except Exception:
        logger.warning("Semantic evaluation failed for %s", segment_id, exc_info=True)

    if semantic is None:
        return None

    return AnalyticsResultMessage.model_validate(
        {
            "message_id": str(_analytics_message_id(segment_id)),
            "handoff": handoff,
            "semantic": semantic,
            "transcription": transcription,
            "acoustic": acoustic_payload,
        }
    )


def _publish_analytics_result(channels: IpcChannels, msg: InferenceControlMessage) -> None:
    analytics = _build_analytics_result(msg)
    if analytics is None:
        return
    channels.analytics_inbox.put(analytics.model_dump(mode="json"))


def run(shutdown_event: mpsync.Event, channels: IpcChannels) -> None:
    logger.info("gpu_ml_worker started")

    from services.desktop_app.os_adapter import resolve_state_dir
    from services.desktop_app.state.heartbeats import HeartbeatRecorder

    state_dir = resolve_state_dir()
    heartbeat = HeartbeatRecorder(state_dir / SQLITE_FILENAME, "gpu_ml_worker")
    heartbeat.start()

    try:
        while not shutdown_event.is_set():
            try:
                raw = channels.ml_inbox.get(timeout=INBOX_POLL_TIMEOUT_S)
            except queue.Empty:
                continue
            try:
                msg = InferenceControlMessage.model_validate(raw)
            except Exception:  # noqa: BLE001
                logger.exception("gpu_ml_worker discarded malformed control message")
                continue
            logger.info(
                "gpu_ml_worker received segment_id=%s audio=%s/%d bytes",
                msg.handoff.get("segment_id", "?"),
                msg.audio.name,
                msg.audio.byte_length,
            )
            try:
                _publish_analytics_result(channels, msg)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "gpu_ml_worker failed to publish analytics result for segment_id=%s",
                    msg.handoff.get("segment_id", "?"),
                )
    finally:
        heartbeat.stop()
        logger.info("gpu_ml_worker stopped")
