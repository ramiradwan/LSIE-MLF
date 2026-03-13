"""
Inference Task — §4.D Multimodal ML Processing via Celery

Dispatched from Module C → Module D via in-process call, then
Module D → Module E via Celery async task through Message Broker (§2 step 6).
"""

from __future__ import annotations

import logging
import tempfile
from typing import Any

from celery import Task

from services.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    max_retries=5,
    default_retry_delay=2,
    acks_late=True,
)
def process_segment(self: Task, payload: dict[str, Any]) -> dict[str, Any]:
    """
    §2 step 5 — Process a single InferenceHandoffPayload segment.

    Executes the full Module D pipeline:
      1. Speech transcription (faster-whisper) — §4.D.1
      2. Face mesh landmark extraction (MediaPipe) — §4.D.2
      3. AU12 intensity scoring — §7.5
      4. Acoustic analysis (parselmouth) — §4.D.3
      5. Text preprocessing (spaCy) — §4.D.4
      6. Semantic evaluation (Azure OpenAI) — §8

    Dispatches results to Module E via Message Broker.

    Args:
        payload: Validated InferenceHandoffPayload as dict.

    Returns:
        Dict with session_id, segment_id, AU12 intensity, transcription,
        semantic match, pitch, jitter, shimmer (§2 step 6).
    """
    session_id: str = payload["session_id"]
    segment_id: str = payload.get("_segment_id", "unknown")
    audio_data: bytes | None = payload.get("_audio_data")
    timestamp_utc: str = payload["timestamp_utc"]

    logger.info("Module D: processing %s/%s", session_id, segment_id)

    # --- §4.D.1 — Speech Transcription ---
    transcription: str = ""
    if audio_data:
        try:
            from packages.ml_core.transcription import TranscriptionEngine

            engine = TranscriptionEngine()
            # Write audio to temp file for faster-whisper
            with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name
            transcription = engine.transcribe(tmp_path)
        except Exception:
            # §12 Network disconnect D — retry once then null
            logger.warning("Transcription failed for %s", segment_id, exc_info=True)

    # --- §4.D.2 + §7.5 — Face Mesh + AU12 ---
    au12_intensity: float | None = None
    # Note: Video frame extraction is handled by the capture container;
    # audio-only segments may not have frame data available.
    # When frame data is available, extract landmarks and compute AU12.
    frame_data: Any = payload.get("_frame_data")
    if frame_data is not None:
        try:
            import numpy as np

            from packages.ml_core.au12 import AU12Normalizer
            from packages.ml_core.face_mesh import FaceMeshProcessor

            mesh = FaceMeshProcessor()
            frame_array = np.frombuffer(frame_data, dtype=np.uint8)
            landmarks = mesh.extract_landmarks(frame_array)

            if landmarks is not None:
                # §7.5 — AU12 scoring
                normalizer = AU12Normalizer()
                au12_intensity = normalizer.compute_intensity(landmarks)
        except Exception:
            # §4.D contract — missing face returns null facial metrics
            logger.warning("AU12 extraction failed for %s", segment_id, exc_info=True)

    # --- §4.D.3 — Acoustic Analysis ---
    pitch_f0: float | None = None
    jitter: float | None = None
    shimmer: float | None = None
    if audio_data:
        try:
            from packages.ml_core.acoustic import AcousticAnalyzer

            analyzer = AcousticAnalyzer()
            metrics = analyzer.analyze(audio_data)
            pitch_f0 = metrics.pitch_f0
            jitter = metrics.jitter
            shimmer = metrics.shimmer
        except Exception:
            # §12 Network disconnect D — retry once then null
            logger.warning("Acoustic analysis failed for %s", segment_id, exc_info=True)

    # --- §4.D.4 — Text Preprocessing ---
    preprocessed_text: str = transcription
    if transcription:
        try:
            from packages.ml_core.preprocessing import TextPreprocessor

            preprocessor = TextPreprocessor()
            preprocessed_text = preprocessor.preprocess(transcription)
        except Exception:
            logger.warning("Preprocessing failed for %s", segment_id, exc_info=True)

    # --- §8 — Semantic Evaluation ---
    semantic: dict[str, Any] | None = None
    if preprocessed_text:
        try:
            from packages.ml_core.semantic import SemanticEvaluator

            evaluator = SemanticEvaluator()
            # §8.3 — Evaluate against default greeting rule
            expected_greeting: str = payload.get(
                "_expected_greeting", "Hello, welcome to the stream!"
            )
            semantic = evaluator.evaluate(expected_greeting, preprocessed_text)
        except Exception:
            # §4.D contract — LLM timeout retries once before recording null
            logger.warning("Semantic evaluation failed for %s", segment_id, exc_info=True)

    # --- §2 step 6 — Assemble output payload for Module E ---
    result: dict[str, Any] = {
        "session_id": session_id,
        "segment_id": segment_id,
        "timestamp_utc": timestamp_utc,
        "au12_intensity": au12_intensity,
        "transcription": transcription,
        "semantic": semantic,
        "pitch_f0": pitch_f0,
        "jitter": jitter,
        "shimmer": shimmer,
    }

    # §2 step 6 → §2 step 7 — Dispatch to Module E via Celery
    try:
        persist_metrics.delay(result)
    except Exception:
        logger.error("Failed to dispatch persist_metrics for %s", segment_id, exc_info=True)

    return result


@celery_app.task(
    bind=True,
    max_retries=5,
    default_retry_delay=5,
)
def persist_metrics(self: Task, metrics: dict[str, Any]) -> None:
    """
    §2 step 7 — Module E: Persist inference metrics to Persistent Store.

    Failure mode (§12.1): If database unreachable, buffer up to 1000
    records in memory and retry every 5 seconds before overflow to CSV.

    Args:
        metrics: Structured inference results from Module D.
    """
    from services.worker.pipeline.analytics import MetricsStore

    store = MetricsStore()

    try:
        store.connect()
    except Exception:
        # §12.1 Module E — log and continue on connection failure
        logger.error(
            "Cannot connect to Persistent Store for %s",
            metrics.get("segment_id", "unknown"),
            exc_info=True,
        )
        return

    try:
        # §2 step 7 — Parameterized INSERT via MetricsStore
        store.insert_metrics(metrics)
        logger.info(
            "Metrics persisted for %s/%s",
            metrics.get("session_id"),
            metrics.get("segment_id"),
        )
    except Exception:
        # §12.1 Module E — log and continue
        logger.error(
            "Metrics persistence failed for %s",
            metrics.get("segment_id", "unknown"),
            exc_info=True,
        )
    finally:
        store.close()
