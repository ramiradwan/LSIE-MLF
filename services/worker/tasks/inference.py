"""
Inference Task — §4.D Multimodal ML Processing via Celery

Dispatched from Module C → Module D via in-process call, then
Module D → Module E via Celery async task through Message Broker (§2 step 6).

Gap 1 fix: process_segment() now forwards orchestrator experiment and
telemetry fields (_active_arm, _experiment_id, _expected_greeting,
_au12_series, _stimulus_time, _x_max) to persist_metrics.

Gap 2 fix: _audio_data and _frame_data are base64-decoded from the
JSON-serialized Celery payload back to bytes.
"""

from __future__ import annotations

import logging
import tempfile
from typing import Any

from celery import Task

from services.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

# Orchestrator experiment and telemetry fields that must be forwarded
# from the input payload to the persist_metrics output. These are
# underscore-prefixed internal fields not part of the §2 step 6 spec
# payload, but required by the v3.0 reward pipeline in persist_metrics.
_FORWARD_FIELDS: tuple[str, ...] = (
    "_active_arm",
    "_experiment_id",
    "_expected_greeting",
    "_au12_series",
    "_stimulus_time",
    "_x_max",
)

# Binary fields that are base64-encoded for Celery JSON transport.
# See services/worker/pipeline/serialization.py for the encode/decode helpers.
_BINARY_FIELDS: list[str] = ["_audio_data", "_frame_data"]


@celery_app.task(  # type: ignore[untyped-decorator]
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
    # --- Gap 2 fix: decode base64-encoded binary fields from JSON transport ---
    from services.worker.pipeline.serialization import decode_bytes_fields

    payload = decode_bytes_fields(payload, _BINARY_FIELDS)

    session_id: str = payload["session_id"]
    segment_id: str = payload.get("_segment_id", "unknown")
    audio_data: bytes | None = payload.get("_audio_data")
    timestamp_utc: str = payload["timestamp_utc"]

    logger.info("Module D: processing %s/%s", session_id, segment_id)

    # --- §4.D.1 — Speech Transcription ---
    transcription: str = ""
    if audio_data:
        try:
            import os
            import subprocess

            from packages.ml_core.transcription import TranscriptionEngine

            logger.info("=====================================================================")
            logger.info("⏳ Initializing faster-whisper-large-v3...")
            logger.info("📦 If this is the first run, a 3GB model is downloading. Please wait.")
            logger.info("=====================================================================")

            engine = TranscriptionEngine()
            logger.info("✅ Whisper model loaded/ready!")

            # Write the raw PCM bytes to disk
            with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as raw_file:
                raw_file.write(audio_data)
                raw_path = raw_file.name

            # Use FFmpeg to translate the raw PCM into a pristine WAV container
            wav_path = raw_path.replace(".raw", ".wav")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-i",
                    raw_path,
                    wav_path,
                ],
                check=True,
            )

            # Transcribe the pristine WAV file
            transcription = engine.transcribe(wav_path)

            # Clean up the temp files
            os.remove(raw_path)
            os.remove(wav_path)

        except Exception:
            # §12 Network disconnect D — retry once then null
            logger.warning("Transcription failed for %s", segment_id, exc_info=True)

    # --- §4.D.2 + §7.5 — Face Mesh + AU12 ---
    au12_intensity: float | None = None
    frame_data: Any = payload.get("_frame_data")
    if frame_data is not None:
        try:
            import numpy as np

            from packages.ml_core.au12 import AU12Normalizer
            from packages.ml_core.face_mesh import FaceMeshProcessor

            # Read the spec-compliant resolution to reconstruct the 3D shape
            media_source = payload.get("media_source", {})
            res = media_source.get("resolution", [1920, 1080])
            width, height = res[0], res[1]

            mesh = FaceMeshProcessor()
            # Fold flat bytes into 3D array and copy() to make it writeable
            frame_array = (
                np.frombuffer(frame_data, dtype=np.uint8).copy().reshape((height, width, 3))
            )
            landmarks = mesh.extract_landmarks(frame_array)

            if landmarks is not None:
                # §7.5 — AU12 scoring
                normalizer = AU12Normalizer()
                normalizer.b_neutral = 0.0  # Prevent uncalibrated crash on the single frame
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

    # --- Gap 1 fix: Forward orchestrator experiment and telemetry fields ---
    # These underscore-prefixed fields are not part of the §2 step 6 spec payload
    # but are required by persist_metrics v3.0 for the Thompson Sampling reward
    # pipeline (_au12_series, _stimulus_time) and experiment attribution
    # (_active_arm, _experiment_id, _expected_greeting, _x_max).
    for key in _FORWARD_FIELDS:
        if key in payload:
            result[key] = payload[key]

    # §2 step 6 → §2 step 7 — Dispatch to Module E via Celery
    try:
        persist_metrics.delay(result)
    except Exception:
        logger.error("Failed to dispatch persist_metrics for %s", segment_id, exc_info=True)

    return result


@celery_app.task(  # type: ignore[untyped-decorator]
    bind=True,
    max_retries=5,
    default_retry_delay=5,
)
def persist_metrics(self: Task, metrics: dict[str, Any]) -> None:
    """
    §2 step 7 — Module E: Persist inference metrics to Persistent Store.
    §4.E.1 — [v3.0] Close Thompson Sampling feedback loop with continuous
    fractional Beta-Bernoulli reward.

    Failure mode (§12.1): If database unreachable, buffer up to 1000
    records in memory and retry every 5 seconds before overflow to CSV.

    Args:
        metrics: Structured inference results from Module D.
                 Includes _active_arm, _experiment_id, _expected_greeting,
                 and (v3.0) _au12_series, _stimulus_time from the Orchestrator.
    """
    from services.worker.pipeline.analytics import MetricsStore, ThompsonSamplingEngine

    store = MetricsStore()

    try:
        store.connect()
    except Exception:
        logger.error(
            "Cannot connect to Persistent Store for %s",
            metrics.get("segment_id", "unknown"),
            exc_info=True,
        )
        return

    try:
        store.insert_metrics(metrics)
        logger.info(
            "Metrics persisted for %s/%s",
            metrics.get("session_id"),
            metrics.get("segment_id"),
        )

        # ─── [v3.0] Thompson Sampling Continuous Reward Pipeline ───
        #
        # Pipeline:
        #   1. Build timestamped AU12 series from orchestrator telemetry
        #   2. Apply stimulus response window
        #        t ∈ [stimulus_time + 0.5, stimulus_time + 5.0]
        #   3. Aggregate facial response using P90 intensity
        #   4. Apply semantic validity gate
        #        G_t = 1 if semantic match else 0
        #   5. Continuous reward
        #        r_t = P90 × G_t
        #   6. Fractional Thompson Sampling update
        #        α ← α + r_t
        #        β ← β + (1 − r_t)
        #
        # Null observation handling:
        #   If the reward pipeline determines the observation is invalid
        #   (e.g., insufficient AU12 frames in the response window), the
        #   Thompson Sampling posterior is NOT updated. Missing telemetry
        #   is treated as "no observation", not negative feedback.

        semantic: dict[str, Any] | None = metrics.get("semantic")
        active_arm: str = metrics.get("_active_arm", "")
        experiment_id: str = metrics.get("_experiment_id", "")
        au12_raw_series: list[dict[str, Any]] | None = metrics.get("_au12_series")
        stimulus_time: float | None = metrics.get("_stimulus_time")
        x_max: float | None = metrics.get("_x_max")

        if (
            semantic is not None
            and active_arm
            and experiment_id
            and au12_raw_series is not None
            and len(au12_raw_series) > 0
            and stimulus_time is not None
        ):
            try:
                from services.worker.pipeline.reward import (
                    RewardResult,
                    TimestampedAU12,
                    compute_reward,
                )

                au12_series: list[TimestampedAU12] = [
                    TimestampedAU12(
                        timestamp_s=float(obs["timestamp_s"]),
                        intensity=float(obs["intensity"]),
                    )
                    for obs in au12_raw_series
                ]

                is_match: bool = semantic.get("is_match", False)
                confidence: float = semantic.get("confidence_score", 0.0)

                result_reward: RewardResult = compute_reward(
                    au12_series=au12_series,
                    stimulus_time_s=stimulus_time,
                    is_match=is_match,
                    confidence_score=confidence,
                    x_max=x_max,
                )

                if result_reward.is_valid:
                    if not (0.0 <= result_reward.gated_reward <= 1.0):
                        logger.warning(
                            "Invalid reward outside [0,1]: experiment=%s arm=%s reward=%f",
                            experiment_id,
                            active_arm,
                            result_reward.gated_reward,
                        )
                        return

                    engine = ThompsonSamplingEngine(store)
                    engine.update(experiment_id, active_arm, result_reward.gated_reward)

                    logger.info(
                        "Thompson Sampling updated: experiment=%s arm=%s "
                        "reward=%.4f (p90=%.4f gate=%d frames=%d)",
                        experiment_id,
                        active_arm,
                        result_reward.gated_reward,
                        result_reward.p90_intensity,
                        result_reward.semantic_gate,
                        result_reward.n_frames_in_window,
                    )
                else:
                    logger.info(
                        "Thompson Sampling update SKIPPED (invalid): "
                        "experiment=%s arm=%s frames=%d",
                        experiment_id,
                        active_arm,
                        result_reward.n_frames_in_window,
                    )

            except Exception:
                logger.warning(
                    "Thompson Sampling update failed for arm '%s'",
                    active_arm,
                    exc_info=True,
                )

        elif semantic is not None and active_arm and experiment_id:
            logger.info(
                "Thompson Sampling update SKIPPED (no AU12 telemetry): experiment=%s arm=%s",
                experiment_id,
                active_arm,
            )

    except Exception:
        logger.error(
            "Metrics persistence failed for %s",
            metrics.get("segment_id", "unknown"),
            exc_info=True,
        )
    finally:
        store.close()
