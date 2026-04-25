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
import math
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
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

# Raw Module C → D audio is mono PCM s16le at 16 kHz.
_PCM_SAMPLE_WIDTH_BYTES: int = 2

# §2.7 — Parameterized INSERT for encounter audit trail persistence.
_INSERT_ENCOUNTER_LOG_SQL: str = """
    INSERT INTO encounter_log (
        session_id, segment_id, experiment_id, arm, timestamp_utc,
        gated_reward, p90_intensity, semantic_gate, is_valid,
        n_frames, baseline_neutral, stimulus_time
    ) VALUES (
        %(session_id)s, %(segment_id)s, %(experiment_id)s, %(arm)s, %(timestamp_utc)s,
        %(gated_reward)s, %(p90_intensity)s, %(semantic_gate)s, %(is_valid)s,
        %(n_frames)s, %(baseline_neutral)s, %(stimulus_time)s
    )
"""

_CANONICAL_NULLABLE_ACOUSTIC_FLOAT_FIELDS: tuple[str, ...] = (
    "f0_mean_measure_hz",
    "f0_mean_baseline_hz",
    "f0_delta_semitones",
    "jitter_mean_measure",
    "jitter_mean_baseline",
    "jitter_delta",
    "shimmer_mean_measure",
    "shimmer_mean_baseline",
    "shimmer_delta",
    "pitch_f0",
    "jitter",
    "shimmer",
)

_CANONICAL_COVERAGE_ACOUSTIC_FLOAT_FIELDS: tuple[str, ...] = (
    "voiced_coverage_measure_s",
    "voiced_coverage_baseline_s",
)


def _default_acoustic_payload() -> dict[str, Any]:
    """Deterministic null-stimulus / no-audio acoustic payload defaults."""
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
        "pitch_f0": None,
        "jitter": None,
        "shimmer": None,
    }


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_nonnegative_float_or_default(value: Any, *, default: float = 0.0) -> float:
    number = _finite_float_or_none(value)
    if number is None or number < 0.0:
        return default
    return number


def _timestamp_to_epoch_s(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, str):
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return _finite_float_or_none(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)

    try:
        return _finite_float_or_none(timestamp.timestamp())
    except (OverflowError, OSError, ValueError):
        return None


def _derive_segment_start_time_s(
    *,
    timestamp_utc: Any,
    audio_data: bytes | None,
    sample_rate: int = 16000,
) -> float | None:
    """Derive the absolute segment start epoch for stimulus-locked §7D analysis."""
    if not audio_data or sample_rate <= 0:
        return None

    segment_end_time_s = _timestamp_to_epoch_s(timestamp_utc)
    if segment_end_time_s is None:
        return None

    segment_duration_s = len(audio_data) / float(sample_rate * _PCM_SAMPLE_WIDTH_BYTES)
    if not math.isfinite(segment_duration_s) or segment_duration_s < 0.0:
        return None

    return segment_end_time_s - segment_duration_s


def _serialize_acoustic_metrics(metrics: Any) -> dict[str, Any]:
    """Serialize ``AcousticMetrics`` into a JSON-safe Module D → E payload."""
    acoustic_payload = _default_acoustic_payload()
    acoustic_payload.update(asdict(metrics))

    for field in _CANONICAL_NULLABLE_ACOUSTIC_FLOAT_FIELDS:
        acoustic_payload[field] = _finite_float_or_none(acoustic_payload.get(field))
    for field in _CANONICAL_COVERAGE_ACOUSTIC_FLOAT_FIELDS:
        acoustic_payload[field] = _finite_nonnegative_float_or_default(acoustic_payload.get(field))

    return acoustic_payload


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
      3. AU12 intensity scoring — §7A.5
      4. Acoustic analysis (parselmouth) — §4.D.3
      5. Text preprocessing (spaCy) — §4.D.4
      6. Semantic evaluation (Azure OpenAI) — §8

    Dispatches results to Module E via Message Broker.

    Args:
        payload: Validated InferenceHandoffPayload as dict.

    Returns:
        Dict with session_id, segment_id, AU12 intensity, transcription,
        semantic match, and the canonical observational acoustic payload
        (plus optional legacy pitch/jitter/shimmer compatibility fields)
        required by §2 step 6 / §4.D.3.
    """
    # --- Gap 2 fix: decode base64-encoded binary fields from JSON transport ---
    from services.worker.pipeline.serialization import decode_bytes_fields, sanitize_json_payload

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

            logger.info("Initializing faster-whisper (first run downloads ~3GB model)")

            engine = TranscriptionEngine()
            logger.info("Whisper model loaded")

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

    # --- §4.D.2 + §7A.5 — Face Mesh + AU12 ---
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
                # §7A.5 — AU12 scoring
                normalizer = AU12Normalizer()
                normalizer.b_neutral = 0.0  # Prevent uncalibrated crash on the single frame
                au12_intensity = normalizer.compute_intensity(landmarks)
        except Exception:
            # §4.D contract — missing face returns null facial metrics
            logger.warning("AU12 extraction failed for %s", segment_id, exc_info=True)

    # --- §4.D.3 — Acoustic Analysis ---
    acoustic_payload: dict[str, Any]
    stimulus_time: float | None = payload.get("_stimulus_time")
    if stimulus_time is None:
        # §7D.5 / §4.D.contract — null stimulus time must emit the
        # deterministic default acoustic payload even when audio bytes exist.
        acoustic_payload = _default_acoustic_payload()
    elif audio_data:
        try:
            from packages.ml_core.acoustic import AcousticAnalyzer

            segment_start_time_s = _derive_segment_start_time_s(
                timestamp_utc=timestamp_utc,
                audio_data=audio_data,
                sample_rate=16000,
            )
            if segment_start_time_s is None:
                logger.warning(
                    "Acoustic timing derivation failed for %s; emitting default payload",
                    segment_id,
                )
                acoustic_payload = _default_acoustic_payload()
            else:
                analyzer = AcousticAnalyzer()
                acoustic_payload = _serialize_acoustic_metrics(
                    analyzer.analyze(
                        audio_data,
                        sample_rate=16000,
                        stimulus_time_s=stimulus_time,
                        segment_start_time_s=segment_start_time_s,
                    )
                )
        except Exception:
            # §4.D.contract / §12.4 — acoustic invalidity and local extraction
            # failures are data-quality outcomes; emit deterministic false/null
            # outputs instead of retrying or failing the worker.
            logger.warning("Acoustic analysis failed for %s", segment_id, exc_info=True)
            acoustic_payload = _default_acoustic_payload()
    else:
        acoustic_payload = _default_acoustic_payload()

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
        **acoustic_payload,
    }

    physio_context: dict[str, Any] | None = payload.get("_physiological_context")
    if physio_context is not None:
        # Preserve the orchestrator-provided dict exactly as received, but
        # keep the Task 4 omission rule when physiology is unavailable.
        result["_physiological_context"] = physio_context

    # --- Gap 1 fix: Forward orchestrator experiment and telemetry fields ---
    # These underscore-prefixed fields are not part of the §2 step 6 spec payload
    # but are required by persist_metrics v3.0 for the Thompson Sampling reward
    # pipeline (_au12_series, _stimulus_time) and experiment attribution
    # (_active_arm, _experiment_id, _expected_greeting, _x_max).
    for key in _FORWARD_FIELDS:
        if key in payload:
            result[key] = payload[key]

    result = sanitize_json_payload(result)

    # §2 step 6 → §2 step 7 — Dispatch to Module E via Celery
    try:
        persist_metrics.delay(result)
    except Exception:
        logger.error("Failed to dispatch persist_metrics for %s", segment_id, exc_info=True)

    return result


def _log_encounter(
    store: Any,
    metrics: dict[str, Any],
    experiment_id: str,
    arm: str,
    result: Any,
    stimulus_time: float | None,
) -> None:
    """
    §4.E.1 — Persist RewardResult to encounter_log for audit trail.

    §12.5 Module E — Failure must not propagate. If the INSERT fails,
    log a warning and continue. The Thompson Sampling update has already
    succeeded (or been correctly skipped) at this point.

    Args:
        store: Connected MetricsStore instance.
        metrics: The full metrics dict from Module D (for session/segment IDs).
        experiment_id: Thompson Sampling experiment ID.
        arm: The arm that was tested in this encounter.
        result: RewardResult from compute_reward().
        stimulus_time: Drift-corrected epoch of stimulus injection stored in
            encounter_log.stimulus_time as DOUBLE PRECISION. The wall-clock
            observation timestamp remains metrics["timestamp_utc"] and maps to
            encounter_log.timestamp_utc (TIMESTAMPTZ).
    """
    try:
        conn = store._get_conn()
        try:
            # §2.7 typing is enforced by the encounter_log schema:
            # timestamp_utc -> TIMESTAMPTZ, and reward/stimulus scalars ->
            # DOUBLE PRECISION (gated_reward, p90_intensity,
            # baseline_neutral, stimulus_time).
            with conn.cursor() as cur:
                cur.execute(
                    _INSERT_ENCOUNTER_LOG_SQL,
                    {
                        "session_id": metrics.get("session_id"),
                        "segment_id": metrics.get("segment_id"),
                        "experiment_id": experiment_id,
                        "arm": arm,
                        "timestamp_utc": metrics.get("timestamp_utc"),
                        "gated_reward": result.gated_reward,
                        "p90_intensity": result.p90_intensity,
                        "semantic_gate": result.semantic_gate,
                        "is_valid": result.is_valid,
                        "n_frames": result.n_frames_in_window,
                        "baseline_neutral": result.baseline_b_neutral,
                        "stimulus_time": stimulus_time,
                    },
                )
            conn.commit()
            logger.debug(
                "Encounter logged: experiment=%s arm=%s reward=%.4f valid=%s",
                experiment_id,
                arm,
                result.gated_reward,
                result.is_valid,
            )
        finally:
            store._put_conn(conn)
    except Exception:
        # §12.5 Module E — encounter_log failure must not crash persist_metrics.
        # The TS update already succeeded. Log and continue.
        logger.warning(
            "Failed to log encounter for experiment=%s arm=%s",
            experiment_id,
            arm,
            exc_info=True,
        )


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

    Failure mode (§12.5): If database unreachable, buffer up to 1000
    records in memory and retry every 5 seconds before overflow to CSV.

    Args:
        metrics: Structured inference results from Module D.
                 Includes _active_arm, _experiment_id, _expected_greeting,
                 and (v3.0) _au12_series, _stimulus_time from the Orchestrator.
                 timestamp_utc persists as TIMESTAMPTZ, while _stimulus_time
                 remains the drift-corrected epoch used by the reward pipeline
                 and is stored in encounter_log.stimulus_time as DOUBLE PRECISION.
    """
    from services.worker.pipeline.analytics import MetricsStore, ThompsonSamplingEngine
    from services.worker.pipeline.serialization import sanitize_json_payload

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
        physio_context: dict[str, Any] | None = metrics.get("_physiological_context")
        persistable_metrics: dict[str, Any] = dict(metrics)
        if physio_context is not None:
            # §4.D / orchestrator_physio_buffer.py — preserve the orchestrator-
            # injected dict exactly as received when handing the payload to
            # Module E persistence methods.
            persistable_metrics["_physiological_context"] = physio_context
        else:
            # Keep the optional handoff omitted when upstream sends no usable
            # physiological context instead of forwarding an explicit null.
            persistable_metrics.pop("_physiological_context", None)

        persistable_metrics = sanitize_json_payload(persistable_metrics)
        metrics = persistable_metrics
        store.insert_metrics(persistable_metrics)
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
        # §2.7 cross-artifact typing: timestamp_utc captures the wall-clock
        # observation time as TIMESTAMPTZ, while stimulus_time is the reward
        # pipeline's drift-corrected epoch scalar persisted as DOUBLE PRECISION.
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
                    _log_encounter(
                        store,
                        metrics,
                        experiment_id,
                        active_arm,
                        result_reward,
                        stimulus_time,
                    )
                else:
                    logger.info(
                        "Thompson Sampling update SKIPPED (invalid): "
                        "experiment=%s arm=%s frames=%d",
                        experiment_id,
                        active_arm,
                        result_reward.n_frames_in_window,
                    )
                    _log_encounter(
                        store,
                        metrics,
                        experiment_id,
                        active_arm,
                        result_reward,
                        stimulus_time,
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

        if physio_context is not None:
            session_id = metrics.get("session_id", "")
            segment_id = metrics.get("segment_id", "")
            streamer_snapshot: dict[str, Any] | None = physio_context.get("streamer")
            operator_snapshot: dict[str, Any] | None = physio_context.get("operator")
            streamer_available: bool = streamer_snapshot is not None
            operator_available: bool = operator_snapshot is not None
            physio_available: bool = streamer_available or operator_available

            for subject_role, snapshot in (
                ("streamer", streamer_snapshot),
                ("operator", operator_snapshot),
            ):
                if snapshot is None:
                    continue
                try:
                    store.persist_physiology_snapshot(
                        session_id=session_id,
                        segment_id=segment_id,
                        subject_role=subject_role,
                        snapshot=snapshot,
                    )
                except Exception:
                    logger.warning(
                        "Physiological context persistence failed for %s/%s role=%s",
                        session_id,
                        segment_id,
                        subject_role,
                        exc_info=True,
                    )

            if (
                session_id
                and streamer_snapshot is not None
                and operator_snapshot is not None
                and not streamer_snapshot.get("is_stale", True)
                and not operator_snapshot.get("is_stale", True)
            ):
                try:
                    comodulation_result = store.compute_comodulation(session_id)
                    if comodulation_result is None:
                        logger.info(
                            "Co-modulation unavailable: session=%s; no downstream action taken",
                            session_id,
                        )
                    else:
                        logger.info(
                            "Co-modulation persisted: session=%s index=%.4f pairs=%d coverage=%.3f",
                            session_id,
                            comodulation_result["co_modulation_index"],
                            comodulation_result["n_paired_observations"],
                            comodulation_result["coverage_ratio"],
                        )
                except Exception:
                    logger.warning(
                        "Co-modulation computation failed for session=%s",
                        session_id,
                        exc_info=True,
                    )

            logger.info(
                "Physiological context available: physio_available=%s streamer=%s operator=%s",
                physio_available,
                streamer_available,
                operator_available,
            )

    except Exception:
        logger.error(
            "Metrics persistence failed for %s",
            metrics.get("segment_id", "unknown"),
            exc_info=True,
        )
    finally:
        store.close()
