"""Analytics and local state worker for the desktop process graph.

This process is the sole writer to local SQLite and the only desktop
surface that applies the §7B reward computation and fractional
Beta-Bernoulli posterior update. It must stay free of the heavy ML
libraries reserved for ``gpu_ml_worker``.
"""

from __future__ import annotations

import logging
import multiprocessing.synchronize as mpsync
import os
import queue
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from packages.ml_core.attribution import build_attribution_ledger_records
from packages.schemas.attribution import AttributionEvent
from packages.schemas.cloud import PosteriorDelta
from packages.schemas.inference_handoff import InferenceHandoffPayload
from services.desktop_app.cloud.outbox import CloudOutbox
from services.desktop_app.ipc import IpcChannels
from services.desktop_app.ipc.control_messages import (
    AnalyticsAttributionInputs,
    AnalyticsResultMessage,
    VisualAnalyticsStateMessage,
)
from services.desktop_app.processes.cloud_sync_worker import DEFAULT_CLIENT_ID
from services.desktop_app.state.sqlite_schema import apply_writer_pragmas
from services.worker.pipeline.reward import RewardResult, TimestampedAU12, compute_reward

logger = logging.getLogger(__name__)

POSTERIOR_EVENT_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "urn:lsie-mlf:cloud:posterior-delta-event:v4",
)
SQLITE_FILENAME = "desktop.sqlite"
INBOX_POLL_TIMEOUT_S = 0.5

_DEFAULT_ACOUSTIC_PAYLOAD: dict[str, bool | float | None] = {
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


class QueueLike(Protocol):
    def get(self, block: bool = True, timeout: float | None = None) -> object: ...


# §9.2: keep cloud-bound artifacts together so enqueue order stays segment-first.
@dataclass(frozen=True)
class _CloudAnalyticsEnqueuePlan:
    handoff: InferenceHandoffPayload
    attribution_event: AttributionEvent | None
    posterior_delta: PosteriorDelta | None


class LocalAnalyticsProcessor:
    def __init__(self, db_path: Path, *, client_id: str) -> None:
        self._db_path = db_path
        self._client_id = client_id
        # One connection per processor: fresh sqlite3.connect on every call
        # gets default synchronous=FULL and busy_timeout=0 — each COMMIT
        # then double-fsyncs and the latency p95 floats up to ~13ms on
        # Windows. Reusing the connection with the writer pragma bundle
        # already applied keeps fsync semantics aligned with WAL+NORMAL
        # like the rest of the desktop runtime.
        self._conn = sqlite3.connect(str(db_path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        apply_writer_pragmas(self._conn)

    def process(self, raw: object) -> _CloudAnalyticsEnqueuePlan | None:
        message = AnalyticsResultMessage.model_validate(raw)
        handoff = message.handoff
        reward: RewardResult | None = None
        delta: PosteriorDelta | None = None
        applied_at_utc = datetime.now(UTC)

        if handoff.stimulus_time is not None:
            is_match = message.semantic.is_match
            # §7B: recompute the gated reward from the validated handoff payload.
            reward = compute_reward(
                au12_series=[
                    TimestampedAU12(timestamp_s=obs.timestamp_s, intensity=obs.intensity)
                    for obs in handoff.au12_series
                ],
                stimulus_time_s=handoff.stimulus_time,
                is_match=is_match,
            )
            delta = PosteriorDelta(
                experiment_id=handoff.experiment_id,
                arm_id=handoff.active_arm,
                delta_alpha=reward.gated_reward,
                delta_beta=1.0 - reward.gated_reward,
                segment_id=handoff.segment_id,
                client_id=self._client_id,
                event_id=_posterior_event_id(
                    handoff.segment_id,
                    self._client_id,
                    handoff.experiment_id,
                    handoff.active_arm,
                ),
                applied_at_utc=applied_at_utc,
                decision_context_hash=handoff.bandit_decision_snapshot.decision_context_hash,
            )

        applied = _apply_local_update(
            self._conn,
            message,
            client_id=self._client_id,
            applied_at_utc=applied_at_utc,
            reward=reward,
            delta=delta,
        )
        if not applied:
            return None
        # §6.1 / §6.4.1 / §7E / §9.2: hand off the validated segment payload,
        # the optional online-provisional AttributionEvent, and the optional
        # posterior delta together so the caller can preserve cloud ordering.
        return _CloudAnalyticsEnqueuePlan(
            handoff=message.handoff,
            attribution_event=_build_attribution_event(
                message,
                reward=reward,
                applied_at_utc=applied_at_utc,
            ),
            posterior_delta=delta,
        )

    def process_visual_state(self, raw: object) -> None:
        message = VisualAnalyticsStateMessage.model_validate(raw)
        _upsert_live_session_state(self._conn, message)

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            logger.debug("LocalAnalyticsProcessor close failed", exc_info=True)


def run(shutdown_event: mpsync.Event, channels: IpcChannels) -> None:
    logger.info("analytics_state_worker started")

    # Late imports preserve the ML-isolation canary contract and keep
    # the parent process free of SQLite handles.
    from services.desktop_app.os_adapter import resolve_state_dir
    from services.desktop_app.state.heartbeats import HeartbeatRecorder
    from services.desktop_app.state.sqlite_writer import SqliteWriter

    state_dir = resolve_state_dir()
    db_path = state_dir / SQLITE_FILENAME
    writer = SqliteWriter(db_path)
    writer.start()
    logger.info("sqlite writer opened at %s", db_path)

    heartbeat = HeartbeatRecorder(db_path, "analytics_state_worker")
    heartbeat.start()
    outbox = CloudOutbox(db_path)
    processor = LocalAnalyticsProcessor(
        db_path,
        client_id=os.environ.get("LSIE_CLOUD_CLIENT_ID", DEFAULT_CLIENT_ID),
    )
    analytics_inbox = getattr(channels, "analytics_inbox", None)

    try:
        if analytics_inbox is None:
            logger.warning("analytics_state_worker started without analytics_inbox")
            shutdown_event.wait()
            return
        _run_loop(shutdown_event, analytics_inbox, processor, outbox)
    finally:
        outbox.close()
        heartbeat.stop()
        writer.close()
        processor.close()
        logger.info("analytics_state_worker stopped")


def _run_loop(
    shutdown_event: mpsync.Event,
    analytics_inbox: QueueLike,
    processor: LocalAnalyticsProcessor,
    outbox: CloudOutbox,
) -> None:
    while not shutdown_event.is_set():
        try:
            raw = analytics_inbox.get(timeout=INBOX_POLL_TIMEOUT_S)
        except queue.Empty:
            continue
        try:
            schema_version = _schema_version(raw)
            if schema_version == "ws5.p4.visual_analytics_state.v1":
                processor.process_visual_state(raw)
                continue
            if schema_version != "ws5.p4.analytics_result.v1":
                raise ValueError(
                    f"Unsupported analytics message schema_version: {schema_version!r}"
                )
            enqueue_plan = processor.process(raw)
            if enqueue_plan is not None:
                # §9.2: the cloud path requires the segment payload before any
                # related attribution event or posterior delta reaches the outbox.
                outbox.enqueue_inference_handoff(enqueue_plan.handoff)
                if enqueue_plan.attribution_event is not None:
                    # §6.4.1 / §7E: live desktop sync emits only the
                    # online_provisional AttributionEvent between segment and delta.
                    outbox.enqueue_attribution_event(enqueue_plan.attribution_event)
                delta = enqueue_plan.posterior_delta
                if delta is not None:
                    outbox.enqueue_posterior_delta(delta)
        except Exception:  # noqa: BLE001
            logger.exception("analytics_state_worker discarded malformed analytics message")


def _apply_local_update(
    conn: sqlite3.Connection,
    message: AnalyticsResultMessage,
    *,
    client_id: str,
    applied_at_utc: datetime,
    reward: RewardResult | None,
    delta: PosteriorDelta | None,
) -> bool:
    handoff = message.handoff
    conn.execute("BEGIN IMMEDIATE")
    try:
        if _analytics_identity_exists(conn, handoff.segment_id, client_id, handoff.active_arm):
            conn.execute("COMMIT")
            return False
        _insert_metrics_row(conn, message)
        _insert_transcript_row(conn, message)
        _insert_evaluation_row(conn, message)
        if reward is not None and delta is not None:
            _apply_reward_update(
                conn,
                handoff.experiment_id,
                handoff.active_arm,
                delta,
                applied_at_utc,
            )
            _insert_encounter_row(conn, message, reward)
        conn.execute(
            """
            INSERT INTO analytics_message_ledger (
                message_id, segment_id, client_id, arm, processed_at_utc
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(message.message_id),
                handoff.segment_id,
                client_id,
                handoff.active_arm,
                _iso_utc(applied_at_utc),
            ),
        )
        conn.execute("COMMIT")
        return True
    except Exception:
        conn.execute("ROLLBACK")
        raise


def _analytics_identity_exists(
    conn: sqlite3.Connection,
    segment_id: str,
    client_id: str,
    arm: str,
) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM analytics_message_ledger
        WHERE segment_id = ? AND client_id = ? AND arm = ?
        LIMIT 1
        """,
        (segment_id, client_id, arm),
    ).fetchone()
    return row is not None


def _upsert_live_session_state(
    conn: sqlite3.Connection,
    message: VisualAnalyticsStateMessage,
) -> None:
    conn.execute(
        """
        INSERT INTO live_session_state (
            session_id, active_arm, expected_greeting, is_calibrating,
            calibration_frames_accumulated, calibration_frames_required,
            face_present, latest_au12_intensity, latest_au12_timestamp_s,
            status, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            active_arm = excluded.active_arm,
            expected_greeting = excluded.expected_greeting,
            is_calibrating = excluded.is_calibrating,
            calibration_frames_accumulated = excluded.calibration_frames_accumulated,
            calibration_frames_required = excluded.calibration_frames_required,
            face_present = excluded.face_present,
            latest_au12_intensity = excluded.latest_au12_intensity,
            latest_au12_timestamp_s = excluded.latest_au12_timestamp_s,
            status = excluded.status,
            updated_at_utc = excluded.updated_at_utc
        """,
        (
            str(message.session_id),
            message.active_arm,
            message.expected_greeting,
            int(message.is_calibrating),
            message.calibration_frames_accumulated,
            message.calibration_frames_required,
            int(message.face_present),
            message.latest_au12_intensity,
            message.latest_au12_timestamp_s,
            message.status,
            _iso_utc(message.timestamp_utc),
        ),
    )


def _schema_version(raw: object) -> str | None:
    # §6.1: the worker accepts only the typed analytics and visual-state
    # schemas on its inbox, so the dispatcher normalizes schema_version first.
    if isinstance(raw, dict):
        value = raw.get("schema_version")
    else:
        value = getattr(raw, "schema_version", None)
    return value if isinstance(value, str) else None


# §6.4.1 / §7E: desktop live upload emits only the online-provisional
# AttributionEvent surface; offline finalization remains a separate deferred path.
def _build_attribution_event(
    message: AnalyticsResultMessage,
    *,
    reward: RewardResult | None,
    applied_at_utc: datetime,
) -> AttributionEvent | None:
    attribution = message.attribution
    if attribution is None or not _has_attribution_inputs(attribution):
        return None
    metrics: dict[str, Any] = {
        "session_id": str(message.handoff.session_id),
        "segment_id": message.handoff.segment_id,
        "timestamp_utc": message.handoff.timestamp_utc,
        "_active_arm": message.handoff.active_arm,
        "_expected_greeting": message.handoff.expected_greeting,
        "_stimulus_time": message.handoff.stimulus_time,
        "_au12_series": [
            observation.model_dump(mode="python") for observation in message.handoff.au12_series
        ],
        "_bandit_decision_snapshot": message.handoff.bandit_decision_snapshot.model_dump(
            mode="python",
            by_alias=True,
        ),
        "semantic": message.semantic.model_dump(mode="python", by_alias=True),
    }
    if attribution.outcome_event is not None:
        metrics["_outcome_event"] = attribution.outcome_event
    if attribution.outcome_events is not None:
        metrics["_outcome_events"] = attribution.outcome_events
    if attribution.creator_follow is not None:
        metrics["_creator_follow"] = attribution.creator_follow
    ledger = build_attribution_ledger_records(
        metrics,
        reward_result=reward,
        created_at=applied_at_utc,
    )
    return None if ledger is None else ledger.event


def _has_attribution_inputs(attribution: AnalyticsAttributionInputs) -> bool:
    return any(
        value is not None and value != []
        for value in (
            attribution.outcome_event,
            attribution.outcome_events,
            attribution.creator_follow,
        )
    )


def _metrics_record(message: AnalyticsResultMessage) -> dict[str, Any]:
    handoff = message.handoff
    acoustic = (
        message.acoustic.model_dump(mode="python")
        if message.acoustic is not None
        else dict(_DEFAULT_ACOUSTIC_PAYLOAD)
    )
    return {
        "session_id": str(handoff.session_id),
        "segment_id": handoff.segment_id,
        "timestamp_utc": _iso_utc(handoff.timestamp_utc),
        "au12_intensity": None,
        **dict(_DEFAULT_ACOUSTIC_PAYLOAD),
        **acoustic,
    }


def _insert_metrics_row(conn: sqlite3.Connection, message: AnalyticsResultMessage) -> None:
    metrics = _metrics_record(message)
    conn.execute(
        """
        INSERT INTO metrics (
            session_id, segment_id, timestamp_utc,
            au12_intensity,
            f0_valid_measure, f0_valid_baseline,
            perturbation_valid_measure, perturbation_valid_baseline,
            voiced_coverage_measure_s, voiced_coverage_baseline_s,
            f0_mean_measure_hz, f0_mean_baseline_hz, f0_delta_semitones,
            jitter_mean_measure, jitter_mean_baseline, jitter_delta,
            shimmer_mean_measure, shimmer_mean_baseline, shimmer_delta
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            metrics["session_id"],
            metrics["segment_id"],
            metrics["timestamp_utc"],
            metrics["au12_intensity"],
            metrics["f0_valid_measure"],
            metrics["f0_valid_baseline"],
            metrics["perturbation_valid_measure"],
            metrics["perturbation_valid_baseline"],
            metrics["voiced_coverage_measure_s"],
            metrics["voiced_coverage_baseline_s"],
            metrics["f0_mean_measure_hz"],
            metrics["f0_mean_baseline_hz"],
            metrics["f0_delta_semitones"],
            metrics["jitter_mean_measure"],
            metrics["jitter_mean_baseline"],
            metrics["jitter_delta"],
            metrics["shimmer_mean_measure"],
            metrics["shimmer_mean_baseline"],
            metrics["shimmer_delta"],
        ),
    )


def _insert_transcript_row(conn: sqlite3.Connection, message: AnalyticsResultMessage) -> None:
    if not message.transcription:
        return
    handoff = message.handoff
    conn.execute(
        """
        INSERT INTO transcripts (session_id, segment_id, timestamp_utc, text)
        VALUES (?, ?, ?, ?)
        """,
        (
            str(handoff.session_id),
            handoff.segment_id,
            _iso_utc(handoff.timestamp_utc),
            message.transcription,
        ),
    )


def _insert_evaluation_row(conn: sqlite3.Connection, message: AnalyticsResultMessage) -> None:
    handoff = message.handoff
    conn.execute(
        """
        INSERT INTO evaluations (
            session_id, segment_id, timestamp_utc, reasoning, is_match, confidence
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            str(handoff.session_id),
            handoff.segment_id,
            _iso_utc(handoff.timestamp_utc),
            message.semantic.reasoning,
            message.semantic.is_match,
            message.semantic.confidence_score,
        ),
    )


def _apply_reward_update(
    conn: sqlite3.Connection,
    experiment_id: int,
    arm: str,
    delta: PosteriorDelta,
    applied_at_utc: datetime,
) -> None:
    row = conn.execute(
        """
        SELECT alpha_param, beta_param
        FROM experiments
        WHERE id = ? AND arm = ?
        """,
        (experiment_id, arm),
    ).fetchone()
    if row is None:
        raise ValueError(f"Arm {arm!r} not found for experiment row {experiment_id}")
    conn.execute(
        """
        UPDATE experiments
        SET alpha_param = ?, beta_param = ?, updated_at = ?
        WHERE id = ? AND arm = ?
        """,
        (
            float(row["alpha_param"]) + delta.delta_alpha,
            float(row["beta_param"]) + delta.delta_beta,
            _iso_utc(applied_at_utc),
            experiment_id,
            arm,
        ),
    )


def _insert_encounter_row(
    conn: sqlite3.Connection,
    message: AnalyticsResultMessage,
    reward: RewardResult,
) -> None:
    handoff = message.handoff
    conn.execute(
        """
        INSERT INTO encounter_log (
            session_id, segment_id, experiment_id, arm, timestamp_utc,
            gated_reward, p90_intensity, semantic_gate, n_frames_in_window,
            au12_baseline_pre, stimulus_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(handoff.session_id),
            handoff.segment_id,
            str(handoff.experiment_id),
            handoff.active_arm,
            _iso_utc(handoff.timestamp_utc),
            reward.gated_reward,
            reward.p90_intensity,
            reward.semantic_gate,
            reward.n_frames_in_window,
            reward.au12_baseline_pre,
            handoff.stimulus_time,
        ),
    )


def _posterior_event_id(
    segment_id: str,
    client_id: str,
    experiment_id: int,
    arm_id: str,
) -> uuid.UUID:
    return uuid.uuid5(
        POSTERIOR_EVENT_NAMESPACE,
        f"{client_id}:{segment_id}:{experiment_id}:{arm_id}",
    )


def _iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
