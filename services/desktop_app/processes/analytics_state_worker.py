"""Analytics + state worker process (v4.0 §4.E / WS3 P1 + WS4 P1b + P2).

Sole writer to the local SQLite store (WS4 P1) and the only process
that runs the §7B reward computation and the Beta-Bernoulli posterior
update. Wraps ``services.worker.pipeline.analytics.MetricsStore`` and
``ThompsonSamplingEngine``, plus
``services.worker.pipeline.reward.compute_reward`` — none of which
transitively touch torch / mediapipe / faster_whisper / ctranslate2.

WS4 P1b owns the SQLite writer's lifecycle here:

* Open and start the :class:`SqliteWriter` against the resolved
  app-data ``desktop.sqlite`` so the operator console reads pick up
  any rows the worker enqueues.
* Drain pending records on cooperative shutdown so the WAL flushes
  cleanly before process teardown.

WS4 P2 adds a per-process :class:`HeartbeatRecorder` so the next
ui_api_shell startup recovery sweep can spot a process that crashed
mid-flight and the operator console can render freshness.

WS5 P4 wires the analytics inbox: validated ML results are reduced
through the unchanged §7B reward function, applied to the local SQLite
posterior, and emitted as replay-safe cloud ``PosteriorDelta`` rows.

ML import discipline: this module MUST NOT import any of the four ML
library roots at any scope.
"""

from __future__ import annotations

import hashlib
import logging
import multiprocessing.synchronize as mpsync
import os
import queue
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from packages.schemas.cloud import PosteriorDelta
from services.desktop_app.cloud.outbox import CloudOutbox
from services.desktop_app.ipc import IpcChannels
from services.desktop_app.ipc.control_messages import AnalyticsResultMessage
from services.desktop_app.processes.cloud_sync_worker import DEFAULT_CLIENT_ID
from services.worker.pipeline.reward import RewardResult, TimestampedAU12, compute_reward

logger = logging.getLogger(__name__)

SQLITE_FILENAME = "desktop.sqlite"
INBOX_POLL_TIMEOUT_S = 0.5


class QueueLike(Protocol):
    def get(self, block: bool = True, timeout: float | None = None) -> object: ...


class LocalAnalyticsProcessor:
    def __init__(self, db_path: Path, *, client_id: str) -> None:
        self._db_path = db_path
        self._client_id = client_id

    def process(self, raw: object) -> PosteriorDelta | None:
        message = AnalyticsResultMessage.model_validate(raw)
        handoff = message.handoff
        if handoff.stimulus_time is None:
            return None
        is_match = message.semantic.is_match
        reward = compute_reward(
            au12_series=[
                TimestampedAU12(timestamp_s=obs.timestamp_s, intensity=obs.intensity)
                for obs in handoff.au12_series
            ],
            stimulus_time_s=handoff.stimulus_time,
            is_match=is_match,
        )
        applied_at_utc = datetime.now(UTC)
        delta = PosteriorDelta(
            experiment_id=handoff.experiment_id,
            arm_id=handoff.active_arm,
            delta_alpha=reward.gated_reward,
            delta_beta=1.0 - reward.gated_reward,
            segment_id=handoff.segment_id,
            client_id=self._client_id,
            event_id=_posterior_event_id(handoff.segment_id, self._client_id, handoff.active_arm),
            applied_at_utc=applied_at_utc,
            decision_context_hash=handoff.bandit_decision_snapshot.decision_context_hash,
        )
        with sqlite3.connect(str(self._db_path), isolation_level=None) as conn:
            conn.row_factory = sqlite3.Row
            applied = _apply_local_update(conn, message, reward, delta, applied_at_utc)
        return delta if applied else None


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
            delta = processor.process(raw)
            if delta is not None:
                outbox.enqueue_posterior_delta(delta)
        except Exception:  # noqa: BLE001
            logger.exception("analytics_state_worker discarded malformed analytics message")


def _apply_local_update(
    conn: sqlite3.Connection,
    message: AnalyticsResultMessage,
    reward: RewardResult,
    delta: PosteriorDelta,
    applied_at_utc: datetime,
) -> bool:
    handoff = message.handoff
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")
    try:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO analytics_message_ledger (
                message_id, segment_id, client_id, arm, processed_at_utc
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(message.message_id),
                handoff.segment_id,
                delta.client_id,
                handoff.active_arm,
                _iso_utc(applied_at_utc),
            ),
        )
        if cursor.rowcount == 0:
            conn.execute("COMMIT")
            return False
        row = conn.execute(
            """
            SELECT alpha_param, beta_param
            FROM experiments
            WHERE id = ? AND arm = ?
            """,
            (handoff.experiment_id, handoff.active_arm),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"Arm {handoff.active_arm!r} not found for experiment row {handoff.experiment_id}"
            )
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
                handoff.experiment_id,
                handoff.active_arm,
            ),
        )
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
        conn.execute("COMMIT")
        return True
    except Exception:
        conn.execute("ROLLBACK")
        raise


def _posterior_event_id(segment_id: str, client_id: str, arm_id: str) -> uuid.UUID:
    seed = f"{segment_id}:{client_id}:{arm_id}".encode()
    digest = bytearray(hashlib.sha256(seed).digest()[:16])
    digest[6] = (digest[6] & 0x0F) | 0x40
    digest[8] = (digest[8] & 0x3F) | 0x80
    return uuid.UUID(bytes=bytes(digest))


def _iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
