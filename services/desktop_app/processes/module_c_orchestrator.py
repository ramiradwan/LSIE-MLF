"""Module C desktop shell for drift updates and safe segment dispatch.

This process owns desktop-safe handoff construction and shared-memory PCM
transport while keeping heavy ML libraries isolated to ``gpu_ml_worker``.
"""

from __future__ import annotations

import audioop
import hashlib
import logging
import multiprocessing.synchronize as mpsync
import queue
import sqlite3
import threading
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from pydantic import ValidationError

from packages.schemas.inference_handoff import InferenceHandoffPayload
from services.desktop_app.ipc import IpcChannels
from services.desktop_app.ipc.control_messages import (
    AudioBlockRef,
    InferenceControlMessage,
    LiveSessionControlMessage,
    PcmBlockAckMessage,
)
from services.desktop_app.ipc.shared_buffers import PcmBlock, write_pcm_block
from services.desktop_app.state.sqlite_schema import apply_reader_pragmas, bootstrap_schema

logger = logging.getLogger(__name__)

DRIFT_DRAIN_POLL_TIMEOUT_S = 0.5
SESSION_POLL_INTERVAL_S = 0.25
AUDIO_SAMPLE_RATE_HZ = 48_000
AUDIO_SAMPLE_WIDTH_BYTES = 2
AUDIO_CHANNELS = 1
AUDIO_CHUNK_SECONDS = 1.0
WAV_HEADER_BYTES = 44
SQLITE_FILENAME = "desktop.sqlite"
AUDIO_FILENAME = "audio_stream.wav"
SEGMENT_WINDOW_SECONDS = 30
DEFAULT_MAX_INFLIGHT_BLOCKS = 128
PCM_ACK_WAIT_TIMEOUT_S = 0.25
ML_INBOX_PUT_TIMEOUT_S = 0.5


@dataclass(frozen=True)
class DesktopSegment:
    session_id: UUID
    stream_url: str
    segment_window_start_utc: datetime
    pcm_s16le_16khz_mono: bytes
    experiment_row_id: int
    experiment_id: str
    active_arm: str
    expected_greeting: str
    stimulus_time_s: float | None


@dataclass(frozen=True)
class _ActiveSession:
    session_id: UUID
    stream_url: str
    experiment_id: str
    experiment_row_id: int
    active_arm: str
    expected_greeting: str
    stimulus_time_s: float | None = None


class _DriftState(Protocol):
    drift_corrector: Any


class DesktopSegmentDispatcher:
    def __init__(
        self,
        ml_inbox: Any,
        pcm_acks: Any | None = None,
        *,
        max_inflight_blocks: int = DEFAULT_MAX_INFLIGHT_BLOCKS,
    ) -> None:
        self._ml_inbox = ml_inbox
        self._pcm_acks = pcm_acks
        self._max_inflight_blocks = max_inflight_blocks
        self._inflight_blocks: dict[str, PcmBlock] = {}
        self._inflight_order: deque[str] = deque()

    def dispatch(
        self,
        segment: DesktopSegment,
        *,
        drift_offset_s: float = 0.0,
        shutdown_event: mpsync.Event | None = None,
    ) -> bool:
        if not self.wait_for_capacity(shutdown_event):
            if shutdown_event is None or not shutdown_event.is_set():
                logger.warning("module_c_orchestrator backpressured waiting for PCM ack capacity")
            return False
        try:
            handoff = _build_handoff(segment, drift_offset_s=drift_offset_s)
        except ValidationError:
            logger.exception("module_c_orchestrator dropped invalid inference handoff payload")
            return False
        block = write_pcm_block(segment.pcm_s16le_16khz_mono)
        self._track_block(block)
        message = InferenceControlMessage(
            handoff=handoff.model_dump(mode="json", by_alias=True),
            audio=AudioBlockRef.from_metadata(block.metadata),
        )
        if not self._enqueue_message(message.model_dump(mode="json"), shutdown_event):
            self._release_block(block.metadata.name)
            if shutdown_event is None or not shutdown_event.is_set():
                logger.warning("module_c_orchestrator skipped segment because ml_inbox is full")
            return False
        return True

    def _track_block(self, block: PcmBlock) -> None:
        name = block.metadata.name
        self._inflight_blocks[name] = block
        self._inflight_order.append(name)

    def _release_block(self, name: str) -> None:
        block = self._inflight_blocks.pop(name, None)
        if block is not None:
            block.close_and_unlink()
        self._prune_inflight_order()

    def _prune_inflight_order(self) -> None:
        while self._inflight_order and self._inflight_order[0] not in self._inflight_blocks:
            self._inflight_order.popleft()

    def _handle_pcm_ack(self, raw: object) -> None:
        try:
            ack = PcmBlockAckMessage.model_validate(raw)
        except Exception:  # noqa: BLE001
            logger.exception("module_c_orchestrator discarded malformed PCM ack message")
            return
        self._release_block(ack.name)

    def release_acked_blocks(self) -> None:
        if self._pcm_acks is None:
            return
        while True:
            try:
                raw = self._pcm_acks.get_nowait()
            except queue.Empty:
                return
            self._handle_pcm_ack(raw)

    def wait_for_capacity(self, shutdown_event: mpsync.Event | None = None) -> bool:
        self.release_acked_blocks()
        while len(self._inflight_blocks) >= self._max_inflight_blocks:
            if shutdown_event is None:
                return False
            if shutdown_event.is_set() or self._pcm_acks is None:
                return False
            try:
                raw = self._pcm_acks.get(timeout=PCM_ACK_WAIT_TIMEOUT_S)
            except queue.Empty:
                continue
            self._handle_pcm_ack(raw)
        return True

    def _enqueue_message(
        self,
        payload: dict[str, Any],
        shutdown_event: mpsync.Event | None = None,
    ) -> bool:
        while True:
            try:
                self._ml_inbox.put(payload, timeout=ML_INBOX_PUT_TIMEOUT_S)
                return True
            except queue.Full:
                self.release_acked_blocks()
                if shutdown_event is not None and shutdown_event.is_set():
                    return False

    def close_inflight_blocks(self) -> None:
        while self._inflight_order:
            name = self._inflight_order.popleft()
            block = self._inflight_blocks.pop(name, None)
            if block is not None:
                block.close_and_unlink()


def _build_handoff(segment: DesktopSegment, *, drift_offset_s: float) -> InferenceHandoffPayload:
    start = segment.segment_window_start_utc.astimezone(UTC)
    end = start + timedelta(seconds=SEGMENT_WINDOW_SECONDS)
    payload = {
        "session_id": str(segment.session_id),
        "segment_id": _segment_id(segment),
        "segment_window_start_utc": start,
        "segment_window_end_utc": end,
        "timestamp_utc": end,
        "media_source": {
            "stream_url": segment.stream_url,
            "codec": "h264",
            "resolution": [1, 1],
        },
        "segments": [],
        "_active_arm": segment.active_arm,
        "_experiment_id": segment.experiment_row_id,
        "_expected_greeting": segment.expected_greeting,
        "_stimulus_time": _correct_stimulus_time(segment.stimulus_time_s, drift_offset_s),
        "_au12_series": [],
        "_bandit_decision_snapshot": _bandit_snapshot(segment, start),
    }
    return InferenceHandoffPayload.model_validate(payload)


def _correct_stimulus_time(stimulus_time_s: float | None, drift_offset_s: float) -> float | None:
    if stimulus_time_s is None:
        return None
    return stimulus_time_s + drift_offset_s


def _segment_id(segment: DesktopSegment) -> str:
    start = segment.segment_window_start_utc.astimezone(UTC).isoformat()
    digest_input = (
        f"{segment.session_id}:{start}:{segment.experiment_row_id}:"
        f"{segment.active_arm}:{len(segment.pcm_s16le_16khz_mono)}"
    )
    return hashlib.sha256(digest_input.encode()).hexdigest()


def _bandit_snapshot(segment: DesktopSegment, selection_time_utc: datetime) -> dict[str, Any]:
    decision_context_hash = hashlib.sha256(
        f"{segment.session_id}:{segment.experiment_id}:{segment.active_arm}".encode()
    ).hexdigest()
    return {
        "selection_method": "thompson_sampling",
        "selection_time_utc": selection_time_utc,
        "experiment_id": segment.experiment_row_id,
        "policy_version": "desktop_replay_v1",
        "selected_arm_id": segment.active_arm,
        "candidate_arm_ids": [segment.active_arm],
        "posterior_by_arm": {segment.active_arm: {"alpha": 1.0, "beta": 1.0}},
        "sampled_theta_by_arm": {segment.active_arm: 0.5},
        "expected_greeting": segment.expected_greeting,
        "decision_context_hash": decision_context_hash,
        "random_seed": 0,
    }


def _reader_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    bootstrap_schema(conn)
    apply_reader_pragmas(conn)
    return conn


def _fetch_active_session(db_path: Path) -> _ActiveSession | None:
    conn = _reader_connection(db_path)
    try:
        row = conn.execute(
            """
            SELECT s.session_id, s.stream_url, s.experiment_id,
                   e.id AS experiment_row_id, e.arm, e.greeting_text
            FROM sessions s
            LEFT JOIN experiments e
                ON e.experiment_id = s.experiment_id
               AND e.enabled = 1
            WHERE s.ended_at IS NULL
            ORDER BY s.started_at DESC, e.alpha_param DESC, e.arm ASC
            LIMIT 1
            """
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return _ActiveSession(
        session_id=UUID(str(row["session_id"])),
        stream_url=str(row["stream_url"]),
        experiment_id=str(row["experiment_id"] or "greeting_line_v1"),
        experiment_row_id=int(row["experiment_row_id"] or 0),
        active_arm=str(row["arm"] or "warm_welcome"),
        expected_greeting=str(row["greeting_text"] or "Hello, welcome to the stream!"),
    )


def _drain_segment_controls(
    channels: IpcChannels,
    active: _ActiveSession | None,
) -> _ActiveSession | None:
    control_queue = channels.segment_control
    if control_queue is None:
        return active
    while True:
        try:
            raw = control_queue.get_nowait()
        except queue.Empty:
            return active
        try:
            control = LiveSessionControlMessage.model_validate(raw)
        except Exception:  # noqa: BLE001
            logger.exception("module_c_orchestrator discarded malformed live control message")
            continue
        if control.action == "end":
            if active is not None and active.session_id == control.session_id:
                active = None
            continue
        if control.action == "start":
            active = _ActiveSession(
                session_id=control.session_id,
                stream_url=str(control.stream_url),
                experiment_id=control.experiment_id or "greeting_line_v1",
                experiment_row_id=0,
                active_arm=control.active_arm or "warm_welcome",
                expected_greeting=control.expected_greeting or "Hello, welcome to the stream!",
            )
            continue
        if active is not None and active.session_id == control.session_id:
            active = _ActiveSession(
                session_id=active.session_id,
                stream_url=control.stream_url or active.stream_url,
                experiment_id=control.experiment_id or active.experiment_id,
                experiment_row_id=active.experiment_row_id,
                active_arm=control.active_arm or active.active_arm,
                expected_greeting=control.expected_greeting or active.expected_greeting,
                stimulus_time_s=control.stimulus_time_s,
            )
    return active


def _read_new_capture_audio(path: Path, cursor_bytes: int) -> tuple[bytes, int]:
    if not path.exists():
        return b"", cursor_bytes
    try:
        size = path.stat().st_size
        if size <= WAV_HEADER_BYTES:
            return b"", cursor_bytes
        if cursor_bytes < WAV_HEADER_BYTES or cursor_bytes > size:
            cursor_bytes = WAV_HEADER_BYTES
        if cursor_bytes == size:
            return b"", cursor_bytes
        with path.open("rb") as audio_file:
            audio_file.seek(cursor_bytes)
            raw = audio_file.read(size - cursor_bytes)
    except OSError:
        return b"", cursor_bytes
    usable = len(raw) - (len(raw) % (AUDIO_SAMPLE_WIDTH_BYTES * AUDIO_CHANNELS))
    if usable <= 0:
        return b"", cursor_bytes
    return raw[:usable], cursor_bytes + usable


def _pcm_48k_to_16k(raw: bytes) -> bytes:
    if not raw:
        return b""
    converted, _ = audioop.ratecv(
        raw,
        AUDIO_SAMPLE_WIDTH_BYTES,
        AUDIO_CHANNELS,
        AUDIO_SAMPLE_RATE_HZ,
        16_000,
        None,
    )
    return bytes(converted)


def _segment_from_active_session(
    active: _ActiveSession,
    *,
    segment_start_utc: datetime,
    pcm_s16le_16khz_mono: bytes,
) -> DesktopSegment:
    return DesktopSegment(
        session_id=active.session_id,
        stream_url=active.stream_url,
        segment_window_start_utc=segment_start_utc,
        pcm_s16le_16khz_mono=pcm_s16le_16khz_mono,
        experiment_row_id=active.experiment_row_id,
        experiment_id=active.experiment_id,
        active_arm=active.active_arm,
        expected_greeting=active.expected_greeting,
        stimulus_time_s=active.stimulus_time_s,
    )


def _run_segment_loop(
    shutdown_event: mpsync.Event,
    channels: IpcChannels,
    dispatcher: DesktopSegmentDispatcher,
    drift_state: _DriftState,
    *,
    db_path: Path,
    audio_path: Path,
) -> None:
    active = _fetch_active_session(db_path)
    cursor_bytes = audio_path.stat().st_size if active is not None and audio_path.exists() else 0
    audio_buffer = bytearray()
    segment_start_utc: datetime | None = None
    segment_bytes = SEGMENT_WINDOW_SECONDS * 16_000 * AUDIO_SAMPLE_WIDTH_BYTES
    while not shutdown_event.is_set():
        dispatcher.release_acked_blocks()
        active = _drain_segment_controls(channels, active)
        if active is None:
            audio_buffer.clear()
            segment_start_utc = None
            cursor_bytes = audio_path.stat().st_size if audio_path.exists() else 0
            shutdown_event.wait(timeout=SESSION_POLL_INTERVAL_S)
            continue
        raw_48k, cursor_bytes = _read_new_capture_audio(audio_path, cursor_bytes)
        pcm_16k = _pcm_48k_to_16k(raw_48k)
        if pcm_16k:
            if segment_start_utc is None:
                segment_start_utc = datetime.now(UTC)
            audio_buffer.extend(pcm_16k)
        while len(audio_buffer) >= segment_bytes and segment_start_utc is not None:
            segment_audio = bytes(audio_buffer[:segment_bytes])
            dispatched = dispatcher.dispatch(
                _segment_from_active_session(
                    active,
                    segment_start_utc=segment_start_utc,
                    pcm_s16le_16khz_mono=segment_audio,
                ),
                drift_offset_s=float(getattr(drift_state.drift_corrector, "drift_offset", 0.0)),
                shutdown_event=shutdown_event,
            )
            if not dispatched:
                if shutdown_event.is_set():
                    return
                logger.warning("module_c_orchestrator dropped undeliverable PCM segment")
            del audio_buffer[:segment_bytes]
            segment_start_utc = segment_start_utc + timedelta(seconds=SEGMENT_WINDOW_SECONDS)
        shutdown_event.wait(timeout=SESSION_POLL_INTERVAL_S)


def _drain_drift_updates(
    channels: IpcChannels,
    orchestrator: object,
    shutdown_event: mpsync.Event,
) -> None:
    """Apply each ``drift_updates`` payload to the orchestrator's corrector.

    The supervisor pushes ``{"drift_offset": float, ...}`` dicts on
    every poll cycle (~30 s). We update the orchestrator's drift
    corrector inline; the orchestrator's apply-side
    ``correct_timestamp`` calls then pick up the new offset on the
    next read.
    """
    corrector = getattr(orchestrator, "drift_corrector", None)
    if corrector is None:
        logger.error("orchestrator has no drift_corrector — drift updates will be dropped")
        return

    while not shutdown_event.is_set():
        try:
            payload = channels.drift_updates.get(timeout=DRIFT_DRAIN_POLL_TIMEOUT_S)
        except queue.Empty:
            continue
        if not isinstance(payload, dict):
            continue
        offset = payload.get("drift_offset")
        if isinstance(offset, int | float):
            corrector.drift_offset = float(offset)


def run(shutdown_event: mpsync.Event, channels: IpcChannels) -> None:
    logger.info("module_c_orchestrator started")

    # Late imports: keeps the parent's import-isolation canary clean.
    from services.desktop_app.os_adapter import resolve_capture_dir, resolve_state_dir
    from services.desktop_app.state.heartbeats import HeartbeatRecorder
    from services.worker.pipeline.orchestrator import DriftCorrector

    orchestrator = DesktopSegmentDispatcher(channels.ml_inbox, channels.pcm_acks)
    drift_state = type("DesktopDriftState", (), {"drift_corrector": DriftCorrector()})()

    drift_thread = threading.Thread(
        target=_drain_drift_updates,
        args=(channels, drift_state, shutdown_event),
        name="module-c-drift-drain",
        daemon=True,
    )
    drift_thread.start()

    state_dir = resolve_state_dir()
    heartbeat = HeartbeatRecorder(state_dir / SQLITE_FILENAME, "module_c_orchestrator")
    heartbeat.start()
    capture_dir = resolve_capture_dir()

    try:
        _run_segment_loop(
            shutdown_event,
            channels,
            orchestrator,
            drift_state,
            db_path=state_dir / SQLITE_FILENAME,
            audio_path=capture_dir / AUDIO_FILENAME,
        )
    finally:
        heartbeat.stop()
        try:
            orchestrator.close_inflight_blocks()
        except Exception:  # noqa: BLE001
            logger.debug("inflight cleanup failed", exc_info=True)
        drift_thread.join(timeout=5.0)
        logger.info("module_c_orchestrator stopped")
