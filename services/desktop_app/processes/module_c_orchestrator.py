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
import struct
import threading
from collections import deque
from dataclasses import dataclass, replace
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
PCM_SAMPLE_RATE_HZ = 16_000
AUDIO_SAMPLE_WIDTH_BYTES = 2
PCM_CHANNELS = 1
AUDIO_CHUNK_SECONDS = 1.0
WAV_HEADER_BYTES = 44
SQLITE_FILENAME = "desktop.sqlite"
AUDIO_FILENAME = "audio_stream.wav"
SEGMENT_WINDOW_SECONDS = 30
MAX_AUDIO_BUFFER_SECONDS = SEGMENT_WINDOW_SECONDS * 2
DEFAULT_MAX_INFLIGHT_BLOCKS = 128
PCM_ACK_WAIT_TIMEOUT_S = 0.25
ML_INBOX_PUT_TIMEOUT_S = 0.5
STIMULUS_MEASUREMENT_WINDOW_END_OFFSET_S = 5.0
STIMULUS_BASELINE_WINDOW_START_OFFSET_S = -5.0


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


@dataclass(frozen=True)
class _CaptureAudioFormat:
    sample_rate_hz: int
    sample_width_bytes: int
    channels: int
    data_offset_bytes: int

    @property
    def frame_width_bytes(self) -> int:
        return self.sample_width_bytes * self.channels


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
            forward_fields={"_drift_offset_s": drift_offset_s},
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
        "_stimulus_time": segment.stimulus_time_s,
        "_au12_series": [],
        "_bandit_decision_snapshot": _bandit_snapshot(segment, start),
    }
    return InferenceHandoffPayload.model_validate(payload)


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
            JOIN experiments e
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
        experiment_row_id=int(row["experiment_row_id"]),
        active_arm=str(row["arm"] or "warm_welcome"),
        expected_greeting=str(row["greeting_text"] or "Hello, welcome to the stream!"),
    )


def _resolve_control_session(
    db_path: Path,
    control: LiveSessionControlMessage,
    active: _ActiveSession | None = None,
) -> _ActiveSession | None:
    experiment_id = control.experiment_id or (
        active.experiment_id if active is not None else "greeting_line_v1"
    )
    active_arm = control.active_arm or (active.active_arm if active is not None else None)
    conn = _reader_connection(db_path)
    try:
        row = conn.execute(
            """
            SELECT id, experiment_id, arm, greeting_text
            FROM experiments
            WHERE experiment_id = ?
              AND enabled = 1
              AND (? IS NULL OR arm = ?)
            ORDER BY CASE WHEN arm = ? THEN 0 ELSE 1 END, alpha_param DESC, arm ASC
            LIMIT 1
            """,
            (experiment_id, active_arm, active_arm, active_arm or ""),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        logger.error(
            "module_c_orchestrator could not resolve experiment row for %s/%s",
            experiment_id,
            active_arm or "<default>",
        )
        return active
    return _ActiveSession(
        session_id=control.session_id,
        stream_url=str(control.stream_url or (active.stream_url if active is not None else "")),
        experiment_id=str(row["experiment_id"]),
        experiment_row_id=int(row["id"]),
        active_arm=str(row["arm"]),
        expected_greeting=str(
            control.expected_greeting
            or (active.expected_greeting if active is not None else row["greeting_text"])
        ),
        stimulus_time_s=control.stimulus_time_s,
    )


def _drain_segment_controls(
    channels: IpcChannels,
    active: _ActiveSession | None,
    *,
    db_path: Path,
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
            active = _resolve_control_session(db_path, control)
            continue
        if active is not None and active.session_id == control.session_id:
            active = _resolve_control_session(db_path, control, active)
    return active


def _read_capture_audio_format(path: Path) -> _CaptureAudioFormat | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as audio_file:
            header = audio_file.read(12)
            if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                return None
            sample_rate_hz: int | None = None
            sample_width_bytes: int | None = None
            channels: int | None = None
            data_offset_bytes: int | None = None
            while True:
                chunk_header = audio_file.read(8)
                if len(chunk_header) < 8:
                    return None
                chunk_id = chunk_header[:4]
                chunk_size = struct.unpack("<I", chunk_header[4:])[0]
                chunk_data_offset = audio_file.tell()
                if chunk_id == b"fmt ":
                    fmt = audio_file.read(chunk_size)
                    if len(fmt) < 16:
                        return None
                    audio_format, channel_count, sample_rate, _, _, bits_per_sample = struct.unpack(
                        "<HHIIHH",
                        fmt[:16],
                    )
                    if audio_format != 1:
                        return None
                    channels = channel_count
                    sample_rate_hz = sample_rate
                    sample_width_bytes = bits_per_sample // 8
                elif chunk_id == b"data":
                    data_offset_bytes = chunk_data_offset
                    break
                else:
                    audio_file.seek(chunk_size, 1)
                if chunk_size % 2:
                    audio_file.seek(1, 1)
    except OSError:
        return None
    if (
        sample_rate_hz is None
        or sample_width_bytes is None
        or channels is None
        or data_offset_bytes is None
        or sample_rate_hz <= 0
        or sample_width_bytes <= 0
        or channels <= 0
    ):
        return None
    return _CaptureAudioFormat(
        sample_rate_hz=sample_rate_hz,
        sample_width_bytes=sample_width_bytes,
        channels=channels,
        data_offset_bytes=data_offset_bytes,
    )


def _read_new_capture_audio(
    path: Path,
    cursor_bytes: int,
    audio_format: _CaptureAudioFormat,
) -> tuple[bytes, int]:
    if not path.exists():
        return b"", cursor_bytes
    try:
        size = path.stat().st_size
        if size <= audio_format.data_offset_bytes:
            return b"", cursor_bytes
        if cursor_bytes < audio_format.data_offset_bytes or cursor_bytes > size:
            cursor_bytes = audio_format.data_offset_bytes
        if cursor_bytes == size:
            return b"", cursor_bytes
        with path.open("rb") as audio_file:
            audio_file.seek(cursor_bytes)
            raw = audio_file.read(size - cursor_bytes)
    except OSError:
        return b"", cursor_bytes
    usable = len(raw) - (len(raw) % audio_format.frame_width_bytes)
    if usable <= 0:
        return b"", cursor_bytes
    return raw[:usable], cursor_bytes + usable


def _source_pcm_to_16k_mono(raw: bytes, audio_format: _CaptureAudioFormat) -> bytes:
    if not raw:
        return b""
    mono = raw
    if audio_format.channels > 1:
        channel_weight = 1.0 / audio_format.channels
        mono = audioop.tomono(
            raw,
            audio_format.sample_width_bytes,
            channel_weight,
            channel_weight,
        )
    if audio_format.sample_width_bytes != AUDIO_SAMPLE_WIDTH_BYTES:
        mono = audioop.lin2lin(mono, audio_format.sample_width_bytes, AUDIO_SAMPLE_WIDTH_BYTES)
    converted, _ = audioop.ratecv(
        mono,
        AUDIO_SAMPLE_WIDTH_BYTES,
        PCM_CHANNELS,
        audio_format.sample_rate_hz,
        PCM_SAMPLE_RATE_HZ,
        None,
    )
    return bytes(converted)


def _stimulus_time_for_segment(
    active: _ActiveSession,
    *,
    segment_start_utc: datetime,
    drift_offset_s: float,
) -> float | None:
    if active.stimulus_time_s is None:
        return None
    segment_start_s = segment_start_utc.astimezone(UTC).timestamp() + drift_offset_s
    segment_end_s = segment_start_s + SEGMENT_WINDOW_SECONDS
    stimulus_baseline_start_s = (
        active.stimulus_time_s + drift_offset_s + STIMULUS_BASELINE_WINDOW_START_OFFSET_S
    )
    stimulus_window_end_s = (
        active.stimulus_time_s + drift_offset_s + STIMULUS_MEASUREMENT_WINDOW_END_OFFSET_S
    )
    if segment_start_s <= stimulus_baseline_start_s and segment_end_s >= stimulus_window_end_s:
        return active.stimulus_time_s
    return None


def _segment_audio_offset_bytes(
    *,
    segment_start_utc: datetime,
    target_start_utc: datetime,
) -> int:
    offset_s = (target_start_utc - segment_start_utc).total_seconds()
    offset_bytes = int(round(offset_s * PCM_SAMPLE_RATE_HZ * AUDIO_SAMPLE_WIDTH_BYTES))
    frame_bytes = AUDIO_SAMPLE_WIDTH_BYTES * PCM_CHANNELS
    return max(0, offset_bytes - (offset_bytes % frame_bytes))


def _stimulus_aligned_segment_start_utc(
    active: _ActiveSession,
    *,
    buffer_start_utc: datetime,
) -> datetime | None:
    if active.stimulus_time_s is None:
        return None
    desired_start_utc = datetime.fromtimestamp(
        active.stimulus_time_s + STIMULUS_MEASUREMENT_WINDOW_END_OFFSET_S - SEGMENT_WINDOW_SECONDS,
        tz=UTC,
    )
    return max(desired_start_utc, buffer_start_utc)


def _audio_duration_for_bytes(byte_count: int) -> timedelta:
    duration_s = byte_count / (PCM_SAMPLE_RATE_HZ * AUDIO_SAMPLE_WIDTH_BYTES * PCM_CHANNELS)
    return timedelta(seconds=duration_s)


def _trim_audio_buffer_to_latest_window(
    audio_buffer: bytearray,
    *,
    buffer_start_utc: datetime,
    max_bytes: int,
) -> datetime:
    if len(audio_buffer) <= max_bytes:
        return buffer_start_utc
    removed_bytes = len(audio_buffer) - max_bytes
    del audio_buffer[:removed_bytes]
    return buffer_start_utc + _audio_duration_for_bytes(removed_bytes)


def _segment_from_active_session(
    active: _ActiveSession,
    *,
    segment_start_utc: datetime,
    pcm_s16le_16khz_mono: bytes,
    drift_offset_s: float,
    stimulus_time_s: float | None = None,
) -> DesktopSegment:
    resolved_stimulus_time_s = stimulus_time_s
    if resolved_stimulus_time_s is None:
        resolved_stimulus_time_s = _stimulus_time_for_segment(
            active,
            segment_start_utc=segment_start_utc,
            drift_offset_s=drift_offset_s,
        )
    return DesktopSegment(
        session_id=active.session_id,
        stream_url=active.stream_url,
        segment_window_start_utc=segment_start_utc,
        pcm_s16le_16khz_mono=pcm_s16le_16khz_mono,
        experiment_row_id=active.experiment_row_id,
        experiment_id=active.experiment_id,
        active_arm=active.active_arm,
        expected_greeting=active.expected_greeting,
        stimulus_time_s=resolved_stimulus_time_s,
    )


def _dispatch_segment(
    dispatcher: DesktopSegmentDispatcher,
    segment: DesktopSegment,
    *,
    drift_offset_s: float,
    shutdown_event: mpsync.Event,
) -> bool:
    dispatched = dispatcher.dispatch(
        segment,
        drift_offset_s=drift_offset_s,
        shutdown_event=shutdown_event,
    )
    if not dispatched:
        if shutdown_event.is_set():
            return False
        logger.warning("module_c_orchestrator dropped undeliverable PCM segment")
    return True


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
    audio_format = _read_capture_audio_format(audio_path)
    cursor_bytes = audio_path.stat().st_size if active is not None and audio_path.exists() else 0
    audio_buffer = bytearray()
    segment_start_utc: datetime | None = None
    segment_bytes = SEGMENT_WINDOW_SECONDS * PCM_SAMPLE_RATE_HZ * AUDIO_SAMPLE_WIDTH_BYTES
    max_buffer_bytes = MAX_AUDIO_BUFFER_SECONDS * PCM_SAMPLE_RATE_HZ * AUDIO_SAMPLE_WIDTH_BYTES
    while not shutdown_event.is_set():
        dispatcher.release_acked_blocks()
        active = _drain_segment_controls(channels, active, db_path=db_path)
        if active is None:
            audio_buffer.clear()
            segment_start_utc = None
            audio_format = _read_capture_audio_format(audio_path)
            cursor_bytes = audio_path.stat().st_size if audio_path.exists() else 0
            shutdown_event.wait(timeout=SESSION_POLL_INTERVAL_S)
            continue
        if audio_format is None:
            audio_format = _read_capture_audio_format(audio_path)
            if audio_format is None:
                shutdown_event.wait(timeout=SESSION_POLL_INTERVAL_S)
                continue
        raw_source, cursor_bytes = _read_new_capture_audio(audio_path, cursor_bytes, audio_format)
        pcm_16k = _source_pcm_to_16k_mono(raw_source, audio_format)
        if pcm_16k:
            if segment_start_utc is None:
                segment_start_utc = datetime.now(UTC) - _audio_duration_for_bytes(len(pcm_16k))
            audio_buffer.extend(pcm_16k)
            segment_start_utc = _trim_audio_buffer_to_latest_window(
                audio_buffer,
                buffer_start_utc=segment_start_utc,
                max_bytes=max_buffer_bytes,
            )
        while active.stimulus_time_s is not None and segment_start_utc is not None:
            drift_offset_s = float(getattr(drift_state.drift_corrector, "drift_offset", 0.0))
            stimulus_segment_start_utc = _stimulus_aligned_segment_start_utc(
                active,
                buffer_start_utc=segment_start_utc,
            )
            if stimulus_segment_start_utc is not None:
                stimulus_offset_bytes = _segment_audio_offset_bytes(
                    segment_start_utc=segment_start_utc,
                    target_start_utc=stimulus_segment_start_utc,
                )
                if len(audio_buffer) < stimulus_offset_bytes + segment_bytes:
                    break
                segment = _segment_from_active_session(
                    active,
                    segment_start_utc=stimulus_segment_start_utc,
                    pcm_s16le_16khz_mono=bytes(
                        audio_buffer[stimulus_offset_bytes : stimulus_offset_bytes + segment_bytes]
                    ),
                    drift_offset_s=drift_offset_s,
                    stimulus_time_s=active.stimulus_time_s,
                )
                if not _dispatch_segment(
                    dispatcher,
                    segment,
                    drift_offset_s=drift_offset_s,
                    shutdown_event=shutdown_event,
                ):
                    return
                active = replace(active, stimulus_time_s=None)
                break
            break
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
