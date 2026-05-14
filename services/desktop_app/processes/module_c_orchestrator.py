"""Module C desktop shell for drift updates and safe segment dispatch.

This process owns desktop-safe handoff construction and shared-memory PCM
transport while keeping heavy ML libraries isolated to ``gpu_ml_worker``.
"""

from __future__ import annotations

import audioop
import hashlib
import json
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

from packages.schemas.evaluation import StimulusDefinition
from packages.schemas.inference_handoff import InferenceHandoffPayload
from packages.schemas.physiology import PhysiologicalContext
from services.desktop_app.ipc import IpcChannels
from services.desktop_app.ipc.control_messages import (
    DESKTOP_LIVE_VISUAL_SOURCE_CONTRACT,
    AudioBlockRef,
    InferenceControlMessage,
    LiveSessionControlMessage,
    PcmBlockAckMessage,
    VisualAnalyticsStateMessage,
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
VISUAL_AU12_BUFFER_MAXLEN = 1800


@dataclass(frozen=True)
class DesktopSegment:
    session_id: UUID
    stream_url: str
    segment_window_start_utc: datetime
    pcm_s16le_16khz_mono: bytes
    experiment_row_id: int
    experiment_id: str
    active_arm: str
    stimulus_definition: StimulusDefinition
    bandit_decision_snapshot: dict[str, Any]
    stimulus_time_s: float | None
    au12_series: tuple[dict[str, float], ...] = ()
    physiological_context: PhysiologicalContext | None = None


@dataclass(frozen=True)
class _ActiveSession:
    session_id: UUID
    stream_url: str
    experiment_id: str
    experiment_row_id: int
    active_arm: str
    stimulus_definition: StimulusDefinition
    bandit_decision_snapshot: dict[str, Any]
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


@dataclass(frozen=True)
class _VisualAu12Telemetry:
    timestamp_s: float
    intensity: float

    def as_payload(self) -> dict[str, float]:
        return {"timestamp_s": self.timestamp_s, "intensity": self.intensity}


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
            visual_source_contract=DESKTOP_LIVE_VISUAL_SOURCE_CONTRACT,
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
        "_stimulus_modality": segment.stimulus_definition.stimulus_modality,
        "_stimulus_payload": segment.stimulus_definition.stimulus_payload.model_dump(mode="json"),
        "_expected_stimulus_rule": segment.stimulus_definition.expected_stimulus_rule,
        "_expected_response_rule": segment.stimulus_definition.expected_response_rule,
        "_stimulus_time": segment.stimulus_time_s,
        "_au12_series": list(segment.au12_series),
        "_bandit_decision_snapshot": segment.bandit_decision_snapshot,
    }
    if segment.physiological_context is not None:
        payload["_physiological_context"] = segment.physiological_context.model_dump(mode="json")
    return InferenceHandoffPayload.model_validate(payload)


def _canonical_utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _segment_id(segment: DesktopSegment) -> str:
    end = segment.segment_window_start_utc + timedelta(seconds=SEGMENT_WINDOW_SECONDS)
    stable_identity = "|".join(
        (
            f"{segment.session_id}",
            _canonical_utc_timestamp(segment.segment_window_start_utc),
            _canonical_utc_timestamp(end),
        )
    )
    return hashlib.sha256(stable_identity.encode("utf-8")).hexdigest()


def _reader_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    bootstrap_schema(conn)
    apply_reader_pragmas(conn)
    return conn


def _latest_physiological_context(db_path: Path, session_id: UUID) -> PhysiologicalContext | None:
    conn = _reader_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT subject_role, rmssd_ms, heart_rate_bpm, source_timestamp_utc,
                   freshness_s, is_stale, provider, source_kind, derivation_method,
                   window_s, validity_ratio, is_valid
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY subject_role
                    ORDER BY source_timestamp_utc DESC, created_at DESC, id DESC
                ) AS rn
                FROM physiology_log
                WHERE session_id = ?
                  AND is_valid = 1
                  AND is_stale = 0
            )
            WHERE rn = 1
            ORDER BY subject_role
            """,
            (str(session_id),),
        ).fetchall()
    finally:
        conn.close()
    snapshots: dict[str, dict[str, object]] = {}
    for row in rows:
        snapshots[str(row["subject_role"])] = {
            "rmssd_ms": row["rmssd_ms"],
            "heart_rate_bpm": row["heart_rate_bpm"],
            "source_timestamp_utc": row["source_timestamp_utc"],
            "freshness_s": row["freshness_s"],
            "is_stale": bool(row["is_stale"]),
            "provider": row["provider"],
            "source_kind": row["source_kind"],
            "derivation_method": row["derivation_method"],
            "window_s": row["window_s"],
            "validity_ratio": row["validity_ratio"],
            "is_valid": bool(row["is_valid"]),
        }
    if not snapshots:
        return None
    try:
        return PhysiologicalContext.model_validate(snapshots)
    except ValidationError:
        logger.exception("module_c_orchestrator ignored invalid physiological context")
        return None


def _fetch_active_session(db_path: Path) -> _ActiveSession | None:
    conn = _reader_connection(db_path)
    try:
        row = conn.execute(
            """
            SELECT s.session_id, s.stream_url, s.experiment_id, s.started_at,
                   s.active_arm, s.stimulus_definition, s.bandit_decision_snapshot,
                   e.id AS experiment_row_id,
                   e.arm,
                   e.stimulus_definition AS experiment_stimulus_definition
            FROM sessions s
            LEFT JOIN experiments e
                ON e.experiment_id = s.experiment_id
               AND e.arm = s.active_arm
            WHERE s.ended_at IS NULL
            ORDER BY s.started_at DESC
            LIMIT 1
            """
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return _active_session_from_row(row)


def _fetch_session_by_id(db_path: Path, session_id: UUID) -> _ActiveSession | None:
    conn = _reader_connection(db_path)
    try:
        row = conn.execute(
            """
            SELECT s.session_id, s.stream_url, s.experiment_id, s.started_at,
                   s.active_arm, s.stimulus_definition, s.bandit_decision_snapshot,
                   e.id AS experiment_row_id,
                   e.arm,
                   e.stimulus_definition AS experiment_stimulus_definition
            FROM sessions s
            LEFT JOIN experiments e
                ON e.experiment_id = s.experiment_id
               AND e.arm = s.active_arm
            WHERE s.session_id = ?
              AND s.ended_at IS NULL
            LIMIT 1
            """,
            (str(session_id),),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return _active_session_from_row(row)


def _active_session_from_row(row: sqlite3.Row) -> _ActiveSession | None:
    session_id = UUID(str(row["session_id"]))
    experiment_id = str(row["experiment_id"] or "greeting_line_v1")
    selection = _selection_from_session_row(row)
    if selection is None:
        return None
    experiment_row_id, active_arm, stimulus_definition, snapshot = selection
    return _ActiveSession(
        session_id=session_id,
        stream_url=str(row["stream_url"]),
        experiment_id=experiment_id,
        experiment_row_id=experiment_row_id,
        active_arm=active_arm,
        stimulus_definition=stimulus_definition,
        bandit_decision_snapshot=snapshot,
    )


def _selection_from_session_row(
    row: sqlite3.Row,
) -> tuple[int, str, StimulusDefinition, dict[str, Any]] | None:
    snapshot_json = row["bandit_decision_snapshot"]
    if snapshot_json is None:
        return None
    snapshot = json.loads(str(snapshot_json))
    active_arm = str(row["active_arm"] or snapshot["selected_arm_id"])
    stimulus_definition_json = row["stimulus_definition"] or row["experiment_stimulus_definition"]
    if stimulus_definition_json is not None:
        stimulus_definition = StimulusDefinition.model_validate_json(str(stimulus_definition_json))
    else:
        stimulus_definition = StimulusDefinition.model_validate(
            {
                "stimulus_modality": snapshot["stimulus_modality"],
                "stimulus_payload": snapshot["stimulus_payload"],
                "expected_stimulus_rule": snapshot["expected_stimulus_rule"],
                "expected_response_rule": snapshot["expected_response_rule"],
            }
        )
    experiment_row_id = int(row["experiment_row_id"] or snapshot["experiment_id"])
    return experiment_row_id, active_arm, stimulus_definition, snapshot


def _parse_sqlite_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace(" ", "T"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _resolve_control_session(
    db_path: Path,
    control: LiveSessionControlMessage,
    active: _ActiveSession | None = None,
) -> _ActiveSession | None:
    if active is not None and control.action == "stimulus":
        return replace(
            active,
            stimulus_definition=control.stimulus_definition or active.stimulus_definition,
            stimulus_time_s=control.stimulus_time_s,
        )
    stored = _fetch_session_by_id(db_path, control.session_id)
    if stored is not None:
        return replace(
            stored,
            stream_url=str(control.stream_url or stored.stream_url),
            stimulus_definition=(
                control.stimulus_definition or stored.stimulus_definition
                if control.action == "stimulus"
                else stored.stimulus_definition
            ),
            stimulus_time_s=control.stimulus_time_s,
        )
    logger.error(
        "module_c_orchestrator could not resolve stored session selection for %s",
        control.session_id,
    )
    return active


def _drain_visual_au12_updates(
    channels: IpcChannels,
    visual_buffer: deque[_VisualAu12Telemetry],
) -> None:
    update_queue = channels.visual_state_updates
    if update_queue is None:
        return
    while True:
        try:
            raw = update_queue.get_nowait()
        except queue.Empty:
            return
        try:
            message = VisualAnalyticsStateMessage.model_validate(raw)
        except Exception:  # noqa: BLE001
            logger.exception("module_c_orchestrator discarded malformed visual state update")
            continue
        if message.latest_au12_timestamp_s is None or message.latest_au12_intensity is None:
            continue
        visual_buffer.append(
            _VisualAu12Telemetry(
                timestamp_s=message.latest_au12_timestamp_s,
                intensity=message.latest_au12_intensity,
            )
        )


def _au12_series_for_window(
    visual_buffer: deque[_VisualAu12Telemetry],
    *,
    start_s: float,
    end_s: float,
) -> tuple[dict[str, float], ...]:
    selected: list[_VisualAu12Telemetry] = []
    retained: deque[_VisualAu12Telemetry] = deque(maxlen=VISUAL_AU12_BUFFER_MAXLEN)
    for observation in visual_buffer:
        if start_s <= observation.timestamp_s <= end_s:
            selected.append(observation)
        if observation.timestamp_s > end_s:
            retained.append(observation)
    visual_buffer.clear()
    visual_buffer.extend(retained)
    return tuple(observation.as_payload() for observation in selected)


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
) -> datetime | None:
    """The §4.C 30 s segment is always anchored at the §7B measurement window.

    Returns the wall-clock UTC instant at which the desired segment
    begins: ``stim_time + 5 s − 30 s`` = ``stim_time − 25 s``. The
    segment ends at ``stim_time + 5 s``, exactly bracketing the §7B
    measurement window. When the orchestrator's audio buffer does not
    extend back 25 s before the stimulus (e.g. on the first stimulus
    of a session, before the buffer has filled), the segment-build
    path pre-pads the missing prefix with silence so the §4.C 30 s
    contract still holds bit-for-bit while dispatch happens as soon
    as ``stim + 5 s`` of post-stimulus audio is available.
    """
    if active.stimulus_time_s is None:
        return None
    return datetime.fromtimestamp(
        active.stimulus_time_s + STIMULUS_MEASUREMENT_WINDOW_END_OFFSET_S - SEGMENT_WINDOW_SECONDS,
        tz=UTC,
    )


def _audio_duration_for_bytes(byte_count: int) -> timedelta:
    duration_s = byte_count / (PCM_SAMPLE_RATE_HZ * AUDIO_SAMPLE_WIDTH_BYTES * PCM_CHANNELS)
    return timedelta(seconds=duration_s)


def _build_stimulus_segment_pcm(
    audio_buffer: bytearray,
    *,
    buffer_start_utc: datetime,
    desired_segment_start_utc: datetime,
    desired_segment_end_offset_bytes: int,
    segment_bytes: int,
) -> bytes:
    """Slice the buffer for a 30 s segment anchored at ``stim − 25 s``.

    When the buffer covers the whole desired segment, returns the
    matching slice. When the buffer started later than the desired
    segment start (typically the first stimulus of a session, before
    the buffer has filled), pre-pads the missing prefix with PCM
    silence so the §4.C 30 s contract holds bit-for-bit. Frame
    alignment may produce a single-frame drift; the result is clamped
    to exactly ``segment_bytes`` either by trimming or by a tail pad.
    """
    if desired_segment_start_utc >= buffer_start_utc:
        skip_bytes = _segment_audio_offset_bytes(
            segment_start_utc=buffer_start_utc,
            target_start_utc=desired_segment_start_utc,
        )
        return bytes(audio_buffer[skip_bytes : skip_bytes + segment_bytes])

    silence_prepad_bytes = _segment_audio_offset_bytes(
        segment_start_utc=desired_segment_start_utc,
        target_start_utc=buffer_start_utc,
    )
    real_audio = bytes(audio_buffer[:desired_segment_end_offset_bytes])
    pcm = (b"\x00" * silence_prepad_bytes) + real_audio
    if len(pcm) >= segment_bytes:
        return pcm[:segment_bytes]
    return pcm + b"\x00" * (segment_bytes - len(pcm))


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
    au12_series: tuple[dict[str, float], ...] = (),
    physiological_context: PhysiologicalContext | None = None,
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
        stimulus_definition=active.stimulus_definition,
        bandit_decision_snapshot=active.bandit_decision_snapshot,
        stimulus_time_s=resolved_stimulus_time_s,
        au12_series=au12_series,
        physiological_context=physiological_context,
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
    visual_buffer: deque[_VisualAu12Telemetry] = deque(maxlen=VISUAL_AU12_BUFFER_MAXLEN)
    segment_start_utc: datetime | None = None
    segment_bytes = SEGMENT_WINDOW_SECONDS * PCM_SAMPLE_RATE_HZ * AUDIO_SAMPLE_WIDTH_BYTES
    max_buffer_bytes = MAX_AUDIO_BUFFER_SECONDS * PCM_SAMPLE_RATE_HZ * AUDIO_SAMPLE_WIDTH_BYTES
    while not shutdown_event.is_set():
        dispatcher.release_acked_blocks()
        _drain_visual_au12_updates(channels, visual_buffer)
        active = _drain_segment_controls(channels, active, db_path=db_path)
        if active is None:
            audio_buffer.clear()
            visual_buffer.clear()
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
            stimulus_segment_start_utc = _stimulus_aligned_segment_start_utc(active)
            if stimulus_segment_start_utc is not None:
                stimulus_segment_end_utc = stimulus_segment_start_utc + timedelta(
                    seconds=SEGMENT_WINDOW_SECONDS
                )
                buffer_end_offset_bytes = _segment_audio_offset_bytes(
                    segment_start_utc=segment_start_utc,
                    target_start_utc=stimulus_segment_end_utc,
                )
                if len(audio_buffer) < buffer_end_offset_bytes:
                    break
                pcm = _build_stimulus_segment_pcm(
                    audio_buffer,
                    buffer_start_utc=segment_start_utc,
                    desired_segment_start_utc=stimulus_segment_start_utc,
                    desired_segment_end_offset_bytes=buffer_end_offset_bytes,
                    segment_bytes=segment_bytes,
                )
                au12_series = _au12_series_for_window(
                    visual_buffer,
                    start_s=stimulus_segment_start_utc.timestamp(),
                    end_s=stimulus_segment_end_utc.timestamp(),
                )
                physiological_context = _latest_physiological_context(db_path, active.session_id)
                segment = _segment_from_active_session(
                    active,
                    segment_start_utc=stimulus_segment_start_utc,
                    pcm_s16le_16khz_mono=pcm,
                    drift_offset_s=drift_offset_s,
                    stimulus_time_s=active.stimulus_time_s,
                    au12_series=au12_series,
                    physiological_context=physiological_context,
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
