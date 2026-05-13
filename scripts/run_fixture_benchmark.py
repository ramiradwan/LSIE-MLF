#!/usr/bin/env python3
"""Run a deterministic v4 desktop fixture benchmark and append a baseline row.

The harness exercises the active desktop data path with a recorded fixture:
``DesktopSegment`` construction, ``DesktopSegmentDispatcher`` shared-memory IPC,
``gpu_ml_worker`` analytics publication, and ``LocalAnalyticsProcessor`` SQLite
persistence. It intentionally avoids retained worker, broker, and replay-stack
compatibility paths.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import sqlite3
import subprocess
import sys
import time
import uuid
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import numpy.typing as npt

from packages.schemas.evaluation import StimulusDefinition

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_BASELINE_PATH: Path = REPO_ROOT / "docs" / "artifacts" / "performance_baseline.md"
FIXTURE_LABEL: str = "v4-fixture:@"
BASELINE_COLUMNS: tuple[str, ...] = (
    "Date",
    "Commit SHA",
    "Scenario",
    "Segments",
    "Dispatch p50 (ms)",
    "Dispatch p95 (ms)",
    "ML publish p50 (ms)",
    "ML publish p95 (ms)",
    "Analytics state p50 (ms)",
    "Analytics state p95 (ms)",
    "Visual AU12 tick p50 (ms)",
    "End-to-end p95 (ms)",
    "Notes",
)
P95_COLUMNS: tuple[str, ...] = (
    "Dispatch p95 (ms)",
    "ML publish p95 (ms)",
    "Analytics state p95 (ms)",
    "End-to-end p95 (ms)",
)
REGRESSION_WARNING_THRESHOLD: float = 1.20
DEFAULT_SEGMENTS: int = 3
_PCM_SAMPLE_RATE_HZ: int = 16_000
_PCM_SAMPLE_WIDTH_BYTES: int = 2

Frame = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class BenchmarkStats:
    segment_count: int
    dispatch_p50_ms: float
    dispatch_p95_ms: float
    ml_publish_p50_ms: float
    ml_publish_p95_ms: float
    analytics_state_p50_ms: float
    analytics_state_p95_ms: float
    visual_au12_tick_p50_ms: float
    end_to_end_p95_ms: float
    persisted_segments: int


@dataclass(frozen=True)
class BenchmarkResult:
    row: str
    row_cells: tuple[str, ...]
    warnings: tuple[str, ...]
    stats: BenchmarkStats


@dataclass(frozen=True)
class _SegmentFixture:
    index: int
    active_arm: str
    stimulus_definition: StimulusDefinition
    stimulus_time_s: float
    pcm_s16le_16khz_mono: bytes
    au12_series: tuple[dict[str, float], ...]


@dataclass(frozen=True)
class _BenchmarkTimings:
    dispatch_ms: tuple[float, ...]
    ml_publish_ms: tuple[float, ...]
    analytics_state_ms: tuple[float, ...]
    visual_au12_tick_ms: tuple[float, ...]
    end_to_end_ms: tuple[float, ...]
    persisted_segments: int


class _BenchmarkTranscriptionEngine:
    def transcribe(self, audio: object) -> str:
        del audio
        return "hello just joined happy to be here"


class _BenchmarkTextPreprocessor:
    def preprocess(self, text: str) -> str:
        return " ".join(text.lower().split())


class _BenchmarkSemanticEvaluator:
    last_semantic_method = "cross_encoder"
    last_semantic_method_version = "v4-fixture-benchmark-v1"

    def evaluate(self, expected_response_rule: str, actual_utterance: str) -> dict[str, Any]:
        del expected_response_rule, actual_utterance
        return {
            "reasoning": "cross_encoder_high_match",
            "is_match": True,
            "confidence_score": 0.91,
        }


class _BenchmarkAcousticAnalyzer:
    def analyze(
        self,
        audio_samples: bytes,
        sample_rate: int = _PCM_SAMPLE_RATE_HZ,
        *,
        stimulus_time_s: float | None = None,
        segment_start_time_s: float | None = None,
    ) -> Any:
        del audio_samples, sample_rate, stimulus_time_s, segment_start_time_s
        from packages.ml_core.acoustic import AcousticMetrics, null_acoustic_result

        return AcousticMetrics.from_observational_result(null_acoustic_result())


class _BenchmarkVideoCapture:
    def __init__(self, frames: Sequence[Frame]) -> None:
        self._frames = list(frames)
        self._started = False

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def get_latest_frame(self) -> Frame | None:
        if not self._started or not self._frames:
            return None
        return self._frames.pop(0)


class _BenchmarkFaceMesh:
    def extract_landmarks(self, frame: object) -> object:
        del frame
        return object()

    def close(self) -> None:
        return None


class _BenchmarkAu12Normalizer:
    def __init__(self, intensities: Sequence[float]) -> None:
        self._intensities = list(intensities)
        self.calibration_buffer: list[float] = []

    def compute_bounded_intensity(self, landmarks: object, *, is_calibrating: bool) -> float:
        del landmarks
        if is_calibrating:
            self.calibration_buffer.append(0.25)
        if not self._intensities:
            return 0.0
        return self._intensities.pop(0)


def _parse_markdown_row(line: str) -> tuple[str, ...]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        raise ValueError(f"not a markdown table row: {line!r}")
    inner = stripped[1:-1]
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for char in inner:
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if escaped:
        current.append("\\")
    cells.append("".join(current).strip())
    return tuple(cells)


def _escape_markdown_cell(cell: str) -> str:
    return cell.replace("\n", " ").replace("|", "\\|").strip()


def _format_markdown_row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(_escape_markdown_cell(cell) for cell in cells) + " |"


def _read_baseline_table(path: Path) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
    if not path.exists():
        raise FileNotFoundError(f"baseline file not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index: int | None = None
    columns: tuple[str, ...] | None = None
    for index, line in enumerate(lines):
        if line.strip().startswith("| Date |"):
            parsed = _parse_markdown_row(line)
            header_index = index
            columns = parsed
            break
    if header_index is None or columns is None:
        raise ValueError(f"baseline table header not found in {path}")
    if columns != BASELINE_COLUMNS:
        raise ValueError(
            "performance baseline table columns changed; expected "
            f"{BASELINE_COLUMNS!r}, found {columns!r}"
        )
    if header_index + 1 >= len(lines):
        raise ValueError("performance baseline table separator missing")

    rows: list[tuple[str, ...]] = []
    for line in lines[header_index + 2 :]:
        if not line.strip().startswith("|"):
            continue
        row = _parse_markdown_row(line)
        if len(row) == len(columns):
            rows.append(row)
    return columns, rows


def _append_row(path: Path, row: str) -> None:
    content = path.read_text(encoding="utf-8")
    newline = "" if content.endswith("\n") else "\n"
    path.write_text(f"{content}{newline}{row}\n", encoding="utf-8")


def _git_sha_short() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return "unknown"
    sha = result.stdout.strip()
    if result.returncode != 0 or not sha:
        return "unknown"
    return sha


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction)


def _format_ms(value: float) -> str:
    if not math.isfinite(value) or value < 0.0:
        value = 0.0
    return f"{value:.3f}"


def _parse_ms_cell(value: str) -> float | None:
    cleaned = value.strip().strip("`").replace(",", "")
    if not cleaned or cleaned.upper() == "TBD":
        return None
    try:
        parsed = float(cleaned)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _load_fixture_script(fixture_path: Path) -> dict[str, Any]:
    script_path = fixture_path / "stimulus_script.json"
    if not script_path.exists():
        raise FileNotFoundError(f"fixture missing stimulus_script.json: {script_path}")
    return cast(dict[str, Any], json.loads(script_path.read_text(encoding="utf-8")))


def _selected_stimuli(script: dict[str, Any], segment_count: int) -> list[dict[str, Any]]:
    stimuli = list(script.get("stimuli") or [])
    if not stimuli:
        raise ValueError("fixture stimulus_script.json contains no stimuli")
    if segment_count > len(stimuli):
        raise ValueError(
            f"fixture contains {len(stimuli)} scripted segments but benchmark requires "
            f"{segment_count}; provide a longer fixture or rerun with --segments"
        )
    return [dict(stimulus) for stimulus in stimuli[:segment_count]]


def _load_fixture_pcm(fixture_path: Path, script: dict[str, Any]) -> bytes:
    audio_path = fixture_path / "audio.wav"
    if not audio_path.exists():
        raise FileNotFoundError(f"fixture missing audio.wav: {audio_path}")
    with wave.open(str(audio_path), "rb") as audio:
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        sample_rate = audio.getframerate()
        pcm = audio.readframes(audio.getnframes())
    if sample_width != _PCM_SAMPLE_WIDTH_BYTES:
        raise ValueError(f"fixture audio.wav must be PCM s16le; found sample_width={sample_width}")
    return _pcm_to_16k_mono(pcm, sample_rate=sample_rate, channels=channels)


def _pcm_to_16k_mono(pcm: bytes, *, sample_rate: int, channels: int) -> bytes:
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    if channels < 1:
        raise ValueError(f"fixture audio.wav must have at least one channel; found {channels}")
    if channels > 1:
        usable = (len(samples) // channels) * channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)
    if sample_rate != _PCM_SAMPLE_RATE_HZ:
        if sample_rate <= 0:
            raise ValueError(f"fixture audio.wav sample_rate must be positive; found {sample_rate}")
        output_len = int(round(len(samples) * (_PCM_SAMPLE_RATE_HZ / sample_rate)))
        if output_len <= 0:
            return b""
        source_x = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
        target_x = np.linspace(0.0, 1.0, num=output_len, endpoint=False)
        samples = np.interp(target_x, source_x, samples)
    clipped = np.clip(np.rint(samples), -32768, 32767).astype("<i2")
    return clipped.tobytes()


def _segment_pcm(pcm: bytes, segment_index: int, segment_duration_s: float) -> bytes:
    bytes_per_second = _PCM_SAMPLE_RATE_HZ * _PCM_SAMPLE_WIDTH_BYTES
    start = int(round(segment_index * segment_duration_s * bytes_per_second))
    end = int(round((segment_index + 1) * segment_duration_s * bytes_per_second))
    segment = pcm[start:end]
    minimum = max(1, int(round(segment_duration_s * bytes_per_second)))
    if len(segment) < minimum:
        raise ValueError(f"fixture audio is too short for segment {segment_index}")
    return segment


def _fixture_segments(fixture_path: Path, segment_count: int) -> tuple[_SegmentFixture, ...]:
    script = _load_fixture_script(fixture_path)
    stimuli = _selected_stimuli(script, segment_count)
    segment_duration_s = float(script["segment_duration_s"])
    pcm = _load_fixture_pcm(fixture_path, script)
    segments: list[_SegmentFixture] = []
    for stimulus in stimuli:
        index = int(stimulus["segment_index"])
        stimulus_offset_s = float(stimulus["stimulus_offset_s"])
        peak = float(stimulus.get("expected_peak_au12", 0.75))
        segments.append(
            _SegmentFixture(
                index=index,
                active_arm=str(stimulus["expected_arm_id"]),
                stimulus_definition=StimulusDefinition.model_validate(stimulus["stimulus_definition"]),
                stimulus_time_s=(index * segment_duration_s) + stimulus_offset_s,
                pcm_s16le_16khz_mono=_segment_pcm(pcm, index, segment_duration_s),
                au12_series=(
                    {"timestamp_s": max(0.0, stimulus_offset_s - 2.0), "intensity": 0.05},
                    {"timestamp_s": stimulus_offset_s + 0.25, "intensity": peak},
                    {"timestamp_s": stimulus_offset_s + 0.75, "intensity": min(1.0, peak * 0.9)},
                ),
            )
        )
    return tuple(segments)


def _benchmark_bandit_snapshot(
    *,
    fixture: _SegmentFixture,
    experiment_row_id: int,
    selection_time_utc: datetime,
) -> dict[str, object]:
    return {
        "selection_method": "thompson_sampling",
        "selection_time_utc": selection_time_utc.isoformat(),
        "experiment_id": experiment_row_id,
        "policy_version": "desktop_replay_v1",
        "selected_arm_id": fixture.active_arm,
        "candidate_arm_ids": [fixture.active_arm],
        "posterior_by_arm": {fixture.active_arm: {"alpha": 1.0, "beta": 1.0}},
        "sampled_theta_by_arm": {fixture.active_arm: 0.5},
        "stimulus_modality": fixture.stimulus_definition.stimulus_modality,
        "stimulus_payload": fixture.stimulus_definition.stimulus_payload.model_dump(mode="json"),
        "expected_stimulus_rule": fixture.stimulus_definition.expected_stimulus_rule,
        "expected_response_rule": fixture.stimulus_definition.expected_response_rule,
        "decision_context_hash": "0" * 64,
        "random_seed": 0,
    }


def _bootstrap_benchmark_db(db_path: Path, session_id: uuid.UUID) -> dict[str, int]:
    from services.desktop_app.state.sqlite_schema import bootstrap_schema

    with sqlite3.connect(str(db_path), isolation_level=None) as conn:
        conn.row_factory = sqlite3.Row
        bootstrap_schema(conn)
        conn.execute(
            """
            INSERT OR IGNORE INTO sessions (session_id, stream_url, experiment_id, started_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                str(session_id),
                "fixture://v4-benchmark",
                "greeting_line_v1",
                "2026-05-02T12:00:00Z",
            ),
        )
        rows = conn.execute(
            """
            SELECT id, arm
            FROM experiments
            WHERE experiment_id = ?
            """,
            ("greeting_line_v1",),
        ).fetchall()
        experiment_rows = {str(row["arm"]): int(row["id"]) for row in rows}
        if not experiment_rows:
            raise RuntimeError("benchmark experiment seed rows were not created")
        return experiment_rows


def _persisted_segment_count(db_path: Path) -> int:
    with sqlite3.connect(str(db_path), isolation_level=None) as conn:
        row = conn.execute("SELECT COUNT(*) FROM analytics_message_ledger").fetchone()
    return int(row[0]) if row is not None else 0


def _make_visual_tracker(intensities: Sequence[float]) -> Any:
    from services.desktop_app.processes import gpu_ml_worker

    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in intensities]
    return gpu_ml_worker.LiveVisualTracker(
        video_capture_factory=lambda _path: _BenchmarkVideoCapture(frames),
        face_mesh_factory=_BenchmarkFaceMesh,
        au12_factory=lambda: _BenchmarkAu12Normalizer(intensities),
    )


def _prime_visual_tracker(tracker: Any, session_id: uuid.UUID, fixture: _SegmentFixture) -> None:
    from services.desktop_app.ipc.control_messages import LiveSessionControlMessage
    from services.desktop_app.processes import gpu_ml_worker

    timestamp = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    tracker.handle_control(
        LiveSessionControlMessage(
            action="start",
            session_id=session_id,
            stream_url="fixture://v4-benchmark",
            experiment_id="greeting_line_v1",
            active_arm=fixture.active_arm,
            stimulus_definition=fixture.stimulus_definition,
            timestamp_utc=timestamp,
        )
    )
    tracker.handle_control(
        LiveSessionControlMessage(
            action="stimulus",
            session_id=session_id,
            stream_url="fixture://v4-benchmark",
            experiment_id="greeting_line_v1",
            active_arm=fixture.active_arm,
            stimulus_definition=fixture.stimulus_definition,
            stimulus_time_s=fixture.stimulus_time_s,
            timestamp_utc=timestamp,
        )
    )
    with patch.object(gpu_ml_worker, "_VISUAL_CALIBRATION_FRAMES_REQUIRED", 1):
        tracker.tick(timestamp)


def _run_v4_fixture_path(fixtures: Sequence[_SegmentFixture]) -> _BenchmarkTimings:
    from services.desktop_app.ipc import IpcChannels
    from services.desktop_app.ipc.control_messages import InferenceControlMessage
    from services.desktop_app.processes import gpu_ml_worker
    from services.desktop_app.processes.analytics_state_worker import LocalAnalyticsProcessor
    from services.desktop_app.processes.module_c_orchestrator import (
        DesktopSegment,
        DesktopSegmentDispatcher,
    )

    with TemporaryDirectory(prefix="lsie-v4-benchmark-", ignore_cleanup_errors=True) as tmp_dir:
        db_path = Path(tmp_dir) / "desktop.sqlite"
        session_id = uuid.uuid4()
        experiment_row_ids = _bootstrap_benchmark_db(db_path, session_id)
        ml_inbox: queue.Queue[object] = queue.Queue()
        analytics_inbox: queue.Queue[object] = queue.Queue()
        pcm_acks: queue.Queue[object] = queue.Queue()
        channels = IpcChannels(
            ml_inbox=cast(Any, ml_inbox),
            drift_updates=cast(Any, queue.Queue()),
            analytics_inbox=cast(Any, analytics_inbox),
            pcm_acks=cast(Any, pcm_acks),
            live_control=cast(Any, queue.Queue()),
            segment_control=cast(Any, queue.Queue()),
        )
        dispatcher = DesktopSegmentDispatcher(ml_inbox, pcm_acks)
        processor = LocalAnalyticsProcessor(db_path, client_id="fixture-benchmark")
        dispatch_ms: list[float] = []
        ml_publish_ms: list[float] = []
        analytics_state_ms: list[float] = []
        visual_au12_tick_ms: list[float] = []
        end_to_end_ms: list[float] = []

        with (
            patch("packages.ml_core.preprocessing.TextPreprocessor", _BenchmarkTextPreprocessor),
            patch("packages.ml_core.semantic.SemanticEvaluator", _BenchmarkSemanticEvaluator),
            patch("packages.ml_core.acoustic.AcousticAnalyzer", _BenchmarkAcousticAnalyzer),
        ):
            try:
                for fixture in fixtures:
                    segment_start = datetime(2026, 5, 2, 12, 0, tzinfo=UTC) + timedelta(
                        seconds=fixture.index * 30
                    )
                    experiment_row_id = experiment_row_ids.get(fixture.active_arm)
                    if experiment_row_id is None:
                        raise RuntimeError(
                            f"benchmark fixture arm has no seed row: {fixture.active_arm}"
                        )
                    desktop_segment = DesktopSegment(
                        session_id=session_id,
                        stream_url="fixture://v4-benchmark",
                        segment_window_start_utc=segment_start,
                        pcm_s16le_16khz_mono=fixture.pcm_s16le_16khz_mono,
                        experiment_row_id=experiment_row_id,
                        experiment_id="greeting_line_v1",
                        active_arm=fixture.active_arm,
                        stimulus_definition=fixture.stimulus_definition,
                        bandit_decision_snapshot=_benchmark_bandit_snapshot(
                            fixture=fixture,
                            experiment_row_id=experiment_row_id,
                            selection_time_utc=segment_start,
                        ),
                        stimulus_time_s=fixture.stimulus_time_s,
                    )
                    tracker = _make_visual_tracker(
                        [row["intensity"] for row in fixture.au12_series]
                    )
                    _prime_visual_tracker(tracker, session_id, fixture)

                    start = time.perf_counter()
                    dispatch_start = time.perf_counter()
                    if not dispatcher.dispatch(desktop_segment):
                        raise RuntimeError(
                            "DesktopSegmentDispatcher failed to dispatch fixture segment"
                        )
                    dispatch_ms.append((time.perf_counter() - dispatch_start) * 1000.0)
                    raw = ml_inbox.get_nowait()
                    control = InferenceControlMessage.model_validate(raw)

                    visual_start = time.perf_counter()
                    tracker.tick(segment_start)
                    visual_au12_tick_ms.append((time.perf_counter() - visual_start) * 1000.0)

                    publish_start = time.perf_counter()
                    gpu_ml_worker._publish_analytics_result(  # noqa: SLF001
                        channels,
                        control,
                        tracker,
                        transcription_engine=_BenchmarkTranscriptionEngine(),
                    )
                    ml_publish_ms.append((time.perf_counter() - publish_start) * 1000.0)

                    raw_analytics = analytics_inbox.get_nowait()
                    analytics_start = time.perf_counter()
                    processor.process(raw_analytics)
                    analytics_state_ms.append((time.perf_counter() - analytics_start) * 1000.0)
                    end_to_end_ms.append((time.perf_counter() - start) * 1000.0)
            finally:
                dispatcher.close_inflight_blocks()
                tracker = locals().get("tracker")
                if tracker is not None:
                    tracker.close()

        persisted_segments = _persisted_segment_count(db_path)

    return _BenchmarkTimings(
        dispatch_ms=tuple(dispatch_ms),
        ml_publish_ms=tuple(ml_publish_ms),
        analytics_state_ms=tuple(analytics_state_ms),
        visual_au12_tick_ms=tuple(visual_au12_tick_ms),
        end_to_end_ms=tuple(end_to_end_ms),
        persisted_segments=persisted_segments,
    )


def _collect_fixture_stats(fixture_path: Path, segment_count: int) -> BenchmarkStats:
    fixtures = _fixture_segments(fixture_path, segment_count)
    timings = _run_v4_fixture_path(fixtures)
    if timings.persisted_segments != segment_count:
        raise RuntimeError(
            "v4 benchmark persisted "
            f"{timings.persisted_segments} segments; expected {segment_count}"
        )
    return BenchmarkStats(
        segment_count=segment_count,
        dispatch_p50_ms=_percentile(timings.dispatch_ms, 50.0),
        dispatch_p95_ms=_percentile(timings.dispatch_ms, 95.0),
        ml_publish_p50_ms=_percentile(timings.ml_publish_ms, 50.0),
        ml_publish_p95_ms=_percentile(timings.ml_publish_ms, 95.0),
        analytics_state_p50_ms=_percentile(timings.analytics_state_ms, 50.0),
        analytics_state_p95_ms=_percentile(timings.analytics_state_ms, 95.0),
        visual_au12_tick_p50_ms=_percentile(timings.visual_au12_tick_ms, 50.0),
        end_to_end_p95_ms=_percentile(timings.end_to_end_ms, 95.0),
        persisted_segments=timings.persisted_segments,
    )


def _find_previous_fixture_row(rows: Sequence[tuple[str, ...]]) -> tuple[str, ...] | None:
    scenario_index = BASELINE_COLUMNS.index("Scenario")
    previous: tuple[str, ...] | None = None
    for row in rows:
        if len(row) == len(BASELINE_COLUMNS) and row[scenario_index].strip() == FIXTURE_LABEL:
            previous = row
    return previous


def _observational_warnings(
    current_cells: Sequence[str],
    previous_fixture_row: Sequence[str] | None,
) -> tuple[str, ...]:
    if previous_fixture_row is None:
        return ()
    warnings: list[str] = []
    for column in P95_COLUMNS:
        index = BASELINE_COLUMNS.index(column)
        current = _parse_ms_cell(current_cells[index])
        previous = _parse_ms_cell(previous_fixture_row[index])
        if current is None or previous is None or previous <= 0.0:
            continue
        ratio = current / previous
        if ratio > REGRESSION_WARNING_THRESHOLD:
            warnings.append(
                f"{column} {current:.3f}ms is {((ratio - 1.0) * 100.0):.1f}% "
                f"above previous v4 fixture row {previous:.3f}ms"
            )
    return tuple(warnings)


def _build_row_cells(
    stats: BenchmarkStats,
    warnings: Sequence[str],
    *,
    notes_suffix: str = "",
) -> tuple[str, ...]:
    notes_parts = [
        "v4 desktop IPC/SQLite fixture benchmark",
        f"persisted={stats.persisted_segments}",
        "live ADB/scrcpy path not measured",
        f"warnings={len(warnings)}",
    ]
    if notes_suffix:
        notes_parts.append(notes_suffix)
    notes = "; ".join(notes_parts)
    return (
        datetime.now(tz=UTC).date().isoformat(),
        f"`{_git_sha_short()}`",
        FIXTURE_LABEL,
        str(stats.segment_count),
        _format_ms(stats.dispatch_p50_ms),
        _format_ms(stats.dispatch_p95_ms),
        _format_ms(stats.ml_publish_p50_ms),
        _format_ms(stats.ml_publish_p95_ms),
        _format_ms(stats.analytics_state_p50_ms),
        _format_ms(stats.analytics_state_p95_ms),
        _format_ms(stats.visual_au12_tick_p50_ms),
        _format_ms(stats.end_to_end_p95_ms),
        notes,
    )


def run_fixture_benchmark(
    fixture_path: Path,
    *,
    segment_count: int = DEFAULT_SEGMENTS,
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    append: bool = True,
    notes_suffix: str = "",
) -> BenchmarkResult:
    if segment_count < 1:
        raise ValueError("segment_count must be >= 1")
    fixture_path = fixture_path.resolve()
    if not fixture_path.exists():
        raise FileNotFoundError(f"fixture path not found: {fixture_path}")
    baseline_path = baseline_path.resolve()

    _columns, rows = _read_baseline_table(baseline_path)
    previous_fixture_row = _find_previous_fixture_row(rows)

    stats = _collect_fixture_stats(fixture_path, segment_count)
    provisional_cells = _build_row_cells(stats, (), notes_suffix=notes_suffix)
    warnings = _observational_warnings(provisional_cells, previous_fixture_row)
    row_cells = _build_row_cells(stats, warnings, notes_suffix=notes_suffix)
    row = _format_markdown_row(row_cells)

    if len(row_cells) != len(BASELINE_COLUMNS):
        raise AssertionError("fixture row column count does not match baseline")
    if append:
        _append_row(baseline_path, row)

    return BenchmarkResult(row=row, row_cells=row_cells, warnings=warnings, stats=stats)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run v4 desktop fixture timing benchmark and append performance baseline row.",
    )
    parser.add_argument("fixture_path", type=Path, help="Deterministic capture fixture directory")
    parser.add_argument(
        "-n",
        "--segments",
        type=int,
        default=DEFAULT_SEGMENTS,
        help=f"Number of fixture segments to run (default: {DEFAULT_SEGMENTS})",
    )
    parser.add_argument(
        "--baseline-path",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help="Performance baseline markdown file to append to",
    )
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Emit the row without appending it (prefer --baseline-path for tests)",
    )
    parser.add_argument(
        "--notes-suffix",
        default="",
        help="Optional suffix appended to the Notes cell",
    )
    parser.add_argument("--verbose", action="store_true", help="Accepted for CLI compatibility")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = run_fixture_benchmark(
            args.fixture_path,
            segment_count=int(args.segments),
            baseline_path=args.baseline_path,
            append=not bool(args.no_append),
            notes_suffix=str(args.notes_suffix),
        )
    except Exception as exc:
        print(f"run_fixture_benchmark: {exc}", file=sys.stderr)
        return 1

    print(result.row)
    for warning in result.warnings:
        print(f"WARNING: observational v4 fixture regression: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
