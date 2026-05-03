#!/usr/bin/env python3
"""Run the offline replay-fixture benchmark and append a baseline row.

The harness exercises the production replay capture source, the normal
Orchestrator.run() → desktop IPC dispatch boundary, and the replay fixture
stimulus schedule while replacing live-only ML/network dependencies with small
in-process shims. Benchmark metrics are sourced from the structured production
log lines emitted by ``orchestrator.py``, ``inference.py``, and ``au12.py``
rather than from harness-local timers.

By default a row is appended to ``docs/artifacts/performance_baseline.md`` and
also emitted to stdout. Tests and ad-hoc dry runs can point ``--baseline-path``
at a copy of the baseline file to keep the repository clean.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import json
import logging
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Sequence, cast
from unittest.mock import patch

import numpy as np
import numpy.typing as npt

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_BASELINE_PATH: Path = REPO_ROOT / "docs" / "artifacts" / "performance_baseline.md"
FIXTURE_LABEL: str = "fixture:@"
BASELINE_COLUMNS: tuple[str, ...] = (
    "Date",
    "Commit SHA",
    "Cycle / PR",
    "Segment-assembly p50 (ms)",
    "Segment-assembly p95 (ms)",
    "ML inference p50 (ms)",
    "ML inference p95 (ms)",
    "AU12 per-frame p50 (ms)",
    "Co-Modulation window compute (ms)",
    "Notes",
)
P95_COLUMNS: tuple[str, ...] = (
    "Segment-assembly p95 (ms)",
    "ML inference p95 (ms)",
)
REGRESSION_WARNING_THRESHOLD: float = 1.20
DEFAULT_SEGMENTS: int = 10

_SEGMENT_ASSEMBLY_RE = re.compile(
    r"^BENCHMARK segment_assembly_ms=(?P<ms>[0-9]+(?:\.[0-9]+)?) segment_id=(?P<segment_id>[^ ]+)$"
)
_ML_INFERENCE_RE = re.compile(
    r"^BENCHMARK ml_inference_ms=(?P<ms>[0-9]+(?:\.[0-9]+)?) segment_id=(?P<segment_id>[^ ]+)$"
)
_AU12_RE = re.compile(
    r"^BENCHMARK au12_bounded_ms=(?P<ms>[0-9]+(?:\.[0-9]+)?) calibrating=(?P<calibrating>True|False)$"
)

logger = logging.getLogger(__name__)

LandmarkArray = npt.NDArray[np.floating[Any]]


@dataclass(frozen=True)
class BenchmarkStats:
    """Aggregate fixture-regime benchmark timings in milliseconds."""

    segment_count: int
    segment_assembly_p50_ms: float
    segment_assembly_p95_ms: float
    ml_inference_p50_ms: float
    ml_inference_p95_ms: float
    au12_per_frame_p50_ms: float
    comodulation_window_compute_ms: float = 0.0


@dataclass(frozen=True)
class BenchmarkResult:
    """Final row and observational warnings from one harness run."""

    row: str
    row_cells: tuple[str, ...]
    warnings: tuple[str, ...]
    stats: BenchmarkStats


class _NoopPersistTask:
    """In-process stand-in for the Module E Celery task during fixture timing."""

    def delay(self, metrics: dict[str, Any]) -> None:
        del metrics
        return None


class _BenchmarkTranscriptionEngine:
    """Deterministic no-GPU transcription shim used by Module D in CI."""

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        del audio_path, language
        return "Hello just joined happy to be here"


class _BenchmarkTextPreprocessor:
    """Tiny tokenizer/normalizer that avoids spaCy model downloads in CI."""

    def preprocess(self, text: str) -> str:
        return " ".join(text.lower().split())


class _BenchmarkSemanticEvaluator:
    """Deterministic semantic evaluator that avoids Azure OpenAI access."""

    def evaluate(self, expected_greeting: str, actual_utterance: str) -> dict[str, Any]:
        return {
            "reasoning": (
                "offline fixture benchmark shim accepted deterministic "
                f"utterance '{actual_utterance}' for expected '{expected_greeting}'"
            ),
            "is_match": True,
            "confidence_score": 1.0,
        }


class _BenchmarkFaceMeshProcessor:
    """Small MediaPipe-free face-mesh shim returning valid AU12 landmarks."""

    def __init__(self) -> None:
        self._calls: int = 0

    def extract_landmarks(self, frame: npt.NDArray[np.uint8]) -> LandmarkArray:
        del frame
        self._calls += 1
        mouth_ratio = 0.55 + (0.015 if self._calls % 2 else 0.0)
        return _landmarks_for_mouth_ratio(mouth_ratio)

    def close(self) -> None:
        return None


class _BenchmarkAcousticAnalyzer:
    """Deterministic local shim that still requires real stimulus timing inputs."""

    def analyze(
        self,
        audio_samples: bytes,
        sample_rate: int = 16000,
        *,
        stimulus_time_s: float | None = None,
        segment_start_time_s: float | None = None,
    ) -> Any:
        del audio_samples, sample_rate
        if stimulus_time_s is None or segment_start_time_s is None:
            raise AssertionError("benchmark acoustic shim requires non-null stimulus timing")

        from packages.ml_core.acoustic import AcousticMetrics, null_acoustic_result

        return AcousticMetrics.from_observational_result(null_acoustic_result())


class _FallbackCeleryTask:
    """Minimal Celery Task stand-in for environments without worker deps."""

    def __init__(self, func: Any | None = None, *, bind: bool = False) -> None:
        self._func = func
        self._bind = bind

    def run(self, *args: Any, **kwargs: Any) -> Any:
        if self._func is None:
            return None
        if self._bind:
            return self._func(self, *args, **kwargs)
        return self._func(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.run(*args, **kwargs)

    def delay(self, *args: Any, **kwargs: Any) -> Any:
        return self.run(*args, **kwargs)


class _FallbackCelery:
    """Tiny subset of Celery used by services.worker.celery_app."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.conf: dict[str, Any] = {}

    def task(self, *decorator_args: Any, **decorator_kwargs: Any) -> Any:
        bind = bool(decorator_kwargs.get("bind", False))

        def decorate(func: Any) -> _FallbackCeleryTask:
            return _FallbackCeleryTask(func, bind=bind)

        if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1:
            return decorate(decorator_args[0])
        return decorate

    def autodiscover_tasks(self, packages: Sequence[str]) -> None:
        del packages


class _BenchmarkLogCapture(logging.Handler):
    """Collect structured benchmark timings from production log messages."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.segment_assembly_ms: list[float] = []
        self.ml_inference_ms: list[float] = []
        self.au12_ms: list[float] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        for pattern, target in (
            (_SEGMENT_ASSEMBLY_RE, self.segment_assembly_ms),
            (_ML_INFERENCE_RE, self.ml_inference_ms),
            (_AU12_RE, self.au12_ms),
        ):
            match = pattern.match(message)
            if match is None:
                continue
            try:
                target.append(float(match.group("ms")))
            except (TypeError, ValueError):
                return
            return


def _process_segment_payload_from_control_message(message: Any) -> dict[str, Any]:
    from services.desktop_app.ipc.control_messages import InferenceControlMessage
    from services.desktop_app.ipc.shared_buffers import read_pcm_block

    control_message = InferenceControlMessage.model_validate(message)
    payload = copy.deepcopy(control_message.handoff)
    payload.update(copy.deepcopy(control_message.forward_fields))
    payload["_audio_data"] = read_pcm_block(control_message.audio.to_metadata())
    return payload


class _InProcessDispatchBridge:
    """Local worker bridge that consumes desktop IPC dispatches in-process."""

    def __init__(
        self,
        *,
        orchestrator: Any,
        inference_module: Any,
        ipc_queue: Any,
        expected_segments: int,
    ) -> None:
        self._orchestrator = orchestrator
        self._inference_module = inference_module
        self._ipc_queue = ipc_queue
        self._expected_segments = expected_segments
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._consume_ipc_queue,
            name="fixture-benchmark-worker",
            daemon=True,
        )
        self.done = threading.Event()
        self.payloads: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []
        self.error: Exception | None = None

    def start(self) -> None:
        self._worker_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._worker_thread.join(timeout=2.0)

    def _consume_ipc_queue(self) -> None:
        while not self._stop_event.is_set():
            try:
                raw = self._ipc_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                payload = _process_segment_payload_from_control_message(raw)
                payload_copy = copy.deepcopy(payload)
                result = _execute_process_segment(self._inference_module, payload)
            except Exception as exc:
                with self._lock:
                    self.error = exc
                self._orchestrator.stop()
                self.done.set()
                return

            with self._lock:
                self.payloads.append(payload_copy)
                self.results.append(result)
                if len(self.results) >= self._expected_segments:
                    self._orchestrator.stop()
                    self.done.set()
                    return

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise RuntimeError("fixture benchmark worker bridge failed") from self.error
        if len(self.payloads) != self._expected_segments:
            raise RuntimeError(
                f"fixture benchmark processed {len(self.payloads)} segments; "
                f"expected {self._expected_segments}"
            )
        if not all(payload.get("_stimulus_time") is not None for payload in self.payloads):
            raise RuntimeError("fixture benchmark dispatched a payload without _stimulus_time")


def _ensure_celery_importable() -> None:
    """Provide a local Celery shim when the runtime lacks worker deps."""
    if "celery" in sys.modules:
        return
    try:
        __import__("celery")
        return
    except ModuleNotFoundError:
        celery_stub = ModuleType("celery")
        setattr(celery_stub, "Celery", _FallbackCelery)
        setattr(celery_stub, "Task", _FallbackCeleryTask)
        sys.modules["celery"] = celery_stub


def _landmarks_for_mouth_ratio(mouth_ratio: float) -> LandmarkArray:
    """Construct the minimal landmark geometry used by ``AU12Normalizer``."""
    landmarks = np.zeros((478, 3), dtype=np.float64)
    landmarks[33] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    landmarks[133] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    landmarks[362] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    landmarks[263] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    landmarks[61] = np.array([0.20, 0.50, 0.0], dtype=np.float64)
    landmarks[291] = np.array([0.20 + mouth_ratio, 0.50, 0.0], dtype=np.float64)
    return landmarks


@contextlib.contextmanager
def _temporary_attr(target: Any, name: str, value: Any) -> Iterator[None]:
    original = getattr(target, name)
    setattr(target, name, value)
    try:
        yield
    finally:
        setattr(target, name, original)


@contextlib.contextmanager
def _benchmark_environment(fixture_path: Path, *, realtime: bool = True) -> Iterator[None]:
    updates = {
        "REPLAY_CAPTURE_FIXTURE": str(fixture_path),
        "REPLAY_CAPTURE_REALTIME": "1" if realtime else "0",
        "AUTO_STIMULUS_DELAY_S": "0",
    }
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def _offline_dependency_shims() -> Iterator[None]:
    """Patch live-only ML/network dependencies with deterministic local shims."""
    _ensure_celery_importable()

    from packages.ml_core import acoustic as acoustic_mod
    from packages.ml_core import face_mesh as face_mesh_mod
    from packages.ml_core import preprocessing as preprocessing_mod
    from packages.ml_core import semantic as semantic_mod
    from packages.ml_core import transcription as transcription_mod
    from services.worker.pipeline import orchestrator as orchestrator_mod
    from services.worker.tasks import inference as inference_mod

    def benchmark_register_session(self: Any) -> None:
        del self
        return None

    def benchmark_select_experiment_arm(self: Any) -> None:
        self._active_arm = "simple_hello"
        self._expected_greeting = "Hello! Just joined, happy to be here!"
        
    def benchmark_poll(self: Any) -> None:
        return None

    with (
        _temporary_attr(face_mesh_mod, "FaceMeshProcessor", _BenchmarkFaceMeshProcessor),
        _temporary_attr(transcription_mod, "TranscriptionEngine", _BenchmarkTranscriptionEngine),
        _temporary_attr(preprocessing_mod, "TextPreprocessor", _BenchmarkTextPreprocessor),
        _temporary_attr(semantic_mod, "SemanticEvaluator", _BenchmarkSemanticEvaluator),
        _temporary_attr(acoustic_mod, "AcousticAnalyzer", _BenchmarkAcousticAnalyzer),
        _temporary_attr(inference_mod, "persist_metrics", _NoopPersistTask()),
        _temporary_attr(orchestrator_mod.Orchestrator, "_register_session", benchmark_register_session),
        _temporary_attr(
            orchestrator_mod.Orchestrator,
            "_select_experiment_arm",
            benchmark_select_experiment_arm,
        ),
        _temporary_attr(orchestrator_mod.DriftCorrector, "poll", benchmark_poll),
    ):
        yield


@contextlib.contextmanager
def _capture_benchmark_logs() -> Iterator[_BenchmarkLogCapture]:
    handler = _BenchmarkLogCapture()
    logger_configs = (
        ("services.worker.pipeline.orchestrator", logging.INFO),
        ("services.worker.tasks.inference", logging.INFO),
        ("packages.ml_core.au12", logging.DEBUG),
    )
    previous_levels: list[tuple[logging.Logger, int]] = []
    try:
        for logger_name, minimum_level in logger_configs:
            target_logger = logging.getLogger(logger_name)
            previous_levels.append((target_logger, target_logger.level))
            target_logger.addHandler(handler)
            if target_logger.level == logging.NOTSET or target_logger.level > minimum_level:
                target_logger.setLevel(minimum_level)
        yield handler
    finally:
        for target_logger, previous_level in previous_levels:
            target_logger.removeHandler(handler)
            target_logger.setLevel(previous_level)


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


def _segment_window_seconds(script: dict[str, Any]) -> int:
    raw_duration = float(script["segment_duration_s"])
    rounded = int(round(raw_duration))
    if rounded < 1 or not math.isclose(raw_duration, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "fixture segment_duration_s must be an integer number of seconds to drive "
            "the production Orchestrator.run() loop"
        )
    return rounded


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


def _execute_process_segment(inference_module: Any, payload: dict[str, Any]) -> dict[str, Any]:
    task = inference_module.process_segment
    if hasattr(task, "run"):
        result = task.run(payload)
    else:
        result = task(None, payload)
    if not isinstance(result, dict):
        raise TypeError(f"process_segment returned non-dict result: {type(result)!r}")
    return dict(result)


def _drive_fixture_stimuli(
    orchestrator: Any,
    replay: Any,
    stimuli: Sequence[dict[str, Any]],
    *,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        if getattr(replay, "is_running", False) and getattr(replay, "_start_monotonic", None) is not None:
            break
        stop_event.wait(0.01)
    else:
        return

    replay_start = float(replay._start_monotonic)
    for stimulus in stimuli:
        if stop_event.is_set():
            return
        target_monotonic = replay_start + float(replay.elapsed_for_stimulus(stimulus))
        while not stop_event.is_set():
            remaining_s = target_monotonic - time.monotonic()
            if remaining_s <= 0.0:
                break
            stop_event.wait(min(remaining_s, 0.01))
        if stop_event.is_set():
            return
        orchestrator._active_arm = str(stimulus.get("expected_arm_id", orchestrator._active_arm))
        orchestrator._expected_greeting = str(
            stimulus.get("expected_greeting_text", orchestrator._expected_greeting)
        )
        orchestrator.record_stimulus_injection()


def _benchmark_timeout_s(segment_window_seconds: int, segment_count: int) -> float:
    return max(15.0, (segment_window_seconds * float(segment_count)) + 15.0)


async def _run_fixture_pipeline(
    fixture_path: Path,
    *,
    script: dict[str, Any],
    stimuli: Sequence[dict[str, Any]],
    segment_window_seconds: int,
) -> _InProcessDispatchBridge:
    _ensure_celery_importable()

    from services.worker.pipeline import orchestrator as orchestrator_mod
    from services.worker.pipeline.orchestrator import Orchestrator
    from services.worker.pipeline.replay_capture import ReplayCaptureSource
    from services.worker.tasks import inference as inference_mod

    del fixture_path, script

    ipc_queue: queue.Queue[object] = queue.Queue()
    orchestrator = Orchestrator(
        stream_url="replay://fixture-benchmark",
        session_id=str(uuid.uuid4()),
        ipc_queue=ipc_queue,
    )
    orchestrator._redis = None
    replay = orchestrator.audio_resampler
    if not isinstance(replay, ReplayCaptureSource):
        raise RuntimeError("orchestrator did not boot with ReplayCaptureSource")

    bridge = _InProcessDispatchBridge(
        orchestrator=orchestrator,
        inference_module=inference_mod,
        ipc_queue=ipc_queue,
        expected_segments=len(stimuli),
    )
    stimulus_stop_event = threading.Event()
    stimulus_thread = threading.Thread(
        target=_drive_fixture_stimuli,
        kwargs={
            "orchestrator": orchestrator,
            "replay": replay,
            "stimuli": list(stimuli),
            "stop_event": stimulus_stop_event,
        },
        name="fixture-benchmark-stimuli",
        daemon=True,
    )

    with patch.object(orchestrator_mod, "SEGMENT_WINDOW_SECONDS", segment_window_seconds):
        run_task = asyncio.create_task(orchestrator.run())
        bridge.start()
        stimulus_thread.start()
        try:
            timeout_s = _benchmark_timeout_s(segment_window_seconds, len(stimuli))
            completed = await asyncio.to_thread(bridge.done.wait, timeout_s)
            if not completed:
                raise TimeoutError("fixture benchmark timed out waiting for segment dispatch")
        finally:
            stimulus_stop_event.set()
            orchestrator.stop()
            bridge.stop()
            stimulus_thread.join(timeout=2.0)
            orchestrator.close_inflight_blocks()
            try:
                await asyncio.wait_for(run_task, timeout=5.0)
            except asyncio.TimeoutError:
                run_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await run_task
        if run_task.done() and not run_task.cancelled():
            exception = run_task.exception()
            if exception is not None:
                raise exception

    bridge.raise_if_failed()
    return bridge


def _collect_fixture_stats(fixture_path: Path, segment_count: int) -> BenchmarkStats:
    script = _load_fixture_script(fixture_path)
    stimuli = _selected_stimuli(script, segment_count)
    segment_window_seconds = _segment_window_seconds(script)

    with (
        _benchmark_environment(fixture_path, realtime=True),
        _offline_dependency_shims(),
        _capture_benchmark_logs() as log_capture,
    ):
        bridge = asyncio.run(
            _run_fixture_pipeline(
                fixture_path,
                script=script,
                stimuli=stimuli,
                segment_window_seconds=segment_window_seconds,
            )
        )

    if len(log_capture.segment_assembly_ms) < segment_count:
        raise RuntimeError(
            "fixture benchmark did not capture enough segment assembly log lines "
            f"({len(log_capture.segment_assembly_ms)}/{segment_count})"
        )
    if len(log_capture.ml_inference_ms) < segment_count:
        raise RuntimeError(
            "fixture benchmark did not capture enough inference log lines "
            f"({len(log_capture.ml_inference_ms)}/{segment_count})"
        )
    if not log_capture.au12_ms:
        raise RuntimeError("fixture benchmark captured no AU12 timing log lines")
    if not bridge.payloads:
        raise RuntimeError("fixture benchmark dispatched no orchestrator payloads")

    return BenchmarkStats(
        segment_count=segment_count,
        segment_assembly_p50_ms=_percentile(log_capture.segment_assembly_ms[:segment_count], 50.0),
        segment_assembly_p95_ms=_percentile(log_capture.segment_assembly_ms[:segment_count], 95.0),
        ml_inference_p50_ms=_percentile(log_capture.ml_inference_ms[:segment_count], 50.0),
        ml_inference_p95_ms=_percentile(log_capture.ml_inference_ms[:segment_count], 95.0),
        au12_per_frame_p50_ms=_percentile(log_capture.au12_ms, 50.0),
    )


def _find_previous_fixture_row(rows: Sequence[tuple[str, ...]]) -> tuple[str, ...] | None:
    try:
        cycle_index = BASELINE_COLUMNS.index("Cycle / PR")
    except ValueError:
        return None
    previous: tuple[str, ...] | None = None
    for row in rows:
        if len(row) == len(BASELINE_COLUMNS) and row[cycle_index].strip() == FIXTURE_LABEL:
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
                f"above previous fixture row {previous:.3f}ms"
            )
    return tuple(warnings)


def _build_row_cells(
    stats: BenchmarkStats,
    warnings: Sequence[str],
    *,
    notes_suffix: str = "",
) -> tuple[str, ...]:
    notes_parts = [
        f"Offline replay fixture benchmark; N={stats.segment_count}",
        "operator-owned live row unchanged",
        f"warnings={len(warnings)}",
    ]
    if notes_suffix:
        notes_parts.append(notes_suffix)
    notes = "; ".join(notes_parts)
    return (
        datetime.now(tz=UTC).date().isoformat(),
        f"`{_git_sha_short()}`",
        FIXTURE_LABEL,
        _format_ms(stats.segment_assembly_p50_ms),
        _format_ms(stats.segment_assembly_p95_ms),
        _format_ms(stats.ml_inference_p50_ms),
        _format_ms(stats.ml_inference_p95_ms),
        _format_ms(stats.au12_per_frame_p50_ms),
        _format_ms(stats.comodulation_window_compute_ms),
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
    """Run the fixture benchmark, emit a markdown row, and optionally append it."""
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
        description="Run replay fixture timing benchmark and append performance baseline row.",
    )
    parser.add_argument("fixture_path", type=Path, help="Replay fixture directory")
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
    parser.add_argument("--verbose", action="store_true", help="Enable verbose harness logging")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
        stream=sys.stderr,
    )
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
        print(f"WARNING: observational fixture regression: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
