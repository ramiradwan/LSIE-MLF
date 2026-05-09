"""Unit tests for the worker-local health endpoint payload."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.ml_core.transcription import TranscriptionEngine
from services.worker import health

_NOW = datetime(2026, 4, 17, 12, 0, tzinfo=UTC)


def _use_readiness_marker(monkeypatch: Any, tmp_path: Path) -> Path:
    marker = tmp_path / "whisper-readiness.json"
    monkeypatch.delenv("LSIE_DEV_FORCE_CPU_SPEECH", raising=False)
    monkeypatch.setattr(health, "TranscriptionEngine", lambda: TranscriptionEngine(device="cuda"))
    monkeypatch.setattr(health, "_WHISPER_READINESS_PATH", marker)
    # CROSS-PLATFORM FIX: Prevent Windows/Mac from failing the Linux /proc/stat check
    monkeypatch.setattr(health, "_process_start_time", lambda pid: "mocked_test_start_time")
    return marker


def test_worker_health_payload_reports_uninitialized_whisper_model(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    _use_readiness_marker(monkeypatch, tmp_path)

    payload = health.build_worker_health_payload(clock=lambda: _NOW)

    assert payload["service"] == "worker"
    assert payload["status"] == "error"
    assert payload["generated_at_utc"] == _NOW.isoformat()
    assert payload["whisper"]["ready"] is False
    assert payload["whisper"]["readiness"] == "not_initialized"
    assert "not been initialized" in payload["whisper"]["detail"]
    assert payload["whisper"]["model_size"] == "large-v3"
    assert payload["whisper"]["device"] == "cuda"
    assert payload["whisper"]["compute_type"] == "int8"


def test_worker_health_payload_reports_authoritative_model_ready(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    _use_readiness_marker(monkeypatch, tmp_path)
    engine = TranscriptionEngine(device="cuda")
    engine._model = object()
    health.record_whisper_model_ready(engine, clock=lambda: _NOW)

    payload = health.build_worker_health_payload(clock=lambda: _NOW)

    assert payload["status"] == "ok"
    assert payload["whisper"]["ready"] is True
    assert payload["whisper"]["readiness"] == "model_ready"
    assert "loaded by worker TranscriptionEngine" in payload["whisper"]["detail"]
    assert payload["whisper"]["updated_at_utc"] == _NOW.isoformat()
    assert payload["whisper"]["model_size"] == "large-v3"
    assert payload["whisper"]["device"] == "cuda"
    assert payload["whisper"]["compute_type"] == "int8"


def test_worker_health_payload_reports_model_initialization_error(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    _use_readiness_marker(monkeypatch, tmp_path)
    engine = TranscriptionEngine(device="cuda")
    health.record_whisper_model_error(engine, RuntimeError("GPU OOM"), clock=lambda: _NOW)

    payload = health.build_worker_health_payload(clock=lambda: _NOW)

    assert payload["status"] == "error"
    assert payload["whisper"]["ready"] is False
    assert payload["whisper"]["readiness"] == "error"
    assert "GPU OOM" in payload["whisper"]["detail"]


def test_worker_health_payload_rejects_stale_model_ready_marker(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    _use_readiness_marker(monkeypatch, tmp_path)
    engine = TranscriptionEngine(device="cuda")
    engine._model = object()
    health.record_whisper_model_ready(engine, clock=lambda: _NOW)

    # Override the stable mock with a "stale" string to verify the health endpoint rejects it
    monkeypatch.setattr(health, "_process_start_time", lambda _pid: "stale")

    payload = health.build_worker_health_payload(clock=lambda: _NOW)

    assert payload["status"] == "error"
    assert payload["whisper"]["ready"] is False
    assert payload["whisper"]["readiness"] == "not_initialized"
    assert "stale" in payload["whisper"]["detail"]
