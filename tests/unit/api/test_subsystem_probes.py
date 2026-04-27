"""Unit tests for bounded operator subsystem probes."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from typing import Any

import pytest

from packages.schemas.operator_console import HealthProbeState
from services.api.services import subsystem_probes


class _Cursor:
    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def execute(self, _sql: str) -> None:
        return None

    def fetchone(self) -> tuple[int]:
        return (1,)


class _Conn:
    def cursor(self) -> _Cursor:
        return _Cursor()


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self, _size: int = -1) -> bytes:
        return self._raw


def _clock() -> datetime:
    return datetime(2026, 4, 17, 12, 0, tzinfo=UTC)


def _disable_worker_health(monkeypatch: Any) -> None:
    def _raise_not_configured() -> str:
        raise subsystem_probes._NotConfiguredError("worker health endpoint unresolved")

    monkeypatch.setattr(subsystem_probes, "_worker_health_url", _raise_not_configured)


def test_collect_subsystem_probes_reports_missing_config_as_not_configured(
    monkeypatch: Any,
) -> None:
    for name in (
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
    ):
        monkeypatch.delenv(name, raising=False)
    _disable_worker_health(monkeypatch)

    results = asyncio.run(
        subsystem_probes.collect_subsystem_probes(
            get_conn=lambda: _Conn(),
            put_conn=lambda _conn: None,
            redis_factory=None,
            clock=_clock,
        )
    )

    by_key = {result.subsystem_key: result for result in results}
    assert by_key["postgres"].state is HealthProbeState.OK
    assert by_key["azure_openai"].state is HealthProbeState.NOT_CONFIGURED
    assert by_key["redis"].state is HealthProbeState.NOT_CONFIGURED
    assert by_key["whisper_worker"].state is HealthProbeState.NOT_CONFIGURED
    assert by_key["orchestrator"].state is HealthProbeState.NOT_CONFIGURED


def test_whisper_probe_uses_worker_health_endpoint(monkeypatch: Any) -> None:
    for name in (
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        subsystem_probes,
        "_worker_health_url",
        lambda: "http://worker:8081/healthz",
    )

    def _open_worker_health(_request: Any, *, timeout: float) -> _Response:
        assert timeout == subsystem_probes.PROBE_TIMEOUT_S
        return _Response(
            {
                "service": "worker",
                "status": "ok",
                "whisper": {
                    "ready": True,
                    "readiness": "model_ready",
                    "model_size": "large-v3",
                    "device": "cuda",
                    "compute_type": "int8",
                    "detail": "Whisper model loaded by worker TranscriptionEngine",
                },
            }
        )

    monkeypatch.setattr(subsystem_probes, "_open_worker_health_request", _open_worker_health)

    results = asyncio.run(
        subsystem_probes.collect_subsystem_probes(
            get_conn=lambda: _Conn(),
            put_conn=lambda _conn: None,
            redis_factory=None,
            clock=_clock,
        )
    )

    by_key = {result.subsystem_key: result for result in results}
    assert by_key["whisper_worker"].state is HealthProbeState.OK
    assert "large-v3" in (by_key["whisper_worker"].detail or "")


def test_whisper_probe_rejects_importability_only_payload(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        subsystem_probes,
        "_worker_health_url",
        lambda: "http://worker:8081/healthz",
    )

    def _open_worker_health(_request: Any, *, timeout: float) -> _Response:
        assert timeout == subsystem_probes.PROBE_TIMEOUT_S
        return _Response(
            {
                "service": "worker",
                "status": "ok",
                "whisper": {
                    "ready": True,
                    "model_size": "large-v3",
                    "device": "cuda",
                    "compute_type": "int8",
                    "detail": "faster-whisper runtime importable",
                },
            }
        )

    monkeypatch.setattr(subsystem_probes, "_open_worker_health_request", _open_worker_health)

    with pytest.raises(subsystem_probes._UnknownError, match="authoritative Whisper"):
        subsystem_probes._probe_whisper_worker()


def test_whisper_probe_reports_worker_not_ready_as_error(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        subsystem_probes,
        "_worker_health_url",
        lambda: "http://worker:8081/healthz",
    )

    def _open_worker_health(_request: Any, *, timeout: float) -> _Response:
        assert timeout == subsystem_probes.PROBE_TIMEOUT_S
        return _Response(
            {
                "service": "worker",
                "status": "error",
                "whisper": {
                    "ready": False,
                    "readiness": "not_initialized",
                    "model_size": "large-v3",
                    "device": "cuda",
                    "compute_type": "int8",
                    "detail": "Whisper model has not been initialized by the worker process",
                },
            }
        )

    monkeypatch.setattr(subsystem_probes, "_open_worker_health_request", _open_worker_health)

    with pytest.raises(RuntimeError, match="not been initialized"):
        subsystem_probes._probe_whisper_worker()


def test_per_probe_timeout_returns_timeout_without_blocking(monkeypatch: Any) -> None:
    for name in (
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
    ):
        monkeypatch.delenv(name, raising=False)
    _disable_worker_health(monkeypatch)

    def slow_get_conn() -> _Conn:
        time.sleep(0.05)
        return _Conn()

    started = time.perf_counter()
    results = asyncio.run(
        subsystem_probes.collect_subsystem_probes(
            get_conn=slow_get_conn,
            put_conn=lambda _conn: None,
            redis_factory=None,
            clock=_clock,
            per_probe_timeout_s=0.001,
            total_timeout_s=0.2,
        )
    )

    elapsed = time.perf_counter() - started
    postgres = next(result for result in results if result.subsystem_key == "postgres")
    assert postgres.state is HealthProbeState.TIMEOUT
    assert elapsed < 0.2
