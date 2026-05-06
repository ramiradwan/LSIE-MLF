"""Lightweight retained-worker health endpoint for internal subsystem probes.

The API-side operator health aggregate reaches this endpoint in retained
server/cloud deployments. It is intentionally read-only at request time: the
handler reports the last bounded readiness signal written by the worker process
after its actual ``TranscriptionEngine`` has initialized a Whisper model,
without touching Redis, Celery inspection, the Persistent Store, or
operator-host resources.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from packages.ml_core.transcription import TranscriptionEngine

logger = logging.getLogger(__name__)

_HEALTH_HOST = "0.0.0.0"
_HEALTH_PORT = 8081
_HEALTH_PATH = "/healthz"

_WHISPER_READINESS_PATH = Path("/tmp/lsie_worker_whisper_readiness.json")
_WHISPER_READINESS_MODEL_READY = "model_ready"
_WHISPER_READINESS_NOT_INITIALIZED = "not_initialized"
_WHISPER_READINESS_ERROR = "error"


@dataclass(frozen=True)
class _WhisperReadiness:
    ready: bool
    readiness: str
    detail: str
    model_size: str
    device: str
    compute_type: str
    updated_at_utc: str | None = None


def record_whisper_model_ready(
    engine: Any,
    *,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> None:
    """Record that the worker process has an initialized Whisper model.

    This function is called by the actual worker task path after the
    ``TranscriptionEngine`` has successfully initialized its backing model. It
    never loads the model itself; it only persists a compact readiness marker
    for the sibling HTTP health process to read.
    """

    metadata = _whisper_metadata(engine)
    if getattr(engine, "_model", None) is None:
        _write_whisper_readiness(
            ready=False,
            readiness=_WHISPER_READINESS_ERROR,
            detail="TranscriptionEngine model is not loaded",
            metadata=metadata,
            clock=clock,
        )
        return

    _write_whisper_readiness(
        ready=True,
        readiness=_WHISPER_READINESS_MODEL_READY,
        detail="Whisper model loaded by worker TranscriptionEngine",
        metadata=metadata,
        clock=clock,
    )


def record_whisper_model_error(
    engine: Any,
    exc: BaseException,
    *,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> None:
    """Record a Whisper model initialization failure from the worker path."""

    _write_whisper_readiness(
        ready=False,
        readiness=_WHISPER_READINESS_ERROR,
        detail=f"Whisper model initialization failed: {_safe_detail(exc)}",
        metadata=_whisper_metadata(engine),
        clock=clock,
    )


def reset_whisper_readiness_state() -> None:
    """Clear stale readiness markers when the worker health process starts."""

    try:
        _WHISPER_READINESS_PATH.unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover - non-critical startup cleanup
        logger.warning("failed to clear Whisper readiness marker: %s", exc)


def build_worker_health_payload(
    *,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict[str, Any]:
    """Build the worker health payload without loading the Whisper model.

    Readiness is authoritative only when it comes from the readiness marker
    written by the worker process after its actual transcriber/model object is
    initialized. Missing, stale, or malformed markers are reported as not ready
    so ``/healthz`` remains bounded and non-mutating.
    """

    whisper = _read_whisper_readiness()
    whisper_payload: dict[str, Any] = {
        "ready": whisper.ready,
        "readiness": whisper.readiness,
        "detail": whisper.detail,
        "model_size": whisper.model_size,
        "device": whisper.device,
        "compute_type": whisper.compute_type,
    }
    if whisper.updated_at_utc is not None:
        whisper_payload["updated_at_utc"] = whisper.updated_at_utc

    return {
        "service": "worker",
        "status": "ok" if whisper.ready else "error",
        "generated_at_utc": clock().astimezone(UTC).isoformat(),
        "whisper": whisper_payload,
    }


class _WorkerHealthHandler(BaseHTTPRequestHandler):
    server_version = "LSIEWorkerHealth/1.0"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler override
        if urlparse(self.path).path != _HEALTH_PATH:
            self._write_json(404, {"detail": "not found"})
            return
        payload = build_worker_health_payload()
        self._write_json(200, payload)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib name
        logger.debug("worker health endpoint: " + format, *args)

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def run_health_server(host: str = _HEALTH_HOST, port: int = _HEALTH_PORT) -> None:
    """Run the worker-local health server until the process exits."""

    logging.basicConfig(level=logging.INFO)
    reset_whisper_readiness_state()
    server = ThreadingHTTPServer((host, port), _WorkerHealthHandler)
    logger.info("worker health endpoint listening on %s:%s", host, port)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()


def _read_whisper_readiness() -> _WhisperReadiness:
    default = _default_whisper_readiness()
    try:
        raw = _WHISPER_READINESS_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default
    except OSError as exc:
        return _default_whisper_readiness(
            readiness=_WHISPER_READINESS_ERROR,
            detail=f"Whisper readiness marker could not be read: {_safe_detail(exc)}",
        )

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return _default_whisper_readiness(
            readiness=_WHISPER_READINESS_ERROR,
            detail="Whisper readiness marker is not valid JSON",
        )
    if not isinstance(payload, dict):
        return _default_whisper_readiness(
            readiness=_WHISPER_READINESS_ERROR,
            detail="Whisper readiness marker is not a JSON object",
        )
    if not _readiness_marker_process_is_current(payload):
        return _default_whisper_readiness(
            detail="Whisper readiness marker is stale or not from a live worker process",
        )

    readiness = str(payload.get("readiness") or "").strip()
    ready = payload.get("ready") is True and readiness == _WHISPER_READINESS_MODEL_READY
    if ready:
        normalized_readiness = _WHISPER_READINESS_MODEL_READY
    else:
        normalized_readiness = readiness or _WHISPER_READINESS_ERROR

    detail = str(payload.get("detail") or "").strip()
    if not detail:
        detail = "Whisper model is ready" if ready else "Whisper model is not ready"

    return _WhisperReadiness(
        ready=ready,
        readiness=normalized_readiness,
        detail=detail,
        model_size=_text_or_default(payload.get("model_size"), default.model_size),
        device=_text_or_default(payload.get("device"), default.device),
        compute_type=_text_or_default(payload.get("compute_type"), default.compute_type),
        updated_at_utc=_optional_text(payload.get("updated_at_utc")),
    )


def _default_whisper_readiness(
    *,
    readiness: str = _WHISPER_READINESS_NOT_INITIALIZED,
    detail: str = "Whisper model has not been initialized by the worker process",
) -> _WhisperReadiness:
    metadata = _whisper_metadata(TranscriptionEngine())
    return _WhisperReadiness(
        ready=False,
        readiness=readiness,
        detail=detail,
        model_size=metadata["model_size"],
        device=metadata["device"],
        compute_type=metadata["compute_type"],
    )


def _write_whisper_readiness(
    *,
    ready: bool,
    readiness: str,
    detail: str,
    metadata: dict[str, str],
    clock: Callable[[], datetime],
) -> None:
    payload: dict[str, Any] = {
        "ready": ready,
        "readiness": readiness,
        "detail": detail,
        "updated_at_utc": clock().astimezone(UTC).isoformat(),
        **metadata,
        **_current_process_identity(),
    }
    try:
        _WHISPER_READINESS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = _WHISPER_READINESS_PATH.with_name(
            f"{_WHISPER_READINESS_PATH.name}.{os.getpid()}.tmp"
        )
        tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, _WHISPER_READINESS_PATH)
    except OSError as exc:  # pragma: no cover - diagnostic persistence only
        logger.warning("failed to write Whisper readiness marker: %s", exc)


def _whisper_metadata(engine: Any) -> dict[str, str]:
    return {
        "model_size": _text_or_default(getattr(engine, "model_size", None), "large-v3"),
        "device": _text_or_default(getattr(engine, "device", None), "cuda"),
        "compute_type": _text_or_default(getattr(engine, "compute_type", None), "int8"),
    }


def _current_process_identity() -> dict[str, Any]:
    pid = os.getpid()
    return {"pid": pid, "process_start_time": _process_start_time(pid)}


def _readiness_marker_process_is_current(payload: dict[str, Any]) -> bool:
    pid = payload.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    expected_start_time = payload.get("process_start_time")
    if not isinstance(expected_start_time, str) or not expected_start_time:
        return False
    return _process_start_time(pid) == expected_start_time


def _process_start_time(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        fields_after_comm = stat.rsplit(")", 1)[1].strip().split()
    except IndexError:
        return None
    # ``fields_after_comm`` starts at field 3 (state); field 22 is the process
    # start time in clock ticks, so the zero-based index is 19.
    if len(fields_after_comm) <= 19:
        return None
    return fields_after_comm[19]


def _text_or_default(value: Any, default: str) -> str:
    text = str(value or "").strip()
    return text or default


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _safe_detail(exc: BaseException) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    return text[:240]


if __name__ == "__main__":
    run_health_server()
