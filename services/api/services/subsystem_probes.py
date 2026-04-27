"""Bounded, read-only subsystem connectivity probes for operator health.

The existing §12 health rows are freshness rollups derived from persisted
pipeline writes. These helpers are active diagnostics: each probe performs one
small read-only check, is bounded by its own timeout, and reports a neutral
``not_configured`` state when the deployment does not provide the required
configuration. Probe results are appended to ``HealthSnapshot`` only; they do
not synthesize §12 alerts or mutate the alert pipeline.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import ProxyHandler, Request, build_opener, urlopen

from packages.schemas.operator_console import HealthProbeState

PROBE_TIMEOUT_S: float = 1.0
PROBE_COLLECTION_TIMEOUT_S: float = 3.0

_ORCHESTRATOR_HEARTBEAT_KEY = "operator:orchestrator:heartbeat"
_ORCHESTRATOR_HEARTBEAT_STALE_AFTER_S = 10.0
_WORKER_HEALTH_HOST = "worker"
_WORKER_HEALTH_PORT = 8081
_WORKER_HEALTH_PATH = "/healthz"


class RedisProbeClientLike(Protocol):
    """Minimal Redis surface required by the probe layer."""

    def ping(self) -> bool: ...

    def get(self, name: str) -> Any: ...

    def close(self) -> None: ...


RedisFactory = Callable[[], RedisProbeClientLike]
GetConnection = Callable[[], Any]
PutConnection = Callable[[Any], None]
Clock = Callable[[], datetime]


@dataclass(frozen=True)
class ProbeResult:
    """Small probe result shape mapped downstream to the public DTO."""

    subsystem_key: str
    label: str
    state: HealthProbeState
    latency_ms: float | None = None
    detail: str | None = None


@dataclass(frozen=True)
class _ProbeSpec:
    subsystem_key: str
    label: str
    run: Callable[[], Any]


class _NotConfiguredError(RuntimeError):
    """Raised internally when a probe lacks required configuration."""


class _UnknownError(RuntimeError):
    """Raised internally when a diagnostic signal has not been emitted yet."""


async def collect_subsystem_probes(
    *,
    get_conn: GetConnection,
    put_conn: PutConnection,
    redis_factory: RedisFactory | None,
    clock: Clock,
    per_probe_timeout_s: float = PROBE_TIMEOUT_S,
    total_timeout_s: float = PROBE_COLLECTION_TIMEOUT_S,
) -> list[ProbeResult]:
    """Run all subsystem probes concurrently under a total timeout envelope."""

    specs = _probe_specs(
        get_conn=get_conn,
        put_conn=put_conn,
        redis_factory=redis_factory,
        clock=clock,
    )
    tasks = [asyncio.create_task(_run_probe(spec, timeout_s=per_probe_timeout_s)) for spec in specs]
    try:
        return list(await asyncio.wait_for(asyncio.gather(*tasks), timeout=total_timeout_s))
    except TimeoutError:
        results: list[ProbeResult] = []
        for spec, task in zip(specs, tasks, strict=True):
            if task.done() and not task.cancelled():
                try:
                    results.append(task.result())
                    continue
                except Exception as exc:  # pragma: no cover - defensive gather fallback
                    results.append(_error_result(spec, exc, latency_ms=None))
                    continue
            task.cancel()
            results.append(
                ProbeResult(
                    subsystem_key=spec.subsystem_key,
                    label=spec.label,
                    state=HealthProbeState.TIMEOUT,
                    latency_ms=total_timeout_s * 1000.0,
                    detail=f"probe collection timed out after {total_timeout_s:.1f}s",
                )
            )
        return results


def _probe_specs(
    *,
    get_conn: GetConnection,
    put_conn: PutConnection,
    redis_factory: RedisFactory | None,
    clock: Clock,
) -> list[_ProbeSpec]:
    return [
        _ProbeSpec(
            subsystem_key="postgres",
            label="Postgres",
            run=lambda: _probe_postgres(get_conn=get_conn, put_conn=put_conn),
        ),
        _ProbeSpec(
            subsystem_key="redis",
            label="Redis Broker",
            run=lambda: _probe_redis(redis_factory=redis_factory),
        ),
        _ProbeSpec(
            subsystem_key="azure_openai",
            label="Azure OpenAI",
            run=_probe_azure_openai,
        ),
        _ProbeSpec(
            subsystem_key="whisper_worker",
            label="Whisper Worker",
            run=_probe_whisper_worker,
        ),
        _ProbeSpec(
            subsystem_key="orchestrator",
            label="Orchestrator Heartbeat",
            run=lambda: _probe_orchestrator_heartbeat(redis_factory=redis_factory, clock=clock),
        ),
    ]


async def _run_probe(spec: _ProbeSpec, *, timeout_s: float) -> ProbeResult:
    started = time.perf_counter()
    try:
        detail = await asyncio.wait_for(asyncio.to_thread(spec.run), timeout=timeout_s)
    except TimeoutError:
        return ProbeResult(
            subsystem_key=spec.subsystem_key,
            label=spec.label,
            state=HealthProbeState.TIMEOUT,
            latency_ms=timeout_s * 1000.0,
            detail=f"probe timed out after {timeout_s:.1f}s",
        )
    except _NotConfiguredError as exc:
        return ProbeResult(
            subsystem_key=spec.subsystem_key,
            label=spec.label,
            state=HealthProbeState.NOT_CONFIGURED,
            latency_ms=_elapsed_ms(started),
            detail=str(exc),
        )
    except _UnknownError as exc:
        return ProbeResult(
            subsystem_key=spec.subsystem_key,
            label=spec.label,
            state=HealthProbeState.UNKNOWN,
            latency_ms=_elapsed_ms(started),
            detail=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics must never propagate
        return _error_result(spec, exc, latency_ms=_elapsed_ms(started))
    return ProbeResult(
        subsystem_key=spec.subsystem_key,
        label=spec.label,
        state=HealthProbeState.OK,
        latency_ms=_elapsed_ms(started),
        detail=str(detail) if detail else None,
    )


def _error_result(spec: _ProbeSpec, exc: BaseException, *, latency_ms: float | None) -> ProbeResult:
    return ProbeResult(
        subsystem_key=spec.subsystem_key,
        label=spec.label,
        state=HealthProbeState.ERROR,
        latency_ms=latency_ms,
        detail=_safe_exception_detail(exc),
    )


def _probe_postgres(*, get_conn: GetConnection, put_conn: PutConnection) -> str:
    """Read-only Persistent Store connectivity check: ``SELECT 1``."""

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            row = cur.fetchone()
    finally:
        put_conn(conn)
    if row is None or row[0] != 1:
        raise RuntimeError("SELECT 1 returned an unexpected result")
    return "SELECT 1 succeeded"


def _probe_redis(*, redis_factory: RedisFactory | None) -> str:
    """Read-only Redis connectivity check: ``PING``."""

    client = _require_redis(redis_factory)
    try:
        pong = client.ping()
    finally:
        _close_redis(client)
    if pong is not True:
        raise RuntimeError("PING returned a non-OK response")
    return "PING succeeded"


def _probe_azure_openai() -> str:
    """Read-only Azure OpenAI deployment diagnostic using existing env config."""

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01").strip()
    missing = [
        name
        for name, value in (
            ("AZURE_OPENAI_ENDPOINT", endpoint),
            ("AZURE_OPENAI_API_KEY", api_key),
            ("AZURE_OPENAI_DEPLOYMENT", deployment),
        )
        if not value
    ]
    if missing:
        raise _NotConfiguredError(f"missing {', '.join(missing)}")

    url = _azure_deployment_url(endpoint, deployment, api_version)
    request = Request(url, method="GET")
    request.add_header("Accept", "application/json")
    request.add_header("api-key", api_key)
    try:
        with urlopen(request, timeout=PROBE_TIMEOUT_S) as response:
            response.read(256)
    except HTTPError as exc:
        raise RuntimeError(f"Azure OpenAI diagnostic HTTP {exc.code}") from exc
    except TimeoutError as exc:
        raise TimeoutError("Azure OpenAI diagnostic timed out") from exc
    except URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(f"Azure OpenAI diagnostic failed: {reason}") from exc
    return "deployment endpoint reachable"


def _probe_whisper_worker() -> str:
    """Read-only worker health endpoint check for Whisper readiness."""

    url = _worker_health_url()
    request = Request(url, method="GET")
    request.add_header("Accept", "application/json")
    try:
        with _open_worker_health_request(request, timeout=PROBE_TIMEOUT_S) as response:
            payload = _json_object(response.read(4096), context="worker health payload")
    except HTTPError as exc:
        detail = _http_error_excerpt(exc)
        raise RuntimeError(f"worker health HTTP {exc.code}: {detail}") from exc
    except TimeoutError as exc:
        raise TimeoutError("worker health endpoint timed out") from exc
    except URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, socket.gaierror):
            raise _NotConfiguredError("worker health endpoint is not resolvable") from exc
        raise RuntimeError(f"worker health endpoint failed: {reason}") from exc

    whisper = payload.get("whisper")
    if not isinstance(whisper, dict):
        raise _UnknownError("worker health endpoint did not report Whisper readiness")
    ready = whisper.get("ready")
    readiness = str(whisper.get("readiness") or "").strip()
    detail = str(whisper.get("detail") or "").strip()
    if ready is not True or readiness != "model_ready":
        if ready is True and not readiness:
            raise _UnknownError(
                "worker health endpoint did not report authoritative Whisper model readiness"
            )
        reason = detail or "Whisper model readiness is false"
        raise RuntimeError(reason)

    model = str(whisper.get("model_size") or "unknown")
    device = str(whisper.get("device") or "unknown")
    compute_type = str(whisper.get("compute_type") or "unknown")
    return f"worker health reachable; {model} model_ready on {device} ({compute_type})"


def _probe_orchestrator_heartbeat(*, redis_factory: RedisFactory | None, clock: Clock) -> str:
    """Read-only freshness check for the orchestrator heartbeat Redis key."""

    client = _require_redis(redis_factory)
    try:
        raw = client.get(_ORCHESTRATOR_HEARTBEAT_KEY)
    finally:
        _close_redis(client)
    if raw is None:
        raise RuntimeError("no orchestrator heartbeat observed")
    payload = _json_object(raw, context="orchestrator heartbeat")
    timestamp = _parse_utc_timestamp(payload.get("timestamp_utc"))
    if timestamp is None:
        raise RuntimeError("orchestrator heartbeat missing timestamp_utc")
    age = max(0.0, (clock() - timestamp).total_seconds())
    if age > _ORCHESTRATOR_HEARTBEAT_STALE_AFTER_S:
        raise RuntimeError(f"orchestrator heartbeat stale ({age:.1f}s old)")
    session_id = str(payload.get("session_id") or "unknown session")
    return f"heartbeat fresh ({age:.1f}s old; {session_id})"


def _require_redis(redis_factory: RedisFactory | None) -> RedisProbeClientLike:
    if redis_factory is None:
        raise _NotConfiguredError("REDIS_URL client is not configured for this service")
    return redis_factory()


def _close_redis(client: RedisProbeClientLike) -> None:
    try:
        client.close()
    except Exception:  # noqa: BLE001 - close failure should not poison diagnostics
        return


def _worker_health_url() -> str:
    try:
        socket.getaddrinfo(
            _WORKER_HEALTH_HOST,
            _WORKER_HEALTH_PORT,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise _NotConfiguredError(
            "worker health endpoint is not resolvable via Docker service discovery"
        ) from exc
    return f"http://{_WORKER_HEALTH_HOST}:{_WORKER_HEALTH_PORT}{_WORKER_HEALTH_PATH}"


def _open_worker_health_request(request: Request, *, timeout: float) -> Any:
    # Internal service discovery should not be routed through operator/API host
    # HTTP proxy settings; the worker lives on the Docker bridge network.
    opener = build_opener(ProxyHandler({}))
    return opener.open(request, timeout=timeout)


def _azure_deployment_url(endpoint: str, deployment: str, api_version: str) -> str:
    base = endpoint.rstrip("/")
    deployment_segment = quote(deployment, safe="")
    query = urlencode({"api-version": api_version})
    return f"{base}/openai/deployments/{deployment_segment}?{query}"


def _json_object(raw: Any, *, context: str) -> dict[str, Any]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        payload = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{context} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} is not a JSON object")
    return payload


def _parse_utc_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _http_error_excerpt(exc: HTTPError) -> str:
    try:
        raw = exc.read(240)
    except Exception:
        return str(exc)
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")[:240]
    return str(raw)[:240]


def _safe_exception_detail(exc: BaseException) -> str:
    text = str(exc).strip()
    if not text:
        text = exc.__class__.__name__
    # Keep diagnostics compact and never echo request headers/secrets.
    return text[:240]


def _elapsed_ms(started: float) -> float:
    return max(0.0, (time.perf_counter() - started) * 1000.0)
