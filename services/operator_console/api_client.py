"""
Typed REST client for the LSIE-MLF Operator Console.

This module is the single network boundary for the console. Every
response is validated into a Pydantic DTO from
`packages.schemas.operator_console`; no `dict[str, Any]` crosses back
into the UI. `Transport` is a narrow Protocol so callers can swap in a
fake in unit tests without monkey-patching `urlopen`, and so we can
later replace `UrllibTransport` with `httpx` without rewriting the
typed method surface.

Errors are surfaced as `ApiError` — never `SystemExit` — because the
console is a long-running UI and a failed call must not kill the app.
`ApiError.retryable` is set by the transport based on whether the
failure is transient (connection refused, timeout, 5xx) or permanent
(4xx client errors, validation failures). The polling layer uses this
flag to decide between a silent retry and a surfaced alert.

Spec references:
  §4.C           — stimulus lifecycle / authoritative stimulus_time
  §4.E.1         — operator-facing aggregate endpoints
  §12            — error-handling matrix (retry semantics)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, TypeVar
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from uuid import UUID

from pydantic import BaseModel, TypeAdapter, ValidationError

from packages.schemas.experiments import (
    ExperimentAdminResponse,
    ExperimentArmAdminResponse,
    ExperimentArmCreateRequest,
    ExperimentArmDeleteResponse,
    ExperimentArmPatchRequest,
    ExperimentCreateRequest,
)
from packages.schemas.operator_console import (
    AlertEvent,
    EncounterSummary,
    ExperimentDetail,
    HealthSnapshot,
    OverviewSnapshot,
    SessionCreateRequest,
    SessionEndRequest,
    SessionLifecycleAccepted,
    SessionPhysiologySnapshot,
    SessionSummary,
    StimulusAccepted,
    StimulusRequest,
)

_ModelT = TypeVar("_ModelT", bound=BaseModel)
_ListItemT = TypeVar("_ListItemT")

# ----------------------------------------------------------------------
# ApiError — with endpoint + retryable
# ----------------------------------------------------------------------


@dataclass
class ApiError(Exception):
    """Raised when an API call fails. Wraps the underlying cause.

    `retryable` is a hint to the polling layer: True for transient
    failures (URLError, TimeoutError, 5xx HTTP) that should be retried
    on the next tick, False for permanent failures (4xx, validation
    errors) that should surface as alerts. `endpoint` is the path that
    failed so the UI can attribute an error to a specific card.
    """

    message: str
    endpoint: str | None = None
    status_code: int | None = None
    retryable: bool = False
    # pydantic validation failures keep the raw payload for debug logs
    payload_excerpt: str | None = field(default=None, repr=False)

    def __str__(self) -> str:
        prefix = f"{self.endpoint} " if self.endpoint else ""
        if self.status_code is not None:
            return f"API error {prefix}({self.status_code}): {self.message}"
        return f"API error {prefix}: {self.message}".rstrip()


# ----------------------------------------------------------------------
# Transport protocol + urllib impl
# ----------------------------------------------------------------------


class Transport(Protocol):
    """Narrow transport surface: bytes in, bytes out, typed errors.

    Implementations translate HTTP-level concerns into `ApiError` with
    the right `retryable` flag. Response bodies are returned as raw
    bytes; JSON decoding and DTO validation happen in `ApiClient`.
    """

    def request(
        self,
        method: str,
        url: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: float,
    ) -> bytes: ...


class UrllibTransport:
    """Standard-library transport — no new dependency on the operator host.

    `retryable=True` on URLError / TimeoutError / 5xx so the polling
    coordinator can silently retry. `retryable=False` on 4xx so the UI
    renders a permanent alert (typically a bug on our side or a missing
    resource).
    """

    def request(
        self,
        method: str,
        url: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: float,
    ) -> bytes:
        request = Request(url, method=method, data=body)
        request.add_header("Accept", "application/json")
        if body is not None:
            request.add_header("Content-Type", "application/json")
        for key, value in (headers or {}).items():
            request.add_header(key, value)
        try:
            with urlopen(request, timeout=timeout_s) as response:
                raw = response.read()
                if isinstance(raw, bytes):
                    return raw
                return bytes(raw)
        except HTTPError as exc:
            detail = _parse_error_body(exc)
            status = exc.code
            raise ApiError(
                message=detail,
                status_code=status,
                retryable=status >= 500,
            ) from exc
        except URLError as exc:
            raise ApiError(
                message=f"cannot reach {url}: {exc.reason}",
                retryable=True,
            ) from exc
        except TimeoutError as exc:
            raise ApiError(
                message=f"request to {url} timed out",
                retryable=True,
            ) from exc


def _parse_error_body(exc: HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8")
    except Exception:
        return str(exc)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(parsed, dict) and "detail" in parsed:
        detail = parsed["detail"]
        return str(detail) if not isinstance(detail, str) else detail
    return raw


# ----------------------------------------------------------------------
# ApiClient — typed methods returning validated DTOs
# ----------------------------------------------------------------------


# Pre-built type adapters for list-returning endpoints. Keeping them at
# module scope avoids re-parsing the schema on every call.
_SESSION_LIST_ADAPTER: TypeAdapter[list[SessionSummary]] = TypeAdapter(list[SessionSummary])
_ENCOUNTER_LIST_ADAPTER: TypeAdapter[list[EncounterSummary]] = TypeAdapter(list[EncounterSummary])
_ALERT_LIST_ADAPTER: TypeAdapter[list[AlertEvent]] = TypeAdapter(list[AlertEvent])


class ApiClient:
    """Thin typed wrapper over the API Server's operator/admin surfaces.

    All read methods return Pydantic DTOs. Write methods stay narrow and
    typed: operator stimulus requests go to `/api/v1/operator/*`, while
    experiment/arm management uses the additive `/api/v1/experiments/*`
    admin surface. A `Transport` can be injected for tests; the default
    is `UrllibTransport`.
    """

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 10.0,
        transport: Transport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._transport: Transport = transport if transport is not None else UrllibTransport()

    # ---- read endpoints ------------------------------------------------

    def get_overview(self) -> OverviewSnapshot:
        """`GET /api/v1/operator/overview` — six-card Overview payload."""
        return self._get_model("/api/v1/operator/overview", OverviewSnapshot)

    def list_sessions(self, *, limit: int = 50) -> list[SessionSummary]:
        """`GET /api/v1/operator/sessions?limit=N` — recent session cards."""
        path = f"/api/v1/operator/sessions?{urlencode({'limit': limit})}"
        return self._get_list(path, _SESSION_LIST_ADAPTER)

    def get_session(self, session_id: UUID | str) -> SessionSummary:
        """`GET /api/v1/operator/sessions/{id}`."""
        path = f"/api/v1/operator/sessions/{_path_segment(session_id)}"
        return self._get_model(path, SessionSummary)

    def list_session_encounters(
        self,
        session_id: UUID | str,
        *,
        limit: int = 100,
        before_utc: datetime | None = None,
    ) -> list[EncounterSummary]:
        """`GET /api/v1/operator/sessions/{id}/encounters?limit=N&before_utc=...`.

        `before_utc` must be UTC-aware — the API expects drift-corrected
        UTC, and the console must not leak a naive wall-clock timestamp
        past its boundary.
        """
        params: dict[str, str] = {"limit": str(limit)}
        if before_utc is not None:
            if before_utc.tzinfo is None or before_utc.utcoffset() is None:
                raise ValueError("before_utc must be UTC-aware")
            params["before_utc"] = before_utc.isoformat()
        path = (
            f"/api/v1/operator/sessions/{_path_segment(session_id)}/encounters?{urlencode(params)}"
        )
        return self._get_list(path, _ENCOUNTER_LIST_ADAPTER)

    def get_experiment_detail(self, experiment_id: str) -> ExperimentDetail:
        """`GET /api/v1/operator/experiments/{id}`."""
        path = f"/api/v1/operator/experiments/{_path_segment(experiment_id)}"
        return self._get_model(path, ExperimentDetail)

    def get_session_physiology(self, session_id: UUID | str) -> SessionPhysiologySnapshot:
        """`GET /api/v1/operator/sessions/{id}/physiology` — §4.E.2 + §7C."""
        path = f"/api/v1/operator/sessions/{_path_segment(session_id)}/physiology"
        return self._get_model(path, SessionPhysiologySnapshot)

    def get_health(self) -> HealthSnapshot:
        """`GET /api/v1/operator/health` — §12 subsystem rollup."""
        return self._get_model("/api/v1/operator/health", HealthSnapshot)

    def list_alerts(
        self,
        *,
        limit: int = 50,
        since_utc: datetime | None = None,
    ) -> list[AlertEvent]:
        """`GET /api/v1/operator/alerts?limit=N&since_utc=...` — attention queue."""
        params: dict[str, str] = {"limit": str(limit)}
        if since_utc is not None:
            if since_utc.tzinfo is None or since_utc.utcoffset() is None:
                raise ValueError("since_utc must be UTC-aware")
            params["since_utc"] = since_utc.isoformat()
        path = f"/api/v1/operator/alerts?{urlencode(params)}"
        return self._get_list(path, _ALERT_LIST_ADAPTER)

    # ---- write endpoints ----------------------------------------------

    def post_stimulus(
        self,
        session_id: UUID | str,
        request: StimulusRequest,
    ) -> StimulusAccepted:
        """`POST /api/v1/operator/sessions/{id}/stimulus` — §4.C.

        The orchestrator assigns the authoritative `stimulus_time` on
        receipt; this call simply hands the request off. Duplicate
        submissions are collapsed server-side via
        `StimulusRequest.client_action_id`.
        """
        path = f"/api/v1/operator/sessions/{_path_segment(session_id)}/stimulus"
        body = request.model_dump_json().encode("utf-8")
        return self._post_model(path, body, StimulusAccepted)

    def post_session_start(self, request: SessionCreateRequest) -> SessionLifecycleAccepted:
        """`POST /api/v1/sessions` — publish a session-start lifecycle intent."""
        body = request.model_dump_json().encode("utf-8")
        return self._post_model("/api/v1/sessions", body, SessionLifecycleAccepted)

    def post_session_end(
        self,
        session_id: UUID | str,
        request: SessionEndRequest,
    ) -> SessionLifecycleAccepted:
        """`POST /api/v1/sessions/{id}/end` — publish a session-end intent."""
        path = f"/api/v1/sessions/{_path_segment(session_id)}/end"
        body = request.model_dump_json().encode("utf-8")
        return self._post_model(path, body, SessionLifecycleAccepted)

    def create_experiment(self, request: ExperimentCreateRequest) -> ExperimentAdminResponse:
        """`POST /api/v1/experiments` — create experiment with Beta(1,1) arms."""
        body = request.model_dump_json().encode("utf-8")
        return self._post_model("/api/v1/experiments", body, ExperimentAdminResponse)

    def add_experiment_arm(
        self,
        experiment_id: str,
        request: ExperimentArmCreateRequest,
    ) -> ExperimentArmAdminResponse:
        """`POST /api/v1/experiments/{id}/arms` — add one Beta(1,1) arm."""
        path = f"/api/v1/experiments/{_path_segment(experiment_id)}/arms"
        body = request.model_dump_json().encode("utf-8")
        return self._post_model(path, body, ExperimentArmAdminResponse)

    def patch_experiment_arm(
        self,
        experiment_id: str,
        arm_id: str,
        request: ExperimentArmPatchRequest,
    ) -> ExperimentArmAdminResponse:
        """`PATCH /api/v1/experiments/{id}/arms/{arm}` — metadata/disable only."""
        path = f"/api/v1/experiments/{_path_segment(experiment_id)}/arms/{_path_segment(arm_id)}"
        body = request.model_dump_json().encode("utf-8")
        return self._patch_model(path, body, ExperimentArmAdminResponse)

    def delete_experiment_arm(
        self,
        experiment_id: str,
        arm_id: str,
    ) -> ExperimentArmDeleteResponse:
        """`DELETE /api/v1/experiments/{id}/arms/{arm}` — guarded arm removal."""
        path = f"/api/v1/experiments/{_path_segment(experiment_id)}/arms/{_path_segment(arm_id)}"
        return self._delete_model(path, ExperimentArmDeleteResponse)

    # ---- internal helpers ---------------------------------------------

    def _get_model(self, path: str, model: type[_ModelT]) -> _ModelT:
        raw = self._request("GET", path, body=None)
        return self._validate_model(path, raw, model)

    def _post_model(self, path: str, body: bytes, model: type[_ModelT]) -> _ModelT:
        raw = self._request("POST", path, body=body)
        return self._validate_model(path, raw, model)

    def _patch_model(self, path: str, body: bytes, model: type[_ModelT]) -> _ModelT:
        raw = self._request("PATCH", path, body=body)
        return self._validate_model(path, raw, model)

    def _delete_model(self, path: str, model: type[_ModelT]) -> _ModelT:
        raw = self._request("DELETE", path, body=None)
        return self._validate_model(path, raw, model)

    def _get_list(self, path: str, adapter: TypeAdapter[list[_ListItemT]]) -> list[_ListItemT]:
        raw = self._request("GET", path, body=None)
        return self._validate_adapter(path, raw, adapter)

    def _request(self, method: str, path: str, *, body: bytes | None) -> bytes:
        url = f"{self._base_url}{path}"
        try:
            return self._transport.request(method, url, body=body, timeout_s=self._timeout)
        except ApiError as exc:
            # Attach endpoint if the transport did not already.
            if exc.endpoint is None:
                exc.endpoint = path
            raise

    @staticmethod
    def _validate_model(path: str, raw: bytes, model: type[_ModelT]) -> _ModelT:
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(
                message=f"malformed JSON from {path}: {exc}",
                endpoint=path,
                retryable=False,
                payload_excerpt=_excerpt(raw),
            ) from exc
        try:
            return model.model_validate(payload)
        except ValidationError as exc:
            raise ApiError(
                message=f"response failed validation for {model.__name__}: {exc}",
                endpoint=path,
                retryable=False,
                payload_excerpt=_excerpt(raw),
            ) from exc

    @staticmethod
    def _validate_adapter(
        path: str, raw: bytes, adapter: TypeAdapter[list[_ListItemT]]
    ) -> list[_ListItemT]:
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(
                message=f"malformed JSON from {path}: {exc}",
                endpoint=path,
                retryable=False,
                payload_excerpt=_excerpt(raw),
            ) from exc
        try:
            return adapter.validate_python(payload)
        except ValidationError as exc:
            raise ApiError(
                message=f"response failed list validation: {exc}",
                endpoint=path,
                retryable=False,
                payload_excerpt=_excerpt(raw),
            ) from exc


def _path_segment(value: UUID | str) -> str:
    """Percent-encode a path segment so UUIDs and arbitrary experiment IDs
    both round-trip safely. Slashes are not allowed in segments."""
    return quote(str(value), safe="")


def _excerpt(raw: bytes, *, limit: int = 512) -> str:
    try:
        decoded = raw.decode("utf-8", errors="replace")
    except Exception:
        return ""
    if len(decoded) <= limit:
        return decoded
    return decoded[:limit] + "…"
