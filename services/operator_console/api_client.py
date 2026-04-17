"""Typed REST client for the LSIE-MLF API Server.

Uses the standard library only (``urllib``) to match the operator CLI and
avoid introducing a new HTTP dependency on the operator host. Every
method returns a parsed JSON payload with a narrow type; callers never
see raw ``Any`` structures beyond the immediate response boundary.

Errors are surfaced as ``ApiError`` — never ``SystemExit`` — because the
console is a long-running UI and a failed call must not kill the app.
The calling worker catches the error and reports it through a Qt signal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ApiError(Exception):
    """Raised when an API call fails. Wraps the underlying cause."""

    message: str
    status_code: int | None = None

    def __str__(self) -> str:
        if self.status_code is not None:
            return f"API error ({self.status_code}): {self.message}"
        return f"API error: {self.message}"


class ApiClient:
    """Thin typed wrapper over the API Server.

    A new ``ApiClient`` can be instantiated per worker thread — ``urlopen``
    is stateless, so there is no shared connection pool to manage. If we
    later swap to ``httpx``, the class surface stays the same.
    """

    def __init__(self, base_url: str, timeout_seconds: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    # -- low-level -------------------------------------------------------

    def _request(self, method: str, path: str, body: bytes | None = None) -> Any:
        url = f"{self._base_url}{path}"
        request = Request(url, method=method, data=body)
        request.add_header("Accept", "application/json")
        if body is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = self._parse_error_body(exc)
            raise ApiError(message=detail, status_code=exc.code) from exc
        except URLError as exc:
            raise ApiError(message=f"cannot reach {self._base_url}: {exc.reason}") from exc
        except (TimeoutError, json.JSONDecodeError) as exc:
            raise ApiError(message=str(exc)) from exc

    @staticmethod
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

    # -- typed endpoints -------------------------------------------------

    def list_sessions(self) -> list[dict[str, Any]]:
        result = self._request("GET", "/api/v1/sessions")
        if not isinstance(result, list):
            raise ApiError(message="expected list of sessions")
        return [item for item in result if isinstance(item, dict)]

    def session_status(self, session_id: str) -> dict[str, Any]:
        result = self._request("GET", f"/api/v1/sessions/{session_id}")
        if not isinstance(result, dict):
            raise ApiError(message="expected session object")
        return result

    def list_experiments(self) -> list[dict[str, Any]]:
        result = self._request("GET", "/api/v1/experiments")
        if not isinstance(result, list):
            raise ApiError(message="expected list of experiments")
        return [item for item in result if isinstance(item, dict)]

    def experiment_show(self, experiment_id: str) -> dict[str, Any]:
        result = self._request("GET", f"/api/v1/experiments/{experiment_id}")
        if not isinstance(result, dict):
            raise ApiError(message="expected experiment object")
        return result

    def encounter_summary(self, experiment_id: str) -> list[dict[str, Any]]:
        result = self._request("GET", f"/api/v1/encounters/{experiment_id}/summary")
        if not isinstance(result, list):
            raise ApiError(message="expected list of arm summaries")
        return [item for item in result if isinstance(item, dict)]

    def physiology_snapshot(
        self, session_id: str, *, series: bool = False, limit: int = 100
    ) -> list[dict[str, Any]]:
        suffix = f"?series=true&limit={limit}" if series else ""
        result = self._request("GET", f"/api/v1/physiology/{session_id}{suffix}")
        if not isinstance(result, list):
            raise ApiError(message="expected list of physiology snapshots")
        return [item for item in result if isinstance(item, dict)]

    def comodulation_history(self, session_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        result = self._request("GET", f"/api/v1/comodulation/{session_id}?limit={limit}")
        if not isinstance(result, list):
            raise ApiError(message="expected list of comodulation points")
        return [item for item in result if isinstance(item, dict)]

    def inject_stimulus(self) -> dict[str, Any]:
        result = self._request("POST", "/api/v1/stimulus", body=b"")
        if not isinstance(result, dict):
            raise ApiError(message="expected stimulus response object")
        return result
