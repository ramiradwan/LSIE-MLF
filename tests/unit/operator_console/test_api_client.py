"""Tests for the Operator Console REST client — Phase 3.

The client validates every response into a Phase-1 DTO and surfaces
errors through `ApiError` with an `endpoint` and `retryable` flag. The
tests below exercise:

  * happy-path DTO validation success
  * URL + querystring assembly (encounters, alerts, stimulus)
  * `ApiError.retryable` semantics: True for URLError/Timeout/5xx,
    False for 4xx + JSON parse + DTO validation
  * `post_stimulus()` serialises `StimulusRequest` to JSON body

Two transport strategies are used: a `FakeTransport` for DTO-layer
tests, and `urlopen`-patching for `UrllibTransport` integration tests.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.message import Message
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError
from uuid import UUID, uuid4

import pytest

from packages.schemas.operator_console import (
    AlertEvent,
    AlertKind,
    AlertSeverity,
    EncounterState,
    HealthSnapshot,
    HealthState,
    OverviewSnapshot,
    SessionSummary,
    StimulusAccepted,
    StimulusRequest,
)
from services.operator_console.api_client import (
    ApiClient,
    ApiError,
    UrllibTransport,
)

# ----------------------------------------------------------------------
# Fake transport — captures calls and returns canned bytes
# ----------------------------------------------------------------------


@dataclass
class _FakeCall:
    method: str
    url: str
    body: bytes | None
    timeout_s: float


@dataclass
class FakeTransport:
    """Records calls and returns canned responses.

    `responses` is a FIFO queue of either raw bytes (success) or an
    `ApiError` to raise.
    """

    responses: list[bytes | ApiError] = field(default_factory=list)
    calls: list[_FakeCall] = field(default_factory=list)

    def enqueue_json(self, payload: Any) -> None:
        self.responses.append(json.dumps(payload).encode("utf-8"))

    def enqueue_raw(self, raw: bytes) -> None:
        self.responses.append(raw)

    def enqueue_error(self, error: ApiError) -> None:
        self.responses.append(error)

    def request(
        self,
        method: str,
        url: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: float,
    ) -> bytes:
        del headers
        self.calls.append(_FakeCall(method=method, url=url, body=body, timeout_s=timeout_s))
        if not self.responses:
            raise AssertionError("FakeTransport got an unexpected call")
        item = self.responses.pop(0)
        if isinstance(item, ApiError):
            raise item
        return item


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def _http_error(code: int, detail: str) -> HTTPError:
    payload = io.BytesIO(json.dumps({"detail": detail}).encode("utf-8"))
    return HTTPError("http://api.test", code, "error", Message(), payload)


# ----------------------------------------------------------------------
# DTO validation — happy path
# ----------------------------------------------------------------------


class TestDtoValidation:
    def test_get_overview_returns_validated_dto(self) -> None:
        transport = FakeTransport()
        transport.enqueue_json(
            {
                "generated_at_utc": _utc(2026, 4, 17, 12, 0).isoformat(),
                "alerts": [],
            }
        )
        client = ApiClient("http://api.test", transport=transport)
        snap = client.get_overview()
        assert isinstance(snap, OverviewSnapshot)
        assert transport.calls[0].method == "GET"
        assert transport.calls[0].url.endswith("/api/v1/operator/overview")

    def test_list_sessions_returns_dto_list(self) -> None:
        transport = FakeTransport()
        session_id = str(uuid4())
        transport.enqueue_json(
            [
                {
                    "session_id": session_id,
                    "status": "live",
                    "started_at_utc": _utc(2026, 4, 17, 10, 0).isoformat(),
                }
            ]
        )
        client = ApiClient("http://api.test", transport=transport)
        sessions = client.list_sessions(limit=25)
        assert len(sessions) == 1
        assert isinstance(sessions[0], SessionSummary)
        assert str(sessions[0].session_id) == session_id
        assert "limit=25" in transport.calls[0].url

    def test_invalid_shape_raises_non_retryable_api_error(self) -> None:
        transport = FakeTransport()
        # Missing required field `generated_at_utc`.
        transport.enqueue_json({"alerts": []})
        client = ApiClient("http://api.test", transport=transport)
        with pytest.raises(ApiError) as info:
            client.get_overview()
        assert info.value.endpoint is not None
        assert "/api/v1/operator/overview" in info.value.endpoint
        assert info.value.retryable is False

    def test_malformed_json_raises_non_retryable(self) -> None:
        transport = FakeTransport()
        transport.enqueue_raw(b"<<not-json>>")
        client = ApiClient("http://api.test", transport=transport)
        with pytest.raises(ApiError) as info:
            client.get_health()
        assert info.value.retryable is False
        assert info.value.endpoint is not None
        assert "/api/v1/operator/health" in info.value.endpoint


# ----------------------------------------------------------------------
# Querystring assembly
# ----------------------------------------------------------------------


class TestQuerystringAssembly:
    def test_list_session_encounters_encodes_before_utc(self) -> None:
        transport = FakeTransport()
        transport.enqueue_json([])
        client = ApiClient("http://api.test", transport=transport)
        session_id = UUID("00000000-0000-0000-0000-000000000001")
        cutoff = _utc(2026, 4, 17, 11, 30)
        client.list_session_encounters(session_id, limit=20, before_utc=cutoff)
        url = transport.calls[0].url
        assert f"/sessions/{session_id}/encounters" in url
        assert "limit=20" in url
        # urlencode quotes `+` and colons in ISO timestamps
        assert "before_utc=" in url
        assert "2026-04-17" in url

    def test_list_session_encounters_rejects_naive_before_utc(self) -> None:
        transport = FakeTransport()
        client = ApiClient("http://api.test", transport=transport)
        with pytest.raises(ValueError, match="UTC-aware"):
            client.list_session_encounters(uuid4(), before_utc=datetime(2026, 4, 17, 11, 30))

    def test_list_alerts_encodes_since_utc(self) -> None:
        transport = FakeTransport()
        transport.enqueue_json([])
        client = ApiClient("http://api.test", transport=transport)
        client.list_alerts(limit=10, since_utc=_utc(2026, 4, 17, 9, 0))
        url = transport.calls[0].url
        assert "/api/v1/operator/alerts?" in url
        assert "limit=10" in url
        assert "since_utc=" in url

    def test_list_alerts_rejects_naive_since_utc(self) -> None:
        client = ApiClient("http://api.test", transport=FakeTransport())
        with pytest.raises(ValueError, match="UTC-aware"):
            client.list_alerts(since_utc=datetime(2026, 4, 17))


# ----------------------------------------------------------------------
# post_stimulus
# ----------------------------------------------------------------------


class TestPostStimulus:
    def test_post_stimulus_serialises_body_and_validates_response(self) -> None:
        transport = FakeTransport()
        session_id = UUID("00000000-0000-0000-0000-000000000002")
        action_id = UUID("11111111-1111-1111-1111-111111111111")
        transport.enqueue_json(
            {
                "session_id": str(session_id),
                "client_action_id": str(action_id),
                "accepted": True,
                "received_at_utc": _utc(2026, 4, 17, 12, 5).isoformat(),
            }
        )
        client = ApiClient("http://api.test", transport=transport)
        request = StimulusRequest(client_action_id=action_id, operator_note="hello")
        accepted = client.post_stimulus(session_id, request)
        assert isinstance(accepted, StimulusAccepted)
        assert accepted.accepted is True
        call = transport.calls[0]
        assert call.method == "POST"
        assert f"/sessions/{session_id}/stimulus" in call.url
        assert call.body is not None
        body = json.loads(call.body.decode("utf-8"))
        assert body["client_action_id"] == str(action_id)
        assert body["operator_note"] == "hello"


# ----------------------------------------------------------------------
# ApiError.retryable semantics via UrllibTransport
# ----------------------------------------------------------------------


class TestRetryableSemantics:
    @patch("services.operator_console.api_client.urlopen")
    def test_url_error_is_retryable(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = URLError("connection refused")
        transport = UrllibTransport()
        with pytest.raises(ApiError) as info:
            transport.request("GET", "http://api.test/x", timeout_s=1.0)
        assert info.value.retryable is True
        assert info.value.status_code is None

    @patch("services.operator_console.api_client.urlopen")
    def test_timeout_error_is_retryable(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = TimeoutError("slow")
        transport = UrllibTransport()
        with pytest.raises(ApiError) as info:
            transport.request("GET", "http://api.test/x", timeout_s=1.0)
        assert info.value.retryable is True

    @patch("services.operator_console.api_client.urlopen")
    def test_5xx_is_retryable(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _http_error(503, "down")
        transport = UrllibTransport()
        with pytest.raises(ApiError) as info:
            transport.request("GET", "http://api.test/x", timeout_s=1.0)
        assert info.value.status_code == 503
        assert info.value.retryable is True

    @patch("services.operator_console.api_client.urlopen")
    def test_4xx_is_not_retryable(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _http_error(404, "missing")
        transport = UrllibTransport()
        with pytest.raises(ApiError) as info:
            transport.request("GET", "http://api.test/x", timeout_s=1.0)
        assert info.value.status_code == 404
        assert info.value.retryable is False


# ----------------------------------------------------------------------
# Endpoint attribution — ApiError carries the failing path
# ----------------------------------------------------------------------


class TestEndpointAttribution:
    def test_transport_error_gets_endpoint_attached_by_client(self) -> None:
        transport = FakeTransport()
        transport.enqueue_error(ApiError(message="boom", retryable=True))
        client = ApiClient("http://api.test", transport=transport)
        with pytest.raises(ApiError) as info:
            client.get_health()
        assert info.value.endpoint is not None
        assert "/api/v1/operator/health" in info.value.endpoint
        assert info.value.retryable is True

    def test_validation_error_marks_endpoint(self) -> None:
        transport = FakeTransport()
        transport.enqueue_json({"wrong": "shape"})
        client = ApiClient("http://api.test", transport=transport)
        with pytest.raises(ApiError) as info:
            client.get_overview()
        assert info.value.endpoint is not None
        assert "/api/v1/operator/overview" in info.value.endpoint


# ----------------------------------------------------------------------
# Round-trip of a minimally-populated AlertEvent through list validation
# ----------------------------------------------------------------------


class TestListAlertsValidation:
    def test_alert_event_round_trip(self) -> None:
        transport = FakeTransport()
        transport.enqueue_json(
            [
                {
                    "alert_id": "a1",
                    "severity": AlertSeverity.WARNING.value,
                    "kind": AlertKind.PHYSIOLOGY_STALE.value,
                    "message": "operator snapshot stale",
                    "emitted_at_utc": _utc(2026, 4, 17, 11, 45).isoformat(),
                }
            ]
        )
        client = ApiClient("http://api.test", transport=transport)
        alerts = client.list_alerts()
        assert len(alerts) == 1
        alert = alerts[0]
        assert isinstance(alert, AlertEvent)
        assert alert.kind is AlertKind.PHYSIOLOGY_STALE
        assert alert.severity is AlertSeverity.WARNING


# ----------------------------------------------------------------------
# HealthSnapshot validates degraded/recovering distinct from error
# ----------------------------------------------------------------------


class TestHealthValidation:
    def test_degraded_and_recovering_round_trip(self) -> None:
        transport = FakeTransport()
        transport.enqueue_json(
            {
                "generated_at_utc": _utc(2026, 4, 17, 12, 0).isoformat(),
                "overall_state": HealthState.DEGRADED.value,
                "subsystems": [
                    {
                        "subsystem_key": "ffmpeg",
                        "label": "FFmpeg transcoder",
                        "state": HealthState.RECOVERING.value,
                        "recovery_mode": "restart in progress",
                    }
                ],
                "degraded_count": 0,
                "recovering_count": 1,
                "error_count": 0,
            }
        )
        client = ApiClient("http://api.test", transport=transport)
        health = client.get_health()
        assert isinstance(health, HealthSnapshot)
        assert health.overall_state is HealthState.DEGRADED
        assert health.subsystems[0].state is HealthState.RECOVERING
        assert health.subsystems[0].recovery_mode == "restart in progress"


# ----------------------------------------------------------------------
# EncounterState round-trip — confirms the reward-explanation fields ride
# through list validation cleanly.
# ----------------------------------------------------------------------


class TestEncounterValidation:
    def test_gate_closed_encounter_round_trips(self) -> None:
        transport = FakeTransport()
        session_id = uuid4()
        transport.enqueue_json(
            [
                {
                    "encounter_id": "e1",
                    "session_id": str(session_id),
                    "segment_timestamp_utc": _utc(2026, 4, 17, 12, 1).isoformat(),
                    "state": EncounterState.REJECTED_GATE_CLOSED.value,
                    "active_arm": "arm_b",
                    "semantic_gate": 0,
                    "p90_intensity": 0.42,
                    "gated_reward": 0.0,
                    "n_frames_in_window": 90,
                    "physiology_attached": True,
                    "physiology_stale": False,
                    "notes": ["gate closed"],
                }
            ]
        )
        client = ApiClient("http://api.test", transport=transport)
        encounters = client.list_session_encounters(session_id)
        assert len(encounters) == 1
        e = encounters[0]
        assert e.state is EncounterState.REJECTED_GATE_CLOSED
        assert e.semantic_gate == 0
        assert e.gated_reward == 0.0
