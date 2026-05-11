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

from packages.schemas.experiments import (
    ExperimentAdminResponse,
    ExperimentArmAdminResponse,
    ExperimentArmCreateRequest,
    ExperimentArmDeleteResponse,
    ExperimentArmPatchRequest,
    ExperimentArmSeedRequest,
    ExperimentCreateRequest,
)
from packages.schemas.operator_console import (
    AlertEvent,
    AlertKind,
    AlertSeverity,
    AttributionSummary,
    CloudActionStatus,
    CloudAuthState,
    CloudAuthStatus,
    CloudExperimentRefreshStatus,
    CloudOutboxSummary,
    CloudSignInResult,
    EncounterState,
    ExperimentBundleRefreshResult,
    HealthSnapshot,
    HealthState,
    ObservationalAcousticSummary,
    OverviewSnapshot,
    SemanticEvaluationSummary,
    SessionCreateRequest,
    SessionEndRequest,
    SessionLifecycleAccepted,
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


class TestCloudEndpoints:
    def test_cloud_status_and_outbox_validate_bounded_dtos(self) -> None:
        transport = FakeTransport()
        transport.enqueue_json(
            {
                "state": CloudAuthState.SIGNED_IN.value,
                "checked_at_utc": _utc(2026, 5, 2, 12, 0).isoformat(),
                "message": "Cloud sign-in is active.",
            }
        )
        transport.enqueue_json(
            {
                "generated_at_utc": _utc(2026, 5, 2, 12, 1).isoformat(),
                "pending_count": 2,
                "in_flight_count": 1,
                "dead_letter_count": 1,
                "retry_scheduled_count": 1,
                "redacted_count": 1,
                "last_error": "HTTP 503",
            }
        )
        client = ApiClient("http://api.test", transport=transport)

        status = client.get_cloud_auth_status()
        summary = client.get_cloud_outbox_summary()

        assert isinstance(status, CloudAuthStatus)
        assert isinstance(summary, CloudOutboxSummary)
        assert transport.calls[0].url.endswith("/api/v1/operator/cloud/auth/status")
        assert transport.calls[1].url.endswith("/api/v1/operator/cloud/outbox")

    def test_cloud_actions_post_empty_body_and_validate_bounded_results(self) -> None:
        transport = FakeTransport()
        transport.enqueue_json(
            {
                "status": CloudActionStatus.SUCCEEDED.value,
                "auth_state": CloudAuthState.SIGNED_IN.value,
                "completed_at_utc": _utc(2026, 5, 2, 12, 0).isoformat(),
                "message": "Cloud sign-in completed.",
            }
        )
        transport.enqueue_json(
            {
                "status": CloudExperimentRefreshStatus.APPLIED.value,
                "completed_at_utc": _utc(2026, 5, 2, 12, 1).isoformat(),
                "message": "Experiment bundle refreshed.",
                "bundle_id": "bundle-a",
                "experiment_count": 2,
            }
        )
        client = ApiClient("http://api.test", transport=transport)

        sign_in = client.post_cloud_sign_in()
        refresh = client.post_experiment_bundle_refresh()

        assert isinstance(sign_in, CloudSignInResult)
        assert isinstance(refresh, ExperimentBundleRefreshResult)
        assert [call.method for call in transport.calls] == ["POST", "POST"]
        assert [json.loads(call.body or b"{}") for call in transport.calls] == [{}, {}]
        assert transport.calls[0].url.endswith("/api/v1/operator/cloud/auth/sign-in")
        assert transport.calls[0].timeout_s == 125.0
        assert transport.calls[1].url.endswith("/api/v1/operator/cloud/experiments/refresh")
        assert transport.calls[1].timeout_s == 10.0


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


class TestSessionLifecyclePosts:
    def test_post_session_start_serialises_body_and_validates_response(self) -> None:
        transport = FakeTransport()
        session_id = UUID("00000000-0000-0000-0000-000000000010")
        action_id = UUID("22222222-2222-2222-2222-222222222222")
        transport.enqueue_json(
            {
                "action": "start",
                "session_id": str(session_id),
                "client_action_id": str(action_id),
                "accepted": True,
                "received_at_utc": _utc(2026, 4, 17, 12, 6).isoformat(),
            }
        )
        client = ApiClient("http://api.test", transport=transport)
        request = SessionCreateRequest(
            stream_url="rtmp://example/live",
            experiment_id="greeting_line_v1",
            client_action_id=action_id,
        )
        accepted = client.post_session_start(request)

        assert isinstance(accepted, SessionLifecycleAccepted)
        assert accepted.action == "start"
        call = transport.calls[0]
        assert call.method == "POST"
        assert call.url.endswith("/api/v1/sessions")
        assert call.body is not None
        body = json.loads(call.body.decode("utf-8"))
        assert body["stream_url"] == "rtmp://example/live"
        assert body["experiment_id"] == "greeting_line_v1"
        assert body["client_action_id"] == str(action_id)

    def test_post_session_end_serialises_body_and_validates_response(self) -> None:
        transport = FakeTransport()
        session_id = UUID("00000000-0000-0000-0000-000000000011")
        action_id = UUID("33333333-3333-3333-3333-333333333333")
        transport.enqueue_json(
            {
                "action": "end",
                "session_id": str(session_id),
                "client_action_id": str(action_id),
                "accepted": True,
                "received_at_utc": _utc(2026, 4, 17, 12, 7).isoformat(),
            }
        )
        client = ApiClient("http://api.test", transport=transport)
        request = SessionEndRequest(client_action_id=action_id)
        accepted = client.post_session_end(session_id, request)

        assert isinstance(accepted, SessionLifecycleAccepted)
        assert accepted.action == "end"
        call = transport.calls[0]
        assert call.method == "POST"
        assert call.url.endswith(f"/api/v1/sessions/{session_id}/end")
        assert call.body is not None
        body = json.loads(call.body.decode("utf-8"))
        assert body == {"client_action_id": str(action_id)}


# ----------------------------------------------------------------------
# Experiment admin writes
# ----------------------------------------------------------------------


class TestExperimentAdminWrites:
    def test_create_experiment_posts_typed_body(self) -> None:
        transport = FakeTransport()
        transport.enqueue_json(
            {
                "experiment_id": "exp-new",
                "label": "Greeting v2",
                "arms": [
                    {
                        "experiment_id": "exp-new",
                        "label": "Greeting v2",
                        "arm": "arm-a",
                        "greeting_text": "Hei",
                        "alpha_param": 1.0,
                        "beta_param": 1.0,
                        "enabled": True,
                    }
                ],
            }
        )
        client = ApiClient("http://api.test", transport=transport)
        result = client.create_experiment(
            ExperimentCreateRequest(
                experiment_id="exp-new",
                label="Greeting v2",
                arms=[ExperimentArmSeedRequest(arm="arm-a", greeting_text="Hei")],
            )
        )
        assert isinstance(result, ExperimentAdminResponse)
        call = transport.calls[0]
        assert call.method == "POST"
        assert call.url.endswith("/api/v1/experiments")
        assert call.body is not None
        body = json.loads(call.body.decode("utf-8"))
        assert body["arms"] == [{"arm": "arm-a", "greeting_text": "Hei"}]

    def test_add_arm_posts_to_nested_endpoint(self) -> None:
        transport = FakeTransport()
        transport.enqueue_json(
            {
                "experiment_id": "exp-new",
                "label": "Greeting v2",
                "arm": "arm-b",
                "greeting_text": "Moi",
                "alpha_param": 1.0,
                "beta_param": 1.0,
                "enabled": True,
            }
        )
        client = ApiClient("http://api.test", transport=transport)
        result = client.add_experiment_arm(
            "exp-new",
            ExperimentArmCreateRequest(arm="arm-b", greeting_text="Moi"),
        )
        assert isinstance(result, ExperimentArmAdminResponse)
        assert transport.calls[0].method == "POST"
        assert transport.calls[0].url.endswith("/api/v1/experiments/exp-new/arms")

    def test_patch_arm_uses_patch_and_never_requires_posterior_fields(self) -> None:
        transport = FakeTransport()
        transport.enqueue_json(
            {
                "experiment_id": "exp-new",
                "label": "Greeting v2",
                "arm": "arm-b",
                "greeting_text": "Moi ystävä",
                "alpha_param": 3.0,
                "beta_param": 2.0,
                "enabled": False,
            }
        )
        client = ApiClient("http://api.test", transport=transport)
        result = client.patch_experiment_arm(
            "exp-new",
            "arm-b",
            ExperimentArmPatchRequest(greeting_text="Moi ystävä", enabled=False),
        )
        assert isinstance(result, ExperimentArmAdminResponse)
        call = transport.calls[0]
        assert call.method == "PATCH"
        assert call.url.endswith("/api/v1/experiments/exp-new/arms/arm-b")
        assert call.body is not None
        body = json.loads(call.body.decode("utf-8"))
        assert body == {"greeting_text": "Moi ystävä", "enabled": False}
        assert "alpha_param" not in body
        assert "beta_param" not in body

    def test_delete_arm_uses_delete_and_validates_guard_response(self) -> None:
        transport = FakeTransport()
        transport.enqueue_json(
            {
                "experiment_id": "exp-new",
                "arm": "arm-b",
                "deleted": False,
                "posterior_preserved": True,
                "reason": "arm has posterior history; disabled instead of hard-deleting",
                "arm_state": {
                    "experiment_id": "exp-new",
                    "label": "Greeting v2",
                    "arm": "arm-b",
                    "greeting_text": "Moi ystävä",
                    "alpha_param": 3.0,
                    "beta_param": 2.0,
                    "enabled": False,
                },
            }
        )
        client = ApiClient("http://api.test", transport=transport)

        result = client.delete_experiment_arm("exp-new", "arm-b")

        assert isinstance(result, ExperimentArmDeleteResponse)
        assert result.deleted is False
        assert result.posterior_preserved is True
        call = transport.calls[0]
        assert call.method == "DELETE"
        assert call.url.endswith("/api/v1/experiments/exp-new/arms/arm-b")
        assert call.body is None


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
                "subsystem_probes": {
                    "whisper_worker": {
                        "subsystem_key": "whisper_worker",
                        "label": "Whisper Worker",
                        "state": "ok",
                        "checked_at_utc": _utc(2026, 4, 17, 12, 0).isoformat(),
                    }
                },
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
        assert "whisper_worker" in health.subsystem_probes


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

    def test_completed_encounter_round_trips_semantic_attribution_diagnostics(self) -> None:
        transport = FakeTransport()
        session_id = uuid4()
        transport.enqueue_json(
            [
                {
                    "encounter_id": "e-semantic-attribution",
                    "session_id": str(session_id),
                    "segment_timestamp_utc": _utc(2026, 4, 17, 12, 2).isoformat(),
                    "state": EncounterState.COMPLETED.value,
                    "active_arm": "arm_a",
                    "semantic_gate": 1,
                    "semantic_confidence": 0.83,
                    "p90_intensity": 0.61,
                    "gated_reward": 0.61,
                    "n_frames_in_window": 120,
                    "semantic_evaluation": {
                        "reasoning": "cross_encoder_high_match",
                        "is_match": True,
                        "confidence_score": 0.83,
                        "semantic_method": "cross_encoder",
                        "semantic_method_version": "ce-v1",
                    },
                    "attribution": {
                        "finality": "online_provisional",
                        "soft_reward_candidate": 0.506,
                        "au12_baseline_pre": 0.10,
                        "au12_lift_p90": 0.51,
                        "au12_lift_peak": 0.72,
                        "au12_peak_latency_ms": 900.0,
                        "sync_peak_corr": 0.37,
                        "sync_peak_lag": 2,
                        "outcome_link_lag_s": 44.0,
                    },
                }
            ]
        )
        client = ApiClient("http://api.test", transport=transport)
        [encounter] = client.list_session_encounters(session_id)

        assert encounter.semantic_evaluation is not None
        assert isinstance(encounter.semantic_evaluation, SemanticEvaluationSummary)
        assert encounter.semantic_evaluation.reasoning == "cross_encoder_high_match"
        assert encounter.semantic_evaluation.semantic_method == "cross_encoder"
        assert encounter.semantic_evaluation.confidence_score == pytest.approx(0.83)
        assert encounter.attribution is not None
        assert isinstance(encounter.attribution, AttributionSummary)
        assert encounter.attribution.finality == "online_provisional"
        assert encounter.attribution.soft_reward_candidate == pytest.approx(0.506)
        assert encounter.attribution.sync_peak_corr == pytest.approx(0.37)
        assert encounter.attribution.outcome_link_lag_s == pytest.approx(44.0)

    def test_completed_encounter_round_trips_observational_acoustic(self) -> None:
        transport = FakeTransport()
        session_id = uuid4()
        transport.enqueue_json(
            [
                {
                    "encounter_id": "e-observational-acoustic",
                    "session_id": str(session_id),
                    "segment_timestamp_utc": _utc(2026, 4, 17, 12, 2).isoformat(),
                    "state": EncounterState.COMPLETED.value,
                    "active_arm": "arm_a",
                    "semantic_gate": 1,
                    "p90_intensity": 0.61,
                    "gated_reward": 0.61,
                    "n_frames_in_window": 120,
                    "physiology_attached": True,
                    "notes": [],
                    "observational_acoustic": {
                        "f0_valid_measure": True,
                        "f0_valid_baseline": True,
                        "perturbation_valid_measure": True,
                        "perturbation_valid_baseline": True,
                        "voiced_coverage_measure_s": 3.2,
                        "voiced_coverage_baseline_s": 2.7,
                        "f0_mean_measure_hz": 210.0,
                        "f0_mean_baseline_hz": 190.0,
                        "f0_delta_semitones": 1.73,
                        "jitter_mean_measure": 0.012,
                        "jitter_mean_baseline": 0.010,
                        "jitter_delta": 0.002,
                        "shimmer_mean_measure": 0.021,
                        "shimmer_mean_baseline": 0.019,
                        "shimmer_delta": 0.002,
                    },
                }
            ]
        )
        client = ApiClient("http://api.test", transport=transport)
        [encounter] = client.list_session_encounters(session_id)

        assert encounter.state is EncounterState.COMPLETED
        observational_acoustic = encounter.observational_acoustic
        assert isinstance(observational_acoustic, ObservationalAcousticSummary)
        assert observational_acoustic.f0_valid_measure is True
        assert isinstance(observational_acoustic.f0_valid_measure, bool)
        assert observational_acoustic.f0_valid_baseline is True
        assert observational_acoustic.perturbation_valid_measure is True
        assert observational_acoustic.perturbation_valid_baseline is True
        assert observational_acoustic.voiced_coverage_measure_s == pytest.approx(3.2)
        assert observational_acoustic.voiced_coverage_baseline_s == pytest.approx(2.7)
        assert observational_acoustic.f0_mean_measure_hz == pytest.approx(210.0)
        assert observational_acoustic.f0_mean_baseline_hz == pytest.approx(190.0)
        assert observational_acoustic.f0_delta_semitones == pytest.approx(1.73)

    def test_completed_encounter_round_trips_observational_acoustic_null_contract(self) -> None:
        transport = FakeTransport()
        session_id = uuid4()
        transport.enqueue_json(
            [
                {
                    "encounter_id": "e-observational-acoustic-null",
                    "session_id": str(session_id),
                    "segment_timestamp_utc": _utc(2026, 4, 17, 12, 2).isoformat(),
                    "state": EncounterState.COMPLETED.value,
                    "active_arm": "arm_a",
                    "semantic_gate": 1,
                    "p90_intensity": 0.55,
                    "gated_reward": 0.55,
                    "n_frames_in_window": 120,
                    "physiology_attached": False,
                    "notes": [],
                    "observational_acoustic": {
                        "f0_valid_measure": False,
                        "f0_valid_baseline": False,
                        "perturbation_valid_measure": False,
                        "perturbation_valid_baseline": False,
                        "voiced_coverage_measure_s": 0.0,
                        "voiced_coverage_baseline_s": 0.0,
                        "f0_mean_measure_hz": None,
                        "f0_mean_baseline_hz": None,
                        "f0_delta_semitones": None,
                        "jitter_mean_measure": None,
                        "jitter_mean_baseline": None,
                        "jitter_delta": None,
                        "shimmer_mean_measure": None,
                        "shimmer_mean_baseline": None,
                        "shimmer_delta": None,
                    },
                }
            ]
        )
        client = ApiClient("http://api.test", transport=transport)
        [encounter] = client.list_session_encounters(session_id)

        assert encounter.state is EncounterState.COMPLETED
        assert encounter.observational_acoustic is not None
        observational_acoustic = encounter.observational_acoustic
        assert observational_acoustic.f0_valid_measure is False
        assert isinstance(observational_acoustic.f0_valid_measure, bool)
        assert observational_acoustic.f0_valid_baseline is False
        assert isinstance(observational_acoustic.f0_valid_baseline, bool)
        assert observational_acoustic.perturbation_valid_measure is False
        assert isinstance(observational_acoustic.perturbation_valid_measure, bool)
        assert observational_acoustic.perturbation_valid_baseline is False
        assert isinstance(observational_acoustic.perturbation_valid_baseline, bool)
        assert observational_acoustic.voiced_coverage_measure_s == 0.0
        assert observational_acoustic.voiced_coverage_baseline_s == 0.0
        assert observational_acoustic.f0_mean_measure_hz is None
        assert observational_acoustic.f0_mean_baseline_hz is None
        assert observational_acoustic.f0_delta_semitones is None
