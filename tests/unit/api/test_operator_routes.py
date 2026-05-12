"""
Route-level tests for `services/api/routes/operator.py`.

Focus:
  * Each handler returns the expected Phase-1 DTO (no raw dicts).
  * 404 on missing session / experiment.
  * 409 on stimulus submission against an already-ended session.
  * 503 on Redis publish failure.
  * Stimulus idempotency: duplicate client_action_id → accepted no-op,
    single publish call.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from packages.schemas.operator_console import (
    AttributionSummary,
    CloudActionStatus,
    CloudAuthState,
    CloudAuthStatus,
    CloudExperimentRefreshStatus,
    CloudOutboxSummary,
    CloudSignInResult,
    EncounterState,
    EncounterSummary,
    ExperimentBundleRefreshPreview,
    ExperimentBundleRefreshRequest,
    ExperimentBundleRefreshResult,
    ExperimentDetail,
    HealthSnapshot,
    HealthState,
    ObservationalAcousticSummary,
    OperatorStateBootstrap,
    OverviewSnapshot,
    SemanticEvaluationSummary,
    SessionPhysiologySnapshot,
    SessionSummary,
    StimulusAccepted,
    StimulusRequest,
)
from services.api.routes.operator import (
    get_cloud_auth_status,
    get_experiment_detail,
    get_health,
    get_latest_cloud_experiment_refresh,
    get_operator_state_bootstrap,
    get_overview,
    get_session,
    get_session_physiology,
    get_supported_event_service,
    list_alerts,
    list_session_encounters,
    list_sessions,
    refresh_cloud_experiment_bundle,
    sign_in_to_cloud,
    stream_operator_state_events,
    submit_stimulus,
)
from services.api.services.operator_action_service import (
    SessionAlreadyEndedError,
    SessionNotFoundError,
    StimulusPublishError,
)


def _now() -> datetime:
    return datetime(2026, 4, 17, 12, 0, tzinfo=UTC)


# ----------------------------------------------------------------------
# Read routes — stub service instances returned DTOs unchanged.
# ----------------------------------------------------------------------


class TestOperatorReadRoutes:
    def test_state_bootstrap_returns_dto(self) -> None:
        health = HealthSnapshot(generated_at_utc=_now(), overall_state=HealthState.OK)
        overview = OverviewSnapshot(generated_at_utc=_now(), health=health)
        payload = OperatorStateBootstrap(
            generated_at_utc=_now(),
            overview=overview,
            health=health,
        )
        svc = MagicMock()
        svc.build_bootstrap = AsyncMock(return_value=payload)
        result = asyncio.run(get_operator_state_bootstrap(service=svc))
        assert result is payload
        svc.build_bootstrap.assert_awaited_once()

    def test_state_events_passes_last_event_id(self) -> None:
        request = MagicMock()

        async def stream() -> AsyncIterator[str]:
            yield "event"

        svc = MagicMock()
        svc.has_event_stream_support.return_value = True
        svc.stream_events.return_value = stream()

        async def collect() -> list[Any]:
            return [
                event
                async for event in stream_operator_state_events(
                    request=request,
                    last_event_id="overview:abc",
                    service=svc,
                )
            ]

        assert asyncio.run(collect()) == ["event"]
        svc.stream_events.assert_called_once_with(request, last_event_id="overview:abc")

    def test_state_events_rejects_unsupported_runtime(self) -> None:
        svc = MagicMock()
        svc.has_event_stream_support.return_value = False

        with pytest.raises(HTTPException) as exc_info:
            get_supported_event_service(service=svc)

        assert exc_info.value.status_code == 503

    def test_overview_returns_dto(self) -> None:
        payload = OverviewSnapshot(generated_at_utc=_now())
        svc = MagicMock()
        svc.get_overview.return_value = payload
        result = asyncio.run(get_overview(service=svc))
        assert isinstance(result, OverviewSnapshot)
        svc.get_overview.assert_called_once()

    def test_list_sessions_returns_dto_list(self) -> None:
        sid = uuid.uuid4()
        rows = [
            SessionSummary(
                session_id=sid,
                status="active",
                started_at_utc=_now(),
            )
        ]
        svc = MagicMock()
        svc.list_sessions.return_value = rows
        result = asyncio.run(list_sessions(limit=10, service=svc))
        assert result == rows
        svc.list_sessions.assert_called_once_with(limit=10)

    def test_get_session_found(self) -> None:
        sid = uuid.uuid4()
        summary = SessionSummary(session_id=sid, status="active", started_at_utc=_now())
        svc = MagicMock()
        svc.get_session.return_value = summary
        result = asyncio.run(get_session(session_id=sid, service=svc))
        assert result is summary

    def test_get_session_missing_raises_404(self) -> None:
        sid = uuid.uuid4()
        svc = MagicMock()
        svc.get_session.return_value = None
        with pytest.raises(Exception) as exc_info:
            asyncio.run(get_session(session_id=sid, service=svc))
        exc: Any = exc_info.value
        assert exc.status_code == 404

    def test_list_encounters_passes_filters(self) -> None:
        sid = uuid.uuid4()
        svc = MagicMock()
        svc.list_encounters.return_value = []
        asyncio.run(list_session_encounters(session_id=sid, limit=25, before_utc=None, service=svc))
        svc.list_encounters.assert_called_once_with(sid, limit=25, before_utc=None)

    def test_get_experiment_detail_missing_raises_404(self) -> None:
        svc = MagicMock()
        svc.get_experiment_detail.return_value = None
        with pytest.raises(Exception) as exc_info:
            asyncio.run(get_experiment_detail(experiment_id="missing", service=svc))
        exc: Any = exc_info.value
        assert exc.status_code == 404

    def test_get_experiment_detail_found(self) -> None:
        detail = ExperimentDetail(experiment_id="greeting_line_v1", arms=[], last_updated_utc=None)
        svc = MagicMock()
        svc.get_experiment_detail.return_value = detail
        result = asyncio.run(get_experiment_detail(experiment_id="greeting_line_v1", service=svc))
        assert result is detail

    def test_get_session_physiology_missing_raises_404(self) -> None:
        sid = uuid.uuid4()
        svc = MagicMock()
        svc.get_session_physiology.return_value = None
        with pytest.raises(Exception) as exc_info:
            asyncio.run(get_session_physiology(session_id=sid, service=svc))
        exc: Any = exc_info.value
        assert exc.status_code == 404

    def test_get_session_physiology_found(self) -> None:
        sid = uuid.uuid4()
        snap = SessionPhysiologySnapshot(session_id=sid, generated_at_utc=_now())
        svc = MagicMock()
        svc.get_session_physiology.return_value = snap
        result = asyncio.run(get_session_physiology(session_id=sid, service=svc))
        assert result is snap

    def test_get_health_returns_dto(self) -> None:
        health = HealthSnapshot(generated_at_utc=_now(), overall_state=HealthState.OK)
        svc = MagicMock()
        svc.get_health.return_value = health
        result = asyncio.run(get_health(service=svc))
        assert result is health

    def test_list_alerts_returns_dto_list(self) -> None:
        svc = MagicMock()
        svc.list_alerts.return_value = []
        result = asyncio.run(list_alerts(limit=10, since_utc=None, service=svc))
        assert result == []
        svc.list_alerts.assert_called_once_with(limit=10, since_utc=None)

    def test_runtime_error_surfaces_as_503(self) -> None:
        svc = MagicMock()
        svc.get_overview.side_effect = RuntimeError("pool not initialized")
        with pytest.raises(Exception) as exc_info:
            asyncio.run(get_overview(service=svc))
        exc: Any = exc_info.value
        assert exc.status_code == 503

    def test_list_encounters_returns_acoustic_dto_payload(self) -> None:
        sid = uuid.uuid4()
        encounter = EncounterSummary(
            encounter_id="enc-acoustic-1",
            session_id=sid,
            segment_timestamp_utc=_now(),
            state=EncounterState.COMPLETED,
            active_arm="warm_welcome",
            semantic_gate=1,
            p90_intensity=0.71,
            gated_reward=0.71,
            n_frames_in_window=128,
            observational_acoustic=ObservationalAcousticSummary(
                f0_valid_measure=True,
                f0_valid_baseline=False,
                perturbation_valid_measure=True,
                perturbation_valid_baseline=False,
                voiced_coverage_measure_s=2.4,
                voiced_coverage_baseline_s=0.0,
                f0_mean_measure_hz=215.5,
                f0_mean_baseline_hz=None,
                f0_delta_semitones=None,
                jitter_mean_measure=0.018,
                jitter_mean_baseline=None,
                jitter_delta=None,
                shimmer_mean_measure=0.027,
                shimmer_mean_baseline=None,
                shimmer_delta=None,
            ),
            semantic_evaluation=SemanticEvaluationSummary(
                reasoning="cross_encoder_high_match",
                is_match=True,
                confidence_score=0.83,
                semantic_method="cross_encoder",
                semantic_method_version="ce-v1.2.3",
            ),
            attribution=AttributionSummary(
                finality="online_provisional",
                soft_reward_candidate=0.59,
                au12_baseline_pre=0.04,
                au12_lift_p90=0.67,
                au12_lift_peak=0.72,
                au12_peak_latency_ms=240.0,
                sync_peak_corr=0.41,
                sync_peak_lag=1,
                outcome_link_lag_s=12.0,
            ),
        )
        svc = MagicMock()
        svc.list_encounters.return_value = [encounter]

        result = asyncio.run(
            list_session_encounters(
                session_id=sid,
                limit=25,
                before_utc=None,
                service=svc,
            )
        )

        assert result == [encounter]
        assert result[0].observational_acoustic is not None
        assert result[0].observational_acoustic.f0_valid_measure is True
        assert isinstance(result[0].observational_acoustic.f0_valid_measure, bool)
        assert result[0].observational_acoustic.f0_valid_baseline is False
        assert result[0].observational_acoustic.voiced_coverage_measure_s == 2.4
        assert result[0].observational_acoustic.f0_mean_measure_hz == 215.5
        assert result[0].observational_acoustic.f0_mean_baseline_hz is None
        assert result[0].semantic_evaluation is not None
        assert result[0].semantic_evaluation.reasoning == "cross_encoder_high_match"
        assert result[0].semantic_evaluation.is_match is True
        assert result[0].semantic_evaluation.confidence_score == 0.83
        assert result[0].attribution is not None
        assert result[0].attribution.finality == "online_provisional"
        assert result[0].attribution.soft_reward_candidate == 0.59
        assert result[0].attribution.sync_peak_lag == 1
        assert isinstance(result[0].attribution.sync_peak_lag, int)
        assert result[0].attribution.outcome_link_lag_s == 12.0

    def test_list_encounters_returns_additive_null_semantic_attribution_fields(self) -> None:
        sid = uuid.uuid4()
        encounter = EncounterSummary(
            encounter_id="enc-no-attribution",
            session_id=sid,
            segment_timestamp_utc=_now(),
            state=EncounterState.COMPLETED,
        )
        svc = MagicMock()
        svc.list_encounters.return_value = [encounter]

        result = asyncio.run(
            list_session_encounters(
                session_id=sid,
                limit=25,
                before_utc=None,
                service=svc,
            )
        )

        assert result == [encounter]
        assert result[0].semantic_evaluation is None
        assert result[0].attribution is None


class TestCloudOperatorRoutes:
    def test_latest_experiment_refresh_returns_persisted_background_state(self) -> None:
        result = ExperimentBundleRefreshResult(
            status=CloudExperimentRefreshStatus.FAILED,
            completed_at_utc=_now(),
            message="Cloud authorization was rejected.",
        )
        svc = MagicMock()
        svc.get_latest_experiment_refresh.return_value = result

        response = asyncio.run(get_latest_cloud_experiment_refresh(service=svc))

        assert response is result
        svc.get_latest_experiment_refresh.assert_called_once()

    def test_cloud_experiment_refresh_passes_preview_token_to_service(self) -> None:
        result = ExperimentBundleRefreshResult(
            status=CloudExperimentRefreshStatus.APPLIED,
            completed_at_utc=_now(),
            message="Experiment bundle refreshed.",
        )
        svc = MagicMock()
        svc.refresh_experiment_bundle.return_value = result
        request = ExperimentBundleRefreshRequest(preview_token="preview-token-a")

        response = asyncio.run(refresh_cloud_experiment_bundle(request, service=svc))

        assert response is result
        svc.refresh_experiment_bundle.assert_called_once_with(request)

    def test_cloud_sign_in_does_not_block_auth_status_route(self) -> None:
        sign_in_entered = threading.Event()
        release_sign_in = threading.Event()

        class Service:
            def sign_in(self) -> CloudSignInResult:
                sign_in_entered.set()
                assert release_sign_in.wait(timeout=5)
                return CloudSignInResult(
                    status=CloudActionStatus.SUCCEEDED,
                    auth_state=CloudAuthState.SIGNED_IN,
                    completed_at_utc=_now(),
                    message="Cloud sign-in completed.",
                )

            def get_auth_status(self) -> CloudAuthStatus:
                return CloudAuthStatus(
                    state=CloudAuthState.SIGNED_OUT,
                    checked_at_utc=_now(),
                    message="Cloud sign-in is required.",
                )

            def get_outbox_summary(self) -> CloudOutboxSummary:
                return CloudOutboxSummary(generated_at_utc=_now())

            def get_latest_experiment_refresh(self) -> ExperimentBundleRefreshResult | None:
                return None

            def preview_experiment_bundle_refresh(self) -> ExperimentBundleRefreshPreview:
                return ExperimentBundleRefreshPreview(
                    status=CloudActionStatus.FAILED,
                    checked_at_utc=_now(),
                    message="not used",
                )

            def refresh_experiment_bundle(
                self,
                _request: ExperimentBundleRefreshRequest,
            ) -> ExperimentBundleRefreshResult:
                return ExperimentBundleRefreshResult(
                    status=CloudExperimentRefreshStatus.FAILED,
                    completed_at_utc=_now(),
                    message="not used",
                )

        service = Service()

        async def exercise_routes() -> CloudAuthStatus:
            sign_in_task = asyncio.create_task(sign_in_to_cloud(service=service))
            assert await asyncio.to_thread(sign_in_entered.wait, 5)
            status = await asyncio.wait_for(
                get_cloud_auth_status(service=service),
                timeout=1,
            )
            release_sign_in.set()
            await sign_in_task
            return status

        status = asyncio.run(exercise_routes())

        assert status.state is CloudAuthState.SIGNED_OUT


# ----------------------------------------------------------------------
# Stimulus submission
# ----------------------------------------------------------------------


class TestStimulusSubmission:
    def test_stimulus_accepted(self) -> None:
        sid = uuid.uuid4()
        action_id = uuid.uuid4()
        request = StimulusRequest(client_action_id=action_id)
        accepted = StimulusAccepted(
            session_id=sid,
            client_action_id=action_id,
            accepted=True,
            received_at_utc=_now(),
        )
        svc = MagicMock()
        svc.submit_stimulus.return_value = accepted
        result = asyncio.run(submit_stimulus(session_id=sid, request=request, service=svc))
        assert result is accepted
        svc.submit_stimulus.assert_called_once_with(sid, request)

    def test_session_not_found_raises_404(self) -> None:
        sid = uuid.uuid4()
        request = StimulusRequest(client_action_id=uuid.uuid4())
        svc = MagicMock()
        svc.submit_stimulus.side_effect = SessionNotFoundError(str(sid))
        with pytest.raises(Exception) as exc_info:
            asyncio.run(submit_stimulus(session_id=sid, request=request, service=svc))
        exc: Any = exc_info.value
        assert exc.status_code == 404

    def test_session_ended_raises_409(self) -> None:
        sid = uuid.uuid4()
        request = StimulusRequest(client_action_id=uuid.uuid4())
        svc = MagicMock()
        svc.submit_stimulus.side_effect = SessionAlreadyEndedError(str(sid))
        with pytest.raises(Exception) as exc_info:
            asyncio.run(submit_stimulus(session_id=sid, request=request, service=svc))
        exc: Any = exc_info.value
        assert exc.status_code == 409

    def test_broker_down_raises_503(self) -> None:
        sid = uuid.uuid4()
        request = StimulusRequest(client_action_id=uuid.uuid4())
        svc = MagicMock()
        svc.submit_stimulus.side_effect = StimulusPublishError("broker down")
        with pytest.raises(Exception) as exc_info:
            asyncio.run(submit_stimulus(session_id=sid, request=request, service=svc))
        exc: Any = exc_info.value
        assert exc.status_code == 503


# ----------------------------------------------------------------------
# OperatorActionService idempotency (service-level, routed through
# here because it shares the action surface under test).
# ----------------------------------------------------------------------


class TestOperatorActionServiceIdempotency:
    def test_duplicate_client_action_id_suppressed(self) -> None:
        from services.api.services.operator_action_service import (
            OperatorActionService,
        )

        sid = uuid.uuid4()
        action_id = uuid.uuid4()
        # DB: session exists, not ended.
        conn = MagicMock()
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        cursor.description = [
            ("session_id",),
            ("started_at",),
            ("ended_at",),
            ("duration_s",),
            ("experiment_id",),
            ("active_arm",),
            ("last_segment_completed_at_utc",),
            ("latest_reward",),
            ("latest_semantic_gate",),
        ]
        cursor.fetchone.return_value = (
            str(sid),
            datetime(2026, 4, 17, 11, 0, tzinfo=UTC),
            None,
            60.0,
            None,
            None,
            None,
            None,
            None,
        )
        conn.cursor.return_value = cursor

        redis = MagicMock()
        redis.set.return_value = None  # SETNX says key already existed.
        redis.publish.return_value = 1
        svc = OperatorActionService(
            get_conn=lambda: conn,
            put_conn=lambda _c: None,
            redis_factory=lambda: redis,
            clock=_now,
        )
        request = StimulusRequest(client_action_id=action_id)
        result = svc.submit_stimulus(sid, request)
        assert result.accepted is True
        assert result.message == "duplicate submission suppressed"
        redis.publish.assert_not_called()

    def test_first_submit_publishes_trigger(self) -> None:
        from services.api.services.operator_action_service import (
            OperatorActionService,
        )

        sid = uuid.uuid4()
        action_id = uuid.uuid4()
        conn = MagicMock()
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        cursor.description = [
            ("session_id",),
            ("started_at",),
            ("ended_at",),
            ("duration_s",),
            ("experiment_id",),
            ("active_arm",),
            ("last_segment_completed_at_utc",),
            ("latest_reward",),
            ("latest_semantic_gate",),
        ]
        cursor.fetchone.return_value = (
            str(sid),
            datetime(2026, 4, 17, 11, 0, tzinfo=UTC),
            None,
            60.0,
            None,
            None,
            None,
            None,
            None,
        )
        conn.cursor.return_value = cursor

        redis = MagicMock()
        redis.set.return_value = True
        redis.publish.return_value = 2
        svc = OperatorActionService(
            get_conn=lambda: conn,
            put_conn=lambda _c: None,
            redis_factory=lambda: redis,
            clock=_now,
        )
        request = StimulusRequest(client_action_id=action_id)
        result = svc.submit_stimulus(sid, request)
        assert result.accepted is True
        assert result.message is None
        redis.publish.assert_called_once_with("stimulus:inject", "inject")

    def test_ended_session_raises(self) -> None:
        from services.api.services.operator_action_service import (
            OperatorActionService,
            SessionAlreadyEndedError,
        )

        sid = uuid.uuid4()
        action_id = uuid.uuid4()
        conn = MagicMock()
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        cursor.description = [
            ("session_id",),
            ("started_at",),
            ("ended_at",),
            ("duration_s",),
            ("experiment_id",),
            ("active_arm",),
            ("last_segment_completed_at_utc",),
            ("latest_reward",),
            ("latest_semantic_gate",),
        ]
        cursor.fetchone.return_value = (
            str(sid),
            datetime(2026, 4, 17, 11, 0, tzinfo=UTC),
            datetime(2026, 4, 17, 11, 30, tzinfo=UTC),
            1800.0,
            None,
            None,
            None,
            None,
            None,
        )
        conn.cursor.return_value = cursor

        redis = MagicMock()
        svc = OperatorActionService(
            get_conn=lambda: conn,
            put_conn=lambda _c: None,
            redis_factory=lambda: redis,
            clock=_now,
        )
        with pytest.raises(SessionAlreadyEndedError):
            svc.submit_stimulus(sid, StimulusRequest(client_action_id=action_id))
        redis.set.assert_not_called()

    def test_publish_failure_raises(self) -> None:
        from services.api.services.operator_action_service import (
            OperatorActionService,
            StimulusPublishError,
        )

        sid = uuid.uuid4()
        action_id = uuid.uuid4()
        conn = MagicMock()
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        cursor.description = [
            ("session_id",),
            ("started_at",),
            ("ended_at",),
            ("duration_s",),
            ("experiment_id",),
            ("active_arm",),
            ("last_segment_completed_at_utc",),
            ("latest_reward",),
            ("latest_semantic_gate",),
        ]
        cursor.fetchone.return_value = (
            str(sid),
            datetime(2026, 4, 17, 11, 0, tzinfo=UTC),
            None,
            60.0,
            None,
            None,
            None,
            None,
            None,
        )
        conn.cursor.return_value = cursor

        redis = MagicMock()
        redis.set.return_value = True
        redis.publish.side_effect = Exception("broker down")
        svc = OperatorActionService(
            get_conn=lambda: conn,
            put_conn=lambda _c: None,
            redis_factory=lambda: redis,
            clock=_now,
        )
        with pytest.raises(StimulusPublishError):
            svc.submit_stimulus(sid, StimulusRequest(client_action_id=action_id))

    def test_no_subscribers_sets_warning_message(self) -> None:
        from services.api.services.operator_action_service import (
            OperatorActionService,
        )

        sid = uuid.uuid4()
        action_id = uuid.uuid4()
        conn = MagicMock()
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        cursor.description = [
            ("session_id",),
            ("started_at",),
            ("ended_at",),
            ("duration_s",),
            ("experiment_id",),
            ("active_arm",),
            ("last_segment_completed_at_utc",),
            ("latest_reward",),
            ("latest_semantic_gate",),
        ]
        cursor.fetchone.return_value = (
            str(sid),
            datetime(2026, 4, 17, 11, 0, tzinfo=UTC),
            None,
            60.0,
            None,
            None,
            None,
            None,
            None,
        )
        conn.cursor.return_value = cursor

        redis = MagicMock()
        redis.set.return_value = True
        redis.publish.return_value = 0
        svc = OperatorActionService(
            get_conn=lambda: conn,
            put_conn=lambda _c: None,
            redis_factory=lambda: redis,
            clock=_now,
        )
        result = svc.submit_stimulus(sid, StimulusRequest(client_action_id=action_id))
        assert result.accepted is True
        assert result.message is not None
        assert "no orchestrator" in result.message.lower()
