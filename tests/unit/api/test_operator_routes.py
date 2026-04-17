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
import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from packages.schemas.operator_console import (
    ExperimentDetail,
    HealthSnapshot,
    HealthState,
    OverviewSnapshot,
    SessionPhysiologySnapshot,
    SessionSummary,
    StimulusAccepted,
    StimulusRequest,
)
from services.api.routes.operator import (
    get_experiment_detail,
    get_health,
    get_overview,
    get_session,
    get_session_physiology,
    list_alerts,
    list_session_encounters,
    list_sessions,
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
        summary = SessionSummary(
            session_id=sid, status="active", started_at_utc=_now()
        )
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
        asyncio.run(
            list_session_encounters(
                session_id=sid, limit=25, before_utc=None, service=svc
            )
        )
        svc.list_encounters.assert_called_once_with(sid, limit=25, before_utc=None)

    def test_get_experiment_detail_missing_raises_404(self) -> None:
        svc = MagicMock()
        svc.get_experiment_detail.return_value = None
        with pytest.raises(Exception) as exc_info:
            asyncio.run(get_experiment_detail(experiment_id="missing", service=svc))
        exc: Any = exc_info.value
        assert exc.status_code == 404

    def test_get_experiment_detail_found(self) -> None:
        detail = ExperimentDetail(
            experiment_id="greeting_line_v1", arms=[], last_updated_utc=None
        )
        svc = MagicMock()
        svc.get_experiment_detail.return_value = detail
        result = asyncio.run(
            get_experiment_detail(experiment_id="greeting_line_v1", service=svc)
        )
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
        snap = SessionPhysiologySnapshot(
            session_id=sid, generated_at_utc=_now()
        )
        svc = MagicMock()
        svc.get_session_physiology.return_value = snap
        result = asyncio.run(get_session_physiology(session_id=sid, service=svc))
        assert result is snap

    def test_get_health_returns_dto(self) -> None:
        health = HealthSnapshot(
            generated_at_utc=_now(), overall_state=HealthState.OK
        )
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
        result = asyncio.run(
            submit_stimulus(session_id=sid, request=request, service=svc)
        )
        assert result is accepted
        svc.submit_stimulus.assert_called_once_with(sid, request)

    def test_session_not_found_raises_404(self) -> None:
        sid = uuid.uuid4()
        request = StimulusRequest(client_action_id=uuid.uuid4())
        svc = MagicMock()
        svc.submit_stimulus.side_effect = SessionNotFoundError(str(sid))
        with pytest.raises(Exception) as exc_info:
            asyncio.run(
                submit_stimulus(session_id=sid, request=request, service=svc)
            )
        exc: Any = exc_info.value
        assert exc.status_code == 404

    def test_session_ended_raises_409(self) -> None:
        sid = uuid.uuid4()
        request = StimulusRequest(client_action_id=uuid.uuid4())
        svc = MagicMock()
        svc.submit_stimulus.side_effect = SessionAlreadyEndedError(str(sid))
        with pytest.raises(Exception) as exc_info:
            asyncio.run(
                submit_stimulus(session_id=sid, request=request, service=svc)
            )
        exc: Any = exc_info.value
        assert exc.status_code == 409

    def test_broker_down_raises_503(self) -> None:
        sid = uuid.uuid4()
        request = StimulusRequest(client_action_id=uuid.uuid4())
        svc = MagicMock()
        svc.submit_stimulus.side_effect = StimulusPublishError("broker down")
        with pytest.raises(Exception) as exc_info:
            asyncio.run(
                submit_stimulus(session_id=sid, request=request, service=svc)
            )
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
        result = svc.submit_stimulus(
            sid, StimulusRequest(client_action_id=action_id)
        )
        assert result.accepted is True
        assert result.message is not None
        assert "no orchestrator" in result.message.lower()
