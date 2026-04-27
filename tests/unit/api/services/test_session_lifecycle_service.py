"""Unit tests for SessionLifecycleService."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from packages.schemas.operator_console import SessionCreateRequest, SessionEndRequest
from services.api.services.session_lifecycle_service import (
    _IDEMPOTENCY_TTL_S,
    SessionLifecycleConflictError,
    SessionLifecyclePublishError,
    SessionLifecycleService,
    _stable_session_id_for_action,
)

_NOW = datetime(2026, 4, 18, 12, 0, tzinfo=UTC)


def _clock() -> datetime:
    return _NOW


def _make_state_connection(
    responses: list[tuple[list[str], tuple[Any, ...] | None]],
) -> tuple[MagicMock, MagicMock]:
    """Create a mock connection/cursor that yields one row per execute call."""
    conn = MagicMock()
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    iterator = iter(responses)

    def _execute(_sql: str, _params: dict[str, Any] | None = None) -> None:
        columns, row = next(iterator)
        cursor.description = [(col,) for col in columns]
        cursor.fetchone.return_value = row

    cursor.execute.side_effect = _execute
    conn.cursor.return_value = cursor
    return conn, cursor


def _assert_idempotency_write(
    redis: MagicMock,
    *,
    action: str,
    session_id: uuid.UUID,
    client_action_id: uuid.UUID,
) -> None:
    redis.set.assert_called_once()
    key, raw_value = redis.set.call_args.args
    assert key == f"operator:session:seen:{client_action_id}"
    assert json.loads(raw_value) == {
        "action": action,
        "session_id": str(session_id),
    }
    assert redis.set.call_args.kwargs == {
        "nx": True,
        "ex": _IDEMPOTENCY_TTL_S,
    }


def _duplicate_end_record(session_id: uuid.UUID) -> bytes:
    return json.dumps(
        {"action": "end", "session_id": str(session_id)},
        separators=(",", ":"),
    ).encode()


class _RetryableRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.publish_attempts = 0
        self.publish_messages: list[tuple[str, str]] = []
        self.delete_calls: list[str] = []

    def set(
        self,
        name: str,
        value: str,
        *,
        nx: bool = False,
        xx: bool = False,
        ex: int = 0,
        keepttl: bool = False,
    ) -> bool | None:
        _ = xx, ex, keepttl
        if nx and name in self.values:
            return None
        self.values[name] = value
        return True

    def get(self, name: str) -> str | None:
        return self.values.get(name)

    def delete(self, *names: str) -> int:
        deleted = 0
        for name in names:
            self.delete_calls.append(name)
            if self.values.pop(name, None) is not None:
                deleted += 1
        return deleted

    def publish(self, channel: str, message: str) -> int:
        self.publish_attempts += 1
        self.publish_messages.append((channel, message))
        if self.publish_attempts == 1:
            raise Exception("broker down")
        return 1

    def close(self) -> None:
        return None


class TestRequestSessionStart:
    def test_first_submit_publishes_lifecycle_intent(self) -> None:
        redis = MagicMock()
        redis.set.return_value = True
        redis.publish.return_value = 2
        svc = SessionLifecycleService(
            get_conn=lambda: None,
            put_conn=lambda _conn: None,
            redis_factory=lambda: redis,
            clock=_clock,
        )
        request = SessionCreateRequest(
            stream_url="https://example.com/live",
            experiment_id="exp-1",
            client_action_id=uuid.uuid4(),
        )

        result = svc.request_session_start(request)

        expected_session_id = _stable_session_id_for_action(request.client_action_id)
        assert result.accepted is True
        assert result.action == "start"
        assert result.session_id == expected_session_id
        assert result.received_at_utc == _NOW
        _assert_idempotency_write(
            redis,
            action="start",
            session_id=expected_session_id,
            client_action_id=request.client_action_id,
        )
        redis.publish.assert_called_once()
        channel, payload_json = redis.publish.call_args.args
        assert channel == "session:lifecycle"
        payload = json.loads(payload_json)
        assert payload == {
            "action": "start",
            "session_id": str(expected_session_id),
            "stream_url": request.stream_url,
            "experiment_id": request.experiment_id,
        }

    def test_duplicate_client_action_id_suppressed(self) -> None:
        redis = MagicMock()
        redis.set.return_value = None
        svc = SessionLifecycleService(
            get_conn=lambda: None,
            put_conn=lambda _conn: None,
            redis_factory=lambda: redis,
            clock=_clock,
        )
        request = SessionCreateRequest(
            stream_url="https://example.com/live",
            experiment_id="exp-1",
            client_action_id=uuid.uuid4(),
        )

        result = svc.request_session_start(request)

        assert result.accepted is True
        assert result.action == "start"
        assert result.message == "duplicate submission suppressed"
        assert result.session_id == _stable_session_id_for_action(request.client_action_id)
        redis.publish.assert_not_called()

    def test_no_subscribers_sets_warning_message(self) -> None:
        redis = MagicMock()
        redis.set.return_value = True
        redis.publish.return_value = 0
        svc = SessionLifecycleService(
            get_conn=lambda: None,
            put_conn=lambda _conn: None,
            redis_factory=lambda: redis,
            clock=_clock,
        )
        request = SessionCreateRequest(
            stream_url="https://example.com/live",
            experiment_id="exp-1",
            client_action_id=uuid.uuid4(),
        )

        result = svc.request_session_start(request)

        assert result.accepted is True
        assert result.message is not None
        assert "no orchestrator" in result.message.lower()

    def test_redis_client_creation_failure_raises_publish_error(self) -> None:
        def _broken_factory() -> Any:
            raise Exception("broker unavailable")

        svc = SessionLifecycleService(
            get_conn=lambda: None,
            put_conn=lambda _conn: None,
            redis_factory=_broken_factory,
            clock=_clock,
        )
        request = SessionCreateRequest(
            stream_url="https://example.com/live",
            experiment_id="exp-1",
            client_action_id=uuid.uuid4(),
        )

        with pytest.raises(SessionLifecyclePublishError):
            svc.request_session_start(request)

    def test_setnx_failure_raises_publish_error(self) -> None:
        redis = MagicMock()
        redis.set.side_effect = Exception("set failed")
        svc = SessionLifecycleService(
            get_conn=lambda: None,
            put_conn=lambda _conn: None,
            redis_factory=lambda: redis,
            clock=_clock,
        )
        request = SessionCreateRequest(
            stream_url="https://example.com/live",
            experiment_id="exp-1",
            client_action_id=uuid.uuid4(),
        )

        with pytest.raises(SessionLifecyclePublishError):
            svc.request_session_start(request)

        redis.publish.assert_not_called()

    def test_publish_failure_raises_and_rolls_back_claimed_key(self) -> None:
        redis = MagicMock()
        redis.set.return_value = True
        redis.publish.side_effect = Exception("broker down")
        redis.delete.return_value = 1
        svc = SessionLifecycleService(
            get_conn=lambda: None,
            put_conn=lambda _conn: None,
            redis_factory=lambda: redis,
            clock=_clock,
        )
        request = SessionCreateRequest(
            stream_url="https://example.com/live",
            experiment_id="exp-1",
            client_action_id=uuid.uuid4(),
        )

        with pytest.raises(SessionLifecyclePublishError):
            svc.request_session_start(request)

        redis.delete.assert_called_once_with(f"operator:session:seen:{request.client_action_id}")

    def test_publish_failure_rolls_back_key_so_retry_can_reattempt(self) -> None:
        redis = _RetryableRedis()
        svc = SessionLifecycleService(
            get_conn=lambda: None,
            put_conn=lambda _conn: None,
            redis_factory=lambda: redis,
            clock=_clock,
        )
        request = SessionCreateRequest(
            stream_url="https://example.com/live",
            experiment_id="exp-1",
            client_action_id=uuid.uuid4(),
        )
        idempotency_key = f"operator:session:seen:{request.client_action_id}"

        with pytest.raises(SessionLifecyclePublishError):
            svc.request_session_start(request)

        assert redis.delete_calls == [idempotency_key]
        assert idempotency_key not in redis.values

        result = svc.request_session_start(request)

        assert result.accepted is True
        assert result.session_id == _stable_session_id_for_action(request.client_action_id)
        assert redis.publish_attempts == 2
        assert idempotency_key in redis.values

    def test_rollback_failure_raises_publish_error(self) -> None:
        primary_client = MagicMock()
        primary_client.set.return_value = True
        primary_client.publish.side_effect = Exception("broker down")
        primary_client.delete.side_effect = Exception("delete failed")

        rollback_client = MagicMock()
        rollback_client.delete.side_effect = Exception("still failing")

        redis_factory = MagicMock(side_effect=[primary_client, rollback_client])
        svc = SessionLifecycleService(
            get_conn=lambda: None,
            put_conn=lambda _conn: None,
            redis_factory=redis_factory,
            clock=_clock,
        )
        request = SessionCreateRequest(
            stream_url="https://example.com/live",
            experiment_id="exp-1",
            client_action_id=uuid.uuid4(),
        )

        with pytest.raises(SessionLifecyclePublishError):
            svc.request_session_start(request)

        rollback_client.delete.assert_called_once_with(
            f"operator:session:seen:{request.client_action_id}"
        )


class TestRequestSessionEnd:
    def test_active_session_publishes_end_intent(self) -> None:
        session_id = uuid.uuid4()
        conn, _cursor = _make_state_connection(
            [
                (
                    ["session_id", "stream_url", "started_at", "ended_at", "experiment_id"],
                    (
                        str(session_id),
                        "https://example.com/live",
                        datetime(2026, 4, 18, 11, 0, tzinfo=UTC),
                        None,
                        None,
                    ),
                ),
                (
                    ["session_id", "stream_url", "started_at", "ended_at", "experiment_id"],
                    (
                        str(session_id),
                        "https://example.com/live",
                        datetime(2026, 4, 18, 11, 0, tzinfo=UTC),
                        None,
                        "exp-1",
                    ),
                ),
            ]
        )
        redis = MagicMock()
        redis.get.return_value = None
        redis.set.return_value = True
        redis.publish.return_value = 1
        put_conn = MagicMock()
        svc = SessionLifecycleService(
            get_conn=lambda: conn,
            put_conn=put_conn,
            redis_factory=lambda: redis,
            clock=_clock,
        )
        request = SessionEndRequest(client_action_id=uuid.uuid4())

        result = svc.request_session_end(session_id, request)

        assert result.accepted is True
        assert result.action == "end"
        assert result.session_id == session_id
        _assert_idempotency_write(
            redis,
            action="end",
            session_id=session_id,
            client_action_id=request.client_action_id,
        )
        redis.publish.assert_called_once()
        channel, payload_json = redis.publish.call_args.args
        assert channel == "session:lifecycle"
        payload = json.loads(payload_json)
        assert payload == {
            "action": "end",
            "session_id": str(session_id),
            "stream_url": "https://example.com/live",
            "experiment_id": "exp-1",
        }
        put_conn.assert_called_once_with(conn)

    def test_duplicate_end_after_ended_at_populated_is_suppressed(self) -> None:
        session_id = uuid.uuid4()
        conn, _cursor = _make_state_connection(
            [
                (
                    ["session_id", "stream_url", "started_at", "ended_at", "experiment_id"],
                    (
                        str(session_id),
                        "https://example.com/live",
                        datetime(2026, 4, 18, 11, 0, tzinfo=UTC),
                        datetime(2026, 4, 18, 11, 45, tzinfo=UTC),
                        "exp-1",
                    ),
                ),
                (["session_id", "stream_url", "started_at", "ended_at", "experiment_id"], None),
            ]
        )
        redis = MagicMock()
        redis.get.return_value = _duplicate_end_record(session_id)
        put_conn = MagicMock()
        svc = SessionLifecycleService(
            get_conn=lambda: conn,
            put_conn=put_conn,
            redis_factory=lambda: redis,
            clock=_clock,
        )
        request = SessionEndRequest(client_action_id=uuid.uuid4())

        result = svc.request_session_end(session_id, request)

        assert result.accepted is True
        assert result.action == "end"
        assert result.session_id == session_id
        assert result.message == "duplicate submission suppressed"
        redis.get.assert_called_once_with(f"operator:session:seen:{request.client_action_id}")
        redis.set.assert_not_called()
        redis.publish.assert_not_called()
        put_conn.assert_called_once_with(conn)

    def test_missing_session_raises_conflict(self) -> None:
        session_id = uuid.uuid4()
        conn, _cursor = _make_state_connection(
            [
                (["session_id", "stream_url", "started_at", "ended_at", "experiment_id"], None),
                (["session_id", "stream_url", "started_at", "ended_at", "experiment_id"], None),
            ]
        )
        redis = MagicMock()
        redis.get.return_value = None
        svc = SessionLifecycleService(
            get_conn=lambda: conn,
            put_conn=lambda _conn: None,
            redis_factory=lambda: redis,
            clock=_clock,
        )

        with pytest.raises(SessionLifecycleConflictError):
            svc.request_session_end(session_id, SessionEndRequest(client_action_id=uuid.uuid4()))

        redis.set.assert_not_called()

    def test_already_ended_session_raises_conflict(self) -> None:
        session_id = uuid.uuid4()
        ended_at = datetime(2026, 4, 18, 11, 30, tzinfo=UTC)
        conn, _cursor = _make_state_connection(
            [
                (
                    ["session_id", "stream_url", "started_at", "ended_at", "experiment_id"],
                    (
                        str(session_id),
                        "https://example.com/live",
                        datetime(2026, 4, 18, 11, 0, tzinfo=UTC),
                        ended_at,
                        "exp-1",
                    ),
                ),
                (["session_id", "stream_url", "started_at", "ended_at", "experiment_id"], None),
            ]
        )
        redis = MagicMock()
        redis.get.return_value = None
        svc = SessionLifecycleService(
            get_conn=lambda: conn,
            put_conn=lambda _conn: None,
            redis_factory=lambda: redis,
            clock=_clock,
        )

        with pytest.raises(SessionLifecycleConflictError):
            svc.request_session_end(session_id, SessionEndRequest(client_action_id=uuid.uuid4()))

        redis.set.assert_not_called()

    def test_non_active_session_raises_conflict(self) -> None:
        target_session_id = uuid.uuid4()
        active_session_id = uuid.uuid4()
        conn, _cursor = _make_state_connection(
            [
                (
                    ["session_id", "stream_url", "started_at", "ended_at", "experiment_id"],
                    (
                        str(target_session_id),
                        "https://example.com/old",
                        datetime(2026, 4, 18, 10, 0, tzinfo=UTC),
                        None,
                        "exp-old",
                    ),
                ),
                (
                    ["session_id", "stream_url", "started_at", "ended_at", "experiment_id"],
                    (
                        str(active_session_id),
                        "https://example.com/new",
                        datetime(2026, 4, 18, 11, 0, tzinfo=UTC),
                        None,
                        "exp-new",
                    ),
                ),
            ]
        )
        redis = MagicMock()
        redis.get.return_value = None
        svc = SessionLifecycleService(
            get_conn=lambda: conn,
            put_conn=lambda _conn: None,
            redis_factory=lambda: redis,
            clock=_clock,
        )

        with pytest.raises(SessionLifecycleConflictError):
            svc.request_session_end(
                target_session_id,
                SessionEndRequest(client_action_id=uuid.uuid4()),
            )

        redis.set.assert_not_called()

    def test_no_active_session_raises_conflict(self) -> None:
        session_id = uuid.uuid4()
        conn, _cursor = _make_state_connection(
            [
                (
                    ["session_id", "stream_url", "started_at", "ended_at", "experiment_id"],
                    (
                        str(session_id),
                        "https://example.com/live",
                        datetime(2026, 4, 18, 11, 0, tzinfo=UTC),
                        None,
                        "exp-1",
                    ),
                ),
                (["session_id", "stream_url", "started_at", "ended_at", "experiment_id"], None),
            ]
        )
        redis = MagicMock()
        redis.get.return_value = None
        svc = SessionLifecycleService(
            get_conn=lambda: conn,
            put_conn=lambda _conn: None,
            redis_factory=lambda: redis,
            clock=_clock,
        )

        with pytest.raises(SessionLifecycleConflictError):
            svc.request_session_end(session_id, SessionEndRequest(client_action_id=uuid.uuid4()))

        redis.set.assert_not_called()
