"""
SessionLifecycleService — write surface behind `/api/v1/sessions*` lifecycle POSTs.

The API Server only accepts operator intent and relays it to Redis. The
orchestrator remains the authoritative owner of session start/end timing,
including `started_at`/`ended_at` writes.

Design constraints:
  - Idempotency by `client_action_id` via Redis SETNX under
    `operator:session:seen:{client_action_id}` with a 24h TTL.
  - Publish normalized JSON payloads on `session:lifecycle` for the
    orchestrator to execute authoritatively.
  - The API may read current session state to reject invalid end-session
    attempts, but it never writes lifecycle timestamps directly.
  - Mirror the OperatorActionService semantics: duplicate submissions are
    accepted as no-ops, invalid end attempts raise 409 at the route layer,
    and Redis broker failures surface as 503.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal, Protocol
from uuid import UUID

from packages.schemas.operator_console import (
    SessionCreateRequest,
    SessionEndRequest,
    SessionLifecycleAccepted,
)
from services.api.db.connection import get_connection, put_connection

logger = logging.getLogger(__name__)

_IDEMPOTENCY_TTL_S: int = 24 * 3600
_SESSION_LIFECYCLE_CHANNEL: str = "session:lifecycle"

_GET_SESSION_STATE_SQL: str = """
    SELECT
        s.session_id,
        s.stream_url,
        s.started_at,
        s.ended_at,
        latest_enc.experiment_id
    FROM sessions s
    LEFT JOIN LATERAL (
        SELECT e.experiment_id
        FROM encounter_log e
        WHERE e.session_id = s.session_id
        ORDER BY e.timestamp_utc DESC
        LIMIT 1
    ) latest_enc ON TRUE
    WHERE s.session_id = %(session_id)s
"""

_GET_ACTIVE_SESSION_STATE_SQL: str = """
    SELECT
        s.session_id,
        s.stream_url,
        s.started_at,
        s.ended_at,
        latest_enc.experiment_id
    FROM sessions s
    LEFT JOIN LATERAL (
        SELECT e.experiment_id
        FROM encounter_log e
        WHERE e.session_id = s.session_id
        ORDER BY e.timestamp_utc DESC
        LIMIT 1
    ) latest_enc ON TRUE
    WHERE s.ended_at IS NULL
    ORDER BY s.started_at DESC
    LIMIT 1
"""


class SessionLifecycleConflictError(Exception):
    """Raised when an end-session request targets a non-endable session."""


class SessionLifecyclePublishError(Exception):
    """Raised when the Redis publish path cannot deliver lifecycle intent."""


class RedisClientLike(Protocol):
    """Minimal Redis surface required by SessionLifecycleService."""

    def set(
        self,
        name: str,
        value: str,
        *,
        nx: bool = ...,
        xx: bool = ...,
        ex: int = ...,
        keepttl: bool = ...,
    ) -> Any: ...

    def get(self, name: str) -> Any: ...

    def delete(self, *names: str) -> int: ...

    def publish(self, channel: str, message: str) -> int: ...

    def close(self) -> None: ...


def _default_redis_factory() -> RedisClientLike:
    """Lazy import — `redis` remains a runtime/container dependency."""

    import redis as redis_lib

    url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
    return redis_lib.from_url(url)  # type: ignore[no-any-return,no-untyped-call]


def _row_to_dict(cursor: Any) -> dict[str, Any] | None:
    if cursor.description is None:
        return None
    columns = [desc[0] for desc in cursor.description]
    row: Any = cursor.fetchone()
    if row is None:
        return None
    return dict(zip(columns, row, strict=True))


def _stable_session_id_for_action(client_action_id: UUID) -> UUID:
    """Derive a deterministic UUIDv4-shaped session id from the action id.

    Create-session retries do not carry a caller-provided `session_id`, so
    we need a stable identifier before Redis SETNX runs. The bytes are
    deterministic but the UUID version/variant bits are normalized to v4 so
    downstream payloads still satisfy the session UUID contract.
    """

    digest = hashlib.sha256(f"session:{client_action_id}".encode()).digest()
    raw = bytearray(digest[:16])
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return UUID(bytes=bytes(raw))


def _idempotency_key(client_action_id: UUID) -> str:
    return f"operator:session:seen:{client_action_id}"


def _idempotency_value(action: Literal["start", "end"], session_id: UUID) -> str:
    return json.dumps(
        {"action": action, "session_id": str(session_id)},
        separators=(",", ":"),
    )


def _normalize_redis_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return str(value)


def _parse_idempotency_value(value: Any) -> tuple[Literal["start", "end"] | None, str | None]:
    text = _normalize_redis_value(value)
    if text is None:
        return None, None

    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        # Backward-compatible with older plain-string values that stored only
        # the session_id before this service started recording minimal request
        # metadata for duplicate end suppression.
        return None, text

    if not isinstance(decoded, dict):
        return None, None

    raw_action = decoded.get("action")
    action: Literal["start", "end"] | None = raw_action if raw_action in ("start", "end") else None

    raw_session_id = decoded.get("session_id")
    session_id = raw_session_id if isinstance(raw_session_id, str) and raw_session_id else None
    return action, session_id


class SessionLifecycleService:
    """Publishes create/end lifecycle intent for orchestrator execution."""

    def __init__(
        self,
        *,
        get_conn: Callable[[], Any] = get_connection,
        put_conn: Callable[[Any], None] = put_connection,
        redis_factory: Callable[[], RedisClientLike] = _default_redis_factory,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._get_conn = get_conn
        self._put_conn = put_conn
        self._redis_factory = redis_factory
        self._clock = clock

    def request_session_start(self, request: SessionCreateRequest) -> SessionLifecycleAccepted:
        """Accept a create-session intent and publish it for the orchestrator."""

        session_id = _stable_session_id_for_action(request.client_action_id)
        return self._publish_intent(
            action="start",
            session_id=session_id,
            client_action_id=request.client_action_id,
            stream_url=request.stream_url,
            experiment_id=request.experiment_id,
        )

    def request_session_end(
        self,
        session_id: UUID,
        request: SessionEndRequest,
    ) -> SessionLifecycleAccepted:
        """Accept an end-session intent if the target session is still active."""

        target_row, active_row = self._load_end_state(session_id)
        active_session_id = str(active_row.get("session_id")) if active_row is not None else None
        has_conflict = (
            target_row is None
            or target_row.get("ended_at") is not None
            or active_row is None
            or active_session_id != str(session_id)
        )

        if has_conflict:
            duplicate_response = self._get_duplicate_end_response(
                session_id=session_id,
                client_action_id=request.client_action_id,
            )
            if duplicate_response is not None:
                return duplicate_response

        if target_row is None:
            raise SessionLifecycleConflictError(
                f"session {session_id} is not active; end not accepted"
            )
        if target_row.get("ended_at") is not None:
            raise SessionLifecycleConflictError(
                f"session {session_id} has already ended; end not accepted"
            )
        if active_row is None:
            raise SessionLifecycleConflictError(
                f"session {session_id} is not active; end not accepted"
            )
        if active_session_id != str(session_id):
            raise SessionLifecycleConflictError(
                f"session {session_id} is not the active session; end not accepted"
            )

        stream_url = target_row.get("stream_url")
        experiment_id = active_row.get("experiment_id")
        return self._publish_intent(
            action="end",
            session_id=session_id,
            client_action_id=request.client_action_id,
            stream_url=str(stream_url) if stream_url is not None else None,
            experiment_id=str(experiment_id) if experiment_id is not None else None,
        )

    def _load_end_state(
        self,
        session_id: UUID,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        conn: Any = None
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(_GET_SESSION_STATE_SQL, {"session_id": str(session_id)})
                target_row = _row_to_dict(cur)
                cur.execute(_GET_ACTIVE_SESSION_STATE_SQL)
                active_row = _row_to_dict(cur)
                return target_row, active_row
        finally:
            if conn is not None:
                self._put_conn(conn)

    def _publish_intent(
        self,
        *,
        action: Literal["start", "end"],
        session_id: UUID,
        client_action_id: UUID,
        stream_url: str | None,
        experiment_id: str | None,
    ) -> SessionLifecycleAccepted:
        now = self._clock()
        idempotency_key = _idempotency_key(client_action_id)
        client = self._create_redis_client()
        try:
            try:
                set_result = client.set(
                    idempotency_key,
                    _idempotency_value(action, session_id),
                    nx=True,
                    ex=_IDEMPOTENCY_TTL_S,
                )
            except Exception as exc:  # noqa: BLE001
                raise SessionLifecyclePublishError(
                    "failed to record session lifecycle idempotency key"
                ) from exc

            if not set_result:
                logger.info(
                    "session lifecycle dedup: client_action_id=%s already recorded",
                    client_action_id,
                )
                return self._duplicate_response(
                    action=action,
                    session_id=session_id,
                    client_action_id=client_action_id,
                    received_at_utc=now,
                )

            payload = json.dumps(
                {
                    "action": action,
                    "session_id": str(session_id),
                    "stream_url": stream_url,
                    "experiment_id": experiment_id,
                }
            )
            try:
                receivers = client.publish(_SESSION_LIFECYCLE_CHANNEL, payload)
            except Exception as exc:  # noqa: BLE001
                self._rollback_idempotency_claim(idempotency_key, client)
                raise SessionLifecyclePublishError(
                    "failed to publish session lifecycle intent to broker"
                ) from exc
        finally:
            self._close_redis_client(client)

        message: str | None = None
        if receivers == 0:
            message = "no orchestrator listening; lifecycle intent will be retried"
        return SessionLifecycleAccepted(
            action=action,
            session_id=session_id,
            client_action_id=client_action_id,
            accepted=True,
            received_at_utc=now,
            message=message,
        )

    def _get_duplicate_end_response(
        self,
        *,
        session_id: UUID,
        client_action_id: UUID,
    ) -> SessionLifecycleAccepted | None:
        stored_action, stored_session_id = self._read_idempotency_record(client_action_id)
        if stored_session_id != str(session_id):
            return None
        if stored_action not in (None, "end"):
            return None

        logger.info(
            "session lifecycle dedup: client_action_id=%s already recorded for session end",
            client_action_id,
        )
        return self._duplicate_response(
            action="end",
            session_id=session_id,
            client_action_id=client_action_id,
            received_at_utc=self._clock(),
        )

    def _read_idempotency_record(
        self,
        client_action_id: UUID,
    ) -> tuple[Literal["start", "end"] | None, str | None]:
        client = self._create_redis_client()
        try:
            try:
                return _parse_idempotency_value(client.get(_idempotency_key(client_action_id)))
            except Exception as exc:  # noqa: BLE001
                raise SessionLifecyclePublishError(
                    "failed to read session lifecycle idempotency key"
                ) from exc
        finally:
            self._close_redis_client(client)

    def _rollback_idempotency_claim(self, idempotency_key: str, client: RedisClientLike) -> None:
        try:
            client.delete(idempotency_key)
            return
        except Exception:  # noqa: BLE001
            logger.debug(
                "session lifecycle idempotency rollback failed on publish client",
                exc_info=True,
            )

        rollback_client: RedisClientLike | None = None
        try:
            rollback_client = self._create_redis_client()
            rollback_client.delete(idempotency_key)
        except Exception as exc:  # noqa: BLE001
            raise SessionLifecyclePublishError(
                "failed to roll back session lifecycle idempotency after broker publish failure"
            ) from exc
        finally:
            if rollback_client is not None and rollback_client is not client:
                self._close_redis_client(rollback_client)

    def _create_redis_client(self) -> RedisClientLike:
        try:
            return self._redis_factory()
        except Exception as exc:  # noqa: BLE001
            raise SessionLifecyclePublishError(
                "failed to create redis client for session lifecycle intent"
            ) from exc

    def _duplicate_response(
        self,
        *,
        action: Literal["start", "end"],
        session_id: UUID,
        client_action_id: UUID,
        received_at_utc: datetime,
    ) -> SessionLifecycleAccepted:
        return SessionLifecycleAccepted(
            action=action,
            session_id=session_id,
            client_action_id=client_action_id,
            accepted=True,
            received_at_utc=received_at_utc,
            message="duplicate submission suppressed",
        )

    def _close_redis_client(self, client: RedisClientLike | None) -> None:
        if client is None:
            return
        try:
            client.close()
        except Exception:  # noqa: BLE001
            logger.debug("redis client close failed", exc_info=True)
