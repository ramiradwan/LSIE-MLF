"""
OperatorActionService — the single write surface behind `/api/v1/operator/*`.

Only the stimulus submission endpoint is a write — the rest of the
Operator Console is read-only aggregation. Keeping writes in a dedicated
service makes idempotency, state-guarding, and side-effect ordering
auditable in one place.

Design constraints:
  - Idempotency by `client_action_id` — Redis `SETNX` under key
    `operator:stimulus:seen:{client_action_id}` collapses accidental
    double-submits (§4.C authoritative stimulus bookkeeping).
  - The API Server never assigns the authoritative `stimulus_time`.
    That is the orchestrator's responsibility: only the drift-corrected
    pipeline clock is allowed to anchor the §7B reward window. The API
    merely publishes a `stimulus:inject` trigger and records a receipt.
  - Reject with 409 when the target session is already ended.
  - No GUI-clock leakage: `received_at_utc` is this service's clock for
    audit, not the reward anchor.

Spec references:
  §4.C       — Orchestrator stimulus lifecycle (`_active_arm`,
               `_expected_greeting`, authoritative `_stimulus_time`).
  §4.E.1     — Operator Intervention route behavior.
  §12        — Error-handling matrix (Redis failure, degraded states).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import UUID

from packages.schemas.operator_console import StimulusAccepted, StimulusRequest
from services.api.db.connection import get_connection, put_connection
from services.api.repos import operator_queries as q

logger = logging.getLogger(__name__)

# 24h TTL on the idempotency key — long enough to cover any realistic
# operator retry without letting keys accumulate indefinitely.
_IDEMPOTENCY_TTL_S: int = 24 * 3600

# Redis channel the orchestrator listener subscribes to (§4.E.1).
_STIMULUS_CHANNEL: str = "stimulus:inject"


class SessionAlreadyEndedError(Exception):
    """Raised when stimulus is targeted at a session with `ended_at` set."""


class SessionNotFoundError(Exception):
    """Raised when stimulus is targeted at a session that does not exist."""


class StimulusPublishError(Exception):
    """Raised when the Redis publish path cannot deliver the trigger."""


class RedisClientLike(Protocol):
    """Minimal Redis client surface — just what this service needs.

    Accepting a protocol keeps the service testable without importing
    the real `redis` package in unit tests.
    """

    def set(
        self, name: str, value: str, *, nx: bool = ..., ex: int = ...
    ) -> Any: ...

    def publish(self, channel: str, message: str) -> int: ...

    def close(self) -> None: ...


def _default_redis_factory() -> RedisClientLike:
    """Lazy import — `redis` is a container-only runtime dep."""

    import redis as redis_lib

    url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
    return redis_lib.from_url(url)  # type: ignore[no-any-return,no-untyped-call]


class OperatorActionService:
    """Handles the operator-initiated stimulus submission path."""

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

    def submit_stimulus(
        self, session_id: UUID, request: StimulusRequest
    ) -> StimulusAccepted:
        """§4.C — operator-issued stimulus with idempotency + state guard.

        Raises:
            SessionNotFoundError: 404 at the route layer.
            SessionAlreadyEndedError: 409 at the route layer.
            StimulusPublishError: 503 at the route layer.
        """
        conn: Any = None
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                row = q.fetch_session_by_id(cur, session_id)
        finally:
            if conn is not None:
                self._put_conn(conn)

        if row is None:
            raise SessionNotFoundError(str(session_id))
        if row.get("ended_at") is not None:
            raise SessionAlreadyEndedError(str(session_id))

        now = self._clock()
        idempotency_key = f"operator:stimulus:seen:{request.client_action_id}"
        client = self._redis_factory()
        try:
            # SETNX succeeds the first time only; subsequent calls return
            # falsy and we short-circuit as an accepted no-op so the
            # operator never sees a 409 for a retried submit.
            set_result = client.set(
                idempotency_key,
                str(session_id),
                nx=True,
                ex=_IDEMPOTENCY_TTL_S,
            )
            if not set_result:
                logger.info(
                    "stimulus dedup: client_action_id=%s already recorded",
                    request.client_action_id,
                )
                return StimulusAccepted(
                    session_id=session_id,
                    client_action_id=request.client_action_id,
                    accepted=True,
                    received_at_utc=now,
                    message="duplicate submission suppressed",
                )

            try:
                receivers = client.publish(_STIMULUS_CHANNEL, "inject")
            except Exception as exc:  # noqa: BLE001
                raise StimulusPublishError(
                    "failed to publish stimulus trigger to broker"
                ) from exc
        finally:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                logger.debug("redis client close failed", exc_info=True)

        message: str | None = None
        if receivers == 0:
            # §12: orchestrator silence → operator-visible warning, but
            # the receipt is still accepted so the audit row exists.
            message = "no orchestrator listening; inject will be retried"
        return StimulusAccepted(
            session_id=session_id,
            client_action_id=request.client_action_id,
            accepted=True,
            received_at_utc=now,
            message=message,
        )
